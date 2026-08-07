!> A deterministic, exact-split second-order boosting foundation.
module fortml_xgboost
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: XGB_OBJECTIVE_SQUARED = 1
    integer, parameter, public :: XGB_OBJECTIVE_LOGISTIC = 2
    integer, parameter, public :: XGB_MISSING_ERROR = 0
    integer, parameter, public :: XGB_MISSING_LEARN = 1
    integer, parameter, public :: XGB_MISSING_LEFT = 2
    integer, parameter, public :: XGB_MISSING_RIGHT = 3

    !> Options for the deterministic exact-split XGBoost-style estimator.
    !>
    !> Numeric splits are enumerated exhaustively.  Trees may grow to
    !> `max_depth` and use the full second-order leaf and split formulas,
    !> including L1/L2 regularisation, gamma, min-child-Hessian and shrinkage.
    !> The exact CPU path accepts IEEE NaNs when `missing_policy` is `learn`,
    !> `left`, or `right`.  `learn` evaluates both default directions for every
    !> candidate split and stores the direction with the best gain, while the
    !> fixed policies route all missing values to the requested child.
    type, public :: xgboost_options_t
        integer :: n_estimators = 50
        integer :: max_depth = 1
        integer :: min_samples_leaf = 1
        real(dp) :: learning_rate = 0.3_dp
        real(dp) :: l1 = 0.0_dp
        real(dp) :: l2 = 1.0_dp
        real(dp) :: gamma = 0.0_dp
        real(dp) :: min_child_weight = 1.0e-3_dp
        character(len=16) :: objective = "squared"
        character(len=16) :: missing_policy = "error"
    end type xgboost_options_t

    type :: xgb_tree_t
        ! Legacy root fields are retained as cheap diagnostics for the public
        ! split-gain/leaf-weight accessors.  The complete tree is represented
        ! by the node arrays below.
        integer :: feature_index = 0
        integer :: left_count = 0
        integer :: right_count = 0
        real(dp) :: threshold = 0.0_dp
        real(dp) :: left_weight = 0.0_dp
        real(dp) :: right_weight = 0.0_dp
        real(dp) :: split_gain = 0.0_dp
        logical :: has_split = .false.
        integer :: n_nodes = 0
        integer :: depth = 0
        integer, allocatable :: feature(:), left_child(:), right_child(:)
        real(dp), allocatable :: node_threshold(:), weight(:), node_gain(:)
        real(dp), allocatable :: node_cover(:)
        logical, allocatable :: leaf(:)
        logical, allocatable :: missing_left(:)
    end type xgb_tree_t

    !> Exact-split second-order boosting for squared and binary logistic
    !> objectives.  Fit is discrete; predictions are deterministic and the
    !> objective's Hessians are aggregated exactly for every candidate split.
    type, public :: xgboost_t
        private
        integer :: n_inputs = 0
        integer :: n_estimators = 0
        integer :: objective_code = 0
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: base_score = 0.0_dp
        integer :: missing_code = XGB_MISSING_ERROR
        type(xgb_tree_t), allocatable :: estimators(:)
        logical :: initialized = .false.
    contains
        procedure, public :: fit => xgb_fit
        procedure, public :: fit_regression => xgb_fit_regression
        procedure, public :: fit_binary => xgb_fit_binary
        procedure, public :: predict_matrix => xgb_predict_matrix
        procedure, public :: predict_vector => xgb_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_margin_matrix => xgb_predict_margin_matrix
        procedure, public :: predict_margin_vector => xgb_predict_margin_vector
        generic, public :: predict_margin => predict_margin_matrix, &
            predict_margin_vector
        procedure, public :: predict_jvp => xgb_predict_jvp
        procedure, public :: predict_staged => xgb_predict_staged
        procedure, public :: predict_staged_margin => xgb_predict_staged_margin
        procedure, public :: predict_proba_staged => xgb_predict_proba_staged
        procedure, public :: predict_proba => xgb_predict_proba
        procedure, public :: decision_function => xgb_decision_function
        procedure, public :: predict_vjp => xgb_predict_vjp
        procedure, public :: split_gain => xgb_split_gain
        procedure, public :: leaf_weights => xgb_leaf_weights
        procedure, public :: feature_importance => xgb_feature_importance
        procedure, public :: tree_node_count => xgb_tree_node_count
        procedure, public :: tree_depth => xgb_tree_depth
        procedure, public :: feature_count => xgb_feature_count
        procedure, public :: estimator_count => xgb_estimator_count
        procedure, public :: base_margin => xgb_base_margin
        procedure, public :: objective_name => xgb_objective_name
        procedure, public :: missing_policy => xgb_missing_policy
        procedure, public :: accepts_missing => xgb_accepts_missing
        procedure, public :: fitted => xgb_fitted
    end type xgboost_t

contains

    subroutine xgb_fit(self, x, y, status, options)
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        type(xgboost_options_t) :: settings
        real(dp), allocatable :: prediction(:), gradient(:), hessian(:)
        real(dp), allocatable :: correction(:)
        real(dp) :: mean_target, rate
        integer :: objective_code, missing_code, i, n_samples, n_features

        settings = xgboost_options_t()
        if (present(options)) settings = options
        objective_code = parse_objective(settings%objective)
        if (objective_code == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: objective must be squared or logistic")
            return
        end if
        missing_code = parse_missing_policy(settings%missing_policy)
        if (missing_code < XGB_MISSING_ERROR) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: missing_policy must be error, learn, left, or right")
            return
        end if

        n_samples = size(x, 1)
        n_features = size(x, 2)
        rate = settings%learning_rate
        if (n_samples < 2 .or. n_features < 1 .or. size(y) /= n_samples .or. &
            settings%n_estimators < 1 .or. settings%max_depth < 1 .or. &
            settings%max_depth > n_samples .or. &
            settings%min_samples_leaf < 1 .or. &
            2*settings%min_samples_leaf > n_samples .or. &
            .not. ieee_is_finite(rate) .or. rate <= 0.0_dp .or. rate > 1.0_dp .or. &
            .not. ieee_is_finite(settings%l1) .or. settings%l1 < 0.0_dp .or. &
            .not. ieee_is_finite(settings%l2) .or. settings%l2 < 0.0_dp .or. &
            .not. ieee_is_finite(settings%gamma) .or. settings%gamma < 0.0_dp .or. &
            .not. ieee_is_finite(settings%min_child_weight) .or. &
            settings%min_child_weight < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: invalid dimensions or hyperparameters")
            return
        end if
        if (any((.not. ieee_is_finite(x)) .and. (.not. ieee_is_nan(x))) .or. &
            any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: inputs must be finite or IEEE NaN and targets finite")
            return
        end if
        if (missing_code == XGB_MISSING_ERROR .and. any(ieee_is_nan(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: NaN inputs require a missing-value policy")
            return
        end if
        if (objective_code == XGB_OBJECTIVE_LOGISTIC) then
            if (any(y < 0.0_dp) .or. any(y > 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: logistic targets must be in [0, 1]")
                return
            end if
        end if

        mean_target = sum(y)/real(n_samples, dp)
        if (objective_code == XGB_OBJECTIVE_LOGISTIC) then
            self%base_score = stable_logit(mean_target)
        else
            self%base_score = mean_target
        end if
        allocate(self%estimators(settings%n_estimators))
        allocate(prediction(n_samples), gradient(n_samples), hessian(n_samples))
        allocate(correction(n_samples))
        prediction = self%base_score

        do i = 1, settings%n_estimators
            call objective_derivatives(objective_code, prediction, y, gradient, &
                hessian, status)
            if (status%code /= FORTNUM_OK) return
            call build_tree(x, gradient, hessian, settings, self%estimators(i), &
                status)
            if (status%code /= FORTNUM_OK) return
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            prediction = prediction + rate*correction
        end do

        self%n_inputs = n_features
        self%n_estimators = settings%n_estimators
        self%objective_code = objective_code
        self%learning_rate = rate
        self%missing_code = missing_code
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_fit

    subroutine xgb_fit_regression(self, x, y, status, options)
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "squared"
        call xgb_fit(self, x, y, status, settings)
    end subroutine xgb_fit_regression

    subroutine xgb_fit_binary(self, x, y, status, options)
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "logistic"
        call xgb_fit(self, x, y, status, settings)
    end subroutine xgb_fit_binary

    subroutine xgb_predict_matrix(self, x, y, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:)

        if (any(shape(y) /= [size(x, 1), 1])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict: output shape is invalid")
            return
        end if
        allocate(margin(size(x, 1)))
        call xgb_predict_margin_vector(self, x, margin, status)
        if (status%code /= FORTNUM_OK) return
        if (self%objective_code == XGB_OBJECTIVE_LOGISTIC) then
            y(:, 1) = stable_sigmoid_array(margin)
        else
            y(:, 1) = margin
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_matrix

    subroutine xgb_predict_vector(self, x, y, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict: output shape is invalid")
            return
        end if
        allocate(margin(size(y)))
        call xgb_predict_margin_vector(self, x, margin, status)
        if (status%code /= FORTNUM_OK) return
        if (self%objective_code == XGB_OBJECTIVE_LOGISTIC) then
            y = stable_sigmoid_array(margin)
        else
            y = margin
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_vector

    subroutine xgb_predict_margin_matrix(self, x, margin, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margin(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (any(shape(margin) /= [size(x, 1), 1])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_margin: output shape is invalid")
            return
        end if
        call xgb_predict_margin_vector(self, x, margin(:, 1), status)
    end subroutine xgb_predict_margin_matrix

    subroutine xgb_predict_margin_vector(self, x, margin, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margin(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(margin) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_margin: model, input, or output shape is invalid")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_margin: input has unsupported nonfinite values")
            return
        end if
        allocate(correction(size(x, 1)))
        margin = self%base_score
        do i = 1, self%n_estimators
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + self%learning_rate*correction
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_margin_vector

    !> Return the cumulative prediction after every boosting stage.
    !>
    !> The second dimension is ordered from the first fitted tree through the
    !> complete ensemble.  Regression stages contain margins; logistic stages
    !> contain positive-class probabilities, matching `predict`.
    subroutine xgb_predict_staged(self, x, staged, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: staged(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:), correction(:)
        integer :: i

        if (.not. self%initialized .or. size(staged, 1) /= size(x, 1) .or. &
            size(staged, 2) /= self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_staged: model, input, or output shape is invalid")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_staged: input has unsupported nonfinite values")
            return
        end if
        allocate(margin(size(x, 1)), correction(size(x, 1)))
        margin = self%base_score
        do i = 1, self%n_estimators
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + self%learning_rate*correction
            if (self%objective_code == XGB_OBJECTIVE_LOGISTIC) then
                staged(:, i) = stable_sigmoid_array(margin)
            else
                staged(:, i) = margin
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_staged

    !> Return cumulative raw margins after every boosting stage.
    subroutine xgb_predict_staged_margin(self, x, staged, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: staged(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized .or. size(staged, 1) /= size(x, 1) .or. &
            size(staged, 2) /= self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_staged_margin: model, input, or "// &
                "output shape is invalid")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_staged_margin: input has unsupported nonfinite values")
            return
        end if
        allocate(correction(size(x, 1)))
        staged = 0.0_dp
        staged(:, 1) = self%base_score
        do i = 1, self%n_estimators
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            if (i == 1) then
                staged(:, i) = self%base_score + self%learning_rate*correction
            else
                staged(:, i) = staged(:, i - 1) + self%learning_rate*correction
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_staged_margin

    !> Return cumulative binary probabilities after every boosting stage.
    !>
    !> The output has shape `(n_samples, 2, n_estimators)` and stores negative
    !> and positive class probabilities in the second dimension.  A regression
    !> estimator refuses this classification-only operation.
    subroutine xgb_predict_proba_staged(self, x, probabilities, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margins(:, :)
        real(dp) :: positive
        integer :: i, j

        if (self%objective_code /= XGB_OBJECTIVE_LOGISTIC) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_proba_staged: only logistic objective "// &
                "has probabilities")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= 2 .or. &
            size(probabilities, 3) /= self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_proba_staged: output shape is invalid")
            return
        end if
        allocate(margins(size(x, 1), self%n_estimators))
        call self%predict_staged_margin(x, margins, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_estimators
            do i = 1, size(x, 1)
                positive = stable_sigmoid(margins(i, j))
                probabilities(i, 1, j) = 1.0_dp - positive
                probabilities(i, 2, j) = positive
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_proba_staged

    subroutine xgb_predict_proba(self, x, probabilities, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:), positive(:)

        if (self%objective_code /= XGB_OBJECTIVE_LOGISTIC) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_proba: only logistic objective has probabilities")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), 2])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_proba: output shape is invalid")
            return
        end if
        allocate(margin(size(x, 1)), positive(size(x, 1)))
        call xgb_predict_margin_vector(self, x, margin, status)
        if (status%code /= FORTNUM_OK) return
        positive = stable_sigmoid_array(margin)
        probabilities(:, 1) = 1.0_dp - positive
        probabilities(:, 2) = positive
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_proba

    !> Input JVP of the piecewise-constant fitted predictor.
    !>
    !> The derivative is zero away from split surfaces.  A query exactly on a
    !> learned threshold is rejected because the tree has no classical
    !> derivative there.
    subroutine xgb_predict_jvp(self, x, x_dot, y, y_dot, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: y(:), y_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, node

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y) /= size(x, 1) .or. &
            size(y_dot) /= size(y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_jvp: model, input, or output shape is invalid")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_jvp: input or tangent has unsupported values")
            return
        end if
        do j = 1, self%n_estimators
            do node = 1, self%estimators(j)%n_nodes
                if (self%estimators(j)%leaf(node)) cycle
                do i = 1, size(x, 1)
                    if (.not. ieee_is_nan(x(i, self%estimators(j)%feature(node))) .and. &
                        x(i, self%estimators(j)%feature(node)) == &
                        self%estimators(j)%node_threshold(node)) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "xgboost predict_jvp: derivative is undefined on split")
                        return
                    end if
                end do
            end do
        end do
        call xgb_predict_vector(self, x, y, status)
        if (status%code /= FORTNUM_OK) return
        y_dot = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_jvp

    subroutine xgb_decision_function(self, x, margin, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margin(:)
        type(fortnum_status_t), intent(out) :: status

        call xgb_predict_margin_vector(self, x, margin, status)
    end subroutine xgb_decision_function

    !> Reverse-mode product of the fitted predictor with respect to query
    !> features.  A fitted tree is piecewise constant, so the product is zero
    !> away from learned split surfaces.  A query exactly on any threshold is
    !> rejected because the classical derivative is undefined there.
    subroutine xgb_predict_vjp(self, x, output_bar, x_bar, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, node

        x_bar = 0.0_dp
        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(output_bar) /= size(x, 1) .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_vjp: model, cotangent, or output shape is invalid")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x) .or. &
            any(.not. ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_vjp: input or cotangent has unsupported values")
            return
        end if
        do j = 1, self%n_estimators
            do node = 1, self%estimators(j)%n_nodes
                if (self%estimators(j)%leaf(node)) cycle
                do i = 1, size(x, 1)
                    if (.not. ieee_is_nan(x(i, self%estimators(j)%feature(node))) .and. &
                        x(i, self%estimators(j)%feature(node)) == &
                        self%estimators(j)%node_threshold(node)) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "xgboost predict_vjp: derivative is undefined on split")
                        return
                    end if
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_vjp

    real(dp) function xgb_split_gain(self, tree_index) result(gain)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: tree_index

        gain = 0.0_dp
        if (.not. allocated(self%estimators)) return
        if (tree_index < 1 .or. tree_index > size(self%estimators)) return
        gain = self%estimators(tree_index)%split_gain
    end function xgb_split_gain

    subroutine xgb_leaf_weights(self, tree_index, weights, status)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: tree_index
        real(dp), intent(out) :: weights(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. allocated(self%estimators)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost leaf_weights: model is not initialized")
            return
        end if
        if (size(weights) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost leaf_weights: output shape is invalid")
            return
        end if
        if (tree_index < 1 .or. tree_index > size(self%estimators)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost leaf_weights: tree index is invalid")
            return
        end if
        if (self%estimators(tree_index)%n_nodes < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost leaf_weights: tree is not initialized")
            return
        end if
        if (self%estimators(tree_index)%has_split) then
            weights = [self%estimators(tree_index)%weight( &
                self%estimators(tree_index)%left_child(1)), &
                self%estimators(tree_index)%weight( &
                self%estimators(tree_index)%right_child(1))]
        else
            weights = [self%estimators(tree_index)%weight(1), &
                self%estimators(tree_index)%weight(1)]
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_leaf_weights

    !> Aggregate split diagnostics by input feature.
    !>
    !> `kind` is `gain` (the sum of regularized split gains), `weight` (the
    !> number of internal split nodes), or `cover` (the sum of Hessians at
    !> internal nodes).  These are raw sums by default; pass `normalize=.true.`
    !> to divide by the total, as in scikit-learn's feature-importances API.
    subroutine xgb_feature_importance(self, importance, status, kind, normalize)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(out) :: importance(:)
        type(fortnum_status_t), intent(out) :: status
        character(len=*), intent(in), optional :: kind
        logical, intent(in), optional :: normalize
        character(len=16) :: metric
        logical :: should_normalize
        real(dp) :: total
        integer :: tree_index, node, feature

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost feature_importance: model is not initialized")
            return
        end if
        if (size(importance) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost feature_importance: output shape is invalid")
            return
        end if
        metric = "gain"
        if (present(kind)) metric = trim(adjustl(kind))
        select case (metric)
        case ("gain", "weight", "cover")
            continue
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost feature_importance: kind must be gain, weight, or cover")
            return
        end select
        should_normalize = .false.
        if (present(normalize)) should_normalize = normalize
        importance = 0.0_dp
        do tree_index = 1, self%n_estimators
            do node = 1, self%estimators(tree_index)%n_nodes
                if (self%estimators(tree_index)%leaf(node)) cycle
                feature = self%estimators(tree_index)%feature(node)
                select case (metric)
                case ("gain")
                    importance(feature) = importance(feature) + &
                        self%estimators(tree_index)%node_gain(node)
                case ("weight")
                    importance(feature) = importance(feature) + 1.0_dp
                case ("cover")
                    importance(feature) = importance(feature) + &
                        self%estimators(tree_index)%node_cover(node)
                end select
            end do
        end do
        if (should_normalize) then
            total = sum(importance)
            if (total > 0.0_dp) importance = importance/total
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_feature_importance

    integer function xgb_tree_node_count(self, tree_index) result(count)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: tree_index

        count = 0
        if (.not. allocated(self%estimators)) return
        if (tree_index < 1 .or. tree_index > size(self%estimators)) return
        count = self%estimators(tree_index)%n_nodes
    end function xgb_tree_node_count

    integer function xgb_tree_depth(self, tree_index) result(depth)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: tree_index

        depth = 0
        if (.not. allocated(self%estimators)) return
        if (tree_index < 1 .or. tree_index > size(self%estimators)) return
        depth = self%estimators(tree_index)%depth
    end function xgb_tree_depth

    integer function xgb_feature_count(self) result(count)
        class(xgboost_t), intent(in) :: self
        count = self%n_inputs
    end function xgb_feature_count

    integer function xgb_estimator_count(self) result(count)
        class(xgboost_t), intent(in) :: self
        count = self%n_estimators
    end function xgb_estimator_count

    real(dp) function xgb_base_margin(self) result(margin)
        class(xgboost_t), intent(in) :: self
        margin = self%base_score
    end function xgb_base_margin

    character(len=16) function xgb_objective_name(self) result(name)
        class(xgboost_t), intent(in) :: self

        select case (self%objective_code)
        case (XGB_OBJECTIVE_SQUARED)
            name = "squared"
        case (XGB_OBJECTIVE_LOGISTIC)
            name = "logistic"
        case default
            name = "unfitted"
        end select
    end function xgb_objective_name

    character(len=16) function xgb_missing_policy(self) result(name)
        class(xgboost_t), intent(in) :: self

        select case (self%missing_code)
        case (XGB_MISSING_ERROR)
            name = "error"
        case (XGB_MISSING_LEARN)
            name = "learn"
        case (XGB_MISSING_LEFT)
            name = "left"
        case (XGB_MISSING_RIGHT)
            name = "right"
        case default
            name = "unfitted"
        end select
    end function xgb_missing_policy

    logical function xgb_accepts_missing(self) result(value)
        class(xgboost_t), intent(in) :: self

        value = self%initialized .and. self%missing_code /= XGB_MISSING_ERROR
    end function xgb_accepts_missing

    logical function xgb_fitted(self) result(fitted)
        class(xgboost_t), intent(in) :: self
        fitted = self%initialized
    end function xgb_fitted

    subroutine objective_derivatives(objective_code, margin, target, gradient, &
            hessian, status)
        integer, intent(in) :: objective_code
        real(dp), intent(in) :: margin(:), target(:)
        real(dp), intent(out) :: gradient(:), hessian(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), parameter :: minimum_hessian = 1.0e-12_dp
        real(dp), allocatable :: probability(:)

        if (size(margin) /= size(target) .or. size(gradient) /= size(target) .or. &
            size(hessian) /= size(target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost derivatives: array shapes differ")
            return
        end if
        select case (objective_code)
        case (XGB_OBJECTIVE_SQUARED)
            gradient = margin - target
            hessian = 1.0_dp
        case (XGB_OBJECTIVE_LOGISTIC)
            allocate(probability(size(margin)))
            probability = stable_sigmoid_array(margin)
            gradient = probability - target
            hessian = max(probability*(1.0_dp - probability), minimum_hessian)
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost derivatives: unsupported objective")
            return
        end select
        if (any(.not. ieee_is_finite(gradient)) .or. &
            any(.not. ieee_is_finite(hessian))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost derivatives: nonfinite gradient or Hessian")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine objective_derivatives

    subroutine build_tree(x, gradient, hessian, options, tree, status)
        real(dp), intent(in) :: x(:, :), gradient(:), hessian(:)
        type(xgboost_options_t), intent(in) :: options
        type(xgb_tree_t), intent(out) :: tree
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: sample_index(:)
        integer :: n_samples, n_features, max_nodes, next_node, root, i

        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (size(gradient) /= n_samples .or. size(hessian) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree: derivative array shape is invalid")
            return
        end if
        if (n_samples < 1 .or. n_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree: at least one sample and feature are required")
            return
        end if

        ! A binary tree containing n leaves has at most 2*n-1 nodes.  This
        ! bound avoids integer exponentiation and is also a hard memory bound
        ! independent of a user-supplied depth.
        max_nodes = 2*n_samples - 1
        allocate(tree%feature(max_nodes), tree%left_child(max_nodes), &
            tree%right_child(max_nodes), tree%node_threshold(max_nodes), &
            tree%weight(max_nodes), tree%node_gain(max_nodes), &
            tree%node_cover(max_nodes), &
            tree%leaf(max_nodes), tree%missing_left(max_nodes))
        tree%feature = 0
        tree%left_child = 0
        tree%right_child = 0
        tree%node_threshold = 0.0_dp
        tree%weight = 0.0_dp
        tree%node_gain = 0.0_dp
        tree%node_cover = 0.0_dp
        tree%leaf = .true.
        tree%missing_left = .true.
        allocate(sample_index(n_samples))
        do i = 1, n_samples
            sample_index(i) = i
        end do
        next_node = 0
        tree%depth = 0
        call build_tree_node(x, gradient, hessian, options, sample_index, 0, &
            tree, next_node, root, status)
        if (status%code /= FORTNUM_OK) return
        tree%n_nodes = next_node
        tree%feature_index = tree%feature(root)
        tree%split_gain = tree%node_gain(root)
        tree%has_split = .not. tree%leaf(root)
        if (tree%has_split) tree%threshold = tree%node_threshold(root)
        tree%left_weight = tree%weight(root)
        tree%right_weight = tree%weight(root)
        if (tree%has_split) then
            tree%left_weight = tree%weight(tree%left_child(root))
            tree%right_weight = tree%weight(tree%right_child(root))
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_tree

    recursive subroutine build_tree_node(x, gradient, hessian, options, &
            sample_index, depth, tree, next_node, node_id, status)
        real(dp), intent(in) :: x(:, :), gradient(:), hessian(:)
        type(xgboost_options_t), intent(in) :: options
        integer, intent(in) :: sample_index(:), depth
        type(xgb_tree_t), intent(inout) :: tree
        integer, intent(inout) :: next_node
        integer, intent(out) :: node_id
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: order(:), left_index(:), right_index(:)
        integer, allocatable :: finite_index(:)
        real(dp), allocatable :: feature_values(:), finite_values(:)
        integer :: n_local, n_features, feature, k, i, left_count
        integer :: n_finite, n_missing, direction, n_directions
        integer :: best_feature, left_node, right_node
        real(dp) :: total_gradient, total_hessian, left_gradient, left_hessian
        real(dp) :: right_gradient, right_hessian, candidate_gain, best_gain
        real(dp) :: best_threshold, candidate_threshold, value
        real(dp) :: missing_gradient, missing_hessian
        logical :: best_missing_left, missing_left

        n_local = size(sample_index)
        n_features = size(x, 2)
        next_node = next_node + 1
        node_id = next_node
        tree%depth = max(tree%depth, depth)
        total_gradient = sum(gradient(sample_index))
        total_hessian = sum(hessian(sample_index))
        value = regularized_leaf_weight(total_gradient, total_hessian, options)
        tree%weight(node_id) = value
        tree%node_gain(node_id) = 0.0_dp
        tree%node_cover(node_id) = total_hessian
        tree%leaf(node_id) = .true.
        tree%missing_left(node_id) = .true.

        ! No split is possible at the depth limit or when both children would
        ! violate the minimum leaf size.  Returning a regularized leaf here is
        ! exactly the XGBoost Newton-leaf contract.
        if (depth >= options%max_depth .or. &
            n_local < 2*options%min_samples_leaf) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        allocate(order(n_local), feature_values(n_local), finite_index(n_local), &
            finite_values(n_local))
        best_gain = 0.0_dp
        best_feature = 0
        best_threshold = 0.0_dp
        best_missing_left = .true.
        do feature = 1, n_features
            n_finite = 0
            n_missing = 0
            missing_gradient = 0.0_dp
            missing_hessian = 0.0_dp
            do i = 1, n_local
                feature_values(i) = x(sample_index(i), feature)
                if (ieee_is_nan(feature_values(i))) then
                    n_missing = n_missing + 1
                    missing_gradient = missing_gradient + &
                        gradient(sample_index(i))
                    missing_hessian = missing_hessian + hessian(sample_index(i))
                else
                    n_finite = n_finite + 1
                    finite_index(n_finite) = sample_index(i)
                    finite_values(n_finite) = feature_values(i)
                end if
            end do
            if (n_finite < 2) cycle
            call sort_feature_indices(finite_values(:n_finite), order(:n_finite))
            left_gradient = 0.0_dp
            left_hessian = 0.0_dp
            n_directions = 1
            if (missing_code_for_options(options) == XGB_MISSING_LEARN) then
                n_directions = 2
            end if
            do k = 1, n_finite - 1
                i = order(k)
                left_gradient = left_gradient + gradient(finite_index(i))
                left_hessian = left_hessian + hessian(finite_index(i))
                if (k < options%min_samples_leaf .or. &
                    n_finite - k + n_missing < options%min_samples_leaf) cycle
                if (finite_values(order(k)) >= finite_values(order(k + 1))) cycle
                candidate_threshold = 0.5_dp*(finite_values(order(k)) + &
                    finite_values(order(k + 1)))
                do direction = 1, n_directions
                    missing_left = direction == 1
                    if (missing_code_for_options(options) == XGB_MISSING_RIGHT) then
                        missing_left = .false.
                    else if (missing_code_for_options(options) == XGB_MISSING_LEFT) then
                        missing_left = .true.
                    end if
                    if (missing_left) then
                        if (k + n_missing < options%min_samples_leaf .or. &
                            n_finite - k < options%min_samples_leaf) cycle
                    else
                        if (k < options%min_samples_leaf .or. &
                            n_finite - k + n_missing < options%min_samples_leaf) cycle
                    end if
                    if (missing_left) then
                        left_gradient = left_gradient + missing_gradient
                        left_hessian = left_hessian + missing_hessian
                        right_gradient = total_gradient - left_gradient
                        right_hessian = total_hessian - left_hessian
                    else
                        right_gradient = total_gradient - left_gradient - &
                            missing_gradient
                        right_hessian = total_hessian - left_hessian - &
                            missing_hessian
                    end if
                    if (left_hessian < options%min_child_weight .or. &
                        right_hessian < options%min_child_weight) then
                        if (missing_left) then
                            left_gradient = left_gradient - missing_gradient
                            left_hessian = left_hessian - missing_hessian
                        end if
                        cycle
                    end if
                    candidate_gain = 0.5_dp*(regularized_leaf_score(left_gradient, &
                        left_hessian, options) + regularized_leaf_score(right_gradient, &
                        right_hessian, options) - regularized_leaf_score(total_gradient, &
                        total_hessian, options)) - options%gamma
                    if (candidate_gain > best_gain) then
                        best_gain = candidate_gain
                        best_feature = feature
                        best_threshold = candidate_threshold
                        best_missing_left = missing_left
                    end if
                    if (missing_left) then
                        left_gradient = left_gradient - missing_gradient
                        left_hessian = left_hessian - missing_hessian
                    end if
                end do
            end do
        end do

        if (best_feature == 0) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        left_count = 0
        do i = 1, n_local
            if (go_left(x(sample_index(i), best_feature), best_threshold, &
                best_missing_left)) then
                left_count = left_count + 1
            end if
        end do
        if (left_count < options%min_samples_leaf .or. &
            n_local - left_count < options%min_samples_leaf) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree: split construction violated leaf constraint")
            return
        end if
        allocate(left_index(left_count), right_index(n_local - left_count))
        left_count = 0
        k = 0
        do i = 1, n_local
            if (go_left(x(sample_index(i), best_feature), best_threshold, &
                best_missing_left)) then
                left_count = left_count + 1
                left_index(left_count) = sample_index(i)
            else
                k = k + 1
                right_index(k) = sample_index(i)
            end if
        end do
        tree%leaf(node_id) = .false.
        tree%feature(node_id) = best_feature
        tree%node_threshold(node_id) = best_threshold
        tree%node_gain(node_id) = best_gain
        tree%missing_left(node_id) = best_missing_left
        call build_tree_node(x, gradient, hessian, options, left_index, depth + 1, &
            tree, next_node, left_node, status)
        if (status%code /= FORTNUM_OK) return
        call build_tree_node(x, gradient, hessian, options, right_index, depth + 1, &
            tree, next_node, right_node, status)
        if (status%code /= FORTNUM_OK) return
        tree%left_child(node_id) = left_node
        tree%right_child(node_id) = right_node
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_tree_node

    subroutine tree_predict(tree, x, values, status)
        type(xgb_tree_t), intent(in) :: tree
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, node

        if (size(values) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree prediction: output shape is invalid")
            return
        end if
        if (tree%n_nodes < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree prediction: tree is not initialized")
            return
        end if
        if (.not. allocated(tree%leaf)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree prediction: tree is not initialized")
            return
        end if
        do i = 1, size(x, 1)
            node = 1
            do while (.not. tree%leaf(node))
                if (go_left(x(i, tree%feature(node)), &
                    tree%node_threshold(node), tree%missing_left(node))) then
                    node = tree%left_child(node)
                else
                    node = tree%right_child(node)
                end if
            end do
            values(i) = tree%weight(node)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine tree_predict

    real(dp) function regularized_leaf_weight(gradient, hessian, options) &
            result(weight)
        real(dp), intent(in) :: gradient, hessian
        type(xgboost_options_t), intent(in) :: options
        real(dp) :: thresholded

        thresholded = max(abs(gradient) - options%l1, 0.0_dp)
        if (thresholded == 0.0_dp) then
            weight = 0.0_dp
        else
            weight = -sign(thresholded, gradient)/(hessian + options%l2)
        end if
    end function regularized_leaf_weight

    real(dp) function regularized_leaf_score(gradient, hessian, options) &
            result(score)
        real(dp), intent(in) :: gradient, hessian
        type(xgboost_options_t), intent(in) :: options
        real(dp) :: thresholded

        thresholded = max(abs(gradient) - options%l1, 0.0_dp)
        score = thresholded**2/(hessian + options%l2)
    end function regularized_leaf_score

    integer function parse_objective(name) result(code)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: normalized

        normalized = trim(adjustl(name))
        select case (normalized)
        case ("squared", "reg:squarederror", "regression")
            code = XGB_OBJECTIVE_SQUARED
        case ("logistic", "binary:logistic", "classification")
            code = XGB_OBJECTIVE_LOGISTIC
        case default
            code = 0
        end select
    end function parse_objective

    integer function parse_missing_policy(name) result(code)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: normalized

        normalized = trim(adjustl(name))
        select case (normalized)
        case ("error", "finite-only", "reject")
            code = XGB_MISSING_ERROR
        case ("learn", "learned", "default")
            code = XGB_MISSING_LEARN
        case ("left", "missing-left")
            code = XGB_MISSING_LEFT
        case ("right", "missing-right")
            code = XGB_MISSING_RIGHT
        case default
            code = -1
        end select
    end function parse_missing_policy

    integer function missing_code_for_options(options) result(code)
        type(xgboost_options_t), intent(in) :: options

        code = parse_missing_policy(options%missing_policy)
    end function missing_code_for_options

    logical function valid_query_values(missing_code, x) result(valid)
        integer, intent(in) :: missing_code
        real(dp), intent(in) :: x(:, :)

        valid = .not. any((.not. ieee_is_finite(x)) .and. (.not. ieee_is_nan(x)))
        if (.not. valid) return
        valid = missing_code /= XGB_MISSING_ERROR .or. .not. any(ieee_is_nan(x))
    end function valid_query_values

    logical function go_left(value, threshold, missing_left) result(value_is_left)
        real(dp), intent(in) :: value, threshold
        logical, intent(in) :: missing_left

        if (ieee_is_nan(value)) then
            value_is_left = missing_left
        else
            value_is_left = value < threshold
        end if
    end function go_left

    real(dp) function stable_logit(probability) result(value)
        real(dp), intent(in) :: probability
        real(dp), parameter :: epsilon = 1.0e-12_dp
        real(dp) :: clipped

        clipped = min(max(probability, epsilon), 1.0_dp - epsilon)
        value = log(clipped) - log(1.0_dp - clipped)
    end function stable_logit

    real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

    pure elemental real(dp) function stable_sigmoid_element(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid_element

    function stable_sigmoid_array(values) result(probabilities)
        real(dp), intent(in) :: values(:)
        real(dp) :: probabilities(size(values))

        probabilities = stable_sigmoid_element(values)
    end function stable_sigmoid_array

    subroutine sort_feature_indices(values, order)
        real(dp), intent(in) :: values(:)
        integer, intent(out) :: order(:)
        integer :: i, j, key

        do i = 1, size(values)
            order(i) = i
        end do
        do i = 2, size(values)
            key = order(i)
            j = i - 1
            do while (j >= 1)
                if (values(order(j)) <= values(key)) exit
                order(j + 1) = order(j)
                j = j - 1
            end do
            order(j + 1) = key
        end do
    end subroutine sort_feature_indices

end module fortml_xgboost
