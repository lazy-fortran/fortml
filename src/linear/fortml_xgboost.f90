!> A deterministic, exact-split second-order boosting foundation.
module fortml_xgboost
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: XGB_OBJECTIVE_SQUARED = 1
    integer, parameter, public :: XGB_OBJECTIVE_LOGISTIC = 2

    !> Options for the bounded exact-split XGBoost-style estimator.
    !>
    !> The current implementation deliberately grows depth-one trees.  It
    !> already uses the full second-order leaf and split formulas (including
    !> L1/L2 regularisation, gamma, min-child-Hessian and shrinkage), so a
    !> histogram/deeper-tree backend can be added without changing objective
    !> or prediction contracts.
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
    end type xgboost_options_t

    type :: xgb_tree_t
        integer :: feature_index = 0
        integer :: left_count = 0
        integer :: right_count = 0
        real(dp) :: threshold = 0.0_dp
        real(dp) :: left_weight = 0.0_dp
        real(dp) :: right_weight = 0.0_dp
        real(dp) :: split_gain = 0.0_dp
        logical :: has_split = .false.
    end type xgb_tree_t

    !> Exact depth-one second-order boosting for squared and binary logistic
    !> objectives.  Fit is discrete; predictions are deterministic and the
    !> objective's Hessians are aggregated exactly for every candidate split.
    type, public :: xgboost_t
        private
        integer :: n_inputs = 0
        integer :: n_estimators = 0
        integer :: objective_code = 0
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: base_score = 0.0_dp
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
        procedure, public :: predict_proba => xgb_predict_proba
        procedure, public :: decision_function => xgb_decision_function
        procedure, public :: split_gain => xgb_split_gain
        procedure, public :: leaf_weights => xgb_leaf_weights
        procedure, public :: feature_count => xgb_feature_count
        procedure, public :: estimator_count => xgb_estimator_count
        procedure, public :: base_margin => xgb_base_margin
        procedure, public :: objective_name => xgb_objective_name
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
        integer :: objective_code, i, n_samples, n_features

        settings = xgboost_options_t()
        if (present(options)) settings = options
        objective_code = parse_objective(settings%objective)
        if (objective_code == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: objective must be squared or logistic")
            return
        end if

        n_samples = size(x, 1)
        n_features = size(x, 2)
        rate = settings%learning_rate
        if (n_samples < 2 .or. n_features < 1 .or. size(y) /= n_samples .or. &
            settings%n_estimators < 1 .or. settings%max_depth /= 1 .or. &
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
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: inputs and targets must be finite")
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
            size(margin) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_margin: model, input, or output shape is invalid")
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
        integer :: i, j

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y) /= size(x, 1) .or. &
            size(y_dot) /= size(y) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_jvp: model, input, or output shape is invalid")
            return
        end if
        do j = 1, self%n_estimators
            if (.not. self%estimators(j)%has_split) cycle
            do i = 1, size(x, 1)
                if (x(i, self%estimators(j)%feature_index) == &
                        self%estimators(j)%threshold) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "xgboost predict_jvp: derivative is undefined on split")
                    return
                end if
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

        if (.not. allocated(self%estimators) .or. size(weights) /= 2 .or. &
            tree_index < 1 .or. tree_index > size(self%estimators)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost leaf_weights: tree index or output shape is invalid")
            return
        end if
        weights = [self%estimators(tree_index)%left_weight, &
            self%estimators(tree_index)%right_weight]
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_leaf_weights

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
        integer, allocatable :: order(:)
        integer :: n_samples, n_features, feature, k, i
        integer :: best_feature, best_left_count, best_right_count
        real(dp) :: total_gradient, total_hessian, left_gradient, left_hessian
        real(dp) :: right_gradient, right_hessian, candidate_gain, best_gain
        real(dp) :: candidate_threshold, best_threshold
        real(dp) :: best_left_weight, best_right_weight, value

        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (size(gradient) /= n_samples .or. size(hessian) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree: derivative array shape is invalid")
            return
        end if
        total_gradient = sum(gradient)
        total_hessian = sum(hessian)
        value = regularized_leaf_weight(total_gradient, total_hessian, options)
        tree%left_weight = value
        tree%right_weight = value
        tree%split_gain = 0.0_dp
        tree%has_split = .false.
        best_gain = 0.0_dp
        best_feature = 0
        best_threshold = 0.0_dp
        best_left_weight = value
        best_right_weight = value
        best_left_count = n_samples
        best_right_count = 0
        allocate(order(n_samples))

        do feature = 1, n_features
            call sort_feature_indices(x(:, feature), order)
            left_gradient = 0.0_dp
            left_hessian = 0.0_dp
            do k = 1, n_samples - 1
                i = order(k)
                left_gradient = left_gradient + gradient(i)
                left_hessian = left_hessian + hessian(i)
                if (k < options%min_samples_leaf .or. &
                    n_samples - k < options%min_samples_leaf) cycle
                if (x(order(k), feature) >= x(order(k + 1), feature)) cycle
                right_gradient = total_gradient - left_gradient
                right_hessian = total_hessian - left_hessian
                if (left_hessian < options%min_child_weight .or. &
                    right_hessian < options%min_child_weight) cycle
                candidate_gain = 0.5_dp*(regularized_leaf_score(left_gradient, &
                    left_hessian, options) + regularized_leaf_score(right_gradient, &
                    right_hessian, options) - regularized_leaf_score(total_gradient, &
                    total_hessian, options)) - options%gamma
                if (candidate_gain > best_gain) then
                    best_gain = candidate_gain
                    best_feature = feature
                    best_threshold = 0.5_dp*(x(order(k), feature) + &
                        x(order(k + 1), feature))
                    best_left_weight = regularized_leaf_weight(left_gradient, &
                        left_hessian, options)
                    best_right_weight = regularized_leaf_weight(right_gradient, &
                        right_hessian, options)
                    best_left_count = k
                    best_right_count = n_samples - k
                end if
            end do
        end do

        if (best_feature > 0) then
            tree%feature_index = best_feature
            tree%threshold = best_threshold
            tree%left_weight = best_left_weight
            tree%right_weight = best_right_weight
            tree%split_gain = best_gain
            tree%left_count = best_left_count
            tree%right_count = best_right_count
            tree%has_split = .true.
        else
            tree%left_count = best_left_count
            tree%right_count = best_right_count
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_tree

    subroutine tree_predict(tree, x, values, status)
        type(xgb_tree_t), intent(in) :: tree
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (size(values) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree prediction: output shape is invalid")
            return
        end if
        if (.not. tree%has_split) then
            values = tree%left_weight
        else
            do i = 1, size(x, 1)
                if (x(i, tree%feature_index) < tree%threshold) then
                    values(i) = tree%left_weight
                else
                    values(i) = tree%right_weight
                end if
            end do
        end if
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
