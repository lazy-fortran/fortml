!> A bounded LightGBM-style leaf-wise histogram boosting estimator.
module fortml_lightgbm
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: iostat_end, int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgb_histogram_cut_positions
    implicit none
    private

    integer, parameter, public :: LIGHTGBM_OBJECTIVE_REGRESSION = 1
    integer, parameter, public :: LIGHTGBM_OBJECTIVE_BINARY = 2
    character(*), parameter, public :: LIGHTGBM_MODEL_TEXT_MAGIC = &
        "FORTML_LIGHTGBM_TEXT"
    integer, parameter, public :: LIGHTGBM_MODEL_TEXT_SCHEMA_VERSION = 2
    integer, parameter, public :: LIGHTGBM_BOOSTING_GBDT = 0
    integer, parameter, public :: LIGHTGBM_BOOSTING_GOSS = 1
    integer, parameter :: LIGHTGBM_MAX_SERIALIZED_NODES = 1000000

    !> Options for the deterministic numeric LightGBM-style path.  The
    !> estimator intentionally exposes only the supported CPU core: weighted
    !> squared regression and binary logistic loss, weighted-quantile
    !> histograms, and best-first (leaf-wise) growth up to `num_leaves`.
    type, public :: lightgbm_options_t
        integer :: n_estimators = 50
        integer :: num_leaves = 31
        integer :: max_depth = 0
        integer :: min_data_in_leaf = 20
        integer :: max_bin = 255
        real(dp) :: learning_rate = 0.1_dp
        real(dp) :: l2 = 1.0_dp
        real(dp) :: min_gain_to_split = 0.0_dp
        character(len=32) :: objective = "regression"
        !! Evaluate an optional validation set after every boosting round.
        !! A positive value stops after this many consecutive rounds without
        !! an improvement larger than `early_stopping_min_delta`.
        integer :: early_stopping_rounds = 0
        real(dp) :: early_stopping_min_delta = 0.0_dp
        logical :: restore_best = .true.
        !! LightGBM's gradient-based one-side sampling.  `goss` retains
        !! `top_rate` of rows with the largest absolute gradients and a
        !! deterministic hash-selected `other_rate` fraction of the full
        !! sample from the rest;
        !! the latter gradients/Hessians are reweighted by
        !! `(1-top_rate)/other_rate` before leaf statistics are accumulated.
        character(len=16) :: boosting_type = "gbdt"
        real(dp) :: top_rate = 0.2_dp
        real(dp) :: other_rate = 0.1_dp
        integer(int64) :: seed = 104729_int64
    end type lightgbm_options_t

    type :: lgbm_node_t
        integer :: feature = 0
        integer :: left_child = 0
        integer :: right_child = 0
        integer :: depth = 0
        real(dp) :: threshold = 0.0_dp
        real(dp) :: weight = 0.0_dp
        real(dp) :: gain = 0.0_dp
        logical :: leaf = .true.
        integer, allocatable :: rows(:)
    end type lgbm_node_t

    type :: lgbm_tree_t
        integer :: n_nodes = 0
        integer :: depth = 0
        type(lgbm_node_t), allocatable :: node(:)
    end type lgbm_tree_t

    type :: lgbm_split_t
        logical :: valid = .false.
        integer :: feature = 0
        integer :: node = 0
        real(dp) :: threshold = 0.0_dp
        real(dp) :: gain = 0.0_dp
        real(dp) :: left_weight = 0.0_dp
        real(dp) :: right_weight = 0.0_dp
        integer, allocatable :: left_rows(:), right_rows(:)
    end type lgbm_split_t

    !> Numeric LightGBM-style estimator.  Unlike `xgboost_t`, which grows
    !> each tree recursively by depth, this policy evaluates all current leaf
    !> candidates and splits the globally best leaf until `num_leaves` is
    !> reached.  It is deliberately separate so the XGBoost API and growth
    !> semantics remain unchanged.
    type, public :: lightgbm_t
        private
        integer :: n_inputs = 0
        integer :: objective_code = 0
        integer :: n_estimators = 0
        integer :: requested_estimators = 0
        integer :: num_leaves_value = 0
        integer :: max_bin_value = 0
        integer :: max_depth_value = 0
        integer :: min_data_in_leaf_value = 0
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2_value = 0.0_dp
        real(dp) :: min_gain_value = 0.0_dp
        real(dp) :: base_score = 0.0_dp
        integer :: early_stopping_rounds_value = 0
        real(dp) :: early_stopping_min_delta_value = 0.0_dp
        logical :: restore_best_value = .true.
        integer :: boosting_type_code = LIGHTGBM_BOOSTING_GBDT
        real(dp) :: top_rate_value = 0.2_dp
        real(dp) :: other_rate_value = 0.1_dp
        integer(int64) :: seed_value = 104729_int64
        integer :: best_iteration_value = 0
        real(dp) :: best_validation_loss_value = huge(1.0_dp)
        logical :: early_stopped_flag = .false.
        type(lgbm_tree_t), allocatable :: estimator(:)
        logical :: initialized = .false.
    contains
        procedure, public :: fit => lgbm_fit
        procedure, public :: fit_regression => lgbm_fit_regression
        procedure, public :: fit_binary => lgbm_fit_binary
        procedure, public :: fit_warm_start => lgbm_fit_warm_start
        procedure, public :: predict => lgbm_predict
        procedure, public :: predict_margin => lgbm_predict_margin
        procedure, public :: predict_staged => lgbm_predict_staged
        procedure, public :: predict_staged_margin => lgbm_predict_staged_margin
        procedure, public :: predict_contributions => lgbm_predict_contributions
        procedure, public :: slice => lgbm_slice
        procedure, public :: save_text => lgbm_save_text
        procedure, public :: load_text => lgbm_load_text
        procedure, public :: predict_proba => lgbm_predict_proba
        procedure, public :: predict_device => lgbm_predict_device
        procedure, public :: predict_jvp => lgbm_predict_jvp
        procedure, public :: predict_vjp => lgbm_predict_vjp
        procedure, public :: device_supported => lgbm_device_supported
        procedure, public :: fitted => lgbm_fitted
        procedure, public :: objective_name => lgbm_objective_name
        procedure, public :: estimator_count => lgbm_estimator_count
        procedure, public :: best_iteration => lgbm_best_iteration
        procedure, public :: best_validation_loss => lgbm_best_validation_loss
        procedure, public :: early_stopped => lgbm_early_stopped
        procedure, public :: num_leaves => lgbm_num_leaves
        procedure, public :: tree_node_count => lgbm_tree_node_count
        procedure, public :: tree_depth => lgbm_tree_depth
        procedure, public :: feature_count => lgbm_feature_count
        procedure, public :: base_margin => lgbm_base_margin
        procedure, public :: boosting_type => lgbm_boosting_type
        procedure, public :: top_rate => lgbm_top_rate
        procedure, public :: other_rate => lgbm_other_rate
    end type lightgbm_t

contains

    subroutine lgbm_fit_regression(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        class(lightgbm_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(lightgbm_options_t) :: settings
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(lightgbm_options_t) :: lightgbm_options_t_default

        settings = lightgbm_options_t_default
        if (present(options)) settings = options
        settings%objective = "regression"
        call lgbm_fit(self, x, y, status, settings, sample_weight, validation_x, &
            validation_y, validation_weight)
    end subroutine lgbm_fit_regression

    subroutine lgbm_fit_binary(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        class(lightgbm_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(lightgbm_options_t) :: settings
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(lightgbm_options_t) :: lightgbm_options_t_default

        settings = lightgbm_options_t_default
        if (present(options)) settings = options
        settings%objective = "binary"
        call lgbm_fit(self, x, y, status, settings, sample_weight, validation_x, &
            validation_y, validation_weight)
    end subroutine lgbm_fit_binary

    !> Continue a fitted LightGBM ensemble to a larger `n_estimators` target.
    !>
    !> The existing tree prefix is copied into temporary storage and only
    !> committed after all new trees succeed.  Options other than
    !> `n_estimators` must match the fitted prefix exactly; validation/early
    !> stopping controls are refused in this bounded continuation API because
    !> the source does not retain training rows or validation state.
    subroutine lgbm_fit_warm_start(self, x, y, status, options, sample_weight)
        class(lightgbm_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        type(lightgbm_options_t) :: settings
        type(lgbm_tree_t), allocatable :: expanded_estimators(:)
        real(dp), allocatable :: weights(:), margin(:), gradient(:), hessian(:), correction(:)
        real(dp), allocatable :: gradient_for_tree(:), hessian_for_tree(:), row_scale(:)
        integer, allocatable :: rows(:), sampled_rows(:)
        integer :: objective_code, boosting_type_code, n_samples, n_features, start_estimators, target_estimators
        integer :: i, j
        real(dp) :: weight_sum

        if (.not. self%initialized .or. .not. allocated(self%estimator)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm warm start: source is not a valid fitted model")
            return
        end if
        if (.not. present(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm warm start: options with a larger n_estimators is required")
            return
        end if
        settings = options
        start_estimators = self%n_estimators
        target_estimators = settings%n_estimators
        if (start_estimators < 1 .or. size(self%estimator) /= start_estimators .or. &
            target_estimators <= start_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm warm start: n_estimators must exceed the fitted prefix")
            return
        end if
        objective_code = parse_lgbm_objective(settings%objective)
        boosting_type_code = parse_lgbm_boosting_type(settings%boosting_type)
        if (objective_code == 0 .or. objective_code /= self%objective_code .or. &
            boosting_type_code /= self%boosting_type_code) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm warm start: objective and boosting type must match the fitted prefix")
            return
        end if
        if (settings%num_leaves /= self%num_leaves_value .or. &
            settings%max_bin /= self%max_bin_value .or. &
            settings%max_depth /= self%max_depth_value .or. &
            settings%min_data_in_leaf /= self%min_data_in_leaf_value .or. &
            settings%learning_rate /= self%learning_rate .or. &
            settings%l2 /= self%l2_value .or. &
            settings%min_gain_to_split /= self%min_gain_value .or. &
            settings%top_rate /= self%top_rate_value .or. settings%other_rate /= self%other_rate_value .or. &
            settings%seed /= self%seed_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm warm start: tree options must match the fitted prefix")
            return
        end if
        if (settings%early_stopping_rounds /= 0 .or. &
            settings%early_stopping_min_delta /= 0.0_dp) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "lightgbm warm start: validation early stopping is not retained")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 2 .or. n_features /= self%n_inputs .or. size(y) /= n_samples .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm warm start: input dimensions or finite-value contract failed")
            return
        end if
        if (objective_code == LIGHTGBM_OBJECTIVE_BINARY) then
            if (any(y < 0.0_dp) .or. any(y > 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm warm start: binary targets must lie in [0,1]")
                return
            end if
        end if
        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm warm start: sample weights must be finite and positive")
                return
            end if
            weights = sample_weight
        end if
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm warm start: sample weights have no positive mass")
            return
        end if
        if (settings%seed <= 0_int64 .or. (boosting_type_code == LIGHTGBM_BOOSTING_GOSS .and. &
            (settings%top_rate <= 0.0_dp .or. settings%top_rate >= 1.0_dp .or. &
            settings%other_rate <= 0.0_dp .or. settings%other_rate > 1.0_dp .or. &
            settings%top_rate + settings%other_rate >= 1.0_dp))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm warm start: invalid GOSS rates or seed")
            return
        end if
        allocate(margin(n_samples), gradient(n_samples), hessian(n_samples), &
            gradient_for_tree(n_samples), hessian_for_tree(n_samples), correction(n_samples))
        allocate(rows(n_samples))
        do j = 1, n_samples
            rows(j) = j
        end do
        margin = self%base_score
        do i = 1, start_estimators
            call lgbm_tree_predict(self%estimator(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + self%learning_rate*correction
        end do
        allocate(expanded_estimators(target_estimators))
        expanded_estimators(:start_estimators) = self%estimator
        do i = start_estimators + 1, target_estimators
            call lgbm_objective_derivatives(objective_code, margin, y, weights, gradient, &
                hessian, status)
            if (status%code /= FORTNUM_OK) return
            gradient_for_tree = gradient
            hessian_for_tree = hessian
            if (boosting_type_code == LIGHTGBM_BOOSTING_GOSS) then
                call select_goss_rows(gradient, settings, i, sampled_rows, row_scale, status)
                if (status%code /= FORTNUM_OK) return
                do j = 1, size(sampled_rows)
                    gradient_for_tree(sampled_rows(j)) = gradient(sampled_rows(j))*row_scale(j)
                    hessian_for_tree(sampled_rows(j)) = hessian(sampled_rows(j))*row_scale(j)
                end do
            else
                sampled_rows = rows
            end if
            call build_leafwise_tree(x, gradient_for_tree, hessian_for_tree, weights, settings, sampled_rows, &
                expanded_estimators(i), status)
            if (status%code /= FORTNUM_OK) return
            call lgbm_tree_predict(expanded_estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + self%learning_rate*correction
        end do
        call move_alloc(expanded_estimators, self%estimator)
        self%n_estimators = target_estimators
        self%requested_estimators = target_estimators
        self%early_stopping_rounds_value = 0
        self%early_stopping_min_delta_value = 0.0_dp
        self%best_iteration_value = target_estimators
        self%best_validation_loss_value = huge(1.0_dp)
        self%early_stopped_flag = .false.
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_fit_warm_start

    subroutine lgbm_fit(self, x, y, status, options, sample_weight, validation_x, &
            validation_y, validation_weight)
        class(lightgbm_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(lightgbm_options_t) :: settings
        real(dp), allocatable :: weights(:), margin(:), gradient(:), hessian(:), correction(:)
        real(dp), allocatable :: gradient_for_tree(:), hessian_for_tree(:), row_scale(:)
        real(dp), allocatable :: validation_weights(:), validation_margin(:), &
            validation_correction(:)
        type(lgbm_tree_t), allocatable :: best_estimators(:), retained_estimators(:)
        integer, allocatable :: rows(:), sampled_rows(:)
        integer :: objective_code, boosting_type_code, i, n_samples, n_features, n_validation
        integer :: completed_estimators, best_iteration, stale_rounds, j
        real(dp) :: weight_sum, mean_target, validation_loss, best_validation_loss
        logical :: have_validation, improved
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(lightgbm_options_t) :: lightgbm_options_t_default

        settings = lightgbm_options_t_default
        if (present(options)) settings = options
        have_validation = present(validation_x) .or. present(validation_y) .or. &
            present(validation_weight)
        if (present(validation_x) .neqv. present(validation_y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: validation_x and validation_y must be supplied together")
            return
        end if
        if (present(validation_weight) .and. .not. present(validation_x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: validation_weight requires validation data")
            return
        end if
        if (settings%early_stopping_rounds < 0 .or. &
            .not. ieee_is_finite(settings%early_stopping_min_delta) .or. &
            settings%early_stopping_min_delta < 0.0_dp .or. &
            (settings%early_stopping_rounds > 0 .and. .not. have_validation)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: invalid early-stopping configuration")
            return
        end if
        objective_code = parse_lgbm_objective(settings%objective)
        if (objective_code == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: objective must be regression or binary")
            return
        end if
        boosting_type_code = parse_lgbm_boosting_type(settings%boosting_type)
        if (boosting_type_code < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: boosting_type must be gbdt or goss")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (have_validation) then
            n_validation = size(validation_x, 1)
            if (n_validation < 1 .or. size(validation_x, 2) /= n_features .or. &
                size(validation_y) /= n_validation) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm fit: validation dimensions do not match training features")
                return
            end if
        else
            n_validation = 0
        end if
        if (n_samples < 2 .or. n_features < 1 .or. size(y) /= n_samples .or. &
            settings%n_estimators < 1 .or. settings%num_leaves < 2 .or. &
            settings%num_leaves > n_samples .or. settings%max_depth < 0 .or. &
            settings%min_data_in_leaf < 1 .or. &
            2*settings%min_data_in_leaf > n_samples .or. settings%max_bin < 2 .or. &
            .not. ieee_is_finite(settings%learning_rate) .or. &
            settings%learning_rate <= 0.0_dp .or. settings%learning_rate > 1.0_dp .or. &
            .not. ieee_is_finite(settings%l2) .or. settings%l2 < 0.0_dp .or. &
            .not. ieee_is_finite(settings%min_gain_to_split) .or. &
            settings%min_gain_to_split < 0.0_dp .or. settings%seed <= 0_int64 .or. &
            (boosting_type_code == LIGHTGBM_BOOSTING_GOSS .and. &
            (.not. ieee_is_finite(settings%top_rate) .or. &
            .not. ieee_is_finite(settings%other_rate) .or. settings%top_rate <= 0.0_dp .or. &
            settings%top_rate >= 1.0_dp .or. settings%other_rate <= 0.0_dp .or. &
            settings%other_rate > 1.0_dp .or. settings%top_rate + settings%other_rate >= 1.0_dp))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: invalid dimensions or options")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: only finite numeric inputs are supported")
            return
        end if
        if (have_validation) then
            if (any(.not. ieee_is_finite(validation_x)) .or. &
                any(.not. ieee_is_finite(validation_y))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm fit: validation inputs must be finite")
                return
            end if
        end if
        if (objective_code == LIGHTGBM_OBJECTIVE_BINARY) then
            if (any(y < 0.0_dp) .or. any(y > 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm fit: binary targets must lie in [0,1]")
                return
            end if
            if (have_validation) then
                if (any(validation_y < 0.0_dp) .or. any(validation_y > 1.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "lightgbm fit: validation binary targets must lie in [0,1]")
                    return
                end if
            end if
        end if
        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm fit: sample weights must be finite and positive")
                return
            end if
            weights = sample_weight
        end if
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: sample weights have no positive mass")
            return
        end if
        if (have_validation) then
            allocate(validation_weights(n_validation), validation_margin(n_validation), &
                validation_correction(n_validation))
            validation_weights = 1.0_dp
            if (present(validation_weight)) then
                if (size(validation_weight) /= n_validation .or. &
                    any(.not. ieee_is_finite(validation_weight)) .or. &
                    any(validation_weight <= 0.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "lightgbm fit: validation weights must be finite and positive")
                    return
                end if
                validation_weights = validation_weight
            end if
            if (.not. ieee_is_finite(sum(validation_weights)) .or. &
                sum(validation_weights) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm fit: validation weights have no positive mass")
                return
            end if
        end if
        mean_target = sum(weights*y)/weight_sum
        if (objective_code == LIGHTGBM_OBJECTIVE_BINARY) then
            self%base_score = stable_logit(mean_target)
        else
            self%base_score = mean_target
        end if
        allocate(self%estimator(settings%n_estimators), margin(n_samples), &
            gradient(n_samples), hessian(n_samples), gradient_for_tree(n_samples), &
            hessian_for_tree(n_samples), correction(n_samples), rows(n_samples))
        rows = [(i, i=1,n_samples)]
        margin = self%base_score
        if (have_validation) then
            validation_margin = self%base_score
            best_validation_loss = huge(1.0_dp)
            best_iteration = 0
            stale_rounds = 0
        else
            best_validation_loss = huge(1.0_dp)
            best_iteration = settings%n_estimators
            stale_rounds = 0
        end if
        completed_estimators = 0
        do i = 1, settings%n_estimators
            call lgbm_objective_derivatives(objective_code, margin, y, weights, gradient, &
                hessian, status)
            if (status%code /= FORTNUM_OK) return
            gradient_for_tree = gradient
            hessian_for_tree = hessian
            if (boosting_type_code == LIGHTGBM_BOOSTING_GOSS) then
                call select_goss_rows(gradient, settings, i, sampled_rows, row_scale, status)
                if (status%code /= FORTNUM_OK) return
                do j = 1, size(sampled_rows)
                    gradient_for_tree(sampled_rows(j)) = gradient(sampled_rows(j))*row_scale(j)
                    hessian_for_tree(sampled_rows(j)) = hessian(sampled_rows(j))*row_scale(j)
                end do
            else
                sampled_rows = rows
            end if
            call build_leafwise_tree(x, gradient_for_tree, hessian_for_tree, weights, settings, sampled_rows, &
                self%estimator(i), status)
            if (status%code /= FORTNUM_OK) return
            call lgbm_tree_predict(self%estimator(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + settings%learning_rate*correction
            completed_estimators = i
            if (have_validation) then
                call lgbm_tree_predict(self%estimator(i), validation_x, &
                    validation_correction, status)
                if (status%code /= FORTNUM_OK) return
                validation_margin = validation_margin + settings%learning_rate* &
                    validation_correction
                call lgbm_objective_loss(objective_code, validation_margin, validation_y, &
                    validation_weights, validation_loss, status)
                if (status%code /= FORTNUM_OK) return
                improved = validation_loss < best_validation_loss - &
                    settings%early_stopping_min_delta
                if (improved) then
                    best_validation_loss = validation_loss
                    best_iteration = i
                    stale_rounds = 0
                    if (settings%restore_best) best_estimators = self%estimator(:i)
                else
                    stale_rounds = stale_rounds + 1
                end if
                if (settings%early_stopping_rounds > 0 .and. &
                    stale_rounds >= settings%early_stopping_rounds) then
                    self%early_stopped_flag = .true.
                    exit
                end if
            end if
        end do
        if (have_validation) then
            if (best_iteration < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm fit: validation objective did not produce a finite score")
                return
            end if
            self%best_iteration_value = best_iteration
            self%best_validation_loss_value = best_validation_loss
            if (settings%restore_best .and. allocated(best_estimators)) then
                if (allocated(retained_estimators)) deallocate(retained_estimators)
                allocate(retained_estimators(best_iteration))
                retained_estimators = best_estimators
                call move_alloc(retained_estimators, self%estimator)
                completed_estimators = best_iteration
            else if (completed_estimators < settings%n_estimators) then
                allocate(retained_estimators(completed_estimators))
                retained_estimators = self%estimator(:completed_estimators)
                call move_alloc(retained_estimators, self%estimator)
            end if
        else
            self%best_iteration_value = settings%n_estimators
            self%best_validation_loss_value = huge(1.0_dp)
        end if
        self%n_inputs = n_features
        self%objective_code = objective_code
        self%n_estimators = completed_estimators
        self%requested_estimators = settings%n_estimators
        self%num_leaves_value = settings%num_leaves
        self%max_bin_value = settings%max_bin
        self%max_depth_value = settings%max_depth
        self%min_data_in_leaf_value = settings%min_data_in_leaf
        self%learning_rate = settings%learning_rate
        self%l2_value = settings%l2
        self%min_gain_value = settings%min_gain_to_split
        self%early_stopping_rounds_value = settings%early_stopping_rounds
        self%early_stopping_min_delta_value = settings%early_stopping_min_delta
        self%restore_best_value = settings%restore_best
        self%boosting_type_code = boosting_type_code
        self%top_rate_value = settings%top_rate
        self%other_rate_value = settings%other_rate
        self%seed_value = settings%seed
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_fit

    subroutine lgbm_predict(self, x, y, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status

        call lgbm_predict_margin(self, x, y, status)
        if (status%code /= FORTNUM_OK) return
        if (self%objective_code == LIGHTGBM_OBJECTIVE_BINARY) y = stable_sigmoid_array(y)
    end subroutine lgbm_predict

    subroutine lgbm_predict_margin(self, x, margin, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margin(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(margin) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict: model, shape, or finite-input contract failed")
            return
        end if
        allocate(correction(size(x, 1)))
        margin = self%base_score
        do i = 1, self%n_estimators
            call lgbm_tree_predict(self%estimator(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + self%learning_rate*correction
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_margin

    !> Return cumulative raw margins after every fitted boosting round.
    !>
    !> The second dimension is ordered from the first fitted tree through the
    !> retained ensemble.  This is deliberately a structural product: it
    !> never refits or changes the leaf-wise tree state.
    subroutine lgbm_predict_staged_margin(self, x, staged, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: staged(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_staged_margin: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_inputs .or. size(staged, 1) /= size(x, 1) .or. &
            size(staged, 2) /= self%n_estimators .or. self%n_estimators < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_staged_margin: output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_staged_margin: input has unsupported nonfinite values")
            return
        end if
        allocate(correction(size(x, 1)))
        staged = 0.0_dp
        do i = 1, self%n_estimators
            call lgbm_tree_predict(self%estimator(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            if (i == 1) then
                staged(:, i) = self%base_score + self%learning_rate*correction
            else
                staged(:, i) = staged(:, i-1) + self%learning_rate*correction
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_staged_margin

    !> Return the fitted prediction after every boosting round.
    !>
    !> Regression stages are raw margins.  Binary stages are positive-class
    !> probabilities, matching `predict`; use `predict_staged_margin` when the
    !> additive link-scale values are required.
    subroutine lgbm_predict_staged(self, x, staged, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: staged(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        call self%predict_staged_margin(x, staged, status)
        if (status%code /= FORTNUM_OK) return
        if (self%objective_code == LIGHTGBM_OBJECTIVE_BINARY) then
            do i = 1, size(staged, 2)
                staged(:, i) = stable_sigmoid_array(staged(:, i))
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_staged

    !> Return additive raw-margin contributions for the fitted ensemble.
    !>
    !> Column one is the base margin and column `i+1` is the learning-rate
    !> scaled contribution of tree `i`.  Summing columns reproduces
    !> `predict_margin`; for binary objectives apply the sigmoid only after
    !> summing the raw-link contributions.
    subroutine lgbm_predict_contributions(self, x, contributions, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: contributions(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_contributions: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_inputs .or. size(contributions, 1) /= size(x, 1) .or. &
            size(contributions, 2) /= self%n_estimators + 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_contributions: output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_contributions: input has unsupported nonfinite values")
            return
        end if
        allocate(correction(size(x, 1)))
        contributions = 0.0_dp
        contributions(:, 1) = self%base_score
        do i = 1, self%n_estimators
            call lgbm_tree_predict(self%estimator(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            contributions(:, i+1) = self%learning_rate*correction
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_contributions

    !> Copy a fitted LightGBM prefix into `destination` without refitting.
    !>
    !> The operation is transactional: malformed requests leave `destination`
    !> unchanged.  Tree arrays use intrinsic allocatable assignment, so all
    !> node/row state is copied rather than aliased.
    subroutine lgbm_slice(self, n_trees, destination, status)
        class(lightgbm_t), intent(in) :: self
        integer, intent(in) :: n_trees
        type(lightgbm_t), intent(inout) :: destination
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_t) :: candidate

        if (.not. self%initialized .or. .not. allocated(self%estimator)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm slice: source is not a valid fitted ensemble")
            return
        end if
        if (self%n_estimators < 1 .or. size(self%estimator) /= self%n_estimators .or. &
            n_trees < 1 .or. n_trees > self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm slice: requested prefix is invalid")
            return
        end if
        candidate%n_inputs = self%n_inputs
        candidate%objective_code = self%objective_code
        candidate%n_estimators = n_trees
        candidate%requested_estimators = max(self%requested_estimators, n_trees)
        candidate%num_leaves_value = self%num_leaves_value
        candidate%max_bin_value = self%max_bin_value
        candidate%max_depth_value = self%max_depth_value
        candidate%min_data_in_leaf_value = self%min_data_in_leaf_value
        candidate%learning_rate = self%learning_rate
        candidate%l2_value = self%l2_value
        candidate%min_gain_value = self%min_gain_value
        candidate%base_score = self%base_score
        candidate%early_stopping_rounds_value = self%early_stopping_rounds_value
        candidate%early_stopping_min_delta_value = self%early_stopping_min_delta_value
        candidate%restore_best_value = self%restore_best_value
        candidate%boosting_type_code = self%boosting_type_code
        candidate%top_rate_value = self%top_rate_value
        candidate%other_rate_value = self%other_rate_value
        candidate%seed_value = self%seed_value
        candidate%best_iteration_value = min(max(self%best_iteration_value, 0), n_trees)
        candidate%best_validation_loss_value = self%best_validation_loss_value
        candidate%early_stopped_flag = self%early_stopped_flag
        allocate(candidate%estimator(n_trees))
        candidate%estimator = self%estimator(:n_trees)
        candidate%initialized = .true.
        destination = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_slice

    !> Save a fitted LightGBM ensemble as a versioned, compiler-independent
    !> formatted text snapshot.  Training-row membership is intentionally not
    !> serialized: prediction and all staged/contribution products depend only
    !> on the learned node arrays.
    subroutine lgbm_save_text(self, path, status)
        class(lightgbm_t), intent(in) :: self
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        integer :: unit, ios, close_ios, i, j

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm save_text: model is not fitted")
            return
        end if
        if (.not. valid_lgbm_scalars(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm save_text: model metadata is invalid")
            return
        end if
        if (.not. allocated(self%estimator) .or. size(self%estimator) /= self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm save_text: tree storage is invalid")
            return
        end if
        do i = 1, self%n_estimators
            if (.not. valid_lgbm_tree(self%estimator(i), self%n_inputs)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm save_text: model contains an invalid tree")
                return
            end if
        end do
        open(newunit=unit, file=path, status="replace", action="write", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm save_text: cannot open destination")
            return
        end if
        write(unit, "(A)", iostat=ios) LIGHTGBM_MODEL_TEXT_MAGIC
        if (ios == 0) call lgbm_write_i(unit, "schema_version", &
            LIGHTGBM_MODEL_TEXT_SCHEMA_VERSION, ios)
        if (ios == 0) call lgbm_write_i(unit, "n_inputs", self%n_inputs, ios)
        if (ios == 0) call lgbm_write_i(unit, "n_estimators", self%n_estimators, ios)
        if (ios == 0) call lgbm_write_i(unit, "requested_estimators", &
            self%requested_estimators, ios)
        if (ios == 0) call lgbm_write_i(unit, "objective_code", self%objective_code, ios)
        if (ios == 0) call lgbm_write_i(unit, "num_leaves", self%num_leaves_value, ios)
        if (ios == 0) call lgbm_write_i(unit, "max_bin", self%max_bin_value, ios)
        if (ios == 0) call lgbm_write_i(unit, "max_depth", self%max_depth_value, ios)
        if (ios == 0) call lgbm_write_i(unit, "min_data_in_leaf", &
            self%min_data_in_leaf_value, ios)
        if (ios == 0) call lgbm_write_r(unit, "learning_rate", self%learning_rate, ios)
        if (ios == 0) call lgbm_write_r(unit, "l2", self%l2_value, ios)
        if (ios == 0) call lgbm_write_r(unit, "min_gain_to_split", self%min_gain_value, ios)
        if (ios == 0) call lgbm_write_r(unit, "base_score", self%base_score, ios)
        if (ios == 0) call lgbm_write_i(unit, "early_stopping_rounds", &
            self%early_stopping_rounds_value, ios)
        if (ios == 0) call lgbm_write_r(unit, "early_stopping_min_delta", &
            self%early_stopping_min_delta_value, ios)
        if (ios == 0) call lgbm_write_l(unit, "restore_best", self%restore_best_value, ios)
        if (ios == 0) call lgbm_write_i(unit, "best_iteration", self%best_iteration_value, ios)
        if (ios == 0) call lgbm_write_r(unit, "best_validation_loss", &
            self%best_validation_loss_value, ios)
        if (ios == 0) call lgbm_write_l(unit, "early_stopped", self%early_stopped_flag, ios)
        if (ios == 0) call lgbm_write_i(unit, "boosting_type_code", self%boosting_type_code, ios)
        if (ios == 0) call lgbm_write_r(unit, "top_rate", self%top_rate_value, ios)
        if (ios == 0) call lgbm_write_r(unit, "other_rate", self%other_rate_value, ios)
        if (ios == 0) call lgbm_write_i64(unit, "seed", self%seed_value, ios)
        if (ios == 0) call lgbm_write_i(unit, "tree_count", self%n_estimators, ios)
        do i = 1, self%n_estimators
            if (ios /= 0) exit
            call lgbm_write_i(unit, "tree_begin", i, ios)
            if (ios == 0) call lgbm_write_i(unit, "n_nodes", self%estimator(i)%n_nodes, ios)
            if (ios == 0) call lgbm_write_i(unit, "depth", self%estimator(i)%depth, ios)
            do j = 1, self%estimator(i)%n_nodes
                if (ios /= 0) exit
                call lgbm_write_i(unit, "node_begin", j, ios)
                if (ios == 0) call lgbm_write_i(unit, "feature", &
                    self%estimator(i)%node(j)%feature, ios)
                if (ios == 0) call lgbm_write_i(unit, "left_child", &
                    self%estimator(i)%node(j)%left_child, ios)
                if (ios == 0) call lgbm_write_i(unit, "right_child", &
                    self%estimator(i)%node(j)%right_child, ios)
                if (ios == 0) call lgbm_write_i(unit, "node_depth", &
                    self%estimator(i)%node(j)%depth, ios)
                if (ios == 0) call lgbm_write_r(unit, "threshold", &
                    self%estimator(i)%node(j)%threshold, ios)
                if (ios == 0) call lgbm_write_r(unit, "weight", &
                    self%estimator(i)%node(j)%weight, ios)
                if (ios == 0) call lgbm_write_r(unit, "gain", &
                    self%estimator(i)%node(j)%gain, ios)
                if (ios == 0) call lgbm_write_l(unit, "leaf", &
                    self%estimator(i)%node(j)%leaf, ios)
            end do
        end do
        if (ios == 0) write(unit, "(A)", iostat=ios) "end"
        close_ios = 0
        close(unit, iostat=close_ios)
        if (ios /= 0 .or. close_ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm save_text: formatted write failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_save_text

    !> Load a complete LightGBM text snapshot transactionally.  Every named
    !> record, tree index, node index, scalar bound, and EOF marker is checked;
    !> unknown, truncated, duplicate, or trailing records refuse without
    !> mutating the destination model.
    subroutine lgbm_load_text(self, path, status)
        class(lightgbm_t), intent(inout) :: self
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_t) :: candidate
        character(len=256) :: line
        integer :: unit, ios, close_ios, schema, i, j, tree_count

        open(newunit=unit, file=path, status="old", action="read", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm load_text: cannot open source")
            return
        end if
        read(unit, "(A)", iostat=ios) line
        if (ios /= 0 .or. trim(line) /= LIGHTGBM_MODEL_TEXT_MAGIC) goto 900
        call lgbm_read_i(unit, "schema_version", schema, ios)
        if (ios /= 0 .or. schema /= LIGHTGBM_MODEL_TEXT_SCHEMA_VERSION) goto 900
        call lgbm_read_i(unit, "n_inputs", candidate%n_inputs, ios)
        if (ios == 0) call lgbm_read_i(unit, "n_estimators", candidate%n_estimators, ios)
        if (ios == 0) call lgbm_read_i(unit, "requested_estimators", &
            candidate%requested_estimators, ios)
        if (ios == 0) call lgbm_read_i(unit, "objective_code", candidate%objective_code, ios)
        if (ios == 0) call lgbm_read_i(unit, "num_leaves", candidate%num_leaves_value, ios)
        if (ios == 0) call lgbm_read_i(unit, "max_bin", candidate%max_bin_value, ios)
        if (ios == 0) call lgbm_read_i(unit, "max_depth", candidate%max_depth_value, ios)
        if (ios == 0) call lgbm_read_i(unit, "min_data_in_leaf", &
            candidate%min_data_in_leaf_value, ios)
        if (ios == 0) call lgbm_read_r(unit, "learning_rate", candidate%learning_rate, ios)
        if (ios == 0) call lgbm_read_r(unit, "l2", candidate%l2_value, ios)
        if (ios == 0) call lgbm_read_r(unit, "min_gain_to_split", candidate%min_gain_value, ios)
        if (ios == 0) call lgbm_read_r(unit, "base_score", candidate%base_score, ios)
        if (ios == 0) call lgbm_read_i(unit, "early_stopping_rounds", &
            candidate%early_stopping_rounds_value, ios)
        if (ios == 0) call lgbm_read_r(unit, "early_stopping_min_delta", &
            candidate%early_stopping_min_delta_value, ios)
        if (ios == 0) call lgbm_read_l(unit, "restore_best", candidate%restore_best_value, ios)
        if (ios == 0) call lgbm_read_i(unit, "best_iteration", candidate%best_iteration_value, ios)
        if (ios == 0) call lgbm_read_r(unit, "best_validation_loss", &
            candidate%best_validation_loss_value, ios)
        if (ios == 0) call lgbm_read_l(unit, "early_stopped", candidate%early_stopped_flag, ios)
        if (ios == 0) call lgbm_read_i(unit, "boosting_type_code", candidate%boosting_type_code, ios)
        if (ios == 0) call lgbm_read_r(unit, "top_rate", candidate%top_rate_value, ios)
        if (ios == 0) call lgbm_read_r(unit, "other_rate", candidate%other_rate_value, ios)
        if (ios == 0) call lgbm_read_i64(unit, "seed", candidate%seed_value, ios)
        if (ios /= 0 .or. .not. valid_lgbm_scalars(candidate)) goto 900
        call lgbm_read_i(unit, "tree_count", tree_count, ios)
        if (ios /= 0 .or. tree_count /= candidate%n_estimators) goto 900
        allocate(candidate%estimator(tree_count), stat=ios)
        if (ios /= 0) goto 900
        do i = 1, tree_count
            call lgbm_read_i(unit, "tree_begin", j, ios)
            if (ios /= 0 .or. j /= i) goto 900
            call lgbm_read_i(unit, "n_nodes", candidate%estimator(i)%n_nodes, ios)
            if (ios == 0) call lgbm_read_i(unit, "depth", candidate%estimator(i)%depth, ios)
            if (ios /= 0 .or. candidate%estimator(i)%n_nodes < 1 .or. &
                candidate%estimator(i)%n_nodes > LIGHTGBM_MAX_SERIALIZED_NODES) goto 900
            allocate(candidate%estimator(i)%node(candidate%estimator(i)%n_nodes), stat=ios)
            if (ios /= 0) goto 900
            do j = 1, candidate%estimator(i)%n_nodes
                call lgbm_read_i(unit, "node_begin", tree_count, ios)
                if (ios /= 0 .or. tree_count /= j) goto 900
                call lgbm_read_i(unit, "feature", candidate%estimator(i)%node(j)%feature, ios)
                if (ios == 0) call lgbm_read_i(unit, "left_child", &
                    candidate%estimator(i)%node(j)%left_child, ios)
                if (ios == 0) call lgbm_read_i(unit, "right_child", &
                    candidate%estimator(i)%node(j)%right_child, ios)
                if (ios == 0) call lgbm_read_i(unit, "node_depth", &
                    candidate%estimator(i)%node(j)%depth, ios)
                if (ios == 0) call lgbm_read_r(unit, "threshold", &
                    candidate%estimator(i)%node(j)%threshold, ios)
                if (ios == 0) call lgbm_read_r(unit, "weight", &
                    candidate%estimator(i)%node(j)%weight, ios)
                if (ios == 0) call lgbm_read_r(unit, "gain", &
                    candidate%estimator(i)%node(j)%gain, ios)
                if (ios == 0) call lgbm_read_l(unit, "leaf", &
                    candidate%estimator(i)%node(j)%leaf, ios)
                if (ios /= 0) goto 900
            end do
            if (.not. valid_lgbm_tree(candidate%estimator(i), candidate%n_inputs)) goto 900
        end do
        read(unit, "(A)", iostat=ios) line
        if (ios /= 0 .or. trim(line) /= "end") goto 900
        read(unit, "(A)", iostat=ios) line
        if (ios /= iostat_end) goto 900
        close_ios = 0
        close(unit, iostat=close_ios)
        if (close_ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm load_text: close failed")
            return
        end if
        candidate%initialized = .true.
        select type (self)
            type is (lightgbm_t)
            self = candidate
        class default
            goto 900
        end select
        call status_set(status, FORTNUM_OK, "")
        return
        900     close_ios = 0
        close(unit, iostat=close_ios)
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            "lightgbm load_text: malformed, truncated, or trailing record")
    end subroutine lgbm_load_text

    subroutine lgbm_predict_proba(self, x, probabilities, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:)

        if (self%objective_code /= LIGHTGBM_OBJECTIVE_BINARY) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_proba: only binary objective has probabilities")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. size(probabilities, 2) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_proba: output shape is invalid")
            return
        end if
        allocate(margin(size(x, 1)))
        call lgbm_predict_margin(self, x, margin, status)
        if (status%code /= FORTNUM_OK) return
        probabilities(:, 2) = stable_sigmoid_array(margin)
        probabilities(:, 1) = 1.0_dp - probabilities(:, 2)
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_proba

    subroutine lgbm_predict_device(self, device, x, y, status)
        class(lightgbm_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, y, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "lightgbm device prediction: no resident CUDA histogram kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm device prediction: device kind is invalid")
        end select
    end subroutine lgbm_predict_device

    subroutine lgbm_predict_jvp(self, x, x_dot, y, y_dot, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: y(:), y_dot(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2) .or. &
            size(y) /= size(x, 1) .or. size(y_dot) /= size(y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_jvp: shape is invalid")
            return
        end if
        call self%predict(x, y, status)
        if (status%code /= FORTNUM_OK) return
        y_dot = 0.0_dp
        if (leaf_boundary_hit(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_jvp: split-boundary derivative is undefined")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_jvp

    subroutine lgbm_predict_vjp(self, x, output_bar, x_bar, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: y(:)

        if (size(output_bar) /= size(x, 1) .or. size(x_bar, 1) /= size(x, 1) .or. &
            size(x_bar, 2) /= size(x, 2) .or. any(.not. ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_vjp: shape or cotangent is invalid")
            return
        end if
        allocate(y(size(x, 1)))
        call self%predict(x, y, status)
        if (status%code /= FORTNUM_OK) return
        x_bar = 0.0_dp
        if (leaf_boundary_hit(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_vjp: split-boundary derivative is undefined")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_vjp

    logical function lgbm_device_supported(self, device_kind) result(supported)
        class(lightgbm_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%initialized .and. device_kind == FORTML_DEVICE_CPU
    end function lgbm_device_supported

    logical function lgbm_fitted(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%initialized
    end function lgbm_fitted

    character(len=16) function lgbm_objective_name(self) result(name)
        class(lightgbm_t), intent(in) :: self
        if (self%objective_code == LIGHTGBM_OBJECTIVE_BINARY) then
            name = "binary"
        else if (self%objective_code == LIGHTGBM_OBJECTIVE_REGRESSION) then
            name = "regression"
        else
            name = "unfitted"
        end if
    end function lgbm_objective_name

    integer function lgbm_estimator_count(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%n_estimators
    end function lgbm_estimator_count

    integer function lgbm_best_iteration(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%best_iteration_value
    end function lgbm_best_iteration

    real(dp) function lgbm_best_validation_loss(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%best_validation_loss_value
    end function lgbm_best_validation_loss

    logical function lgbm_early_stopped(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%early_stopped_flag
    end function lgbm_early_stopped

    integer function lgbm_num_leaves(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%num_leaves_value
    end function lgbm_num_leaves

    integer function lgbm_tree_node_count(self, tree_index) result(value)
        class(lightgbm_t), intent(in) :: self
        integer, intent(in) :: tree_index
        value = 0
        if (.not. self%initialized .or. tree_index < 1 .or. tree_index > self%n_estimators) return
        value = self%estimator(tree_index)%n_nodes
    end function lgbm_tree_node_count

    integer function lgbm_tree_depth(self, tree_index) result(value)
        class(lightgbm_t), intent(in) :: self
        integer, intent(in) :: tree_index
        value = 0
        if (.not. self%initialized .or. tree_index < 1 .or. tree_index > self%n_estimators) return
        value = self%estimator(tree_index)%depth
    end function lgbm_tree_depth

    integer function lgbm_feature_count(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%n_inputs
    end function lgbm_feature_count

    real(dp) function lgbm_base_margin(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%base_score
    end function lgbm_base_margin

    character(len=16) function lgbm_boosting_type(self) result(name)
        class(lightgbm_t), intent(in) :: self
        if (self%boosting_type_code == LIGHTGBM_BOOSTING_GOSS) then
            name = "goss"
        else if (self%boosting_type_code == LIGHTGBM_BOOSTING_GBDT) then
            name = "gbdt"
        else
            name = "unfitted"
        end if
    end function lgbm_boosting_type

    real(dp) function lgbm_top_rate(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%top_rate_value
    end function lgbm_top_rate

    real(dp) function lgbm_other_rate(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%other_rate_value
    end function lgbm_other_rate

    subroutine lgbm_objective_derivatives(code, margin, target, weights, gradient, hessian, status)
        integer, intent(in) :: code
        real(dp), intent(in) :: margin(:), target(:), weights(:)
        real(dp), intent(out) :: gradient(:), hessian(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: p
        integer :: i

        if (size(margin) /= size(target) .or. size(weights) /= size(target) .or. &
            size(gradient) /= size(target) .or. size(hessian) /= size(target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "lightgbm derivatives: shape is invalid")
            return
        end if
        if (code == LIGHTGBM_OBJECTIVE_REGRESSION) then
            gradient = weights*(margin-target)
            hessian = weights
        else if (code == LIGHTGBM_OBJECTIVE_BINARY) then
            do i = 1, size(target)
                p = stable_sigmoid(margin(i))
                gradient(i) = weights(i)*(p-target(i))
                hessian(i) = weights(i)*max(p*(1.0_dp-p), 1.0e-12_dp)
            end do
        else
            call status_set(status, FORTNUM_DOMAIN_ERROR, "lightgbm derivatives: objective is unsupported")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_objective_derivatives

    subroutine lgbm_objective_loss(code, margin, target, weights, loss, status)
        integer, intent(in) :: code
        real(dp), intent(in) :: margin(:), target(:), weights(:)
        real(dp), intent(out) :: loss
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: total_weight, term
        integer :: i

        loss = 0.0_dp
        if (size(margin) /= size(target) .or. size(weights) /= size(target) .or. &
            size(target) < 1 .or. any(.not. ieee_is_finite(margin)) .or. &
            any(.not. ieee_is_finite(target)) .or. any(.not. ieee_is_finite(weights)) .or. &
            any(weights <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm validation objective: shape or finite-value contract failed")
            return
        end if
        total_weight = sum(weights)
        if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm validation objective: weights have no positive mass")
            return
        end if
        if (code == LIGHTGBM_OBJECTIVE_REGRESSION) then
            loss = 0.5_dp*sum(weights*(margin-target)**2)/total_weight
        else if (code == LIGHTGBM_OBJECTIVE_BINARY) then
            do i = 1, size(target)
                if (margin(i) >= 0.0_dp) then
                    term = (1.0_dp-target(i))*margin(i) + &
                        log(1.0_dp+exp(-margin(i)))
                else
                    term = -target(i)*margin(i) + log(1.0_dp+exp(margin(i)))
                end if
                loss = loss + weights(i)*term
            end do
            loss = loss/total_weight
        else
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm validation objective: objective is unsupported")
            return
        end if
        if (.not. ieee_is_finite(loss)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm validation objective: loss is nonfinite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_objective_loss

    subroutine build_leafwise_tree(x, gradient, hessian, observation_weight, options, rows, tree, status)
        real(dp), intent(in) :: x(:, :), gradient(:), hessian(:), observation_weight(:)
        type(lightgbm_options_t), intent(in) :: options
        integer, intent(in) :: rows(:)
        type(lgbm_tree_t), intent(out) :: tree
        type(fortnum_status_t), intent(out) :: status
        type(lgbm_split_t) :: candidate, best
        integer :: max_nodes, next_node, leaves, i, parent, left_node, right_node
        real(dp) :: total_g, total_h

        max_nodes = 2*options%num_leaves - 1
        allocate(tree%node(max_nodes))
        tree%n_nodes = 1
        tree%depth = 0
        allocate(tree%node(1)%rows(size(rows)))
        tree%node(1)%rows = rows
        total_g = sum(gradient(rows))
        total_h = sum(hessian(rows))
        tree%node(1)%weight = -total_g/(total_h + options%l2)
        next_node = 1
        leaves = 1
        do while (leaves < options%num_leaves)
            best%valid = .false.
            do i = 1, next_node
                if (.not. tree%node(i)%leaf .or. .not. allocated(tree%node(i)%rows)) cycle
                call best_split_for_leaf(x, gradient, hessian, observation_weight, options, &
                    tree%node(i)%rows, i, tree%node(i)%depth, candidate, status)
                if (status%code /= FORTNUM_OK) return
                if (candidate%valid) then
                    if (.not. best%valid .or. better_split(candidate, best)) then
                        call clear_split(best)
                        best = candidate
                        call clear_split(candidate)
                    else
                        call clear_split(candidate)
                    end if
                else
                    call clear_split(candidate)
                end if
            end do
            if (.not. best%valid) exit
            parent = best%node
            next_node = next_node + 1
            left_node = next_node
            next_node = next_node + 1
            right_node = next_node
            tree%n_nodes = next_node
            tree%node(parent)%leaf = .false.
            tree%node(parent)%feature = best%feature
            tree%node(parent)%threshold = best%threshold
            tree%node(parent)%gain = best%gain
            tree%node(parent)%left_child = left_node
            tree%node(parent)%right_child = right_node
            deallocate(tree%node(parent)%rows)
            tree%node(left_node)%depth = tree%node(parent)%depth + 1
            tree%node(right_node)%depth = tree%node(parent)%depth + 1
            tree%depth = max(tree%depth, tree%node(left_node)%depth)
            tree%node(left_node)%weight = best%left_weight
            tree%node(right_node)%weight = best%right_weight
            allocate(tree%node(left_node)%rows(size(best%left_rows)), &
                tree%node(right_node)%rows(size(best%right_rows)))
            tree%node(left_node)%rows = best%left_rows
            tree%node(right_node)%rows = best%right_rows
            call clear_split(best)
            leaves = leaves + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_leafwise_tree

    subroutine best_split_for_leaf(x, gradient, hessian, observation_weight, options, rows, node_id, depth, best, status)
        real(dp), intent(in) :: x(:, :), gradient(:), hessian(:), observation_weight(:)
        type(lightgbm_options_t), intent(in) :: options
        integer, intent(in) :: rows(:), node_id, depth
        type(lgbm_split_t), intent(out) :: best
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:), finite_values(:)
        integer, allocatable :: order(:), finite_rows(:), positions(:)
        integer :: feature, i, n, nfinite, npositions, p, k, left_count
        real(dp) :: total_g, total_h, left_g, left_h, right_g, right_h, gain, threshold

        best%valid = .false.
        if (size(rows) < 2*options%min_data_in_leaf .or. &
            (options%max_depth > 0 .and. depth >= options%max_depth)) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        n = size(rows)
        total_g = sum(gradient(rows))
        total_h = sum(hessian(rows))
        allocate(values(n), finite_values(n), order(n), finite_rows(n), positions(max(1,n-1)))
        do feature = 1, size(x, 2)
            nfinite = 0
            do i = 1, n
                if (.not. ieee_is_finite(x(rows(i), feature))) cycle
                nfinite = nfinite + 1
                finite_values(nfinite) = x(rows(i), feature)
                finite_rows(nfinite) = rows(i)
            end do
            if (nfinite < 2*options%min_data_in_leaf) cycle
            call sort_indices(finite_values(:nfinite), order(:nfinite))
            call xgb_histogram_cut_positions(finite_values(:nfinite), order(:nfinite), &
                finite_rows(:nfinite), observation_weight, options%max_bin, positions, npositions)
            left_g = 0.0_dp
            left_h = 0.0_dp
            p = 1
            do k = 1, nfinite-1
                i = order(k)
                left_g = left_g + gradient(finite_rows(i))
                left_h = left_h + hessian(finite_rows(i))
                if (p > npositions) cycle
                if (positions(p) /= k) cycle
                p = p + 1
                if (finite_values(order(k)) >= finite_values(order(k+1))) cycle
                left_count = k
                if (left_count < options%min_data_in_leaf .or. &
                    nfinite-left_count < options%min_data_in_leaf) cycle
                right_g = total_g-left_g
                right_h = total_h-left_h
                if (left_h <= 0.0_dp .or. right_h <= 0.0_dp) cycle
                gain = 0.5_dp*(left_g*left_g/(left_h+options%l2) + &
                    right_g*right_g/(right_h+options%l2) - &
                    total_g*total_g/(total_h+options%l2))
                threshold = 0.5_dp*finite_values(order(k)) + &
                    0.5_dp*finite_values(order(k+1))
                if (gain <= options%min_gain_to_split) cycle
                if (.not. best%valid .or. gain > best%gain + 1.0e-14_dp .or. &
                    (abs(gain-best%gain) <= 1.0e-14_dp .and. &
                    (feature < best%feature .or. (feature == best%feature .and. threshold < best%threshold)))) then
                    call clear_split(best)
                    best%valid = .true.
                    best%feature = feature
                    best%node = node_id
                    best%threshold = threshold
                    best%gain = gain
                    best%left_weight = -left_g/(left_h+options%l2)
                    best%right_weight = -right_g/(right_h+options%l2)
                    allocate(best%left_rows(left_count), best%right_rows(nfinite-left_count))
                    best%left_rows = finite_rows(order(:left_count))
                    best%right_rows = finite_rows(order(left_count+1:nfinite))
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine best_split_for_leaf

    logical function better_split(a, b) result(value)
        type(lgbm_split_t), intent(in) :: a, b
        value = a%gain > b%gain + 1.0e-14_dp .or. &
            (abs(a%gain-b%gain) <= 1.0e-14_dp .and. &
            (a%node < b%node .or. (a%node == b%node .and. &
            (a%feature < b%feature .or. (a%feature == b%feature .and. a%threshold < b%threshold)))))
    end function better_split

    subroutine clear_split(split)
        type(lgbm_split_t), intent(inout) :: split
        if (allocated(split%left_rows)) deallocate(split%left_rows)
        if (allocated(split%right_rows)) deallocate(split%right_rows)
        split%valid = .false.
    end subroutine clear_split

    subroutine lgbm_tree_predict(tree, x, values, status)
        type(lgbm_tree_t), intent(in) :: tree
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, node

        if (tree%n_nodes < 1 .or. size(values) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "lightgbm tree prediction: shape or finite-input contract failed")
            return
        end if
        do i = 1, size(x, 1)
            node = 1
            do while (.not. tree%node(node)%leaf)
                if (x(i, tree%node(node)%feature) < tree%node(node)%threshold) then
                    node = tree%node(node)%left_child
                else
                    node = tree%node(node)%right_child
                end if
            end do
            values(i) = tree%node(node)%weight
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_tree_predict

    logical function leaf_boundary_hit(self, x) result(hit)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer :: t, n, f
        real(dp) :: scale

        hit = .false.
        do t = 1, self%n_estimators
            do n = 1, self%estimator(t)%n_nodes
                if (self%estimator(t)%node(n)%leaf) cycle
                f = self%estimator(t)%node(n)%feature
                scale = max(1.0_dp, abs(self%estimator(t)%node(n)%threshold))
                if (any(abs(x(:, f)-self%estimator(t)%node(n)%threshold) <= 1.0e-12_dp*scale)) then
                    hit = .true.
                    return
                end if
            end do
        end do
    end function leaf_boundary_hit

    logical function valid_lgbm_scalars(model) result(valid)
        class(lightgbm_t), intent(in) :: model

        valid = model%n_inputs >= 1
        if (.not. valid) return
        valid = model%n_estimators >= 1 .and. model%requested_estimators >= model%n_estimators
        if (.not. valid) return
        valid = model%objective_code == LIGHTGBM_OBJECTIVE_REGRESSION .or. &
            model%objective_code == LIGHTGBM_OBJECTIVE_BINARY
        if (.not. valid) return
        valid = model%num_leaves_value >= 2 .and. model%max_bin_value >= 2 .and. &
            model%max_depth_value >= 0 .and. model%min_data_in_leaf_value >= 1
        if (.not. valid) return
        valid = ieee_is_finite(model%learning_rate) .and. model%learning_rate > 0.0_dp .and. &
            model%learning_rate <= 1.0_dp .and. ieee_is_finite(model%l2_value) .and. &
            model%l2_value >= 0.0_dp .and. ieee_is_finite(model%min_gain_value) .and. &
            model%min_gain_value >= 0.0_dp .and. ieee_is_finite(model%base_score)
        if (.not. valid) return
        valid = model%boosting_type_code == LIGHTGBM_BOOSTING_GBDT .or. &
            model%boosting_type_code == LIGHTGBM_BOOSTING_GOSS
        if (.not. valid) return
        valid = model%seed_value > 0_int64 .and. ieee_is_finite(model%top_rate_value) .and. &
            ieee_is_finite(model%other_rate_value)
        if (model%boosting_type_code == LIGHTGBM_BOOSTING_GOSS) then
            valid = valid .and. model%top_rate_value > 0.0_dp .and. model%top_rate_value < 1.0_dp .and. &
                model%other_rate_value > 0.0_dp .and. model%other_rate_value <= 1.0_dp .and. &
                model%top_rate_value + model%other_rate_value < 1.0_dp
        end if
        if (.not. valid) return
        valid = model%early_stopping_rounds_value >= 0 .and. &
            ieee_is_finite(model%early_stopping_min_delta_value) .and. &
            model%early_stopping_min_delta_value >= 0.0_dp .and. &
            model%best_iteration_value >= 0 .and. &
            model%best_iteration_value <= model%requested_estimators .and. &
            ieee_is_finite(model%best_validation_loss_value)
    end function valid_lgbm_scalars

    logical function valid_lgbm_tree(tree, n_inputs) result(valid)
        type(lgbm_tree_t), intent(in) :: tree
        integer, intent(in) :: n_inputs
        integer :: i

        valid = tree%n_nodes >= 1 .and. tree%n_nodes <= LIGHTGBM_MAX_SERIALIZED_NODES .and. &
            tree%depth >= 0 .and. allocated(tree%node)
        if (.not. valid) return
        if (size(tree%node) < tree%n_nodes) then
            valid = .false.
            return
        end if
        do i = 1, tree%n_nodes
            if (.not. ieee_is_finite(tree%node(i)%threshold) .or. &
                .not. ieee_is_finite(tree%node(i)%weight) .or. &
                .not. ieee_is_finite(tree%node(i)%gain) .or. &
                tree%node(i)%depth < 0) then
                valid = .false.
                return
            end if
            if (tree%node(i)%leaf) then
                if (tree%node(i)%feature /= 0 .or. tree%node(i)%left_child /= 0 .or. &
                    tree%node(i)%right_child /= 0) then
                    valid = .false.
                    return
                end if
            else
                if (tree%node(i)%feature < 1 .or. tree%node(i)%feature > n_inputs .or. &
                    tree%node(i)%left_child <= i .or. tree%node(i)%left_child > tree%n_nodes .or. &
                    tree%node(i)%right_child <= i .or. tree%node(i)%right_child > tree%n_nodes) then
                    valid = .false.
                    return
                end if
            end if
        end do
        valid = .not. tree%node(1)%leaf .or. tree%n_nodes == 1
    end function valid_lgbm_tree

    subroutine lgbm_write_i(unit, key, value, ios)
        integer, intent(in) :: unit, value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), value
    end subroutine lgbm_write_i

    subroutine lgbm_write_i64(unit, key, value, ios)
        integer, intent(in) :: unit
        integer(int64), intent(in) :: value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), value
    end subroutine lgbm_write_i64

    subroutine lgbm_write_l(unit, key, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: key
        logical, intent(in) :: value
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), merge(1, 0, value)
    end subroutine lgbm_write_l

    subroutine lgbm_write_r(unit, key, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: key
        real(dp), intent(in) :: value
        integer, intent(out) :: ios

        write(unit, "(A,1X,ES26.17E3)", iostat=ios) trim(key), value
    end subroutine lgbm_write_r

    subroutine lgbm_read_i(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        integer, intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine lgbm_read_i

    subroutine lgbm_read_i64(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        integer(int64), intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine lgbm_read_i64

    subroutine lgbm_read_l(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        logical, intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key
        integer :: encoded

        encoded = 0
        read(unit, *, iostat=ios) key, encoded
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
        if (ios == 0 .and. encoded /= 0 .and. encoded /= 1) ios = 1
        value = encoded == 1
    end subroutine lgbm_read_l

    subroutine lgbm_read_r(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        real(dp), intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine lgbm_read_r

    integer function parse_lgbm_objective(name) result(code)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: normalized
        normalized = trim(adjustl(name))
        select case (normalized)
        case ("regression", "reg:squarederror", "squared")
            code = LIGHTGBM_OBJECTIVE_REGRESSION
        case ("binary", "binary_logloss", "binary:logistic", "logistic")
            code = LIGHTGBM_OBJECTIVE_BINARY
        case default
            code = 0
        end select
    end function parse_lgbm_objective

    integer function parse_lgbm_boosting_type(name) result(code)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: normalized

        normalized = trim(adjustl(name))
        select case (normalized)
        case ("gbdt"); code = LIGHTGBM_BOOSTING_GBDT
        case ("goss"); code = LIGHTGBM_BOOSTING_GOSS
        case default; code = -1
        end select
    end function parse_lgbm_boosting_type

    !> Select one GOSS row set for a boosting round.  The largest absolute
    !> gradients are retained exactly; the remaining rows are ranked by a
    !> stable integer hash of `(seed, round, row)` so repeated fits and
    !> compiler-independent text snapshots have deterministic membership.
    subroutine select_goss_rows(gradient, options, round, rows, row_scale, status)
        real(dp), intent(in) :: gradient(:)
        type(lightgbm_options_t), intent(in) :: options
        integer, intent(in) :: round
        integer, allocatable, intent(out) :: rows(:)
        real(dp), allocatable, intent(out) :: row_scale(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: magnitude(:), hash_value(:)
        integer, allocatable :: order(:), remainder(:), remainder_order(:)
        integer :: n, top_count, other_count, i, j, k, remainder_count, selected_count
        real(dp) :: scale

        n = size(gradient)
        if (n < 2 .or. options%top_rate <= 0.0_dp .or. options%top_rate >= 1.0_dp .or. &
            options%other_rate <= 0.0_dp .or. options%other_rate > 1.0_dp .or. &
            options%top_rate + options%other_rate >= 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm GOSS: rates or sample size are invalid")
            return
        end if
        top_count = max(1, min(n-1, ceiling(options%top_rate*real(n, dp))))
        remainder_count = n-top_count
        other_count = max(1, min(remainder_count, ceiling(options%other_rate*real(n, dp))))
        selected_count = top_count + other_count
        scale = (1.0_dp-options%top_rate)/options%other_rate
        if (.not. ieee_is_finite(scale) .or. scale <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "lightgbm GOSS: row reweight is invalid")
            return
        end if
        allocate(magnitude(n), order(n), remainder(remainder_count), hash_value(remainder_count), &
            remainder_order(remainder_count), rows(selected_count), row_scale(selected_count))
        magnitude = abs(gradient)
        call sort_indices(magnitude, order)
        k = 0
        do i = n, n-top_count+1, -1
            k = k+1
            rows(k) = order(i)
            row_scale(k) = 1.0_dp
        end do
        k = 0
        do i = 1, n
            if (.not. row_is_selected(i, rows(:top_count))) then
                k = k+1
                remainder(k) = i
                hash_value(k) = real(goss_hash(options%seed, round, i), dp)
            end if
        end do
        call sort_indices(hash_value, remainder_order)
        do j = 1, other_count
            rows(top_count+j) = remainder(remainder_order(j))
            row_scale(top_count+j) = scale
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine select_goss_rows

    logical function row_is_selected(row, selected) result(value)
        integer, intent(in) :: row, selected(:)
        value = any(selected == row)
    end function row_is_selected

    pure integer(int64) function goss_hash(seed, round, row) result(value)
        integer(int64), intent(in) :: seed
        integer, intent(in) :: round, row
        integer(int64), parameter :: modulus = 2147483629_int64

        value = modulo(seed + 104729_int64*int(row, int64) + &
            13007_int64*int(round, int64), modulus)
        value = modulo(value*48271_int64 + 17_int64, modulus)
    end function goss_hash

    subroutine sort_indices(values, order)
        real(dp), intent(in) :: values(:)
        integer, intent(out) :: order(:)
        integer :: i, j, key
        do i = 1, size(values)
            order(i) = i
        end do
        do i = 2, size(values)
            key = order(i)
            j = i-1
            do while (j >= 1)
                if (values(order(j)) <= values(key)) exit
                order(j+1) = order(j)
                j = j-1
            end do
            order(j+1) = key
        end do
    end subroutine sort_indices

    real(dp) function stable_logit(probability) result(value)
        real(dp), intent(in) :: probability
        real(dp) :: p
        p = min(max(probability, 1.0e-12_dp), 1.0_dp-1.0e-12_dp)
        value = log(p)-log(1.0_dp-p)
    end function stable_logit

    pure real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp+exp(-value))
        else
            probability = exp(value)/(1.0_dp+exp(value))
        end if
    end function stable_sigmoid

    pure function stable_sigmoid_array(values) result(probabilities)
        real(dp), intent(in) :: values(:)
        real(dp) :: probabilities(size(values))
        integer :: i
        do i = 1, size(values)
            probabilities(i) = stable_sigmoid(values(i))
        end do
    end function stable_sigmoid_array

end module fortml_lightgbm
