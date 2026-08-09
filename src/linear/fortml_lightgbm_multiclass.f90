!> One-vs-rest multiclass classification over the bounded LightGBM-style
!> leaf-wise binary estimator.
module fortml_lightgbm_multiclass
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    implicit none
    private

    !> Deterministic one-vs-rest LightGBM-style multiclass classifier.
    !>
    !> The child estimators share all tree options and are fitted without
    !> child-level early stopping.  Validation stopping is applied once to
    !> the normalized multiclass probabilities, so every class retains an
    !> identical tree prefix.  Labels are sorted before the child fits and
    !> are returned unchanged by `predict`.
    type, public :: lightgbm_multiclass_t
        private
        type(lightgbm_t), allocatable :: one_vs_rest(:)
        integer, allocatable :: class_label(:)
        integer :: n_inputs = 0
        integer :: requested_estimators = 0
        integer :: best_iteration_value = 0
        real(dp) :: best_validation_loss_value = huge(1.0_dp)
        logical :: early_stopped_flag = .false.
        logical :: initialized = .false.
    contains
        procedure, public :: fit => lgbm_multiclass_fit
        procedure, public :: predict_proba => lgbm_multiclass_predict_proba
        procedure, public :: predict_log_proba => lgbm_multiclass_predict_log_proba
        procedure, public :: predict_proba_device => &
            lgbm_multiclass_predict_proba_device
        procedure, public :: predict_log_proba_device => &
            lgbm_multiclass_predict_log_proba_device
        procedure, public :: predict_proba_staged => &
            lgbm_multiclass_predict_proba_staged
        procedure, public :: decision_function_staged => &
            lgbm_multiclass_decision_function_staged
        procedure, public :: predict_proba_jvp => lgbm_multiclass_predict_proba_jvp
        procedure, public :: predict_log_proba_jvp => &
            lgbm_multiclass_predict_log_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            lgbm_multiclass_predict_proba_parameter_jvp
        procedure, public :: predict_log_proba_parameter_jvp => &
            lgbm_multiclass_predict_log_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => lgbm_multiclass_predict_proba_vjp
        procedure, public :: predict_log_proba_vjp => &
            lgbm_multiclass_predict_log_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            lgbm_multiclass_predict_proba_parameter_vjp
        procedure, public :: predict_log_proba_parameter_vjp => &
            lgbm_multiclass_predict_log_proba_parameter_vjp
        procedure, public :: predict_proba_parameter_jvp_device => &
            lgbm_multiclass_predict_proba_parameter_jvp_device
        procedure, public :: predict_proba_parameter_vjp_device => &
            lgbm_multiclass_predict_proba_parameter_vjp_device
        procedure, public :: predict_log_proba_parameter_jvp_device => &
            lgbm_multiclass_predict_log_proba_parameter_jvp_device
        procedure, public :: predict_log_proba_parameter_vjp_device => &
            lgbm_multiclass_predict_log_proba_parameter_vjp_device
        procedure, public :: decision_function => lgbm_multiclass_decision_function
        procedure, public :: predict => lgbm_multiclass_predict
        procedure, public :: predict_device => lgbm_multiclass_predict_device
        procedure, public :: classes => lgbm_multiclass_classes
        procedure, public :: feature_count => lgbm_multiclass_feature_count
        procedure, public :: class_count => lgbm_multiclass_class_count
        procedure, public :: parameter_count => lgbm_multiclass_parameter_count
        procedure, public :: parameters => lgbm_multiclass_parameters
        procedure, public :: leaf_parameter_count => lgbm_multiclass_parameter_count
        procedure, public :: leaf_parameters => lgbm_multiclass_parameters
        procedure, public :: estimator_count => lgbm_multiclass_estimator_count
        procedure, public :: requested_estimator_count => &
            lgbm_multiclass_requested_estimator_count
        procedure, public :: best_iteration => lgbm_multiclass_best_iteration
        procedure, public :: best_validation_loss => &
            lgbm_multiclass_best_validation_loss
        procedure, public :: early_stopped => lgbm_multiclass_early_stopped
        procedure, public :: boosting_type => lgbm_multiclass_boosting_type
        procedure, public :: num_leaves => lgbm_multiclass_num_leaves
        procedure, public :: fitted => lgbm_multiclass_fitted
        procedure, public :: device_supported => lgbm_multiclass_device_supported
    end type lightgbm_multiclass_t

contains

    subroutine lgbm_multiclass_fit(self, x, labels, status, options, sample_weight, &
            validation_x, validation_labels, validation_weight)
        class(lightgbm_multiclass_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :)
        integer, intent(in), optional :: validation_labels(:)
        real(dp), intent(in), optional :: validation_weight(:)
        type(lightgbm_options_t) :: settings, child_settings
        type(lightgbm_multiclass_t) :: candidate
        type(lightgbm_t) :: sliced
        integer, allocatable :: classes(:), binary_labels(:), validation_binary(:)
        real(dp), allocatable :: staged(:, :, :), child_staged(:, :), totals(:, :)
        real(dp), allocatable :: validation_observation_weight(:)
        integer :: i, j, n_classes, n_samples, n_validation
        integer :: completed_estimators, best_iteration, stale_rounds, keep_estimators
        real(dp) :: validation_loss, best_validation_loss, weight_sum
        logical :: have_validation, improved
        type(lightgbm_options_t) :: lightgbm_options_t_default

        settings = lightgbm_options_t_default
        if (present(options)) settings = options
        have_validation = present(validation_x) .or. present(validation_labels) .or. &
            present(validation_weight)
        if (present(validation_x) .neqv. present(validation_labels)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass fit: validation_x and validation_labels must be supplied together")
            return
        end if
        if (present(validation_weight) .and. .not. present(validation_x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass fit: validation_weight requires validation data")
            return
        end if
        if (settings%early_stopping_rounds < 0 .or. &
            .not. ieee_is_finite(settings%early_stopping_min_delta) .or. &
            settings%early_stopping_min_delta < 0.0_dp .or. &
            (settings%early_stopping_rounds > 0 .and. .not. have_validation)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass fit: invalid early-stopping configuration")
            return
        end if
        n_samples = size(x, 1)
        if (n_samples < 2 .or. size(x, 2) < 1 .or. size(labels) /= n_samples .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass fit: input dimensions or finite-value contract failed")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "LightGBM multiclass fit: sample_weight must be positive and finite")
                return
            end if
        end if
        call unique_sorted_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass fit: at least two classes are required")
            return
        end if
        if (have_validation) then
            n_validation = size(validation_x, 1)
            if (n_validation < 1 .or. size(validation_x, 2) /= size(x, 2) .or. &
                size(validation_labels) /= n_validation .or. &
                any(.not. ieee_is_finite(validation_x))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "LightGBM multiclass fit: validation dimensions or finite-value contract failed")
                return
            end if
            if (present(validation_weight)) then
                if (size(validation_weight) /= n_validation .or. &
                    any(.not. ieee_is_finite(validation_weight)) .or. &
                    any(validation_weight <= 0.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "LightGBM multiclass fit: validation_weight must be positive and finite")
                    return
                end if
            end if
            do i = 1, n_validation
                if (find_label_index(classes, validation_labels(i)) < 1) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "LightGBM multiclass fit: validation labels must belong to training classes")
                    return
                end if
            end do
        else
            n_validation = 0
        end if

        child_settings = settings
        child_settings%objective = "binary"
        child_settings%early_stopping_rounds = 0
        child_settings%restore_best = .false.
        allocate(candidate%one_vs_rest(n_classes), candidate%class_label(n_classes))
        candidate%class_label = classes
        allocate(binary_labels(n_samples))
        if (have_validation) allocate(validation_binary(n_validation))
        do i = 1, n_classes
            binary_labels = 0
            where (labels == classes(i)) binary_labels = 1
            if (have_validation) then
                validation_binary = 0
                where (validation_labels == classes(i)) validation_binary = 1
            end if
            if (present(sample_weight)) then
                if (have_validation) then
                    if (present(validation_weight)) then
                        call candidate%one_vs_rest(i)%fit_binary(x, &
                            real(binary_labels, dp), status, child_settings, sample_weight, &
                            validation_x, real(validation_binary, dp), validation_weight)
                    else
                        call candidate%one_vs_rest(i)%fit_binary(x, &
                            real(binary_labels, dp), status, child_settings, sample_weight, &
                            validation_x, real(validation_binary, dp))
                    end if
                else
                    call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                        status, child_settings, sample_weight)
                end if
            else if (have_validation) then
                if (present(validation_weight)) then
                    call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                        status, child_settings, validation_x=validation_x, &
                        validation_y=real(validation_binary, dp), &
                        validation_weight=validation_weight)
                else
                    call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                        status, child_settings, validation_x=validation_x, &
                        validation_y=real(validation_binary, dp))
                end if
            else
                call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                    status, child_settings)
            end if
            if (status%code /= FORTNUM_OK) return
        end do
        candidate%n_inputs = size(x, 2)
        candidate%requested_estimators = settings%n_estimators
        candidate%best_iteration_value = settings%n_estimators
        candidate%best_validation_loss_value = huge(1.0_dp)
        candidate%early_stopped_flag = .false.
        completed_estimators = candidate%one_vs_rest(1)%estimator_count()
        do i = 2, n_classes
            if (candidate%one_vs_rest(i)%estimator_count() /= completed_estimators) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "LightGBM multiclass fit: child estimator counts differ")
                return
            end if
        end do

        if (have_validation) then
            allocate(staged(n_validation, n_classes, completed_estimators), &
                child_staged(n_validation, completed_estimators), &
                totals(n_validation, completed_estimators), &
                validation_observation_weight(n_validation))
            validation_observation_weight = 1.0_dp
            if (present(validation_weight)) validation_observation_weight = validation_weight
            weight_sum = sum(validation_observation_weight)
            if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "LightGBM multiclass fit: validation_weight has no positive mass")
                return
            end if
            do i = 1, n_classes
                call candidate%one_vs_rest(i)%predict_staged(validation_x, child_staged, status)
                if (status%code /= FORTNUM_OK) return
                staged(:, i, :) = child_staged
            end do
            totals = sum(staged, dim=2)
            if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "LightGBM multiclass fit: validation normalization failed")
                return
            end if
            best_validation_loss = huge(1.0_dp)
            best_iteration = 0
            stale_rounds = 0
            do j = 1, completed_estimators
                validation_loss = multiclass_log_loss(staged(:, :, j), totals(:, j), &
                    validation_labels, classes, validation_observation_weight, weight_sum)
                if (.not. ieee_is_finite(validation_loss)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "LightGBM multiclass fit: validation objective is nonfinite")
                    return
                end if
                improved = validation_loss < best_validation_loss - &
                    settings%early_stopping_min_delta
                if (improved) then
                    best_validation_loss = validation_loss
                    best_iteration = j
                    stale_rounds = 0
                else
                    stale_rounds = stale_rounds + 1
                end if
                if (settings%early_stopping_rounds > 0 .and. &
                    stale_rounds >= settings%early_stopping_rounds) then
                    candidate%early_stopped_flag = .true.
                    exit
                end if
            end do
            if (best_iteration < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "LightGBM multiclass fit: validation objective did not produce a finite score")
                return
            end if
            candidate%best_iteration_value = best_iteration
            candidate%best_validation_loss_value = best_validation_loss
            keep_estimators = completed_estimators
            if (settings%restore_best) keep_estimators = best_iteration
            if (candidate%early_stopped_flag .and. .not. settings%restore_best) then
                keep_estimators = j
            end if
            if (keep_estimators < 1) keep_estimators = best_iteration
            if (keep_estimators < completed_estimators) then
                do i = 1, n_classes
                    call candidate%one_vs_rest(i)%slice(keep_estimators, sliced, status)
                    if (status%code /= FORTNUM_OK) return
                    candidate%one_vs_rest(i) = sliced
                end do
            end if
        end if
        candidate%initialized = .true.
        if (.not. valid_multiclass_model(candidate)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass fit: child model metadata is inconsistent")
            return
        end if
        self%one_vs_rest = candidate%one_vs_rest
        self%class_label = candidate%class_label
        self%n_inputs = candidate%n_inputs
        self%requested_estimators = candidate%requested_estimators
        self%best_iteration_value = candidate%best_iteration_value
        self%best_validation_loss_value = candidate%best_validation_loss_value
        self%early_stopped_flag = candidate%early_stopped_flag
        self%initialized = candidate%initialized
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_fit

    subroutine lgbm_multiclass_predict_proba(self, x, probabilities, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: child_probabilities(:, :), candidate(:, :), totals(:)
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict_proba: model or input is invalid")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= self%class_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict_proba: output shape is invalid")
            return
        end if
        allocate(child_probabilities(size(x, 1), 2), &
            candidate(size(x, 1), self%class_count()), totals(size(x, 1)))
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_proba(x, child_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            candidate(:, i) = child_probabilities(:, 2)
        end do
        totals = sum(candidate, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict_proba: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            candidate(:, i) = candidate(:, i)/totals
        end do
        probabilities = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_proba

    !> Return normalized OVR probabilities in log space.
    !>
    !> Each child margin is transformed with a stable log-sigmoid and the
    !> class reduction uses log-sum-exp.  This keeps the API finite even when
    !> a fitted leaf margin is far outside the representable probability
    !> range.
    subroutine lgbm_multiclass_predict_log_proba(self, x, log_probabilities, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margins(:, :), log_positive(:, :), log_totals(:)
        integer :: i, j

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict_log_proba: model or input is invalid")
            return
        end if
        if (any(shape(log_probabilities) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict_log_proba: output shape is invalid")
            return
        end if
        allocate(margins(size(x, 1), self%class_count()), &
            log_positive(size(x, 1), self%class_count()), log_totals(size(x, 1)))
        do j = 1, self%class_count()
            call self%one_vs_rest(j)%predict_margin(x, margins(:, j), status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, size(x, 1)
                log_positive(i, j) = stable_log_sigmoid(margins(i, j))
            end do
        end do
        do i = 1, size(x, 1)
            log_totals(i) = stable_logsumexp(log_positive(i, :))
            if (.not. ieee_is_finite(log_totals(i))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "LightGBM multiclass predict_log_proba: normalization failed")
                return
            end if
        end do
        do j = 1, self%class_count()
            log_probabilities(:, j) = log_positive(:, j) - log_totals
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_log_proba

    subroutine lgbm_multiclass_predict_proba_device(self, device, x, probabilities, &
            status)
        class(lightgbm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "LightGBM multiclass device prediction: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass device prediction: device kind is invalid")
        end select
    end subroutine lgbm_multiclass_predict_proba_device

    subroutine lgbm_multiclass_predict_log_proba_device(self, device, x, &
            log_probabilities, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_log_proba(x, log_probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "LightGBM multiclass log-probability device: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability device: device kind is invalid")
        end select
    end subroutine lgbm_multiclass_predict_log_proba_device

    subroutine lgbm_multiclass_predict_proba_staged(self, x, probabilities, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: child_staged(:, :), candidate(:, :, :), totals(:, :)
        integer :: i, stages

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict_proba_staged: model or input is invalid")
            return
        end if
        stages = self%estimator_count()
        if (size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= self%class_count() .or. &
            size(probabilities, 3) /= stages .or. stages < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict_proba_staged: output shape is invalid")
            return
        end if
        allocate(child_staged(size(x, 1), stages), &
            candidate(size(x, 1), self%class_count(), stages), &
            totals(size(x, 1), stages))
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_staged(x, child_staged, status)
            if (status%code /= FORTNUM_OK) return
            candidate(:, i, :) = child_staged
        end do
        totals = sum(candidate, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict_proba_staged: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            candidate(:, i, :) = candidate(:, i, :)/totals
        end do
        probabilities = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_proba_staged

    subroutine lgbm_multiclass_decision_function_staged(self, x, margins, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: child_margins(:, :), candidate(:, :, :)
        integer :: i, stages

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass decision_function_staged: model or input is invalid")
            return
        end if
        stages = self%estimator_count()
        if (size(margins, 1) /= size(x, 1) .or. size(margins, 2) /= self%class_count() .or. &
            size(margins, 3) /= stages .or. stages < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass decision_function_staged: output shape is invalid")
            return
        end if
        allocate(child_margins(size(x, 1), stages), &
            candidate(size(x, 1), self%class_count(), stages))
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_staged_margin(x, child_margins, status)
            if (status%code /= FORTNUM_OK) return
            candidate(:, i, :) = child_margins
        end do
        margins = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_decision_function_staged

    subroutine lgbm_multiclass_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: child_value(:), child_dot(:), raw(:, :), raw_dot(:, :)
        real(dp), allocatable :: totals(:), totals_dot(:), candidate(:, :), candidate_dot(:, :)
        integer :: i

        if (.not. valid_query(self, x) .or. size(x_dot, 1) /= size(x, 1) .or. &
            size(x_dot, 2) /= size(x, 2) .or. any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability JVP: model or input is invalid")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= self%class_count() .or. &
            size(probabilities_dot, 1) /= size(x, 1) .or. &
            size(probabilities_dot, 2) /= self%class_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability JVP: output shape is invalid")
            return
        end if
        allocate(child_value(size(x, 1)), child_dot(size(x, 1)), &
            raw(size(x, 1), self%class_count()), raw_dot(size(x, 1), self%class_count()), &
            totals(size(x, 1)), totals_dot(size(x, 1)), &
            candidate(size(x, 1), self%class_count()), &
            candidate_dot(size(x, 1), self%class_count()))
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_jvp(x, x_dot, child_value, child_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, i) = child_value
            raw_dot(:, i) = child_dot
        end do
        totals = sum(raw, dim=2)
        totals_dot = sum(raw_dot, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp) .or. &
            any(.not. ieee_is_finite(totals_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability JVP: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            candidate(:, i) = raw(:, i)/totals
            candidate_dot(:, i) = (raw_dot(:, i)*totals - raw(:, i)*totals_dot)/ &
                (totals*totals)
        end do
        probabilities = candidate
        probabilities_dot = candidate_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_proba_jvp

    subroutine lgbm_multiclass_predict_proba_vjp(self, x, probabilities_bar, x_bar, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: child_probabilities(:, :), raw(:, :), totals(:)
        real(dp), allocatable :: raw_bar(:, :), child_bar(:, :), candidate(:, :)
        integer :: i

        x_bar = 0.0_dp
        if (.not. valid_query(self, x) .or. size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%class_count() .or. &
            size(x_bar, 1) /= size(x, 1) .or. size(x_bar, 2) /= size(x, 2) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability VJP: model, cotangent, or output shape is invalid")
            return
        end if
        allocate(child_probabilities(size(x, 1), 2), raw(size(x, 1), self%class_count()), &
            totals(size(x, 1)), raw_bar(size(x, 1), self%class_count()), &
            child_bar(size(x, 1), size(x, 2)), candidate(size(x, 1), size(x, 2)))
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_proba(x, child_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, i) = child_probabilities(:, 2)
        end do
        totals = sum(raw, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability VJP: normalization failed")
            return
        end if
        raw_bar = 0.0_dp
        do i = 1, self%class_count()
            raw_bar(:, i) = probabilities_bar(:, i)/totals
        end do
        do i = 1, size(x, 1)
            raw_bar(i, :) = raw_bar(i, :) - &
                sum(probabilities_bar(i, :)*raw(i, :))/ (totals(i)*totals(i))
        end do
        candidate = 0.0_dp
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_vjp(x, raw_bar(:, i), child_bar, status)
            if (status%code /= FORTNUM_OK) return
            candidate = candidate + child_bar
        end do
        x_bar = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_proba_vjp

    !> Forward product of normalized OVR probabilities with respect to the
    !> packed `[base_score, leaf weights]` coordinates of all child trees.
    subroutine lgbm_multiclass_predict_proba_parameter_jvp(self, x, parameter_dot, &
            probabilities, probabilities_dot, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), parameter_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margins(:, :), margins_dot(:, :), positive(:, :)
        real(dp), allocatable :: positive_dot(:, :), totals(:), totals_dot(:)
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter JVP: model or input is invalid")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter JVP: output shape is invalid")
            return
        end if
        if (size(parameter_dot) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(parameter_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter JVP: parameter tangent is invalid")
            return
        end if
        allocate(margins(size(x, 1), self%class_count()), &
            margins_dot(size(x, 1), self%class_count()), &
            positive(size(x, 1), self%class_count()), &
            positive_dot(size(x, 1), self%class_count()), &
            totals(size(x, 1)), totals_dot(size(x, 1)))
        call multiclass_parameter_margin_jvp(self, x, parameter_dot, margins, &
            margins_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%class_count()
            positive(:, i) = stable_sigmoid_array(margins(:, i))
            positive_dot(:, i) = positive(:, i)*(1.0_dp - positive(:, i))* &
                margins_dot(:, i)
        end do
        totals = sum(positive, dim=2)
        totals_dot = sum(positive_dot, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp) .or. &
            any(.not. ieee_is_finite(totals_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter JVP: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            probabilities(:, i) = positive(:, i)/totals
            probabilities_dot(:, i) = (positive_dot(:, i)*totals - &
                positive(:, i)*totals_dot)/(totals*totals)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_proba_parameter_jvp

    !> Reverse product of normalized OVR probabilities with respect to the
    !> packed fixed-structure leaf coordinates of all child trees.
    subroutine lgbm_multiclass_predict_proba_parameter_vjp(self, x, &
            probabilities_bar, parameter_bar, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margins(:, :), positive(:, :), probabilities(:, :)
        real(dp), allocatable :: margin_bar(:), child_bar(:)
        real(dp) :: normalization, dot_product_bar
        integer :: i, j, first, last, count

        parameter_bar = 0.0_dp
        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter VJP: model or input is invalid")
            return
        end if
        if (any(shape(probabilities_bar) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter VJP: cotangent shape is invalid")
            return
        end if
        if (size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter VJP: parameter or cotangent is invalid")
            return
        end if
        allocate(margins(size(x, 1), self%class_count()), &
            positive(size(x, 1), self%class_count()), &
            probabilities(size(x, 1), self%class_count()), &
            margin_bar(size(x, 1)), child_bar(0))
        call multiclass_margin_probabilities(self, x, margins, positive, &
            probabilities, status)
        if (status%code /= FORTNUM_OK) return
        first = 1
        do i = 1, self%class_count()
            count = self%one_vs_rest(i)%leaf_parameter_count()
            last = first + count - 1
            deallocate(child_bar)
            allocate(child_bar(count))
            do j = 1, size(x, 1)
                normalization = sum(positive(j, :))
                dot_product_bar = sum(probabilities_bar(j, :)*probabilities(j, :))
                margin_bar(j) = positive(j, i)*(1.0_dp - positive(j, i))/ &
                    normalization*(probabilities_bar(j, i) - dot_product_bar)
            end do
            call self%one_vs_rest(i)%predict_leaf_vjp(x, margin_bar, child_bar, status)
            if (status%code /= FORTNUM_OK) return
            parameter_bar(first:last) = child_bar
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_proba_parameter_vjp

    !> Forward product of stable log probabilities with respect to inputs.
    subroutine lgbm_multiclass_predict_log_proba_jvp(self, x, x_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)

        if (any(shape(log_probabilities) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability JVP: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%class_count()), &
            probabilities_dot(size(x, 1), self%class_count()))
        call self%predict_proba_jvp(x, x_dot, probabilities, probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        log_probabilities_dot = probabilities_dot/max(probabilities, tiny(1.0_dp))
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_log_proba_jvp

    !> Forward product of stable log probabilities with respect to packed
    !> fixed-structure leaf coordinates.
    subroutine lgbm_multiclass_predict_log_proba_parameter_jvp(self, x, parameter_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), parameter_dot(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margins(:, :), margins_dot(:, :), positive(:, :)
        real(dp), allocatable :: log_positive(:, :), log_totals(:), log_dot(:, :)
        real(dp), allocatable :: positive_dot(:, :)
        integer :: i, j

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter JVP: model or input is invalid")
            return
        end if
        if (any(shape(log_probabilities) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter JVP: output shape is invalid")
            return
        end if
        if (size(parameter_dot) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(parameter_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter JVP: parameter tangent is invalid")
            return
        end if
        allocate(margins(size(x, 1), self%class_count()), &
            margins_dot(size(x, 1), self%class_count()), &
            positive(size(x, 1), self%class_count()), &
            positive_dot(size(x, 1), self%class_count()), &
            log_positive(size(x, 1), self%class_count()), &
            log_totals(size(x, 1)), log_dot(size(x, 1), self%class_count()))
        call multiclass_parameter_margin_jvp(self, x, parameter_dot, margins, &
            margins_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%class_count()
            positive(:, i) = stable_sigmoid_array(margins(:, i))
            positive_dot(:, i) = positive(:, i)*(1.0_dp - positive(:, i))* &
                margins_dot(:, i)
            do j = 1, size(x, 1)
                log_positive(j, i) = stable_log_sigmoid(margins(j, i))
            end do
        end do
        do j = 1, size(x, 1)
            log_totals(j) = stable_logsumexp(log_positive(j, :))
            if (.not. ieee_is_finite(log_totals(j))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "LightGBM multiclass log-probability parameter JVP: normalization failed")
                return
            end if
        end do
        do i = 1, self%class_count()
            log_probabilities(:, i) = log_positive(:, i) - log_totals
            do j = 1, size(x, 1)
                log_dot(j, i) = margins_dot(j, i)*(1.0_dp - positive(j, i)) - &
                    sum(exp(log_positive(j, :) - log_totals(j))* &
                    margins_dot(j, :)*(1.0_dp - positive(j, :)))
            end do
        end do
        log_probabilities_dot = log_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_log_proba_parameter_jvp

    !> Reverse product of normalized OVR probabilities with respect to inputs.
    subroutine lgbm_multiclass_predict_log_proba_vjp(self, x, log_probabilities_bar, &
            x_bar, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margins(:, :), positive(:, :), probabilities(:, :)
        real(dp), allocatable :: margin_bar(:), child_x_bar(:, :)
        real(dp) :: dot_product_bar
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability VJP: model or input is invalid")
            return
        end if
        if (any(shape(log_probabilities_bar) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability VJP: cotangent or output shape is invalid")
            return
        end if
        allocate(margins(size(x, 1), self%class_count()), &
            positive(size(x, 1), self%class_count()), &
            probabilities(size(x, 1), self%class_count()), &
            margin_bar(size(x, 1)), child_x_bar(size(x, 1), self%n_inputs))
        call multiclass_margin_probabilities(self, x, margins, positive, &
            probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%class_count()
            do j = 1, size(x, 1)
                dot_product_bar = sum(log_probabilities_bar(j, :))*probabilities(j, i)
                margin_bar(j) = (1.0_dp - positive(j, i))* &
                    (log_probabilities_bar(j, i) - dot_product_bar)
            end do
            call self%one_vs_rest(i)%predict_vjp(x, margin_bar, child_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + child_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_log_proba_vjp

    !> Reverse product of stable log probabilities with respect to packed
    !> fixed-structure leaf coordinates.
    subroutine lgbm_multiclass_predict_log_proba_parameter_vjp(self, x, &
            log_probabilities_bar, parameter_bar, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margins(:, :), positive(:, :), probabilities(:, :)
        real(dp), allocatable :: margin_bar(:), child_bar(:)
        real(dp) :: dot_product_bar
        integer :: i, j, first, last, count

        parameter_bar = 0.0_dp
        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter VJP: model or input is invalid")
            return
        end if
        if (any(shape(log_probabilities_bar) /= [size(x, 1), self%class_count()]) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter VJP: cotangent shape or values are invalid")
            return
        end if
        if (size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter VJP: parameter shape is invalid")
            return
        end if
        allocate(margins(size(x, 1), self%class_count()), &
            positive(size(x, 1), self%class_count()), &
            probabilities(size(x, 1), self%class_count()), &
            margin_bar(size(x, 1)), child_bar(0))
        call multiclass_margin_probabilities(self, x, margins, positive, &
            probabilities, status)
        if (status%code /= FORTNUM_OK) return
        first = 1
        do i = 1, self%class_count()
            count = self%one_vs_rest(i)%leaf_parameter_count()
            last = first + count - 1
            deallocate(child_bar)
            allocate(child_bar(count))
            do j = 1, size(x, 1)
                dot_product_bar = sum(log_probabilities_bar(j, :))*probabilities(j, i)
                margin_bar(j) = (1.0_dp - positive(j, i))* &
                    (log_probabilities_bar(j, i) - dot_product_bar)
            end do
            call self%one_vs_rest(i)%predict_leaf_vjp(x, margin_bar, child_bar, status)
            if (status%code /= FORTNUM_OK) return
            parameter_bar(first:last) = child_bar
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict_log_proba_parameter_vjp

    subroutine lgbm_multiclass_predict_proba_parameter_jvp_device(self, device, x, &
            parameter_dot, probabilities, probabilities_dot, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), parameter_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_jvp(x, parameter_dot, probabilities, &
                probabilities_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "LightGBM multiclass probability parameter JVP device: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter JVP device: device kind is invalid")
        end select
    end subroutine lgbm_multiclass_predict_proba_parameter_jvp_device

    subroutine lgbm_multiclass_predict_proba_parameter_vjp_device(self, device, x, &
            probabilities_bar, parameter_bar, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_vjp(x, probabilities_bar, parameter_bar, &
                status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "LightGBM multiclass probability parameter VJP device: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass probability parameter VJP device: device kind is invalid")
        end select
    end subroutine lgbm_multiclass_predict_proba_parameter_vjp_device

    subroutine lgbm_multiclass_predict_log_proba_parameter_jvp_device(self, device, x, &
            parameter_dot, log_probabilities, log_probabilities_dot, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), parameter_dot(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_log_proba_parameter_jvp(x, parameter_dot, &
                log_probabilities, log_probabilities_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "LightGBM multiclass log-probability parameter JVP device: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter JVP device: device kind is invalid")
        end select
    end subroutine lgbm_multiclass_predict_log_proba_parameter_jvp_device

    subroutine lgbm_multiclass_predict_log_proba_parameter_vjp_device(self, device, x, &
            log_probabilities_bar, parameter_bar, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_log_proba_parameter_vjp(x, log_probabilities_bar, &
                parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "LightGBM multiclass log-probability parameter VJP device: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass log-probability parameter VJP device: device kind is invalid")
        end select
    end subroutine lgbm_multiclass_predict_log_proba_parameter_vjp_device

    subroutine lgbm_multiclass_decision_function(self, x, margins, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: child(:), candidate(:, :)
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass decision_function: model or input is invalid")
            return
        end if
        if (size(margins, 1) /= size(x, 1) .or. size(margins, 2) /= self%class_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass decision_function: output shape is invalid")
            return
        end if
        allocate(child(size(x, 1)), candidate(size(x, 1), self%class_count()))
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_margin(x, child, status)
            if (status%code /= FORTNUM_OK) return
            candidate(:, i) = child
        end do
        margins = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_decision_function

    subroutine lgbm_multiclass_predict(self, x, labels, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict: output shape is invalid")
            return
        end if
        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass predict: model or input is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%class_count()))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multiclass_predict

    subroutine lgbm_multiclass_predict_device(self, device, x, labels, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "LightGBM multiclass device prediction: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass device prediction: device kind is invalid")
        end select
    end subroutine lgbm_multiclass_predict_device

    function lgbm_multiclass_classes(self) result(classes)
        class(lightgbm_multiclass_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function lgbm_multiclass_classes

    integer function lgbm_multiclass_feature_count(self) result(count)
        class(lightgbm_multiclass_t), intent(in) :: self
        count = self%n_inputs
    end function lgbm_multiclass_feature_count

    integer function lgbm_multiclass_class_count(self) result(count)
        class(lightgbm_multiclass_t), intent(in) :: self
        count = 0
        if (allocated(self%class_label)) count = size(self%class_label)
    end function lgbm_multiclass_class_count

    integer function lgbm_multiclass_parameter_count(self) result(count)
        class(lightgbm_multiclass_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. self%initialized .or. .not. allocated(self%one_vs_rest)) return
        do i = 1, size(self%one_vs_rest)
            count = count + self%one_vs_rest(i)%leaf_parameter_count()
        end do
    end function lgbm_multiclass_parameter_count

    function lgbm_multiclass_parameters(self, status) result(parameters)
        class(lightgbm_multiclass_t), intent(in) :: self
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: parameters(:), child_parameters(:)
        integer :: i, first, last, count

        allocate(parameters(max(0, self%parameter_count())))
        parameters = 0.0_dp
        if (.not. self%initialized .or. .not. allocated(self%one_vs_rest)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass parameters: model is not initialized")
            return
        end if
        first = 1
        do i = 1, size(self%one_vs_rest)
            child_parameters = self%one_vs_rest(i)%leaf_parameters(status)
            if (status%code /= FORTNUM_OK) return
            count = size(child_parameters)
            last = first + count - 1
            parameters(first:last) = child_parameters
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end function lgbm_multiclass_parameters

    integer function lgbm_multiclass_estimator_count(self) result(count)
        class(lightgbm_multiclass_t), intent(in) :: self
        count = 0
        if (.not. allocated(self%one_vs_rest)) return
        if (size(self%one_vs_rest) < 1) return
        count = self%one_vs_rest(1)%estimator_count()
    end function lgbm_multiclass_estimator_count

    integer function lgbm_multiclass_requested_estimator_count(self) result(count)
        class(lightgbm_multiclass_t), intent(in) :: self
        count = self%requested_estimators
    end function lgbm_multiclass_requested_estimator_count

    integer function lgbm_multiclass_best_iteration(self) result(iteration)
        class(lightgbm_multiclass_t), intent(in) :: self
        iteration = self%best_iteration_value
    end function lgbm_multiclass_best_iteration

    real(dp) function lgbm_multiclass_best_validation_loss(self) result(loss)
        class(lightgbm_multiclass_t), intent(in) :: self
        loss = self%best_validation_loss_value
    end function lgbm_multiclass_best_validation_loss

    logical function lgbm_multiclass_early_stopped(self) result(stopped)
        class(lightgbm_multiclass_t), intent(in) :: self
        stopped = self%early_stopped_flag
    end function lgbm_multiclass_early_stopped

    character(len=16) function lgbm_multiclass_boosting_type(self) result(name)
        class(lightgbm_multiclass_t), intent(in) :: self
        name = "unfitted"
        if (.not. self%initialized) return
        name = self%one_vs_rest(1)%boosting_type()
    end function lgbm_multiclass_boosting_type

    integer function lgbm_multiclass_num_leaves(self) result(value)
        class(lightgbm_multiclass_t), intent(in) :: self
        value = 0
        if (.not. self%initialized) return
        value = self%one_vs_rest(1)%num_leaves()
    end function lgbm_multiclass_num_leaves

    logical function lgbm_multiclass_device_supported(self, device_kind) result(supported)
        class(lightgbm_multiclass_t), intent(in) :: self
        integer, intent(in) :: device_kind
        supported = self%initialized .and. device_kind == FORTML_DEVICE_CPU
    end function lgbm_multiclass_device_supported

    logical function lgbm_multiclass_fitted(self) result(fitted)
        class(lightgbm_multiclass_t), intent(in) :: self
        fitted = self%initialized
    end function lgbm_multiclass_fitted

    subroutine multiclass_margin_probabilities(self, x, margins, positive, &
            probabilities, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :), positive(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: totals(:)
        integer :: i, j

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass margin products: model or input is invalid")
            return
        end if
        if (any(shape(margins) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(positive) /= shape(margins)) .or. &
            any(shape(probabilities) /= shape(margins))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass margin products: output shape is invalid")
            return
        end if
        allocate(totals(size(x, 1)))
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_margin(x, margins(:, i), status)
            if (status%code /= FORTNUM_OK) return
            positive(:, i) = stable_sigmoid_array(margins(:, i))
        end do
        totals = sum(positive, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass margin products: normalization failed")
            return
        end if
        do j = 1, self%class_count()
            probabilities(:, j) = positive(:, j)/totals
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_margin_probabilities

    subroutine multiclass_parameter_margin_jvp(self, x, parameter_dot, margins, &
            margins_dot, status)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), parameter_dot(:)
        real(dp), intent(out) :: margins(:, :), margins_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: child_dot(:)
        integer :: i, first, last, count

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass parameter products: model or input is invalid")
            return
        end if
        if (any(shape(margins) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(margins_dot) /= shape(margins))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass parameter products: output shape is invalid")
            return
        end if
        if (size(parameter_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "LightGBM multiclass parameter products: parameter shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%class_count()
            count = self%one_vs_rest(i)%leaf_parameter_count()
            last = first + count - 1
            allocate(child_dot(count))
            child_dot = parameter_dot(first:last)
            call self%one_vs_rest(i)%predict_leaf_jvp(x, child_dot, margins(:, i), &
                margins_dot(:, i), status)
            deallocate(child_dot)
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_parameter_margin_jvp

    logical function valid_query(self, x) result(valid)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)

        valid = self%initialized
        if (.not. valid) return
        valid = size(x, 1) > 0 .and. size(x, 2) == self%n_inputs
        if (.not. valid) return
        valid = all(ieee_is_finite(x))
    end function valid_query

    pure real(dp) function stable_log_sigmoid(value) result(log_probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            log_probability = -log(1.0_dp + exp(-value))
        else
            log_probability = value - log(1.0_dp + exp(value))
        end if
    end function stable_log_sigmoid

    pure elemental real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

    pure function stable_sigmoid_array(values) result(probabilities)
        real(dp), intent(in) :: values(:)
        real(dp) :: probabilities(size(values))

        probabilities = stable_sigmoid(values)
    end function stable_sigmoid_array

    pure real(dp) function stable_logsumexp(values) result(value)
        real(dp), intent(in) :: values(:)
        real(dp) :: maximum, shifted

        maximum = maxval(values)
        if (.not. ieee_is_finite(maximum)) then
            value = maximum
            return
        end if
        shifted = sum(exp(values - maximum))
        if (shifted <= 0.0_dp .or. .not. ieee_is_finite(shifted)) then
            value = huge(1.0_dp)
        else
            value = maximum + log(shifted)
        end if
    end function stable_logsumexp

    logical function valid_multiclass_model(self) result(valid)
        class(lightgbm_multiclass_t), intent(in) :: self
        integer :: i, estimators

        valid = self%initialized
        if (.not. valid) return
        if (.not. allocated(self%class_label)) return
        if (.not. allocated(self%one_vs_rest)) return
        if (self%n_inputs < 1 .or. size(self%class_label) < 2) return
        if (size(self%one_vs_rest) /= size(self%class_label)) return
        if (self%requested_estimators < 1 .or. self%best_iteration_value < 1 .or. &
            self%best_iteration_value > self%requested_estimators .or. &
            .not. ieee_is_finite(self%best_validation_loss_value)) return
        do i = 2, size(self%class_label)
            if (self%class_label(i) <= self%class_label(i - 1)) return
        end do
        estimators = 0
        do i = 1, size(self%one_vs_rest)
            if (.not. self%one_vs_rest(i)%fitted()) return
            if (self%one_vs_rest(i)%feature_count() /= self%n_inputs) return
            if (trim(self%one_vs_rest(i)%objective_name()) /= "binary") return
            if (i == 1) then
                estimators = self%one_vs_rest(i)%estimator_count()
                if (estimators < 1) return
            else if (self%one_vs_rest(i)%estimator_count() /= estimators) then
                return
            end if
        end do
        valid = .true.
    end function valid_multiclass_model

    subroutine unique_sorted_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer :: i, j, n_unique, temporary

        allocate(classes(size(labels)))
        classes = labels
        do i = 2, size(classes)
            temporary = classes(i)
            j = i - 1
            do while (j >= 1)
                if (classes(j) <= temporary) exit
                classes(j + 1) = classes(j)
                j = j - 1
            end do
            classes(j + 1) = temporary
        end do
        n_unique = 1
        do i = 2, size(classes)
            if (classes(i) /= classes(n_unique)) then
                n_unique = n_unique + 1
                classes(n_unique) = classes(i)
            end if
        end do
        if (n_unique < size(classes)) classes = classes(:n_unique)
    end subroutine unique_sorted_labels

    integer function find_label_index(classes, label) result(index)
        integer, intent(in) :: classes(:), label
        integer :: i

        index = 0
        do i = 1, size(classes)
            if (classes(i) == label) then
                index = i
                return
            end if
        end do
    end function find_label_index

    real(dp) function multiclass_log_loss(probabilities, totals, labels, classes, &
            observation_weight, weight_sum) result(loss)
        real(dp), intent(in) :: probabilities(:, :), totals(:)
        integer, intent(in) :: labels(:), classes(:)
        real(dp), intent(in) :: observation_weight(:), weight_sum
        integer :: i, class_index

        loss = 0.0_dp
        do i = 1, size(labels)
            class_index = find_label_index(classes, labels(i))
            if (class_index < 1 .or. class_index > size(probabilities, 2)) then
                loss = huge(1.0_dp)
                return
            end if
            loss = loss - observation_weight(i)*log(max( &
                probabilities(i, class_index)/totals(i), 1.0e-15_dp))
        end do
        loss = loss/weight_sum
    end function multiclass_log_loss

end module fortml_lightgbm_multiclass
