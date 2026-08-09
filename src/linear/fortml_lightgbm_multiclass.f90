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
        procedure, public :: predict_proba_device => &
            lgbm_multiclass_predict_proba_device
        procedure, public :: predict_proba_staged => &
            lgbm_multiclass_predict_proba_staged
        procedure, public :: decision_function_staged => &
            lgbm_multiclass_decision_function_staged
        procedure, public :: predict_proba_jvp => lgbm_multiclass_predict_proba_jvp
        procedure, public :: predict_proba_vjp => lgbm_multiclass_predict_proba_vjp
        procedure, public :: decision_function => lgbm_multiclass_decision_function
        procedure, public :: predict => lgbm_multiclass_predict
        procedure, public :: predict_device => lgbm_multiclass_predict_device
        procedure, public :: classes => lgbm_multiclass_classes
        procedure, public :: feature_count => lgbm_multiclass_feature_count
        procedure, public :: class_count => lgbm_multiclass_class_count
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

    logical function valid_query(self, x) result(valid)
        class(lightgbm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)

        valid = self%initialized
        if (.not. valid) return
        valid = size(x, 1) > 0 .and. size(x, 2) == self%n_inputs
        if (.not. valid) return
        valid = all(ieee_is_finite(x))
    end function valid_query

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
