module fortml_ovr_logistic_classifier
    !! One-vs-rest multiclass wrapper around binary logistic regression.
    !!
    !! Each class owns an independent, regularized binary logistic model.  The
    !! positive class probabilities are normalized row-wise for deployment so
    !! that arbitrary integer labels share the same probability-simplex
    !! contract as the multinomial and GP classifiers.  Fit is a discrete
    !! operation.  Prediction products are smooth with respect to inputs and
    !! packed fitted parameters.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_classification_state, only: classification_state_t
    use fortml_logistic_regression, only: logistic_regression_t
    implicit none
    private

    type, public :: ovr_logistic_classifier_t
        private
        type(logistic_regression_t), allocatable :: models(:)
        integer, allocatable :: class_label(:)
        integer :: n_classes = 0
        integer :: n_features = 0
        integer :: n_parameters_per_model = 0
        logical :: is_fitted = .false.
        type(classification_state_t) :: state
        real(dp), allocatable :: history_x(:, :), history_weight(:)
        integer, allocatable :: history_labels(:)
        real(dp), allocatable :: configured_class_weight(:)
        real(dp) :: configured_l2 = 1.0_dp
        real(dp) :: configured_tolerance = 1.0e-8_dp
        integer :: configured_max_iterations = 200
        logical :: configured_fit_intercept = .true.
        logical :: configuration_initialized = .false.
    contains
        procedure, public :: fit => ovr_logistic_fit
        procedure, public :: partial_fit => ovr_logistic_partial_fit
        procedure, public :: decision_function => ovr_logistic_decision
        procedure, public :: predict_proba => ovr_logistic_predict_proba
        procedure, public :: predict_proba_device => ovr_logistic_predict_proba_device
        procedure, public :: predict_proba_jvp => ovr_logistic_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            ovr_logistic_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => ovr_logistic_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            ovr_logistic_predict_proba_parameter_vjp
        procedure, public :: predict => ovr_logistic_predict
        procedure, public :: predict_device => ovr_logistic_predict_device
        procedure, public :: classes => ovr_logistic_classes
        procedure, public :: class_count => ovr_logistic_class_count
        procedure, public :: feature_count => ovr_logistic_feature_count
        procedure, public :: parameter_count => ovr_logistic_parameter_count
        procedure, public :: parameters => ovr_logistic_parameters
        procedure, public :: set_parameters => ovr_logistic_set_parameters
        procedure, public :: fitted => ovr_logistic_fitted
        procedure, public :: metadata => ovr_logistic_metadata
        procedure, public :: device_supported => ovr_logistic_device_supported
    end type ovr_logistic_classifier_t

    public :: ovr_logistic_fit
    public :: ovr_logistic_decision
    public :: ovr_logistic_predict_proba
    public :: ovr_logistic_predict_proba_device
    public :: ovr_logistic_predict_proba_jvp
    public :: ovr_logistic_predict_proba_parameter_jvp
    public :: ovr_logistic_predict_proba_vjp
    public :: ovr_logistic_predict_proba_parameter_vjp
    public :: ovr_logistic_predict
    public :: ovr_logistic_predict_device
    public :: ovr_logistic_partial_fit

contains

    subroutine ovr_logistic_fit(self, x, labels, status, l2, fit_intercept, &
            max_iterations, tolerance, sample_weight, class_weight)
        class(ovr_logistic_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        integer, allocatable :: classes(:), binary_labels(:)
        real(dp), allocatable :: effective_weight(:), class_factors(:)
        integer :: i, j, n_samples, n_features, n_classes
        integer :: requested_iterations
        real(dp) :: requested_l2, requested_tolerance
        logical :: requested_fit_intercept
        type(fortnum_status_t) :: state_status
        character(256) :: failure_message

        self%is_fitted = .false.
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic fit: inputs must be finite")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic fit: at least two classes are required")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        requested_l2 = 1.0_dp
        if (present(l2)) requested_l2 = l2
        requested_fit_intercept = .true.
        if (present(fit_intercept)) requested_fit_intercept = fit_intercept
        requested_iterations = 200
        if (present(max_iterations)) requested_iterations = max_iterations
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        allocate(effective_weight(n_samples), binary_labels(n_samples))
        effective_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic fit: sample weights must be finite and nonnegative")
                return
            end if
            effective_weight = sample_weight
        end if
        allocate(class_factors(n_classes))
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic fit: class weights must be finite and positive "// &
                    "in sorted class order")
                return
            end if
            class_factors = class_weight
            do i = 1, n_samples
                do j = 1, n_classes
                    if (labels(i) == classes(j)) then
                        effective_weight(i) = effective_weight(i)*class_factors(j)
                        exit
                    end if
                end do
            end do
        end if
        if (.not. ieee_is_finite(sum(effective_weight)) .or. &
            sum(effective_weight) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic fit: effective weights must have positive mass")
            return
        end if

        allocate(self%models(n_classes), self%class_label(n_classes))
        self%class_label = classes
        self%n_classes = n_classes
        self%n_features = n_features
        do j = 1, n_classes
            binary_labels = 0
            where (labels == classes(j)) binary_labels = 1
            call self%models(j)%fit(x, binary_labels, status, l2=l2, &
                fit_intercept=fit_intercept, max_iterations=max_iterations, &
                tolerance=tolerance, sample_weight=effective_weight)
            if (status%code /= FORTNUM_OK) then
                write (failure_message, '(a,i0,a,a)') &
                    "OVR logistic fit: binary estimator ", j, " failed: ", &
                    trim(status%msg)
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    trim(failure_message))
                return
            end if
        end do
        self%n_parameters_per_model = self%models(1)%parameter_count()
        self%is_fitted = .true.
        self%configured_l2 = requested_l2
        self%configured_fit_intercept = requested_fit_intercept
        self%configured_max_iterations = requested_iterations
        self%configured_tolerance = requested_tolerance
        if (allocated(self%configured_class_weight)) then
            deallocate(self%configured_class_weight)
        end if
        allocate(self%configured_class_weight(n_classes))
        self%configured_class_weight = class_factors
        self%configuration_initialized = .true.
        call self%state%initialize(classes, n_features, state_status)
        if (state_status%code /= FORTNUM_OK) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(state_status%msg))
            return
        end if
        call self%state%append_batch(n_samples, state_status)
        if (state_status%code /= FORTNUM_OK) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(state_status%msg))
            return
        end if
        call ovr_store_history(self, x, labels, sample_weight, state_status)
        if (state_status%code /= FORTNUM_OK) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(state_status%msg))
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_fit

    subroutine ovr_logistic_partial_fit(self, x, labels, status, classes, &
            l2, fit_intercept, max_iterations, tolerance, sample_weight, &
            class_weight)
        class(ovr_logistic_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: classes(:)
        real(dp), intent(in), optional :: l2, tolerance, sample_weight(:), &
            class_weight(:)
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        type(ovr_logistic_classifier_t) :: candidate
        type(classification_state_t) :: candidate_state
        type(fortnum_status_t) :: local_status
        integer, allocatable :: batch_classes(:), state_classes(:)
        integer, allocatable :: combined_labels(:)
        real(dp), allocatable :: combined_x(:, :), combined_weight(:)
        real(dp), allocatable :: requested_class_weight(:)
        real(dp) :: requested_l2, requested_tolerance
        integer :: requested_iterations, n_old, n_new, n_features
        logical :: requested_fit_intercept, has_all_classes

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic partial_fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic partial_fit: inputs must be finite")
            return
        end if
        n_features = size(x, 2)
        n_new = size(x, 1)
        allocate(combined_weight(n_new))
        combined_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_new .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic partial_fit: sample weights are invalid")
                return
            end if
            combined_weight = sample_weight
        end if

        if (.not. self%state%initialized()) then
            if (present(classes)) then
                allocate(batch_classes(size(classes)))
                batch_classes = classes
            else
                call sorted_unique_labels(labels, batch_classes)
            end if
            call candidate_state%initialize(batch_classes, n_features, local_status)
            if (local_status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, trim(local_status%msg))
                return
            end if
            do n_old = 1, size(labels)
                if (.not. label_in_classes(labels(n_old), batch_classes)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "OVR logistic partial_fit: batch contains an unknown class")
                    return
                end if
            end do
            requested_l2 = 1.0_dp
            if (present(l2)) requested_l2 = l2
            requested_fit_intercept = .true.
            if (present(fit_intercept)) requested_fit_intercept = fit_intercept
            requested_iterations = 200
            if (present(max_iterations)) requested_iterations = max_iterations
            requested_tolerance = 1.0e-8_dp
            if (present(tolerance)) requested_tolerance = tolerance
            allocate(requested_class_weight(size(batch_classes)))
            requested_class_weight = 1.0_dp
            if (present(class_weight)) then
                if (size(class_weight) /= size(batch_classes) .or. &
                    any(.not. ieee_is_finite(class_weight)) .or. &
                    any(class_weight <= 0.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "OVR logistic partial_fit: class weights are invalid")
                    return
                end if
                requested_class_weight = class_weight
            end if
            has_all_classes = labels_cover_classes(labels, batch_classes)
            if (has_all_classes) then
                call candidate%fit(x, labels, local_status, l2=requested_l2, &
                    fit_intercept=requested_fit_intercept, &
                    max_iterations=requested_iterations, tolerance=requested_tolerance, &
                    sample_weight=combined_weight, class_weight=requested_class_weight)
                if (local_status%code /= FORTNUM_OK) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, trim(local_status%msg))
                    return
                end if
            else
                candidate%is_fitted = .false.
                candidate%n_features = n_features
                candidate%n_classes = size(batch_classes)
                allocate(candidate%class_label(size(batch_classes)))
                candidate%class_label = batch_classes
            end if
            candidate%configured_l2 = requested_l2
            candidate%configured_fit_intercept = requested_fit_intercept
            candidate%configured_max_iterations = requested_iterations
            candidate%configured_tolerance = requested_tolerance
            candidate%configured_class_weight = requested_class_weight
            candidate%configuration_initialized = .true.
            candidate%state = candidate_state
            call candidate%state%append_batch(n_new, local_status)
            if (local_status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, trim(local_status%msg))
                return
            end if
            call ovr_store_history(candidate, x, labels, combined_weight, local_status)
            if (local_status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, trim(local_status%msg))
                return
            end if
            select type (self)
            type is (ovr_logistic_classifier_t)
                self = candidate
            end select
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        call self%state%validate_batch(labels, n_features, local_status)
        if (local_status%code /= FORTNUM_OK) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(local_status%msg))
            return
        end if
        state_classes = self%state%classes()
        if (present(classes)) then
            if (size(classes) /= size(state_classes) .or. &
                any(classes /= state_classes)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic partial_fit: classes do not match initialized metadata")
                return
            end if
        end if
        requested_l2 = self%configured_l2
        if (present(l2)) requested_l2 = l2
        requested_fit_intercept = self%configured_fit_intercept
        if (present(fit_intercept)) requested_fit_intercept = fit_intercept
        requested_iterations = self%configured_max_iterations
        if (present(max_iterations)) requested_iterations = max_iterations
        requested_tolerance = self%configured_tolerance
        if (present(tolerance)) requested_tolerance = tolerance
        requested_class_weight = self%configured_class_weight
        if (present(class_weight)) then
            if (size(class_weight) /= size(state_classes) .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic partial_fit: class weights are invalid")
                return
            end if
            requested_class_weight = class_weight
        end if
        n_old = size(self%history_labels)
        allocate(combined_x(n_old + n_new, n_features), &
            combined_labels(n_old + n_new))
        combined_x(:n_old, :) = self%history_x
        combined_x(n_old + 1:, :) = x
        combined_labels(:n_old) = self%history_labels
        combined_labels(n_old + 1:) = labels
        if (allocated(combined_weight)) deallocate(combined_weight)
        allocate(combined_weight(n_old + n_new))
        combined_weight(:n_old) = self%history_weight
        if (present(sample_weight)) then
            combined_weight(n_old + 1:) = sample_weight
        else
            combined_weight(n_old + 1:) = 1.0_dp
        end if
        has_all_classes = labels_cover_classes(combined_labels, state_classes)
        candidate = self
        if (has_all_classes) then
            call candidate%fit(combined_x, combined_labels, local_status, &
                l2=requested_l2, fit_intercept=requested_fit_intercept, &
                max_iterations=requested_iterations, tolerance=requested_tolerance, &
                sample_weight=combined_weight, class_weight=requested_class_weight)
            if (local_status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, trim(local_status%msg))
                return
            end if
            candidate%state = self%state
            call candidate%state%append_batch(n_new, local_status)
        else
            call candidate%state%append_batch(n_new, local_status)
        end if
        if (local_status%code /= FORTNUM_OK) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(local_status%msg))
            return
        end if
        candidate%configured_l2 = requested_l2
        candidate%configured_fit_intercept = requested_fit_intercept
        candidate%configured_max_iterations = requested_iterations
        candidate%configured_tolerance = requested_tolerance
        candidate%configured_class_weight = requested_class_weight
        candidate%configuration_initialized = .true.
        call ovr_store_history(candidate, combined_x, combined_labels, combined_weight, &
            local_status)
        if (local_status%code /= FORTNUM_OK) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(local_status%msg))
            return
        end if
        select type (self)
        type is (ovr_logistic_classifier_t)
            self = candidate
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_partial_fit

    subroutine ovr_logistic_decision(self, x, scores, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_scores(:)
        integer :: j

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic decision: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(scores) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic decision: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic decision: inputs must be finite")
            return
        end if
        allocate(binary_scores(size(x, 1)))
        do j = 1, self%n_classes
            call self%models(j)%decision_function(x, binary_scores, status)
            if (status%code /= FORTNUM_OK) return
            scores(:, j) = binary_scores
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_decision

    subroutine ovr_logistic_predict_proba(self, x, probabilities, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :)
        real(dp) :: normalization
        integer :: i, j

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability: inputs must be finite")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(probabilities(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic probability: invalid normalization")
                return
            end if
            probabilities(i, :) = probabilities(i, :)/normalization
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba

    subroutine ovr_logistic_predict_proba_device(self, device, x, probabilities, &
            status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "OVR logistic device probability: resident CUDA kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic device probability: device kind is invalid")
        end select
    end subroutine ovr_logistic_predict_proba_device

    subroutine ovr_logistic_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), binary_dot(:, :)
        real(dp), allocatable :: raw(:, :), raw_dot(:, :)
        real(dp), allocatable :: theta_dot(:)
        real(dp) :: normalization, normalization_dot
        integer :: i, j

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability JVP: tangent or output shape is invalid")
            return
        end if
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability JVP: model or input shape is invalid")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2), binary_dot(size(x, 1), 2))
        allocate(raw(size(x, 1), self%n_classes), raw_dot(size(x, 1), self%n_classes))
        allocate(theta_dot(self%n_parameters_per_model))
        theta_dot = 0.0_dp
        do j = 1, self%n_classes
            call self%models(j)%predict_proba_jvp(x, theta_dot, x_dot, &
                binary_probabilities, binary_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
            raw_dot(:, j) = binary_dot(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(raw(i, :))
            normalization_dot = sum(raw_dot(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic probability JVP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/normalization
            probabilities_dot(i, :) = (raw_dot(i, :)*normalization - &
                raw(i, :)*normalization_dot)/(normalization*normalization)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba_jvp

    subroutine ovr_logistic_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), binary_dot(:, :)
        real(dp), allocatable :: raw(:, :), raw_dot(:, :), x_dot(:, :)
        real(dp), allocatable :: theta_slice(:)
        real(dp) :: normalization, normalization_dot
        integer :: i, j, first, last

        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic parameter JVP: model, parameter, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic parameter JVP: inputs and tangents must be finite")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2), binary_dot(size(x, 1), 2))
        allocate(raw(size(x, 1), self%n_classes), raw_dot(size(x, 1), self%n_classes))
        allocate(x_dot(size(x, 1), self%n_features), &
            theta_slice(self%n_parameters_per_model))
        x_dot = 0.0_dp
        do j = 1, self%n_classes
            first = (j - 1)*self%n_parameters_per_model + 1
            last = j*self%n_parameters_per_model
            theta_slice = theta_dot(first:last)
            call self%models(j)%predict_proba_jvp(x, theta_slice, x_dot, &
                binary_probabilities, binary_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
            raw_dot(:, j) = binary_dot(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(raw(i, :))
            normalization_dot = sum(raw_dot(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic parameter JVP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/normalization
            probabilities_dot(i, :) = (raw_dot(i, :)*normalization - &
                raw(i, :)*normalization_dot)/(normalization*normalization)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba_parameter_jvp

    subroutine ovr_logistic_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), raw(:, :), raw_bar(:)
        real(dp), allocatable :: binary_probabilities(:, :)
        real(dp), allocatable :: binary_probabilities_bar(:, :), binary_theta_bar(:)
        real(dp), allocatable :: binary_x_bar(:, :)
        real(dp) :: dot_product_bar
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability VJP: model or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            raw(size(x, 1), self%n_classes), binary_probabilities(size(x, 1), 2))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
        end do
        do i = 1, size(x, 1)
            dot_product_bar = sum(raw(i, :))
            if (.not. ieee_is_finite(dot_product_bar) .or. &
                dot_product_bar <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic probability VJP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/dot_product_bar
        end do
        allocate(binary_probabilities_bar(size(x, 1), 2), &
            binary_x_bar(size(x, 1), self%n_features))
        allocate(binary_theta_bar(self%n_parameters_per_model), raw_bar(size(x, 1)))
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                dot_product_bar = sum(probabilities_bar(i, :)*probabilities(i, :))
                raw_bar(i) = (probabilities_bar(i, j) - dot_product_bar)/ &
                    sum(raw(i, :))
            end do
            binary_probabilities_bar(:, 1) = 0.0_dp
            binary_probabilities_bar(:, 2) = raw_bar
            call self%models(j)%predict_proba_vjp(x, binary_probabilities_bar, &
                binary_theta_bar, binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + binary_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba_vjp

    subroutine ovr_logistic_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), raw(:, :), raw_bar(:)
        real(dp), allocatable :: binary_probabilities(:, :), binary_probabilities_bar(:, :)
        real(dp), allocatable :: binary_theta_bar(:), binary_x_bar(:, :)
        real(dp) :: normalization, dot_product_bar
        integer :: i, j, first, last

        theta_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic parameter VJP: model, parameter, or cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic parameter VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            raw(size(x, 1), self%n_classes), binary_probabilities(size(x, 1), 2))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(raw(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic parameter VJP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/normalization
        end do
        allocate(raw_bar(size(x, 1)), &
            binary_probabilities_bar(size(x, 1), 2), &
            binary_theta_bar(self%n_parameters_per_model), &
            binary_x_bar(size(x, 1), self%n_features))
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                dot_product_bar = sum(probabilities_bar(i, :)*probabilities(i, :))
                raw_bar(i) = (probabilities_bar(i, j) - dot_product_bar)/ &
                    sum(raw(i, :))
            end do
            binary_probabilities_bar(:, 1) = 0.0_dp
            binary_probabilities_bar(:, 2) = raw_bar
            call self%models(j)%predict_proba_vjp(x, binary_probabilities_bar, &
                binary_theta_bar, binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            first = (j - 1)*self%n_parameters_per_model + 1
            last = j*self%n_parameters_per_model
            theta_bar(first:last) = binary_theta_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba_parameter_vjp

    subroutine ovr_logistic_predict(self, x, labels, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, j, best

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            best = 1
            do j = 2, self%n_classes
                if (probabilities(i, j) > probabilities(i, best)) best = j
            end do
            labels(i) = self%class_label(best)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict

    subroutine ovr_logistic_predict_device(self, device, x, labels, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic device predict: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "OVR logistic device predict: resident CUDA kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic device predict: device kind is invalid")
        end select
    end subroutine ovr_logistic_predict_device

    function ovr_logistic_classes(self) result(labels)
        class(ovr_logistic_classifier_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function ovr_logistic_classes

    integer function ovr_logistic_class_count(self) result(count)
        class(ovr_logistic_classifier_t), intent(in) :: self

        count = self%n_classes
    end function ovr_logistic_class_count

    integer function ovr_logistic_feature_count(self) result(count)
        class(ovr_logistic_classifier_t), intent(in) :: self

        count = self%n_features
    end function ovr_logistic_feature_count

    integer function ovr_logistic_parameter_count(self) result(count)
        class(ovr_logistic_classifier_t), intent(in) :: self

        count = self%n_classes*self%n_parameters_per_model
    end function ovr_logistic_parameter_count

    function ovr_logistic_parameters(self) result(values)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:), model_values(:)
        integer :: i, first, last

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        do i = 1, self%n_classes
            model_values = self%models(i)%parameters()
            first = (i - 1)*self%n_parameters_per_model + 1
            last = i*self%n_parameters_per_model
            values(first:last) = model_values
        end do
    end function ovr_logistic_parameters

    subroutine ovr_logistic_set_parameters(self, values, status)
        class(ovr_logistic_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic set_parameters: model, shape, or values are invalid")
            return
        end if
        do i = 1, self%n_classes
            first = (i - 1)*self%n_parameters_per_model + 1
            last = i*self%n_parameters_per_model
            call self%models(i)%set_parameters(values(first:last), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_set_parameters

    logical function ovr_logistic_fitted(self) result(is_fitted)
        class(ovr_logistic_classifier_t), intent(in) :: self

        is_fitted = self%is_fitted
    end function ovr_logistic_fitted

    function ovr_logistic_metadata(self) result(metadata)
        class(ovr_logistic_classifier_t), intent(in) :: self
        type(classification_state_t) :: metadata

        metadata = self%state
    end function ovr_logistic_metadata

    logical function ovr_logistic_device_supported(self, device_kind) result(supported)
        class(ovr_logistic_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function ovr_logistic_device_supported

    subroutine ovr_store_history(self, x, labels, sample_weight, status)
        class(ovr_logistic_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(in), optional :: sample_weight(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate_x(:, :), candidate_weight(:)
        integer, allocatable :: candidate_labels(:)

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic history: batch dimensions are invalid")
            return
        end if
        allocate(candidate_x(size(x, 1), size(x, 2)), &
            candidate_labels(size(labels)), candidate_weight(size(labels)))
        candidate_x = x
        candidate_labels = labels
        candidate_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic history: sample weights are invalid")
                return
            end if
            candidate_weight = sample_weight
        end if
        call move_alloc(candidate_x, self%history_x)
        call move_alloc(candidate_labels, self%history_labels)
        call move_alloc(candidate_weight, self%history_weight)
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_store_history

    logical function label_in_classes(label, classes) result(found)
        integer, intent(in) :: label, classes(:)
        integer :: i

        found = .false.
        do i = 1, size(classes)
            if (label == classes(i)) then
                found = .true.
                return
            end if
        end do
    end function label_in_classes

    logical function labels_cover_classes(labels, classes) result(covered)
        integer, intent(in) :: labels(:), classes(:)
        integer :: i

        covered = .true.
        do i = 1, size(classes)
            if (.not. any(labels == classes(i))) then
                covered = .false.
                return
            end if
        end do
    end function labels_cover_classes

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, n
        logical :: found

        allocate(work(size(labels)))
        n = 0
        do i = 1, size(labels)
            found = .false.
            do j = 1, n
                if (work(j) == labels(i)) then
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) then
                n = n + 1
                work(n) = labels(i)
            end if
        end do
        do i = 2, n
            j = i
            do while (j > 1)
                if (work(j) >= work(j - 1)) exit
                call swap(work(j), work(j - 1))
                j = j - 1
            end do
        end do
        allocate(classes(n))
        classes = work(:n)
    end subroutine sorted_unique_labels

    subroutine swap(left, right)
        integer, intent(inout) :: left, right
        integer :: temporary

        temporary = left
        left = right
        right = temporary
    end subroutine swap

end module fortml_ovr_logistic_classifier
