module fortml_multilabel_logistic_classifier
    !! Independent binary logistic heads for multilabel-indicator targets.
    !!
    !! The indicator matrix has shape ``(n_samples, n_labels)`` and contains
    !! only zero and one.  Every column is fitted with the same weighted
    !! logistic objective and the packed parameter vector is laid out by
    !! label, in model order.  Positive probabilities are returned as an
    !! ``(n_samples,n_labels)`` matrix; this avoids the ambiguous list-of-
    !! matrices representation used by some host language APIs.
    !!
    !! Prediction products are exact JVP/VJP products with respect to inputs
    !! and packed parameters.  Fitting is a host operation.  CPU device calls
    !! dispatch to the same implementation, while CUDA calls return a typed
    !! refusal until a resident multi-head kernel is linked.  There is no
    !! hidden host fallback for an accelerator request.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_logistic_regression, only: logistic_regression_t
    implicit none
    private

    type, public :: multilabel_logistic_classifier_t
        private
        type(logistic_regression_t), allocatable :: models(:)
        real(dp), allocatable :: decision_threshold(:)
        integer :: n_labels = 0
        integer :: n_features = 0
        integer :: n_parameters_per_label = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => multilabel_logistic_fit
        procedure, public :: decision_function => &
            multilabel_logistic_decision
        procedure, public :: decision_function_device => &
            multilabel_logistic_decision_device
        procedure, public :: predict_proba => multilabel_logistic_predict_proba
        procedure, public :: predict_proba_device => &
            multilabel_logistic_predict_proba_device
        procedure, public :: predict_proba_jvp => &
            multilabel_logistic_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            multilabel_logistic_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => &
            multilabel_logistic_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            multilabel_logistic_predict_proba_parameter_vjp
        procedure, public :: predict => multilabel_logistic_predict
        procedure, public :: predict_device => multilabel_logistic_predict_device
        procedure, public :: set_parameters => multilabel_logistic_set_parameters
        procedure, public :: parameters => multilabel_logistic_parameters
        procedure, public :: parameter_count => multilabel_logistic_parameter_count
        procedure, public :: label_count => multilabel_logistic_label_count
        procedure, public :: feature_count => multilabel_logistic_feature_count
        procedure, public :: thresholds => multilabel_logistic_thresholds
        procedure, public :: set_thresholds => multilabel_logistic_set_thresholds
        procedure, public :: fitted => multilabel_logistic_fitted
        procedure, public :: device_supported => &
            multilabel_logistic_device_supported
    end type multilabel_logistic_classifier_t

    public :: multilabel_logistic_fit
    public :: multilabel_logistic_decision
    public :: multilabel_logistic_decision_device
    public :: multilabel_logistic_predict_proba
    public :: multilabel_logistic_predict_proba_device
    public :: multilabel_logistic_predict_proba_jvp
    public :: multilabel_logistic_predict_proba_parameter_jvp
    public :: multilabel_logistic_predict_proba_vjp
    public :: multilabel_logistic_predict_proba_parameter_vjp
    public :: multilabel_logistic_predict
    public :: multilabel_logistic_predict_device

contains

    subroutine multilabel_logistic_fit(self, x, indicators, status, l2, &
            fit_intercept, max_iterations, tolerance, sample_weight, &
            class_weight, thresholds)
        class(multilabel_logistic_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:, :)
        real(dp), intent(in), optional :: thresholds(:)
        integer, allocatable :: binary_labels(:)
        integer :: i, j, n_samples, n_features, n_labels
        real(dp) :: weight_sum
        character(256) :: failure_message

        self%is_fitted = .false.
        self%n_labels = 0
        self%n_features = 0
        self%n_parameters_per_label = 0
        if (allocated(self%models)) deallocate(self%models)
        if (allocated(self%decision_threshold)) deallocate(self%decision_threshold)
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(indicators, 1) /= size(x, 1) .or. size(indicators, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic fit: inputs must be finite")
            return
        end if
        if (any((indicators /= 0) .and. (indicators /= 1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic fit: indicators must be zero or one")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_labels = size(indicators, 2)
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multilabel logistic fit: sample weights must be finite "// &
                    "and nonnegative")
                return
            end if
            weight_sum = sum(sample_weight)
        else
            weight_sum = real(n_samples, dp)
        end if
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic fit: sample weights have no positive mass")
            return
        end if
        if (present(class_weight)) then
            if (any(shape(class_weight) /= [2, n_labels]) .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multilabel logistic fit: class_weight must have shape "// &
                    "(2,n_labels) and contain positive finite values")
                return
            end if
        end if
        allocate(self%models(n_labels), self%decision_threshold(n_labels), &
            binary_labels(n_samples))
        self%decision_threshold = 0.5_dp
        if (present(thresholds)) then
            if (size(thresholds) /= n_labels .or. &
                any(.not. ieee_is_finite(thresholds)) .or. &
                any(thresholds <= 0.0_dp) .or. any(thresholds >= 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multilabel logistic fit: thresholds must be finite in (0,1)")
                return
            end if
            self%decision_threshold = thresholds
        end if
        self%n_features = n_features
        self%n_labels = n_labels
        do j = 1, n_labels
            if (minval(indicators(:, j)) == maxval(indicators(:, j))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multilabel logistic fit: every label needs both classes")
                return
            end if
            binary_labels = indicators(:, j)
            if (present(class_weight)) then
                call self%models(j)%fit(x, binary_labels, status, l2=l2, &
                    fit_intercept=fit_intercept, max_iterations=max_iterations, &
                    tolerance=tolerance, sample_weight=sample_weight, &
                    class_weight=class_weight(:, j))
            else
                call self%models(j)%fit(x, binary_labels, status, l2=l2, &
                    fit_intercept=fit_intercept, max_iterations=max_iterations, &
                    tolerance=tolerance, sample_weight=sample_weight)
            end if
            if (status%code /= FORTNUM_OK) then
                write (failure_message, '(a,i0,a,a)') &
                    "multilabel logistic fit: label head ", j, " failed: ", &
                    trim(status%msg)
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    trim(failure_message))
                return
            end if
        end do
        self%n_parameters_per_label = self%models(1)%parameter_count()
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_fit

    subroutine multilabel_logistic_decision(self, x, scores, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic decision: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(scores) /= [size(x, 1), self%n_labels])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic decision: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic decision: inputs must be finite")
            return
        end if
        do j = 1, self%n_labels
            call self%models(j)%decision_function(x, scores(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_decision

    subroutine multilabel_logistic_decision_device(self, device, x, scores, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multilabel logistic device decision: no resident CUDA kernel "// &
                "is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic device decision: device kind is invalid")
        end select
    end subroutine multilabel_logistic_decision_device

    subroutine multilabel_logistic_predict_proba(self, x, probabilities, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :)
        integer :: j

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic probability: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_labels])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic probability: input or output shape is invalid")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2))
        do j = 1, self%n_labels
            call self%models(j)%predict_proba(x, binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_predict_proba

    subroutine multilabel_logistic_predict_proba_device(self, device, x, &
            probabilities, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multilabel logistic device probability: no resident CUDA kernel "// &
                "is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic device probability: device kind is invalid")
        end select
    end subroutine multilabel_logistic_predict_proba_device

    subroutine multilabel_logistic_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), binary_dot(:, :)
        real(dp), allocatable :: theta_dot(:)
        integer :: j

        if (.not. model_and_shapes_valid(self, x, probabilities, status, &
            "multilabel logistic probability JVP")) return
        if (any(shape(x_dot) /= shape(x)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic probability JVP: tangent or output shape "// &
                "is invalid")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2), binary_dot(size(x, 1), 2), &
            theta_dot(self%n_parameters_per_label))
        theta_dot = 0.0_dp
        do j = 1, self%n_labels
            call self%models(j)%predict_proba_jvp(x, theta_dot, x_dot, &
                binary_probabilities, binary_dot, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
            probabilities_dot(:, j) = binary_dot(:, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_predict_proba_jvp

    subroutine multilabel_logistic_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), binary_dot(:, :)
        real(dp), allocatable :: theta_slice(:), x_dot(:, :)
        integer :: j, first, last

        if (.not. model_and_shapes_valid(self, x, probabilities, status, &
            "multilabel logistic parameter JVP")) return
        if (size(theta_dot) /= self%parameter_count() .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic parameter JVP: tangent or output shape "// &
                "is invalid")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2), binary_dot(size(x, 1), 2), &
            theta_slice(self%n_parameters_per_label), &
            x_dot(size(x, 1), self%n_features))
        x_dot = 0.0_dp
        do j = 1, self%n_labels
            first = (j - 1)*self%n_parameters_per_label + 1
            last = j*self%n_parameters_per_label
            theta_slice = theta_dot(first:last)
            call self%models(j)%predict_proba_jvp(x, theta_slice, x_dot, &
                binary_probabilities, binary_dot, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
            probabilities_dot(:, j) = binary_dot(:, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_predict_proba_parameter_jvp

    subroutine multilabel_logistic_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities_bar(:, :), binary_theta_bar(:)
        real(dp), allocatable :: binary_x_bar(:, :)
        integer :: j

        x_bar = 0.0_dp
        if (.not. model_and_shapes_valid(self, x, probabilities_bar, status, &
            "multilabel logistic probability VJP")) return
        if (any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic probability VJP: cotangent or output shape "// &
                "is invalid")
            return
        end if
        allocate(binary_probabilities_bar(size(x, 1), 2), &
            binary_theta_bar(self%n_parameters_per_label), &
            binary_x_bar(size(x, 1), self%n_features))
        binary_probabilities_bar(:, 1) = 0.0_dp
        do j = 1, self%n_labels
            binary_probabilities_bar(:, 2) = probabilities_bar(:, j)
            call self%models(j)%predict_proba_vjp(x, binary_probabilities_bar, &
                binary_theta_bar, binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + binary_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_predict_proba_vjp

    subroutine multilabel_logistic_predict_proba_parameter_vjp(self, x, &
            probabilities_bar, theta_bar, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities_bar(:, :), binary_theta_bar(:)
        real(dp), allocatable :: binary_x_bar(:, :)
        integer :: j, first, last

        theta_bar = 0.0_dp
        if (.not. model_and_shapes_valid(self, x, probabilities_bar, status, &
            "multilabel logistic parameter VJP")) return
        if (size(theta_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic parameter VJP: cotangent or output shape "// &
                "is invalid")
            return
        end if
        allocate(binary_probabilities_bar(size(x, 1), 2), &
            binary_theta_bar(self%n_parameters_per_label), &
            binary_x_bar(size(x, 1), self%n_features))
        binary_probabilities_bar(:, 1) = 0.0_dp
        do j = 1, self%n_labels
            binary_probabilities_bar(:, 2) = probabilities_bar(:, j)
            call self%models(j)%predict_proba_vjp(x, binary_probabilities_bar, &
                binary_theta_bar, binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            first = (j - 1)*self%n_parameters_per_label + 1
            last = j*self%n_parameters_per_label
            theta_bar(first:last) = binary_theta_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_predict_proba_parameter_vjp

    subroutine multilabel_logistic_predict(self, x, indicators, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, j

        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(indicators) /= [size(x, 1), self%n_labels])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic predict: model, input, or output shape "// &
                "is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_labels))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_labels
            do i = 1, size(x, 1)
                indicators(i, j) = 0
                if (probabilities(i, j) >= self%decision_threshold(j)) &
                    indicators(i, j) = 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_predict

    subroutine multilabel_logistic_predict_device(self, device, x, indicators, status)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic device predict: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, indicators, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multilabel logistic device predict: no resident CUDA kernel "// &
                "is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic device predict: device kind is invalid")
        end select
    end subroutine multilabel_logistic_predict_device

    function multilabel_logistic_parameters(self) result(values)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:), model_values(:)
        integer :: j, first, last

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        do j = 1, self%n_labels
            model_values = self%models(j)%parameters()
            first = (j - 1)*self%n_parameters_per_label + 1
            last = j*self%n_parameters_per_label
            values(first:last) = model_values
        end do
    end function multilabel_logistic_parameters

    subroutine multilabel_logistic_set_parameters(self, values, status)
        class(multilabel_logistic_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, first, last

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic set_parameters: model or parameter shape "// &
                "is invalid")
            return
        end if
        do j = 1, self%n_labels
            first = (j - 1)*self%n_parameters_per_label + 1
            last = j*self%n_parameters_per_label
            call self%models(j)%set_parameters(values(first:last), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_set_parameters

    integer function multilabel_logistic_parameter_count(self) result(count)
        class(multilabel_logistic_classifier_t), intent(in) :: self

        count = self%n_labels*self%n_parameters_per_label
    end function multilabel_logistic_parameter_count

    integer function multilabel_logistic_label_count(self) result(count)
        class(multilabel_logistic_classifier_t), intent(in) :: self

        count = self%n_labels
    end function multilabel_logistic_label_count

    integer function multilabel_logistic_feature_count(self) result(count)
        class(multilabel_logistic_classifier_t), intent(in) :: self

        count = self%n_features
    end function multilabel_logistic_feature_count

    function multilabel_logistic_thresholds(self) result(values)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%decision_threshold)) then
            values = self%decision_threshold
        else
            allocate(values(0))
        end if
    end function multilabel_logistic_thresholds

    subroutine multilabel_logistic_set_thresholds(self, values, status)
        class(multilabel_logistic_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted .or. size(values) /= self%n_labels .or. &
            any(.not. ieee_is_finite(values)) .or. any(values <= 0.0_dp) .or. &
            any(values >= 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel logistic set_thresholds: values must be finite "// &
                "and in (0,1)")
            return
        end if
        self%decision_threshold = values
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_logistic_set_thresholds

    logical function multilabel_logistic_fitted(self) result(is_fitted)
        class(multilabel_logistic_classifier_t), intent(in) :: self

        is_fitted = self%is_fitted
    end function multilabel_logistic_fitted

    logical function multilabel_logistic_device_supported(self, device_kind) &
            result(supported)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function multilabel_logistic_device_supported

    logical function model_and_shapes_valid(self, x, values, status, operation) &
            result(valid)
        class(multilabel_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), values(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        valid = .false.
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(values) /= [size(x, 1), self%n_labels])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model, input, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": inputs must be finite")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function model_and_shapes_valid

end module fortml_multilabel_logistic_classifier
