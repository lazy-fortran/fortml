module fortml_gp_multiclass_classification
    !! One-vs-rest multiclass Gaussian-process classification.
    !!
    !! Each class owns an independent Laplace GP classifier from
    !! ``fortml_gp_classification``.  The positive columns are normalized over
    !! classes after prediction, giving a deterministic probability simplex
    !! while preserving the binary likelihood and kernel contracts.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t, &
        GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
    implicit none
    private

    type, public :: gp_multiclass_classification_options_t
        integer :: likelihood = GP_LIKELIHOOD_LOGISTIC
        integer :: max_iterations = 50
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: jitter = 1.0e-8_dp
        real(dp) :: damping = 1.0_dp
    end type gp_multiclass_classification_options_t

    type, public :: gp_multiclass_classification_state_t
        integer :: class_count = 0
        integer :: total_iterations = 0
        real(dp) :: log_posterior = -huge(1.0_dp)
        logical :: converged = .false.
    end type gp_multiclass_classification_state_t

    type, public :: gp_multiclass_classification_t
        private
        type(gp_classification_t), allocatable :: models(:)
        integer, allocatable :: class_label(:)
        integer :: n_classes = 0
        integer :: n_features = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => gp_multiclass_classification_fit
        procedure, public :: predict_proba => &
            gp_multiclass_classification_predict_proba
        procedure, public :: predict_log_proba => &
            gp_multiclass_classification_predict_log_proba
        procedure, public :: predict_log_proba_device => &
            gp_multiclass_classification_predict_log_proba_device
        procedure, public :: predict_proba_device => &
            gp_multiclass_classification_predict_proba_device
        procedure, public :: predict_proba_jvp => &
            gp_multiclass_classification_predict_proba_jvp
        procedure, public :: predict_proba_vjp => &
            gp_multiclass_classification_predict_proba_vjp
        procedure, public :: predict_log_proba_jvp => &
            gp_multiclass_classification_predict_log_proba_jvp
        procedure, public :: predict_log_proba_vjp => &
            gp_multiclass_classification_predict_log_proba_vjp
        procedure, public :: predict_proba_parameter_jvp => &
            gp_multiclass_classification_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            gp_multiclass_classification_predict_proba_parameter_vjp
        procedure, public :: predict_log_proba_parameter_jvp => &
            gp_multiclass_classification_predict_log_proba_parameter_jvp
        procedure, public :: predict_log_proba_parameter_vjp => &
            gp_multiclass_classification_predict_log_proba_parameter_vjp
        procedure, public :: predict_proba_parameter_jvp_device => &
            gp_multiclass_classification_predict_proba_parameter_jvp_device
        procedure, public :: predict_proba_parameter_vjp_device => &
            gp_multiclass_classification_predict_proba_parameter_vjp_device
        procedure, public :: decision_function => &
            gp_multiclass_classification_decision_function
        procedure, public :: decision_function_jvp => &
            gp_multiclass_classification_decision_function_jvp
        procedure, public :: decision_function_vjp => &
            gp_multiclass_classification_decision_function_vjp
        procedure, public :: decision_function_parameter_jvp => &
            gp_multiclass_classification_decision_function_parameter_jvp
        procedure, public :: decision_function_parameter_vjp => &
            gp_multiclass_classification_decision_function_parameter_vjp
        procedure, public :: predict => gp_multiclass_classification_predict
        procedure, public :: predict_device => &
            gp_multiclass_classification_predict_device
        procedure, public :: classes => gp_multiclass_classification_classes
        procedure, public :: class_count => gp_multiclass_classification_class_count
        procedure, public :: feature_count => &
            gp_multiclass_classification_feature_count
        procedure, public :: parameter_count => &
            gp_multiclass_classification_parameter_count
        procedure, public :: parameters => gp_multiclass_classification_parameters
        procedure, public :: hyperparameter_gradient => &
            gp_multiclass_classification_hyperparameter_gradient
        procedure, public :: hyperparameter_hvp => &
            gp_multiclass_classification_hyperparameter_hvp
        procedure, public :: hyperparameter_hvp_device => &
            gp_multiclass_classification_hyperparameter_hvp_device
        procedure, public :: fitted => gp_multiclass_classification_fitted
        procedure, public :: device_supported => &
            gp_multiclass_classification_device_supported
    end type gp_multiclass_classification_t

    public :: gp_multiclass_classification_fit
    public :: gp_multiclass_classification_predict_proba
    public :: gp_multiclass_classification_predict_log_proba
    public :: gp_multiclass_classification_predict_log_proba_device
    public :: gp_multiclass_classification_predict_proba_device
    public :: gp_multiclass_classification_predict_proba_jvp
    public :: gp_multiclass_classification_predict_proba_vjp
    public :: gp_multiclass_classification_predict_log_proba_jvp
    public :: gp_multiclass_classification_predict_log_proba_vjp
    public :: gp_multiclass_classification_predict_proba_parameter_jvp
    public :: gp_multiclass_classification_predict_proba_parameter_vjp
    public :: gp_multiclass_classification_predict_log_proba_parameter_jvp
    public :: gp_multiclass_classification_predict_log_proba_parameter_vjp
    public :: gp_multiclass_classification_predict_proba_parameter_jvp_device
    public :: gp_multiclass_classification_predict_proba_parameter_vjp_device
    public :: gp_multiclass_classification_decision_function
    public :: gp_multiclass_classification_decision_function_jvp
    public :: gp_multiclass_classification_decision_function_vjp
    public :: gp_multiclass_classification_decision_function_parameter_jvp
    public :: gp_multiclass_classification_decision_function_parameter_vjp
    public :: gp_multiclass_classification_predict
    public :: gp_multiclass_classification_predict_device
    public :: gp_multiclass_classification_hyperparameter_hvp
    public :: gp_multiclass_classification_hyperparameter_hvp_device

contains

    subroutine gp_multiclass_classification_fit( &
            self, x, labels, kernel, status, options, state, sample_weight)
        class(gp_multiclass_classification_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), intent(in) :: kernel
        type(fortnum_status_t), intent(out) :: status
        type(gp_multiclass_classification_options_t), intent(in), optional :: options
        type(gp_multiclass_classification_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:)
        type(gp_multiclass_classification_options_t) :: requested
        type(gp_classification_options_t) :: binary_options
        type(gp_classification_state_t) :: binary_state
        type(gp_multiclass_classification_state_t) :: result
        integer, allocatable :: unique_labels(:), binary_labels(:)
        integer :: i, class_count
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(gp_multiclass_classification_options_t) :: gp_multiclass_classification_options_t_default
        type(gp_multiclass_classification_state_t) :: gp_multiclass_classification_state_t_default

        result = gp_multiclass_classification_state_t_default
        if (present(state)) state = result
        requested = gp_multiclass_classification_options_t_default
        if (present(options)) requested = options
        if (.not. valid_options(requested)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: options are invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: inputs must be finite")
            return
        end if
        call validate_sample_weights(size(x, 1), sample_weight, status)
        if (status%code /= FORTNUM_OK) return
        call sorted_unique_labels(labels, unique_labels)
        class_count = size(unique_labels)
        if (class_count < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: at least two classes are required")
            return
        end if
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: kernel dimension is invalid")
            return
        end if

        allocate(self%models(class_count), self%class_label(class_count))
        self%class_label = unique_labels
        self%n_classes = class_count
        self%n_features = size(x, 2)
        self%is_fitted = .false.
        allocate(binary_labels(size(labels)))
        binary_options%likelihood = requested%likelihood
        binary_options%max_iterations = requested%max_iterations
        binary_options%tolerance = requested%tolerance
        binary_options%jitter = requested%jitter
        binary_options%damping = requested%damping
        result%class_count = class_count
        result%log_posterior = 0.0_dp
        do i = 1, class_count
            binary_labels = 0
            where (labels == unique_labels(i)) binary_labels = 1
            if (present(sample_weight)) then
                call self%models(i)%fit(x, binary_labels, kernel, status, &
                    binary_options, binary_state, sample_weight=sample_weight)
            else
                call self%models(i)%fit(x, binary_labels, kernel, status, &
                    binary_options, binary_state)
            end if
            if (status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass classification fit: one-vs-rest model failed")
                return
            end if
            result%total_iterations = result%total_iterations + binary_state%iterations
            result%log_posterior = result%log_posterior + binary_state%log_posterior
        end do
        result%converged = .true.
        self%is_fitted = .true.
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_fit

    subroutine gp_multiclass_classification_predict_proba( &
            self, x, probabilities, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :)
        real(dp) :: normalization
        integer :: i, j

        if (.not. prediction_shapes(self, x, probabilities, status)) return
        allocate(binary_probabilities(size(x, 1), 2))
        probabilities = 0.0_dp
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(probabilities(i, :))
            if (.not. ieee_is_finite(normalization) .or. normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass classification prediction: invalid probability sum")
                return
            end if
            probabilities(i, :) = probabilities(i, :)/normalization
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_proba

    subroutine gp_multiclass_classification_predict_proba_device(self, device, x, &
            probabilities, status)
        !! Predict through the explicit multiclass device contract.
        !!
        !! The one-vs-rest wrapper owns one covariance/Laplace state per class;
        !! none of those states is resident on CUDA yet.  CPU dispatch is the
        !! exact reference path.  CUDA therefore returns a typed refusal and
        !! never stages an unaccounted host computation.
        class(gp_multiclass_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP multiclass device prediction: no resident CUDA Laplace kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass device prediction: device kind is invalid")
        end select
    end subroutine gp_multiclass_classification_predict_proba_device

    !> Return the natural logarithm of the normalized one-vs-rest
    !! probabilities.  The class columns retain ``classes()`` order and the
    !! finite floor mirrors the binary Laplace-GP contract at floating-point
    !! tails.
    subroutine gp_multiclass_classification_predict_log_proba(self, x, &
            log_probabilities, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. prediction_shapes(self, x, log_probabilities, status)) return
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        log_probabilities = log(max(probabilities, tiny(1.0_dp)))
        if (any(.not. ieee_is_finite(log_probabilities))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass log probability: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_log_proba

    !> Device-dispatched multiclass log-probability prediction.  CUDA remains
    !! a typed refusal until all OVR Laplace states and the normalization graph
    !! are resident; CPU dispatch is the exact reference path.
    subroutine gp_multiclass_classification_predict_log_proba_device(self, device, x, &
            log_probabilities, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass log probability device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_log_proba(x, log_probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP multiclass log probability device: no resident CUDA Laplace graph is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass log probability device: device kind is invalid")
        end select
    end subroutine gp_multiclass_classification_predict_log_proba_device

    !> Forward query-input product of multiclass log probabilities.
    subroutine gp_multiclass_classification_predict_log_proba_jvp(self, x, x_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)
        integer :: i, j

        if (.not. prediction_shapes(self, x, log_probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass log probability JVP: input or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            probabilities_dot(size(x, 1), self%n_classes))
        call self%predict_proba_jvp(x, x_dot, probabilities, probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                log_probabilities(i, j) = log(max(probabilities(i, j), tiny(1.0_dp)))
                if (probabilities(i, j) > tiny(1.0_dp)) then
                    log_probabilities_dot(i, j) = probabilities_dot(i, j) / &
                        probabilities(i, j)
                else
                    log_probabilities_dot(i, j) = 0.0_dp
                end if
            end do
        end do
        if (any(.not. ieee_is_finite(log_probabilities)) .or. &
            any(.not. ieee_is_finite(log_probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass log probability JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_log_proba_jvp

    !> Reverse query-input product of multiclass log probabilities.
    subroutine gp_multiclass_classification_predict_log_proba_vjp(self, x, &
            log_probabilities_bar, x_bar, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probability_bar(:, :)
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. prediction_shapes(self, x, log_probabilities_bar, status)) return
        if (any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass log probability VJP: input or cotangent is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            probability_bar(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                probability_bar(i, j) = log_probabilities_bar(i, j) / &
                    max(probabilities(i, j), tiny(1.0_dp))
            end do
        end do
        if (any(.not. ieee_is_finite(probability_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass log probability VJP: cotangent is not finite")
            return
        end if
        call self%predict_proba_vjp(x, probability_bar, x_bar, status)
    end subroutine gp_multiclass_classification_predict_log_proba_vjp

    !> Forward fixed-state kernel-parameter product of multiclass log
    !! probabilities.  The packed per-class direction uses the same layout as
    !! ``predict_proba_parameter_jvp``.
    subroutine gp_multiclass_classification_predict_log_proba_parameter_jvp(self, x, &
            direction, log_probabilities, log_probabilities_dot, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)
        integer :: i, j

        if (.not. prediction_shapes(self, x, log_probabilities, status)) return
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass log probability parameter JVP: direction or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            probabilities_dot(size(x, 1), self%n_classes))
        call self%predict_proba_parameter_jvp(x, direction, probabilities, &
            probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                log_probabilities(i, j) = log(max(probabilities(i, j), tiny(1.0_dp)))
                if (probabilities(i, j) > tiny(1.0_dp)) then
                    log_probabilities_dot(i, j) = probabilities_dot(i, j) / &
                        probabilities(i, j)
                else
                    log_probabilities_dot(i, j) = 0.0_dp
                end if
            end do
        end do
        if (any(.not. ieee_is_finite(log_probabilities)) .or. &
            any(.not. ieee_is_finite(log_probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass log probability parameter JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_log_proba_parameter_jvp

    !> Reverse fixed-state kernel-parameter product of multiclass log
    !! probabilities.
    subroutine gp_multiclass_classification_predict_log_proba_parameter_vjp(self, x, &
            log_probabilities_bar, parameter_bar, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probability_bar(:, :)
        integer :: i, j

        parameter_bar = 0.0_dp
        if (.not. prediction_shapes(self, x, log_probabilities_bar, status)) return
        if (size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass log probability parameter VJP: input or cotangent is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            probability_bar(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                probability_bar(i, j) = log_probabilities_bar(i, j) / &
                    max(probabilities(i, j), tiny(1.0_dp))
            end do
        end do
        if (any(.not. ieee_is_finite(probability_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass log probability parameter VJP: cotangent is not finite")
            return
        end if
        call self%predict_proba_parameter_vjp(x, probability_bar, parameter_bar, status)
    end subroutine gp_multiclass_classification_predict_log_proba_parameter_vjp

    subroutine gp_multiclass_classification_predict_proba_jvp( &
            self, x, x_dot, probabilities, probabilities_dot, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), binary_probability_dot(:, :)
        real(dp), allocatable :: raw(:, :), raw_dot(:, :)
        real(dp) :: normalization, normalization_dot
        integer :: i, j

        if (.not. prediction_shapes(self, x, probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification probability JVP: shape is invalid")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2))
        allocate(binary_probability_dot(size(x, 1), 2))
        allocate(raw(size(x, 1), self%n_classes), raw_dot(size(x, 1), self%n_classes))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba_jvp(x, x_dot, binary_probabilities, &
                binary_probability_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
            raw_dot(:, j) = binary_probability_dot(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(raw(i, :))
            normalization_dot = sum(raw_dot(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass classification probability JVP: invalid sum")
                return
            end if
            probabilities(i, :) = raw(i, :)/normalization
            probabilities_dot(i, :) = (raw_dot(i, :)*normalization - &
                raw(i, :)*normalization_dot)/(normalization*normalization)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_proba_jvp

    !> Reverse-mode product of normalized multiclass probabilities with
    !> respect to query features.
    subroutine gp_multiclass_classification_predict_proba_vjp(self, x, &
            probabilities_bar, x_bar, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: raw(:, :), raw_bar(:, :), binary_bar(:, :), &
            binary_x_bar(:, :)
        real(dp), allocatable :: totals(:), raw_cotangent(:)
        real(dp) :: weighted
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(probabilities_bar) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass probability VJP: input or cotangent is invalid")
            return
        end if
        allocate(raw(size(x, 1), self%n_classes), raw_bar(size(x, 1), self%n_classes), &
            totals(size(x, 1)), binary_bar(size(x, 1), 2), &
            binary_x_bar(size(x, 1), size(x, 2)), raw_cotangent(self%n_classes))
        call self%predict_proba(x, raw, status)
        if (status%code /= FORTNUM_OK) return
        ! Recover the positive one-vs-rest probabilities before normalization.
        do i = 1, self%n_classes
            call self%models(i)%predict_proba(x, binary_bar, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, i) = binary_bar(:, 2)
        end do
        totals = sum(raw, dim=2)
        do j = 1, size(x, 1)
            weighted = dot_product(probabilities_bar(j, :), raw(j, :))
            raw_cotangent = probabilities_bar(j, :)/totals(j) - weighted/ &
                (totals(j)*totals(j))
            raw_bar(j, :) = raw_cotangent
        end do
        do i = 1, self%n_classes
            binary_bar = 0.0_dp
            binary_bar(:, 2) = raw_bar(:, i)
            call self%models(i)%predict_proba_vjp(x, binary_bar, binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + binary_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_proba_vjp

    !> Forward product of normalized one-vs-rest probabilities with respect
    !! to the packed per-class kernel log parameters.  Each binary Laplace
    !! state is held fixed, matching the binary fixed-state product contract.
    subroutine gp_multiclass_classification_predict_proba_parameter_jvp(self, x, &
            direction, probabilities, probabilities_dot, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: raw(:, :), raw_dot(:, :), local(:, :), local_dot(:, :)
        real(dp) :: total, total_dot
        integer :: i, j, first, last, local_count

        if (.not. prediction_shapes(self, x, probabilities, status)) return
        if (any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass probability parameter JVP: shape is invalid")
            return
        end if
        allocate(raw(size(x, 1), self%n_classes), &
            raw_dot(size(x, 1), self%n_classes), local(size(x, 1), 2), &
            local_dot(size(x, 1), 2))
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            last = first + local_count - 1
            call self%models(i)%predict_proba_parameter_jvp(x, direction(first:last), &
                local, local_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, i) = local(:, 2)
            raw_dot(:, i) = local_dot(:, 2)
            first = last + 1
        end do
        do j = 1, size(x, 1)
            total = sum(raw(j, :))
            total_dot = sum(raw_dot(j, :))
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass probability parameter JVP: invalid normalization")
                return
            end if
            probabilities(j, :) = raw(j, :)/total
            probabilities_dot(j, :) = (raw_dot(j, :)*total - &
                raw(j, :)*total_dot)/(total*total)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_proba_parameter_jvp

    !> Reverse product of normalized one-vs-rest probabilities with respect
    !! to the packed per-class kernel log parameters.  The simplex adjoint is
    !! applied before dispatching each class's binary fixed-state VJP.
    subroutine gp_multiclass_classification_predict_proba_parameter_vjp(self, x, &
            probabilities_bar, parameter_bar, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: raw(:, :), raw_bar(:, :), local(:, :)
        real(dp), allocatable :: local_parameter_bar(:)
        real(dp) :: total, projection
        integer :: i, j, first, last, local_count

        parameter_bar = 0.0_dp
        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(probabilities_bar) /= [size(x, 1), self%n_classes]) .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass probability parameter VJP: input is invalid")
            return
        end if
        allocate(raw(size(x, 1), self%n_classes), &
            raw_bar(size(x, 1), self%n_classes), local(size(x, 1), 2))
        do i = 1, self%n_classes
            call self%models(i)%predict_proba(x, local, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, i) = local(:, 2)
        end do
        do j = 1, size(x, 1)
            total = sum(raw(j, :))
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass probability parameter VJP: invalid normalization")
                return
            end if
            projection = sum(probabilities_bar(j, :)*raw(j, :))/total
            raw_bar(j, :) = (probabilities_bar(j, :) - projection)/total
        end do
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            last = first + local_count - 1
            local = 0.0_dp
            local(:, 2) = raw_bar(:, i)
            allocate(local_parameter_bar(local_count))
            call self%models(i)%predict_proba_parameter_vjp(x, local, &
                local_parameter_bar, status)
            if (status%code /= FORTNUM_OK) return
            parameter_bar(first:last) = local_parameter_bar
            deallocate(local_parameter_bar)
            first = last + 1
        end do
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass probability parameter VJP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_proba_parameter_vjp

    !> Dispatch the packed probability parameter JVP through the explicit
    !! device contract.  CUDA remains refused until the OVR Laplace states
    !! and normalization reduction are resident.
    subroutine gp_multiclass_classification_predict_proba_parameter_jvp_device(self, &
            device, x, direction, probabilities, probabilities_dot, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass probability parameter JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_jvp(x, direction, probabilities, &
                probabilities_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP multiclass probability parameter JVP device: no resident CUDA Laplace graph is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass probability parameter JVP device: device kind is invalid")
        end select
    end subroutine gp_multiclass_classification_predict_proba_parameter_jvp_device

    !> Dispatch the packed probability parameter VJP through the explicit
    !! device contract.  CUDA remains refused until the OVR reverse graph is
    !! resident; no host fallback is implied.
    subroutine gp_multiclass_classification_predict_proba_parameter_vjp_device(self, &
            device, x, probabilities_bar, parameter_bar, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass probability parameter VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_vjp(x, probabilities_bar, parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP multiclass probability parameter VJP device: no resident CUDA Laplace graph is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass probability parameter VJP device: device kind is invalid")
        end select
    end subroutine gp_multiclass_classification_predict_proba_parameter_vjp_device

    !> Return the one-vs-rest latent posterior means for every class.
    !>
    !> The columns follow ``classes()`` and are the binary GP latent means
    !> before the probability-simplex normalization.  This is the multiclass
    !> analogue of a classifier ``decision_function`` and is useful for
    !> margins, calibration, and downstream differentiable composition.
    subroutine gp_multiclass_classification_decision_function(self, x, margins, &
            status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        integer :: i

        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(margins) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass decision_function: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), variance(size(x, 1)))
        do i = 1, self%n_classes
            call self%models(i)%predict_latent(x, mean, variance, status)
            if (status%code /= FORTNUM_OK) return
            margins(:, i) = mean
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_decision_function

    !> JVP of ``decision_function`` with respect to the query features.
    subroutine gp_multiclass_classification_decision_function_jvp(self, x, x_dot, &
            margins, margins_dot, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: margins(:, :), margins_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        integer :: i

        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass decision_function JVP: input tangent is invalid")
            return
        end if
        if (any(shape(margins) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(margins_dot) /= shape(margins))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass decision_function JVP: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), &
            variance(size(x, 1)), variance_dot(size(x, 1)))
        do i = 1, self%n_classes
            call self%models(i)%predict_latent_jvp(x, x_dot, mean, mean_dot, &
                variance, variance_dot, status)
            if (status%code /= FORTNUM_OK) return
            margins(:, i) = mean
            margins_dot(:, i) = mean_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_decision_function_jvp

    !> Reverse-mode product of the one-vs-rest latent margins with respect to
    !> query features.
    subroutine gp_multiclass_classification_decision_function_vjp(self, x, &
            margins_bar, x_bar, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), margins_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean_bar(:), variance_bar(:), binary_x_bar(:, :)
        integer :: i

        x_bar = 0.0_dp
        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(margins_bar) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(margins_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass decision_function VJP: input or cotangent is invalid")
            return
        end if
        allocate(mean_bar(size(x, 1)), variance_bar(size(x, 1)), &
            binary_x_bar(size(x, 1), size(x, 2)))
        variance_bar = 0.0_dp
        do i = 1, self%n_classes
            mean_bar = margins_bar(:, i)
            call self%models(i)%predict_latent_vjp(x, mean_bar, variance_bar, &
                binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + binary_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_decision_function_vjp

    !> Forward product of the one-vs-rest latent margins with respect to the
    !! packed per-class kernel log parameters under the binary fixed-state
    !! contract.
    subroutine gp_multiclass_classification_decision_function_parameter_jvp(self, x, &
            direction, margins, margins_dot, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: margins(:, :), margins_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        integer :: i, first, last, local_count

        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(margins) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(margins_dot) /= shape(margins)) .or. &
            size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass decision parameter JVP: shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), &
            variance(size(x, 1)), variance_dot(size(x, 1)))
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            last = first + local_count - 1
            call self%models(i)%predict_latent_parameter_jvp(x, direction(first:last), &
                mean, mean_dot, variance, variance_dot, status)
            if (status%code /= FORTNUM_OK) return
            margins(:, i) = mean
            margins_dot(:, i) = mean_dot
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_decision_function_parameter_jvp

    !> Reverse product of one-vs-rest latent margins with respect to the
    !! packed per-class kernel log parameters under the binary fixed-state
    !! contract.
    subroutine gp_multiclass_classification_decision_function_parameter_vjp(self, x, &
            margins_bar, parameter_bar, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), margins_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: variance_bar(:), local_parameter_bar(:)
        integer :: i, first, last, local_count

        parameter_bar = 0.0_dp
        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(margins_bar) /= [size(x, 1), self%n_classes]) .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(margins_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass decision parameter VJP: input is invalid")
            return
        end if
        allocate(variance_bar(size(x, 1)))
        variance_bar = 0.0_dp
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            last = first + local_count - 1
            allocate(local_parameter_bar(local_count))
            call self%models(i)%predict_latent_parameter_vjp(x, margins_bar(:, i), &
                variance_bar, local_parameter_bar, status)
            if (status%code /= FORTNUM_OK) return
            parameter_bar(first:last) = local_parameter_bar
            deallocate(local_parameter_bar)
            first = last + 1
        end do
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass decision parameter VJP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_decision_function_parameter_vjp

    subroutine gp_multiclass_classification_predict(self, x, labels, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. prediction_input_valid(self, x, status)) return
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification prediction: label shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict

    subroutine gp_multiclass_classification_predict_device(self, device, x, labels, status)
        !! Predict labels through the explicit multiclass device contract.
        class(gp_multiclass_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass device label prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP multiclass device label prediction: no resident CUDA Laplace kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass device label prediction: device kind is invalid")
        end select
    end subroutine gp_multiclass_classification_predict_device

    function gp_multiclass_classification_classes(self) result(classes)
        class(gp_multiclass_classification_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function gp_multiclass_classification_classes

    integer function gp_multiclass_classification_class_count(self) result(count)
        class(gp_multiclass_classification_t), intent(in) :: self

        count = self%n_classes
    end function gp_multiclass_classification_class_count

    integer function gp_multiclass_classification_feature_count(self) result(count)
        class(gp_multiclass_classification_t), intent(in) :: self

        count = self%n_features
    end function gp_multiclass_classification_feature_count

    integer function gp_multiclass_classification_parameter_count(self) result(count)
        class(gp_multiclass_classification_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. self%fitted()) return
        do i = 1, self%n_classes
            count = count + self%models(i)%parameter_count()
        end do
    end function gp_multiclass_classification_parameter_count

    function gp_multiclass_classification_parameters(self) result(parameters)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        real(dp), allocatable :: model_parameters(:)
        integer :: i, first, last

        if (.not. self%fitted()) then
            allocate(parameters(0))
            return
        end if
        allocate(parameters(self%parameter_count()))
        first = 1
        do i = 1, self%n_classes
            model_parameters = self%models(i)%parameters()
            last = first + size(model_parameters) - 1
            parameters(first:last) = model_parameters
            first = last + 1
        end do
    end function gp_multiclass_classification_parameters

    subroutine gp_multiclass_classification_hyperparameter_gradient(self, gradient, &
            status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_gradient(:)
        integer :: i, first, last, local_count

        gradient = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass hyperparameter gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass hyperparameter gradient: output shape is invalid")
            return
        end if
        ! The one-vs-rest state objective is the sum of the independent
        ! binary mode log posteriors.  Pack each exact binary envelope
        ! gradient in the same sorted-class order as parameters().
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            allocate(local_gradient(local_count))
            call self%models(i)%hyperparameter_gradient(local_gradient, status)
            if (status%code /= FORTNUM_OK) return
            last = first + local_count - 1
            gradient(first:last) = local_gradient
            first = last + 1
            deallocate(local_gradient)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_hyperparameter_gradient

    subroutine gp_multiclass_classification_hyperparameter_hvp(self, direction, &
            parameter_hvp, status)
        !! Directional HVP of the sum of the independent binary Laplace
        !! envelope objectives.  Each class owns one contiguous kernel/noise
        !! block, so the multiclass product is a transactional block dispatch
        !! with no hidden cross-class coupling.
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_hvp(:)
        integer :: i, first, last, local_count

        parameter_hvp = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass hyperparameter HVP: model is not fitted")
            return
        end if
        if (size(direction) /= self%parameter_count() .or. &
            size(parameter_hvp) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass hyperparameter HVP: parameter shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            last = first + local_count - 1
            allocate(local_hvp(local_count))
            call self%models(i)%hyperparameter_hvp(direction(first:last), &
                local_hvp, status)
            if (status%code /= FORTNUM_OK) return
            parameter_hvp(first:last) = local_hvp
            deallocate(local_hvp)
            first = last + 1
        end do
        if (any(.not. ieee_is_finite(parameter_hvp))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass hyperparameter HVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_hyperparameter_hvp

    subroutine gp_multiclass_classification_hyperparameter_hvp_device(self, device, &
            direction, parameter_hvp, status)
        !! Explicit device boundary; the binary Laplace HVP is CPU-only until
        !! a resident multiclass factorization and derivative graph is linked.
        class(gp_multiclass_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_hvp = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass hyperparameter HVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hyperparameter_hvp(direction, parameter_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP multiclass hyperparameter HVP device: no resident CUDA Laplace graph is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass hyperparameter HVP device: device kind is invalid")
        end select
    end subroutine gp_multiclass_classification_hyperparameter_hvp_device

    logical function gp_multiclass_classification_fitted(self) result(fitted)
        class(gp_multiclass_classification_t), intent(in) :: self

        fitted = self%is_fitted .and. allocated(self%models) .and. &
            allocated(self%class_label) .and. self%n_classes >= 2
    end function gp_multiclass_classification_fitted

    logical function gp_multiclass_classification_device_supported(self, device_kind) &
            result(supported)
        !! Report capability without inferring a host fallback for CUDA.
        class(gp_multiclass_classification_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%fitted()
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function gp_multiclass_classification_device_supported

    logical function prediction_shapes(self, x, probabilities, status) result(valid)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification prediction: probability shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_shapes

    logical function prediction_input_valid(self, x, status) result(valid)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification prediction: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification prediction: input shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_input_valid

    subroutine sorted_unique_labels(labels, unique_labels)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: unique_labels(:)
        integer, allocatable :: work(:)
        integer :: i, j, count, candidate
        logical :: found

        allocate(work(size(labels)))
        count = 0
        do i = 1, size(labels)
            candidate = labels(i)
            found = .false.
            do j = 1, count
                if (work(j) == candidate) then
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) then
                count = count + 1
                work(count) = candidate
            end if
        end do
        do i = 1, count - 1
            do j = i + 1, count
                if (work(j) < work(i)) then
                    candidate = work(i)
                    work(i) = work(j)
                    work(j) = candidate
                end if
            end do
        end do
        allocate(unique_labels(count))
        unique_labels = work(:count)
    end subroutine sorted_unique_labels

    subroutine validate_sample_weights(n_samples, sample_weight, status)
        integer, intent(in) :: n_samples
        real(dp), intent(in), optional :: sample_weight(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: weight_mass

        if (.not. present(sample_weight)) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (size(sample_weight) /= n_samples .or. &
            any(.not. ieee_is_finite(sample_weight)) .or. &
            any(sample_weight < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: sample weights are invalid")
            return
        end if
        weight_mass = sum(sample_weight)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: sample weights need positive mass")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_sample_weights

    logical function valid_options(options) result(valid)
        type(gp_multiclass_classification_options_t), intent(in) :: options

        valid = (options%likelihood == GP_LIKELIHOOD_LOGISTIC .or. &
            options%likelihood == GP_LIKELIHOOD_PROBIT) .and. &
            options%max_iterations >= 1 .and. &
            ieee_is_finite(options%tolerance) .and. options%tolerance > 0.0_dp .and. &
            ieee_is_finite(options%jitter) .and. options%jitter >= 0.0_dp .and. &
            ieee_is_finite(options%damping) .and. options%damping > 0.0_dp .and. &
            options%damping <= 1.0_dp
    end function valid_options

end module fortml_gp_multiclass_classification
