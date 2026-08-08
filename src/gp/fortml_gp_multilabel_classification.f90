module fortml_gp_multilabel_classification
    !! Independent binary Laplace-GP heads for multilabel indicators.
    !!
    !! A multilabel target is a dense ``(n_samples,n_labels)`` indicator
    !! matrix.  Each column owns a binary Laplace GP with the same kernel
    !! contract; unlike the multiclass OVR wrapper, probabilities are not
    !! normalized across labels.  Packed kernel parameters are concatenated
    !! in label order and all prediction products hold the fitted Laplace
    !! states fixed, matching ``gp_classification_t``'s explicit derivative
    !! contract.  CUDA requests are refused until all heads and reductions
    !! have resident kernels; CPU dispatch never hides a host fallback.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t
    implicit none
    private

    type, public :: gp_multilabel_classification_options_t
        integer :: likelihood = 1
        integer :: max_iterations = 50
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: jitter = 1.0e-8_dp
        real(dp) :: damping = 1.0_dp
    end type gp_multilabel_classification_options_t

    type, public :: gp_multilabel_classification_state_t
        integer :: label_count = 0
        integer :: total_iterations = 0
        real(dp) :: log_posterior = -huge(1.0_dp)
        logical :: converged = .false.
    end type gp_multilabel_classification_state_t

    type, public :: gp_multilabel_classification_t
        private
        type(gp_classification_t), allocatable :: models(:)
        real(dp), allocatable :: decision_threshold(:)
        integer :: n_labels = 0
        integer :: n_features = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => gp_multilabel_fit
        procedure, public :: predict_latent => gp_multilabel_predict_latent
        procedure, public :: predict_latent_jvp => gp_multilabel_predict_latent_jvp
        procedure, public :: predict_latent_vjp => gp_multilabel_predict_latent_vjp
        procedure, public :: predict_latent_parameter_jvp => &
            gp_multilabel_predict_latent_parameter_jvp
        procedure, public :: predict_latent_parameter_vjp => &
            gp_multilabel_predict_latent_parameter_vjp
        procedure, public :: predict_proba => gp_multilabel_predict_proba
        procedure, public :: predict_proba_device => &
            gp_multilabel_predict_proba_device
        procedure, public :: predict_proba_jvp => gp_multilabel_predict_proba_jvp
        procedure, public :: predict_proba_vjp => gp_multilabel_predict_proba_vjp
        procedure, public :: predict_proba_parameter_jvp => &
            gp_multilabel_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            gp_multilabel_predict_proba_parameter_vjp
        procedure, public :: predict => gp_multilabel_predict
        procedure, public :: predict_device => gp_multilabel_predict_device
        procedure, public :: parameter_count => gp_multilabel_parameter_count
        procedure, public :: parameters => gp_multilabel_parameters
        procedure, public :: hyperparameter_gradient => &
            gp_multilabel_hyperparameter_gradient
        procedure, public :: label_count => gp_multilabel_label_count
        procedure, public :: feature_count => gp_multilabel_feature_count
        procedure, public :: thresholds => gp_multilabel_thresholds
        procedure, public :: set_thresholds => gp_multilabel_set_thresholds
        procedure, public :: fitted => gp_multilabel_fitted
        procedure, public :: device_supported => gp_multilabel_device_supported
    end type gp_multilabel_classification_t

    public :: gp_multilabel_fit
    public :: gp_multilabel_predict_latent
    public :: gp_multilabel_predict_latent_jvp
    public :: gp_multilabel_predict_latent_vjp
    public :: gp_multilabel_predict_latent_parameter_jvp
    public :: gp_multilabel_predict_latent_parameter_vjp
    public :: gp_multilabel_predict_proba
    public :: gp_multilabel_predict_proba_device
    public :: gp_multilabel_predict_proba_jvp
    public :: gp_multilabel_predict_proba_vjp
    public :: gp_multilabel_predict_proba_parameter_jvp
    public :: gp_multilabel_predict_proba_parameter_vjp
    public :: gp_multilabel_predict
    public :: gp_multilabel_predict_device

contains

    subroutine gp_multilabel_fit(self, x, indicators, kernel, status, options, state, &
            sample_weight, thresholds)
        class(gp_multilabel_classification_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: indicators(:, :)
        type(kernel_t), intent(in) :: kernel
        type(fortnum_status_t), intent(out) :: status
        type(gp_multilabel_classification_options_t), intent(in), optional :: options
        type(gp_multilabel_classification_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:), thresholds(:)
        type(gp_multilabel_classification_options_t) :: requested
        type(gp_multilabel_classification_state_t) :: result
        type(gp_classification_options_t) :: binary_options
        type(gp_classification_state_t) :: binary_state
        integer, allocatable :: labels(:)
        real(dp), allocatable :: weights(:)
        integer :: i, n_samples, n_labels
        type(gp_multilabel_classification_options_t) :: options_default
        type(gp_multilabel_classification_state_t) :: state_default

        self%is_fitted = .false.
        self%n_labels = 0
        self%n_features = 0
        if (allocated(self%models)) deallocate(self%models)
        if (allocated(self%decision_threshold)) deallocate(self%decision_threshold)
        requested = options_default
        if (present(options)) requested = options
        result = state_default
        if (present(state)) state = result
        n_samples = size(x, 1)
        n_labels = size(indicators, 2)
        if (n_samples < 1 .or. size(x, 2) < 1 .or. n_labels < 1 .or. &
            size(indicators, 1) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any((indicators /= 0) .and. &
            (indicators /= 1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP fit: inputs must be finite indicators")
            return
        end if
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP fit: kernel dimension is invalid")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp) .or. sum(sample_weight) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multilabel GP fit: sample weights must be finite, nonnegative, and positive")
                return
            end if
            allocate(weights, source=sample_weight)
        else
            allocate(weights(n_samples)); weights = 1.0_dp
        end if
        allocate(self%models(n_labels), self%decision_threshold(n_labels), labels(n_samples))
        self%decision_threshold = 0.5_dp
        if (present(thresholds)) then
            if (size(thresholds) /= n_labels .or. any(.not. ieee_is_finite(thresholds)) .or. &
                any(thresholds <= 0.0_dp) .or. any(thresholds >= 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multilabel GP fit: thresholds must be finite in (0,1)")
                return
            end if
            self%decision_threshold = thresholds
        end if
        binary_options%likelihood = requested%likelihood
        binary_options%max_iterations = requested%max_iterations
        binary_options%tolerance = requested%tolerance
        binary_options%jitter = requested%jitter
        binary_options%damping = requested%damping
        self%n_features = size(x, 2)
        self%n_labels = n_labels
        result%label_count = n_labels
        result%log_posterior = 0.0_dp
        do i = 1, n_labels
            if (minval(indicators(:, i)) == maxval(indicators(:, i))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multilabel GP fit: every label needs both classes")
                return
            end if
            labels = indicators(:, i)
            if (present(sample_weight)) then
                call self%models(i)%fit(x, labels, kernel, status, binary_options, &
                    binary_state, sample_weight=weights)
            else
                call self%models(i)%fit(x, labels, kernel, status, binary_options, &
                    binary_state)
            end if
            if (status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multilabel GP fit: binary Laplace head failed")
                return
            end if
            result%total_iterations = result%total_iterations + binary_state%iterations
            result%log_posterior = result%log_posterior + binary_state%log_posterior
        end do
        result%converged = .true.
        self%is_fitted = .true.
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_fit

    subroutine gp_multilabel_predict_latent(self, x, mean, variance, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:, :), variance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_mean(:), local_variance(:)
        integer :: i
        if (.not. valid_query(self, x, mean, status)) return
        if (any(shape(variance) /= shape(mean))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP latent prediction: output shape is invalid")
            return
        end if
        allocate(local_mean(size(x, 1)), local_variance(size(x, 1)))
        do i = 1, self%n_labels
            call self%models(i)%predict_latent(x, local_mean, local_variance, status)
            if (status%code /= FORTNUM_OK) return
            mean(:, i) = local_mean
            variance(:, i) = local_variance
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_latent

    subroutine gp_multilabel_predict_proba(self, x, probabilities, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local(:, :)
        integer :: i
        if (.not. valid_query(self, x, probabilities, status)) return
        allocate(local(size(x, 1), 2))
        do i = 1, self%n_labels
            call self%models(i)%predict_proba(x, local, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i) = local(:, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_proba

    subroutine gp_multilabel_predict_proba_device(self, device, x, probabilities, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP probability device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multilabel GP probability device: no resident CUDA Laplace heads are linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP probability device: device kind is invalid")
        end select
    end subroutine gp_multilabel_predict_proba_device

    subroutine gp_multilabel_predict_latent_jvp(self, x, x_dot, mean, mean_dot, &
            variance, variance_dot, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :), variance(:, :), variance_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_mean(:), local_mean_dot(:), local_variance(:), local_variance_dot(:)
        integer :: i
        if (.not. valid_query(self, x, mean, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. any(shape(mean_dot) /= shape(mean)) .or. &
            any(shape(variance) /= shape(mean)) .or. any(shape(variance_dot) /= shape(mean))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP latent JVP: input or output shape is invalid")
            return
        end if
        allocate(local_mean(size(x, 1)), local_mean_dot(size(x, 1)), &
            local_variance(size(x, 1)), local_variance_dot(size(x, 1)))
        do i = 1, self%n_labels
            call self%models(i)%predict_latent_jvp(x, x_dot, local_mean, local_mean_dot, &
                local_variance, local_variance_dot, status)
            if (status%code /= FORTNUM_OK) return
            mean(:, i) = local_mean; mean_dot(:, i) = local_mean_dot
            variance(:, i) = local_variance; variance_dot(:, i) = local_variance_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_latent_jvp

    subroutine gp_multilabel_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local(:, :), local_dot(:, :)
        integer :: i
        if (.not. valid_query(self, x, probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP probability JVP: input or output shape is invalid")
            return
        end if
        allocate(local(size(x, 1), 2), local_dot(size(x, 1), 2))
        do i = 1, self%n_labels
            call self%models(i)%predict_proba_jvp(x, x_dot, local, local_dot, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i) = local(:, 2); probabilities_dot(:, i) = local_dot(:, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_proba_jvp

    subroutine gp_multilabel_predict_latent_vjp(self, x, mean_bar, variance_bar, x_bar, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:, :), variance_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_x_bar(:, :)
        integer :: i
        x_bar = 0.0_dp
        if (.not. valid_query(self, x, mean_bar, status)) return
        if (any(shape(variance_bar) /= shape(mean_bar)) .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP latent VJP: cotangent shape is invalid")
            return
        end if
        allocate(binary_x_bar(size(x, 1), size(x, 2)))
        do i = 1, self%n_labels
            call self%models(i)%predict_latent_vjp(x, mean_bar(:, i), variance_bar(:, i), &
                binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + binary_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_latent_vjp

    subroutine gp_multilabel_predict_proba_vjp(self, x, probabilities_bar, x_bar, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_bar(:, :), binary_x_bar(:, :)
        integer :: i
        x_bar = 0.0_dp
        if (.not. valid_query(self, x, probabilities_bar, status)) return
        if (any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP probability VJP: cotangent shape is invalid")
            return
        end if
        allocate(binary_bar(size(x, 1), 2), binary_x_bar(size(x, 1), size(x, 2)))
        binary_bar(:, 1) = 0.0_dp
        do i = 1, self%n_labels
            binary_bar(:, 2) = probabilities_bar(:, i)
            call self%models(i)%predict_proba_vjp(x, binary_bar, binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + binary_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_proba_vjp

    subroutine gp_multilabel_predict_latent_parameter_jvp(self, x, direction, mean, &
            mean_dot, variance, variance_dot, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :), variance(:, :), variance_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_mean(:), local_mean_dot(:), local_variance(:), local_variance_dot(:)
        integer :: i, first, last, local_count
        if (.not. valid_query(self, x, mean, status)) return
        if (any(shape(mean_dot) /= shape(mean)) .or. any(shape(variance) /= shape(mean)) .or. &
            any(shape(variance_dot) /= shape(mean)) .or. size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP latent parameter JVP: shape is invalid")
            return
        end if
        allocate(local_mean(size(x, 1)), local_mean_dot(size(x, 1)), &
            local_variance(size(x, 1)), local_variance_dot(size(x, 1)))
        first = 1
        do i = 1, self%n_labels
            local_count = self%models(i)%parameter_count(); last = first + local_count - 1
            call self%models(i)%predict_latent_parameter_jvp(x, direction(first:last), &
                local_mean, local_mean_dot, local_variance, local_variance_dot, status)
            if (status%code /= FORTNUM_OK) return
            mean(:, i) = local_mean; mean_dot(:, i) = local_mean_dot
            variance(:, i) = local_variance; variance_dot(:, i) = local_variance_dot
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_latent_parameter_jvp

    subroutine gp_multilabel_predict_proba_parameter_jvp(self, x, direction, probabilities, &
            probabilities_dot, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local(:, :), local_dot(:, :)
        integer :: i, first, last, local_count
        if (.not. valid_query(self, x, probabilities, status)) return
        if (any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP probability parameter JVP: shape is invalid")
            return
        end if
        allocate(local(size(x, 1), 2), local_dot(size(x, 1), 2))
        first = 1
        do i = 1, self%n_labels
            local_count = self%models(i)%parameter_count(); last = first + local_count - 1
            call self%models(i)%predict_proba_parameter_jvp(x, direction(first:last), &
                local, local_dot, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i) = local(:, 2); probabilities_dot(:, i) = local_dot(:, 2)
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_proba_parameter_jvp

    subroutine gp_multilabel_predict_latent_parameter_vjp(self, x, mean_bar, variance_bar, &
            parameter_bar, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:, :), variance_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_bar(:)
        integer :: i, first, last, local_count
        parameter_bar = 0.0_dp
        if (.not. valid_query(self, x, mean_bar, status)) return
        if (any(shape(variance_bar) /= shape(mean_bar)) .or. size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP latent parameter VJP: shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_labels
            local_count = self%models(i)%parameter_count(); last = first + local_count - 1
            allocate(local_bar(local_count))
            call self%models(i)%predict_latent_parameter_vjp(x, mean_bar(:, i), variance_bar(:, i), &
                local_bar, status)
            if (status%code /= FORTNUM_OK) return
            parameter_bar(first:last) = local_bar
            deallocate(local_bar); first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_latent_parameter_vjp

    subroutine gp_multilabel_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            parameter_bar, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_bar(:, :), local_bar(:)
        integer :: i, first, last, local_count
        parameter_bar = 0.0_dp
        if (.not. valid_query(self, x, probabilities_bar, status)) return
        if (size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP probability parameter VJP: shape is invalid")
            return
        end if
        allocate(binary_bar(size(x, 1), 2)); binary_bar(:, 1) = 0.0_dp
        first = 1
        do i = 1, self%n_labels
            local_count = self%models(i)%parameter_count(); last = first + local_count - 1
            binary_bar(:, 2) = probabilities_bar(:, i)
            allocate(local_bar(local_count))
            call self%models(i)%predict_proba_parameter_vjp(x, binary_bar, local_bar, status)
            if (status%code /= FORTNUM_OK) return
            parameter_bar(first:last) = local_bar
            deallocate(local_bar); first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_proba_parameter_vjp

    subroutine gp_multilabel_predict(self, x, indicators, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i
        if (.not. valid_integer_query(self, x, indicators, status)) return
        allocate(probabilities(size(x, 1), self%n_labels))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%n_labels
            indicators(:, i) = 0
            where (probabilities(:, i) > self%decision_threshold(i)) indicators(:, i) = 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict

    subroutine gp_multilabel_predict_device(self, device, x, indicators, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP label device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, indicators, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multilabel GP label device: no resident CUDA Laplace heads are linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP label device: device kind is invalid")
        end select
    end subroutine gp_multilabel_predict_device

    integer function gp_multilabel_parameter_count(self) result(count)
        class(gp_multilabel_classification_t), intent(in) :: self
        integer :: i
        count = 0
        if (.not. self%fitted()) return
        do i = 1, self%n_labels
            count = count + self%models(i)%parameter_count()
        end do
    end function gp_multilabel_parameter_count

    function gp_multilabel_parameters(self) result(parameters)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:), local(:)
        integer :: i, first, last
        if (.not. self%fitted()) then
            allocate(parameters(0)); return
        end if
        allocate(parameters(self%parameter_count())); first = 1
        do i = 1, self%n_labels
            local = self%models(i)%parameters(); last = first + size(local) - 1
            parameters(first:last) = local; first = last + 1
        end do
    end function gp_multilabel_parameters

    subroutine gp_multilabel_hyperparameter_gradient(self, gradient, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local(:)
        integer :: i, first, last, local_count
        gradient = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP hyperparameter gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP hyperparameter gradient: output shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_labels
            local_count = self%models(i)%parameter_count(); last = first + local_count - 1
            allocate(local(local_count))
            call self%models(i)%hyperparameter_gradient(local, status)
            if (status%code /= FORTNUM_OK) return
            gradient(first:last) = local; deallocate(local); first = last + 1
        end do
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP hyperparameter gradient: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_hyperparameter_gradient

    integer function gp_multilabel_label_count(self) result(count)
        class(gp_multilabel_classification_t), intent(in) :: self
        count = self%n_labels
    end function gp_multilabel_label_count

    integer function gp_multilabel_feature_count(self) result(count)
        class(gp_multilabel_classification_t), intent(in) :: self
        count = self%n_features
    end function gp_multilabel_feature_count

    function gp_multilabel_thresholds(self) result(thresholds)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), allocatable :: thresholds(:)
        if (allocated(self%decision_threshold)) then
            allocate(thresholds(self%n_labels)); thresholds = self%decision_threshold
        else
            allocate(thresholds(0))
        end if
    end function gp_multilabel_thresholds

    subroutine gp_multilabel_set_thresholds(self, thresholds, status)
        class(gp_multilabel_classification_t), intent(inout) :: self
        real(dp), intent(in) :: thresholds(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. self%fitted() .or. size(thresholds) /= self%n_labels .or. &
            any(.not. ieee_is_finite(thresholds)) .or. any(thresholds <= 0.0_dp) .or. &
            any(thresholds >= 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP thresholds: model or thresholds are invalid")
            return
        end if
        self%decision_threshold = thresholds
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_set_thresholds

    logical function gp_multilabel_fitted(self) result(fitted)
        class(gp_multilabel_classification_t), intent(in) :: self
        fitted = self%is_fitted .and. allocated(self%models) .and. &
            allocated(self%decision_threshold) .and. self%n_labels >= 1
    end function gp_multilabel_fitted

    logical function gp_multilabel_device_supported(self, device_kind) result(supported)
        class(gp_multilabel_classification_t), intent(in) :: self
        integer, intent(in) :: device_kind
        supported = self%fitted() .and. device_kind == FORTML_DEVICE_CPU
    end function gp_multilabel_device_supported

    logical function valid_query(self, x, output, status) result(valid)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(in) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP prediction: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. any(shape(output) /= [size(x, 1), self%n_labels]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP prediction: input or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_query

    logical function valid_integer_query(self, x, output, status) result(valid)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP prediction: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(output) /= [size(x, 1), self%n_labels]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP prediction: input or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_integer_query

end module fortml_gp_multilabel_classification
