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
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
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
        procedure, public :: predict_latent_device => &
            gp_multilabel_predict_latent_device
        procedure, public :: predict_latent_jvp => gp_multilabel_predict_latent_jvp
        procedure, public :: predict_latent_vjp => gp_multilabel_predict_latent_vjp
        procedure, public :: predict_latent_parameter_jvp => &
            gp_multilabel_predict_latent_parameter_jvp
        procedure, public :: predict_latent_parameter_vjp => &
            gp_multilabel_predict_latent_parameter_vjp
        procedure, public :: predict_proba => gp_multilabel_predict_proba
        procedure, public :: predict_proba_device => &
            gp_multilabel_predict_proba_device
        procedure, public :: predict_log_proba => gp_multilabel_predict_log_proba
        procedure, public :: predict_log_proba_device => &
            gp_multilabel_predict_log_proba_device
        procedure, public :: predict_proba_jvp => gp_multilabel_predict_proba_jvp
        procedure, public :: predict_proba_vjp => gp_multilabel_predict_proba_vjp
        procedure, public :: predict_log_proba_jvp => gp_multilabel_predict_log_proba_jvp
        procedure, public :: predict_log_proba_vjp => gp_multilabel_predict_log_proba_vjp
        procedure, public :: predict_proba_parameter_jvp => &
            gp_multilabel_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            gp_multilabel_predict_proba_parameter_vjp
        procedure, public :: predict_log_proba_parameter_jvp => &
            gp_multilabel_predict_log_proba_parameter_jvp
        procedure, public :: predict_log_proba_parameter_vjp => &
            gp_multilabel_predict_log_proba_parameter_vjp
        procedure, public :: predict_log_proba_shared_parameter_jvp => &
            gp_multilabel_predict_log_proba_shared_parameter_jvp
        procedure, public :: predict_log_proba_shared_parameter_vjp => &
            gp_multilabel_predict_log_proba_shared_parameter_vjp
        procedure, public :: predict_log_proba_shared_parameter_jvp_device => &
            gp_multilabel_predict_log_proba_shared_parameter_jvp_device
        procedure, public :: predict_log_proba_shared_parameter_vjp_device => &
            gp_multilabel_predict_log_proba_shared_parameter_vjp_device
        procedure, public :: predict_log_proba_kernel_parameter_jvp => &
            gp_multilabel_predict_log_proba_shared_parameter_jvp
        procedure, public :: predict_log_proba_kernel_parameter_vjp => &
            gp_multilabel_predict_log_proba_shared_parameter_vjp
        procedure, public :: predict_log_proba_kernel_parameter_jvp_device => &
            gp_multilabel_predict_log_proba_shared_parameter_jvp_device
        procedure, public :: predict_log_proba_kernel_parameter_vjp_device => &
            gp_multilabel_predict_log_proba_shared_parameter_vjp_device
        procedure, public :: predict => gp_multilabel_predict
        procedure, public :: predict_device => gp_multilabel_predict_device
        procedure, public :: parameter_count => gp_multilabel_parameter_count
        procedure, public :: parameters => gp_multilabel_parameters
        procedure, public :: hyperparameter_gradient => &
            gp_multilabel_hyperparameter_gradient
        procedure, public :: shared_parameter_count => &
            gp_multilabel_shared_parameter_count
        procedure, public :: hyperparameter_count => &
            gp_multilabel_shared_parameter_count
        procedure, public :: shared_parameters => gp_multilabel_shared_parameters
        procedure, public :: set_shared_parameters => &
            gp_multilabel_set_shared_parameters
        procedure, public :: fixed_state_value_gradient => &
            gp_multilabel_fixed_state_value_gradient
        procedure, public :: fixed_state_independent_value_gradient => &
            gp_multilabel_fixed_state_independent_value_gradient
        procedure, public :: shared_hyperparameter_gradient => &
            gp_multilabel_shared_hyperparameter_gradient
        procedure, public :: optimize_lbfgsb => gp_multilabel_optimize_lbfgsb
        procedure, public :: optimize_independent_lbfgsb => &
            gp_multilabel_optimize_independent_lbfgsb
        procedure, public :: label_count => gp_multilabel_label_count
        procedure, public :: feature_count => gp_multilabel_feature_count
        procedure, public :: thresholds => gp_multilabel_thresholds
        procedure, public :: set_thresholds => gp_multilabel_set_thresholds
        procedure, public :: fitted => gp_multilabel_fitted
        procedure, public :: device_supported => gp_multilabel_device_supported
    end type gp_multilabel_classification_t

    type, public :: gp_multilabel_training_objective_t
        !! FortOpt adapter for shared or independent fixed-state objectives.
        private
        type(gp_multilabel_classification_t), pointer :: model => null()
        logical :: independent = .false.
    contains
        procedure, public :: initialize => gp_multilabel_objective_initialize
        procedure, public :: initialize_independent => &
            gp_multilabel_objective_initialize_independent
        procedure, public :: parameter_count => gp_multilabel_objective_parameter_count
        procedure, public :: parameters => gp_multilabel_objective_parameters
        procedure, public :: value_gradient => gp_multilabel_objective_value_gradient
        procedure, public :: jvp => gp_multilabel_objective_jvp
        procedure, public :: vjp => gp_multilabel_objective_vjp
        procedure, public :: fortopt => gp_multilabel_objective_fortopt
    end type gp_multilabel_training_objective_t

    type, public :: gp_multilabel_lbfgsb_options_t
        !! Bounds and convergence controls for shared kernel-log fitting.
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
    end type gp_multilabel_lbfgsb_options_t

    type, public :: gp_multilabel_lbfgsb_result_t
        !! Diagnostics returned by `gp_multilabel_optimize_lbfgsb`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type gp_multilabel_lbfgsb_result_t

    public :: gp_multilabel_fit
    public :: gp_multilabel_predict_latent
    public :: gp_multilabel_predict_latent_device
    public :: gp_multilabel_predict_latent_jvp
    public :: gp_multilabel_predict_latent_vjp
    public :: gp_multilabel_predict_latent_parameter_jvp
    public :: gp_multilabel_predict_latent_parameter_vjp
    public :: gp_multilabel_predict_proba
    public :: gp_multilabel_predict_proba_device
    public :: gp_multilabel_predict_log_proba
    public :: gp_multilabel_predict_log_proba_device
    public :: gp_multilabel_predict_proba_jvp
    public :: gp_multilabel_predict_proba_vjp
    public :: gp_multilabel_predict_log_proba_jvp
    public :: gp_multilabel_predict_log_proba_vjp
    public :: gp_multilabel_predict_proba_parameter_jvp
    public :: gp_multilabel_predict_proba_parameter_vjp
    public :: gp_multilabel_predict_log_proba_parameter_jvp
    public :: gp_multilabel_predict_log_proba_parameter_vjp
    public :: gp_multilabel_predict_log_proba_shared_parameter_jvp
    public :: gp_multilabel_predict_log_proba_shared_parameter_vjp
    public :: gp_multilabel_predict_log_proba_shared_parameter_jvp_device
    public :: gp_multilabel_predict_log_proba_shared_parameter_vjp_device
    public :: gp_multilabel_predict
    public :: gp_multilabel_predict_device
    public :: gp_multilabel_optimize_lbfgsb
    public :: gp_multilabel_optimize_independent_lbfgsb
    public :: gp_multilabel_set_shared_parameters
    public :: gp_multilabel_fixed_state_value_gradient
    public :: gp_multilabel_fixed_state_independent_value_gradient

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

    subroutine gp_multilabel_predict_latent_device(self, device, x, mean, variance, status)
        !! Dispatch latent posterior prediction without hidden host fallback.
        class(gp_multilabel_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:, :), variance(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP latent device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_latent(x, mean, variance, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multilabel GP latent device: no resident CUDA Laplace heads are linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP latent device: device kind is invalid")
        end select
    end subroutine gp_multilabel_predict_latent_device

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

    !> Return the natural logarithm of each independent label probability.
    !!
    !! The columns retain indicator order and are not normalized across labels.
    !! The fitted Laplace modes and curvatures remain fixed, matching the
    !! probability prediction and product contracts.  A finite positive
    !! probability is required before taking its logarithm; this keeps the
    !! result usable by downstream log-loss reductions without hidden NaNs.
    subroutine gp_multilabel_predict_log_proba(self, x, log_probabilities, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. valid_query(self, x, log_probabilities, status)) return
        allocate(probabilities(size(x, 1), self%n_labels))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(probabilities)) .or. &
            any(probabilities <= 0.0_dp) .or. any(probabilities > 1.0_dp)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability: probabilities are not finite and positive")
            return
        end if
        log_probabilities = log(probabilities)
        if (any(.not. ieee_is_finite(log_probabilities))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_log_proba

    !> Dispatch log-probability prediction without an implicit host fallback.
    subroutine gp_multilabel_predict_log_proba_device(self, device, x, &
            log_probabilities, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_log_proba(x, log_probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multilabel GP log probability device: no resident CUDA Laplace heads are linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability device: device kind is invalid")
        end select
    end subroutine gp_multilabel_predict_log_proba_device

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

    !> Forward query-input product of independent label log probabilities.
    subroutine gp_multilabel_predict_log_proba_jvp(self, x, x_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)

        if (.not. valid_query(self, x, log_probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability JVP: input or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_labels), &
            probabilities_dot(size(x, 1), self%n_labels))
        call self%predict_proba_jvp(x, x_dot, probabilities, probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(probabilities)) .or. &
            any(probabilities <= 0.0_dp) .or. any(probabilities > 1.0_dp)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability JVP: probabilities are not finite and positive")
            return
        end if
        log_probabilities = log(probabilities)
        log_probabilities_dot = probabilities_dot / probabilities
        if (any(.not. ieee_is_finite(log_probabilities)) .or. &
            any(.not. ieee_is_finite(log_probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_log_proba_jvp

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

    !> Reverse query-input product of independent label log probabilities.
    subroutine gp_multilabel_predict_log_proba_vjp(self, x, log_probabilities_bar, &
            x_bar, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probability_bar(:, :)

        x_bar = 0.0_dp
        if (.not. valid_query(self, x, log_probabilities_bar, status)) return
        if (any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability VJP: input or cotangent is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_labels), &
            probability_bar(size(x, 1), self%n_labels))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(probabilities)) .or. &
            any(probabilities <= 0.0_dp) .or. any(probabilities > 1.0_dp)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability VJP: probabilities are not finite and positive")
            return
        end if
        probability_bar = log_probabilities_bar / probabilities
        if (any(.not. ieee_is_finite(probability_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability VJP: cotangent is not finite")
            return
        end if
        call self%predict_proba_vjp(x, probability_bar, x_bar, status)
    end subroutine gp_multilabel_predict_log_proba_vjp

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

    !> Forward fixed-state product of log probabilities with packed per-label
    !! kernel-log parameters.  The direction is concatenated in label order.
    subroutine gp_multilabel_predict_log_proba_parameter_jvp(self, x, direction, &
            log_probabilities, log_probabilities_dot, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)

        if (.not. valid_query(self, x, log_probabilities, status)) return
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability parameter JVP: direction or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_labels), &
            probabilities_dot(size(x, 1), self%n_labels))
        call self%predict_proba_parameter_jvp(x, direction, probabilities, &
            probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(probabilities)) .or. &
            any(probabilities <= 0.0_dp) .or. any(probabilities > 1.0_dp)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability parameter JVP: probabilities are not finite and positive")
            return
        end if
        log_probabilities = log(probabilities)
        log_probabilities_dot = probabilities_dot / probabilities
        if (any(.not. ieee_is_finite(log_probabilities)) .or. &
            any(.not. ieee_is_finite(log_probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability parameter JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_log_proba_parameter_jvp

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

    !> Reverse fixed-state product of log probabilities with packed per-label
    !! kernel-log parameters.  The returned cotangent uses label order.
    subroutine gp_multilabel_predict_log_proba_parameter_vjp(self, x, &
            log_probabilities_bar, parameter_bar, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probability_bar(:, :)

        parameter_bar = 0.0_dp
        if (.not. valid_query(self, x, log_probabilities_bar, status)) return
        if (size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability parameter VJP: input or cotangent is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_labels), &
            probability_bar(size(x, 1), self%n_labels))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(probabilities)) .or. &
            any(probabilities <= 0.0_dp) .or. any(probabilities > 1.0_dp)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability parameter VJP: probabilities are not finite and positive")
            return
        end if
        probability_bar = log_probabilities_bar / probabilities
        if (any(.not. ieee_is_finite(probability_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability parameter VJP: cotangent is not finite")
            return
        end if
        call self%predict_proba_parameter_vjp(x, probability_bar, parameter_bar, status)
    end subroutine gp_multilabel_predict_log_proba_parameter_vjp

    !> Forward fixed-state product of log probabilities with one packed kernel
    !! direction shared by every independent label head.
    subroutine gp_multilabel_predict_log_proba_shared_parameter_jvp(self, x, direction, &
            log_probabilities, log_probabilities_dot, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local(:, :), local_dot(:, :)
        integer :: i

        if (.not. valid_query(self, x, log_probabilities, status)) return
        if (size(direction) /= self%shared_parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability shared parameter JVP: direction or output shape is invalid")
            return
        end if
        allocate(local(size(x, 1), 2), local_dot(size(x, 1), 2))
        do i = 1, self%n_labels
            call self%models(i)%predict_proba_parameter_jvp(x, direction, local, &
                local_dot, status)
            if (status%code /= FORTNUM_OK) return
            if (any(.not. ieee_is_finite(local(:, 2))) .or. &
                any(local(:, 2) <= 0.0_dp) .or. any(local(:, 2) > 1.0_dp)) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "multilabel GP log probability shared parameter JVP: probabilities are not finite and positive")
                return
            end if
            log_probabilities(:, i) = log(local(:, 2))
            log_probabilities_dot(:, i) = local_dot(:, 2) / local(:, 2)
        end do
        if (any(.not. ieee_is_finite(log_probabilities)) .or. &
            any(.not. ieee_is_finite(log_probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability shared parameter JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_log_proba_shared_parameter_jvp

    !> Reverse fixed-state product of log probabilities with one packed kernel
    !! cotangent shared by every independent label head.  Head cotangents are
    !! summed into the common kernel-log vector.
    subroutine gp_multilabel_predict_log_proba_shared_parameter_vjp(self, x, &
            log_probabilities_bar, parameter_bar, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), binary_bar(:, :), local_bar(:)
        integer :: i

        parameter_bar = 0.0_dp
        if (.not. valid_query(self, x, log_probabilities_bar, status)) return
        if (size(parameter_bar) /= self%shared_parameter_count() .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability shared parameter VJP: input or cotangent is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_labels), &
            binary_bar(size(x, 1), 2), local_bar(size(parameter_bar)))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(probabilities)) .or. &
            any(probabilities <= 0.0_dp) .or. any(probabilities > 1.0_dp)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability shared parameter VJP: probabilities are not finite and positive")
            return
        end if
        binary_bar(:, 1) = 0.0_dp
        do i = 1, self%n_labels
            binary_bar(:, 2) = log_probabilities_bar(:, i) / probabilities(:, i)
            if (any(.not. ieee_is_finite(binary_bar(:, 2)))) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "multilabel GP log probability shared parameter VJP: cotangent is not finite")
                return
            end if
            call self%models(i)%predict_proba_parameter_vjp(x, binary_bar, local_bar, status)
            if (status%code /= FORTNUM_OK) return
            parameter_bar = parameter_bar + local_bar
        end do
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP log probability shared parameter VJP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_predict_log_proba_shared_parameter_vjp

    !> Device-dispatched shared-kernel log-probability parameter JVP.
    subroutine gp_multilabel_predict_log_proba_shared_parameter_jvp_device(self, device, &
            x, direction, log_probabilities, log_probabilities_dot, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(inout) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability shared parameter JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_log_proba_shared_parameter_jvp(x, direction, &
                log_probabilities, log_probabilities_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multilabel GP log probability shared parameter JVP device: no resident CUDA Laplace heads are linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability shared parameter JVP device: device kind is invalid")
        end select
    end subroutine gp_multilabel_predict_log_proba_shared_parameter_jvp_device

    !> Device-dispatched shared-kernel log-probability parameter VJP.
    subroutine gp_multilabel_predict_log_proba_shared_parameter_vjp_device(self, device, &
            x, log_probabilities_bar, parameter_bar, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability shared parameter VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_log_proba_shared_parameter_vjp(x, log_probabilities_bar, &
                parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multilabel GP log probability shared parameter VJP device: no resident CUDA Laplace heads are linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP log probability shared parameter VJP device: device kind is invalid")
        end select
    end subroutine gp_multilabel_predict_log_proba_shared_parameter_vjp_device

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
        if (.not. gp_multilabel_fitted(self)) return
        do i = 1, self%n_labels
            count = count + self%models(i)%parameter_count()
        end do
    end function gp_multilabel_parameter_count

    function gp_multilabel_parameters(self) result(parameters)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:), local(:)
        integer :: i, first, last
        if (.not. gp_multilabel_fitted(self)) then
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
        if (.not. gp_multilabel_fitted(self)) then
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

    integer function gp_multilabel_shared_parameter_count(self) result(count)
        class(gp_multilabel_classification_t), intent(in) :: self

        count = 0
        if (.not. gp_multilabel_fitted(self)) return
        count = self%models(1)%parameter_count()
    end function gp_multilabel_shared_parameter_count

    function gp_multilabel_shared_parameters(self) result(parameters)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (.not. gp_multilabel_fitted(self)) then
            allocate(parameters(0))
            return
        end if
        parameters = self%models(1)%parameters()
    end function gp_multilabel_shared_parameters

    !> Transactionally set one shared kernel-log vector on every label head.
    !!
    !! Candidate heads are built first.  A failed factorization, unsupported
    !! kernel product, or invalid value therefore leaves every fitted head at
    !! its previous state.
    subroutine gp_multilabel_set_shared_parameters(self, parameters, status)
        class(gp_multilabel_classification_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        type(gp_classification_t), allocatable :: candidate(:)
        integer :: i

        if (.not. gp_multilabel_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP shared parameters: model is not fitted")
            return
        end if
        if (size(parameters) /= self%shared_parameter_count() .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP shared parameters: shape or values are invalid")
            return
        end if
        allocate(candidate(self%n_labels))
        candidate = self%models
        do i = 1, self%n_labels
            call candidate(i)%set_parameters(parameters, status)
            if (.not. status_ok(status)) return
        end do
        self%models = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_set_shared_parameters

    !> Minimize the negative shared fixed-state Laplace mode posterior.
    !!
    !! The mode and likelihood curvature from the original fit are held fixed
    !! while FortOpt changes common logarithmic kernel parameters.  This is a
    !! deliberate, smooth outer-HPO slice; a fresh `fit` remains the API for
    !! recomputing all per-label Newton states.
    subroutine gp_multilabel_fixed_state_value_gradient(self, parameters, value, &
            gradient, status)
        class(gp_multilabel_classification_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        type(gp_classification_t), allocatable :: candidate(:)
        real(dp), allocatable :: local_gradient(:)
        real(dp) :: local_value
        integer :: i

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. gp_multilabel_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP fixed-state objective: model is not fitted")
            return
        end if
        if (size(parameters) /= self%shared_parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP fixed-state objective: shape or values are invalid")
            return
        end if
        allocate(candidate(self%n_labels), local_gradient(size(parameters)))
        candidate = self%models
        do i = 1, self%n_labels
            call candidate(i)%set_parameters(parameters, status)
            if (.not. status_ok(status)) return
        end do
        value = 0.0_dp
        gradient = 0.0_dp
        do i = 1, self%n_labels
            call candidate(i)%fixed_state_log_posterior(local_value, status)
            if (.not. status_ok(status)) return
            call candidate(i)%fixed_state_log_posterior_gradient(local_gradient, status)
            if (.not. status_ok(status)) return
            value = value - local_value
            gradient = gradient - local_gradient
        end do
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP fixed-state objective: result is not finite")
            return
        end if
        self%models = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_fixed_state_value_gradient

    !> Evaluate the negative fixed-state mode posterior with one packed
    !! kernel-log block per label.  The fitted modes and likelihood
    !! curvatures remain fixed; candidate covariance factors are constructed
    !! for every head before any state is committed.
    subroutine gp_multilabel_fixed_state_independent_value_gradient(self, parameters, &
            value, gradient, status)
        class(gp_multilabel_classification_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        type(gp_classification_t), allocatable :: candidate(:)
        real(dp), allocatable :: local_gradient(:)
        real(dp) :: local_value
        integer :: i, first, last, local_count

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. gp_multilabel_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP independent objective: model is not fitted")
            return
        end if
        if (size(parameters) /= self%parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP independent objective: shape or values are invalid")
            return
        end if

        allocate(candidate(self%n_labels))
        candidate = self%models
        first = 1
        do i = 1, self%n_labels
            local_count = candidate(i)%parameter_count()
            last = first + local_count - 1
            call candidate(i)%set_parameters(parameters(first:last), status)
            if (.not. status_ok(status)) return
            first = last + 1
        end do

        value = 0.0_dp
        first = 1
        do i = 1, self%n_labels
            local_count = candidate(i)%parameter_count()
            last = first + local_count - 1
            allocate(local_gradient(local_count))
            call candidate(i)%fixed_state_log_posterior(local_value, status)
            if (.not. status_ok(status)) return
            call candidate(i)%fixed_state_log_posterior_gradient(local_gradient, status)
            if (.not. status_ok(status)) return
            value = value - local_value
            gradient(first:last) = -local_gradient
            deallocate(local_gradient)
            first = last + 1
        end do
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP independent objective: result is not finite")
            return
        end if
        self%models = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_fixed_state_independent_value_gradient

    subroutine gp_multilabel_shared_hyperparameter_gradient(self, gradient, status)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local(:)
        integer :: i

        gradient = 0.0_dp
        if (.not. gp_multilabel_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP shared gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%shared_parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP shared gradient: output shape is invalid")
            return
        end if
        allocate(local(size(gradient)))
        do i = 1, self%n_labels
            call self%models(i)%fixed_state_log_posterior_gradient(local, status)
            if (.not. status_ok(status)) return
            gradient = gradient + local
        end do
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP shared gradient: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_shared_hyperparameter_gradient

    subroutine gp_multilabel_objective_initialize(self, model, status)
        class(gp_multilabel_training_objective_t), intent(out) :: self
        class(gp_multilabel_classification_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        nullify(self%model)
        self%independent = .false.
        if (.not. gp_multilabel_fitted(model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP objective: model is not fitted")
            return
        end if
        self%model => model
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_objective_initialize

    subroutine gp_multilabel_objective_initialize_independent(self, model, status)
        class(gp_multilabel_training_objective_t), intent(out) :: self
        class(gp_multilabel_classification_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        nullify(self%model)
        self%independent = .true.
        if (.not. gp_multilabel_fitted(model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP independent objective: model is not fitted")
            return
        end if
        self%model => model
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_objective_initialize_independent

    integer function gp_multilabel_objective_parameter_count(self) result(count)
        class(gp_multilabel_training_objective_t), intent(in) :: self

        count = 0
        if (.not. associated(self%model)) return
        if (self%independent) then
            count = self%model%parameter_count()
        else
            count = self%model%shared_parameter_count()
        end if
    end function gp_multilabel_objective_parameter_count

    function gp_multilabel_objective_parameters(self) result(parameters)
        class(gp_multilabel_training_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (.not. associated(self%model)) then
            allocate(parameters(0))
            return
        end if
        if (self%independent) then
            parameters = self%model%parameters()
        else
            parameters = self%model%shared_parameters()
        end if
    end function gp_multilabel_objective_parameters

    subroutine gp_multilabel_objective_value_gradient(self, parameters, value, &
            gradient, status)
        class(gp_multilabel_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP objective: adapter is not initialized")
            return
        end if
        if (self%independent) then
            call self%model%fixed_state_independent_value_gradient(parameters, value, &
                gradient, status)
        else
            call self%model%fixed_state_value_gradient(parameters, value, gradient, status)
        end if
    end subroutine gp_multilabel_objective_value_gradient

    subroutine gp_multilabel_objective_jvp(self, parameters, direction, value, &
            tangent, status)
        class(gp_multilabel_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP objective JVP: direction is invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP objective JVP: tangent is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_objective_jvp

    subroutine gp_multilabel_objective_vjp(self, parameters, output_bar, gradient, status)
        class(gp_multilabel_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP objective VJP: cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        gradient = output_bar*gradient
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP objective VJP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_objective_vjp

    subroutine gp_multilabel_objective_fortopt(self, objective, status)
        class(gp_multilabel_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            gp_multilabel_objective_context_callback, status)
    end subroutine gp_multilabel_objective_fortopt

    subroutine gp_multilabel_objective_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (gp_multilabel_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP objective: context has the wrong type")
        end select
    end subroutine gp_multilabel_objective_context_callback

    subroutine gp_multilabel_optimize_lbfgsb(model, options, result, status)
        !! Optimize shared logarithmic kernel parameters with FortOpt L-BFGS-B.
        class(gp_multilabel_classification_t), target, intent(inout) :: model
        type(gp_multilabel_lbfgsb_options_t), intent(in) :: options
        type(gp_multilabel_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(gp_multilabel_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_parameters
        type(gp_multilabel_lbfgsb_result_t) :: result_default

        result = result_default
        if (.not. valid_lbfgsb_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP L-BFGS-B: options are invalid")
            return
        end if
        call adapter%initialize(model, status)
        if (.not. status_ok(status)) return
        n_parameters = adapter%parameter_count()
        parameters = model%shared_parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower = options%lower_bound
        upper = options%upper_bound
        parameters = min(max(parameters, lower), upper)
        call adapter%fortopt(objective, status)
        if (.not. status_ok(status)) return
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call optimizer%minimize(objective, parameters, lower, upper, optimizer_options, &
            optimizer_result, status)
        if (.not. status_ok(status)) return
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) return
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = sqrt(sum(gradient*gradient))
        if (.not. ieee_is_finite(result%objective) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP L-BFGS-B: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_optimize_lbfgsb

    subroutine gp_multilabel_optimize_independent_lbfgsb(model, options, result, status)
        !! Optimize one fixed-state kernel-log block per label with FortOpt
        !! L-BFGS-B.  Each block is independent, but one optimizer sees the
        !! concatenated vector and one exact objective/gradient callback.
        class(gp_multilabel_classification_t), target, intent(inout) :: model
        type(gp_multilabel_lbfgsb_options_t), intent(in) :: options
        type(gp_multilabel_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(gp_multilabel_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_parameters
        type(gp_multilabel_lbfgsb_result_t) :: result_default

        result = result_default
        if (.not. valid_lbfgsb_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel GP independent L-BFGS-B: options are invalid")
            return
        end if
        call adapter%initialize_independent(model, status)
        if (.not. status_ok(status)) return
        n_parameters = adapter%parameter_count()
        parameters = model%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower = options%lower_bound
        upper = options%upper_bound
        parameters = min(max(parameters, lower), upper)
        call adapter%fortopt(objective, status)
        if (.not. status_ok(status)) return
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call optimizer%minimize(objective, parameters, lower, upper, optimizer_options, &
            optimizer_result, status)
        if (.not. status_ok(status)) return
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) return
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = sqrt(sum(gradient*gradient))
        if (.not. ieee_is_finite(result%objective) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP independent L-BFGS-B: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multilabel GP independent L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multilabel_optimize_independent_lbfgsb

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
        if (.not. gp_multilabel_fitted(self) .or. size(thresholds) /= self%n_labels .or. &
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
        fitted = self%is_fitted
        if (.not. fitted) return
        if (.not. allocated(self%models)) then
            fitted = .false.
            return
        end if
        if (.not. allocated(self%decision_threshold)) then
            fitted = .false.
            return
        end if
        if (self%n_labels < 1 .or. size(self%models) /= self%n_labels .or. &
            size(self%decision_threshold) /= self%n_labels) then
            fitted = .false.
            return
        end if
        if (any(.not. ieee_is_finite(self%decision_threshold)) .or. &
            any(self%decision_threshold <= 0.0_dp) .or. &
            any(self%decision_threshold >= 1.0_dp)) fitted = .false.
    end function gp_multilabel_fitted

    logical function gp_multilabel_device_supported(self, device_kind) result(supported)
        class(gp_multilabel_classification_t), intent(in) :: self
        integer, intent(in) :: device_kind
        supported = gp_multilabel_fitted(self) .and. device_kind == FORTML_DEVICE_CPU
    end function gp_multilabel_device_supported

    logical function valid_query(self, x, output, status) result(valid)
        class(gp_multilabel_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(in) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        valid = .false.
        if (.not. gp_multilabel_fitted(self)) then
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
        if (.not. gp_multilabel_fitted(self)) then
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

    logical function valid_lbfgsb_options(options) result(valid)
        type(gp_multilabel_lbfgsb_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. options%lower_bound <= options%upper_bound .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp
        valid = valid .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. ieee_is_finite(options%objective_tolerance) .and. &
            ieee_is_finite(options%lower_bound) .and. ieee_is_finite(options%upper_bound)
    end function valid_lbfgsb_options

end module fortml_gp_multilabel_classification
