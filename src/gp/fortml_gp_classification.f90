module fortml_gp_classification
    !! Laplace-approximated binary Gaussian-process classification.
    !!
    !! The classifier supports Bernoulli logistic and probit likelihoods.  The
    !! latent mode is found with the standard damped Newton iteration and the
    !! posterior covariance is represented by the symmetric matrix
    !! ``I + sqrt(W) K sqrt(W)``.  Consequently prediction uses exactly the
    !! same kernel implementation as GP regression, including composite and
    !! user kernels that satisfy the kernel validation contract.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t, clone_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    real(dp), parameter :: PI = 3.1415926535897932384626433832795_dp
    real(dp), parameter :: SQRT_TWO = 1.4142135623730950488016887242097_dp
    real(dp), parameter :: SQRT_TWO_PI = 2.506628274631000502415765284811_dp
    real(dp), parameter :: MIN_LIKELIHOOD_CURVATURE = 1.0e-12_dp

    integer, parameter, public :: GP_LIKELIHOOD_LOGISTIC = 1
    integer, parameter, public :: GP_LIKELIHOOD_PROBIT = 2

    type, public :: gp_classification_options_t
        integer :: likelihood = GP_LIKELIHOOD_LOGISTIC
        integer :: max_iterations = 50
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: jitter = 1.0e-8_dp
        real(dp) :: damping = 1.0_dp
    end type gp_classification_options_t

    type, public :: gp_classification_state_t
        integer :: iterations = 0
        real(dp) :: final_step_norm = huge(1.0_dp)
        real(dp) :: log_posterior = -huge(1.0_dp)
        logical :: converged = .false.
    end type gp_classification_state_t

    type, public :: gp_classification_t
        private
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: prior_factorization
        type(cholesky_factorization_t) :: posterior_factorization
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: covariance(:, :)
        real(dp), allocatable :: mode(:)
        real(dp), allocatable :: alpha(:)
        real(dp), allocatable :: sqrt_w(:)
        integer, allocatable :: training_labels(:)
        real(dp), allocatable :: sample_weights(:)
        integer :: class_label(2) = 0
        integer :: n_samples = 0
        integer :: n_features = 0
        integer :: likelihood = GP_LIKELIHOOD_LOGISTIC
        real(dp) :: jitter = 1.0e-8_dp
    contains
        procedure, public :: fit => gp_classification_fit
        procedure, public :: predict_latent => gp_classification_predict_latent
        procedure, public :: predict_latent_device => gp_classification_predict_latent_device
        procedure, public :: predict_latent_jvp => gp_classification_predict_latent_jvp
        procedure, public :: predict_latent_vjp => gp_classification_predict_latent_vjp
        procedure, public :: predict_latent_parameter_jvp => &
            gp_classification_predict_latent_parameter_jvp
        procedure, public :: predict_latent_parameter_vjp => &
            gp_classification_predict_latent_parameter_vjp
        procedure, public :: predict_latent_hyperparameter_jvp => &
            gp_classification_predict_latent_hyperparameter_jvp
        procedure, public :: predict_proba => gp_classification_predict_proba
        procedure, public :: predict_proba_device => gp_classification_predict_proba_device
        procedure, public :: predict_log_proba => gp_classification_predict_log_proba
        procedure, public :: predict_log_proba_device => &
            gp_classification_predict_log_proba_device
        procedure, public :: predict_proba_jvp => gp_classification_predict_proba_jvp
        procedure, public :: predict_proba_vjp => gp_classification_predict_proba_vjp
        procedure, public :: predict_log_proba_jvp => gp_classification_predict_log_proba_jvp
        procedure, public :: predict_log_proba_vjp => gp_classification_predict_log_proba_vjp
        procedure, public :: predict_proba_parameter_jvp => &
            gp_classification_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            gp_classification_predict_proba_parameter_vjp
        procedure, public :: predict_proba_hyperparameter_jvp => &
            gp_classification_predict_proba_hyperparameter_jvp
        procedure, public :: predict_proba_hyperparameter_jvp_device => &
            gp_classification_predict_proba_hyperparameter_jvp_device
        procedure, public :: predict_log_proba_parameter_jvp => &
            gp_classification_predict_log_proba_parameter_jvp
        procedure, public :: predict_log_proba_parameter_vjp => &
            gp_classification_predict_log_proba_parameter_vjp
        procedure, public :: predict => gp_classification_predict
        procedure, public :: classes => gp_classification_classes
        procedure, public :: feature_count => gp_classification_feature_count
        procedure, public :: parameter_count => gp_classification_parameter_count
        procedure, public :: parameters => gp_classification_parameters
        procedure, public :: set_parameters => gp_classification_set_parameters
        procedure, public :: fixed_state_log_posterior => &
            gp_classification_fixed_state_log_posterior
        procedure, public :: fixed_state_log_posterior_gradient => &
            gp_classification_fixed_state_log_posterior_gradient
        procedure, public :: hyperparameter_gradient => &
            gp_classification_hyperparameter_gradient
        procedure, public :: hyperparameter_hvp => gp_classification_hyperparameter_hvp
        procedure, public :: hyperparameter_hvp_device => &
            gp_classification_hyperparameter_hvp_device
        procedure, public :: fitted => gp_classification_fitted
        procedure, public :: likelihood_kind => gp_classification_likelihood
        procedure, public :: device_supported => gp_classification_device_supported
    end type gp_classification_t

    public :: gp_classification_fit
    public :: gp_classification_predict_latent
    public :: gp_classification_predict_latent_device
    public :: gp_classification_predict_latent_jvp
    public :: gp_classification_predict_latent_vjp
    public :: gp_classification_predict_latent_parameter_jvp
    public :: gp_classification_predict_latent_parameter_vjp
    public :: gp_classification_predict_latent_hyperparameter_jvp
    public :: gp_classification_predict_proba
    public :: gp_classification_predict_proba_device
    public :: gp_classification_predict_log_proba
    public :: gp_classification_predict_log_proba_device
    public :: gp_classification_predict_proba_jvp
    public :: gp_classification_predict_proba_vjp
    public :: gp_classification_predict_log_proba_jvp
    public :: gp_classification_predict_log_proba_vjp
    public :: gp_classification_predict_proba_parameter_jvp
    public :: gp_classification_predict_proba_parameter_vjp
    public :: gp_classification_predict_proba_hyperparameter_jvp
    public :: gp_classification_predict_proba_hyperparameter_jvp_device
    public :: gp_classification_predict_log_proba_parameter_jvp
    public :: gp_classification_predict_log_proba_parameter_vjp
    public :: gp_classification_predict
    public :: gp_classification_set_parameters
    public :: gp_classification_fixed_state_log_posterior
    public :: gp_classification_fixed_state_log_posterior_gradient
    public :: gp_classification_log_likelihood_value
    public :: gp_classification_log_likelihood_jvp
    public :: gp_classification_log_likelihood_vjp
    public :: gp_classification_hyperparameter_gradient
    public :: gp_classification_hyperparameter_hvp
    public :: gp_classification_hyperparameter_hvp_device
    public :: gp_classification_likelihood_device_supported

contains

    subroutine gp_classification_fit(self, x, labels, kernel, status, options, state, &
            sample_weight)
        class(gp_classification_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), intent(in) :: kernel
        type(fortnum_status_t), intent(out) :: status
        type(gp_classification_options_t), intent(in), optional :: options
        type(gp_classification_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:)
        type(gp_classification_options_t) :: requested
        type(gp_classification_state_t) :: result
        real(dp), allocatable :: b(:), rhs(:), sqrt_w(:), mode_new(:)
        real(dp), allocatable :: matrix(:, :), weights(:)
        real(dp) :: eta, probability, likelihood_gradient, curvature
        real(dp) :: step_norm, scale
        integer :: i, iteration
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(gp_classification_options_t) :: gp_classification_options_t_default
        type(gp_classification_state_t) :: gp_classification_state_t_default

        result = gp_classification_state_t_default
        if (present(state)) state = result
        requested = gp_classification_options_t_default
        if (present(options)) requested = options
        if (.not. valid_options(requested)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: options are invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: inputs must be finite")
            return
        end if
        call build_sample_weights(size(x, 1), sample_weight, weights, status)
        if (status%code /= FORTNUM_OK) return
        self%class_label = [minval(labels), maxval(labels)]
        if (self%class_label(1) == self%class_label(2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: exactly two classes are required")
            return
        end if
        do i = 1, size(labels)
            if (labels(i) /= self%class_label(1) .and. &
                labels(i) /= self%class_label(2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP classification fit: exactly two classes are required")
                return
            end if
        end do
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: kernel dimension is invalid")
            return
        end if

        self%kernel = kernel
        allocate(self%x_train, source=x)
        allocate(self%training_labels, source=labels)
        allocate(self%sample_weights, source=weights)
        self%n_samples = size(x, 1)
        self%n_features = size(x, 2)
        self%likelihood = requested%likelihood
        allocate(self%covariance(self%n_samples, self%n_samples))
        call self%kernel%matrix(self%x_train, self%x_train, self%covariance, status)
        if (status%code /= FORTNUM_OK) return
        self%jitter = requested%jitter
        do i = 1, self%n_samples
            self%covariance(i, i) = self%covariance(i, i) + requested%jitter
        end do
        call self%prior_factorization%factorize(self%covariance, status)
        if (status%code /= FORTNUM_OK) return

        allocate(self%mode(self%n_samples), self%alpha(self%n_samples))
        allocate(self%sqrt_w(self%n_samples), b(self%n_samples), rhs(self%n_samples))
        allocate(mode_new(self%n_samples), sqrt_w(self%n_samples), &
            matrix(self%n_samples, self%n_samples))
        self%mode = 0.0_dp
        do iteration = 1, requested%max_iterations
            do i = 1, self%n_samples
                eta = encoded_label(labels(i), self%class_label)*self%mode(i)
                call likelihood_terms(eta, self%likelihood, probability, &
                    likelihood_gradient, curvature)
                if (weights(i) > 0.0_dp) then
                    sqrt_w(i) = sqrt(max(weights(i)*curvature, MIN_LIKELIHOOD_CURVATURE))
                else
                    sqrt_w(i) = 0.0_dp
                end if
                b(i) = weights(i)*curvature*self%mode(i) + &
                    encoded_label(labels(i), self%class_label)*weights(i)*likelihood_gradient
            end do
            call posterior_system(self%covariance, sqrt_w, matrix)
            call self%posterior_factorization%factorize(matrix, status)
            if (status%code /= FORTNUM_OK) return
            rhs = sqrt_w*matmul(self%covariance, b)
            call self%posterior_factorization%solve(rhs, status)
            if (status%code /= FORTNUM_OK) return
            mode_new = matmul(self%covariance, b - sqrt_w*rhs)
            scale = max(1.0_dp, maxval(abs(self%mode)))
            step_norm = requested%damping*maxval(abs(mode_new - self%mode))/scale
            self%mode = self%mode + requested%damping*(mode_new - self%mode)
            result%iterations = iteration
            result%final_step_norm = step_norm
            if (step_norm <= requested%tolerance) then
                result%converged = .true.
                exit
            end if
        end do
        if (.not. result%converged) then
            if (present(state)) state = result
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification fit: Newton iteration limit reached")
            return
        end if

        ! Rebuild W and the posterior factorization at the damped final mode.
        do i = 1, self%n_samples
            eta = encoded_label(labels(i), self%class_label)*self%mode(i)
            call likelihood_terms(eta, self%likelihood, probability, &
                likelihood_gradient, curvature)
            if (weights(i) > 0.0_dp) then
                self%sqrt_w(i) = sqrt(max(weights(i)*curvature, &
                    MIN_LIKELIHOOD_CURVATURE))
            else
                self%sqrt_w(i) = 0.0_dp
            end if
        end do
        call posterior_system(self%covariance, self%sqrt_w, matrix)
        call self%posterior_factorization%factorize(matrix, status)
        if (status%code /= FORTNUM_OK) return
        self%alpha = self%mode
        call self%prior_factorization%solve(self%alpha, status)
        if (status%code /= FORTNUM_OK) return
        result%log_posterior = log_posterior(self%mode, self%alpha, labels, &
            self%class_label, self%likelihood, weights)
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_fit

    subroutine gp_classification_predict_latent(self, x, mean, variance, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status

        call predict_core(self, x, mean, variance, status)
    end subroutine gp_classification_predict_latent

    subroutine gp_classification_predict_latent_device(self, device, x, mean, &
            variance, status)
        !! Latent prediction through the explicit device contract.
        !!
        !! The Laplace solve and covariance workspaces are not resident on a
        !! CUDA backend yet.  CPU dispatch is exact; CUDA returns a typed
        !! refusal rather than staging a hidden host computation.
        class(gp_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_latent(x, mean, variance, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP classification device prediction: no resident CUDA Laplace kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification device prediction: device kind is invalid")
        end select
    end subroutine gp_classification_predict_latent_device

    subroutine gp_classification_predict_latent_jvp( &
            self, x, x_dot, mean, mean_dot, variance, variance_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), work(:, :), work_dot(:, :)
        real(dp), allocatable :: prior(:), prior_dot(:)
        real(dp) :: value, value_dot
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        integer :: i, j

        if (.not. prediction_shapes(self, x, mean, variance, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. size(mean_dot) /= size(mean) .or. &
            size(variance_dot) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification JVP: input or output shape is invalid")
            return
        end if
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(cross_dot, mold=cross)
        allocate(work, mold=cross)
        allocate(work_dot, mold=cross)
        allocate(prior(size(x, 1)), prior_dot(size(x, 1)))
        allocate(gradient_x1(self%n_features), gradient_x2(self%n_features))
        allocate(mixed_hessian(self%n_features, self%n_features))
        do j = 1, size(x, 1)
            do i = 1, self%n_samples
                call self%kernel%input_derivatives(self%x_train(i, :), x(j, :), &
                    value, gradient_x1, gradient_x2, mixed_hessian, status)
                if (status%code /= FORTNUM_OK) return
                cross(i, j) = value
                cross_dot(i, j) = dot_product(gradient_x2, x_dot(j, :))
            end do
            call self%kernel%input_derivatives(x(j, :), x(j, :), value, &
                gradient_x1, gradient_x2, mixed_hessian, status)
            if (status%code /= FORTNUM_OK) return
            prior(j) = value
            prior_dot(j) = dot_product(gradient_x1 + gradient_x2, x_dot(j, :))
        end do
        mean = matmul(transpose(cross), self%alpha)
        mean_dot = matmul(transpose(cross_dot), self%alpha)
        call scale_rows(cross, self%sqrt_w, work)
        call scale_rows(cross_dot, self%sqrt_w, work_dot)
        call self%posterior_factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        call self%posterior_factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        variance = prior - sum(work*work, dim=1)
        variance_dot = prior_dot - 2.0_dp*sum(work*work_dot, dim=1)
        call clamp_variance(variance, status)
    end subroutine gp_classification_predict_latent_jvp

    !> Reverse-mode product of the latent posterior prediction with respect
    !> to query features.  The fitted state is held fixed.
    subroutine gp_classification_predict_latent_vjp(self, x, mean_bar, &
            variance_bar, x_bar, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), lambda(:, :)
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        real(dp) :: value, variance_weight
        real(dp), allocatable :: cross_bar(:)
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. prediction_input_valid(self, x, status)) return
        if (size(mean_bar) /= size(x, 1) .or. size(variance_bar) /= size(x, 1) .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification latent VJP: input or cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(mean_bar)) .or. &
            any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification latent VJP: cotangents must be finite")
            return
        end if

        allocate(cross(self%n_samples, size(x, 1)), work(self%n_samples, size(x, 1)), &
            lambda(self%n_samples, size(x, 1)))
        allocate(cross_bar(self%n_samples))
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(x, 1)
            work(:, j) = self%sqrt_w*cross(:, j)
        end do
        call self%posterior_factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        lambda = work
        do j = 1, size(x, 1)
            call self%posterior_factorization%solve(lambda(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        allocate(gradient_x1(self%n_features), gradient_x2(self%n_features), &
            mixed_hessian(self%n_features, self%n_features))
        do j = 1, size(x, 1)
            cross_bar = self%alpha*mean_bar(j) - 2.0_dp*self%sqrt_w*lambda(:, j)* &
                variance_bar(j)
            do i = 1, self%n_samples
                call self%kernel%input_derivatives(self%x_train(i, :), x(j, :), &
                    value, gradient_x1, gradient_x2, mixed_hessian, status)
                if (status%code /= FORTNUM_OK) return
                x_bar(j, :) = x_bar(j, :) + cross_bar(i)*gradient_x2
            end do
            call self%kernel%input_derivatives(x(j, :), x(j, :), value, &
                gradient_x1, gradient_x2, mixed_hessian, status)
            if (status%code /= FORTNUM_OK) return
            variance_weight = variance_bar(j)
            x_bar(j, :) = x_bar(j, :) + variance_weight*(gradient_x1 + gradient_x2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_latent_vjp

    !> Forward product of the latent prediction with respect to packed kernel
    !! log parameters.  The fitted Laplace state (mode, alpha, and W) is held
    !! fixed; this is the same fixed-state contract used by variational GP
    !! prediction products and avoids silently differentiating a Newton solve.
    subroutine gp_classification_predict_latent_parameter_jvp(self, x, direction, &
            mean, mean_dot, variance, variance_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i
        real(dp) :: single_prior(1, 1), single_prior_dot(1, 1)
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), prior_diagonal(:), prior_dot_diagonal(:)
        real(dp), allocatable :: train(:, :), train_dot(:, :), matrix_dot(:, :)
        real(dp), allocatable :: work(:, :), work_dot(:, :)
        if (.not. prediction_shapes(self, x, mean, variance, status)) return
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. size(mean_dot) /= size(mean) .or. &
            size(variance_dot) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification parameter JVP: direction or output shape is invalid")
            return
        end if
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(cross_dot(self%n_samples, size(x, 1)))
        allocate(prior_diagonal(size(x, 1)), prior_dot_diagonal(size(x, 1)))
        allocate(train(self%n_samples, self%n_samples))
        allocate(train_dot(self%n_samples, self%n_samples))
        allocate(matrix_dot(self%n_samples, self%n_samples))
        call self%kernel%matrix_jvp(self%x_train, x, direction, cross, cross_dot, status)
        if (status%code /= FORTNUM_OK) return
        ! Diagonal only: forming the full query-by-query prior and its tangent
        ! to read two diagonals costs `m**2` kernel evaluations for `2m`
        ! numbers.
        do i = 1, size(x, 1)
            call self%kernel%matrix_jvp(x(i:i, :), x(i:i, :), direction, &
                single_prior, single_prior_dot, status)
            if (status%code /= FORTNUM_OK) return
            prior_diagonal(i) = single_prior(1, 1)
            prior_dot_diagonal(i) = single_prior_dot(1, 1)
        end do
        call self%kernel%matrix_jvp(self%x_train, self%x_train, direction, train, &
            train_dot, status)
        if (status%code /= FORTNUM_OK) return
        allocate(work, mold=cross)
        allocate(work_dot, mold=cross)
        call scale_rows(cross, self%sqrt_w, work)
        call scale_rows(cross_dot, self%sqrt_w, work_dot)
        call self%posterior_factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        call scale_rows(train_dot, self%sqrt_w, matrix_dot)
        call scale_columns(matrix_dot, self%sqrt_w)
        work_dot = work_dot - matmul(matrix_dot, work)
        call self%posterior_factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        mean = matmul(transpose(cross), self%alpha)
        mean_dot = matmul(transpose(cross_dot), self%alpha)
        variance = prior_diagonal - sum(work*work, dim=1)
        variance_dot = prior_dot_diagonal - 2.0_dp*sum(work*work_dot, dim=1)
        call clamp_variance(variance, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(mean_dot)) .or. &
            any(.not. ieee_is_finite(variance_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification parameter JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_latent_parameter_jvp

    !> Forward prediction product through the converged Laplace fit.
    !!
    !! Unlike `predict_latent_parameter_jvp`, this method differentiates the
    !! fitted mode, likelihood curvature, posterior system, and prior solve.
    !! The training rows, labels, sample weights, likelihood kind, and Newton
    !! convergence branch remain fixed.  Kernel coordinates use the packed
    !! logarithmic layout returned by `parameters()`.
    subroutine gp_classification_predict_latent_hyperparameter_jvp(self, x, &
            direction, mean, mean_dot, variance, variance_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), train(:, :)
        real(dp), allocatable :: train_dot(:, :), prior(:), prior_dot(:)
        real(dp), allocatable :: mode_dot(:), alpha_dot(:), tangent_rhs(:)
        real(dp), allocatable :: tangent_solution(:), sqrt_w_dot(:)
        real(dp), allocatable :: work(:, :), work_dot(:, :), matrix_dot(:, :)
        real(dp) :: single_prior(1, 1), single_prior_dot(1, 1)
        real(dp) :: eta, probability, likelihood_gradient, curvature
        real(dp) :: raw_curvature, curvature_derivative, eta_dot
        integer :: i, j

        if (.not. prediction_shapes(self, x, mean, variance, status)) return
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. &
            size(mean_dot) /= size(mean) .or. &
            size(variance_dot) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification implicit prediction JVP: direction or output shape is invalid")
            return
        end if

        allocate(cross(self%n_samples, size(x, 1)))
        allocate(cross_dot(self%n_samples, size(x, 1)))
        allocate(train(self%n_samples, self%n_samples))
        allocate(train_dot(self%n_samples, self%n_samples))
        allocate(prior(size(x, 1)), prior_dot(size(x, 1)))
        allocate(mode_dot(self%n_samples), alpha_dot(self%n_samples))
        allocate(tangent_rhs(self%n_samples), tangent_solution(self%n_samples))
        allocate(sqrt_w_dot(self%n_samples))
        allocate(work(self%n_samples, size(x, 1)))
        allocate(work_dot(self%n_samples, size(x, 1)))
        allocate(matrix_dot(self%n_samples, self%n_samples))

        call self%kernel%matrix_jvp(self%x_train, x, direction, cross, cross_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix_jvp(self%x_train, self%x_train, direction, train, &
            train_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            call self%kernel%matrix_jvp(x(i:i, :), x(i:i, :), direction, &
                single_prior, single_prior_dot, status)
            if (status%code /= FORTNUM_OK) return
            prior(i) = single_prior(1, 1)
            prior_dot(i) = single_prior_dot(1, 1)
        end do

        ! Differentiate the converged mode equation using the factored
        ! I + sqrt(W) K sqrt(W) system.
        tangent_rhs = matmul(train_dot, self%alpha)
        tangent_solution = self%sqrt_w*tangent_rhs
        call self%posterior_factorization%solve(tangent_solution, status)
        if (status%code /= FORTNUM_OK) return
        mode_dot = tangent_rhs - matmul(self%covariance, &
            self%sqrt_w*tangent_solution)
        alpha_dot = mode_dot - tangent_rhs
        call self%prior_factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return

        ! The posterior variance also depends on the curvature at the fitted
        ! mode.  At the numerical curvature floor, use the fixed active-set
        ! derivative of zero.
        sqrt_w_dot = 0.0_dp
        do i = 1, self%n_samples
            if (self%sample_weights(i) <= 0.0_dp) cycle
            eta = encoded_label(self%training_labels(i), self%class_label)*self%mode(i)
            eta_dot = encoded_label(self%training_labels(i), self%class_label)*mode_dot(i)
            call likelihood_terms(eta, self%likelihood, probability, &
                likelihood_gradient, curvature)
            if (self%likelihood == GP_LIKELIHOOD_LOGISTIC) then
                raw_curvature = probability*(1.0_dp - probability)
                curvature_derivative = raw_curvature*(1.0_dp - 2.0_dp*probability)
            else
                raw_curvature = likelihood_gradient*(likelihood_gradient + eta)
                curvature_derivative = -raw_curvature*(likelihood_gradient + eta) + &
                    likelihood_gradient*(1.0_dp - raw_curvature)
            end if
            if (self%sample_weights(i)*raw_curvature > &
                MIN_LIKELIHOOD_CURVATURE) then
                sqrt_w_dot(i) = 0.5_dp*self%sample_weights(i)* &
                    curvature_derivative*eta_dot/self%sqrt_w(i)
            end if
        end do

        call scale_rows(cross, self%sqrt_w, work)
        call self%posterior_factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        call scale_rows(cross_dot, self%sqrt_w, work_dot)
        do i = 1, self%n_samples
            work_dot(i, :) = work_dot(i, :) + sqrt_w_dot(i)*cross(i, :)
        end do
        do j = 1, self%n_samples
            do i = 1, self%n_samples
                matrix_dot(i, j) = sqrt_w_dot(i)*train(i, j)*self%sqrt_w(j) + &
                    self%sqrt_w(i)*train_dot(i, j)*self%sqrt_w(j) + &
                    self%sqrt_w(i)*train(i, j)*sqrt_w_dot(j)
            end do
        end do
        work_dot = work_dot - matmul(matrix_dot, work)
        call self%posterior_factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return

        mean = matmul(transpose(cross), self%alpha)
        mean_dot = matmul(transpose(cross_dot), self%alpha) + &
            matmul(transpose(cross), alpha_dot)
        variance = prior - sum(work*work, dim=1)
        variance_dot = prior_dot - 2.0_dp*sum(work*work_dot, dim=1)
        call clamp_variance(variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(variance)
            if (variance(i) == 0.0_dp) variance_dot(i) = 0.0_dp
        end do
        if (any(.not. ieee_is_finite(mean_dot)) .or. &
            any(.not. ieee_is_finite(variance_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification implicit prediction JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_latent_hyperparameter_jvp

    !> Reverse product of the latent prediction with respect to packed kernel
    !! log parameters under the same fixed fitted-state contract as the JVP.
    subroutine gp_classification_predict_latent_parameter_vjp(self, x, mean_bar, &
            variance_bar, parameter_bar, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), lambda(:, :)
        real(dp), allocatable :: cross_bar(:, :), prior_bar(:, :), train_bar(:, :)
        real(dp), allocatable :: local_bar(:), left(:), right(:)
        integer :: i, j

        parameter_bar = 0.0_dp
        if (.not. prediction_input_valid(self, x, status)) return
        if (size(mean_bar) /= size(x, 1) .or. size(variance_bar) /= size(x, 1) .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(mean_bar)) .or. &
            any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification parameter VJP: input or cotangent shape is invalid")
            return
        end if
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(work(self%n_samples, size(x, 1)))
        allocate(lambda(self%n_samples, size(x, 1)))
        allocate(cross_bar(self%n_samples, size(x, 1)))
        allocate(train_bar(self%n_samples, self%n_samples))
        allocate(prior_bar(size(x, 1), size(x, 1)))
        allocate(local_bar(self%parameter_count()))
        allocate(left(self%n_samples), right(self%n_samples))
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        call scale_rows(cross, self%sqrt_w, work)
        call self%posterior_factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        cross_bar = spread(self%alpha, dim=2, ncopies=size(x, 1))* &
            spread(mean_bar, dim=1, ncopies=self%n_samples)
        lambda = -2.0_dp*work*spread(variance_bar, dim=1, ncopies=self%n_samples)
        call self%posterior_factorization%solve(lambda, status)
        if (status%code /= FORTNUM_OK) return
        block
            real(dp), allocatable :: lambda_scaled(:, :)
            allocate(lambda_scaled(self%n_samples, size(x, 1)))
            call scale_rows(lambda, self%sqrt_w, lambda_scaled)
            cross_bar = cross_bar + lambda_scaled
        end block
        train_bar = 0.0_dp
        do j = 1, size(x, 1)
            left = self%sqrt_w*lambda(:, j)
            right = self%sqrt_w*work(:, j)
            train_bar = train_bar - 0.5_dp*(outer_product(left, right) + &
                outer_product(right, left))
        end do
        prior_bar = 0.0_dp
        do i = 1, size(x, 1)
            prior_bar(i, i) = variance_bar(i)
        end do
        call self%kernel%parameter_vjp(self%x_train, self%x_train, train_bar, &
            local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = local_bar
        call self%kernel%parameter_vjp(self%x_train, x, cross_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = parameter_bar + local_bar
        call self%kernel%parameter_vjp(x, x, prior_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = parameter_bar + local_bar
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification parameter VJP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_latent_parameter_vjp

    !> Forward product of observed probabilities with respect to fixed-state
    !! kernel parameters.
    subroutine gp_classification_predict_proba_parameter_jvp(self, x, direction, &
            probabilities, probabilities_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        real(dp) :: p, p_mu, p_variance, scale, z, density
        integer :: i

        if (.not. prediction_probability_shapes(self, x, probabilities, status)) return
        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification probability parameter JVP: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_parameter_jvp(x, direction, mean, mean_dot, variance, &
            variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            p = predictive_probability(self%likelihood, mean(i), variance(i))
            if (self%likelihood == GP_LIKELIHOOD_LOGISTIC) then
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p_mu = p*(1.0_dp - p)/scale
                p_variance = p*(1.0_dp - p)*(-mean(i)*PI/(16.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            end if
            probabilities(i, 2) = p
            probabilities(i, 1) = 1.0_dp - p
            probabilities_dot(i, 2) = p_mu*mean_dot(i) + p_variance*variance_dot(i)
            probabilities_dot(i, 1) = -probabilities_dot(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_proba_parameter_jvp

    !> Forward probability product through the converged Laplace fit.
    subroutine gp_classification_predict_proba_hyperparameter_jvp(self, x, &
            direction, probabilities, probabilities_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        real(dp) :: p, p_mu, p_variance, scale, z, density
        integer :: i

        if (.not. prediction_probability_shapes(self, x, probabilities, status)) return
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification implicit probability JVP: direction or output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_hyperparameter_jvp(x, direction, mean, mean_dot, &
            variance, variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            p = predictive_probability(self%likelihood, mean(i), variance(i))
            if (self%likelihood == GP_LIKELIHOOD_LOGISTIC) then
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p_mu = p*(1.0_dp - p)/scale
                p_variance = p*(1.0_dp - p)*(-mean(i)*PI/(16.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            end if
            probabilities(i, 2) = p
            probabilities(i, 1) = 1.0_dp - p
            probabilities_dot(i, 2) = p_mu*mean_dot(i) + &
                p_variance*variance_dot(i)
            probabilities_dot(i, 1) = -probabilities_dot(i, 2)
        end do
        if (any(.not. ieee_is_finite(probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification implicit probability JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_proba_hyperparameter_jvp

    !> Device boundary for the implicit-fit probability product.
    subroutine gp_classification_predict_proba_hyperparameter_jvp_device(self, &
            device, x, direction, probabilities, probabilities_dot, status)
        class(gp_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification implicit probability JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_hyperparameter_jvp(x, direction, probabilities, &
                probabilities_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP classification implicit probability JVP device: no resident CUDA Laplace graph is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification implicit probability JVP device: device kind is invalid")
        end select
    end subroutine gp_classification_predict_proba_hyperparameter_jvp_device

    !> Forward kernel-hyperparameter product of ``predict_log_proba``.
    subroutine gp_classification_predict_log_proba_parameter_jvp(self, x, direction, &
            log_probabilities, log_probabilities_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)
        integer :: i, j

        if (.not. prediction_probability_shapes(self, x, log_probabilities, status)) return
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification log probability parameter JVP: direction or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), 2), probabilities_dot(size(x, 1), 2))
        call self%predict_proba_parameter_jvp(x, direction, probabilities, probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, 2
            do i = 1, size(x, 1)
                log_probabilities(i, j) = log(max(probabilities(i, j), tiny(1.0_dp)))
                if (probabilities(i, j) > tiny(1.0_dp)) then
                    log_probabilities_dot(i, j) = probabilities_dot(i, j)/probabilities(i, j)
                else
                    log_probabilities_dot(i, j) = 0.0_dp
                end if
            end do
        end do
        if (any(.not. ieee_is_finite(log_probabilities)) .or. &
            any(.not. ieee_is_finite(log_probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification log probability parameter JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_log_proba_parameter_jvp

    !> Reverse product of observed probabilities with respect to fixed-state
    !! kernel parameters.
    subroutine gp_classification_predict_proba_parameter_vjp(self, x, &
            probabilities_bar, parameter_bar, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)
        real(dp) :: probability, p_mu, p_variance, scale, z, density
        integer :: i

        parameter_bar = 0.0_dp
        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(probabilities_bar) /= [size(x, 1), 2]) .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification probability parameter VJP: input is invalid")
            return
        end if
        allocate(mean(size(x, 1)), variance(size(x, 1)), mean_bar(size(x, 1)), &
            variance_bar(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probability = predictive_probability(self%likelihood, mean(i), variance(i))
            if (self%likelihood == GP_LIKELIHOOD_LOGISTIC) then
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p_mu = probability*(1.0_dp - probability)/scale
                p_variance = probability*(1.0_dp - probability)*(-mean(i)*PI/ &
                    (16.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            end if
            mean_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*p_mu
            variance_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*p_variance
        end do
        call self%predict_latent_parameter_vjp(x, mean_bar, variance_bar, parameter_bar, &
            status)
    end subroutine gp_classification_predict_proba_parameter_vjp

    !> Reverse kernel-hyperparameter product of ``predict_log_proba``.
    subroutine gp_classification_predict_log_proba_parameter_vjp(self, x, &
            log_probabilities_bar, parameter_bar, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probability_bar(:, :)
        integer :: i, j

        parameter_bar = 0.0_dp
        if (.not. prediction_probability_shapes(self, x, log_probabilities_bar, status)) return
        if (size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification log probability parameter VJP: input or cotangent is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), 2), probability_bar(size(x, 1), 2))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, 2
            do i = 1, size(x, 1)
                probability_bar(i, j) = log_probabilities_bar(i, j) / &
                    max(probabilities(i, j), tiny(1.0_dp))
            end do
        end do
        if (any(.not. ieee_is_finite(probability_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification log probability parameter VJP: cotangent is not finite")
            return
        end if
        call self%predict_proba_parameter_vjp(x, probability_bar, parameter_bar, status)
    end subroutine gp_classification_predict_log_proba_parameter_vjp

    subroutine gp_classification_predict_proba(self, x, probabilities, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        integer :: i

        if (.not. prediction_probability_shapes(self, x, probabilities, status)) return
        allocate(mean(size(x, 1)), variance(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities(i, 2) = predictive_probability(self%likelihood, mean(i), variance(i))
            probabilities(i, 1) = 1.0_dp - probabilities(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_proba

    subroutine gp_classification_predict_proba_device(self, device, x, &
            probabilities, status)
        !! Observed-probability prediction through the device contract.
        class(gp_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP classification device prediction: no resident CUDA Laplace kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification device prediction: device kind is invalid")
        end select
    end subroutine gp_classification_predict_proba_device

    !> Return the natural logarithm of the two predictive probabilities.
    !!
    !! This is the scikit-learn ``predict_log_proba`` companion to
    !! ``predict_proba``.  The returned columns retain ``classes()`` order;
    !! the fitted Laplace state is held fixed just as for the probability
    !! products.  A finite floor is used only at the floating-point boundary
    !! where a probit tail rounds to zero, so callers never receive ``-Inf``.
    subroutine gp_classification_predict_log_proba(self, x, log_probabilities, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. prediction_probability_shapes(self, x, log_probabilities, status)) return
        allocate(probabilities(size(x, 1), 2))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        log_probabilities = log(max(probabilities, tiny(1.0_dp)))
        if (any(.not. ieee_is_finite(log_probabilities))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification log probability: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_log_proba

    !> Device-dispatched log-probability prediction with no hidden host fallback.
    subroutine gp_classification_predict_log_proba_device(self, device, x, &
            log_probabilities, status)
        class(gp_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification log probability device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_log_proba(x, log_probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP classification log probability device: no resident CUDA Laplace kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification log probability device: device kind is invalid")
        end select
    end subroutine gp_classification_predict_log_proba_device

    subroutine gp_classification_predict_proba_jvp( &
            self, x, x_dot, probabilities, probabilities_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        real(dp) :: p, p_mu, p_variance, scale, z, density
        integer :: i

        if (.not. prediction_probability_shapes(self, x, probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. any(shape(probabilities_dot) /= &
            shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification probability JVP: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_jvp(x, x_dot, mean, mean_dot, variance, &
            variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            p = predictive_probability(self%likelihood, mean(i), variance(i))
            if (self%likelihood == GP_LIKELIHOOD_LOGISTIC) then
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                z = mean(i)/scale
                p_mu = p*(1.0_dp - p)/scale
                p_variance = p*(1.0_dp - p)*(-mean(i)*PI/(16.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            end if
            probabilities(i, 2) = p
            probabilities(i, 1) = 1.0_dp - p
            probabilities_dot(i, 2) = p_mu*mean_dot(i) + p_variance*variance_dot(i)
            probabilities_dot(i, 1) = -probabilities_dot(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_proba_jvp

    !> Forward input product of ``predict_log_proba``.
    subroutine gp_classification_predict_log_proba_jvp(self, x, x_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)
        integer :: i, j

        if (.not. prediction_probability_shapes(self, x, log_probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification log probability JVP: input or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), 2), probabilities_dot(size(x, 1), 2))
        call self%predict_proba_jvp(x, x_dot, probabilities, probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, 2
            do i = 1, size(x, 1)
                log_probabilities(i, j) = log(max(probabilities(i, j), tiny(1.0_dp)))
                if (probabilities(i, j) > tiny(1.0_dp)) then
                    log_probabilities_dot(i, j) = probabilities_dot(i, j)/probabilities(i, j)
                else
                    log_probabilities_dot(i, j) = 0.0_dp
                end if
            end do
        end do
        if (any(.not. ieee_is_finite(log_probabilities)) .or. &
            any(.not. ieee_is_finite(log_probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification log probability JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_log_proba_jvp

    !> Reverse-mode product of observed probabilities with respect to query
    !> features.  Probability columns are ordered as ``classes()``.
    subroutine gp_classification_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)
        real(dp) :: probability, p_mu, p_variance, scale, z, density, cotangent
        integer :: i

        x_bar = 0.0_dp
        if (.not. prediction_probability_shapes(self, x, probabilities_bar, status)) then
            return
        end if
        if (any(.not. ieee_is_finite(probabilities_bar)) .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification probability VJP: input or cotangent is invalid")
            return
        end if
        allocate(mean(size(x, 1)), variance(size(x, 1)), mean_bar(size(x, 1)), &
            variance_bar(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probability = predictive_probability(self%likelihood, mean(i), variance(i))
            if (self%likelihood == GP_LIKELIHOOD_LOGISTIC) then
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p_mu = probability*(1.0_dp - probability)/scale
                p_variance = probability*(1.0_dp - probability)*(-mean(i)*PI/ &
                    (16.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            end if
            cotangent = probabilities_bar(i, 2) - probabilities_bar(i, 1)
            mean_bar(i) = cotangent*p_mu
            variance_bar(i) = cotangent*p_variance
        end do
        call self%predict_latent_vjp(x, mean_bar, variance_bar, x_bar, status)
    end subroutine gp_classification_predict_proba_vjp

    !> Reverse input product of ``predict_log_proba``.
    subroutine gp_classification_predict_log_proba_vjp(self, x, log_probabilities_bar, &
            x_bar, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probability_bar(:, :)
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. prediction_probability_shapes(self, x, log_probabilities_bar, status)) return
        if (any(.not. ieee_is_finite(log_probabilities_bar)) .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification log probability VJP: input or cotangent is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), 2), probability_bar(size(x, 1), 2))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, 2
            do i = 1, size(x, 1)
                probability_bar(i, j) = log_probabilities_bar(i, j) / &
                    max(probabilities(i, j), tiny(1.0_dp))
            end do
        end do
        if (any(.not. ieee_is_finite(probability_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification log probability VJP: cotangent is not finite")
            return
        end if
        call self%predict_proba_vjp(x, probability_bar, x_bar, status)
    end subroutine gp_classification_predict_log_proba_vjp

    subroutine gp_classification_predict(self, x, labels, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. prediction_input_valid(self, x, status)) return
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: label shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), 2))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            if (probabilities(i, 2) > probabilities(i, 1)) then
                labels(i) = self%class_label(2)
            else
                labels(i) = self%class_label(1)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict

    logical function gp_classification_likelihood_device_supported(device_kind) result(supported)
        !! Report device support for the scalar likelihood products.
        !!
        !! The value, JVP, and VJP routines are backend-independent host
        !! references.  No resident CUDA reduction is linked, and derivative
        !! products must not silently execute on the host for a CUDA request.
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = .true.
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function gp_classification_likelihood_device_supported

    !> Evaluate the sum of log likelihoods for signed latent margins.
    !!
    !! ``eta(i)`` is the latent function value multiplied by the encoded
    !! class label (``+1`` for the positive class and ``-1`` for the negative
    !! class).  The result is therefore the Bernoulli log likelihood for the
    !! selected Laplace likelihood, without the GP prior term.  Keeping this
    !! scalar product public gives training objectives and hyperparameter
    !! search a common analytic likelihood primitive instead of duplicating
    !! logistic/probit tail handling.
    subroutine gp_classification_log_likelihood_value(eta, likelihood, value, status)
        real(dp), intent(in) :: eta(:)
        integer, intent(in) :: likelihood
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_probability, derivative, curvature
        integer :: i

        value = 0.0_dp
        if (.not. valid_likelihood(likelihood) .or. size(eta) < 1 .or. &
            any(.not. ieee_is_finite(eta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification likelihood: inputs are invalid")
            return
        end if
        do i = 1, size(eta)
            call likelihood_log_terms(eta(i), likelihood, log_probability, &
                derivative, curvature)
            value = value + log_probability
        end do
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification likelihood: value is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_log_likelihood_value

    !> Forward product of the signed-margin log likelihood.
    subroutine gp_classification_log_likelihood_jvp(eta, likelihood, eta_dot, &
            value, value_dot, status)
        real(dp), intent(in) :: eta(:), eta_dot(:)
        integer, intent(in) :: likelihood
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_probability, derivative, curvature
        integer :: i

        value = 0.0_dp
        value_dot = 0.0_dp
        if (size(eta_dot) /= size(eta) .or. any(.not. ieee_is_finite(eta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification likelihood JVP: tangent is invalid")
            return
        end if
        if (.not. valid_likelihood(likelihood) .or. size(eta) < 1 .or. &
            any(.not. ieee_is_finite(eta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification likelihood JVP: inputs are invalid")
            return
        end if
        do i = 1, size(eta)
            call likelihood_log_terms(eta(i), likelihood, log_probability, &
                derivative, curvature)
            value = value + log_probability
            value_dot = value_dot + derivative*eta_dot(i)
        end do
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(value_dot)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification likelihood JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_log_likelihood_jvp

    !> Reverse product of the signed-margin log likelihood.
    subroutine gp_classification_log_likelihood_vjp(eta, likelihood, value_bar, &
            eta_bar, status)
        real(dp), intent(in) :: eta(:)
        integer, intent(in) :: likelihood
        real(dp), intent(in) :: value_bar
        real(dp), intent(out) :: eta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_probability, derivative, curvature
        integer :: i

        eta_bar = 0.0_dp
        if (size(eta_bar) /= size(eta) .or. .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification likelihood VJP: cotangent is invalid")
            return
        end if
        if (.not. valid_likelihood(likelihood) .or. size(eta) < 1 .or. &
            any(.not. ieee_is_finite(eta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification likelihood VJP: inputs are invalid")
            return
        end if
        do i = 1, size(eta)
            call likelihood_log_terms(eta(i), likelihood, log_probability, &
                derivative, curvature)
            eta_bar(i) = value_bar*derivative
        end do
        if (any(.not. ieee_is_finite(eta_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification likelihood VJP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_log_likelihood_vjp

    function gp_classification_classes(self) result(classes)
        class(gp_classification_t), intent(in) :: self
        integer :: classes(2)

        classes = self%class_label
    end function gp_classification_classes

    integer function gp_classification_feature_count(self) result(count)
        class(gp_classification_t), intent(in) :: self

        count = self%n_features
    end function gp_classification_feature_count

    integer function gp_classification_parameter_count(self) result(count)
        class(gp_classification_t), intent(in) :: self

        count = 0
        if (.not. self%fitted()) return
        count = self%kernel%parameter_count()
    end function gp_classification_parameter_count

    function gp_classification_parameters(self) result(parameters)
        class(gp_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (.not. self%fitted()) then
            allocate(parameters(0))
            return
        end if
        parameters = self%kernel%parameters()
    end function gp_classification_parameters

    !> Update kernel log parameters while holding the fitted Laplace mode fixed.
    !!
    !! Prediction JVP/VJP products use this same fixed-state contract.  The
    !! covariance and both Cholesky factors are rebuilt transactionally for
    !! the new kernel, making central-difference checks and outer HPO callers
    !! agree on which state is differentiated.  ``fit`` remains the API that
    !! recomputes the Newton mode.
    subroutine gp_classification_set_parameters(self, parameters, status)
        class(gp_classification_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: candidate
        type(cholesky_factorization_t) :: prior_candidate, posterior_candidate
        real(dp), allocatable :: covariance(:, :), matrix(:, :)
        integer :: i

        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification set_parameters: model is not fitted")
            return
        end if
        if (size(parameters) /= self%kernel%parameter_count() .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification set_parameters: parameter shape or values are invalid")
            return
        end if
        candidate = clone_kernel(self%kernel)
        call candidate%set_parameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        allocate(covariance(self%n_samples, self%n_samples))
        call candidate%matrix(self%x_train, self%x_train, covariance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%n_samples
            covariance(i, i) = covariance(i, i) + self%jitter
        end do
        call prior_candidate%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(matrix(self%n_samples, self%n_samples))
        call posterior_system(covariance, self%sqrt_w, matrix)
        call posterior_candidate%factorize(matrix, status)
        if (status%code /= FORTNUM_OK) return
        self%kernel = candidate
        self%covariance = covariance
        self%prior_factorization = prior_candidate
        self%posterior_factorization = posterior_candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_set_parameters

    !> Evaluate the mode log posterior after a kernel transaction.
    !!
    !! The Newton mode itself, and therefore the likelihood contribution, is
    !! held fixed.  The prior term is evaluated with the current kernel
    !! factorization, so callers can use this routine as a smooth objective
    !! while changing the packed logarithmic kernel parameters with
    !! `set_parameters`.
    subroutine gp_classification_fixed_state_log_posterior(self, value, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: alpha_candidate(:)

        value = -huge(1.0_dp)
        if (.not. self%fitted() .or. .not. allocated(self%training_labels) .or. &
            .not. allocated(self%sample_weights)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fixed-state objective: model is not fitted")
            return
        end if
        allocate(alpha_candidate(self%n_samples))
        alpha_candidate = self%mode
        call self%prior_factorization%solve(alpha_candidate, status)
        if (status%code /= FORTNUM_OK) return
        value = log_posterior(self%mode, alpha_candidate, self%training_labels, &
            self%class_label, self%likelihood, self%sample_weights)
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification fixed-state objective: value is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_fixed_state_log_posterior

    !> Gradient of `fixed_state_log_posterior` with respect to kernel logs.
    !!
    !! This is the exact prior-envelope contraction at the fixed mode.  The
    !! candidate inverse-kernel solve is intentional: after a transactional
    !! `set_parameters`, `alpha = K(theta)^(-1) f_mode` changes even though
    !! the mode and likelihood state remain fixed.
    subroutine gp_classification_fixed_state_log_posterior_gradient(self, gradient, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: alpha_candidate(:), matrix_bar(:, :)

        gradient = 0.0_dp
        if (.not. self%fitted() .or. .not. allocated(self%training_labels) .or. &
            .not. allocated(self%sample_weights)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fixed-state gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fixed-state gradient: output shape is invalid")
            return
        end if
        allocate(alpha_candidate(self%n_samples), matrix_bar(self%n_samples, self%n_samples))
        alpha_candidate = self%mode
        call self%prior_factorization%solve(alpha_candidate, status)
        if (status%code /= FORTNUM_OK) return
        matrix_bar = 0.5_dp*outer_product(alpha_candidate, alpha_candidate)
        call self%kernel%parameter_vjp(self%x_train, self%x_train, matrix_bar, gradient, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification fixed-state gradient: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_fixed_state_log_posterior_gradient

    subroutine gp_classification_hyperparameter_gradient(self, gradient, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: matrix_bar(:, :)
        integer :: i, j

        gradient = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification hyperparameter gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification hyperparameter gradient: output shape is invalid")
            return
        end if
        ! The reported objective is the Laplace mode log posterior without
        ! the optional evidence correction.  At a converged mode the envelope
        ! theorem removes the implicit mode derivative, leaving
        ! d log p(y,f_mode | theta) / d theta =
        ! 1/2 * alpha^T (dK/dtheta) alpha, where alpha = K^{-1} f_mode.
        ! The kernel VJP supplies this contraction for every supported kernel
        ! expression, including sums and products.
        allocate(matrix_bar(self%n_samples, self%n_samples))
        do j = 1, self%n_samples
            do i = 1, self%n_samples
                matrix_bar(i, j) = 0.5_dp*self%alpha(i)*self%alpha(j)
            end do
        end do
        call self%kernel%parameter_vjp(self%x_train, self%x_train, matrix_bar, &
            gradient, status)
    end subroutine gp_classification_hyperparameter_gradient

    !> Directional hyperparameter Hessian product of the fitted Laplace
    !! mode-posterior envelope.  Unlike `set_parameters`, which is a
    !! fixed-state transaction for prediction products, this operation
    !! differentiates the converged mode implicitly.  The mode tangent is
    !! obtained from `(K^{-1}+W) f_dot = K^{-1} K_dot alpha` using the already
    !! factored posterior system, and the kernel HVP/VJP primitives then form
    !! the exact smooth product for every kernel with generated products.
    subroutine gp_classification_hyperparameter_hvp(self, direction, parameter_hvp, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance_dot(:, :), matrix_bar(:, :), matrix_bar_dot(:, :)
        real(dp), allocatable :: alpha_dot(:), mode_dot(:), kernel_direction(:)
        real(dp), allocatable :: tangent_rhs(:), tangent_solution(:)
        real(dp), allocatable :: local_bar(:), local_bar_dot(:)

        parameter_hvp = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification hyperparameter HVP: model is not fitted")
            return
        end if
        if (size(direction) /= self%parameter_count() .or. &
            size(parameter_hvp) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification hyperparameter HVP: parameter shape is invalid")
            return
        end if

        allocate(covariance_dot(self%n_samples, self%n_samples))
        allocate(mode_dot(self%n_samples), alpha_dot(self%n_samples))
        allocate(tangent_rhs(self%n_samples), tangent_solution(self%n_samples))
        allocate(matrix_bar(self%n_samples, self%n_samples))
        allocate(matrix_bar_dot(self%n_samples, self%n_samples))
        allocate(local_bar(self%parameter_count()), local_bar_dot(self%parameter_count()))
        allocate(kernel_direction(size(direction)))
        kernel_direction = direction

        ! `matrix_jvp` returns the value and directional derivative.  The
        ! value is intentionally discarded here; the fitted covariance,
        ! including jitter, is already the transactional state used by the
        ! Cholesky factors.
        block
            real(dp), allocatable :: covariance(:, :)
            allocate(covariance(self%n_samples, self%n_samples))
            call self%kernel%matrix_jvp(self%x_train, self%x_train, kernel_direction, &
                covariance, covariance_dot, status)
        end block
        if (status%code /= FORTNUM_OK) return

        ! Let t = K_dot alpha.  The implicit mode equation can be solved
        ! without forming K^{-1}+W explicitly: with u = sqrt(W) f_dot,
        ! (I + sqrt(W) K sqrt(W)) u = sqrt(W) t and
        ! f_dot = t - K sqrt(W) u.
        tangent_rhs = matmul(covariance_dot, self%alpha)
        tangent_solution = self%sqrt_w*tangent_rhs
        call self%posterior_factorization%solve(tangent_solution, status)
        if (status%code /= FORTNUM_OK) return
        mode_dot = tangent_rhs - matmul(self%covariance, self%sqrt_w*tangent_solution)

        ! alpha = K^{-1} f_mode, so alpha_dot = K^{-1}(f_dot-K_dot alpha).
        alpha_dot = mode_dot - tangent_rhs
        call self%prior_factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return

        matrix_bar = 0.5_dp*outer_product(self%alpha, self%alpha)
        matrix_bar_dot = 0.5_dp*(outer_product(alpha_dot, self%alpha) + &
            outer_product(self%alpha, alpha_dot))
        call self%kernel%parameter_hvp(self%x_train, self%x_train, matrix_bar, &
            direction, local_bar, local_bar_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%parameter_vjp(self%x_train, self%x_train, matrix_bar_dot, &
            local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_hvp = local_bar + local_bar_dot
        if (any(.not. ieee_is_finite(parameter_hvp))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification hyperparameter HVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_hyperparameter_hvp

    !> Device-dispatched hyperparameter HVP with no hidden host fallback.
    subroutine gp_classification_hyperparameter_hvp_device(self, device, direction, &
            parameter_hvp, status)
        class(gp_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_hvp = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification hyperparameter HVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hyperparameter_hvp(direction, parameter_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP classification hyperparameter HVP device: no resident CUDA Laplace kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification hyperparameter HVP device: device kind is invalid")
        end select
    end subroutine gp_classification_hyperparameter_hvp_device

    logical function gp_classification_fitted(self) result(fitted)
        class(gp_classification_t), intent(in) :: self

        fitted = allocated(self%x_train) .and. allocated(self%covariance) .and. &
            allocated(self%mode) .and. allocated(self%alpha) .and. &
            allocated(self%sqrt_w) .and. self%n_samples > 0 .and. self%n_features > 0
    end function gp_classification_fitted

    logical function gp_classification_device_supported(self, device_kind) result(supported)
        !! Report support without inferring a host fallback for accelerators.
        class(gp_classification_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%fitted()
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function gp_classification_device_supported

    integer function gp_classification_likelihood(self) result(kind)
        class(gp_classification_t), intent(in) :: self

        kind = self%likelihood
    end function gp_classification_likelihood

    subroutine predict_core(self, x, mean, variance, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), prior(:)
        integer :: i, j

        if (.not. prediction_shapes(self, x, mean, variance, status)) return
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(work, mold=cross)
        allocate(prior(size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, self%n_samples
                cross(i, j) = self%kernel%value(self%x_train(i, :), x(j, :))
            end do
            prior(j) = self%kernel%value(x(j, :), x(j, :))
        end do
        mean = matmul(transpose(cross), self%alpha)
        call scale_rows(cross, self%sqrt_w, work)
        call self%posterior_factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        variance = prior - sum(work*work, dim=1)
        call clamp_variance(variance, status)
    end subroutine predict_core

    logical function prediction_shapes(self, x, mean, variance, status) result(valid)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            size(mean) /= size(x, 1) .or. &
            size(variance) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: input or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_shapes

    logical function prediction_probability_shapes(self, x, probabilities, status) &
            result(valid)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), 2]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: input or probability shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_probability_shapes

    subroutine clamp_variance(variance, status)
        real(dp), intent(inout) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        do i = 1, size(variance)
            if (.not. ieee_is_finite(variance(i))) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "GP classification prediction: variance is not finite")
                return
            end if
            if (variance(i) < 0.0_dp) then
                if (variance(i) < -1.0e-8_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "GP classification prediction: variance is not positive")
                    return
                end if
                variance(i) = 0.0_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine clamp_variance

    subroutine posterior_system(covariance, sqrt_w, matrix)
        real(dp), intent(in) :: covariance(:, :), sqrt_w(:)
        real(dp), intent(out) :: matrix(:, :)
        integer :: i, j

        matrix = 0.0_dp
        do i = 1, size(matrix, 1)
            matrix(i, i) = 1.0_dp
            do j = 1, size(matrix, 2)
                matrix(i, j) = matrix(i, j) + sqrt_w(i)*covariance(i, j)*sqrt_w(j)
            end do
        end do
    end subroutine posterior_system

    subroutine scale_rows(matrix, weights, scaled)
        real(dp), intent(in) :: matrix(:, :), weights(:)
        real(dp), intent(out) :: scaled(:, :)
        integer :: i

        do i = 1, size(matrix, 1)
            scaled(i, :) = weights(i)*matrix(i, :)
        end do
    end subroutine scale_rows

    subroutine scale_columns(matrix, weights)
        real(dp), intent(inout) :: matrix(:, :)
        real(dp), intent(in) :: weights(:)
        integer :: j

        do j = 1, size(matrix, 2)
            matrix(:, j) = weights(j)*matrix(:, j)
        end do
    end subroutine scale_columns

    function diagonal_matrix(matrix) result(values)
        real(dp), intent(in) :: matrix(:, :)
        real(dp) :: values(min(size(matrix, 1), size(matrix, 2)))
        integer :: i

        do i = 1, size(values)
            values(i) = matrix(i, i)
        end do
    end function diagonal_matrix

    function outer_product(left, right) result(matrix)
        real(dp), intent(in) :: left(:), right(:)
        real(dp) :: matrix(size(left), size(right))
        integer :: i

        do i = 1, size(left)
            matrix(i, :) = left(i)*right
        end do
    end function outer_product

    subroutine likelihood_terms(eta, likelihood, probability, gradient, curvature)
        real(dp), intent(in) :: eta
        integer, intent(in) :: likelihood
        real(dp), intent(out) :: probability, gradient, curvature
        real(dp) :: ratio, density

        if (likelihood == GP_LIKELIHOOD_LOGISTIC) then
            probability = stable_sigmoid(eta)
            gradient = 1.0_dp - probability
            curvature = max(probability*(1.0_dp - probability), &
                MIN_LIKELIHOOD_CURVATURE)
        else
            probability = normal_cdf(eta)
            density = exp(-0.5_dp*eta*eta)/SQRT_TWO_PI
            if (probability > 1.0e-14_dp) then
                ratio = density/probability
            else
                ratio = max(1.0_dp, -eta) + 1.0_dp/max(1.0_dp, -eta)
            end if
            gradient = ratio
            curvature = max(ratio*(ratio + eta), MIN_LIKELIHOOD_CURVATURE)
        end if
    end subroutine likelihood_terms

    subroutine likelihood_log_terms(eta, likelihood, log_probability, gradient, curvature)
        real(dp), intent(in) :: eta
        integer, intent(in) :: likelihood
        real(dp), intent(out) :: log_probability, gradient, curvature
        real(dp) :: probability

        call likelihood_terms(eta, likelihood, probability, gradient, curvature)
        if (likelihood == GP_LIKELIHOOD_LOGISTIC) then
            ! log(sigmoid(eta)) without overflow for either sign of eta.
            if (eta >= 0.0_dp) then
                log_probability = -log(1.0_dp + exp(-eta))
            else
                log_probability = eta - log(1.0_dp + exp(eta))
            end if
        else
            log_probability = log_normal_cdf_stable(eta)
        end if
    end subroutine likelihood_log_terms

    real(dp) function log_normal_cdf_stable(value) result(log_probability)
        real(dp), intent(in) :: value
        real(dp), parameter :: LOG_SQRT_TWO_PI = 0.91893853320467274178032973640562_dp
        real(dp) :: inverse_square, correction

        if (value > -8.0_dp) then
            log_probability = log(max(normal_cdf(value), tiny(1.0_dp)))
            return
        end if
        ! Mills-ratio expansion, retaining two terms, avoids erfc underflow in
        ! the negative probit tail while agreeing smoothly with the direct CDF
        ! branch over the transition used by the Laplace solver.
        inverse_square = 1.0_dp/(value*value)
        correction = 1.0_dp - inverse_square + 3.0_dp*inverse_square*inverse_square
        log_probability = -0.5_dp*value*value - log(-value) - &
            LOG_SQRT_TWO_PI + log(max(correction, tiny(1.0_dp)))
    end function log_normal_cdf_stable

    logical function valid_likelihood(likelihood) result(valid)
        integer, intent(in) :: likelihood

        valid = likelihood == GP_LIKELIHOOD_LOGISTIC .or. &
            likelihood == GP_LIKELIHOOD_PROBIT
    end function valid_likelihood

    real(dp) function predictive_probability(likelihood, mean, variance) result(probability)
        integer, intent(in) :: likelihood
        real(dp), intent(in) :: mean, variance
        real(dp) :: scale

        if (likelihood == GP_LIKELIHOOD_LOGISTIC) then
            scale = sqrt(1.0_dp + PI*variance/8.0_dp)
            probability = stable_sigmoid(mean/scale)
        else
            scale = sqrt(1.0_dp + variance)
            probability = normal_cdf(mean/scale)
        end if
    end function predictive_probability

    real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        real(dp) :: exponential

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            exponential = exp(value)
            probability = exponential/(1.0_dp + exponential)
        end if
    end function stable_sigmoid

    real(dp) function normal_cdf(value) result(probability)
        real(dp), intent(in) :: value

        probability = 0.5_dp*erfc(-value/SQRT_TWO)
        probability = min(1.0_dp, max(0.0_dp, probability))
    end function normal_cdf

    real(dp) function encoded_label(label, classes) result(value)
        integer, intent(in) :: label, classes(2)

        value = merge(1.0_dp, -1.0_dp, label == classes(2))
    end function encoded_label

    real(dp) function log_posterior(mode, alpha, labels, classes, likelihood, weights) result(value)
        real(dp), intent(in) :: mode(:), alpha(:)
        integer, intent(in) :: labels(:), classes(2), likelihood
        real(dp), intent(in) :: weights(:)
        real(dp) :: eta, probability, gradient, curvature
        integer :: i

        value = -0.5_dp*dot_product(mode, alpha)
        do i = 1, size(mode)
            eta = encoded_label(labels(i), classes)*mode(i)
            call likelihood_terms(eta, likelihood, probability, gradient, curvature)
            value = value + weights(i)*log(max(probability, tiny(1.0_dp)))
        end do
    end function log_posterior

    subroutine build_sample_weights(n_samples, sample_weight, weights, status)
        integer, intent(in) :: n_samples
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable, intent(out) :: weights(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: weight_mass

        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP classification fit: sample weights are invalid")
                return
            end if
            weight_mass = sum(sample_weight)
            if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP classification fit: sample weights need positive mass")
                return
            end if
            weights = sample_weight
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_sample_weights

    logical function valid_options(options) result(valid)
        type(gp_classification_options_t), intent(in) :: options

        valid = (options%likelihood == GP_LIKELIHOOD_LOGISTIC .or. &
            options%likelihood == GP_LIKELIHOOD_PROBIT) .and. &
            options%max_iterations >= 1 .and. &
            ieee_is_finite(options%tolerance) .and. options%tolerance > 0.0_dp .and. &
            ieee_is_finite(options%jitter) .and. options%jitter >= 0.0_dp .and. &
            ieee_is_finite(options%damping) .and. options%damping > 0.0_dp .and. &
            options%damping <= 1.0_dp
    end function valid_options

    logical function prediction_input_valid(self, x, status) result(valid)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: input dimension is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_input_valid

end module fortml_gp_classification
