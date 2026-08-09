module fortml_gaussian_process
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_gp_mean, only: gp_mean_t, make_zero_mean
    implicit none
    private

    real(dp), parameter :: LOG_TWO_PI = 1.837877066409345483560659472811_dp

    type, public :: gp_regression_t
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: y_train(:, :)
        real(dp), allocatable :: alpha(:, :)
        real(dp) :: log_noise_variance = log(1.0e-6_dp)
        real(dp) :: jitter = 1.0e-10_dp
        type(gp_mean_t) :: mean
        real(dp), allocatable :: mean_coefficients(:)
        integer :: n_samples = 0
        integer :: n_features = 0
        integer :: n_outputs = 0
    contains
        procedure, public :: fit => gp_fit
        procedure, public :: parameter_count => gp_parameter_count
        procedure, public :: parameters => gp_parameters
        procedure, public :: set_parameters => gp_set_parameters
        procedure, public :: mean_parameter_count => gp_mean_parameter_count
        procedure, public :: mean_parameters => gp_mean_parameters
        procedure, public :: predict => gp_predict
        procedure, public :: predict_covariance => gp_predict_covariance
        procedure, public :: predict_covariance_device => gp_predict_covariance_device
        procedure, public :: predict_covariance_jvp => gp_predict_covariance_jvp
        procedure, public :: predict_covariance_vjp => gp_predict_covariance_vjp
        procedure, public :: predict_covariance_jvp_device => &
            gp_predict_covariance_jvp_device
        procedure, public :: predict_covariance_vjp_device => &
            gp_predict_covariance_vjp_device
        procedure, public :: predict_jvp => gp_predict_jvp
        procedure, public :: predict_vjp => gp_predict_vjp
        procedure, public :: predict_hvp => gp_predict_hvp
        procedure, public :: log_marginal_likelihood => gp_log_marginal_likelihood
        procedure, public :: hyperparameter_gradient => gp_hyperparameter_gradient
        procedure, public :: hyperparameter_hvp => gp_hyperparameter_hvp
        procedure, public :: log_marginal_likelihood_jvp => gp_lml_jvp
    end type gp_regression_t

    public :: gp_fit
    public :: gp_predict
    public :: gp_predict_covariance
    public :: gp_predict_covariance_device
    public :: gp_predict_covariance_jvp
    public :: gp_predict_covariance_vjp
    public :: gp_predict_covariance_jvp_device
    public :: gp_predict_covariance_vjp_device
    public :: gp_predict_jvp
    public :: gp_predict_vjp
    public :: gp_predict_hvp
    public :: gp_log_marginal_likelihood
    public :: gp_hyperparameter_gradient
    public :: gp_hyperparameter_hvp

contains

    subroutine gp_fit(self, x, y, kernel, noise_variance, status, jitter, mean)
        class(gp_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: jitter
        type(gp_mean_t), intent(in), optional :: mean
        type(gp_mean_t) :: zero_mean
        integer :: mean_count, output_index, base_count

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(y, 1) /= size(x, 1) .or. &
            size(y, 2) < 1 .or. noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP fit: invalid data dimensions or noise variance")
            return
        end if
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP fit: kernel input dimension or parameters are invalid")
            return
        end if

        self%kernel = kernel
        allocate(self%x_train(size(x, 1), size(x, 2)))
        allocate(self%y_train(size(y, 1), size(y, 2)))
        self%x_train = x
        self%y_train = y
        self%n_samples = size(x, 1)
        self%n_features = size(x, 2)
        self%n_outputs = size(y, 2)
        self%log_noise_variance = log(noise_variance)
        self%jitter = 1.0e-10_dp
        if (present(jitter)) self%jitter = jitter
        if (self%jitter < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP fit: jitter must be nonnegative")
            return
        end if
        if (present(mean)) then
            call mean%validate(size(x, 2), status)
            if (status%code /= FORTNUM_OK) return
            self%mean = mean
        else
            zero_mean = make_zero_mean(size(x, 2), status)
            if (status%code /= FORTNUM_OK) return
            self%mean = zero_mean
        end if
        base_count = self%mean%parameter_count()
        if (allocated(self%mean_coefficients)) deallocate(self%mean_coefficients)
        allocate(self%mean_coefficients(base_count*self%n_outputs))
        if (base_count > 0) then
            do output_index = 1, self%n_outputs
                self%mean_coefficients( &
                    (output_index - 1)*base_count + 1: &
                    output_index*base_count) = self%mean%parameters
            end do
        end if
        call gp_refactor(self, status)
    end subroutine gp_fit

    integer function gp_parameter_count(self) result(count)
        class(gp_regression_t), intent(in) :: self

        count = self%kernel%parameter_count() + 1 + &
            self%mean%parameter_count()*self%n_outputs
    end function gp_parameter_count

    function gp_parameters(self) result(parameters)
        class(gp_regression_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: kernel_count
        integer :: mean_count

        kernel_count = self%kernel%parameter_count()
        mean_count = self%mean%parameter_count()*self%n_outputs
        allocate(parameters(kernel_count + 1 + mean_count))
        if (kernel_count > 0) parameters(:kernel_count) = self%kernel%parameters()
        parameters(kernel_count + 1) = self%log_noise_variance
        if (mean_count > 0) parameters(kernel_count + 2:) = self%mean_coefficients
    end function gp_parameters

    subroutine gp_set_parameters(self, parameters, status)
        class(gp_regression_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: kernel_count
        integer :: mean_count

        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP set_parameters: model is not fitted")
            return
        end if
        if (size(parameters) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP set_parameters: parameter shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP set_parameters: parameters must be finite")
            return
        end if
        kernel_count = self%kernel%parameter_count()
        call self%kernel%set_parameters(parameters(:kernel_count), status)
        if (status%code /= FORTNUM_OK) return
        self%log_noise_variance = parameters(kernel_count + 1)
        mean_count = self%mean%parameter_count()*self%n_outputs
        if (mean_count > 0) self%mean_coefficients = parameters(kernel_count + 2:)
        call gp_refactor(self, status)
    end subroutine gp_set_parameters

    integer function gp_mean_parameter_count(self) result(count)
        class(gp_regression_t), intent(in) :: self

        count = self%mean%parameter_count()*self%n_outputs
    end function gp_mean_parameter_count

    function gp_mean_parameters(self) result(parameters)
        class(gp_regression_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(self%mean_parameter_count()))
        if (size(parameters) > 0) parameters = self%mean_coefficients
    end function gp_mean_parameters

    subroutine gp_predict(self, x, mean, variance, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:, :), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :)
        real(dp), allocatable :: prior_diagonal(:)
        real(dp), allocatable :: mean_query(:, :)
        integer :: i

        call check_prediction_shapes(self, x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(work, mold=cross)
        allocate(mean_query(size(x, 1), self%n_outputs))
        allocate(prior_diagonal(size(x, 1)))
        call gp_mean_values(self, x, mean_query, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return

        ! Only the diagonal of the prior is needed, so only the diagonal is
        ! computed. This previously built the full `m` by `m` prior covariance
        ! and immediately discarded everything off the diagonal. At the
        ! candidate counts Bayesian optimization actually uses -- TuRBO scores
        ! `min(100d, 5000)` candidates per region per step -- that is a
        ! 5000-by-5000 matrix, twenty-five million kernel evaluations and a
        ! 200 MB allocation, to obtain five thousand numbers. The work done
        ! here is `m` kernel evaluations rather than `m**2`.
        do i = 1, size(x, 1)
            prior_diagonal(i) = self%kernel%value(x(i, :), x(i, :))
        end do

        mean = mean_query + matmul(transpose(cross), self%alpha)

        ! One triangular solve, not two. The variance needs `k^T K^-1 k`, and
        ! `K = L L^T` makes that `|L^-1 k|^2`, so the back substitution
        ! `dpotrs` would perform produces a result this expression discards.
        ! The solve is the dominant cost of the posterior at Bayesian-
        ! optimization sizes -- measured at 70 percent for forty training
        ! points against four thousand candidates -- so the saving is half of
        ! the largest term.
        work = cross
        call self%factorization%solve_lower_matrix(work, status)
        if (status%code /= FORTNUM_OK) return
        variance = prior_diagonal - sum(work*work, dim=1)
        do i = 1, size(variance)
            if (variance(i) < 0.0_dp) then
                if (variance(i) < -1.0e-9_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "GP predict: posterior variance is not positive")
                    return
                end if
                variance(i) = 0.0_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict

    subroutine gp_predict_covariance(self, x, covariance, status)
        !! Dense posterior covariance for a shared query set.
        !!
        !! The covariance is the latent covariance (observation noise is not
        !! added) and is shared by all independent output columns.  Returning
        !! the full matrix matches the exact-GP posterior contract used by
        !! GPyTorch/GPflow and avoids callers reconstructing it from marginal
        !! variances.  The Cholesky factor is reused, so the solve is the same
        !! one used by `predict` and no host fallback is hidden here.
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior(:, :), work(:, :)
        integer :: m, i

        covariance = 0.0_dp
        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_covariance: model is not fitted")
            return
        end if
        m = size(x, 1)
        if (m < 1 .or. size(x, 2) /= self%n_features .or. &
            any(shape(covariance) /= [m, m])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_covariance: input or covariance shape is invalid")
            return
        end if
        allocate(cross(self%n_samples, m), prior(m, m), work(self%n_samples, m))
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix(x, x, prior, status)
        if (status%code /= FORTNUM_OK) return
        work = cross
        call self%factorization%solve_lower_matrix(work, status)
        if (status%code /= FORTNUM_OK) return
        covariance = prior - matmul(transpose(work), work)
        covariance = 0.5_dp*(covariance + transpose(covariance))
        do i = 1, m
            if (covariance(i, i) < -1.0e-9_dp) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "GP predict_covariance: posterior covariance is not positive")
                covariance = 0.0_dp
                return
            end if
            if (covariance(i, i) < 0.0_dp) covariance(i, i) = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict_covariance

    subroutine gp_predict_covariance_device(self, device, x, covariance, status)
        !! Device boundary for the exact posterior covariance.
        !!
        !! CPU dispatches to the reference dense implementation.  CUDA is a
        !! typed refusal until a resident covariance/Cholesky plan is linked;
        !! the routine never silently copies data back to the host.
        class(gp_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status

        covariance = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP covariance device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_covariance(x, covariance, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP covariance device: resident CUDA covariance or Cholesky kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP covariance device: device kind is invalid")
        end select
    end subroutine gp_predict_covariance_device

    subroutine gp_predict_covariance_jvp(self, x, direction, covariance, &
            covariance_dot, status)
        !! Directional hyperparameter product of the dense latent covariance.
        !!
        !! The fitted solve state is differentiated implicitly without
        !! differentiating a Cholesky factor.  With ``W=K^{-1}K(X,x)`` the
        !! posterior is ``K(x,x)-K(X,x)^T W`` and the tangent reuses the
        !! factorization for ``W_dot=K^{-1}(K_dot(X,x)-K_dot W)``.
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: covariance(:, :), covariance_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), prior(:, :)
        real(dp), allocatable :: prior_dot(:, :), train_dot(:, :)
        real(dp), allocatable :: train(:, :), work(:, :), work_dot(:, :)
        real(dp) :: noise_dot
        integer :: m, i, kernel_count

        covariance = 0.0_dp
        covariance_dot = 0.0_dp
        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_covariance_jvp: model is not fitted")
            return
        end if
        m = size(x, 1)
        kernel_count = self%kernel%parameter_count()
        if (m < 1 .or. size(x, 2) /= self%n_features .or. &
            any(shape(covariance) /= [m, m]) .or. &
            any(shape(covariance_dot) /= [m, m]) .or. &
            size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_covariance_jvp: input, tangent, or output shape is invalid")
            return
        end if

        allocate(cross(self%n_samples, m), cross_dot(self%n_samples, m))
        allocate(prior(m, m), prior_dot(m, m))
        allocate(train(self%n_samples, self%n_samples))
        allocate(train_dot, mold=train)
        allocate(work(self%n_samples, m), work_dot(self%n_samples, m))
        call self%kernel%matrix_jvp(self%x_train, x, direction(:kernel_count), &
            cross, cross_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix_jvp(x, x, direction(:kernel_count), &
            prior, prior_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix_jvp(self%x_train, self%x_train, &
            direction(:kernel_count), train, train_dot, status)
        if (status%code /= FORTNUM_OK) return
        noise_dot = exp(self%log_noise_variance)*direction(kernel_count + 1)
        do i = 1, self%n_samples
            train_dot(i, i) = train_dot(i, i) + noise_dot
        end do

        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        work_dot = cross_dot - matmul(train_dot, work)
        call self%factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        covariance = prior - matmul(transpose(cross), work)
        covariance_dot = prior_dot - matmul(transpose(cross_dot), work) - &
            matmul(transpose(cross), work_dot)
        covariance = 0.5_dp*(covariance + transpose(covariance))
        covariance_dot = 0.5_dp*(covariance_dot + transpose(covariance_dot))
        if (any(.not. ieee_is_finite(covariance)) .or. &
            any(.not. ieee_is_finite(covariance_dot))) then
            covariance = 0.0_dp
            covariance_dot = 0.0_dp
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP predict_covariance_jvp: nonfinite covariance product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict_covariance_jvp

    subroutine gp_predict_covariance_vjp(self, x, covariance_bar, parameter_bar, status)
        !! Reverse hyperparameter product of the dense latent covariance.
        !!
        !! The cotangent is symmetrized because the public covariance is
        !! explicitly symmetrized.  Mean coefficients have zero cotangent:
        !! the latent posterior covariance is independent of the mean.
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), covariance_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), cross_bar(:, :)
        real(dp), allocatable :: prior_bar(:, :), train_bar(:, :), local_bar(:)
        real(dp), allocatable :: effective_bar(:, :)
        integer :: m, i, kernel_count

        parameter_bar = 0.0_dp
        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_covariance_vjp: model is not fitted")
            return
        end if
        m = size(x, 1)
        kernel_count = self%kernel%parameter_count()
        if (m < 1 .or. size(x, 2) /= self%n_features .or. &
            any(shape(covariance_bar) /= [m, m]) .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(covariance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_covariance_vjp: input, cotangent, or output shape is invalid")
            return
        end if

        allocate(cross(self%n_samples, m), work(self%n_samples, m))
        allocate(cross_bar(self%n_samples, m), prior_bar(m, m))
        allocate(train_bar(self%n_samples, self%n_samples))
        allocate(effective_bar(m, m), local_bar(kernel_count))
        effective_bar = 0.5_dp*(covariance_bar + transpose(covariance_bar))
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        prior_bar = effective_bar
        cross_bar = -matmul(work, covariance_bar + transpose(covariance_bar))
        train_bar = matmul(work, matmul(effective_bar, transpose(work)))

        call self%kernel%parameter_vjp(self%x_train, self%x_train, train_bar, &
            local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kernel_count) = local_bar
        call self%kernel%parameter_vjp(self%x_train, x, cross_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kernel_count) = parameter_bar(:kernel_count) + local_bar
        call self%kernel%parameter_vjp(x, x, prior_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kernel_count) = parameter_bar(:kernel_count) + local_bar
        parameter_bar(kernel_count + 1) = exp(self%log_noise_variance)* &
            sum([(train_bar(i, i), i = 1, self%n_samples)])
        if (size(parameter_bar) > kernel_count + 1) then
            parameter_bar(kernel_count + 2:) = 0.0_dp
        end if
        if (any(.not. ieee_is_finite(parameter_bar))) then
            parameter_bar = 0.0_dp
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP predict_covariance_vjp: nonfinite covariance product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict_covariance_vjp

    subroutine gp_predict_covariance_jvp_device(self, device, x, direction, &
            covariance, covariance_dot, status)
        !! Explicit device boundary for covariance JVP products.
        class(gp_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: covariance(:, :), covariance_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        covariance = 0.0_dp
        covariance_dot = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP covariance JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_covariance_jvp(x, direction, covariance, &
                covariance_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP covariance JVP device: resident CUDA covariance kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP covariance JVP device: device kind is invalid")
        end select
    end subroutine gp_predict_covariance_jvp_device

    subroutine gp_predict_covariance_vjp_device(self, device, x, covariance_bar, &
            parameter_bar, status)
        !! Explicit device boundary for covariance VJP products.
        class(gp_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), covariance_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP covariance VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_covariance_vjp(x, covariance_bar, parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "GP covariance VJP device: resident CUDA covariance kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP covariance VJP device: device kind is invalid")
        end select
    end subroutine gp_predict_covariance_vjp_device

    subroutine gp_predict_jvp(self, x, direction, mean, mean_dot, variance, &
            variance_dot, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: single_prior(1, 1), single_prior_dot(1, 1)
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), prior_diagonal(:), prior_dot_diagonal(:)
        real(dp), allocatable :: train_dot(:, :), train_matrix_dot(:, :)
        real(dp), allocatable :: alpha_dot(:, :), work(:, :)
        real(dp), allocatable :: work_dot(:, :)
        real(dp), allocatable :: mean_train(:, :), mean_train_dot(:, :)
        real(dp), allocatable :: mean_query(:, :), mean_query_dot(:, :)
        real(dp) :: noise_dot
        integer :: i, kernel_count, mean_count

        call check_prediction_shapes(self, x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        if (any(shape(mean_dot) /= shape(mean)) .or. size(variance_dot) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_jvp: tangent output shape is invalid")
            return
        end if
        if (size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_jvp: parameter tangent shape is invalid")
            return
        end if

        kernel_count = self%kernel%parameter_count()
        mean_count = self%mean_parameter_count()
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(cross_dot, mold=cross)
        allocate(prior_diagonal(size(x, 1)), prior_dot_diagonal(size(x, 1)))
        allocate(train_dot(self%n_samples, self%n_samples))
        allocate(train_matrix_dot, mold=train_dot)
        allocate(alpha_dot, mold=self%alpha)
        allocate(work, mold=cross)
        allocate(work_dot, mold=cross)
        allocate(mean_train_dot(self%n_samples, self%n_outputs))
        allocate(mean_train, mold=mean_train_dot)
        allocate(mean_query(size(x, 1), self%n_outputs))
        allocate(mean_query_dot, mold=mean_query)
        if (mean_count > 0) then
            call gp_mean_values_jvp(self, self%x_train, direction(kernel_count + 2:), &
                mean_train, mean_train_dot, status)
            if (status%code /= FORTNUM_OK) return
            call gp_mean_values_jvp(self, x, direction(kernel_count + 2:), &
                mean_query, mean_query_dot, status)
            if (status%code /= FORTNUM_OK) return
        else
            mean_train = 0.0_dp
            mean_train_dot = 0.0_dp
            mean_query = 0.0_dp
            mean_query_dot = 0.0_dp
        end if
        call self%kernel%matrix_jvp(self%x_train, x, direction(:kernel_count), &
            cross, cross_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix_jvp(self%x_train, self%x_train, &
            direction(:kernel_count), train_dot, train_matrix_dot, status)
        if (status%code /= FORTNUM_OK) return
        noise_dot = exp(self%log_noise_variance)*direction(kernel_count + 1)
        do i = 1, self%n_samples
            train_matrix_dot(i, i) = train_matrix_dot(i, i) + noise_dot
        end do
        alpha_dot = -matmul(train_matrix_dot, self%alpha) - mean_train_dot
        call self%factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return
        mean = mean_query + matmul(transpose(cross), self%alpha)
        mean_dot = mean_query_dot + matmul(transpose(cross_dot), self%alpha) + &
            matmul(transpose(cross), alpha_dot)
        ! Diagonal only, one query point at a time. Forming the full `m` by
        ! `m` prior and its tangent to read two diagonals costs `m**2` kernel
        ! evaluations and two `m**2` allocations; at Bayesian-optimization
        ! candidate counts that is twenty-five million evaluations and 400 MB
        ! for `2m` numbers.
        do i = 1, size(x, 1)
            call self%kernel%matrix_jvp(x(i:i, :), x(i:i, :), &
                direction(:kernel_count), single_prior, single_prior_dot, status)
            if (status%code /= FORTNUM_OK) return
            prior_diagonal(i) = single_prior(1, 1)
            prior_dot_diagonal(i) = single_prior_dot(1, 1)
        end do
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        work_dot = cross_dot - matmul(train_matrix_dot, work)
        call self%factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        variance = prior_diagonal - sum(cross*work, dim=1)
        variance_dot = prior_dot_diagonal - sum(cross_dot*work + &
            cross*work_dot, dim=1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict_jvp

    subroutine gp_predict_vjp(self, x, mean_bar, variance_bar, parameter_bar, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:, :), variance_bar(:)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), cross_bar(:, :)
        real(dp), allocatable :: train_bar(:, :), prior_bar(:, :), alpha_bar(:, :)
        real(dp), allocatable :: lambda(:, :), local_bar(:)
        real(dp), allocatable :: mean_train_basis(:, :), mean_query_basis(:, :)
        integer :: i, kernel_count, mean_count, output_index, first, last

        call check_prediction_shapes(self, x, mean_bar, variance_bar, status)
        if (status%code /= FORTNUM_OK) return
        if (size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_vjp: parameter output shape is invalid")
            return
        end if

        kernel_count = self%kernel%parameter_count()
        mean_count = self%mean_parameter_count()
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(work, mold=cross)
        allocate(cross_bar, mold=cross)
        allocate(train_bar(self%n_samples, self%n_samples))
        allocate(prior_bar(size(x, 1), size(x, 1)))
        allocate(alpha_bar, mold=self%alpha)
        allocate(lambda, mold=self%alpha)
        allocate(local_bar(kernel_count))
        if (mean_count > 0) then
            allocate(mean_train_basis(self%n_samples, self%mean%parameter_count()))
            allocate(mean_query_basis(size(x, 1), self%mean%parameter_count()))
            call self%mean%basis(self%x_train, mean_train_basis, status)
            if (status%code /= FORTNUM_OK) return
            call self%mean%basis(x, mean_query_basis, status)
            if (status%code /= FORTNUM_OK) return
        end if
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return

        cross_bar = matmul(self%alpha, transpose(mean_bar))
        alpha_bar = matmul(cross, mean_bar)
        prior_bar = 0.0_dp
        do i = 1, size(x, 1)
            prior_bar(i, i) = variance_bar(i)
        end do
        cross_bar = cross_bar - 2.0_dp*work*spread(variance_bar, dim=1, &
            ncopies=self%n_samples)
        train_bar = matmul(work*spread(variance_bar, dim=1, ncopies=self%n_samples), &
            transpose(work))
        lambda = alpha_bar
        call self%factorization%solve(lambda, status)
        if (status%code /= FORTNUM_OK) return
        train_bar = train_bar - 0.5_dp*(matmul(lambda, transpose(self%alpha)) + &
            matmul(self%alpha, transpose(lambda)))
        parameter_bar = 0.0_dp
        call self%kernel%parameter_vjp(self%x_train, self%x_train, train_bar, &
            local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kernel_count) = parameter_bar(:kernel_count) + local_bar
        call self%kernel%parameter_vjp(self%x_train, x, cross_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kernel_count) = parameter_bar(:kernel_count) + local_bar
        call self%kernel%parameter_vjp(x, x, prior_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kernel_count) = parameter_bar(:kernel_count) + local_bar
        parameter_bar(kernel_count + 1) = exp(self%log_noise_variance)* &
            sum(diagonal(train_bar))
        if (mean_count > 0) then
            do output_index = 1, self%n_outputs
                first = kernel_count + 1 + (output_index - 1)*self%mean%parameter_count() + 1
                last = first + self%mean%parameter_count() - 1
                parameter_bar(first:last) = matmul(transpose(mean_query_basis), &
                    mean_bar(:, output_index)) - matmul(transpose(mean_train_basis), &
                    lambda(:, output_index))
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict_vjp

    subroutine gp_predict_hvp(self, x, mean_bar, direction, parameter_hvp, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:, :), direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :)
        real(dp), allocatable :: train_matrix(:, :), train_matrix_dot(:, :)
        real(dp), allocatable :: alpha_dot(:, :), alpha_bar(:, :)
        real(dp), allocatable :: alpha_bar_dot(:, :), lambda(:, :), lambda_dot(:, :)
        real(dp), allocatable :: cross_bar(:, :), cross_bar_dot(:, :)
        real(dp), allocatable :: train_bar(:, :), train_bar_dot(:, :)
        real(dp), allocatable :: local_bar(:), local_bar_dot(:)
        real(dp), allocatable :: mean_train(:, :), mean_train_dot(:, :), mean_train_basis(:, :)
        integer :: i, kernel_count, mean_count, output_index, first, last
        real(dp) :: noise_dot, trace_train_bar, trace_train_bar_dot

        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_hvp: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(mean_bar) /= [size(x, 1), self%n_outputs])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_hvp: input or cotangent shape is invalid")
            return
        end if
        if (size(direction) /= self%parameter_count() .or. &
            size(parameter_hvp) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_hvp: parameter shape is invalid")
            return
        end if

        kernel_count = self%kernel%parameter_count()
        mean_count = self%mean_parameter_count()
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(cross_dot, mold=cross)
        allocate(train_matrix(self%n_samples, self%n_samples))
        allocate(train_matrix_dot, mold=train_matrix)
        allocate(alpha_dot, mold=self%alpha)
        allocate(alpha_bar, mold=self%alpha)
        allocate(alpha_bar_dot, mold=self%alpha)
        allocate(lambda, mold=self%alpha)
        allocate(lambda_dot, mold=self%alpha)
        allocate(cross_bar, mold=cross)
        allocate(cross_bar_dot, mold=cross)
        allocate(train_bar, mold=train_matrix)
        allocate(train_bar_dot, mold=train_matrix)
        allocate(local_bar(kernel_count), local_bar_dot(kernel_count))
        if (mean_count > 0) then
            allocate(mean_train_dot(self%n_samples, self%n_outputs))
            allocate(mean_train, mold=mean_train_dot)
            allocate(mean_train_basis(self%n_samples, self%mean%parameter_count()))
            call gp_mean_values_jvp(self, self%x_train, direction(kernel_count + 2:), &
                mean_train, mean_train_dot, status)
            if (status%code /= FORTNUM_OK) return
            call self%mean%basis(self%x_train, mean_train_basis, status)
            if (status%code /= FORTNUM_OK) return
        end if

        call self%kernel%matrix_jvp(self%x_train, x, direction(:kernel_count), &
            cross, cross_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix_jvp(self%x_train, self%x_train, &
            direction(:kernel_count), train_matrix, train_matrix_dot, status)
        if (status%code /= FORTNUM_OK) return

        noise_dot = exp(self%log_noise_variance)*direction(kernel_count + 1)
        do i = 1, self%n_samples
            train_matrix_dot(i, i) = train_matrix_dot(i, i) + noise_dot
        end do

        if (mean_count > 0) then
            alpha_dot = -matmul(train_matrix_dot, self%alpha) - mean_train_dot
        else
            alpha_dot = -matmul(train_matrix_dot, self%alpha)
        end if
        call self%factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return

        cross_bar = matmul(self%alpha, transpose(mean_bar))
        cross_bar_dot = matmul(alpha_dot, transpose(mean_bar))
        alpha_bar = matmul(cross, mean_bar)
        alpha_bar_dot = matmul(cross_dot, mean_bar)
        lambda = alpha_bar
        call self%factorization%solve(lambda, status)
        if (status%code /= FORTNUM_OK) return
        lambda_dot = alpha_bar_dot - matmul(train_matrix_dot, lambda)
        call self%factorization%solve(lambda_dot, status)
        if (status%code /= FORTNUM_OK) return

        train_bar = -0.5_dp*(matmul(lambda, transpose(self%alpha)) + &
            matmul(self%alpha, transpose(lambda)))
        train_bar_dot = -0.5_dp*( &
            matmul(lambda_dot, transpose(self%alpha)) + &
            matmul(lambda, transpose(alpha_dot)) + &
            matmul(alpha_dot, transpose(lambda)) + &
            matmul(self%alpha, transpose(lambda_dot)))

        parameter_hvp = 0.0_dp
        call self%kernel%parameter_hvp( &
            self%x_train, self%x_train, train_bar, direction(:kernel_count), &
            local_bar, local_bar_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%parameter_vjp( &
            self%x_train, self%x_train, train_bar_dot, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_hvp(:kernel_count) = local_bar + local_bar_dot

        call self%kernel%parameter_hvp( &
            self%x_train, x, cross_bar, direction(:kernel_count), &
            local_bar, local_bar_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%parameter_vjp( &
            self%x_train, x, cross_bar_dot, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_hvp(:kernel_count) = parameter_hvp(:kernel_count) + &
            local_bar + local_bar_dot

        trace_train_bar = 0.0_dp
        trace_train_bar_dot = 0.0_dp
        do i = 1, self%n_samples
            trace_train_bar = trace_train_bar + train_bar(i, i)
            trace_train_bar_dot = trace_train_bar_dot + train_bar_dot(i, i)
        end do
        parameter_hvp(kernel_count + 1) = exp(self%log_noise_variance)* &
            (direction(kernel_count + 1)*trace_train_bar + trace_train_bar_dot)
        if (mean_count > 0) then
            do output_index = 1, self%n_outputs
                first = kernel_count + 1 + (output_index - 1)*self%mean%parameter_count() + 1
                last = first + self%mean%parameter_count() - 1
                parameter_hvp(first:last) = -matmul(transpose(mean_train_basis), &
                    lambda_dot(:, output_index))
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict_hvp

    subroutine gp_log_marginal_likelihood(self, value, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: logdet
        real(dp), allocatable :: residual(:, :)

        value = 0.0_dp
        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP log marginal likelihood: model is not fitted")
            return
        end if
        call self%factorization%log_determinant(logdet, status)
        if (status%code /= FORTNUM_OK) return
        allocate(residual, source=self%y_train)
        call gp_mean_values(self, self%x_train, residual, status)
        if (status%code /= FORTNUM_OK) return
        residual = self%y_train - residual
        value = -0.5_dp*sum(residual*self%alpha) - &
            0.5_dp*real(self%n_outputs, dp)*logdet - &
            0.5_dp*real(self%n_samples*self%n_outputs, dp)*LOG_TWO_PI
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_log_marginal_likelihood

    subroutine gp_hyperparameter_gradient(self, gradient, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: identity(:, :), inverse(:, :), matrix_bar(:, :)
        integer :: i, kernel_count, mean_count, output_index, first, last

        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP hyperparameter_gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP hyperparameter_gradient: output shape is invalid")
            return
        end if
        allocate(identity(self%n_samples, self%n_samples))
        allocate(inverse, mold=identity)
        allocate(matrix_bar, mold=identity)
        identity = 0.0_dp
        do i = 1, self%n_samples
            identity(i, i) = 1.0_dp
        end do
        inverse = identity
        call self%factorization%solve(inverse, status)
        if (status%code /= FORTNUM_OK) return
        matrix_bar = 0.5_dp*(matmul(self%alpha, transpose(self%alpha)) - &
            real(self%n_outputs, dp)*inverse)
        kernel_count = self%kernel%parameter_count()
        call self%kernel%parameter_vjp(self%x_train, self%x_train, matrix_bar, &
            gradient(:kernel_count), status)
        if (status%code /= FORTNUM_OK) return
        gradient(kernel_count + 1) = exp(self%log_noise_variance)* &
            sum(diagonal(matrix_bar))
        mean_count = self%mean%parameter_count()
        if (mean_count > 0) then
            block
                real(dp), allocatable :: basis(:, :)
                allocate(basis(self%n_samples, mean_count))
                call self%mean%basis(self%x_train, basis, status)
                if (status%code /= FORTNUM_OK) return
                do output_index = 1, self%n_outputs
                    first = kernel_count + 1 + (output_index - 1)*mean_count + 1
                    last = first + mean_count - 1
                    gradient(first:last) = matmul(transpose(basis), &
                        self%alpha(:, output_index))
                end do
            end block
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_hyperparameter_gradient

    subroutine gp_hyperparameter_hvp(self, direction, parameter_hvp, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance(:, :), covariance_dot(:, :)
        real(dp), allocatable :: alpha_dot(:, :), identity(:, :), inverse(:, :)
        real(dp), allocatable :: inverse_dot(:, :)
        real(dp), allocatable :: matrix_bar(:, :), matrix_bar_dot(:, :)
        real(dp), allocatable :: local_bar(:), local_bar_dot(:)
        real(dp) :: noise_dot, trace_matrix_bar, trace_matrix_bar_dot
        integer :: i, kernel_count, mean_count, output_index, first, last
        real(dp), allocatable :: mean_train(:, :), mean_train_dot(:, :), mean_train_basis(:, :)

        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP hyperparameter_hvp: model is not fitted")
            return
        end if
        if (size(direction) /= self%parameter_count() .or. &
            size(parameter_hvp) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP hyperparameter_hvp: parameter shape is invalid")
            return
        end if

        kernel_count = self%kernel%parameter_count()
        mean_count = self%mean%parameter_count()
        allocate(covariance(self%n_samples, self%n_samples))
        allocate(covariance_dot, mold=covariance)
        allocate(alpha_dot, mold=self%alpha)
        allocate(identity, mold=covariance)
        allocate(inverse, mold=covariance)
        allocate(inverse_dot, mold=covariance)
        allocate(matrix_bar, mold=covariance)
        allocate(matrix_bar_dot, mold=covariance)
        allocate(local_bar(kernel_count), local_bar_dot(kernel_count))
        if (mean_count > 0) then
            allocate(mean_train_dot(self%n_samples, self%n_outputs))
            allocate(mean_train, mold=mean_train_dot)
            allocate(mean_train_basis(self%n_samples, mean_count))
            call gp_mean_values_jvp(self, self%x_train, direction(kernel_count + 2:), &
                mean_train, mean_train_dot, status)
            if (status%code /= FORTNUM_OK) return
            call self%mean%basis(self%x_train, mean_train_basis, status)
            if (status%code /= FORTNUM_OK) return
        end if

        call self%kernel%matrix_jvp(self%x_train, self%x_train, &
            direction(:kernel_count), covariance, covariance_dot, status)
        if (status%code /= FORTNUM_OK) return
        noise_dot = exp(self%log_noise_variance)*direction(kernel_count + 1)
        do i = 1, self%n_samples
            covariance_dot(i, i) = covariance_dot(i, i) + noise_dot
        end do

        if (mean_count > 0) then
            alpha_dot = -matmul(covariance_dot, self%alpha) - mean_train_dot
        else
            alpha_dot = -matmul(covariance_dot, self%alpha)
        end if
        call self%factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return

        identity = 0.0_dp
        do i = 1, self%n_samples
            identity(i, i) = 1.0_dp
        end do
        inverse = identity
        call self%factorization%solve(inverse, status)
        if (status%code /= FORTNUM_OK) return
        inverse_dot = -matmul(inverse, matmul(covariance_dot, inverse))

        matrix_bar = 0.5_dp*(matmul(self%alpha, transpose(self%alpha)) - &
            real(self%n_outputs, dp)*inverse)
        matrix_bar_dot = 0.5_dp*( &
            matmul(alpha_dot, transpose(self%alpha)) + &
            matmul(self%alpha, transpose(alpha_dot)) - &
            real(self%n_outputs, dp)*inverse_dot)

        parameter_hvp = 0.0_dp
        call self%kernel%parameter_hvp( &
            self%x_train, self%x_train, matrix_bar, direction(:kernel_count), &
            local_bar, local_bar_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%parameter_vjp( &
            self%x_train, self%x_train, matrix_bar_dot, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_hvp(:kernel_count) = local_bar + local_bar_dot

        trace_matrix_bar = 0.0_dp
        trace_matrix_bar_dot = 0.0_dp
        do i = 1, self%n_samples
            trace_matrix_bar = trace_matrix_bar + matrix_bar(i, i)
            trace_matrix_bar_dot = trace_matrix_bar_dot + matrix_bar_dot(i, i)
        end do
        parameter_hvp(kernel_count + 1) = exp(self%log_noise_variance)* &
            (direction(kernel_count + 1)*trace_matrix_bar + &
            trace_matrix_bar_dot)
        if (mean_count > 0) then
            do output_index = 1, self%n_outputs
                first = kernel_count + 1 + (output_index - 1)*mean_count + 1
                last = first + mean_count - 1
                parameter_hvp(first:last) = matmul(transpose(mean_train_basis), &
                    alpha_dot(:, output_index))
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_hyperparameter_hvp

    subroutine gp_lml_jvp(self, direction, value_dot, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value_dot = 0.0_dp
        if (size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP log marginal likelihood_jvp: direction shape is invalid")
            return
        end if
        allocate(gradient(size(direction)))
        call self%hyperparameter_gradient(gradient, status)
        if (status%code /= FORTNUM_OK) return
        value_dot = dot_product(gradient, direction)
    end subroutine gp_lml_jvp

    subroutine gp_refactor(self, status)
        class(gp_regression_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance(:, :), mean_train(:, :)
        real(dp) :: noise_variance
        integer :: i

        if (.not. allocated(self%x_train) .or. .not. allocated(self%y_train)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP refactor: training data is not initialized")
            return
        end if
        allocate(covariance(self%n_samples, self%n_samples))
        allocate(mean_train(self%n_samples, self%n_outputs))
        call gp_mean_values(self, self%x_train, mean_train, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix(self%x_train, self%x_train, covariance, status)
        if (status%code /= FORTNUM_OK) return
        noise_variance = exp(self%log_noise_variance)
        do i = 1, self%n_samples
            covariance(i, i) = covariance(i, i) + noise_variance + self%jitter
        end do
        call self%factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        if (allocated(self%alpha)) deallocate(self%alpha)
        allocate(self%alpha, source=self%y_train)
        self%alpha = self%alpha - mean_train
        call self%factorization%solve(self%alpha, status)
    end subroutine gp_refactor

    subroutine gp_mean_values(self, x, values, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: basis(:, :)
        integer :: p, output_index, first, last

        p = self%mean%parameter_count()
        if (any(shape(values) /= [size(x, 1), self%n_outputs])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP mean values: output shape is invalid")
            return
        end if
        values = 0.0_dp
        if (p == 0) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        allocate(basis(size(x, 1), p))
        call self%mean%basis(x, basis, status)
        if (status%code /= FORTNUM_OK) return
        do output_index = 1, self%n_outputs
            first = (output_index - 1)*p + 1
            last = first + p - 1
            values(:, output_index) = matmul(basis, self%mean_coefficients(first:last))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_mean_values

    subroutine gp_mean_values_jvp(self, x, direction, values, values_dot, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: values(:, :), values_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: basis(:, :)
        integer :: p, output_index, first, last

        p = self%mean%parameter_count()
        if (size(direction) /= p*self%n_outputs .or. &
            any(shape(values) /= [size(x, 1), self%n_outputs]) .or. &
            any(shape(values_dot) /= shape(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP mean JVP: direction or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP mean JVP: direction must be finite")
            return
        end if
        values = 0.0_dp
        values_dot = 0.0_dp
        if (p == 0) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        allocate(basis(size(x, 1), p))
        call self%mean%basis(x, basis, status)
        if (status%code /= FORTNUM_OK) return
        do output_index = 1, self%n_outputs
            first = (output_index - 1)*p + 1
            last = first + p - 1
            values(:, output_index) = matmul(basis, self%mean_coefficients(first:last))
            values_dot(:, output_index) = matmul(basis, direction(first:last))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_mean_values_jvp

    subroutine check_prediction_shapes(self, x, mean, variance, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean(:, :), variance(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP prediction: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP prediction: input dimension is invalid")
            return
        end if
        if (any(shape(mean) /= [size(x, 1), self%n_outputs]) .or. &
            size(variance) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP prediction: output shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_prediction_shapes

    logical function gp_fitted(self) result(fitted)
        class(gp_regression_t), intent(in) :: self

        fitted = allocated(self%x_train)
        if (.not. fitted) return
        fitted = allocated(self%y_train)
        if (.not. fitted) return
        fitted = allocated(self%alpha)
        if (.not. fitted) return
        fitted = self%n_samples > 0 .and. self%n_features > 0 .and. &
            self%n_outputs > 0
    end function gp_fitted

    function diagonal(matrix) result(values)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), allocatable :: values(:)
        integer :: i

        allocate(values(min(size(matrix, 1), size(matrix, 2))))
        do i = 1, size(values)
            values(i) = matrix(i, i)
        end do
    end function diagonal

end module fortml_gaussian_process
