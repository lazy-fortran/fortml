module fortml_sparse_gp
    !! Inducing-point variational Gaussian process with a Gaussian likelihood.
    !!
    !! The variational posterior `q(u) = N(m, S)` lives on `m` inducing inputs
    !! `Z`. Writing `A = K_uu^{-1} K_uf`, the marginal of `f_i` under `q` is
    !!
    !!     mean_i = a_i^T m
    !!     var_i  = k_ii - a_i^T K_uu a_i + a_i^T S a_i
    !!
    !! and for a Gaussian likelihood the expected log likelihood is closed
    !! form, so the ELBO needs no sampling:
    !!
    !!     ELBO = sum_i [ log N(y_i | mean_i, sigma^2) - var_i/(2 sigma^2) ]
    !!            - KL(q(u) || N(0, K_uu))
    !!
    !! The identity the tests use as their oracle: with the inducing inputs
    !! placed on the data and `q` set to the exact posterior over `f`, the ELBO
    !! equals the exact log marginal likelihood. Away from that setting it is a
    !! strict lower bound, which is the other property worth checking.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t
    implicit none
    private

    real(dp), parameter :: PI = 3.141592653589793238462643_dp

    type, public :: sparse_gp_t
        type(kernel_t) :: kernel
        real(dp), allocatable :: inducing_points(:, :)
        real(dp), allocatable :: variational_mean(:)
        real(dp), allocatable :: variational_factor(:, :)
        real(dp) :: noise_variance = 1.0_dp
        integer :: n_inducing = 0
    contains
        procedure, public :: initialize => sparse_gp_initialize
        procedure, public :: parameter_count => sparse_gp_parameter_count
        procedure, public :: parameters => sparse_gp_parameters
        procedure, public :: set_parameters => sparse_gp_set_parameters
        procedure, public :: set_variational => sparse_gp_set_variational
        procedure, public :: elbo => sparse_gp_elbo
        procedure, public :: elbo_gradient => sparse_gp_elbo_gradient
        procedure, public :: elbo_jvp => sparse_gp_elbo_jvp
        procedure, public :: elbo_vjp => sparse_gp_elbo_vjp
        procedure, public :: elbo_device => sparse_gp_elbo_device
        procedure, public :: elbo_gradient_device => sparse_gp_elbo_gradient_device
        procedure, public :: elbo_jvp_device => sparse_gp_elbo_jvp_device
        procedure, public :: device_supported => sparse_gp_device_supported
        procedure, public :: predict => sparse_gp_predict
    end type sparse_gp_t

contains

    subroutine sparse_gp_initialize(self, inducing_points, kernel, &
            noise_variance, status)
        class(sparse_gp_t), intent(out) :: self
        real(dp), intent(in) :: inducing_points(:, :)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (size(inducing_points, 1) < 1 .or. noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: inducing points and noise must be present and positive")
            return
        end if
        if (size(inducing_points, 2) /= kernel%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: inducing points do not match the kernel dimension")
            return
        end if
        self%kernel = kernel
        self%noise_variance = noise_variance
        self%n_inducing = size(inducing_points, 1)
        allocate(self%inducing_points, source=inducing_points)
        allocate(self%variational_mean(self%n_inducing))
        allocate(self%variational_factor(self%n_inducing, self%n_inducing))
        self%variational_mean = 0.0_dp
        self%variational_factor = 0.0_dp
        do i = 1, self%n_inducing
            self%variational_factor(i, i) = 1.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_gp_initialize

    integer function sparse_gp_parameter_count(self) result(count)
        class(sparse_gp_t), intent(in) :: self

        count = 0
        if (self%n_inducing < 1) return
        count = self%n_inducing + self%n_inducing*(self%n_inducing + 1)/2
    end function sparse_gp_parameter_count

    function sparse_gp_parameters(self) result(parameters)
        !! Packed variational coordinates: mean, then log-diagonal/lower
        !! columns of the covariance Cholesky factor.
        class(sparse_gp_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: i, j, position

        allocate(parameters(self%parameter_count()))
        if (self%n_inducing < 1) return
        parameters(:self%n_inducing) = self%variational_mean
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            parameters(position) = log(self%variational_factor(j, j))
            position = position + 1
            do i = j + 1, self%n_inducing
                parameters(position) = self%variational_factor(i, j)
                position = position + 1
            end do
        end do
    end function sparse_gp_parameters

    subroutine sparse_gp_set_parameters(self, parameters, status)
        !! Set the packed variational vector without changing kernel state.
        class(sparse_gp_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, position

        if (self%n_inducing < 1 .or. size(parameters) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: parameter shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: parameters must be finite")
            return
        end if
        self%variational_mean = parameters(:self%n_inducing)
        self%variational_factor = 0.0_dp
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            if (parameters(position) < log(tiny(1.0_dp)) .or. &
                parameters(position) > log(huge(1.0_dp))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "sparse GP: covariance diagonal overflows")
                return
            end if
            self%variational_factor(j, j) = exp(parameters(position))
            position = position + 1
            do i = j + 1, self%n_inducing
                self%variational_factor(i, j) = parameters(position)
                position = position + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_gp_set_parameters

    subroutine sparse_gp_set_variational(self, mean, factor, status)
        !! `factor` is the lower-triangular Cholesky factor of `S`.
        class(sparse_gp_t), intent(inout) :: self
        real(dp), intent(in) :: mean(:), factor(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (self%n_inducing < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: set variational parameters after initialize")
            return
        end if
        if (size(mean) /= self%n_inducing .or. &
            any(shape(factor) /= [self%n_inducing, self%n_inducing])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: variational parameter shape is invalid")
            return
        end if
        do i = 1, self%n_inducing
            if (factor(i, i) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "sparse GP: the covariance factor needs a positive diagonal")
                return
            end if
        end do
        self%variational_mean = mean
        self%variational_factor = factor
        do i = 1, self%n_inducing
            self%variational_factor(i, i + 1:) = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_gp_set_variational

    subroutine sparse_gp_elbo_gradient(self, x, y, value, gradient, status)
        !! Analytic gradient of the Gaussian variational ELBO.
        class(sparse_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: a(:, :), mean(:), variance(:), b(:, :)
        real(dp), allocatable :: mean_solve(:), factor_solve(:, :), mean_gradient(:)
        real(dp), allocatable :: factor_gradient(:, :)
        type(cholesky_factorization_t) :: factorization
        real(dp) :: likelihood, divergence, residual, inverse_noise
        integer :: i, j, position

        value = 0.0_dp
        gradient = 0.0_dp
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: ELBO gradient shape is invalid")
            return
        end if
        call sparse_gp_elbo_terms(self, x, y, a, mean, variance, b, factorization, &
            likelihood, divergence, status)
        if (status%code /= FORTNUM_OK) return

        allocate(mean_solve(self%n_inducing), factor_solve(self%n_inducing, &
            self%n_inducing), mean_gradient(self%n_inducing), &
            factor_gradient(self%n_inducing, self%n_inducing))
        mean_solve = self%variational_mean
        call factorization%solve(mean_solve, status)
        if (status%code /= FORTNUM_OK) return
        factor_solve = self%variational_factor
        call factorization%solve(factor_solve, status)
        if (status%code /= FORTNUM_OK) return

        inverse_noise = 1.0_dp/self%noise_variance
        mean_gradient = -mean_solve
        factor_gradient = -factor_solve
        do i = 1, size(x, 1)
            residual = y(i) - mean(i)
            mean_gradient = mean_gradient + inverse_noise*residual*a(:, i)
            do j = 1, self%n_inducing
                factor_gradient(j:, j) = factor_gradient(j:, j) - &
                    inverse_noise*b(j, i)*a(j:, i)
            end do
        end do
        do j = 1, self%n_inducing
            factor_gradient(j, j) = factor_gradient(j, j) + &
                1.0_dp/self%variational_factor(j, j)
        end do

        gradient(:self%n_inducing) = mean_gradient
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            gradient(position) = factor_gradient(j, j)*self%variational_factor(j, j)
            position = position + 1
            do i = j + 1, self%n_inducing
                gradient(position) = factor_gradient(i, j)
                position = position + 1
            end do
        end do
        value = likelihood - divergence
        if (any(.not. ieee_is_finite(gradient)) .or. .not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "sparse GP: ELBO gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_gp_elbo_gradient

    subroutine sparse_gp_elbo_jvp(self, x, y, direction, value, tangent, status)
        !! Forward directional product through the packed variational ELBO.
        class(sparse_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: a(:, :), mean(:), variance(:), b(:, :)
        real(dp), allocatable :: mean_dot(:), factor_dot(:, :), b_dot(:)
        real(dp), allocatable :: mean_solve(:), factor_solve(:, :)
        type(cholesky_factorization_t) :: factorization
        real(dp) :: likelihood, divergence, likelihood_dot, divergence_dot
        real(dp) :: residual, mean_tangent, variance_tangent, inverse_noise
        integer :: i, j, position

        value = 0.0_dp
        tangent = 0.0_dp
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: ELBO JVP direction is invalid")
            return
        end if
        call sparse_gp_elbo_terms(self, x, y, a, mean, variance, b, factorization, &
            likelihood, divergence, status)
        if (status%code /= FORTNUM_OK) return

        allocate(mean_dot(self%n_inducing), factor_dot(self%n_inducing, &
            self%n_inducing), b_dot(self%n_inducing), mean_solve(self%n_inducing), &
            factor_solve(self%n_inducing, self%n_inducing))
        mean_dot = direction(:self%n_inducing)
        factor_dot = 0.0_dp
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            factor_dot(j, j) = self%variational_factor(j, j)*direction(position)
            position = position + 1
            do i = j + 1, self%n_inducing
                factor_dot(i, j) = direction(position)
                position = position + 1
            end do
        end do
        mean_dot = matmul(transpose(a), mean_dot)
        inverse_noise = 1.0_dp/self%noise_variance
        likelihood_dot = 0.0_dp
        do i = 1, size(x, 1)
            b_dot = matmul(transpose(factor_dot), a(:, i))
            mean_tangent = mean_dot(i)
            variance_tangent = 2.0_dp*dot_product(b(:, i), b_dot)
            residual = y(i) - mean(i)
            likelihood_dot = likelihood_dot + inverse_noise*residual*mean_tangent &
                - 0.5_dp*inverse_noise*variance_tangent
        end do

        mean_solve = self%variational_mean
        call factorization%solve(mean_solve, status)
        if (status%code /= FORTNUM_OK) return
        factor_solve = self%variational_factor
        call factorization%solve(factor_solve, status)
        if (status%code /= FORTNUM_OK) return
        divergence_dot = dot_product(mean_solve, direction(:self%n_inducing)) + &
            sum(factor_solve*factor_dot)
        do j = 1, self%n_inducing
            divergence_dot = divergence_dot - factor_dot(j, j)/ &
                self%variational_factor(j, j)
        end do
        value = likelihood - divergence
        tangent = likelihood_dot - divergence_dot
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "sparse GP: ELBO JVP is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_gp_elbo_jvp

    subroutine sparse_gp_elbo_vjp(self, x, y, value_bar, parameter_bar, status)
        !! Reverse product of the scalar ELBO with an explicit cotangent.
        class(sparse_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:), value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value
        real(dp), allocatable :: gradient(:)

        parameter_bar = 0.0_dp
        if (size(parameter_bar) /= self%parameter_count() .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: ELBO VJP cotangent or output shape is invalid")
            return
        end if
        allocate(gradient(size(parameter_bar)))
        call self%elbo_gradient(x, y, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = value_bar*gradient
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "sparse GP: ELBO VJP is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_gp_elbo_vjp

    subroutine sparse_gp_elbo(self, x, y, value, status, &
            expected_log_likelihood, kl_value)
        class(sparse_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(out), optional :: expected_log_likelihood, kl_value

        real(dp), allocatable :: k_uu(:, :), k_uf(:, :), a(:, :)
        real(dp), allocatable :: prior_solve(:, :), mean(:)
        type(cholesky_factorization_t) :: factorization
        real(dp) :: likelihood, divergence, marginal, residual, trace_term
        real(dp) :: quadratic, log_det_prior, log_det_posterior, diagonal
        integer :: i, j, n_samples

        value = 0.0_dp
        if (present(expected_log_likelihood)) expected_log_likelihood = 0.0_dp
        if (present(kl_value)) kl_value = 0.0_dp
        n_samples = size(x, 1)
        if (.not. valid_data(self, x, y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: data shape is invalid")
            return
        end if

        call build_blocks(self, x, k_uu, k_uf, factorization, status)
        if (status%code /= FORTNUM_OK) return
        allocate(a(self%n_inducing, n_samples))
        a = k_uf
        call factorization%solve(a, status)
        if (status%code /= FORTNUM_OK) return

        allocate(mean(n_samples))
        mean = matmul(transpose(a), self%variational_mean)
        likelihood = 0.0_dp
        do i = 1, n_samples
            diagonal = self%kernel%value(x(i, :), x(i, :))
            ! k_ii - a_i^T K_uu a_i + a_i^T S a_i, with S = L L^T.
            marginal = diagonal - sum(a(:, i)*k_uf(:, i))
            marginal = marginal + sum(matmul(a(:, i), self%variational_factor)**2)
            residual = y(i) - mean(i)
            likelihood = likelihood - 0.5_dp*log(2.0_dp*PI*self%noise_variance) &
                - 0.5_dp*residual*residual/self%noise_variance &
                - 0.5_dp*marginal/self%noise_variance
        end do

        ! KL(N(m, S) || N(0, K_uu)).
        allocate(prior_solve(self%n_inducing, self%n_inducing))
        prior_solve = self%variational_factor
        call factorization%solve(prior_solve, status)
        if (status%code /= FORTNUM_OK) return
        trace_term = sum(prior_solve*self%variational_factor)
        block
            real(dp), allocatable :: mean_solve(:)

            allocate(mean_solve(self%n_inducing))
            mean_solve = self%variational_mean
            call factorization%solve(mean_solve, status)
            if (status%code /= FORTNUM_OK) return
            quadratic = sum(self%variational_mean*mean_solve)
        end block
        call factorization%log_determinant(log_det_prior, status)
        if (status%code /= FORTNUM_OK) return
        log_det_posterior = 0.0_dp
        do j = 1, self%n_inducing
            log_det_posterior = log_det_posterior + &
                2.0_dp*log(self%variational_factor(j, j))
        end do
        divergence = 0.5_dp*(trace_term + quadratic - &
            real(self%n_inducing, dp) + log_det_prior - log_det_posterior)

        value = likelihood - divergence
        if (present(expected_log_likelihood)) expected_log_likelihood = likelihood
        if (present(kl_value)) kl_value = divergence
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_gp_elbo

    subroutine sparse_gp_predict(self, x_star, mean, variance, status)
        !! Predictive marginals of `q(f_*)`, without observation noise.
        class(sparse_gp_t), intent(in) :: self
        real(dp), intent(in) :: x_star(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: k_uu(:, :), k_us(:, :), a(:, :)
        type(cholesky_factorization_t) :: factorization
        integer :: i

        mean = 0.0_dp
        variance = 0.0_dp
        if (self%n_inducing < 1 .or. size(x_star, 2) /= self%kernel%input_dim &
            .or. size(mean) /= size(x_star, 1) .or. &
            size(variance) /= size(x_star, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: prediction shape is invalid")
            return
        end if

        call build_blocks(self, x_star, k_uu, k_us, factorization, status)
        if (status%code /= FORTNUM_OK) return
        allocate(a(self%n_inducing, size(x_star, 1)))
        a = k_us
        call factorization%solve(a, status)
        if (status%code /= FORTNUM_OK) return

        mean = matmul(transpose(a), self%variational_mean)
        do i = 1, size(x_star, 1)
            variance(i) = self%kernel%value(x_star(i, :), x_star(i, :)) &
                - sum(a(:, i)*k_us(:, i)) &
                + sum(matmul(a(:, i), self%variational_factor)**2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_gp_predict

    subroutine sparse_gp_elbo_device(self, device, x, y, value, status)
        !! Device boundary for the scalar variational objective.
        class(sparse_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), y(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%elbo(x, y, value, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "sparse GP device: resident CUDA inducing graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP device: device kind is invalid")
        end select
    end subroutine sparse_gp_elbo_device

    subroutine sparse_gp_elbo_gradient_device(self, device, x, y, value, gradient, status)
        !! Device boundary for the scalar ELBO gradient.
        class(sparse_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), y(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP gradient device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%elbo_gradient(x, y, value, gradient, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "sparse GP gradient device: resident CUDA inducing graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP gradient device: device kind is invalid")
        end select
    end subroutine sparse_gp_elbo_gradient_device

    subroutine sparse_gp_elbo_jvp_device(self, device, x, y, direction, value, tangent, status)
        !! Device boundary for the forward variational product.
        class(sparse_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), y(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        tangent = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%elbo_jvp(x, y, direction, value, tangent, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "sparse GP JVP device: resident CUDA inducing graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP JVP device: device kind is invalid")
        end select
    end subroutine sparse_gp_elbo_jvp_device

    logical function sparse_gp_device_supported(self, device_kind) result(supported)
        class(sparse_gp_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function sparse_gp_device_supported

    subroutine sparse_gp_elbo_terms(self, x, y, a, mean, variance, b, factorization, &
            likelihood, divergence, status)
        !! Build the fixed inducing projection and the scalar Gaussian ELBO terms.
        class(sparse_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        real(dp), allocatable, intent(out) :: a(:, :), mean(:), variance(:), b(:, :)
        type(cholesky_factorization_t), intent(out) :: factorization
        real(dp), intent(out) :: likelihood, divergence
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: k_uu(:, :), k_uf(:, :), prior_solve(:, :)
        real(dp), allocatable :: mean_solve(:)
        real(dp) :: marginal, residual, trace_term, quadratic
        real(dp) :: log_det_prior, log_det_posterior
        integer :: i, j, n_samples

        likelihood = 0.0_dp
        divergence = 0.0_dp
        n_samples = size(x, 1)
        if (.not. valid_data(self, x, y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse GP: data shape is invalid")
            return
        end if

        call build_blocks(self, x, k_uu, k_uf, factorization, status)
        if (status%code /= FORTNUM_OK) return
        allocate(a(self%n_inducing, n_samples), mean(n_samples), &
            variance(n_samples), b(self%n_inducing, n_samples))
        a = k_uf
        call factorization%solve(a, status)
        if (status%code /= FORTNUM_OK) return
        mean = matmul(transpose(a), self%variational_mean)
        do i = 1, n_samples
            marginal = self%kernel%value(x(i, :), x(i, :)) - &
                sum(a(:, i)*k_uf(:, i))
            b(:, i) = matmul(transpose(self%variational_factor), a(:, i))
            marginal = marginal + sum(b(:, i)**2)
            variance(i) = marginal
            residual = y(i) - mean(i)
            likelihood = likelihood - 0.5_dp*log(2.0_dp*PI*self%noise_variance) &
                - 0.5_dp*residual*residual/self%noise_variance &
                - 0.5_dp*marginal/self%noise_variance
        end do

        allocate(prior_solve(self%n_inducing, self%n_inducing), &
            mean_solve(self%n_inducing))
        prior_solve = self%variational_factor
        call factorization%solve(prior_solve, status)
        if (status%code /= FORTNUM_OK) return
        trace_term = sum(prior_solve*self%variational_factor)
        mean_solve = self%variational_mean
        call factorization%solve(mean_solve, status)
        if (status%code /= FORTNUM_OK) return
        quadratic = sum(self%variational_mean*mean_solve)
        call factorization%log_determinant(log_det_prior, status)
        if (status%code /= FORTNUM_OK) return
        log_det_posterior = 0.0_dp
        do j = 1, self%n_inducing
            log_det_posterior = log_det_posterior + &
                2.0_dp*log(self%variational_factor(j, j))
        end do
        divergence = 0.5_dp*(trace_term + quadratic - &
            real(self%n_inducing, dp) + log_det_prior - log_det_posterior)
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_gp_elbo_terms

    subroutine build_blocks(self, x, k_uu, k_ux, factorization, status)
        !! `K_uu` with a small jitter and its Cholesky, plus `K_ux`.
        class(sparse_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: k_uu(:, :), k_ux(:, :)
        type(cholesky_factorization_t), intent(out) :: factorization
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: jitter
        integer :: i

        allocate(k_uu(self%n_inducing, self%n_inducing))
        allocate(k_ux(self%n_inducing, size(x, 1)))
        call self%kernel%matrix(self%inducing_points, self%inducing_points, &
            k_uu, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix(self%inducing_points, x, k_ux, status)
        if (status%code /= FORTNUM_OK) return
        jitter = 1.0e-10_dp*max(maxval(abs(k_uu)), 1.0_dp)
        do i = 1, self%n_inducing
            k_uu(i, i) = k_uu(i, i) + jitter
        end do
        call factorization%factorize(k_uu, status)
    end subroutine build_blocks

    logical function valid_data(self, x, y) result(valid)
        class(sparse_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:)

        valid = self%n_inducing > 0 .and. allocated(self%variational_mean)
        if (.not. valid) return
        valid = size(x, 1) > 0 .and. size(x, 1) == size(y) .and. &
            size(x, 2) == self%kernel%input_dim
    end function valid_data

end module fortml_sparse_gp
