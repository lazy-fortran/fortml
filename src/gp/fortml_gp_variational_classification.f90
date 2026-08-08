module fortml_gp_variational_classification
    !! Inducing-point variational Bernoulli GP classification.
    !!
    !! The posterior is ``q(u) = N(m, L L^T)`` at fixed inducing points.  For
    !! a training input ``x_i``, the marginal ``q(f_i)`` is evaluated from
    !! ``A = K_uu^{-1} K_ux`` and sampled with a deterministic, seeded
    !! reparameterization.  The expected Bernoulli log likelihood is therefore
    !! a deterministic objective, while the KL divergence to ``N(0,K_uu)`` is
    !! analytic.  `elbo_gradient` and `elbo_jvp` differentiate this exact
    !! deterministic objective with respect to the packed variational vector.
    !!
    !! This is deliberately a CPU reference path.  `elbo_device` refuses a
    !! CUDA request until the inducing solve, likelihood table, and reduction
    !! are resident kernels; it never hides a host fallback behind a device
    !! argument.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t
    implicit none
    private

    real(dp), parameter :: PI = 3.1415926535897932384626433832795_dp
    real(dp), parameter :: SQRT_TWO = 1.4142135623730950488016887242097_dp
    real(dp), parameter :: SQRT_TWO_PI = 2.506628274631000502415765284811_dp

    integer, parameter, public :: GP_VARIATIONAL_LOGISTIC = 1
    integer, parameter, public :: GP_VARIATIONAL_PROBIT = 2

    type, public :: gp_variational_classification_t
        private
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: prior_factorization
        real(dp), allocatable :: inducing_points(:, :)
        real(dp), allocatable :: variational_mean(:)
        real(dp), allocatable :: variational_factor(:, :)
        real(dp), allocatable :: noise(:, :)
        integer :: n_inducing = 0
        integer :: n_mc = 0
        integer :: noise_samples = 0
        integer :: seed = 1
        integer :: likelihood = GP_VARIATIONAL_LOGISTIC
    contains
        procedure, public :: initialize => gvc_initialize
        procedure, public :: parameter_count => gvc_parameter_count
        procedure, public :: parameters => gvc_parameters
        procedure, public :: set_parameters => gvc_set_parameters
        procedure, public :: elbo => gvc_elbo
        procedure, public :: elbo_gradient => gvc_elbo_gradient
        procedure, public :: elbo_jvp => gvc_elbo_jvp
        procedure, public :: predict_latent => gvc_predict_latent
        procedure, public :: predict_proba => gvc_predict_proba
        procedure, public :: predict_latent_parameter_jvp => &
            gvc_predict_latent_parameter_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            gvc_predict_proba_parameter_jvp
        procedure, public :: predict_latent_parameter_vjp => &
            gvc_predict_latent_parameter_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            gvc_predict_proba_parameter_vjp
        procedure, public :: kernel_parameter_count => &
            gvc_kernel_parameter_count
        procedure, public :: predict_latent_kernel_parameter_jvp => &
            gvc_predict_latent_kernel_parameter_jvp
        procedure, public :: predict_proba_kernel_parameter_jvp => &
            gvc_predict_proba_kernel_parameter_jvp
        procedure, public :: predict_latent_kernel_parameter_vjp => &
            gvc_predict_latent_kernel_parameter_vjp
        procedure, public :: predict_proba_kernel_parameter_vjp => &
            gvc_predict_proba_kernel_parameter_vjp
        procedure, public :: predict_proba_kernel_parameter_vjp_device => &
            gvc_predict_proba_kernel_parameter_vjp_device
        procedure, public :: predict_latent_input_jvp => &
            gvc_predict_latent_input_jvp
        procedure, public :: predict_proba_input_jvp => &
            gvc_predict_proba_input_jvp
        procedure, public :: predict_latent_input_vjp => &
            gvc_predict_latent_input_vjp
        procedure, public :: predict_proba_input_vjp => &
            gvc_predict_proba_input_vjp
        procedure, public :: predict_proba_input_vjp_device => &
            gvc_predict_proba_input_vjp_device
        procedure, public :: predict_proba_parameter_vjp_device => &
            gvc_predict_proba_parameter_vjp_device
        procedure, public :: elbo_device => gvc_elbo_device
        procedure, public :: device_supported => gvc_device_supported
    end type gp_variational_classification_t

contains

    subroutine gvc_initialize(self, inducing_points, kernel, n_mc_samples, seed, &
            status, likelihood, jitter)
        class(gp_variational_classification_t), intent(out) :: self
        real(dp), intent(in) :: inducing_points(:, :)
        type(kernel_t), intent(in) :: kernel
        integer, intent(in) :: n_mc_samples, seed
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: likelihood
        real(dp), intent(in), optional :: jitter
        real(dp), allocatable :: covariance(:, :)
        real(dp) :: requested_jitter
        integer :: i, requested_likelihood

        requested_likelihood = GP_VARIATIONAL_LOGISTIC
        if (present(likelihood)) requested_likelihood = likelihood
        requested_jitter = 1.0e-8_dp
        if (present(jitter)) requested_jitter = jitter
        if (size(inducing_points, 1) < 1 .or. &
            size(inducing_points, 2) /= kernel%input_dim .or. n_mc_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: initialization shape/options are invalid")
            return
        end if
        if (requested_jitter <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: initialization shape/options are invalid")
            return
        end if
        if (.not. ieee_is_finite(requested_jitter)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: initialization shape/options are invalid")
            return
        end if
        if (requested_likelihood /= GP_VARIATIONAL_LOGISTIC .and. &
            requested_likelihood /= GP_VARIATIONAL_PROBIT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: likelihood kind is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(inducing_points))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: inducing points must be finite")
            return
        end if

        self%kernel = kernel
        self%n_inducing = size(inducing_points, 1)
        self%n_mc = n_mc_samples
        self%seed = seed
        self%likelihood = requested_likelihood
        allocate(self%inducing_points, source=inducing_points)
        allocate(covariance(self%n_inducing, self%n_inducing))
        call self%kernel%matrix(self%inducing_points, self%inducing_points, &
            covariance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%n_inducing
            covariance(i, i) = covariance(i, i) + requested_jitter
        end do
        call self%prior_factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return

        allocate(self%variational_mean(self%n_inducing))
        allocate(self%variational_factor(self%n_inducing, self%n_inducing))
        self%variational_mean = 0.0_dp
        self%variational_factor = self%prior_factorization%lower
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_initialize

    integer function gvc_parameter_count(self) result(count)
        class(gp_variational_classification_t), intent(in) :: self

        count = 0
        if (self%n_inducing < 1) return
        count = self%n_inducing + self%n_inducing*(self%n_inducing + 1)/2
    end function gvc_parameter_count

    function gvc_parameters(self) result(lambda)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), allocatable :: lambda(:)
        integer :: i, j, position

        allocate(lambda(self%parameter_count()))
        if (self%n_inducing < 1) return
        lambda(1:self%n_inducing) = self%variational_mean
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            lambda(position) = log(self%variational_factor(j, j))
            position = position + 1
            do i = j + 1, self%n_inducing
                lambda(position) = self%variational_factor(i, j)
                position = position + 1
            end do
        end do
    end function gvc_parameters

    subroutine gvc_set_parameters(self, lambda, status)
        class(gp_variational_classification_t), intent(inout) :: self
        real(dp), intent(in) :: lambda(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, position

        if (self%n_inducing < 1 .or. size(lambda) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(lambda))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: parameter shape/value is invalid")
            return
        end if
        self%variational_mean = lambda(1:self%n_inducing)
        self%variational_factor = 0.0_dp
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            if (lambda(position) > log(huge(1.0_dp)) .or. &
                lambda(position) < log(tiny(1.0_dp))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP classification: covariance diagonal overflows")
                return
            end if
            self%variational_factor(j, j) = exp(lambda(position))
            position = position + 1
            do i = j + 1, self%n_inducing
                self%variational_factor(i, j) = lambda(position)
                position = position + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_set_parameters

    subroutine gvc_elbo(self, x, labels, value, status, expected_log_likelihood, &
            kl_value, scale)
        class(gp_variational_classification_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(out), optional :: expected_log_likelihood, kl_value
        real(dp), intent(in), optional :: scale
        real(dp), allocatable :: projection(:, :), mean(:), variance(:), latent(:)
        real(dp) :: likelihood, divergence, multiplier, term, unused_gradient
        integer :: i, s

        value = 0.0_dp
        if (present(expected_log_likelihood)) expected_log_likelihood = 0.0_dp
        if (present(kl_value)) kl_value = 0.0_dp
        if (.not. valid_data(self, x, labels, status)) return
        multiplier = 1.0_dp
        if (present(scale)) multiplier = scale
        if (multiplier <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: likelihood scale is invalid")
            return
        end if
        if (.not. ieee_is_finite(multiplier)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: likelihood scale is invalid")
            return
        end if
        call build_projection(self, x, projection, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ensure_noise(self, size(x, 1), status)
        if (status%code /= FORTNUM_OK) return
        allocate(latent(size(x, 1)))
        likelihood = 0.0_dp
        do s = 1, self%n_mc
            latent = mean + sqrt(variance)*self%noise(:, s)
            do i = 1, size(x, 1)
                call bernoulli_terms(labels(i), latent(i), self%likelihood, &
                    term, unused_gradient)
                likelihood = likelihood + term
            end do
        end do
        likelihood = likelihood/real(self%n_mc, dp)
        call variational_kl(self, divergence, status)
        if (status%code /= FORTNUM_OK) return
        value = multiplier*likelihood - divergence
        if (present(expected_log_likelihood)) expected_log_likelihood = likelihood
        if (present(kl_value)) kl_value = divergence
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_elbo

    subroutine gvc_elbo_gradient(self, x, labels, value, gradient, status, scale)
        class(gp_variational_classification_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale
        real(dp), allocatable :: projection(:, :), mean(:), variance(:), latent(:)
        real(dp), allocatable :: mean_gradient(:), factor_gradient(:, :), kl_mean(:), &
            kl_factor(:, :), solve_factor(:, :)
        real(dp) :: multiplier, likelihood, divergence, term, term_gradient, &
            coefficient, likelihood_scale
        real(dp), allocatable :: tangent_factor(:)
        integer :: i, j, s, position

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. valid_data(self, x, labels, status)) return
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: gradient shape is invalid")
            return
        end if
        multiplier = 1.0_dp
        if (present(scale)) multiplier = scale
        if (multiplier <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: likelihood scale is invalid")
            return
        end if
        if (.not. ieee_is_finite(multiplier)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: likelihood scale is invalid")
            return
        end if
        call build_projection(self, x, projection, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ensure_noise(self, size(x, 1), status)
        if (status%code /= FORTNUM_OK) return
        allocate(latent(size(x, 1)), mean_gradient(self%n_inducing), &
            factor_gradient(self%n_inducing, self%n_inducing), &
            kl_mean(self%n_inducing), kl_factor(self%n_inducing, self%n_inducing), &
            solve_factor(self%n_inducing, self%n_inducing), tangent_factor(self%n_inducing))
        mean_gradient = 0.0_dp
        factor_gradient = 0.0_dp
        likelihood = 0.0_dp
        do s = 1, self%n_mc
            latent = mean + sqrt(variance)*self%noise(:, s)
            do i = 1, size(x, 1)
                call bernoulli_terms(labels(i), latent(i), self%likelihood, &
                    term, term_gradient)
                likelihood = likelihood + term
                mean_gradient = mean_gradient + term_gradient*projection(:, i)
                tangent_factor = matmul(transpose(self%variational_factor), projection(:, i))
                coefficient = term_gradient*self%noise(i, s)/sqrt(variance(i))
                do j = 1, self%n_inducing
                    factor_gradient(:, j) = factor_gradient(:, j) + &
                        coefficient*tangent_factor(j)*projection(:, i)
                end do
            end do
        end do
        likelihood_scale = multiplier/real(self%n_mc, dp)
        likelihood = likelihood/real(self%n_mc, dp)
        mean_gradient = likelihood_scale*mean_gradient
        factor_gradient = likelihood_scale*factor_gradient

        call variational_kl_gradient(self, kl_mean, kl_factor, solve_factor, status)
        if (status%code /= FORTNUM_OK) return
        mean_gradient = mean_gradient - kl_mean
        factor_gradient = factor_gradient - kl_factor
        gradient(1:self%n_inducing) = mean_gradient
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            gradient(position) = factor_gradient(j, j)*self%variational_factor(j, j)
            position = position + 1
            do i = j + 1, self%n_inducing
                gradient(position) = factor_gradient(i, j)
                position = position + 1
            end do
        end do
        call variational_kl(self, divergence, status)
        if (status%code /= FORTNUM_OK) return
        value = multiplier*likelihood - divergence
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_elbo_gradient

    subroutine gvc_elbo_jvp(self, x, labels, direction, value, tangent, status, scale)
        class(gp_variational_classification_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale
        real(dp), allocatable :: projection(:, :), mean(:), variance(:), latent(:)
        real(dp), allocatable :: mean_tangent(:), factor_tangent(:, :), solve_factor(:, :)
        real(dp), allocatable :: kl_mean(:), kl_factor(:, :), noise_factor(:)
        real(dp) :: multiplier, likelihood, likelihood_tangent, divergence, &
            divergence_tangent, term, term_gradient, latent_tangent, variance_tangent
        integer :: i, j, s, position

        value = 0.0_dp
        tangent = 0.0_dp
        if (.not. valid_data(self, x, labels, status)) return
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: JVP direction is invalid")
            return
        end if
        multiplier = 1.0_dp
        if (present(scale)) multiplier = scale
        if (multiplier <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: likelihood scale is invalid")
            return
        end if
        if (.not. ieee_is_finite(multiplier)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: likelihood scale is invalid")
            return
        end if
        call build_projection(self, x, projection, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ensure_noise(self, size(x, 1), status)
        if (status%code /= FORTNUM_OK) return
        allocate(latent(size(x, 1)), mean_tangent(self%n_inducing), &
            factor_tangent(self%n_inducing, self%n_inducing), &
            solve_factor(self%n_inducing, self%n_inducing), kl_mean(self%n_inducing), &
            kl_factor(self%n_inducing, self%n_inducing), noise_factor(self%n_inducing))
        mean_tangent = direction(1:self%n_inducing)
        factor_tangent = 0.0_dp
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            factor_tangent(j, j) = self%variational_factor(j, j)*direction(position)
            position = position + 1
            do i = j + 1, self%n_inducing
                factor_tangent(i, j) = direction(position)
                position = position + 1
            end do
        end do
        mean_tangent = matmul(transpose(projection), mean_tangent)
        likelihood = 0.0_dp
        likelihood_tangent = 0.0_dp
        do s = 1, self%n_mc
            latent = mean + sqrt(variance)*self%noise(:, s)
            do i = 1, size(x, 1)
                noise_factor = matmul(transpose(self%variational_factor), projection(:, i))
                variance_tangent = 2.0_dp*dot_product(noise_factor, &
                    matmul(transpose(factor_tangent), projection(:, i)))
                latent_tangent = mean_tangent(i) + 0.5_dp*self%noise(i, s)* &
                    variance_tangent/sqrt(variance(i))
                call bernoulli_terms(labels(i), latent(i), self%likelihood, &
                    term, term_gradient)
                likelihood = likelihood + term
                likelihood_tangent = likelihood_tangent + term_gradient*latent_tangent
            end do
        end do
        likelihood = likelihood/real(self%n_mc, dp)
        likelihood_tangent = multiplier*likelihood_tangent/real(self%n_mc, dp)
        call variational_kl(self, divergence, status)
        if (status%code /= FORTNUM_OK) return
        call variational_kl_gradient(self, kl_mean, kl_factor, solve_factor, status)
        if (status%code /= FORTNUM_OK) return
        divergence_tangent = dot_product(kl_mean, direction(1:self%n_inducing))
        do j = 1, self%n_inducing
            divergence_tangent = divergence_tangent + &
                dot_product(kl_factor(j:, j), factor_tangent(j:, j))
        end do
        value = multiplier*likelihood - divergence
        tangent = likelihood_tangent - divergence_tangent
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_elbo_jvp

    !> Return the variational posterior marginal at query points.
    !!
    !! The fitted inducing state is held fixed.  The returned variance is the
    !! diagonal of the inducing posterior approximation, including the prior
    !! Schur complement.  This is a prediction primitive rather than an
    !! optimization step; callers own the update of ``parameters()``.
    subroutine gvc_predict_latent(self, x, mean, variance, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: projection(:, :), local_mean(:), local_variance(:)

        if (.not. prediction_valid(self, x, mean, variance, status)) return
        call build_projection(self, x, projection, local_mean, local_variance, status)
        if (status%code /= FORTNUM_OK) return
        mean = local_mean
        variance = local_variance
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_latent

    !> Predict Bernoulli probabilities from variational latent marginals.
    !! Columns are ``[negative, positive]``.  Logistic uses the standard
    !! variance correction and probit uses the analytic Gaussian integral.
    subroutine gvc_predict_proba(self, x, probabilities, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        real(dp) :: p, scale
        integer :: i

        if (.not. prediction_probability_valid(self, x, probabilities, status)) return
        allocate(mean(size(x, 1)), variance(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            if (self%likelihood == GP_VARIATIONAL_PROBIT) then
                scale = sqrt(1.0_dp + variance(i))
                p = 0.5_dp*erfc(-mean(i)/(scale*SQRT_TWO))
            else
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p = 1.0_dp/(1.0_dp + exp(-mean(i)/scale))
            end if
            probabilities(i, 2) = p
            probabilities(i, 1) = 1.0_dp - p
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_proba

    !> Forward product of latent prediction with respect to packed variational
    !! parameters.  The query points and kernel hyperparameters are held fixed.
    subroutine gvc_predict_latent_parameter_jvp(self, x, direction, mean, &
            mean_dot, variance, variance_dot, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: projection(:, :), local_mean(:), local_variance(:)
        real(dp), allocatable :: mean_tangent(:), factor_tangent(:, :), noise_factor(:)
        integer :: i, j, position

        if (.not. prediction_valid(self, x, mean, variance, status)) return
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. &
            size(mean_dot) /= size(mean) .or. size(variance_dot) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP prediction JVP: parameter direction or output shape is invalid")
            return
        end if
        call build_projection(self, x, projection, local_mean, local_variance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(mean_tangent(self%n_inducing), factor_tangent(self%n_inducing, &
            self%n_inducing), noise_factor(self%n_inducing))
        mean_tangent = direction(1:self%n_inducing)
        factor_tangent = 0.0_dp
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            factor_tangent(j, j) = self%variational_factor(j, j)*direction(position)
            position = position + 1
            do i = j + 1, self%n_inducing
                factor_tangent(i, j) = direction(position)
                position = position + 1
            end do
        end do
        mean = local_mean
        variance = local_variance
        mean_dot = matmul(transpose(projection), mean_tangent)
        do i = 1, size(x, 1)
            noise_factor = matmul(transpose(self%variational_factor), projection(:, i))
            variance_dot(i) = 2.0_dp*dot_product(noise_factor, &
                matmul(transpose(factor_tangent), projection(:, i)))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_latent_parameter_jvp

    !> Forward product of Bernoulli probabilities with respect to packed
    !! variational parameters.
    subroutine gvc_predict_proba_parameter_jvp(self, x, direction, probabilities, &
            probabilities_dot, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        real(dp) :: p, p_mu, p_variance, scale, z, density
        integer :: i

        if (.not. prediction_probability_valid(self, x, probabilities, status)) return
        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP probability JVP: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_parameter_jvp(x, direction, mean, mean_dot, &
            variance, variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            if (self%likelihood == GP_VARIATIONAL_PROBIT) then
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p = 0.5_dp*erfc(-z/SQRT_TWO)
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p = 1.0_dp/(1.0_dp + exp(-mean(i)/scale))
                p_mu = p*(1.0_dp-p)/scale
                p_variance = p*(1.0_dp-p)*(-mean(i)*PI/(16.0_dp*scale**3))
            end if
            probabilities(i, 2) = p
            probabilities(i, 1) = 1.0_dp - p
            probabilities_dot(i, 2) = p_mu*mean_dot(i) + p_variance*variance_dot(i)
            probabilities_dot(i, 1) = -probabilities_dot(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_proba_parameter_jvp

    !> Reverse product of the latent prediction with respect to packed
    !! variational parameters.  Query points and the inducing prior are fixed.
    !! The packed ordering is exactly ``parameters()``: posterior mean,
    !! log-diagonal Cholesky entries, then strict lower-triangular entries.
    subroutine gvc_predict_latent_parameter_vjp(self, x, mean_bar, variance_bar, &
            parameter_bar, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: projection(:, :), local_mean(:), local_variance(:)
        real(dp), allocatable :: factor_bar(:, :), tangent(:)
        integer :: i, j, position

        parameter_bar = 0.0_dp
        if (.not. latent_vjp_valid(self, x, mean_bar, variance_bar, parameter_bar, status)) return
        call build_projection(self, x, projection, local_mean, local_variance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(factor_bar(self%n_inducing, self%n_inducing), tangent(self%n_inducing))
        parameter_bar(1:self%n_inducing) = matmul(projection, mean_bar)
        factor_bar = 0.0_dp
        do i = 1, size(x, 1)
            tangent = matmul(transpose(self%variational_factor), projection(:, i))
            do j = 1, self%n_inducing
                factor_bar(:, j) = factor_bar(:, j) + &
                    2.0_dp*variance_bar(i)*projection(:, i)*tangent(j)
            end do
        end do
        position = self%n_inducing + 1
        do j = 1, self%n_inducing
            parameter_bar(position) = self%variational_factor(j, j)*factor_bar(j, j)
            position = position + 1
            do i = j + 1, self%n_inducing
                parameter_bar(position) = factor_bar(i, j)
                position = position + 1
            end do
        end do
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP latent VJP: nonfinite parameter cotangent")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_latent_parameter_vjp

    !> Reverse product of Bernoulli predictive probabilities with respect to
    !! packed variational parameters.  Both probability columns may carry a
    !! cotangent; the implementation reduces them to the positive-column
    !! scalar derivative and then applies the exact latent reverse product.
    subroutine gvc_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            parameter_bar, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)
        real(dp) :: scale, z, density, p, p_mu, p_variance
        integer :: i

        parameter_bar = 0.0_dp
        if (.not. prediction_probability_cotangent_valid(self, x, probabilities_bar, &
            parameter_bar, status)) return
        allocate(mean(size(x, 1)), variance(size(x, 1)), mean_bar(size(x, 1)), &
            variance_bar(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        mean_bar = 0.0_dp
        variance_bar = 0.0_dp
        do i = 1, size(x, 1)
            if (self%likelihood == GP_VARIATIONAL_PROBIT) then
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p = 0.5_dp*erfc(-z/SQRT_TWO)
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p = 1.0_dp/(1.0_dp + exp(-mean(i)/scale))
                p_mu = p*(1.0_dp-p)/scale
                p_variance = p*(1.0_dp-p)*(-mean(i)*PI/(16.0_dp*scale**3))
            end if
            mean_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*p_mu
            variance_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*p_variance
        end do
        call self%predict_latent_parameter_vjp(x, mean_bar, variance_bar, parameter_bar, status)
    end subroutine gvc_predict_proba_parameter_vjp

    integer function gvc_kernel_parameter_count(self) result(count)
        class(gp_variational_classification_t), intent(in) :: self

        if (self%n_inducing < 1) then
            count = 0
        else
            count = self%kernel%parameter_count()
        end if
    end function gvc_kernel_parameter_count

    !> Forward product of a latent prediction with respect to kernel
    !! log-hyperparameters.  The inducing points, variational mean/factor,
    !! and posterior state remain fixed.  The product includes all three
    !! kernel blocks used by the posterior projection: K_uu, K_ux, and the
    !! diagonal of K_xx.
    subroutine gvc_predict_latent_kernel_parameter_jvp(self, x, direction, mean, &
            mean_dot, variance, variance_dot, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: projection(:, :), local_mean(:), local_variance(:)
        real(dp), allocatable :: k_ux(:, :), k_ux_dot(:, :)
        real(dp), allocatable :: k_uu(:, :), k_uu_dot(:, :), k_xx(:, :), k_xx_dot(:, :)
        real(dp), allocatable :: projection_dot(:, :), factor(:), factor_dot(:)
        integer :: i

        if (.not. prediction_valid(self, x, mean, variance, status)) return
        if (size(direction) /= self%kernel%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. size(mean_dot) /= size(mean) .or. &
            size(variance_dot) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel JVP: direction or output shape is invalid")
            return
        end if
        call build_projection(self, x, projection, local_mean, local_variance, status)
        if (status%code /= FORTNUM_OK) return
        mean = local_mean
        variance = local_variance
        allocate(k_ux(self%n_inducing, size(x, 1)), k_ux_dot(self%n_inducing, size(x, 1)))
        allocate(k_uu(self%n_inducing, self%n_inducing), &
            k_uu_dot(self%n_inducing, self%n_inducing))
        allocate(k_xx(size(x, 1), size(x, 1)), k_xx_dot(size(x, 1), size(x, 1)))
        call self%kernel%matrix_jvp(self%inducing_points, x, direction, k_ux, k_ux_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix_jvp(self%inducing_points, self%inducing_points, direction, &
            k_uu, k_uu_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix_jvp(x, x, direction, k_xx, k_xx_dot, status)
        if (status%code /= FORTNUM_OK) return
        allocate(projection_dot(self%n_inducing, size(x, 1)))
        projection_dot = k_ux_dot - matmul(k_uu_dot, projection)
        call self%prior_factorization%solve(projection_dot, status)
        if (status%code /= FORTNUM_OK) return
        allocate(factor(self%n_inducing), factor_dot(self%n_inducing))
        mean_dot = matmul(transpose(projection_dot), self%variational_mean)
        do i = 1, size(x, 1)
            factor = matmul(transpose(self%variational_factor), projection(:, i))
            factor_dot = matmul(transpose(self%variational_factor), projection_dot(:, i))
            variance_dot(i) = k_xx_dot(i, i) - dot_product(projection_dot(:, i), k_ux(:, i)) - &
                dot_product(projection(:, i), k_ux_dot(:, i)) + 2.0_dp*dot_product(factor, factor_dot)
        end do
        if (any(.not. ieee_is_finite(mean_dot)) .or. any(.not. ieee_is_finite(variance_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel JVP: nonfinite tangent")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_latent_kernel_parameter_jvp

    !> Forward product of Bernoulli probabilities with respect to kernel
    !! log-hyperparameters, including the variance correction used by the
    !! selected logistic or probit likelihood.
    subroutine gvc_predict_proba_kernel_parameter_jvp(self, x, direction, probabilities, &
            probabilities_dot, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        real(dp) :: p, p_mu, p_variance, scale, z, density
        integer :: i

        if (.not. prediction_probability_valid(self, x, probabilities, status)) return
        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel probability JVP: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_kernel_parameter_jvp(x, direction, mean, mean_dot, &
            variance, variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            if (self%likelihood == GP_VARIATIONAL_PROBIT) then
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p = 0.5_dp*erfc(-z/SQRT_TWO)
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p = 1.0_dp/(1.0_dp + exp(-mean(i)/scale))
                p_mu = p*(1.0_dp-p)/scale
                p_variance = p*(1.0_dp-p)*(-mean(i)*PI/(16.0_dp*scale**3))
            end if
            probabilities(i, 2) = p
            probabilities(i, 1) = 1.0_dp - p
            probabilities_dot(i, 2) = p_mu*mean_dot(i) + p_variance*variance_dot(i)
            probabilities_dot(i, 1) = -probabilities_dot(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_proba_kernel_parameter_jvp

    !> Reverse product of latent predictions with respect to kernel
    !! log-hyperparameters.  This is the exact adjoint of the fixed-state
    !! projection, including the K_uu solve and diagonal K_xx term.
    subroutine gvc_predict_latent_kernel_parameter_vjp(self, x, mean_bar, variance_bar, &
            parameter_bar, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: projection(:, :), mean(:), variance(:), k_ux(:, :)
        real(dp), allocatable :: projection_bar(:, :), solve_bar(:, :), k_ux_bar(:, :)
        real(dp), allocatable :: k_uu_bar(:, :), k_xx_bar(:, :), factor(:), local_bar(:)
        integer :: i

        parameter_bar = 0.0_dp
        if (.not. latent_kernel_vjp_valid(self, x, mean_bar, variance_bar, parameter_bar, &
            status)) return
        call build_projection(self, x, projection, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(k_ux(self%n_inducing, size(x, 1)), projection_bar(size(projection, 1), size(projection, 2)))
        call self%kernel%matrix(self%inducing_points, x, k_ux, status)
        if (status%code /= FORTNUM_OK) return
        allocate(k_ux_bar(self%n_inducing, size(x, 1)), &
            solve_bar(self%n_inducing, size(x, 1)), k_uu_bar(self%n_inducing, self%n_inducing), &
            k_xx_bar(size(x, 1), size(x, 1)), factor(self%n_inducing))
        projection_bar = 0.0_dp
        k_ux_bar = 0.0_dp
        k_xx_bar = 0.0_dp
        do i = 1, size(x, 1)
            factor = matmul(transpose(self%variational_factor), projection(:, i))
            projection_bar(:, i) = mean_bar(i)*self%variational_mean + variance_bar(i)* &
                (-k_ux(:, i) + 2.0_dp*matmul(self%variational_factor, factor))
            k_ux_bar(:, i) = -variance_bar(i)*projection(:, i)
            k_xx_bar(i, i) = variance_bar(i)
        end do
        solve_bar = projection_bar
        call self%prior_factorization%solve(solve_bar, status)
        if (status%code /= FORTNUM_OK) return
        k_ux_bar = k_ux_bar + solve_bar
        k_uu_bar = -matmul(solve_bar, transpose(projection))
        allocate(local_bar(size(parameter_bar)))
        call self%kernel%parameter_vjp(self%inducing_points, self%inducing_points, &
            k_uu_bar, parameter_bar, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%parameter_vjp(self%inducing_points, x, k_ux_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = parameter_bar + local_bar
        call self%kernel%parameter_vjp(x, x, k_xx_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = parameter_bar + local_bar
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel VJP: nonfinite parameter cotangent")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_latent_kernel_parameter_vjp

    !> Reverse product of Bernoulli probabilities with respect to kernel
    !! log-hyperparameters.  The likelihood derivative is reduced to latent
    !! mean/variance cotangents before applying the exact kernel adjoint.
    subroutine gvc_predict_proba_kernel_parameter_vjp(self, x, probabilities_bar, &
            parameter_bar, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)
        real(dp) :: scale, z, density, p, p_mu, p_variance
        integer :: i

        parameter_bar = 0.0_dp
        if (.not. prediction_probability_kernel_vjp_valid(self, x, probabilities_bar, &
            parameter_bar, status)) return
        allocate(mean(size(x, 1)), variance(size(x, 1)), mean_bar(size(x, 1)), &
            variance_bar(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            if (self%likelihood == GP_VARIATIONAL_PROBIT) then
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p = 0.5_dp*erfc(-z/SQRT_TWO)
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p = 1.0_dp/(1.0_dp + exp(-mean(i)/scale))
                p_mu = p*(1.0_dp-p)/scale
                p_variance = p*(1.0_dp-p)*(-mean(i)*PI/(16.0_dp*scale**3))
            end if
            mean_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*p_mu
            variance_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*p_variance
        end do
        call self%predict_latent_kernel_parameter_vjp(x, mean_bar, variance_bar, &
            parameter_bar, status)
    end subroutine gvc_predict_proba_kernel_parameter_vjp

    !> Device boundary for the variational-GP kernel reverse product.
    !! CUDA remains explicitly refused until the inducing projection and
    !! kernel-product graph are resident; CPU dispatch is exact.
    subroutine gvc_predict_proba_kernel_parameter_vjp_device(self, device, x, &
            probabilities_bar, parameter_bar, status)
        class(gp_variational_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_kernel_parameter_vjp(x, probabilities_bar, parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "variational GP kernel VJP device: resident CUDA graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel VJP device: device kind is invalid")
        end select
    end subroutine gvc_predict_proba_kernel_parameter_vjp_device

    !> Forward product of the variational latent prediction with respect to
    !! query coordinates.  The inducing points, variational state, and kernel
    !! hyperparameters remain fixed.  Kernel first input derivatives are used
    !! directly; unsupported/nonsmooth kernel leaves return their typed status.
    subroutine gvc_predict_latent_input_jvp(self, x, x_dot, mean, mean_dot, &
            variance, variance_dot, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: projection(:, :), local_mean(:), local_variance(:)
        real(dp), allocatable :: k_ux(:, :), projection_dot(:, :), k_dot(:)
        real(dp), allocatable :: noise_factor(:), noise_factor_dot(:)
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        real(dp) :: value, diagonal_dot
        integer :: i, j

        if (.not. prediction_valid(self, x, mean, variance, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. size(mean_dot) /= size(mean) .or. &
                size(variance_dot) /= size(variance) .or. &
                any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP input JVP: input tangent or output shape is invalid")
            return
        end if
        call build_projection(self, x, projection, local_mean, local_variance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(k_ux(self%n_inducing, size(x, 1)), projection_dot(self%n_inducing, size(x, 1)), &
            k_dot(self%n_inducing), noise_factor(self%n_inducing), &
            noise_factor_dot(self%n_inducing), gradient_x1(self%kernel%input_dim), &
            gradient_x2(self%kernel%input_dim), mixed_hessian(self%kernel%input_dim, &
                self%kernel%input_dim))
        call self%kernel%matrix(self%inducing_points, x, k_ux, status)
        if (status%code /= FORTNUM_OK) return
        mean = local_mean
        variance = local_variance
        do i = 1, size(x, 1)
            do j = 1, self%n_inducing
                call self%kernel%input_derivatives(self%inducing_points(j, :), x(i, :), &
                    value, gradient_x1, gradient_x2, mixed_hessian, status)
                if (status%code /= FORTNUM_OK) return
                k_dot(j) = dot_product(gradient_x2, x_dot(i, :))
            end do
            projection_dot(:, i) = k_dot
            call self%prior_factorization%solve(projection_dot(:, i), status)
            if (status%code /= FORTNUM_OK) return
            mean_dot(i) = dot_product(projection_dot(:, i), self%variational_mean)
            noise_factor = matmul(transpose(self%variational_factor), projection(:, i))
            noise_factor_dot = matmul(transpose(self%variational_factor), projection_dot(:, i))
            variance_dot(i) = -dot_product(projection_dot(:, i), k_ux(:, i)) - &
                dot_product(projection(:, i), k_dot) + 2.0_dp*dot_product(noise_factor, &
                noise_factor_dot)
            call self%kernel%input_derivatives(x(i, :), x(i, :), value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
            if (status%code /= FORTNUM_OK) return
            diagonal_dot = dot_product(gradient_x1 + gradient_x2, x_dot(i, :))
            variance_dot(i) = variance_dot(i) + diagonal_dot
        end do
        if (any(.not. ieee_is_finite(mean_dot)) .or. any(.not. ieee_is_finite(variance_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP input JVP: nonfinite tangent")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_latent_input_jvp

    !> Reverse product of the variational latent prediction with respect to
    !! query coordinates.  This is the adjoint of
    !! `predict_latent_input_jvp` and has no finite-difference fallback.
    subroutine gvc_predict_latent_input_vjp(self, x, mean_bar, variance_bar, &
            x_bar, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: projection(:, :), local_mean(:), local_variance(:)
        real(dp), allocatable :: k_ux(:, :), projection_bar(:), k_bar(:), noise_factor(:)
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        real(dp) :: value
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. latent_input_vjp_valid(self, x, mean_bar, variance_bar, x_bar, status)) return
        call build_projection(self, x, projection, local_mean, local_variance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(k_ux(self%n_inducing, size(x, 1)), projection_bar(self%n_inducing), &
            k_bar(self%n_inducing), noise_factor(self%n_inducing), &
            gradient_x1(self%kernel%input_dim), gradient_x2(self%kernel%input_dim), &
            mixed_hessian(self%kernel%input_dim, self%kernel%input_dim))
        call self%kernel%matrix(self%inducing_points, x, k_ux, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            projection_bar = mean_bar(i)*self%variational_mean - variance_bar(i)*k_ux(:, i)
            noise_factor = matmul(transpose(self%variational_factor), projection(:, i))
            projection_bar = projection_bar + 2.0_dp*variance_bar(i)* &
                matmul(self%variational_factor, noise_factor)
            ! `k_ux` occurs directly in the Schur-complement term
            ! `-P^T k_ux` as well as indirectly through
            ! `P = K_uu^{-1} k_ux`; retain both reverse contributions.
            k_bar = -variance_bar(i)*projection(:, i)
            call self%prior_factorization%solve(projection_bar, status)
            if (status%code /= FORTNUM_OK) return
            do j = 1, self%n_inducing
                call self%kernel%input_derivatives(self%inducing_points(j, :), x(i, :), &
                    value, gradient_x1, gradient_x2, mixed_hessian, status)
                if (status%code /= FORTNUM_OK) return
                x_bar(i, :) = x_bar(i, :) + (k_bar(j) + projection_bar(j))*gradient_x2
            end do
            call self%kernel%input_derivatives(x(i, :), x(i, :), value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
            if (status%code /= FORTNUM_OK) return
            x_bar(i, :) = x_bar(i, :) + variance_bar(i)*(gradient_x1 + gradient_x2)
        end do
        if (any(.not. ieee_is_finite(x_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP input VJP: nonfinite input cotangent")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_latent_input_vjp

    !> Probability wrapper for the query-input JVP.
    subroutine gvc_predict_proba_input_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        real(dp) :: p, p_mu, p_variance, scale, z, density
        integer :: i

        if (.not. prediction_probability_valid(self, x, probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. any(shape(probabilities_dot) /= &
                shape(probabilities)) .or. any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP probability input JVP: input tangent or output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_input_jvp(x, x_dot, mean, mean_dot, variance, variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            if (self%likelihood == GP_VARIATIONAL_PROBIT) then
                scale = sqrt(1.0_dp + variance(i)); z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p = 0.5_dp*erfc(-z/SQRT_TWO)
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p = 1.0_dp/(1.0_dp + exp(-mean(i)/scale))
                p_mu = p*(1.0_dp-p)/scale
                p_variance = p*(1.0_dp-p)*(-mean(i)*PI/(16.0_dp*scale**3))
            end if
            probabilities(i, 2) = p
            probabilities(i, 1) = 1.0_dp - p
            probabilities_dot(i, 2) = p_mu*mean_dot(i) + p_variance*variance_dot(i)
            probabilities_dot(i, 1) = -probabilities_dot(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvc_predict_proba_input_jvp

    !> Probability wrapper for the query-input VJP.
    subroutine gvc_predict_proba_input_vjp(self, x, probabilities_bar, x_bar, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)
        real(dp) :: scale, z, density, p, p_mu, p_variance
        integer :: i

        x_bar = 0.0_dp
        if (.not. prediction_probability_input_vjp_valid(self, x, probabilities_bar, &
                x_bar, status)) return
        allocate(mean(size(x, 1)), variance(size(x, 1)), mean_bar(size(x, 1)), &
            variance_bar(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            if (self%likelihood == GP_VARIATIONAL_PROBIT) then
                scale = sqrt(1.0_dp + variance(i)); z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p = 0.5_dp*erfc(-z/SQRT_TWO)
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                p = 1.0_dp/(1.0_dp + exp(-mean(i)/scale))
                p_mu = p*(1.0_dp-p)/scale
                p_variance = p*(1.0_dp-p)*(-mean(i)*PI/(16.0_dp*scale**3))
            end if
            mean_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*p_mu
            variance_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*p_variance
        end do
        call self%predict_latent_input_vjp(x, mean_bar, variance_bar, x_bar, status)
    end subroutine gvc_predict_proba_input_vjp

    !> Device boundary for the query-input probability VJP.
    subroutine gvc_predict_proba_input_vjp_device(self, device, x, probabilities_bar, &
            x_bar, status)
        class(gp_variational_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        x_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP input VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_input_vjp(x, probabilities_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "variational GP input VJP device: resident CUDA graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP input VJP device: device kind is invalid")
        end select
    end subroutine gvc_predict_proba_input_vjp_device

    !> Explicit backend boundary for the predictive reverse product.  CUDA is
    !! refused until a resident inducing solve and probability-reverse kernel
    !! are linked; no hidden host fallback is permitted.
    subroutine gvc_predict_proba_parameter_vjp_device(self, device, x, probabilities_bar, &
            parameter_bar, status)
        class(gp_variational_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_vjp(x, probabilities_bar, parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "variational GP classification VJP device: resident CUDA graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification VJP device: device kind is invalid")
        end select
    end subroutine gvc_predict_proba_parameter_vjp_device

    subroutine gvc_elbo_device(self, device, x, labels, value, status, scale)
        class(gp_variational_classification_t), intent(inout) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale

        value = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            if (present(scale)) then
                call self%elbo(x, labels, value, status, scale=scale)
            else
                call self%elbo(x, labels, value, status)
            end if
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "variational GP classification device: resident CUDA inducing graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification device: device kind is invalid")
        end select
    end subroutine gvc_elbo_device

    logical function gvc_device_supported(self, device_kind) result(supported)
        class(gp_variational_classification_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function gvc_device_supported

    subroutine build_projection(self, x, projection, mean, variance, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: projection(:, :), mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: k_ux(:, :), covariance_tangent(:)
        real(dp) :: diagonal, marginal
        integer :: i

        allocate(k_ux(self%n_inducing, size(x, 1)), projection(self%n_inducing, size(x, 1)))
        call self%kernel%matrix(self%inducing_points, x, k_ux, status)
        if (status%code /= FORTNUM_OK) return
        projection = k_ux
        call self%prior_factorization%solve(projection, status)
        if (status%code /= FORTNUM_OK) return
        allocate(mean(size(x, 1)), variance(size(x, 1)), covariance_tangent(self%n_inducing))
        mean = matmul(transpose(projection), self%variational_mean)
        do i = 1, size(x, 1)
            diagonal = self%kernel%value(x(i, :), x(i, :))
            marginal = diagonal - dot_product(projection(:, i), k_ux(:, i))
            covariance_tangent = matmul(transpose(self%variational_factor), projection(:, i))
            marginal = marginal + dot_product(covariance_tangent, covariance_tangent)
            if (.not. ieee_is_finite(marginal) .or. marginal <= 0.0_dp) then
                if (marginal < -1.0e-8_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "variational GP classification: latent variance is invalid")
                    return
                end if
                marginal = tiny(1.0_dp)
            end if
            variance(i) = marginal
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_projection

    subroutine variational_kl(self, value, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: solve_mean(:), solve_factor(:, :)
        real(dp) :: log_det_prior, log_det_posterior
        integer :: j

        value = 0.0_dp
        if (self%n_inducing < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: model is not initialized")
            return
        end if
        allocate(solve_mean(self%n_inducing), solve_factor(self%n_inducing, self%n_inducing))
        solve_mean = self%variational_mean
        solve_factor = self%variational_factor
        call self%prior_factorization%solve(solve_mean, status)
        if (status%code /= FORTNUM_OK) return
        call self%prior_factorization%solve(solve_factor, status)
        if (status%code /= FORTNUM_OK) return
        call self%prior_factorization%log_determinant(log_det_prior, status)
        if (status%code /= FORTNUM_OK) return
        log_det_posterior = 0.0_dp
        do j = 1, self%n_inducing
            log_det_posterior = log_det_posterior + &
                2.0_dp*log(self%variational_factor(j, j))
        end do
        value = 0.5_dp*(sum(solve_factor*self%variational_factor) + &
            dot_product(self%variational_mean, solve_mean) - &
            real(self%n_inducing, dp) + log_det_prior - log_det_posterior)
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: KL is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine variational_kl

    subroutine variational_kl_gradient(self, mean_gradient, factor_gradient, &
            solve_factor, status)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(out) :: mean_gradient(:), factor_gradient(:, :), solve_factor(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: solve_mean(:), inverse_factor(:, :)
        integer :: i, j

        if (size(mean_gradient) /= self%n_inducing .or. &
            any(shape(factor_gradient) /= [self%n_inducing, self%n_inducing])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: KL gradient shape is invalid")
            return
        end if
        allocate(solve_mean(self%n_inducing), inverse_factor(self%n_inducing, self%n_inducing))
        solve_mean = self%variational_mean
        solve_factor = self%variational_factor
        call self%prior_factorization%solve(solve_mean, status)
        if (status%code /= FORTNUM_OK) return
        call self%prior_factorization%solve(solve_factor, status)
        if (status%code /= FORTNUM_OK) return
        inverse_factor = 0.0_dp
        do j = 1, self%n_inducing
            inverse_factor(j, j) = 1.0_dp
        end do
        call triangular_inverse(self%variational_factor, inverse_factor, status)
        if (status%code /= FORTNUM_OK) return
        mean_gradient = solve_mean
        factor_gradient = solve_factor - transpose(inverse_factor)
        do j = 1, self%n_inducing
            do i = j + 1, self%n_inducing
                factor_gradient(j, i) = 0.0_dp
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine variational_kl_gradient

    subroutine triangular_inverse(factor, inverse, status)
        real(dp), intent(in) :: factor(:, :)
        real(dp), intent(inout) :: inverse(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: n, i, j
        real(dp) :: diagonal

        n = size(factor, 1)
        if (size(factor, 2) /= n .or. any(shape(inverse) /= [n, n])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: triangular inverse shape is invalid")
            return
        end if
        do i = 1, n
            diagonal = factor(i, i)
            if (diagonal <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP classification: covariance factor is invalid")
                return
            end if
            if (.not. ieee_is_finite(diagonal)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP classification: covariance factor is invalid")
                return
            end if
        end do
        ! Solve each unit vector explicitly.  The inverse is lower triangular.
        inverse = 0.0_dp
        do j = 1, n
            inverse(j, j) = 1.0_dp/factor(j, j)
            do i = j + 1, n
                inverse(i, j) = -sum(factor(i, j:i-1)*inverse(j:i-1, j))/factor(i, i)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine triangular_inverse

    subroutine ensure_noise(self, n_samples, status)
        class(gp_variational_classification_t), intent(inout) :: self
        integer, intent(in) :: n_samples
        type(fortnum_status_t), intent(out) :: status
        type(rng_t) :: generator
        integer :: i, s

        if (n_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: sample count is invalid")
            return
        end if
        if (self%noise_samples == n_samples .and. allocated(self%noise)) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (allocated(self%noise)) deallocate(self%noise)
        allocate(self%noise(n_samples, self%n_mc))
        call rng_seed(generator, int(self%seed, int64), status)
        if (status%code /= FORTNUM_OK) return
        do s = 1, self%n_mc
            do i = 1, n_samples
                call rng_normal(generator, self%noise(i, s))
            end do
        end do
        self%noise_samples = n_samples
        call status_set(status, FORTNUM_OK, "")
    end subroutine ensure_noise

    logical function valid_data(self, x, labels, status) result(valid)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        valid = .false.
        if (self%n_inducing < 1 .or. .not. allocated(self%variational_mean)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%kernel%input_dim .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP classification: data shape or values are invalid")
            return
        end if
        do i = 1, size(labels)
            if (labels(i) /= 0 .and. labels(i) /= 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP classification: labels must be zero or one")
                return
            end if
        end do
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_data

    logical function prediction_valid(self, x, mean, variance, status) result(valid)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (self%n_inducing < 1 .or. .not. allocated(self%variational_mean)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP prediction: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%kernel%input_dim .or. &
            size(mean) /= size(x, 1) .or. size(variance) /= size(x, 1) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP prediction: input or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_valid

    logical function prediction_probability_valid(self, x, probabilities, status) &
            result(valid)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. allocated(self%variational_mean) .or. self%n_inducing < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP probability prediction: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%kernel%input_dim .or. &
            any(shape(probabilities) /= [size(x, 1), 2]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP probability prediction: input or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_probability_valid

    logical function latent_vjp_valid(self, x, mean_bar, variance_bar, parameter_bar, status) &
            result(valid)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:), parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. allocated(self%variational_mean) .or. self%n_inducing < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP latent VJP: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%kernel%input_dim .or. &
            size(mean_bar) /= size(x, 1) .or. size(variance_bar) /= size(x, 1) .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(mean_bar)) .or. &
            any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP latent VJP: input, cotangent, or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function latent_vjp_valid

    logical function prediction_probability_cotangent_valid(self, x, probabilities_bar, &
            parameter_bar, status) result(valid)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :), parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (size(x, 1) < 1 .or. size(x, 2) /= self%kernel%input_dim .or. &
            any(shape(probabilities_bar) /= [size(x, 1), 2]) .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP probability VJP: input, cotangent, or output shape is invalid")
            return
        end if
        if (.not. allocated(self%variational_mean) .or. self%n_inducing < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP probability VJP: model is not initialized")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_probability_cotangent_valid

    logical function latent_kernel_vjp_valid(self, x, mean_bar, variance_bar, &
            parameter_bar, status) result(valid)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:), parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (self%n_inducing < 1 .or. .not. allocated(self%variational_mean)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel VJP: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%kernel%input_dim .or. &
            size(mean_bar) /= size(x, 1) .or. size(variance_bar) /= size(x, 1) .or. &
            size(parameter_bar) /= self%kernel%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(mean_bar)) .or. &
            any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel VJP: input, cotangent, or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function latent_kernel_vjp_valid

    logical function prediction_probability_kernel_vjp_valid(self, x, probabilities_bar, &
            parameter_bar, status) result(valid)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :), parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (self%n_inducing < 1 .or. .not. allocated(self%variational_mean)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel probability VJP: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%kernel%input_dim .or. &
            any(shape(probabilities_bar) /= [size(x, 1), 2]) .or. &
            size(parameter_bar) /= self%kernel%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP kernel probability VJP: input, cotangent, or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_probability_kernel_vjp_valid

    logical function latent_input_vjp_valid(self, x, mean_bar, variance_bar, x_bar, status) &
            result(valid)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. allocated(self%variational_mean) .or. self%n_inducing < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP input VJP: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%kernel%input_dim .or. &
                size(mean_bar) /= size(x, 1) .or. size(variance_bar) /= size(x, 1) .or. &
                any(shape(x_bar) /= shape(x)) .or. any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(mean_bar)) .or. &
                any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP input VJP: input, cotangent, or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function latent_input_vjp_valid

    logical function prediction_probability_input_vjp_valid(self, x, probabilities_bar, &
            x_bar, status) result(valid)
        class(gp_variational_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. allocated(self%variational_mean) .or. self%n_inducing < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP probability input VJP: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%kernel%input_dim .or. &
                any(shape(probabilities_bar) /= [size(x, 1), 2]) .or. &
                any(shape(x_bar) /= shape(x)) .or. any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP probability input VJP: input, cotangent, or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_probability_input_vjp_valid

    subroutine bernoulli_terms(label, latent, likelihood, value, gradient)
        integer, intent(in) :: label, likelihood
        real(dp), intent(in) :: latent
        real(dp), intent(out) :: value, gradient
        real(dp) :: eta, probability, density

        eta = real(2*label - 1, dp)*latent
        if (likelihood == GP_VARIATIONAL_PROBIT) then
            probability = 0.5_dp*erfc(-eta/SQRT_TWO)
            probability = max(probability, tiny(1.0_dp))
            density = exp(-0.5_dp*eta*eta)/SQRT_TWO_PI
            value = log(probability)
            gradient = real(2*label - 1, dp)*density/probability
        else
            if (eta > 40.0_dp) then
                value = -exp(-eta)
                probability = exp(-eta)
            else if (eta < -40.0_dp) then
                value = eta
                probability = 1.0_dp
            else if (eta >= 0.0_dp) then
                probability = 1.0_dp/(1.0_dp + exp(eta))
                value = -log(1.0_dp + exp(-eta))
            else
                probability = 1.0_dp/(1.0_dp + exp(eta))
                value = eta - log(1.0_dp + exp(eta))
            end if
            gradient = real(2*label - 1, dp)*probability
        end if
    end subroutine bernoulli_terms

end module fortml_gp_variational_classification
