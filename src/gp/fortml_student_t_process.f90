module fortml_student_t_process
    !! Student-t process regression.
    !!
    !! Written against Shah, Wilson and Ghahramani, *Student-t Processes as
    !! Alternatives to Gaussian Processes* (arXiv:1402.4306), fetched by
    !! `fortml-bench/scripts/fetch_literature.py`.
    !!
    !! A Student-t process arises from placing an inverse Wishart prior on a
    !! GP's covariance and marginalizing it out. The result keeps everything
    !! that makes a GP usable — analytic marginals, analytic conditionals,
    !! consistency under marginalization — and changes one thing that matters:
    !!
    !! **The predictive covariance depends on the observed values.** A GP's does
    !! not. Conditioning a TP on `n1` observations gives
    !!
    !!     y2 | y1 ~ MVT(nu + n1, phi_tilde,
    !!                   (nu + beta1 - 2)/(nu + n1 - 2) * K_tilde),
    !!
    !! with `beta1 = (y1 - phi1)' K11^-1 (y1 - phi1)` the Mahalanobis distance
    !! of what was actually seen. Since `E[beta1] = n1`, data that turns out
    !! more surprising than the prior expected scales the predictive covariance
    !! *up*, and unusually consistent data scales it *down*. A GP cannot express
    !! either: its posterior variance is fixed once the inputs are chosen,
    !! whatever the outputs turn out to be.
    !!
    !! **The parameterization is the paper's, and it is not the usual one.**
    !! `MVT(nu, phi, K)` here has `cov[y] = K` exactly — the generative form is
    !! written as `y|sigma ~ GP(phi, (nu - 2) sigma)` precisely so that comes
    !! out. Most references parameterize with a scale matrix `Sigma` where
    !! `cov = nu/(nu - 2) Sigma`, and carrying that convention across would
    !! inflate every predictive variance by a factor that vanishes only as `nu`
    !! grows. The tests pin the convention by checking the large-`nu` limit
    !! against an exact GP.
    !!
    !! `nu > 2` is required, not merely preferred: at or below two the
    !! covariance does not exist and the model has no variance to report.

    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, clone_kernel_into
    implicit none
    private

    public :: student_t_process_t

    !! Smallest admissible degrees of freedom. Below this the covariance is
    !! undefined; the bound is the model's, not a tuning choice.
    real(dp), parameter, public :: STUDENT_T_MIN_DOF = 2.0_dp

    type :: student_t_process_t
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: alpha(:)
        real(dp) :: nu = 5.0_dp
        real(dp) :: noise_variance = 1.0e-6_dp
        !! Diagonal regularization, matching the GP's default. Separate from
        !! the noise because it is a numerical safeguard rather than a claim
        !! about the measurement.
        real(dp) :: jitter = 1.0e-10_dp
        !! `(y - phi)' K^-1 (y - phi)`, the observed Mahalanobis distance. This
        !! is the quantity a GP has no use for and a TP conditions on.
        real(dp) :: beta = 0.0_dp
        integer :: n_samples = 0
        integer :: n_features = 0
        logical :: fitted = .false.
    contains
        procedure, public :: fit => student_t_fit
        procedure, public :: predict => student_t_predict
        procedure, public :: posterior_dof => student_t_posterior_dof
        procedure, public :: covariance_scale => student_t_covariance_scale
        procedure, public :: log_marginal_likelihood => student_t_lml
        procedure, public :: likelihood_parameter_count => student_t_likelihood_parameter_count
        procedure, public :: likelihood_parameters => student_t_likelihood_parameters
        procedure, public :: set_likelihood_parameters => student_t_set_likelihood_parameters
        procedure, public :: log_marginal_likelihood_likelihood_parameter_jvp => &
            student_t_likelihood_parameter_jvp
        procedure, public :: log_marginal_likelihood_likelihood_parameter_vjp => &
            student_t_likelihood_parameter_vjp
        procedure, public :: log_marginal_likelihood_likelihood_parameter_hvp => &
            student_t_likelihood_parameter_hvp
        procedure, public :: log_marginal_likelihood_likelihood_parameter_jvp_device => &
            student_t_likelihood_parameter_jvp_device
        procedure, public :: log_marginal_likelihood_likelihood_parameter_vjp_device => &
            student_t_likelihood_parameter_vjp_device
        procedure, public :: log_marginal_likelihood_likelihood_parameter_hvp_device => &
            student_t_likelihood_parameter_hvp_device
    end type student_t_process_t

contains

    subroutine student_t_fit(self, x, y, kernel, nu, noise_variance, status, jitter)
        class(student_t_process_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(in) :: y(:)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: nu
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: jitter
        real(dp), allocatable :: gram(:, :), work(:)
        integer :: n, i

        n = size(x, 1)
        if (n < 1 .or. size(y) /= n .or. size(x, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: input and target shapes disagree")
            return
        end if
        if (nu <= STUDENT_T_MIN_DOF) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: nu must exceed two for a covariance to exist")
            return
        end if
        if (noise_variance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: noise variance must not be negative")
            return
        end if

        self%n_samples = n
        self%n_features = size(x, 2)
        self%nu = nu
        self%noise_variance = noise_variance
        if (present(jitter)) then
            if (jitter < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "student-t process: jitter must not be negative")
                return
            end if
            self%jitter = jitter
        end if
        call clone_kernel_into(kernel, self%kernel)
        allocate (self%x_train, source=x)

        allocate (gram(n, n))
        call self%kernel%matrix(x, x, gram, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            gram(i, i) = gram(i, i) + noise_variance + self%jitter
        end do

        call self%factorization%factorize(gram, status)
        if (status%code /= FORTNUM_OK) return

        allocate (self%alpha(n), work(n))
        self%alpha = y
        call self%factorization%solve(self%alpha, status)
        if (status%code /= FORTNUM_OK) return

        ! The observed Mahalanobis distance. Its expectation under the prior is
        ! exactly `n`, which is what makes the ratio below interpretable: above
        ! one means the data were more surprising than the prior expected.
        self%beta = dot_product(y, self%alpha)
        if (.not. ieee_is_finite(self%beta)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: nonfinite Mahalanobis distance")
            return
        end if

        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_fit

    !! `nu + n`: every observation adds one degree of freedom, so a TP conditioned
    !! on enough data becomes indistinguishable from a GP.
    pure real(dp) function student_t_posterior_dof(self) result(dof)
        class(student_t_process_t), intent(in) :: self

        dof = self%nu + real(self%n_samples, dp)
    end function student_t_posterior_dof

    !! `(nu + beta - 2)/(nu + n - 2)`, the factor by which the observed data
    !! rescale the predictive covariance. Exactly one when the data were as
    !! surprising as the prior expected.
    pure real(dp) function student_t_covariance_scale(self) result(scale)
        class(student_t_process_t), intent(in) :: self

        scale = 1.0_dp
        if (.not. self%fitted) return
        scale = (self%nu + self%beta - 2.0_dp) &
            /(self%nu + real(self%n_samples, dp) - 2.0_dp)
    end function student_t_covariance_scale

    !! Predictive mean and marginal variance.
    !!
    !! The mean is the GP's — the inverse Wishart marginalization does not touch
    !! it. Only the covariance differs, and only by the scalar above.
    subroutine student_t_predict(self, x, mean, variance, status)
        class(student_t_process_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior_diagonal(:), work(:, :)
        real(dp) :: scale
        integer :: n, i

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: predict before fit")
            return
        end if
        n = size(x, 1)
        if (size(x, 2) /= self%n_features .or. size(mean) /= n .or. &
            size(variance) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: query or output shape is invalid")
            return
        end if

        allocate (cross(self%n_samples, n), prior_diagonal(n), work(self%n_samples, n))
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        ! Diagonal only. Building the full `n` by `n` prior covariance to
        ! take its diagonal costs `n**2` kernel evaluations and an `n**2`
        ! allocation for `n` numbers; at Bayesian-optimization candidate counts
        ! that is twenty-five million evaluations and 200 MB per call.
        do i = 1, n
            prior_diagonal(i) = self%kernel%value(x(i, :), x(i, :))
        end do

        mean = matmul(transpose(cross), self%alpha)
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return

        scale = self%covariance_scale()
        do i = 1, n
            variance(i) = scale*(prior_diagonal(i) - dot_product(cross(:, i), work(:, i)))
            if (variance(i) < 0.0_dp) variance(i) = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_predict

    pure integer function student_t_likelihood_parameter_count(self) result(count)
        !! One transformed likelihood coordinate: `log(nu - 2)`.
        class(student_t_process_t), intent(in) :: self

        count = 1
    end function student_t_likelihood_parameter_count

    function student_t_likelihood_parameters(self) result(parameters)
        !! The log offset keeps `nu > 2` by construction.
        class(student_t_process_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate (parameters(1))
        parameters(1) = log(self%nu - STUDENT_T_MIN_DOF)
    end function student_t_likelihood_parameters

    subroutine student_t_set_likelihood_parameters(self, parameters, status)
        !! Transactionally update the Student-t likelihood degrees of freedom.
        class(student_t_process_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: candidate_nu

        if (size(parameters) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: likelihood parameter count must be one")
            return
        end if
        candidate_nu = STUDENT_T_MIN_DOF + exp(parameters(1))
        if (.not. ieee_is_finite(candidate_nu) .or. &
            candidate_nu <= STUDENT_T_MIN_DOF) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: likelihood degrees of freedom are invalid")
            return
        end if
        self%nu = candidate_nu
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_set_likelihood_parameters

    !! Log marginal likelihood of the multivariate-t marginal.
    !!
    !! The `(nu - 2)` factors are the paper's parameterization showing through:
    !! its `K` is the covariance rather than the scale matrix, so the density
    !! carries `(nu - 2)` where a scale-matrix form would carry `nu`.
    subroutine student_t_lml(self, y, value, status)
        class(student_t_process_t), intent(in) :: self
        real(dp), intent(in) :: y(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient, hessian

        value = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: likelihood before fit")
            return
        end if
        if (size(y) /= self%n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: target length does not match the fit")
            return
        end if

        call student_t_likelihood_terms(self, value, gradient, hessian, status)
    end subroutine student_t_lml

    subroutine student_t_likelihood_parameter_jvp(self, y, direction, value, tangent, status)
        !! Fixed-state JVP for `log(nu - 2)`.
        class(student_t_process_t), intent(in) :: self
        real(dp), intent(in) :: y(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient, hessian

        value = 0.0_dp
        tangent = 0.0_dp
        if (size(direction) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: likelihood JVP direction count must be one")
            return
        end if
        if (size(y) /= self%n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: target length does not match the fit")
            return
        end if
        call student_t_likelihood_terms(self, value, gradient, hessian, status)
        if (status%code /= FORTNUM_OK) return
        tangent = gradient*direction(1)
    end subroutine student_t_likelihood_parameter_jvp

    subroutine student_t_likelihood_parameter_vjp(self, y, value_bar, parameter_bar, status)
        !! Fixed-state VJP for `log(nu - 2)`.
        class(student_t_process_t), intent(in) :: self
        real(dp), intent(in) :: y(:), value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value, gradient, hessian

        parameter_bar = 0.0_dp
        if (size(parameter_bar) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: likelihood VJP parameter count must be one")
            return
        end if
        if (size(y) /= self%n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: target length does not match the fit")
            return
        end if
        call student_t_likelihood_terms(self, value, gradient, hessian, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(1) = value_bar*gradient
    end subroutine student_t_likelihood_parameter_vjp

    subroutine student_t_likelihood_parameter_hvp(self, y, direction, parameter_hvp, status)
        !! Fixed-state HVP for `log(nu - 2)`.
        class(student_t_process_t), intent(in) :: self
        real(dp), intent(in) :: y(:), direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value, gradient, hessian

        parameter_hvp = 0.0_dp
        if (size(direction) /= 1 .or. size(parameter_hvp) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: likelihood HVP coordinate count must be one")
            return
        end if
        if (size(y) /= self%n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: target length does not match the fit")
            return
        end if
        call student_t_likelihood_terms(self, value, gradient, hessian, status)
        if (status%code /= FORTNUM_OK) return
        parameter_hvp(1) = hessian*direction(1)
    end subroutine student_t_likelihood_parameter_hvp

    subroutine student_t_likelihood_parameter_jvp_device(self, device, y, direction, &
            value, tangent, status)
        class(student_t_process_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: y(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        tangent = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process likelihood JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%log_marginal_likelihood_likelihood_parameter_jvp( &
                y, direction, value, tangent, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "student-t process likelihood JVP device: resident CUDA factorization is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process likelihood JVP device: device kind is invalid")
        end select
    end subroutine student_t_likelihood_parameter_jvp_device

    subroutine student_t_likelihood_parameter_vjp_device(self, device, y, value_bar, &
            parameter_bar, status)
        class(student_t_process_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: y(:), value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process likelihood VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%log_marginal_likelihood_likelihood_parameter_vjp( &
                y, value_bar, parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "student-t process likelihood VJP device: resident CUDA factorization is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process likelihood VJP device: device kind is invalid")
        end select
    end subroutine student_t_likelihood_parameter_vjp_device

    subroutine student_t_likelihood_parameter_hvp_device(self, device, y, direction, &
            parameter_hvp, status)
        class(student_t_process_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: y(:), direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_hvp = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process likelihood HVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%log_marginal_likelihood_likelihood_parameter_hvp( &
                y, direction, parameter_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "student-t process likelihood HVP device: resident CUDA factorization is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process likelihood HVP device: device kind is invalid")
        end select
    end subroutine student_t_likelihood_parameter_hvp_device

    subroutine student_t_likelihood_terms(self, value, gradient, hessian, status)
        !! Value and first two derivatives in `log(nu - 2)`, with the fitted
        !! covariance and Mahalanobis statistic held fixed.
        class(student_t_process_t), intent(in) :: self
        real(dp), intent(out) :: value, gradient, hessian
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_det, n_real, pi_value, offset, denominator, dof_gradient, &
            dof_hessian, log_ratio, numerator

        value = 0.0_dp
        gradient = 0.0_dp
        hessian = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: likelihood before fit")
            return
        end if
        offset = self%nu - STUDENT_T_MIN_DOF
        if (.not. ieee_is_finite(offset) .or. offset <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "student-t process: likelihood degrees of freedom are invalid")
            return
        end if
        call self%factorization%log_determinant(log_det, status)
        if (status%code /= FORTNUM_OK) return
        n_real = real(self%n_samples, dp)
        pi_value = 4.0_dp*atan(1.0_dp)
        denominator = offset*(offset + self%beta)
        log_ratio = log(1.0_dp + self%beta/offset)
        numerator = self%nu + n_real
        value = log_gamma(0.5_dp*numerator) - log_gamma(0.5_dp*self%nu) &
            - 0.5_dp*n_real*log(offset*pi_value) - 0.5_dp*log_det &
            - 0.5_dp*numerator*log_ratio
        dof_gradient = 0.5_dp*digamma_positive(0.5_dp*numerator) &
            - 0.5_dp*digamma_positive(0.5_dp*self%nu) - 0.5_dp*n_real/offset &
            - 0.5_dp*log_ratio + 0.5_dp*numerator*self%beta/denominator
        dof_hessian = 0.25_dp*trigamma_positive(0.5_dp*numerator) &
            - 0.25_dp*trigamma_positive(0.5_dp*self%nu) + 0.5_dp*n_real/(offset*offset) &
            + 0.5_dp*self%beta*(2.0_dp*denominator - numerator*(2.0_dp*offset + self%beta)) &
            /(denominator*denominator)
        gradient = offset*dof_gradient
        hessian = offset*dof_gradient + offset*offset*dof_hessian
        call status_set(status, FORTNUM_OK, "")
    end subroutine student_t_likelihood_terms

    pure real(dp) function digamma_positive(argument) result(value)
        real(dp), intent(in) :: argument
        real(dp) :: x, inverse, inverse_square

        x = argument
        value = 0.0_dp
        do while (x < 8.0_dp)
            value = value - 1.0_dp/x
            x = x + 1.0_dp
        end do
        inverse = 1.0_dp/x
        inverse_square = inverse*inverse
        value = value + log(x) - 0.5_dp*inverse - inverse_square*(1.0_dp/12.0_dp &
            - inverse_square*(1.0_dp/120.0_dp - inverse_square*(1.0_dp/252.0_dp &
            - inverse_square*(1.0_dp/240.0_dp - inverse_square*5.0_dp/660.0_dp))))
    end function digamma_positive

    pure real(dp) function trigamma_positive(argument) result(value)
        real(dp), intent(in) :: argument
        real(dp) :: x, inverse, inverse_square

        x = argument
        value = 0.0_dp
        do while (x < 8.0_dp)
            value = value + 1.0_dp/(x*x)
            x = x + 1.0_dp
        end do
        inverse = 1.0_dp/x
        inverse_square = inverse*inverse
        value = value + inverse + 0.5_dp*inverse_square + inverse_square*inverse/6.0_dp &
            - inverse_square**2*inverse/30.0_dp + inverse_square**3*inverse/42.0_dp &
            - inverse_square**4*inverse/30.0_dp + 5.0_dp*inverse_square**5*inverse/66.0_dp
    end function trigamma_positive

end module fortml_student_t_process
