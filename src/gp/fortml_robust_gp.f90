module fortml_robust_gp
    !! Laplace-approximated GP regression under non-Gaussian observation models.
    !!
    !! Two likelihoods, both of which an ordinary GP handles badly for the same
    !! underlying reason — it assumes the residual is Gaussian with a variance
    !! that does not depend on the latent value:
    !!
    !!   * **Poisson**, for counts. `y | f ~ Poisson(exp(f))`. A Gaussian
    !!     likelihood on count data puts mass on negative counts, and treats a
    !!     spread of 3 around a mean of 4 the same as a spread of 3 around a
    !!     mean of 400 — where the first is enormous and the second is nothing.
    !!     Modelling the log rate keeps the mean positive by construction and
    !!     ties the variance to it, which is what a count actually does.
    !!   * **Student-t**, for robustness. `y | f ~ t_nu(f, scale)`. A Gaussian
    !!     log density is quadratic, so one outlier pulls the fit with unbounded
    !!     force. A Student-t's is logarithmic in the residual, so its influence
    !!     *saturates*: a point ten scales away pulls barely harder than one
    !!     five scales away, and the bulk of the data decides the fit.
    !!
    !! Both are fitted by the same damped Newton iteration on the latent mode
    !! followed by a Laplace approximation, the construction already used for
    !! classification here. Only the first two derivatives of the log density
    !! differ, which is why those are stated separately and everything else is
    !! shared.
    !!
    !! **The Student-t posterior is not log-concave.** Beyond `sqrt(nu) * scale`
    !! its curvature turns negative, which would make a Newton step ascend. The
    !! curvature is floored at zero, so a far-tail point contributes no
    !! confidence rather than negative confidence — which *is* the influence
    !! saturation the likelihood was chosen for. `converged` reports whether the
    !! iteration settled; a caller ignoring it is reading a mode that may not be
    !! one.
    !!
    !! The Newton system is solved in the `B = I + W^(1/2) K W^(1/2)` form
    !! rather than the direct `I + W K`. The former is symmetric positive
    !! definite and the latter is neither, so the direct version would need
    !! normal equations and would square the condition number — on a Student-t
    !! fit with floored curvature, exactly where that can least be afforded.

    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t, clone_kernel_into
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    implicit none
    private

    public :: robust_gp_t
    public :: robust_poisson_log_likelihood_value
    public :: robust_poisson_log_likelihood_gradient
    public :: robust_poisson_log_likelihood_jvp
    public :: robust_poisson_log_likelihood_vjp
    public :: robust_poisson_log_likelihood_hvp

    integer, parameter, public :: FORTML_LIKELIHOOD_POISSON = 1
    integer, parameter, public :: FORTML_LIKELIHOOD_STUDENT_T = 2

    integer, parameter, public :: FORTML_ROBUST_MAX_ITERATIONS = 200
    real(dp), parameter, public :: FORTML_ROBUST_TOLERANCE = 1.0e-10_dp

    type :: robust_gp_t
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: prior_factorization
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: covariance(:, :)
        real(dp), allocatable :: y_train(:)
        real(dp), allocatable :: mode(:)
        real(dp), allocatable :: curvature(:)
        !! `K^-1 f` at the mode, which stationarity makes equal to the
        !! log-likelihood gradient there. Stored because the predictive mean is
        !! `K_* alpha` and recovering it from the mode would need `K^-1`.
        real(dp), allocatable :: alpha(:)
        integer :: likelihood = FORTML_LIKELIHOOD_POISSON
        real(dp) :: nu = 4.0_dp
        real(dp) :: scale = 1.0_dp
        real(dp) :: jitter = 1.0e-8_dp
        integer :: n_samples = 0
        integer :: n_features = 0
        integer :: iterations = 0
        logical :: converged = .false.
        logical :: fitted = .false.
    contains
        procedure, public :: fit => robust_fit
        procedure, public :: predict_latent => robust_predict_latent
        procedure, public :: predict_response => robust_predict_response
        procedure, public :: predict_latent_jvp => robust_predict_latent_jvp
        procedure, public :: predict_latent_vjp => robust_predict_latent_vjp
        procedure, public :: predict_response_jvp => robust_predict_response_jvp
        procedure, public :: predict_response_vjp => robust_predict_response_vjp
        procedure, public :: predict_response_device => robust_predict_response_device
        procedure, public :: latent_parameter_count => robust_latent_parameter_count
        procedure, public :: latent_parameters => robust_latent_parameters
        procedure, public :: set_latent_parameters => robust_set_latent_parameters
        procedure, public :: log_posterior => robust_log_posterior
        procedure, public :: log_posterior_gradient => robust_log_posterior_gradient
        procedure, public :: log_posterior_jvp => robust_log_posterior_jvp
        procedure, public :: log_posterior_vjp => robust_log_posterior_vjp
        procedure, public :: log_posterior_hvp => robust_log_posterior_hvp
        procedure, public :: log_posterior_device => robust_log_posterior_device
    end type robust_gp_t

contains

    !! First derivative and negative second derivative of the log likelihood in
    !! the latent value. Everything likelihood-specific lives here.
    pure subroutine log_likelihood_derivatives(likelihood, nu, scale, y, f, &
            gradient, curvature)
        integer, intent(in) :: likelihood
        real(dp), intent(in) :: nu, scale, y, f
        real(dp), intent(out) :: gradient
        real(dp), intent(out) :: curvature
        real(dp) :: rate, residual, denominator

        select case (likelihood)
        case (FORTML_LIKELIHOOD_POISSON)
            ! log p = y f - exp(f) + constant.
            rate = exp(f)
            gradient = y - rate
            ! Always positive: Poisson is log-concave, so no safeguard is needed.
            curvature = rate
        case default
            ! log p = -((nu + 1)/2) log(1 + r^2 / (nu s^2)), with r = y - f.
            residual = y - f
            denominator = nu*scale*scale + residual*residual
            gradient = (nu + 1.0_dp)*residual/denominator
            curvature = (nu + 1.0_dp)*(nu*scale*scale - residual*residual) &
                /(denominator*denominator)
            if (curvature < 0.0_dp) curvature = 0.0_dp
        end select
    end subroutine log_likelihood_derivatives

    subroutine robust_fit(self, x, y, kernel, likelihood, status, nu, scale, &
            jitter)
        class(robust_gp_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(in) :: y(:)
        type(kernel_t), intent(in) :: kernel
        integer, intent(in) :: likelihood
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: nu, scale, jitter
        real(dp), allocatable :: gram(:, :), system(:, :)
        real(dp), allocatable :: gradient(:), curvature(:), root_w(:)
        real(dp), allocatable :: b_vector(:), a_vector(:), temporary(:)
        real(dp), allocatable :: previous(:)
        real(dp) :: shift, damping
        integer :: n, i, j, iteration

        n = size(x, 1)
        if (n < 1 .or. size(y) /= n .or. size(x, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP: input and target shapes disagree")
            return
        end if
        if (likelihood /= FORTML_LIKELIHOOD_POISSON .and. &
            likelihood /= FORTML_LIKELIHOOD_STUDENT_T) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP: unknown likelihood")
            return
        end if
        if (likelihood == FORTML_LIKELIHOOD_POISSON .and. any(y < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP: Poisson counts must not be negative")
            return
        end if
        if (present(nu)) then
            if (nu <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "robust GP: nu must be positive")
                return
            end if
            self%nu = nu
        end if
        if (present(scale)) then
            if (scale <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "robust GP: scale must be positive")
                return
            end if
            self%scale = scale
        end if
        if (present(jitter)) then
            if (jitter < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "robust GP: jitter must not be negative")
                return
            end if
            self%jitter = jitter
        end if

        self%n_samples = n
        self%n_features = size(x, 2)
        self%likelihood = likelihood
        call clone_kernel_into(kernel, self%kernel)
        allocate (self%x_train, source=x)
        allocate (self%y_train, source=y)

        allocate (gram(n, n))
        call self%kernel%matrix(x, x, gram, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            gram(i, i) = gram(i, i) + self%jitter
        end do
        allocate (self%covariance, source=gram)
        call self%prior_factorization%factorize(self%covariance, status)
        if (status%code /= FORTNUM_OK) return

        allocate (self%mode(n), self%curvature(n), self%alpha(n))
        allocate (gradient(n), curvature(n), root_w(n), system(n, n))
        allocate (b_vector(n), a_vector(n), temporary(n), previous(n))

        ! Starting from the data is far better than from zero for a Student-t
        ! fit, whose posterior is multimodal when outliers disagree.
        self%mode = 0.0_dp
        if (likelihood == FORTML_LIKELIHOOD_STUDENT_T) self%mode = y
        self%alpha = 0.0_dp
        self%converged = .false.

        damping = 1.0_dp
        if (likelihood == FORTML_LIKELIHOOD_STUDENT_T) damping = 0.5_dp

        do iteration = 1, FORTML_ROBUST_MAX_ITERATIONS
            previous = self%mode
            do i = 1, n
                call log_likelihood_derivatives(likelihood, self%nu, self%scale, &
                    y(i), self%mode(i), gradient(i), &
                    curvature(i))
                root_w(i) = sqrt(curvature(i))
            end do

            do j = 1, n
                do i = 1, n
                    system(i, j) = root_w(i)*gram(i, j)*root_w(j)
                end do
                system(j, j) = system(j, j) + 1.0_dp
            end do
            call self%factorization%factorize(system, status)
            if (status%code /= FORTNUM_OK) return

            b_vector = curvature*self%mode + gradient
            temporary = root_w*matmul(gram, b_vector)
            call self%factorization%solve(temporary, status)
            if (status%code /= FORTNUM_OK) return
            a_vector = b_vector - root_w*temporary

            self%mode = (1.0_dp - damping)*previous + damping*matmul(gram, a_vector)
            self%alpha = a_vector
            if (any(.not. ieee_is_finite(self%mode))) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "robust GP: latent mode diverged")
                return
            end if

            shift = maxval(abs(self%mode - previous))
            self%iterations = iteration
            if (shift < FORTML_ROBUST_TOLERANCE) then
                self%converged = .true.
                exit
            end if
        end do

        ! Curvature and factorization at the mode actually reached, so the
        ! predictive variance belongs to the same latent as the mean.
        do i = 1, n
            call log_likelihood_derivatives(likelihood, self%nu, self%scale, &
                y(i), self%mode(i), gradient(i), &
                self%curvature(i))
            root_w(i) = sqrt(self%curvature(i))
        end do
        do j = 1, n
            do i = 1, n
                system(i, j) = root_w(i)*gram(i, j)*root_w(j)
            end do
            system(j, j) = system(j, j) + 1.0_dp
        end do
        call self%factorization%factorize(system, status)
        if (status%code /= FORTNUM_OK) return

        self%fitted = .true.
        ! A non-converged fit reports through `converged` rather than through
        ! the status: the mode is usable and the caller decides, which beats
        ! discarding a nearly-converged fit outright.
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_fit

    subroutine robust_predict_latent(self, x, mean, variance, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior_diagonal(:), weighted(:, :)
        real(dp), allocatable :: root_w(:)
        integer :: n, i, j

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP: predict before fit")
            return
        end if
        n = size(x, 1)
        if (size(x, 2) /= self%n_features .or. size(mean) /= n .or. &
            size(variance) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP: query or output shape is invalid")
            return
        end if

        allocate (cross(self%n_samples, n), prior_diagonal(n))
        allocate (weighted(self%n_samples, n), root_w(self%n_samples))
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

        root_w = sqrt(self%curvature)
        do j = 1, n
            do i = 1, self%n_samples
                weighted(i, j) = root_w(i)*cross(i, j)
            end do
        end do
        call self%factorization%solve(weighted, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, n
            do i = 1, self%n_samples
                weighted(i, j) = root_w(i)*weighted(i, j)
            end do
        end do

        do i = 1, n
            variance(i) = prior_diagonal(i) - dot_product(cross(:, i), weighted(:, i))
            if (variance(i) < 0.0_dp) variance(i) = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_predict_latent

    !! Expected response on the observation scale.
    !!
    !! For Poisson this is `exp(f + v/2)`, the log-normal mean — *not* `exp(f)`,
    !! which is the median. Reporting the median as the mean understates every
    !! rate, and the gap grows with the posterior variance, so it is worst
    !! exactly where the model is least sure.
    subroutine robust_predict_response(self, x, response, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: response(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        integer :: n

        response = 0.0_dp
        n = size(x, 1)
        if (size(response) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP: response length does not match the query")
            return
        end if
        allocate (mean(n), variance(n))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return

        select case (self%likelihood)
        case (FORTML_LIKELIHOOD_POISSON)
            response = exp(mean + 0.5_dp*variance)
        case default
            ! A Student-t observation is centred on the latent, so its mean is
            ! the latent mean wherever it exists at all.
            response = mean
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_predict_response

    !> Return the Poisson log likelihood in log-rate coordinates.
    !!
    !! The normalising ``log_gamma(y+1)`` term is retained, so this primitive
    !! is a true log probability rather than an optimisation-only score.  It
    !! is shared by the Laplace GP and external FortOpt objectives.
    subroutine robust_poisson_log_likelihood_value(counts, log_rate, value, status)
        real(dp), intent(in) :: counts(:), log_rate(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        value = 0.0_dp
        if (.not. valid_poisson_vectors(counts, log_rate, status)) return
        do i = 1, size(counts)
            value = value + counts(i)*log_rate(i) - exp(log_rate(i)) - &
                log_gamma(counts(i) + 1.0_dp)
        end do
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Poisson likelihood: value is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_log_likelihood_value

    !> Gradient with respect to the log-rate vector.
    subroutine robust_poisson_log_likelihood_gradient(counts, log_rate, gradient, status)
        real(dp), intent(in) :: counts(:), log_rate(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        gradient = 0.0_dp
        if (size(gradient) /= size(counts)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood gradient: output shape is invalid")
            return
        end if
        if (.not. valid_poisson_vectors(counts, log_rate, status)) return
        do i = 1, size(counts)
            gradient(i) = counts(i) - exp(log_rate(i))
        end do
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Poisson likelihood gradient: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_log_likelihood_gradient

    !> Value and forward product through a log-rate vector.
    subroutine robust_poisson_log_likelihood_jvp(counts, log_rate, log_rate_dot, &
            value, value_dot, status)
        real(dp), intent(in) :: counts(:), log_rate(:), log_rate_dot(:)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: rate
        integer :: i

        value = 0.0_dp
        value_dot = 0.0_dp
        if (size(log_rate_dot) /= size(log_rate)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood JVP: tangent shape is invalid")
            return
        end if
        if (.not. valid_poisson_vectors(counts, log_rate, status)) return
        if (any(.not. ieee_is_finite(log_rate_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood JVP: tangent is not finite")
            return
        end if
        do i = 1, size(counts)
            rate = exp(log_rate(i))
            value = value + counts(i)*log_rate(i) - rate - &
                log_gamma(counts(i) + 1.0_dp)
            value_dot = value_dot + (counts(i) - rate)*log_rate_dot(i)
        end do
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(value_dot)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Poisson likelihood JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_log_likelihood_jvp

    !> Reverse product through a scalar cotangent.
    subroutine robust_poisson_log_likelihood_vjp(counts, log_rate, value_bar, &
            log_rate_bar, status)
        real(dp), intent(in) :: counts(:), log_rate(:), value_bar
        real(dp), intent(out) :: log_rate_bar(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        log_rate_bar = 0.0_dp
        if (size(log_rate_bar) /= size(log_rate) .or. .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood VJP: cotangent shape or value is invalid")
            return
        end if
        if (.not. valid_poisson_vectors(counts, log_rate, status)) return
        do i = 1, size(counts)
            log_rate_bar(i) = value_bar*(counts(i) - exp(log_rate(i)))
        end do
        if (any(.not. ieee_is_finite(log_rate_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Poisson likelihood VJP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_log_likelihood_vjp

    !> Hessian-vector product through the log-rate vector.
    subroutine robust_poisson_log_likelihood_hvp(counts, log_rate, log_rate_dot, &
            product, status)
        real(dp), intent(in) :: counts(:), log_rate(:), log_rate_dot(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        product = 0.0_dp
        if (size(product) /= size(log_rate) .or. size(log_rate_dot) /= size(log_rate)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood HVP: direction or output shape is invalid")
            return
        end if
        if (.not. valid_poisson_vectors(counts, log_rate, status)) return
        if (any(.not. ieee_is_finite(log_rate_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood HVP: direction is not finite")
            return
        end if
        do i = 1, size(log_rate)
            product(i) = -exp(log_rate(i))*log_rate_dot(i)
        end do
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Poisson likelihood HVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_poisson_log_likelihood_hvp

    integer function robust_latent_parameter_count(self) result(count)
        class(robust_gp_t), intent(in) :: self

        count = 0
        if (self%fitted) count = self%n_samples
    end function robust_latent_parameter_count

    function robust_latent_parameters(self) result(parameters)
        class(robust_gp_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(self%latent_parameter_count()))
        if (self%fitted) parameters = self%mode
    end function robust_latent_parameters

    !> Replace only the latent mode and rebuild its Laplace curvature.
    !!
    !! The candidate factorization is built before any state is committed.  A
    !! malformed vector or a failed Cholesky therefore leaves the complete
    !! fitted state untouched, which is important when FortOpt rejects a line
    !! search point.
    subroutine robust_set_latent_parameters(self, parameters, status)
        class(robust_gp_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate_curvature(:), candidate_alpha(:)
        type(cholesky_factorization_t) :: candidate_factor

        if (.not. self%fitted .or. size(parameters) /= self%n_samples .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP latent state: parameter shape or values are invalid")
            return
        end if
        allocate(candidate_curvature(self%n_samples), candidate_alpha(self%n_samples))
        call build_laplace_state(self, parameters, candidate_curvature, candidate_alpha, &
            candidate_factor, status)
        if (status%code /= FORTNUM_OK) return
        self%mode = parameters
        self%curvature = candidate_curvature
        self%alpha = candidate_alpha
        self%factorization = candidate_factor
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_set_latent_parameters

    subroutine robust_log_posterior(self, value, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: alpha(:)

        value = -huge(1.0_dp)
        if (.not. robust_state_valid(self, status)) return
        allocate(alpha(self%n_samples))
        alpha = self%mode
        call self%prior_factorization%solve(alpha, status)
        if (status%code /= FORTNUM_OK) return
        call log_posterior_from_state(self, self%mode, alpha, value, status)
    end subroutine robust_log_posterior

    subroutine robust_log_posterior_gradient(self, gradient, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: alpha(:)

        gradient = 0.0_dp
        if (.not. robust_state_valid(self, status)) return
        if (size(gradient) /= self%n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP posterior gradient: output shape is invalid")
            return
        end if
        allocate(alpha(self%n_samples))
        alpha = self%mode
        call self%prior_factorization%solve(alpha, status)
        if (status%code /= FORTNUM_OK) return
        call posterior_gradient_from_state(self, self%mode, alpha, gradient, status)
    end subroutine robust_log_posterior_gradient

    subroutine robust_log_posterior_jvp(self, direction, value, tangent, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = -huge(1.0_dp)
        tangent = 0.0_dp
        if (.not. robust_state_valid(self, status)) return
        if (size(direction) /= self%n_samples .or. any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP posterior JVP: direction is invalid")
            return
        end if
        allocate(gradient(self%n_samples))
        call self%log_posterior(value, status)
        if (status%code /= FORTNUM_OK) return
        call self%log_posterior_gradient(gradient, status)
        if (status%code /= FORTNUM_OK) return
        tangent = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_log_posterior_jvp

    subroutine robust_log_posterior_vjp(self, value_bar, gradient, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: value_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        if (.not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP posterior VJP: cotangent is invalid")
            return
        end if
        call self%log_posterior_gradient(gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = value_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_log_posterior_vjp

    subroutine robust_log_posterior_hvp(self, direction, product, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: prior_direction(:)
        real(dp) :: gradient, curvature
        integer :: i

        product = 0.0_dp
        if (.not. robust_state_valid(self, status)) return
        if (size(direction) /= self%n_samples .or. size(product) /= self%n_samples .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP posterior HVP: direction or output shape is invalid")
            return
        end if
        allocate(prior_direction(self%n_samples))
        prior_direction = direction
        call self%prior_factorization%solve(prior_direction, status)
        if (status%code /= FORTNUM_OK) return
        product = -prior_direction
        do i = 1, self%n_samples
            call log_likelihood_derivatives(self%likelihood, self%nu, self%scale, &
                self%y_train(i), self%mode(i), gradient, curvature)
            product(i) = product(i) - curvature*direction(i)
        end do
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "robust GP posterior HVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_log_posterior_hvp

    subroutine robust_log_posterior_device(self, device, value, status)
        class(robust_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = -huge(1.0_dp)
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP posterior device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%log_posterior(value, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "robust GP posterior device: resident CUDA Laplace kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP posterior device: device kind is invalid")
        end select
    end subroutine robust_log_posterior_device

    subroutine robust_predict_latent_jvp(self, x, x_dot, mean, mean_dot, variance, &
            variance_dot, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), weighted(:, :), weighted_dot(:, :)
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        real(dp) :: kernel_value, prior_dot
        integer :: i, j, n_query

        mean = 0.0_dp
        mean_dot = 0.0_dp
        variance = 0.0_dp
        variance_dot = 0.0_dp
        if (.not. prediction_jvp_valid(self, x, x_dot, mean, mean_dot, variance, &
            variance_dot, status)) return
        n_query = size(x, 1)
        allocate(cross(self%n_samples, n_query), cross_dot(self%n_samples, n_query), &
            weighted(self%n_samples, n_query), weighted_dot(self%n_samples, n_query))
        allocate(gradient_x1(self%n_features), gradient_x2(self%n_features), &
            mixed_hessian(self%n_features, self%n_features))
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        call scale_rows(cross, sqrt(self%curvature), weighted)
        call self%factorization%solve(weighted, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%n_samples
            weighted(i, :) = sqrt(self%curvature(i))*weighted(i, :)
        end do
        do j = 1, n_query
            do i = 1, self%n_samples
                call self%kernel%input_derivatives(self%x_train(i, :), x(j, :), &
                    kernel_value, gradient_x1, gradient_x2, mixed_hessian, status)
                if (status%code /= FORTNUM_OK) return
                cross_dot(i, j) = dot_product(gradient_x2, x_dot(j, :))
            end do
            call self%kernel%input_derivatives(x(j, :), x(j, :), kernel_value, &
                gradient_x1, gradient_x2, mixed_hessian, status)
            if (status%code /= FORTNUM_OK) return
            prior_dot = dot_product(gradient_x1 + gradient_x2, x_dot(j, :))
            mean_dot(j) = dot_product(cross_dot(:, j), self%alpha)
            variance_dot(j) = prior_dot
        end do
        call scale_rows(cross_dot, sqrt(self%curvature), weighted_dot)
        call self%factorization%solve(weighted_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%n_samples
            weighted_dot(i, :) = sqrt(self%curvature(i))*weighted_dot(i, :)
        end do
        mean = matmul(transpose(cross), self%alpha)
        do j = 1, n_query
            variance(j) = self%kernel%value(x(j, :), x(j, :)) - &
                dot_product(cross(:, j), weighted(:, j))
            variance_dot(j) = variance_dot(j) - dot_product(cross_dot(:, j), weighted(:, j)) - &
                dot_product(cross(:, j), weighted_dot(:, j))
        end do
        call clamp_prediction_variance(variance, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(mean_dot)) .or. any(.not. ieee_is_finite(variance_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "robust GP latent JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_predict_latent_jvp

    subroutine robust_predict_latent_vjp(self, x, mean_bar, variance_bar, x_bar, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), weighted(:, :), cross_bar(:, :)
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        real(dp) :: kernel_value, prior_bar
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. prediction_vjp_valid(self, x, mean_bar, variance_bar, x_bar, status)) return
        allocate(cross(self%n_samples, size(x, 1)), weighted(self%n_samples, size(x, 1)), &
            cross_bar(self%n_samples, size(x, 1)))
        allocate(gradient_x1(self%n_features), gradient_x2(self%n_features), &
            mixed_hessian(self%n_features, self%n_features))
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        call scale_rows(cross, sqrt(self%curvature), weighted)
        call self%factorization%solve(weighted, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%n_samples
            weighted(i, :) = sqrt(self%curvature(i))*weighted(i, :)
        end do
        do j = 1, size(x, 1)
            cross_bar(:, j) = mean_bar(j)*self%alpha - 2.0_dp*variance_bar(j)*weighted(:, j)
            prior_bar = variance_bar(j)
            do i = 1, self%n_samples
                call self%kernel%input_derivatives(self%x_train(i, :), x(j, :), &
                    kernel_value, gradient_x1, gradient_x2, mixed_hessian, status)
                if (status%code /= FORTNUM_OK) return
                x_bar(j, :) = x_bar(j, :) + cross_bar(i, j)*gradient_x2
            end do
            call self%kernel%input_derivatives(x(j, :), x(j, :), kernel_value, &
                gradient_x1, gradient_x2, mixed_hessian, status)
            if (status%code /= FORTNUM_OK) return
            x_bar(j, :) = x_bar(j, :) + prior_bar*(gradient_x1 + gradient_x2)
        end do
        if (any(.not. ieee_is_finite(x_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "robust GP latent VJP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_predict_latent_vjp

    subroutine robust_predict_response_jvp(self, x, x_dot, response, response_dot, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: response(:), response_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        integer :: n, i

        n = size(x, 1)
        response = 0.0_dp
        response_dot = 0.0_dp
        if (size(response) /= n .or. size(response_dot) /= n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP response JVP: output shape is invalid")
            return
        end if
        allocate(mean(n), mean_dot(n), variance(n), variance_dot(n))
        call self%predict_latent_jvp(x, x_dot, mean, mean_dot, variance, variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            if (self%likelihood == FORTML_LIKELIHOOD_POISSON) then
                response(i) = exp(mean(i) + 0.5_dp*variance(i))
                response_dot(i) = response(i)*(mean_dot(i) + 0.5_dp*variance_dot(i))
            else
                response(i) = mean(i)
                response_dot(i) = mean_dot(i)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_predict_response_jvp

    subroutine robust_predict_response_vjp(self, x, response_bar, x_bar, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), response_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)
        integer :: n, i

        x_bar = 0.0_dp
        n = size(x, 1)
        if (size(response_bar) /= n .or. size(x_bar, 1) /= n .or. &
            size(x_bar, 2) /= self%n_features .or. any(.not. ieee_is_finite(response_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP response VJP: input or output shape is invalid")
            return
        end if
        allocate(mean(n), variance(n), mean_bar(n), variance_bar(n))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        if (self%likelihood == FORTML_LIKELIHOOD_POISSON) then
            do i = 1, n
                mean_bar(i) = response_bar(i)*exp(mean(i) + 0.5_dp*variance(i))
                variance_bar(i) = 0.5_dp*mean_bar(i)
            end do
        else
            mean_bar = response_bar
            variance_bar = 0.0_dp
        end if
        call self%predict_latent_vjp(x, mean_bar, variance_bar, x_bar, status)
    end subroutine robust_predict_response_vjp

    subroutine robust_predict_response_device(self, device, x, response, status)
        class(robust_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: response(:)
        type(fortnum_status_t), intent(out) :: status

        response = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP response device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_response(x, response, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "robust GP response device: resident CUDA Laplace kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP response device: device kind is invalid")
        end select
    end subroutine robust_predict_response_device

    logical function robust_state_valid(self, status) result(valid)
        class(robust_gp_t), intent(in) :: self
        type(fortnum_status_t), intent(out) :: status

        valid = self%fitted .and. allocated(self%covariance) .and. &
            allocated(self%y_train) .and. allocated(self%mode)
        if (.not. valid) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP posterior: model is not fitted")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function robust_state_valid

    logical function prediction_jvp_valid(self, x, x_dot, mean, mean_dot, variance, &
            variance_dot, status) result(valid)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :), mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status

        valid = robust_state_valid(self, status)
        if (.not. valid) return
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(shape(x_dot) /= shape(x)) .or. size(mean) /= size(x, 1) .or. &
            size(mean_dot) /= size(mean) .or. size(variance) /= size(mean) .or. &
            size(variance_dot) /= size(mean) .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP latent JVP: input or output shape is invalid")
            valid = .false.
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function prediction_jvp_valid

    logical function prediction_vjp_valid(self, x, mean_bar, variance_bar, x_bar, status) result(valid)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = robust_state_valid(self, status)
        if (.not. valid) return
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            size(mean_bar) /= size(x, 1) .or. size(variance_bar) /= size(x, 1) .or. &
            any(shape(x_bar) /= shape(x)) .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(mean_bar)) .or. any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP latent VJP: input or output shape is invalid")
            valid = .false.
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function prediction_vjp_valid

    subroutine build_laplace_state(self, mode, curvature, alpha, factor, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: mode(:)
        real(dp), intent(out) :: curvature(:), alpha(:)
        type(cholesky_factorization_t), intent(out) :: factor
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: system(:, :), root_w(:)
        real(dp) :: gradient
        integer :: i, j

        if (size(mode) /= self%n_samples .or. size(curvature) /= self%n_samples .or. &
            size(alpha) /= self%n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust GP latent state: internal shape is invalid")
            return
        end if
        allocate(system(self%n_samples, self%n_samples), root_w(self%n_samples))
        do i = 1, self%n_samples
            call log_likelihood_derivatives(self%likelihood, self%nu, self%scale, &
                self%y_train(i), mode(i), gradient, curvature(i))
            if (.not. ieee_is_finite(curvature(i))) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "robust GP latent state: curvature is not finite")
                return
            end if
            root_w(i) = sqrt(max(curvature(i), 0.0_dp))
        end do
        system = 0.0_dp
        do i = 1, self%n_samples
            system(i, i) = 1.0_dp
            do j = 1, self%n_samples
                system(i, j) = system(i, j) + root_w(i)*self%covariance(i, j)*root_w(j)
            end do
        end do
        call factor%factorize(system, status)
        if (status%code /= FORTNUM_OK) return
        alpha = mode
        call self%prior_factorization%solve(alpha, status)
    end subroutine build_laplace_state

    subroutine log_posterior_from_state(self, mode, alpha, value, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: mode(:), alpha(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient, curvature
        integer :: i

        value = -0.5_dp*dot_product(mode, alpha)
        do i = 1, size(mode)
            call log_likelihood_log_value(self%likelihood, self%nu, self%scale, &
                self%y_train(i), mode(i), gradient, curvature)
            value = value + gradient
        end do
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "robust GP posterior: value is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine log_posterior_from_state

    subroutine posterior_gradient_from_state(self, mode, alpha, gradient_out, status)
        class(robust_gp_t), intent(in) :: self
        real(dp), intent(in) :: mode(:), alpha(:)
        real(dp), intent(out) :: gradient_out(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gradient, curvature
        integer :: i

        gradient_out = -alpha
        do i = 1, size(mode)
            call log_likelihood_derivatives(self%likelihood, self%nu, self%scale, &
                self%y_train(i), mode(i), gradient, curvature)
            gradient_out(i) = gradient_out(i) + gradient
        end do
        if (any(.not. ieee_is_finite(gradient_out))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "robust GP posterior: gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine posterior_gradient_from_state

    subroutine log_likelihood_log_value(likelihood, nu, scale, y, f, value, curvature)
        integer, intent(in) :: likelihood
        real(dp), intent(in) :: nu, scale, y, f
        real(dp), intent(out) :: value, curvature
        real(dp) :: residual, denominator, ignored_gradient

        if (likelihood == FORTML_LIKELIHOOD_POISSON) then
            value = y*f - exp(f) - log_gamma(y + 1.0_dp)
            curvature = exp(f)
        else
            residual = y - f
            denominator = nu*scale*scale + residual*residual
            value = -0.5_dp*(nu + 1.0_dp)*log(1.0_dp + residual*residual/(nu*scale*scale))
            call log_likelihood_derivatives(likelihood, nu, scale, y, f, ignored_gradient, curvature)
        end if
    end subroutine log_likelihood_log_value

    logical function valid_poisson_vectors(counts, log_rate, status) result(valid)
        real(dp), intent(in) :: counts(:), log_rate(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (size(counts) < 1 .or. size(log_rate) /= size(counts) .or. &
            any(.not. ieee_is_finite(counts)) .or. any(counts < 0.0_dp) .or. &
            any(.not. ieee_is_finite(log_rate)) .or. &
            any(log_rate > log(huge(1.0_dp)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson likelihood: counts or log-rates are invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_poisson_vectors

    subroutine clamp_prediction_variance(variance, status)
        real(dp), intent(inout) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        do i = 1, size(variance)
            if (.not. ieee_is_finite(variance(i))) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "robust GP latent prediction: variance is not finite")
                return
            end if
            if (variance(i) < 0.0_dp) variance(i) = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine clamp_prediction_variance

    subroutine scale_rows(matrix, weights, scaled)
        real(dp), intent(in) :: matrix(:, :), weights(:)
        real(dp), intent(out) :: scaled(:, :)
        integer :: i

        do i = 1, size(matrix, 1)
            scaled(i, :) = weights(i)*matrix(i, :)
        end do
    end subroutine scale_rows

end module fortml_robust_gp
