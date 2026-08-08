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
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t, clone_kernel_into
    implicit none
    private

    public :: robust_gp_t

    integer, parameter, public :: FORTML_LIKELIHOOD_POISSON = 1
    integer, parameter, public :: FORTML_LIKELIHOOD_STUDENT_T = 2

    integer, parameter, public :: FORTML_ROBUST_MAX_ITERATIONS = 200
    real(dp), parameter, public :: FORTML_ROBUST_TOLERANCE = 1.0e-10_dp

    type :: robust_gp_t
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: x_train(:, :)
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

        allocate (gram(n, n))
        call self%kernel%matrix(x, x, gram, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            gram(i, i) = gram(i, i) + self%jitter
        end do

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

end module fortml_robust_gp
