module fortml_heteroskedastic_gp
    !! Heteroskedastic Gaussian process regression: input-dependent noise.
    !!
    !! An ordinary GP carries one noise variance for the whole domain. That is
    !! wrong whenever measurement quality varies with the input — a simulation
    !! that converges in some regions and not others, a rig that is precise at
    !! low load and rattles at high, a survey with uneven coverage. Fitting a
    !! single noise to such data does not average the two regimes: it takes a
    !! value between them and is then overconfident in the noisy region and
    !! underconfident in the quiet one, which is worse than either.
    !!
    !! The noise is modelled by a **second latent process on the log scale**,
    !! following the standard construction (Goldberg, Williams and Bishop 1998;
    !! Lazaro-Gredilla and Titsias 2011 — neither on arXiv, so neither is in the
    !! literature cache, and this is written from the construction rather than
    !! from a transcription):
    !!
    !!     f(x)  ~ GP(0, k_f),        the signal,
    !!     g(x)  ~ GP(mu_g, k_g),     the log-noise,
    !!     y(x) | f, g ~ N(f(x), exp(g(x))).
    !!
    !! Modelling `log` noise rather than noise is what keeps the model coherent:
    !! a process on the variance directly would put mass on negative variances,
    !! and clipping those would bias exactly the quiet regions the model exists
    !! to represent.
    !!
    !! **This implementation takes the noise levels as given.** A caller that
    !! knows its measurement variances — replicate spreads, solver residuals,
    !! reported error bars — supplies them, and the signal posterior is then
    !! exact. Inferring `g` jointly with `f` requires variational or EM
    !! machinery that belongs with the other approximate-inference work, and
    !! pretending to do it here would produce a model whose uncertainty nobody
    !! could account for. `noise_at` exposes the log-noise interpolation so a
    !! caller can see what the model believes between its supplied points.

    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t, clone_kernel_into
    implicit none
    private

    public :: heteroskedastic_gp_t

    type :: heteroskedastic_gp_t
        type(kernel_t) :: kernel
        !! Kernel of the *log-noise* process. Usually smoother and longer in
        !! lengthscale than the signal's: noise levels vary with the regime,
        !! not with every wiggle of the function.
        type(kernel_t) :: noise_kernel
        type(cholesky_factorization_t) :: factorization
        type(cholesky_factorization_t) :: noise_factorization
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: alpha(:)
        !! Log of the supplied noise variance at each training input, centred,
        !! and the centre removed. Interpolating centred values means the model
        !! reverts to the *mean* log-noise away from data rather than to zero
        !! log-noise, which would be a claim of unit variance nobody made.
        real(dp), allocatable :: log_noise_alpha(:)
        real(dp) :: log_noise_mean = 0.0_dp
        real(dp) :: jitter = 1.0e-10_dp
        integer :: n_samples = 0
        integer :: n_features = 0
        logical :: fitted = .false.
    contains
        procedure, public :: fit => heteroskedastic_fit
        procedure, public :: predict => heteroskedastic_predict
        procedure, public :: noise_at => heteroskedastic_noise_at
    end type heteroskedastic_gp_t

contains

    !! Condition on inputs, targets, and the observation variance at each input.
    subroutine heteroskedastic_fit(self, x, y, noise_variance, kernel, &
            noise_kernel, status, jitter)
        class(heteroskedastic_gp_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(in) :: y(:)
        !! Observation variance at each training input. Positive: a zero would
        !! assert a noiseless measurement, whose log does not exist.
        real(dp), intent(in) :: noise_variance(:)
        type(kernel_t), intent(in) :: kernel
        type(kernel_t), intent(in) :: noise_kernel
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: jitter
        real(dp), allocatable :: gram(:, :), noise_gram(:, :), centred(:)
        integer :: n, i

        n = size(x, 1)
        if (n < 1 .or. size(y) /= n .or. size(noise_variance) /= n .or. &
            size(x, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "heteroskedastic GP: input, target, and noise shapes disagree")
            return
        end if
        if (any(noise_variance <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "heteroskedastic GP: every noise variance must be positive")
            return
        end if
        if (present(jitter)) then
            if (jitter < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "heteroskedastic GP: jitter must not be negative")
                return
            end if
            self%jitter = jitter
        end if

        self%n_samples = n
        self%n_features = size(x, 2)
        call clone_kernel_into(kernel, self%kernel)
        call clone_kernel_into(noise_kernel, self%noise_kernel)
        allocate (self%x_train, source=x)

        ! Signal posterior, with each row carrying *its own* noise. This is the
        ! whole difference from an ordinary GP: the diagonal is no longer
        ! constant.
        allocate (gram(n, n))
        call self%kernel%matrix(x, x, gram, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            gram(i, i) = gram(i, i) + noise_variance(i) + self%jitter
        end do
        call self%factorization%factorize(gram, status)
        if (status%code /= FORTNUM_OK) return

        allocate (self%alpha, source=y)
        call self%factorization%solve(self%alpha, status)
        if (status%code /= FORTNUM_OK) return

        ! Log-noise process, fitted to the centred log variances so that
        ! extrapolation reverts to the mean noise level rather than to one.
        allocate (centred(n))
        centred = log(noise_variance)
        self%log_noise_mean = sum(centred)/real(n, dp)
        centred = centred - self%log_noise_mean

        allocate (noise_gram(n, n))
        call self%noise_kernel%matrix(x, x, noise_gram, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            noise_gram(i, i) = noise_gram(i, i) + self%jitter
        end do
        call self%noise_factorization%factorize(noise_gram, status)
        if (status%code /= FORTNUM_OK) return

        allocate (self%log_noise_alpha, source=centred)
        call self%noise_factorization%solve(self%log_noise_alpha, status)
        if (status%code /= FORTNUM_OK) return

        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine heteroskedastic_fit

    !! Latent signal posterior. Excludes observation noise, matching the rest of
    !! FortML's GPs — `noise_at` supplies the noise separately, so a caller can
    !! choose which of the two it wants rather than being handed a sum it then
    !! has to unpick.
    subroutine heteroskedastic_predict(self, x, mean, variance, status)
        class(heteroskedastic_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior(:, :), work(:, :)
        integer :: n, i

        mean = 0.0_dp
        variance = 0.0_dp
        call check_query(self, x, size(mean), size(variance), status)
        if (status%code /= FORTNUM_OK) return

        n = size(x, 1)
        allocate (cross(self%n_samples, n), prior(n, n), work(self%n_samples, n))
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix(x, x, prior, status)
        if (status%code /= FORTNUM_OK) return

        mean = matmul(transpose(cross), self%alpha)
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            variance(i) = prior(i, i) - dot_product(cross(:, i), work(:, i))
            if (variance(i) < 0.0_dp) variance(i) = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine heteroskedastic_predict

    !! The model's belief about observation variance at arbitrary inputs.
    !!
    !! Interpolated on the log scale and exponentiated, so the result is
    !! positive by construction rather than by clipping — which is the reason
    !! the latent process lives in logs.
    subroutine heteroskedastic_noise_at(self, x, noise_variance, status)
        class(heteroskedastic_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: noise_variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :)
        real(dp), allocatable :: log_noise(:)
        integer :: n

        noise_variance = 0.0_dp
        call check_query(self, x, size(noise_variance), size(noise_variance), &
            status)
        if (status%code /= FORTNUM_OK) return

        n = size(x, 1)
        allocate (cross(self%n_samples, n), log_noise(n))
        call self%noise_kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        log_noise = self%log_noise_mean + matmul(transpose(cross), &
            self%log_noise_alpha)
        noise_variance = exp(log_noise)
        if (any(.not. ieee_is_finite(noise_variance))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "heteroskedastic GP: interpolated noise overflowed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine heteroskedastic_noise_at

    pure subroutine check_query(self, x, n_mean, n_variance, status)
        class(heteroskedastic_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: n_mean, n_variance
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "heteroskedastic GP: predict before fit")
            return
        end if
        if (size(x, 2) /= self%n_features .or. n_mean /= size(x, 1) .or. &
            n_variance /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "heteroskedastic GP: query or output shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_query

end module fortml_heteroskedastic_gp
