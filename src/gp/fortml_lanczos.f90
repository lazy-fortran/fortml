module fortml_lanczos
    !! Lanczos estimators over any `linear_operator_t`: a stochastic Lanczos
    !! quadrature log determinant and a LOVE-style predictive variance.
    !!
    !! Both avoid forming the covariance matrix. The log determinant uses the
    !! Hutchinson estimator with Gauss quadrature: for a probe `z`, the
    !! Lanczos tridiagonal `T` after `k` steps gives
    !! `z^T log(A) z ~ ||z||^2 e_1^T log(T) e_1`, evaluated through the
    !! eigendecomposition of `T`. Rademacher probes make `||z||^2` exactly the
    !! sample count, and the probe stream is seeded, so an estimate is a
    !! reproducible function of the seed.
    !!
    !! The predictive variance runs Lanczos from the cross-covariance vector
    !! itself, so `k_*^T A^{-1} k_* = ||k_*||^2 e_1^T T^{-1} e_1` becomes a
    !! tridiagonal solve. With as many steps as samples this is exact up to
    !! round-off, which is what the test uses as its oracle; with fewer steps it
    !! is the usual low-rank approximation.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortml_linear_operator, only: linear_operator_t
    use, intrinsic :: iso_fortran_env, only: int64
    implicit none
    private

    public :: lanczos_log_determinant
    public :: lanczos_predictive_variance

    interface
        subroutine dstev(jobz, n, d, e, z, ldz, work, info)
            import :: dp
            character, intent(in) :: jobz
            integer, intent(in) :: n, ldz
            real(dp), intent(inout) :: d(*), e(*)
            real(dp), intent(out) :: z(ldz, *)
            real(dp), intent(out) :: work(*)
            integer, intent(out) :: info
        end subroutine dstev
    end interface

contains

    subroutine lanczos_log_determinant(operator, n_probes, n_steps, seed, &
            value, status)
        !! Stochastic Lanczos quadrature estimate of `log det(A)`.
        class(linear_operator_t), intent(in) :: operator
        integer, intent(in) :: n_probes, n_steps, seed
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        type(rng_t) :: generator
        real(dp), allocatable :: probe(:), basis(:, :)
        real(dp), allocatable :: alpha(:), beta(:), eigenvalues(:)
        real(dp), allocatable :: eigenvectors(:, :), work(:), offdiagonal(:)
        real(dp) :: draw, quadrature, total
        integer :: n_samples, requested_steps, steps, probe_index, i, j, info

        value = 0.0_dp
        n_samples = operator%sample_count()
        if (n_samples < 1 .or. n_probes < 1 .or. n_steps < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lanczos log determinant: shape is invalid")
            return
        end if
        requested_steps = min(n_steps, n_samples)
        call rng_seed(generator, int(seed, int64), status)
        if (status%code /= FORTNUM_OK) return

        allocate(probe(n_samples), basis(n_samples, requested_steps))
        allocate(alpha(requested_steps), beta(max(requested_steps - 1, 1)))
        allocate(eigenvalues(requested_steps))
        allocate(offdiagonal(max(requested_steps - 1, 1)))
        allocate(eigenvectors(requested_steps, requested_steps))
        allocate(work(max(2*requested_steps - 2, 1)))

        total = 0.0_dp
        do probe_index = 1, n_probes
            do i = 1, n_samples
                call rng_uniform(generator, draw)
                probe(i) = merge(1.0_dp, -1.0_dp, draw >= 0.5_dp)
            end do
            call run_lanczos(operator, probe, requested_steps, basis, alpha, &
                beta, steps, status)
            if (status%code /= FORTNUM_OK) return

            eigenvalues(1:steps) = alpha(1:steps)
            if (steps > 1) offdiagonal(1:steps - 1) = beta(1:steps - 1)
            call dstev('V', steps, eigenvalues, offdiagonal, eigenvectors, &
                requested_steps, work, info)
            if (info /= 0) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "Lanczos log determinant: tridiagonal eigensolve failed")
                return
            end if
            quadrature = 0.0_dp
            do j = 1, steps
                if (eigenvalues(j) <= 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "Lanczos log determinant: operator is not positive definite")
                    return
                end if
                quadrature = quadrature + &
                    eigenvectors(1, j)*eigenvectors(1, j)*log(eigenvalues(j))
            end do
            ! Rademacher probes have ||z||^2 = n exactly.
            total = total + real(n_samples, dp)*quadrature
        end do

        value = total/real(n_probes, dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine lanczos_log_determinant

    subroutine lanczos_predictive_variance(operator, cross_covariance, &
            prior_variance, n_steps, variance, status)
        !! LOVE-style `prior_variance - k_*^T A^{-1} k_*`.
        class(linear_operator_t), intent(in) :: operator
        real(dp), intent(in) :: cross_covariance(:)
        real(dp), intent(in) :: prior_variance
        integer, intent(in) :: n_steps
        real(dp), intent(out) :: variance
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: basis(:, :), alpha(:), beta(:)
        real(dp), allocatable :: diagonal(:), lower(:), rhs(:)
        real(dp) :: norm_squared, factor
        integer :: n_samples, requested_steps, steps, i

        variance = 0.0_dp
        n_samples = operator%sample_count()
        if (n_samples < 1 .or. n_steps < 1 .or. &
            size(cross_covariance) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lanczos predictive variance: shape is invalid")
            return
        end if
        norm_squared = sum(cross_covariance*cross_covariance)
        if (norm_squared <= 0.0_dp) then
            variance = prior_variance
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        requested_steps = min(n_steps, n_samples)
        allocate(basis(n_samples, requested_steps), alpha(requested_steps))
        allocate(beta(max(requested_steps - 1, 1)))
        call run_lanczos(operator, cross_covariance, requested_steps, basis, &
            alpha, beta, steps, status)
        if (status%code /= FORTNUM_OK) return

        ! Solve T y = e_1 by Thomas elimination; y(1) is e_1^T T^{-1} e_1.
        allocate(diagonal(steps), lower(max(steps - 1, 1)), rhs(steps))
        diagonal = alpha(1:steps)
        if (steps > 1) lower(1:steps - 1) = beta(1:steps - 1)
        rhs = 0.0_dp
        rhs(1) = 1.0_dp
        do i = 2, steps
            if (diagonal(i - 1) == 0.0_dp) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "Lanczos predictive variance: tridiagonal solve broke down")
                return
            end if
            factor = lower(i - 1)/diagonal(i - 1)
            diagonal(i) = diagonal(i) - factor*lower(i - 1)
            rhs(i) = rhs(i) - factor*rhs(i - 1)
        end do
        if (diagonal(steps) == 0.0_dp) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "Lanczos predictive variance: tridiagonal solve broke down")
            return
        end if
        rhs(steps) = rhs(steps)/diagonal(steps)
        do i = steps - 1, 1, -1
            rhs(i) = (rhs(i) - lower(i)*rhs(i + 1))/diagonal(i)
        end do

        variance = prior_variance - norm_squared*rhs(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine lanczos_predictive_variance

    subroutine run_lanczos(operator, start, steps, basis, alpha, beta, &
            actual_steps, status)
        !! Symmetric Lanczos with full reorthogonalization. The step count is
        !! small by construction here, so keeping the basis orthogonal costs
        !! little and keeps the quadrature weights meaningful.
        class(linear_operator_t), intent(in) :: operator
        real(dp), intent(in) :: start(:)
        integer, intent(in) :: steps
        real(dp), intent(out) :: basis(:, :), alpha(:), beta(:)
        integer, intent(out) :: actual_steps
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: work(:)
        real(dp) :: norm, projection
        integer :: i, j

        allocate(work(size(start)))
        norm = sqrt(sum(start*start))
        if (norm <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lanczos: the start vector is zero")
            return
        end if
        basis = 0.0_dp
        alpha = 0.0_dp
        beta = 0.0_dp
        actual_steps = steps
        basis(:, 1) = start/norm

        do i = 1, steps
            call operator%matvec(basis(:, i), work)
            alpha(i) = sum(basis(:, i)*work)
            if (i == steps) exit
            work = work - alpha(i)*basis(:, i)
            if (i > 1) work = work - beta(i - 1)*basis(:, i - 1)
            do j = 1, i
                projection = sum(basis(:, j)*work)
                work = work - projection*basis(:, j)
            end do
            beta(i) = sqrt(sum(work*work))
            if (beta(i) <= 1.0e-14_dp*max(abs(alpha(i)), 1.0_dp)) then
                ! The start vector reached an invariant subspace. The reduced
                ! tridiagonal is already exact on that subspace; padding it
                ! with zero rows would introduce false zero eigenvalues.
                beta(i) = 0.0_dp
                actual_steps = i
                exit
            end if
            basis(:, i + 1) = work/beta(i)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine run_lanczos

end module fortml_lanczos
