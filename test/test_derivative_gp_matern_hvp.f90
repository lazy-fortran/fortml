program test_derivative_gp_matern_hvp
    !! Independent mixed-observation Matérn HVP oracle.
    !! The reference covariance and central differences are implemented here,
    !! without calling FortML's derivative-covariance helpers.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_matern32_kernel, make_matern52_kernel
    implicit none

    integer :: failures

    failures = 0
    call check_kernel(32, failures)
    call check_kernel(52, failures)
    if (failures /= 0) then
        write (error_unit, '(a,i0)') 'FAIL Matérn derivative-GP HVP cases: ', failures
        error stop 1
    end if
    write (*, '(a)') 'PASS Matérn 3/2 and 5/2 derivative-GP HVP independent oracles'

contains

    subroutine check_kernel(kind, failures)
        integer, intent(in) :: kind
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortml_device_t) :: cuda
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), theta(3), direction(3), hvp(3)
        real(dp) :: gradient(3), finite_gradient(3), hvp_reference(3)
        real(dp) :: mean(1, 1), variance(1), value
        real(dp) :: h_gradient, h_hvp, error_gradient, error_hvp
        integer :: components(3), i

        x(:, 1) = [-0.20_dp, 0.35_dp, 0.90_dp]
        y(:, 1) = [0.70_dp, -0.10_dp, 0.55_dp]
        components = [0, 1, 0]
        if (kind == 32) then
            kernel = make_matern32_kernel(1, 1.35_dp, 0.78_dp, status)
        else
            kernel = make_matern52_kernel(1, 1.35_dp, 0.78_dp, status)
        end if
        call check(status_ok(status), 'Matern kernel constructor', failures)
        call model%fit(x, components, y, kernel, 0.045_dp, status, jitter=1.0e-10_dp)
        call check(status_ok(status), 'mixed-observation Matern fit', failures)
        if (.not. status_ok(status)) return

        theta = model%parameters()
        call model%log_marginal_likelihood(value, status)
        call check(status_ok(status) .and. abs(value - oracle_lml(theta, x, components, y, &
            kind, 1.0e-10_dp)) < 3.0e-10_dp, 'Matern likelihood oracle', failures)
        call model%hyperparameter_gradient(gradient, status)
        call check(status_ok(status), 'Matern likelihood gradient', failures)
        h_gradient = 2.0e-5_dp
        do i = 1, size(theta)
            finite_gradient(i) = (oracle_lml(theta + h_gradient*unit_vector(3, i), x, &
                components, y, kind, 1.0e-10_dp) - oracle_lml(theta - &
                h_gradient*unit_vector(3, i), x, components, y, kind, 1.0e-10_dp))/ &
                (2.0_dp*h_gradient)
        end do
        error_gradient = maxval(abs(gradient - finite_gradient))
        call check(error_gradient < 3.0e-7_dp, 'Matern likelihood gradient oracle', failures)

        direction = [0.17_dp, -0.11_dp, 0.08_dp]
        call model%hyperparameter_hvp(direction, hvp, status)
        call check(status_ok(status), 'Matern mixed-observation HVP support', failures)
        h_hvp = 2.0e-3_dp
        hvp_reference(1) = mixed_second_difference(theta, direction, unit_vector(3, 1), &
            x, components, y, kind, 1.0e-10_dp, h_hvp)
        hvp_reference(2) = mixed_second_difference(theta, direction, unit_vector(3, 2), &
            x, components, y, kind, 1.0e-10_dp, h_hvp)
        hvp_reference(3) = mixed_second_difference(theta, direction, unit_vector(3, 3), &
            x, components, y, kind, 1.0e-10_dp, h_hvp)
        error_hvp = maxval(abs(hvp - hvp_reference))
        call check(error_hvp < 3.0e-4_dp, 'Matern mixed HVP central oracle', failures)

        ! The new leaves remain CPU-reference only until a resident derivative
        ! covariance graph exists.  Refusal must happen before output writes.
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        mean = 1234.0_dp
        variance = 5678.0_dp
        call model%predict_device(cuda, reshape([0.25_dp], [1, 1]), [0], mean, variance, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            'Matern CUDA prediction refusal', failures)
        call check(all(mean == 1234.0_dp) .and. all(variance == 5678.0_dp), &
            'Matern CUDA refusal leaves outputs untouched', failures)

        call model%log_marginal_likelihood(value, status)
        call check(status_ok(status), 'Matern likelihood remains usable after refusal', failures)
        write (*, '(a,i0,a,es10.3,a,es10.3)') '  Matern-', kind, &
            '/2 gradient error ', error_gradient, ', HVP error ', error_hvp
    end subroutine check_kernel

    real(dp) function mixed_second_difference(theta, direction, probe, x, components, y, &
            kind, jitter, h) result(value)
        real(dp), intent(in) :: theta(:), direction(:), probe(:), x(:, :), y(:, :), jitter, h
        integer, intent(in) :: components(:), kind
        real(dp) :: pp(size(theta)), pm(size(theta)), mp(size(theta)), mm(size(theta))

        pp = theta + h*direction + h*probe
        pm = theta + h*direction - h*probe
        mp = theta - h*direction + h*probe
        mm = theta - h*direction - h*probe
        value = (oracle_lml(pp, x, components, y, kind, jitter) - &
            oracle_lml(pm, x, components, y, kind, jitter) - &
            oracle_lml(mp, x, components, y, kind, jitter) + &
            oracle_lml(mm, x, components, y, kind, jitter))/(4.0_dp*h*h)
    end function mixed_second_difference

    real(dp) function oracle_lml(theta, x, components, y, kind, jitter) result(value)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), jitter
        integer, intent(in) :: components(:), kind
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: status
        real(dp), allocatable :: covariance(:, :), alpha(:, :)
        real(dp) :: logdet
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)))
        allocate(alpha, source=y)
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = oracle_covariance(theta, x(i, 1), components(i), &
                    x(j, 1), components(j), kind)
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(3)) + jitter
        end do
        call factor%factorize(covariance, status)
        call factor%solve(alpha, status)
        call factor%log_determinant(logdet, status)
        value = -0.5_dp*sum(y*alpha) - 0.5_dp*logdet - &
            0.5_dp*real(size(x, 1), dp)*log(2.0_dp*acos(-1.0_dp))
    end function oracle_lml

    real(dp) function oracle_covariance(theta, x1, component1, x2, component2, kind) result(value)
        real(dp), intent(in) :: theta(:), x1, x2
        integer, intent(in) :: component1, component2, kind
        real(dp) :: delta, distance, variance, lengthscale, z, a, exponential
        real(dp) :: f, first, second

        delta = x1 - x2
        distance = abs(delta)
        variance = exp(theta(1))
        lengthscale = exp(theta(2))
        if (kind == 32) then
            a = sqrt(3.0_dp)
            z = distance/lengthscale
            exponential = exp(-a*z)
            f = variance*(1.0_dp + a*z)*exponential
            first = -3.0_dp*variance*z*exponential/lengthscale
            second = 3.0_dp*variance*(a*z - 1.0_dp)*exponential/(lengthscale**2)
        else
            a = sqrt(5.0_dp)
            z = distance/lengthscale
            exponential = exp(-a*z)
            f = variance*(1.0_dp + a*z + 5.0_dp*z*z/3.0_dp)*exponential
            first = -(5.0_dp/3.0_dp)*variance*z*(1.0_dp + a*z)*exponential/lengthscale
            second = (5.0_dp/3.0_dp)*variance*(5.0_dp*z*z - a*z - 1.0_dp)* &
                exponential/(lengthscale**2)
        end if
        if (component1 == 0 .and. component2 == 0) then
            value = f
        else if (component1 > 0 .and. component2 == 0) then
            if (distance == 0.0_dp) then
                value = 0.0_dp
            else
                value = first*delta/distance
            end if
        else if (component1 == 0 .and. component2 > 0) then
            if (distance == 0.0_dp) then
                value = 0.0_dp
            else
                value = -first*delta/distance
            end if
        else if (distance == 0.0_dp) then
            value = -second
        else
            value = -second
        end if
    end function oracle_covariance

    function unit_vector(n, position) result(vector)
        integer, intent(in) :: n, position
        real(dp) :: vector(n)

        vector = 0.0_dp
        vector(position) = 1.0_dp
    end function unit_vector

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') '  FAIL [matern-derivative-hvp] '//description
        end if
    end subroutine check

end program test_derivative_gp_matern_hvp
