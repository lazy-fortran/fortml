program test_derivative_gp_periodic_hvp
    !! Independent finite-difference oracle for periodic mixed-observation HVPs.
    !! The oracle re-derives the radial covariance and refactors its own dense
    !! covariance; it never calls FortML derivative covariance or gradients.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_kernels, only: kernel_t, make_periodic_kernel
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    implicit none

    type(gp_derivative_regression_t) :: model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    real(dp) :: x(5, 2), y(5, 1), theta(4), direction(4), hvp(4)
    real(dp) :: gradient_plus(4), gradient_minus(4), hvp_ref(4)
    real(dp) :: value, expected, h_gradient, h_hvp, error
    integer :: components(5), failures

    failures = 0
    x = reshape([ &
        -0.70_dp, 0.20_dp, 0.35_dp, -0.40_dp, 0.90_dp, 0.65_dp, &
        -0.10_dp, 1.10_dp, 0.55_dp, -0.85_dp], shape(x))
    y(:, 1) = [0.80_dp, -0.25_dp, 0.45_dp, 1.10_dp, -0.60_dp]
    components = [0, 1, 2, 0, 1]
    kernel = make_periodic_kernel(2, 1.25_dp, 0.83_dp, 1.47_dp, status)
    call check(status_ok(status), "periodic kernel constructor", failures)
    call model%fit(x, components, y, kernel, 0.055_dp, status, jitter=1.0e-10_dp)
    call check(status_ok(status), "periodic derivative-GP fit", failures)
    theta = model%parameters()

    call model%log_marginal_likelihood(value, status)
    expected = oracle_lml(theta, x, components, y, 1.0e-10_dp)
    call check(status_ok(status) .and. abs(value - expected) < 3.0e-10_dp, &
        "independent periodic likelihood oracle", failures)

    direction = [0.13_dp, -0.08_dp, 0.11_dp, -0.05_dp]
    call model%hyperparameter_hvp(direction, hvp, status)
    call check(status_ok(status), "periodic mixed HVP is supported", failures)
    h_gradient = 3.0e-6_dp
    h_hvp = 2.0e-4_dp
    call oracle_gradient(theta + h_hvp*direction, x, components, y, 1.0e-10_dp, &
        h_gradient, gradient_plus)
    call oracle_gradient(theta - h_hvp*direction, x, components, y, 1.0e-10_dp, &
        h_gradient, gradient_minus)
    hvp_ref = (gradient_plus - gradient_minus)/(2.0_dp*h_hvp)
    error = maxval(abs(hvp - hvp_ref))
    call check(error < 3.0e-4_dp, "periodic mixed HVP finite-difference oracle", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL periodic derivative-GP HVP cases: ", failures
        error stop 1
    end if
    write (*, '(a,es12.4)') "PASS periodic derivative-GP HVP oracle, max error ", error

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [periodic-derivative-hvp] "//description
        end if
    end subroutine check

    real(dp) function oracle_lml(theta, x, components, y, jitter) result(value)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), jitter
        integer, intent(in) :: components(:)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: local_status
        real(dp), allocatable :: covariance(:, :), alpha(:, :)
        real(dp) :: logdet
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)))
        allocate(alpha, source=y)
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = oracle_covariance(theta, x(i, :), components(i), &
                    x(j, :), components(j))
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(size(theta))) + jitter
        end do
        call factor%factorize(covariance, local_status)
        call factor%solve(alpha, local_status)
        call factor%log_determinant(logdet, local_status)
        value = -0.5_dp*sum(y*alpha) - 0.5_dp*logdet - &
            0.5_dp*real(size(x, 1), dp)*log(2.0_dp*acos(-1.0_dp))
    end function oracle_lml

    subroutine oracle_gradient(theta, x, components, y, jitter, h, gradient)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), jitter, h
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: gradient(:)
        real(dp) :: plus(size(theta)), minus(size(theta))
        integer :: i
        do i = 1, size(theta)
            plus = theta
            minus = theta
            plus(i) = plus(i) + h
            minus(i) = minus(i) - h
            gradient(i) = (oracle_lml(plus, x, components, y, jitter) - &
                oracle_lml(minus, x, components, y, jitter))/(2.0_dp*h)
        end do
    end subroutine oracle_gradient

    real(dp) function oracle_covariance(theta, x1, component1, x2, component2) result(value)
        real(dp), intent(in) :: theta(:), x1(:), x2(:)
        integer, intent(in) :: component1, component2
        real(dp) :: difference(size(x1)), squared_distance, distance
        real(dp) :: variance, lengthscale, period, b, c, t0, t1, t2
        real(dp) :: f, f1, f2
        integer :: i, j

        difference = x1 - x2
        squared_distance = dot_product(difference, difference)
        distance = sqrt(squared_distance)
        variance = exp(theta(1))
        lengthscale = exp(theta(2))
        period = exp(theta(3))
        b = 2.0_dp/(lengthscale*lengthscale)
        c = acos(-1.0_dp)/period
        if (distance <= 1.0e-8_dp) then
            t0 = c*c*squared_distance - c**4*squared_distance**2/3.0_dp
            t1 = c*c - 2.0_dp*c**4*squared_distance/3.0_dp
            t2 = -2.0_dp*c**4/3.0_dp + 4.0_dp*c**6*squared_distance/15.0_dp
        else
            t0 = sin(c*distance)**2
            t1 = c*sin(2.0_dp*c*distance)/(2.0_dp*distance)
            t2 = c*(2.0_dp*c*distance*cos(2.0_dp*c*distance) - &
                sin(2.0_dp*c*distance))/(4.0_dp*distance**3)
        end if
        f = variance*exp(-b*t0)
        f1 = -b*t1*f
        f2 = (b*b*t1*t1 - b*t2)*f
        if (component1 == 0 .and. component2 == 0) then
            value = f
        else if (component1 > 0 .and. component2 == 0) then
            value = 2.0_dp*f1*difference(component1)
        else if (component1 == 0 .and. component2 > 0) then
            value = -2.0_dp*f1*difference(component2)
        else
            value = -2.0_dp*f1*merge(1.0_dp, 0.0_dp, component1 == component2) - &
                4.0_dp*f2*difference(component1)*difference(component2)
        end if
    end function oracle_covariance

end program test_derivative_gp_periodic_hvp
