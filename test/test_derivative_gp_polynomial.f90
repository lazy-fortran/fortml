program test_derivative_gp_polynomial
    !! Independent finite-difference oracle for polynomial derivative GPs.
    !!
    !! Polynomial kernels already supported value/first-derivative covariance
    !! blocks, but their parameter and query-input products used to refuse.
    !! Keep this fixture separate from the production covariance helpers: the
    !! likelihood oracle assembles the polynomial blocks directly.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_polynomial_kernel
    implicit none

    type(gp_derivative_regression_t) :: model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    real(dp) :: x(3, 2), y(3, 1), query(2, 2), direction(2, 2)
    real(dp) :: mean(2, 1), mean_dot(2, 1), variance(2), variance_dot(2)
    real(dp) :: mean_plus(2, 1), mean_minus(2, 1), variance_plus(2), variance_minus(2)
    real(dp) :: mean_bar(2, 1), variance_bar(2), x_bar(2, 2)
    real(dp) :: theta(5), gradient(5), finite_gradient(5), value, plus, minus
    real(dp) :: parameter_direction(5), hvp(5), gradient_plus(5), gradient_minus(5)
    real(dp) :: finite_hvp(5), h, query_h, hvp_h, lhs, rhs
    integer :: i, j, failures

    failures = 0
    x = reshape([0.2_dp, 0.4_dp, 0.6_dp, 0.8_dp, 1.0_dp, 1.2_dp], shape(x))
    y(:, 1) = [0.8_dp, -0.3_dp, 1.1_dp]
    query = reshape([0.4_dp, 0.5_dp, 0.9_dp, 0.7_dp], shape(query))
    direction = reshape([0.03_dp, -0.02_dp, -0.01_dp, 0.04_dp], shape(direction))
    kernel = make_polynomial_kernel(2, 1.3_dp, 0.4_dp, 1.5_dp, 2.2_dp, status)
    call check(status_ok(status), "polynomial kernel construction", failures)
    call model%fit(x, [0, 1, 2], y, kernel, 0.06_dp, status, jitter=1.0e-10_dp)
    call check(status_ok(status), "polynomial derivative-GP fit", failures)
    if (.not. status_ok(status)) error stop 1

    theta = model%parameters()
    call model%hyperparameter_gradient(gradient, status)
    h = 2.0e-5_dp
    do i = 1, size(theta)
        plus = oracle_lml(theta + h*unit_vector(size(theta), i), x, [0, 1, 2], y, &
            0.06_dp, 1.0e-10_dp)
        minus = oracle_lml(theta - h*unit_vector(size(theta), i), x, [0, 1, 2], y, &
            0.06_dp, 1.0e-10_dp)
        finite_gradient(i) = (plus - minus)/(2.0_dp*h)
    end do
    call check(status_ok(status) .and. maxval(abs(gradient - finite_gradient)) < 3.0e-7_dp, &
        "polynomial derivative-GP hyperparameter gradient oracle", failures)

    parameter_direction = [0.07_dp, -0.03_dp, 0.02_dp, -0.05_dp, 0.06_dp]
    call model%hyperparameter_hvp(parameter_direction, hvp, status)
    hvp_h = 2.0e-4_dp
    gradient_plus = 0.0_dp
    gradient_minus = 0.0_dp
    do j = 1, size(theta)
        gradient_plus(j) = (oracle_lml(theta + hvp_h*parameter_direction + &
            h*unit_vector(size(theta), j), x, [0, 1, 2], y, 0.06_dp, 1.0e-10_dp) - &
            oracle_lml(theta + hvp_h*parameter_direction - &
            h*unit_vector(size(theta), j), x, [0, 1, 2], y, 0.06_dp, 1.0e-10_dp))/ &
            (2.0_dp*h)
        gradient_minus(j) = (oracle_lml(theta - hvp_h*parameter_direction + &
            h*unit_vector(size(theta), j), x, [0, 1, 2], y, 0.06_dp, 1.0e-10_dp) - &
            oracle_lml(theta - hvp_h*parameter_direction - &
            h*unit_vector(size(theta), j), x, [0, 1, 2], y, 0.06_dp, 1.0e-10_dp))/ &
            (2.0_dp*h)
    end do
    do i = 1, size(theta)
        finite_hvp(i) = (gradient_plus(i) - gradient_minus(i))/(2.0_dp*hvp_h)
    end do
    call check(status_ok(status) .and. maxval(abs(hvp - finite_hvp)) < 5.0e-4_dp, &
        "polynomial derivative-GP mixed HVP oracle", failures)
    if (.not. status_ok(status) .or. maxval(abs(hvp - finite_hvp)) >= 5.0e-4_dp) then
        write (error_unit, '(a,i0,2a)') "  polynomial HVP status=", status%code, ": ", trim(status%msg)
        write (error_unit, '(a,es14.6)') "  polynomial HVP max error=", &
            maxval(abs(hvp - finite_hvp))
        write (error_unit, '(a,5es14.6)') "  hvp=", hvp
        write (error_unit, '(a,5es14.6)') "  fd=", finite_hvp
        write (error_unit, '(a,5es14.6)') "  theta=", theta
    end if

    call model%predict_input_jvp(query, [0, 1], direction, mean, mean_dot, variance, &
        variance_dot, status)
    query_h = 2.0e-5_dp
    call model%predict(query + query_h*direction, [0, 1], mean_plus, variance_plus, status)
    call model%predict(query - query_h*direction, [0, 1], mean_minus, variance_minus, status)
    call check(status_ok(status) .and. maxval(abs(mean_dot - (mean_plus - mean_minus)/ &
        (2.0_dp*query_h))) < 3.0e-7_dp .and. maxval(abs(variance_dot - &
        (variance_plus - variance_minus)/(2.0_dp*query_h))) < 3.0e-7_dp, &
        "polynomial derivative-GP query-input JVP oracle", failures)

    mean_bar(:, 1) = [0.7_dp, -0.4_dp]
    variance_bar = [0.2_dp, -0.3_dp]
    call model%predict_input_vjp(query, [0, 1], mean_bar, variance_bar, x_bar, status)
    lhs = sum(mean_bar*mean_dot) + sum(variance_bar*variance_dot)
    rhs = sum(x_bar*direction)
    call check(status_ok(status) .and. abs(lhs - rhs) < 4.0e-7_dp, &
        "polynomial derivative-GP input VJP adjoint", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL polynomial derivative-GP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS polynomial derivative-GP finite-difference oracle"

contains

    function unit_vector(n, position) result(vector)
        integer, intent(in) :: n, position
        real(dp) :: vector(n)

        vector = 0.0_dp
        vector(position) = 1.0_dp
    end function unit_vector

    real(dp) function polynomial_covariance(x1, component1, x2, component2, parameters) result(value)
        real(dp), intent(in) :: x1(:), x2(:), parameters(:)
        integer, intent(in) :: component1, component2
        real(dp) :: variance, scale, offset, degree, base, coefficient, curvature

        variance = exp(parameters(1))
        scale = exp(parameters(2))
        offset = exp(parameters(3))
        degree = exp(parameters(4))
        base = offset + scale*dot_product(x1, x2)
        if (base <= 0.0_dp) then
            value = 0.0_dp
            return
        end if
        if (component1 == 0 .and. component2 == 0) then
            value = variance*base**degree
        else
            coefficient = variance*degree*scale*base**(degree - 1.0_dp)
            if (component1 > 0 .and. component2 == 0) then
                value = coefficient*x2(component1)
            else if (component1 == 0 .and. component2 > 0) then
                value = coefficient*x1(component2)
            else
                curvature = variance*degree*(degree - 1.0_dp)*scale*scale* &
                    base**(degree - 2.0_dp)
                value = coefficient*merge(1.0_dp, 0.0_dp, component1 == component2) + &
                    curvature*x2(component1)*x1(component2)
            end if
        end if
    end function polynomial_covariance

    real(dp) function oracle_lml(parameters, x, components, y, noise, jitter) result(value)
        real(dp), intent(in) :: parameters(:), x(:, :), y(:, :), noise, jitter
        integer, intent(in) :: components(:)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: local_status
        real(dp), allocatable :: covariance(:, :), alpha(:, :)
        real(dp) :: logdet
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)), alpha(size(y, 1), size(y, 2)))
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = polynomial_covariance(x(i, :), components(i), &
                    x(j, :), components(j), parameters)
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(parameters(5)) + jitter
        end do
        call factor%factorize(covariance, local_status)
        if (.not. status_ok(local_status)) then
            value = huge(1.0_dp)
            return
        end if
        alpha = y
        call factor%solve(alpha, local_status)
        call factor%log_determinant(logdet, local_status)
        value = -0.5_dp*sum(y*alpha) - 0.5_dp*logdet - &
            0.5_dp*real(size(x, 1), dp)*log(2.0_dp*acos(-1.0_dp))
    end function oracle_lml

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [polynomial derivative-GP] "//description
        end if
    end subroutine check

end program test_derivative_gp_polynomial
