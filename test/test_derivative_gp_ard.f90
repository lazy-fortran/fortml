program test_derivative_gp_ard
    !! Independent finite-difference oracle for ARD-RBF derivative GPs.
    !! The covariance oracle below is written directly from the ARD equations
    !! and does not call the production derivative-covariance helpers.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_rbf_ard_kernel
    implicit none

    type(gp_derivative_regression_t) :: model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 2), y(4, 1), theta(4), gradient(4), finite_gradient(4)
    real(dp) :: direction(4), hvp(4), gradient_plus(4), gradient_minus(4)
    real(dp) :: h, hvp_h
    integer :: i, failures

    failures = 0
    x = reshape([0.1_dp, -0.4_dp, 0.7_dp, 0.9_dp, &
        -0.2_dp, 0.5_dp, 1.1_dp, -0.8_dp], shape(x))
    y(:, 1) = [0.6_dp, -0.2_dp, 1.0_dp, 0.3_dp]
    kernel = make_rbf_ard_kernel(2, 1.4_dp, [0.75_dp, 1.25_dp], status)
    call check(status_ok(status), "ARD RBF construction", failures)
    call model%fit(x, [0, 1, 2, 0], y, kernel, 0.08_dp, status, jitter=1.0e-10_dp)
    call check(status_ok(status), "ARD derivative-GP fit", failures)
    if (.not. status_ok(status)) error stop 1

    theta = model%parameters()
    call model%hyperparameter_gradient(gradient, status)
    h = 2.0e-5_dp
    do i = 1, size(theta)
        finite_gradient(i) = (oracle_lml(theta + h*unit_vector(size(theta), i), x, &
            [0, 1, 2, 0], y, 0.08_dp, 1.0e-10_dp) - &
            oracle_lml(theta - h*unit_vector(size(theta), i), x, [0, 1, 2, 0], y, &
            0.08_dp, 1.0e-10_dp))/(2.0_dp*h)
    end do
    call check(status_ok(status) .and. maxval(abs(gradient - finite_gradient)) < 4.0e-6_dp, &
        "ARD derivative-GP hyperparameter gradient oracle", failures)

    direction = [0.11_dp, -0.07_dp, 0.19_dp, 0.05_dp]
    call model%hyperparameter_hvp(direction, hvp, status)
    hvp_h = 2.0e-4_dp
    gradient_plus = oracle_gradient(theta + hvp_h*direction, x, [0, 1, 2, 0], y, &
        0.08_dp, 1.0e-10_dp)
    gradient_minus = oracle_gradient(theta - hvp_h*direction, x, [0, 1, 2, 0], y, &
        0.08_dp, 1.0e-10_dp)
    call check(status_ok(status) .and. maxval(abs(hvp - (gradient_plus - gradient_minus)/ &
        (2.0_dp*hvp_h))) < 7.0e-4_dp, &
        "ARD derivative-GP mixed HVP oracle", failures)
    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL ARD derivative-GP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS ARD derivative-GP finite-difference oracle"

contains

    function unit_vector(n, position) result(vector)
        integer, intent(in) :: n, position
        real(dp) :: vector(n)

        vector = 0.0_dp
        vector(position) = 1.0_dp
    end function unit_vector

    real(dp) function ard_covariance(x1, component1, x2, component2, parameters) result(value)
        real(dp), intent(in) :: x1(:), x2(:), parameters(:)
        integer, intent(in) :: component1, component2
        real(dp) :: difference(size(x1)), q(size(x1)), base
        integer :: i, j

        do i = 1, size(x1)
            difference(i) = x1(i) - x2(i)
            q(i) = exp(-2.0_dp*parameters(i + 1))
        end do
        base = exp(parameters(1) - 0.5_dp*sum(difference*difference*q))
        if (component1 == 0 .and. component2 == 0) then
            value = base
        else if (component1 > 0 .and. component2 == 0) then
            value = -base*difference(component1)*q(component1)
        else if (component1 == 0 .and. component2 > 0) then
            value = base*difference(component2)*q(component2)
        else
            value = base*(q(component1)*merge(1.0_dp, 0.0_dp, component1 == component2) - &
                difference(component1)*difference(component2)*q(component1)*q(component2))
        end if
    end function ard_covariance

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
                covariance(i, j) = ard_covariance(x(i, :), components(i), x(j, :), &
                    components(j), parameters)
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(parameters(size(parameters))) + jitter
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

    function oracle_gradient(parameters, x, components, y, noise, jitter) result(gradient)
        real(dp), intent(in) :: parameters(:), x(:, :), y(:, :), noise, jitter
        integer, intent(in) :: components(:)
        real(dp) :: gradient(size(parameters)), h_local
        integer :: i

        h_local = 2.0e-5_dp
        do i = 1, size(parameters)
            gradient(i) = (oracle_lml(parameters + h_local*unit_vector(size(parameters), i), &
                x, components, y, noise, jitter) - &
                oracle_lml(parameters - h_local*unit_vector(size(parameters), i), &
                x, components, y, noise, jitter))/(2.0_dp*h_local)
        end do
    end function oracle_gradient

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [ARD derivative-GP] "//description
        end if
    end subroutine check

end program test_derivative_gp_ard
