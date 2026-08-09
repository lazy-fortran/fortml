program test_positive_linear_regression
    !! Independent weighted constrained least-squares and product oracle.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_positive_linear_regression, only: positive_linear_regression_t
    implicit none
    integer, parameter :: dp = real64
    integer :: failures

    failures = 0
    call test_weighted_constraint_oracle(failures)
    call test_fixed_state_products(failures)
    call test_transactional_and_device_contract(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " positive-linear test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS positive linear regression independent oracle"

contains

    subroutine test_weighted_constraint_oracle(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(4, 2), y(4, 2), weights(4), prediction(4, 2)
        real(dp), allocatable :: theta(:), coefficients(:, :)
        type(positive_linear_regression_t) :: model
        type(fortnum_status_t) :: status

        x = reshape([1.0_dp, 0.0_dp, -1.0_dp, 0.0_dp, &
            0.0_dp, 1.0_dp, 0.0_dp, -1.0_dp], shape(x))
        y(:, 1) = 1.0_dp + 2.0_dp*x(:, 1) - 3.0_dp*x(:, 2)
        y(:, 2) = 0.5_dp + 0.5_dp*x(:, 1)
        weights = [1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp]
        call model%fit(x, y, status, sample_weight=weights, tolerance=1.0e-11_dp)
        call check(status_ok(status), "weighted constrained fit", failures)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        coefficients = model%coefficients()
        call check(size(theta) == 6 .and. all(shape(coefficients) == [3, 2]), &
            "packed multi-output metadata", failures)
        call check(maxval(abs(coefficients - reshape([1.0_dp, 2.0_dp, 0.0_dp, &
            0.5_dp, 0.5_dp, 0.0_dp], shape(coefficients)))) < 2.0e-7_dp, &
            "weighted nonnegative coefficient oracle", failures)
        call model%predict(x, prediction, status)
        call check(status_ok(status), "constrained prediction", failures)
        call check(minval(coefficients(2:, :)) >= -1.0e-13_dp, &
            "feature coefficients projected nonnegative", failures)
        call check(model%fit_intercept() .and. .not. model%nonnegative_intercept() &
            .and. model%fitted(), "fit metadata", failures)
    end subroutine test_weighted_constraint_oracle

    subroutine test_fixed_state_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(4, 2), x_dot(4, 2), y(4, 2), y_dot(4, 2)
        real(dp) :: prediction(4, 2), y_plus(4, 2), y_minus(4, 2)
        real(dp) :: theta_dot(6), theta_bar(6), x_bar(4, 2), u(4, 2)
        real(dp), allocatable :: theta(:), shifted(:)
        real(dp) :: h, fd_error, lhs, rhs
        type(positive_linear_regression_t) :: model
        type(fortnum_status_t) :: status

        x = reshape([1.0_dp, 0.0_dp, -1.0_dp, 0.0_dp, &
            0.0_dp, 1.0_dp, 0.0_dp, -1.0_dp], shape(x))
        y(:, 1) = 1.0_dp + 2.0_dp*x(:, 1) - 3.0_dp*x(:, 2)
        y(:, 2) = 0.5_dp + 0.5_dp*x(:, 1)
        x_dot = reshape([0.2_dp, -0.1_dp, 0.3_dp, -0.4_dp, &
            -0.3_dp, 0.5_dp, 0.2_dp, 0.1_dp], shape(x_dot))
        call model%fit(x, y, status, tolerance=1.0e-11_dp)
        call check(status_ok(status), "products fit", failures)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        theta_dot = [0.03_dp, -0.02_dp, 0.0_dp, 0.04_dp, -0.05_dp, 0.0_dp]
        u = reshape([0.02_dp, -0.03_dp, 0.04_dp, 0.01_dp, &
            -0.05_dp, 0.06_dp, 0.07_dp, -0.08_dp], shape(u))
        h = 2.0e-6_dp
        call model%predict_jvp(x, theta_dot, x_dot, prediction, y_dot, status)
        call check(status_ok(status), "prediction JVP status", failures)
        shifted = theta + h*theta_dot
        call model%set_parameters(shifted, status)
        call model%predict(x + h*x_dot, y_plus, status)
        shifted = theta - h*theta_dot
        call model%set_parameters(shifted, status)
        call model%predict(x - h*x_dot, y_minus, status)
        fd_error = maxval(abs(y_dot - (y_plus-y_minus)/(2.0_dp*h)))
        call check(fd_error < 3.0e-8_dp, "prediction JVP finite difference", failures)
        call model%set_parameters(theta, status)
        call model%predict_vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*y_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-11_dp, &
            "prediction VJP duality", failures)
    end subroutine test_fixed_state_products

    subroutine test_transactional_and_device_contract(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(4, 1), y(4), prediction(4, 1)
        real(dp) :: weights(4), bad(2)
        real(dp), allocatable :: before(:)
        type(positive_linear_regression_t) :: model
        type(fortml_device_t) :: cuda
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
        y = [1.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        weights = 1.0_dp
        call model%fit(x, y, status, sample_weight=weights)
        before = model%parameters()
        bad = [before(1), -1.0_dp]
        call model%set_parameters(bad, status)
        call check(.not. status_ok(status), "negative parameter refusal", failures)
        call check(maxval(abs(model%parameters()-before)) == 0.0_dp, &
            "set-parameters transactional state", failures)
        call model%fit(x, y, status, fit_intercept=.false., &
            nonnegative_intercept=.true.)
        call check(.not. status_ok(status), &
            "intercept constraint without intercept refusal", failures)
        call model%predict(x, prediction, status)
        call check(status_ok(status), "state retained after failed fit", failures)
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        call model%predict_device(cuda, x, prediction, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "typed CUDA refusal", failures)
        call check(model%device_supported(FORTML_DEVICE_CUDA) .eqv. .false. &
            .and. model%device_supported(FORTML_DEVICE_CPU), &
            "device capability metadata", failures)
    end subroutine test_transactional_and_device_contract

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: "//trim(label)
        end if
    end subroutine check

end program test_positive_linear_regression
