program test_basis_linear_regression
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis, &
        make_polynomial_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortml_basis_linear_regression, only: basis_linear_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_fit_and_products(failures)
    call test_polynomial_fit_and_lowering(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " basis linear regression test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_fit_and_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(8, 1), x_dot(8, 1), y(8, 2), prediction(8, 2)
        real(dp) :: prediction_dot(8, 2), u(8, 2), x_bar(8, 1)
        real(dp) :: theta_dot(7), theta_bar(7), theta(7), theta_plus(7)
        real(dp) :: theta_minus(7), prediction_plus(8, 2), prediction_minus(8, 2)
        real(dp) :: lhs, rhs, h, frequency
        type(basis_map_t) :: fourier
        type(basis_pipeline_t) :: pipeline
        type(basis_linear_regression_t) :: model
        type(fortnum_status_t) :: status
        integer :: i

        x(:, 1) = [-1.2_dp, -0.8_dp, -0.35_dp, 0.1_dp, 0.45_dp, 0.9_dp, &
            1.3_dp, 1.8_dp]
        x_dot(:, 1) = [0.2_dp, -0.1_dp, 0.4_dp, 0.3_dp, -0.2_dp, 0.5_dp, &
            -0.4_dp, 0.1_dp]
        frequency = 0.7_dp
        y(:, 1) = 1.2_dp + 2.0_dp*sin(frequency*x(:, 1)) - &
            0.3_dp*cos(frequency*x(:, 1))
        y(:, 2) = -0.2_dp + 0.4_dp*sin(frequency*x(:, 1)) + &
            1.3_dp*cos(frequency*x(:, 1))
        fourier = make_fourier_basis(1, reshape([frequency], [1, 1]), status)
        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(fourier, status)
        call model%fit(pipeline, x, y, status)
        call model%predict(x, prediction, status)
        if (.not. status_ok(status) .or. .not. model%is_fitted() .or. &
            model%feature_count() /= 2 .or. model%output_count() /= 2 .or. &
            model%parameter_count() /= 7 .or. &
            maxval(abs(prediction - y)) > 1.0e-11_dp) then
            write (error_unit, '(a)') &
                "FAIL [basis linear] fit or direct prediction oracle"
            failures = failures + 1
            return
        end if

        theta = model%parameters()
        theta_dot = [0.13_dp, 0.2_dp, -0.1_dp, 0.05_dp, -0.3_dp, 0.08_dp, &
            0.17_dp]
        call model%predict_jvp(x, theta_dot, x_dot, prediction, prediction_dot, &
            status)
        h = 1.0e-6_dp
        theta_plus = theta + h*theta_dot
        theta_minus = theta - h*theta_dot
        call model%set_parameters(theta_plus, status)
        call model%predict(x + h*x_dot, prediction_plus, status)
        call model%set_parameters(theta_minus, status)
        call model%predict(x - h*x_dot, prediction_minus, status)
        call model%set_parameters(theta, status)
        if (.not. status_ok(status) .or. maxval(abs(prediction_dot - &
            (prediction_plus - prediction_minus)/(2.0_dp*h))) > 2.0e-8_dp) then
            write (error_unit, '(a)') &
                "FAIL [basis linear] chained JVP finite-difference oracle"
            failures = failures + 1
        end if

        do i = 1, 2
            u(:, i) = [0.13_dp, -0.22_dp, 0.31_dp, -0.17_dp, 0.29_dp, &
                -0.11_dp, 0.08_dp, 0.19_dp] + 0.04_dp*real(i, dp)
        end do
        call model%predict_vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*prediction_dot)
        rhs = dot_product(theta_bar, theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 2.0e-9_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [basis linear] chained VJP adjoint identity=", abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_fit_and_products

    subroutine test_polynomial_fit_and_lowering(failures)
        integer, intent(inout) :: failures
        integer, parameter :: n_samples = 9
        real(dp) :: x(n_samples, 1), y(n_samples, 1), prediction(n_samples, 1)
        real(dp) :: features(n_samples, 2)
        type(basis_map_t) :: polynomial
        type(basis_pipeline_t) :: pipeline
        type(basis_linear_regression_t) :: model
        type(fortnum_status_t) :: status
        integer :: i

        do i = 1, n_samples
            x(i, 1) = -1.2_dp + 0.3_dp*real(i - 1, dp)
            y(i, 1) = 1.5_dp - 0.7_dp*x(i, 1) + 0.25_dp*x(i, 1)**2
        end do
        polynomial = make_polynomial_basis(1, 2, status)
        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(polynomial, status, name="powers")
        call model%fit(pipeline, x, y, status, fit_intercept=.true.)
        call model%predict(x, prediction, status)
        call model%transform(x, features, status)
        if (.not. status_ok(status) .or. .not. model%is_fitted() .or. &
            model%feature_count() /= 2 .or. model%parameter_count() /= 3 .or. &
            .not. model%static_lowering_eligible() .or. &
            maxval(abs(features(:, 1) - x(:, 1))) > 1.0e-14_dp .or. &
            maxval(abs(features(:, 2) - x(:, 1)**2)) > 1.0e-14_dp .or. &
            maxval(abs(prediction - y)) > 1.0e-11_dp) then
            write (error_unit, '(a)') &
                "FAIL [basis linear] polynomial fit/lowering oracle"
            failures = failures + 1
        end if
    end subroutine test_polynomial_fit_and_lowering

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 1), y(2, 1), output(2, 1)
        real(dp) :: bad_theta(1)
        type(basis_pipeline_t) :: pipeline
        type(basis_linear_regression_t) :: model
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 1.0_dp]
        y(:, 1) = [0.0_dp, 1.0_dp]
        pipeline = make_basis_pipeline(1, status)
        call model%fit(pipeline, x, y, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [basis linear] empty-pipeline refusal"
            failures = failures + 1
        end if
        bad_theta = 0.0_dp
        call model%set_parameters(bad_theta, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [basis linear] unfitted parameter refusal"
            failures = failures + 1
        end if
        output = 0.0_dp
        call model%predict(x, output, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [basis linear] unfitted prediction refusal"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_basis_linear_regression
