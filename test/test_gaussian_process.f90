program test_gaussian_process
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures
    failures = 0
    call test_fit_likelihood_and_prediction(failures)
    call test_prediction_products(failures)
    call test_multioutput(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " GP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_multioutput(failures)
        integer, intent(inout) :: failures
        type(gp_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x_train(3, 1), y_train(3, 2), x_test(2, 1)
        real(dp) :: mean(2, 2), variance(2)

        x_train(:, 1) = [0.0_dp, 0.5_dp, 1.0_dp]
        y_train(:, 1) = [1.0_dp, 2.0_dp, 1.5_dp]
        y_train(:, 2) = 2.0_dp*y_train(:, 1)
        x_test(:, 1) = [0.25_dp, 0.75_dp]
        kernel = make_rbf_kernel(1, 1.5_dp, 0.8_dp, status)
        call model%fit(x_train, y_train, kernel, 0.1_dp, status)
        call model%predict(x_test, mean, variance, status)
        if (.not. status_ok(status) .or. maxval(abs(mean(:, 2) - &
            2.0_dp*mean(:, 1))) > 3.0e-10_dp .or. model%n_outputs /= 2) then
            write (error_unit, '(a)') "FAIL [multioutput] shared GP solve"
            failures = failures + 1
        end if
    end subroutine test_multioutput

    subroutine test_fit_likelihood_and_prediction(failures)
        integer, intent(inout) :: failures
        type(gp_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x_train(2, 1), y_train(2, 1), x_test(2, 1)
        real(dp) :: mean(2, 1), variance(2), expected_mean(2, 1), expected_variance(2)
        real(dp) :: lml, expected_lml, theta(3), gradient(3), direction(3)
        real(dp) :: lml_plus, lml_minus, finite_difference, value_dot, h

        x_train(:, 1) = [0.0_dp, 1.0_dp]
        y_train(:, 1) = [1.0_dp, 2.0_dp]
        x_test(:, 1) = [0.25_dp, 0.75_dp]
        kernel = make_rbf_kernel(1, 1.5_dp, 0.8_dp, status)
        call model%fit(x_train, y_train, kernel, 0.1_dp, status, jitter=1.0e-10_dp)
        call model%predict(x_test, mean, variance, status)
        theta = model%parameters()
        call reference_prediction(theta, x_train, y_train, x_test, expected_mean, &
            expected_variance)
        call model%log_marginal_likelihood(lml, status)
        expected_lml = reference_lml(theta, y_train)
        if (.not. status_ok(status) .or. maxval(abs(mean - expected_mean)) > 3.0e-10_dp .or. &
            maxval(abs(variance - expected_variance)) > 3.0e-10_dp .or. &
            abs(lml - expected_lml) > 3.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [gp] value, variance, or likelihood"
            failures = failures + 1
        end if
        direction = [0.17_dp, -0.23_dp, 0.11_dp]
        call model%hyperparameter_gradient(gradient, status)
        h = 1.0e-6_dp
        lml_plus = reference_lml(theta + h*direction, y_train)
        lml_minus = reference_lml(theta - h*direction, y_train)
        finite_difference = (lml_plus - lml_minus)/(2.0_dp*h)
        call model%log_marginal_likelihood_jvp(direction, value_dot, status)
        if (.not. status_ok(status) .or. abs(finite_difference - dot_product(gradient, &
            direction)) > 4.0e-8_dp .or. abs(value_dot - finite_difference) > 4.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [gp_gradient] finite difference"
            failures = failures + 1
        end if
    end subroutine test_fit_likelihood_and_prediction

    subroutine test_prediction_products(failures)
        integer, intent(inout) :: failures
        type(gp_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x_train(2, 1), y_train(2, 1), x_test(3, 1)
        real(dp) :: direction(3), mean(3, 1), mean_dot(3, 1), variance(3)
        real(dp) :: variance_dot(3), mean_plus(3, 1), mean_minus(3, 1)
        real(dp) :: variance_plus(3), variance_minus(3), fd_mean(3, 1), fd_variance(3)
        real(dp) :: mean_bar(3, 1), variance_bar(3), parameter_bar(3)
        real(dp) :: lhs, rhs, h

        x_train(:, 1) = [0.0_dp, 1.0_dp]
        y_train(:, 1) = [1.0_dp, 2.0_dp]
        x_test(:, 1) = [0.25_dp, 0.75_dp, 1.25_dp]
        kernel = make_rbf_kernel(1, 1.5_dp, 0.8_dp, status)
        call model%fit(x_train, y_train, kernel, 0.1_dp, status, jitter=1.0e-10_dp)
        direction = [0.17_dp, -0.23_dp, 0.11_dp]
        call model%predict_jvp(x_test, direction, mean, mean_dot, variance, &
            variance_dot, status)
        h = 1.0e-6_dp
        call reference_prediction(model%parameters() + h*direction, x_train, y_train, &
            x_test, mean_plus, variance_plus)
        call reference_prediction(model%parameters() - h*direction, x_train, y_train, &
            x_test, mean_minus, variance_minus)
        fd_mean = (mean_plus - mean_minus)/(2.0_dp*h)
        fd_variance = (variance_plus - variance_minus)/(2.0_dp*h)
        if (.not. status_ok(status) .or. maxval(abs(mean_dot - fd_mean)) > 4.0e-8_dp .or. &
            maxval(abs(variance_dot - fd_variance)) > 4.0e-8_dp) then
            write (error_unit, '(a)') "FAIL [gp_jvp] finite difference"
            failures = failures + 1
        end if

        mean_bar(:, 1) = [0.4_dp, -0.2_dp, 0.1_dp]
        variance_bar = [0.3_dp, -0.5_dp, 0.2_dp]
        call model%predict_vjp(x_test, mean_bar, variance_bar, parameter_bar, status)
        lhs = sum(mean_bar*mean_dot) + sum(variance_bar*variance_dot)
        rhs = sum(parameter_bar*direction)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 5.0e-10_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [gp_vjp] adjoint identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_prediction_products

    real(dp) function reference_lml(parameters, y) result(value)
        real(dp), intent(in) :: parameters(:), y(:, :)
        real(dp) :: variance, lengthscale, noise, cross, determinant
        real(dp) :: alpha_1, alpha_2

        variance = exp(parameters(1))
        lengthscale = exp(parameters(2))
        noise = exp(parameters(3))
        cross = variance*exp(-0.5_dp/(lengthscale*lengthscale))
        determinant = (variance + noise + 1.0e-10_dp)**2 - cross**2
        alpha_1 = ((variance + noise + 1.0e-10_dp)*y(1, 1) - cross*y(2, 1)) / &
            determinant
        alpha_2 = ((variance + noise + 1.0e-10_dp)*y(2, 1) - cross*y(1, 1)) / &
            determinant
        value = -0.5_dp*(y(1, 1)*alpha_1 + y(2, 1)*alpha_2) - &
            0.5_dp*log(determinant) - log(2.0_dp*acos(-1.0_dp))
    end function reference_lml

    subroutine reference_prediction(parameters, x_train, y_train, x_test, mean, variance)
        real(dp), intent(in) :: parameters(:), x_train(:, :), y_train(:, :), x_test(:, :)
        real(dp), intent(out) :: mean(:, :), variance(:)
        real(dp) :: signal_variance, lengthscale, noise, covariance, determinant
        real(dp) :: alpha(2), cross(2), inverse_cross(2), prior
        integer :: i

        signal_variance = exp(parameters(1))
        lengthscale = exp(parameters(2))
        noise = exp(parameters(3))
        covariance = signal_variance*exp(-0.5_dp/(lengthscale*lengthscale))
        determinant = (signal_variance + noise + 1.0e-10_dp)**2 - covariance**2
        alpha(1) = ((signal_variance + noise + 1.0e-10_dp)*y_train(1, 1) - &
            covariance*y_train(2, 1))/determinant
        alpha(2) = ((signal_variance + noise + 1.0e-10_dp)*y_train(2, 1) - &
            covariance*y_train(1, 1))/determinant
        do i = 1, size(x_test, 1)
            cross(1) = signal_variance*exp(-0.5_dp*(x_train(1, 1) - x_test(i, 1))**2/ &
                (lengthscale*lengthscale))
            cross(2) = signal_variance*exp(-0.5_dp*(x_train(2, 1) - x_test(i, 1))**2/ &
                (lengthscale*lengthscale))
            mean(i, 1) = cross(1)*alpha(1) + cross(2)*alpha(2)
            inverse_cross(1) = ((signal_variance + noise + 1.0e-10_dp)*cross(1) - &
                covariance*cross(2))/determinant
            inverse_cross(2) = ((signal_variance + noise + 1.0e-10_dp)*cross(2) - &
                covariance*cross(1))/determinant
            prior = signal_variance
            variance(i) = prior - cross(1)*inverse_cross(1) - &
                cross(2)*inverse_cross(2)
        end do
    end subroutine reference_prediction

end program test_gaussian_process
