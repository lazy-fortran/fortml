program test_gaussian_process
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, kernel_multiply, make_linear_kernel, &
        make_matern12_kernel, make_matern32_kernel, make_matern52_kernel, &
        make_rbf_kernel, make_white_noise_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_derivative_gaussian_process, only: &
        gp_derivative_regression_t
    use fortnum_linalg, only: dense_solve, LINALG_OK
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures
    failures = 0
    call test_fit_likelihood_and_prediction(failures)
    call test_prediction_products(failures)
    call test_multioutput(failures)
    call test_kernel_input_derivatives(failures)
    call test_matern_derivative_contract(failures)
    call test_mixed_observations(failures)
    call test_matern_observations(failures)
    call test_white_noise_observations(failures)
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
        real(dp) :: gradient_plus(3), gradient_minus(3), parameter_hvp(3)
        real(dp) :: parameter_hvp_fd(3)
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

        call model%hyperparameter_hvp(direction, parameter_hvp, status)
        call model%set_parameters(theta + h*direction, status)
        call model%hyperparameter_gradient(gradient_plus, status)
        call model%set_parameters(theta - h*direction, status)
        call model%hyperparameter_gradient(gradient_minus, status)
        call model%set_parameters(theta, status)
        parameter_hvp_fd = (gradient_plus - gradient_minus)/(2.0_dp*h)
        if (.not. status_ok(status) .or. maxval(abs(parameter_hvp - &
            parameter_hvp_fd)) > 3.0e-5_dp) then
            write (error_unit, '(a)') "FAIL [gp_hvp] gradient finite difference"
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

    subroutine test_kernel_input_derivatives(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: kernel, left, right, product
        type(fortnum_status_t) :: status
        real(dp) :: x1(2), x2(2), gradient_x1(2), gradient_x2(2)
        real(dp) :: kernel_value, mixed_hessian(2, 2), h, h_half
        real(dp) :: fd_x1(2), fd_x2(2), fd_mixed(2, 2)
        real(dp) :: fd_x1_half(2), fd_x2_half(2), fd_mixed_half(2, 2)
        integer :: i, j

        x1 = [0.4_dp, -0.2_dp]
        x2 = [-0.3_dp, 0.7_dp]
        kernel = make_rbf_kernel(2, 1.7_dp, 0.9_dp, status)
        call kernel%input_derivatives( &
            x1, x2, kernel_value, gradient_x1, gradient_x2, mixed_hessian, status)
        h = 1.0e-4_dp
        h_half = 0.5_dp*h
        do i = 1, 2
            fd_x1(i) = central_gradient(kernel, x1, x2, i, h, .true.)
            fd_x2(i) = central_gradient(kernel, x1, x2, i, h, .false.)
            fd_x1_half(i) = central_gradient(kernel, x1, x2, i, h_half, .true.)
            fd_x2_half(i) = central_gradient(kernel, x1, x2, i, h_half, .false.)
            do j = 1, 2
                fd_mixed(i, j) = central_mixed(kernel, x1, x2, i, j, h)
                fd_mixed_half(i, j) = &
                    central_mixed(kernel, x1, x2, i, j, h_half)
            end do
        end do
        if (.not. status_ok(status) .or. &
            maxval(abs(gradient_x1 - fd_x1)) > 2.0e-8_dp .or. &
            maxval(abs(gradient_x2 - fd_x2)) > 2.0e-8_dp .or. &
            maxval(abs(mixed_hessian - fd_mixed)) > 2.0e-7_dp .or. &
            maxval(abs(fd_x1_half - fd_x1)) > 2.0e-8_dp .or. &
            maxval(abs(fd_x2_half - fd_x2)) > 2.0e-8_dp .or. &
            maxval(abs(fd_mixed_half - fd_mixed)) > 2.0e-7_dp) then
            write (error_unit, '(a)') &
                "FAIL [kernel_input_derivatives] finite-difference oracle"
            failures = failures + 1
        end if

        left = make_rbf_kernel(2, 1.7_dp, 0.9_dp, status)
        right = make_linear_kernel(2, 0.8_dp, status)
        product = kernel_multiply(left, right, status)
        call product%input_derivatives( &
            x1, x2, kernel_value, gradient_x1, gradient_x2, mixed_hessian, status)
        do i = 1, 2
            fd_x1(i) = central_gradient(product, x1, x2, i, h, .true.)
            fd_x2(i) = central_gradient(product, x1, x2, i, h, .false.)
            fd_x1_half(i) = central_gradient(product, x1, x2, i, h_half, .true.)
            fd_x2_half(i) = central_gradient(product, x1, x2, i, h_half, .false.)
            do j = 1, 2
                fd_mixed(i, j) = central_mixed(product, x1, x2, i, j, h)
                fd_mixed_half(i, j) = &
                    central_mixed(product, x1, x2, i, j, h_half)
            end do
        end do
        if (.not. status_ok(status) .or. &
            maxval(abs(gradient_x1 - fd_x1)) > 2.0e-7_dp .or. &
            maxval(abs(gradient_x2 - fd_x2)) > 2.0e-7_dp .or. &
            maxval(abs(mixed_hessian - fd_mixed)) > 2.0e-6_dp .or. &
            maxval(abs(fd_x1_half - fd_x1)) > 2.0e-7_dp .or. &
            maxval(abs(fd_x2_half - fd_x2)) > 2.0e-7_dp .or. &
            maxval(abs(fd_mixed_half - fd_mixed)) > 2.0e-6_dp) then
            write (error_unit, '(a)') &
                "FAIL [kernel_product_input_derivatives] finite-difference oracle"
            failures = failures + 1
        end if
    end subroutine test_kernel_input_derivatives

    subroutine test_matern_derivative_contract(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: matern12, matern32, matern52
        type(fortnum_status_t) :: status
        real(dp) :: x1(2), x2(2), coincident(2)
        real(dp) :: value, gradient_x1(2), gradient_x2(2), mixed_hessian(2, 2)
        real(dp) :: expected_curvature

        x1 = [0.4_dp, -0.2_dp]
        x2 = [-0.3_dp, 0.7_dp]
        coincident = [0.2_dp, -0.1_dp]
        matern12 = make_matern12_kernel(2, 1.7_dp, 0.9_dp, status)
        call check_matern_derivatives(matern12, x1, x2, "matern12", failures)
        matern32 = make_matern32_kernel(2, 1.7_dp, 0.9_dp, status)
        call check_matern_derivatives(matern32, x1, x2, "matern32", failures)
        matern52 = make_matern52_kernel(2, 1.7_dp, 0.9_dp, status)
        call check_matern_derivatives(matern52, x1, x2, "matern52", failures)

        call matern12%input_derivatives( &
            coincident, coincident, value, gradient_x1, gradient_x2, mixed_hessian, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [matern12] coincident derivative should refuse"
            failures = failures + 1
        end if

        call matern32%input_derivatives( &
            coincident, coincident, value, gradient_x1, gradient_x2, mixed_hessian, status)
        expected_curvature = 3.0_dp*1.7_dp/(0.9_dp*0.9_dp)
        if (.not. status_ok(status) .or. maxval(abs(gradient_x1)) > 2.0e-14_dp .or. &
            maxval(abs(gradient_x2)) > 2.0e-14_dp .or. &
            maxval(abs(mixed_hessian - reshape([expected_curvature, 0.0_dp, &
            0.0_dp, expected_curvature], [2, 2]))) > 2.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [matern32] coincident derivative limit"
            failures = failures + 1
        end if

        call matern52%input_derivatives( &
            coincident, coincident, value, gradient_x1, gradient_x2, mixed_hessian, status)
        expected_curvature = 5.0_dp*1.7_dp/(3.0_dp*0.9_dp*0.9_dp)
        if (.not. status_ok(status) .or. maxval(abs(gradient_x1)) > 2.0e-14_dp .or. &
            maxval(abs(gradient_x2)) > 2.0e-14_dp .or. &
            maxval(abs(mixed_hessian - reshape([expected_curvature, 0.0_dp, &
            0.0_dp, expected_curvature], [2, 2]))) > 2.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [matern52] coincident derivative limit"
            failures = failures + 1
        end if
    end subroutine test_matern_derivative_contract

    subroutine check_matern_derivatives(kernel, x1, x2, label, failures)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:)
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: value, gradient_x1(size(x1)), gradient_x2(size(x2))
        real(dp) :: mixed_hessian(size(x1), size(x2))
        real(dp) :: fd_x1(size(x1)), fd_x2(size(x2)), fd_mixed(size(x1), size(x2))
        real(dp) :: fd_x1_half(size(x1)), fd_x2_half(size(x2))
        real(dp) :: fd_mixed_half(size(x1), size(x2))
        real(dp) :: h, h_half
        integer :: i, j

        call kernel%input_derivatives( &
            x1, x2, value, gradient_x1, gradient_x2, mixed_hessian, status)
        h = 1.0e-4_dp
        h_half = 0.5_dp*h
        do i = 1, size(x1)
            fd_x1(i) = central_gradient(kernel, x1, x2, i, h, .true.)
            fd_x2(i) = central_gradient(kernel, x1, x2, i, h, .false.)
            fd_x1_half(i) = central_gradient(kernel, x1, x2, i, h_half, .true.)
            fd_x2_half(i) = central_gradient(kernel, x1, x2, i, h_half, .false.)
            do j = 1, size(x2)
                fd_mixed(i, j) = central_mixed(kernel, x1, x2, i, j, h)
                fd_mixed_half(i, j) = central_mixed(kernel, x1, x2, i, j, h_half)
            end do
        end do
        if (.not. status_ok(status) .or. maxval(abs(gradient_x1 - fd_x1)) > 3.0e-8_dp .or. &
            maxval(abs(gradient_x2 - fd_x2)) > 3.0e-8_dp .or. &
            maxval(abs(mixed_hessian - fd_mixed)) > 3.0e-7_dp .or. &
            maxval(abs(fd_x1_half - fd_x1)) > 3.0e-8_dp .or. &
            maxval(abs(fd_x2_half - fd_x2)) > 3.0e-8_dp .or. &
            maxval(abs(fd_mixed_half - fd_mixed)) > 3.0e-7_dp) then
            write (error_unit, '(2a)') "FAIL [matern derivatives] ", trim(label)
            failures = failures + 1
        end if
    end subroutine check_matern_derivatives

    subroutine test_mixed_observations(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x_train(3, 1), x_query(3, 1)
        real(dp) :: y_train(3, 2), mean(3, 2), variance(3)
        real(dp) :: expected_mean(3, 2), expected_variance(3)
        real(dp) :: covariance(3, 3), cross(3, 3), work(3, 3)
        real(dp) :: alpha(3, 2)
        integer :: components(3), query_components(3), info
        integer :: i, j

        x_train(:, 1) = [0.0_dp, 0.5_dp, 1.0_dp]
        components = [0, 1, 0]
        y_train(:, 1) = [1.0_dp, -0.2_dp, 0.8_dp]
        y_train(:, 2) = [0.4_dp, 1.3_dp, -0.7_dp]
        x_query(:, 1) = [0.25_dp, 0.75_dp, 1.25_dp]
        query_components = [0, 1, 0]
        kernel = make_rbf_kernel(1, 1.3_dp, 0.7_dp, status)
        call model%fit( &
            x_train, components, y_train, kernel, 0.05_dp, status, &
            jitter=1.0e-10_dp)
        call model%predict( &
            x_query, query_components, mean, variance, status)

        do j = 1, 3
            do i = 1, 3
                covariance(i, j) = mixed_rbf_covariance( &
                    x_train(i, 1), components(i), x_train(j, 1), components(j), &
                    1.3_dp, 0.7_dp)
                cross(i, j) = mixed_rbf_covariance( &
                    x_train(i, 1), components(i), x_query(j, 1), &
                    query_components(j), 1.3_dp, 0.7_dp)
            end do
        end do
        do i = 1, 3
            covariance(i, i) = covariance(i, i) + 0.05_dp + 1.0e-10_dp
        end do
        call dense_solve(covariance, y_train, alpha, info)
        call dense_solve(covariance, cross, work, info)
        do j = 1, 3
            expected_mean(j, 1) = dot_product(cross(:, j), alpha(:, 1))
            expected_mean(j, 2) = dot_product(cross(:, j), alpha(:, 2))
            expected_variance(j) = mixed_rbf_covariance( &
                x_query(j, 1), query_components(j), x_query(j, 1), &
                query_components(j), 1.3_dp, 0.7_dp) - &
                dot_product(cross(:, j), work(:, j))
        end do
        if (.not. status_ok(status) .or. info /= LINALG_OK .or. &
            model%observation_count() /= 3 .or. &
            maxval(abs(mean - expected_mean)) > 3.0e-10_dp .or. &
            maxval(abs(variance - expected_variance)) > 3.0e-10_dp) then
            write (error_unit, '(a)') &
                "FAIL [mixed_gp] independent derivative covariance oracle"
            failures = failures + 1
        end if
    end subroutine test_mixed_observations

    subroutine test_matern_observations(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x_train(3, 1), x_query(3, 1)
        real(dp) :: y_train(3, 2), mean(3, 2), variance(3)
        real(dp) :: expected_mean(3, 2), expected_variance(3)
        real(dp) :: covariance(3, 3), cross(3, 3), work(3, 3)
        real(dp) :: alpha(3, 2)
        integer :: components(3), query_components(3), info
        integer :: i, j

        x_train(:, 1) = [0.0_dp, 0.5_dp, 1.0_dp]
        components = [0, 1, 0]
        y_train(:, 1) = [1.0_dp, -0.2_dp, 0.8_dp]
        y_train(:, 2) = [0.4_dp, 1.3_dp, -0.7_dp]
        x_query(:, 1) = [0.25_dp, 0.75_dp, 1.25_dp]
        query_components = [0, 1, 0]
        kernel = make_matern32_kernel(1, 1.3_dp, 0.7_dp, status)
        call model%fit( &
            x_train, components, y_train, kernel, 0.05_dp, status, jitter=1.0e-10_dp)
        call model%predict(x_query, query_components, mean, variance, status)

        do j = 1, 3
            do i = 1, 3
                covariance(i, j) = mixed_matern32_covariance( &
                    x_train(i, 1), components(i), x_train(j, 1), components(j), &
                    1.3_dp, 0.7_dp)
                cross(i, j) = mixed_matern32_covariance( &
                    x_train(i, 1), components(i), x_query(j, 1), &
                    query_components(j), 1.3_dp, 0.7_dp)
            end do
        end do
        do i = 1, 3
            covariance(i, i) = covariance(i, i) + 0.05_dp + 1.0e-10_dp
        end do
        call dense_solve(covariance, y_train, alpha, info)
        call dense_solve(covariance, cross, work, info)
        do j = 1, 3
            expected_mean(j, 1) = dot_product(cross(:, j), alpha(:, 1))
            expected_mean(j, 2) = dot_product(cross(:, j), alpha(:, 2))
            expected_variance(j) = mixed_matern32_covariance( &
                x_query(j, 1), query_components(j), x_query(j, 1), &
                query_components(j), 1.3_dp, 0.7_dp) - &
                dot_product(cross(:, j), work(:, j))
        end do
        if (.not. status_ok(status) .or. info /= LINALG_OK .or. &
            maxval(abs(mean - expected_mean)) > 3.0e-10_dp .or. &
            maxval(abs(variance - expected_variance)) > 3.0e-10_dp) then
            write (error_unit, '(a)') &
                "FAIL [matern32_gp] independent derivative covariance oracle"
            failures = failures + 1
        end if
    end subroutine test_matern_observations

    subroutine test_white_noise_observations(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x_train(3, 1), x_query(3, 1), y_train(3, 1)
        real(dp) :: mean(3, 1), variance(3), expected_mean(3, 1), expected_variance(3)
        real(dp) :: denominator

        x_train(:, 1) = [0.0_dp, 0.5_dp, 1.0_dp]
        y_train(:, 1) = [1.0_dp, 2.0_dp, 3.0_dp]
        x_query(:, 1) = [0.0_dp, 0.25_dp, 1.0_dp]
        kernel = make_white_noise_kernel(1, 0.2_dp, status)
        call model%fit( &
            x_train, [0, 0, 0], y_train, kernel, 0.05_dp, status, jitter=1.0e-10_dp)
        call model%predict(x_query, [0, 0, 0], mean, variance, status)
        denominator = 0.2_dp + 0.05_dp + 1.0e-10_dp
        expected_mean(:, 1) = [0.2_dp/denominator, 0.0_dp, 0.6_dp/denominator]
        expected_variance = [0.2_dp - 0.04_dp/denominator, 0.2_dp, &
            0.2_dp - 0.04_dp/denominator]
        if (.not. status_ok(status) .or. maxval(abs(mean - expected_mean)) > 3.0e-10_dp .or. &
            maxval(abs(variance - expected_variance)) > 3.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [white_noise_gp] function observations"
            failures = failures + 1
        end if

        call model%fit( &
            x_train, [1, 0, 0], y_train, kernel, 0.05_dp, status, jitter=1.0e-10_dp)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [white_noise_gp] derivative observations should refuse"
            failures = failures + 1
        end if
    end subroutine test_white_noise_observations

    real(dp) function central_gradient(kernel, x1, x2, component, h, first) &
            result(value)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:), h
        integer, intent(in) :: component
        logical, intent(in) :: first
        real(dp) :: plus, minus
        real(dp) :: shifted_x1(size(x1)), shifted_x2(size(x2))

        shifted_x1 = x1
        shifted_x2 = x2
        if (first) then
            shifted_x1(component) = shifted_x1(component) + h
            plus = kernel%value(shifted_x1, shifted_x2)
            shifted_x1(component) = shifted_x1(component) - 2.0_dp*h
            minus = kernel%value(shifted_x1, shifted_x2)
        else
            shifted_x2(component) = shifted_x2(component) + h
            plus = kernel%value(shifted_x1, shifted_x2)
            shifted_x2(component) = shifted_x2(component) - 2.0_dp*h
            minus = kernel%value(shifted_x1, shifted_x2)
        end if
        value = (plus - minus)/(2.0_dp*h)
    end function central_gradient

    real(dp) function central_mixed(kernel, x1, x2, component1, component2, h) &
            result(value)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:), h
        integer, intent(in) :: component1, component2
        real(dp) :: x1_plus(size(x1)), x1_minus(size(x1))
        real(dp) :: x2_plus(size(x2)), x2_minus(size(x2))

        x1_plus = x1
        x1_minus = x1
        x2_plus = x2
        x2_minus = x2
        x1_plus(component1) = x1_plus(component1) + h
        x1_minus(component1) = x1_minus(component1) - h
        x2_plus(component2) = x2_plus(component2) + h
        x2_minus(component2) = x2_minus(component2) - h
        value = (kernel%value(x1_plus, x2_plus) - &
            kernel%value(x1_plus, x2_minus) - &
            kernel%value(x1_minus, x2_plus) + &
            kernel%value(x1_minus, x2_minus))/(4.0_dp*h*h)
    end function central_mixed

    real(dp) function mixed_rbf_covariance(x1, component1, x2, component2, &
            variance, lengthscale) result(value)
        real(dp), intent(in) :: x1, x2, variance, lengthscale
        integer, intent(in) :: component1, component2
        real(dp) :: delta, kernel_value, inverse_length_squared

        delta = x1 - x2
        inverse_length_squared = 1.0_dp/(lengthscale*lengthscale)
        kernel_value = variance*exp(-0.5_dp*delta*delta*inverse_length_squared)
        if (component1 == 0 .and. component2 == 0) then
            value = kernel_value
        else if (component1 > 0 .and. component2 == 0) then
            value = -kernel_value*delta*inverse_length_squared
        else if (component1 == 0 .and. component2 > 0) then
            value = kernel_value*delta*inverse_length_squared
        else
            value = kernel_value*(inverse_length_squared - &
                delta*delta*inverse_length_squared*inverse_length_squared)
        end if
    end function mixed_rbf_covariance

    real(dp) function mixed_matern32_covariance(x1, component1, x2, component2, &
            variance, lengthscale) result(value)
        real(dp), intent(in) :: x1, x2, variance, lengthscale
        integer, intent(in) :: component1, component2
        real(dp) :: delta, distance, a, exponential

        delta = x1 - x2
        distance = abs(delta)
        a = sqrt(3.0_dp)/lengthscale
        exponential = exp(-a*distance)
        if (component1 == 0 .and. component2 == 0) then
            value = variance*(1.0_dp + a*distance)*exponential
        else if (component1 > 0 .and. component2 == 0) then
            value = -variance*a*a*delta*exponential
        else if (component1 == 0 .and. component2 > 0) then
            value = variance*a*a*delta*exponential
        else
            value = variance*a*a*(1.0_dp - a*distance)*exponential
        end if
    end function mixed_matern32_covariance

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
