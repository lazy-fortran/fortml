program test_derivative_gp_products
    !! Independent likelihood/product checks for mixed value/derivative GPs.
    !! The oracle assembles the covariance from the RBF partials directly;
    !! it does not call the production derivative-covariance helper.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernel_formula, only: kernel_formula_t
    use fortml_kernels, only: kernel_t, kernel_multiply, make_linear_kernel, &
        make_matern32_kernel, make_matern52_kernel, &
        make_periodic_kernel, make_rational_quadratic_kernel, &
        make_rbf_kernel, make_user_kernel
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_likelihood_products(failures)
    call test_matern_parameter_products(failures)
    call test_periodic_rational_parameter_products(failures)
    call test_product_parameter_products(failures)
    call test_prediction_products(failures)
    call test_joint_covariance(failures)
    call test_query_input_products(failures)
    call test_periodic_rational_query_products(failures)
    call test_user_formula_observations(failures)
    call test_parameter_guards(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " derivative GP product test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: derivative GP likelihood products"

contains

    subroutine test_likelihood_products(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), theta(3), direction(3)
        real(dp) :: gradient(3), hvp(3), gradient_plus(3), gradient_minus(3)
        real(dp) :: expected, value, value_dot, fd_gradient(3), fd_hvp(3)
        real(dp) :: h, h_hvp
        integer :: i

        x(:, 1) = [0.0_dp, 0.45_dp, 1.05_dp]
        y(:, 1) = [1.2_dp, -0.3_dp, 0.8_dp]
        kernel = make_rbf_kernel(1, 1.4_dp, 0.75_dp, status)
        call model%fit(x, [0, 1, 0], y, kernel, 0.07_dp, status, &
            jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [derivative GP] fit"
            failures = failures + 1
            return
        end if

        theta = model%parameters()
        call kernel%set_parameters([log(2.1_dp), log(0.55_dp)], status)
        if (.not. status_ok(status) .or. maxval(abs(theta(:2) - &
            [log(1.4_dp), log(0.75_dp)])) > 1.0e-14_dp) then
            write (error_unit, '(a)') &
                "FAIL [derivative GP] fit did not own its kernel tree"
            failures = failures + 1
        end if

        theta = model%parameters()
        call model%log_marginal_likelihood(value, status)
        expected = oracle_lml(theta, x, [0, 1, 0], y, 0.07_dp, 1.0e-10_dp)
        if (.not. status_ok(status) .or. abs(value - expected) > 2.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [derivative GP] independent likelihood oracle ", abs(value - expected)
            failures = failures + 1
        end if

        call model%hyperparameter_gradient(gradient, status)
        h = 2.0e-5_dp
        do i = 1, 3
            fd_gradient(i) = (oracle_lml(theta + h*unit_vector(3, i), x, &
                [0, 1, 0], y, 0.07_dp, 1.0e-10_dp) - &
                oracle_lml(theta - h*unit_vector(3, i), x, [0, 1, 0], y, &
                0.07_dp, 1.0e-10_dp))/(2.0_dp*h)
        end do
        if (.not. status_ok(status) .or. maxval(abs(gradient - fd_gradient)) > 2.0e-7_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [derivative GP] gradient oracle ", maxval(abs(gradient - fd_gradient))
            failures = failures + 1
        end if

        direction = [0.21_dp, -0.13_dp, 0.17_dp]
        call model%log_marginal_likelihood_jvp(direction, value_dot, status)
        if (.not. status_ok(status) .or. abs(value_dot - dot_product(gradient, direction)) > &
            3.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [derivative GP] likelihood JVP"
            failures = failures + 1
        end if

        call model%hyperparameter_hvp(direction, hvp, status)
        h_hvp = 2.0e-4_dp
        gradient_plus = oracle_gradient(theta + h_hvp*direction, x, [0, 1, 0], y, &
            0.07_dp, 1.0e-10_dp)
        gradient_minus = oracle_gradient(theta - h_hvp*direction, x, [0, 1, 0], y, &
            0.07_dp, 1.0e-10_dp)
        fd_hvp = (gradient_plus - gradient_minus)/(2.0_dp*h_hvp)
        if (.not. status_ok(status) .or. maxval(abs(hvp - fd_hvp)) > 2.0e-4_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [derivative GP] HVP directional oracle ", maxval(abs(hvp - fd_hvp))
            failures = failures + 1
        end if
    end subroutine test_likelihood_products

    subroutine test_matern_parameter_products(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), theta(3), gradient(3), finite_gradient(3)
        real(dp) :: h
        integer :: i

        x(:, 1) = [0.0_dp, 0.4_dp, 1.0_dp]
        y(:, 1) = [0.5_dp, -0.3_dp, 0.9_dp]
        kernel = make_matern32_kernel(1, 1.4_dp, 0.75_dp, status)
        call model%fit(x, [0, 1, 0], y, kernel, 0.06_dp, status, &
            jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [Matern32 derivative GP] fit"
            failures = failures + 1
            return
        end if
        theta = model%parameters()
        call model%hyperparameter_gradient(gradient, status)
        h = 2.0e-5_dp
        do i = 1, 3
            finite_gradient(i) = (oracle_lml_matern32(theta + h*unit_vector(3, i), &
                x, [0, 1, 0], y, 0.06_dp, 1.0e-10_dp) - &
                oracle_lml_matern32(theta - h*unit_vector(3, i), x, [0, 1, 0], y, &
                0.06_dp, 1.0e-10_dp))/(2.0_dp*h)
        end do
        if (.not. status_ok(status) .or. maxval(abs(gradient - finite_gradient)) > &
            3.0e-6_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [Matern32 derivative GP] gradient oracle ", &
                maxval(abs(gradient - finite_gradient))
            failures = failures + 1
        end if
    end subroutine test_matern_parameter_products

    subroutine test_periodic_rational_parameter_products(failures)
        integer, intent(inout) :: failures
        type(kernel_t) :: periodic, rational_quadratic
        type(fortnum_status_t) :: status

        periodic = make_periodic_kernel(1, 1.2_dp, 0.75_dp, 1.8_dp, status)
        call check_kernel_parameter_gradient(periodic, "periodic", failures)
        rational_quadratic = make_rational_quadratic_kernel(1, 1.2_dp, 0.75_dp, &
            1.6_dp, status)
        call check_kernel_parameter_gradient(rational_quadratic, "rational-quadratic", failures)
    end subroutine test_periodic_rational_parameter_products

    subroutine check_kernel_parameter_gradient(kernel, name, failures)
        type(kernel_t), intent(in) :: kernel
        character(len=*), intent(in) :: name
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), theta(4), gradient(4), finite_gradient(4)
        real(dp) :: plus, minus, h
        integer :: i

        x(:, 1) = [0.0_dp, 0.42_dp, 1.03_dp]
        y(:, 1) = [0.8_dp, -0.2_dp, 0.6_dp]
        call model%fit(x, [0, 1, 0], y, kernel, 0.08_dp, status, jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a,a)') "FAIL [", trim(name)//" parameter fit]"
            failures = failures + 1
            return
        end if
        theta = model%parameters()
        call model%hyperparameter_gradient(gradient, status)
        h = 2.0e-5_dp
        do i = 1, size(theta)
            call model%set_parameters(theta + h*unit_vector(size(theta), i), status)
            call model%log_marginal_likelihood(plus, status)
            call model%set_parameters(theta - h*unit_vector(size(theta), i), status)
            call model%log_marginal_likelihood(minus, status)
            finite_gradient(i) = (plus - minus)/(2.0_dp*h)
        end do
        call model%set_parameters(theta, status)
        if (.not. status_ok(status) .or. maxval(abs(gradient - finite_gradient)) > 3.0e-5_dp) then
            write (error_unit, '(a,a,es12.4)') "FAIL [", trim(name)//" parameter gradient] ", &
                maxval(abs(gradient - finite_gradient))
            failures = failures + 1
        end if
    end subroutine check_kernel_parameter_gradient

    subroutine test_product_parameter_products(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: left, right, kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), theta(3), gradient(3), plus, minus
        real(dp) :: finite_gradient(3), h, value
        integer :: i

        x(:, 1) = [0.1_dp, 0.45_dp, 0.95_dp]
        y(:, 1) = [0.8_dp, -0.4_dp, 0.6_dp]
        left = make_rbf_kernel(1, 1.3_dp, 0.8_dp, status)
        right = make_linear_kernel(1, 0.7_dp, status)
        kernel = kernel_multiply(left, right, status)
        call model%fit(x, [0, 1, 0], y, kernel, 0.08_dp, status, &
            jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [product derivative GP] fit"
            failures = failures + 1
            return
        end if
        theta = model%parameters()
        call model%hyperparameter_gradient(gradient, status)
        h = 2.0e-5_dp
        do i = 1, 3
            call model%set_parameters(theta + h*unit_vector(3, i), status)
            call model%log_marginal_likelihood(plus, status)
            call model%set_parameters(theta - h*unit_vector(3, i), status)
            call model%log_marginal_likelihood(minus, status)
            finite_gradient(i) = (plus - minus)/(2.0_dp*h)
        end do
        call model%set_parameters(theta, status)
        call model%log_marginal_likelihood(value, status)
        if (.not. status_ok(status) .or. maxval(abs(gradient - finite_gradient)) > 2.0e-6_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [product derivative GP] parameter gradient ", &
                maxval(abs(gradient - finite_gradient))
            failures = failures + 1
        end if
    end subroutine test_product_parameter_products

    subroutine test_prediction_products(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), x_test(2, 1), theta(3), direction(3)
        integer :: components(3), test_components(2), i
        real(dp) :: mean(2, 1), mean_dot(2, 1), variance(2), variance_dot(2)
        real(dp) :: mean_plus(2, 1), mean_minus(2, 1), variance_plus(2), variance_minus(2)
        real(dp) :: mean_base(2, 1), variance_base(2), zero_direction(3)
        real(dp) :: mean_bar(2, 1), variance_bar(2), parameter_bar(3), fd_bar(3)
        real(dp) :: objective_plus, objective_minus, h

        x(:, 1) = [0.0_dp, 0.45_dp, 1.05_dp]
        y(:, 1) = [1.2_dp, -0.3_dp, 0.8_dp]
        components = [0, 1, 0]
        x_test(:, 1) = [0.25_dp, 0.8_dp]
        test_components = [1, 0]
        kernel = make_rbf_kernel(1, 1.4_dp, 0.75_dp, status)
        call model%fit(x, components, y, kernel, 0.07_dp, status, jitter=1.0e-10_dp)
        theta = model%parameters()
        zero_direction = 0.0_dp
        call model%predict(x_test, test_components, mean_base, variance_base, status)
        call model%predict_jvp(x_test, test_components, zero_direction, mean, mean_dot, &
            variance, variance_dot, status)
        if (.not. status_ok(status) .or. maxval(abs(mean - mean_base)) > 1.0e-12_dp .or. &
            maxval(abs(variance - variance_base)) > 1.0e-12_dp .or. &
            maxval(abs(mean_dot)) > 1.0e-12_dp .or. maxval(abs(variance_dot)) > 1.0e-12_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [derivative GP prediction zero-direction] variance=", &
                maxval(abs(variance - variance_base)), maxval(abs(variance_dot))
            failures = failures + 1
        end if
        direction = [0.21_dp, -0.13_dp, 0.17_dp]
        call model%predict_jvp(x_test, test_components, direction, mean, mean_dot, &
            variance, variance_dot, status)
        h = 2.0e-6_dp
        call oracle_predict(theta + h*direction, x, components, y, x_test, &
            test_components, 0.07_dp, 1.0e-10_dp, mean_plus, variance_plus)
        call oracle_predict(theta - h*direction, x, components, y, x_test, &
            test_components, 0.07_dp, 1.0e-10_dp, mean_minus, variance_minus)
        if (.not. status_ok(status) .or. maxval(abs(mean_dot - (mean_plus - mean_minus)/ &
            (2.0_dp*h))) > 4.0e-7_dp .or. maxval(abs(variance_dot - &
            (variance_plus - variance_minus)/(2.0_dp*h))) > 4.0e-7_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [derivative GP prediction_jvp] independent finite difference ", &
                maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h))), &
                maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h)))
            write (error_unit, '(a,6es12.4)') "  variance/pm/jvp/fd=", variance(1), &
                variance_plus(1), variance_minus(1), variance_dot(1), &
                (variance_plus(1) - variance_minus(1))/(2.0_dp*h), &
                variance_dot(1) - (variance_plus(1) - variance_minus(1))/(2.0_dp*h)
            failures = failures + 1
        end if

        mean_bar(:, 1) = [0.35_dp, -0.2_dp]
        variance_bar = [0.25_dp, -0.15_dp]
        call model%predict_vjp(x_test, test_components, mean_bar, variance_bar, &
            parameter_bar, status)
        do i = 1, size(theta)
            fd_bar(i) = (oracle_prediction_objective(theta + h*unit_vector(3, i), x, &
                components, y, x_test, test_components, 0.07_dp, 1.0e-10_dp, &
                mean_bar, variance_bar) - oracle_prediction_objective(theta - &
                h*unit_vector(3, i), x, components, y, x_test, test_components, &
                0.07_dp, 1.0e-10_dp, mean_bar, variance_bar))/(2.0_dp*h)
        end do
        if (.not. status_ok(status) .or. maxval(abs(parameter_bar - fd_bar)) > 6.0e-7_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [derivative GP prediction_vjp] independent finite difference ", &
                maxval(abs(parameter_bar - fd_bar))
            failures = failures + 1
        end if
    end subroutine test_prediction_products

    subroutine test_joint_covariance(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), x_test(2, 1), covariance(2, 2), expected(2, 2)

        x(:, 1) = [0.0_dp, 0.45_dp, 1.05_dp]
        y(:, 1) = [1.2_dp, -0.3_dp, 0.8_dp]
        x_test(:, 1) = [0.25_dp, 0.8_dp]
        kernel = make_rbf_kernel(1, 1.4_dp, 0.75_dp, status)
        call model%fit(x, [0, 1, 0], y, kernel, 0.07_dp, status, jitter=1.0e-10_dp)
        call model%joint_covariance(x_test, [1, 0], covariance, status)
        call oracle_joint_covariance(model%parameters(), x, [0, 1, 0], y, x_test, [1, 0], &
            0.07_dp, 1.0e-10_dp, expected)
        if (.not. status_ok(status) .or. maxval(abs(covariance - expected)) > 2.0e-11_dp .or. &
            maxval(abs(covariance - transpose(covariance))) > 2.0e-14_dp) then
            write (error_unit, '(a,2es12.4)') "FAIL [derivative GP joint covariance] ", &
                maxval(abs(covariance - expected)), maxval(abs(covariance - transpose(covariance)))
            failures = failures + 1
        end if
    end subroutine test_joint_covariance

    subroutine test_query_input_products(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), x_test(2, 1), direction(2, 1)
        real(dp) :: mean(2, 1), mean_dot(2, 1), variance(2), variance_dot(2)
        real(dp) :: mean_plus(2, 1), mean_minus(2, 1), variance_plus(2), variance_minus(2)
        real(dp) :: mean_bar(2, 1), variance_bar(2), x_bar(2, 1), fd_bar(2, 1)
        real(dp) :: objective_plus, objective_minus, adjoint_left, adjoint_right, h
        integer :: components(3), test_components(2), i

        x(:, 1) = [0.0_dp, 0.45_dp, 1.05_dp]
        y(:, 1) = [1.2_dp, -0.3_dp, 0.8_dp]
        components = [0, 1, 0]
        x_test(:, 1) = [0.25_dp, 0.8_dp]
        test_components = [1, 0]
        direction(:, 1) = [0.07_dp, -0.11_dp]
        kernel = make_rbf_kernel(1, 1.4_dp, 0.75_dp, status)
        call model%fit(x, components, y, kernel, 0.07_dp, status, jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [derivative GP query products] fit"
            failures = failures + 1
            return
        end if

        call model%predict_input_jvp(x_test, test_components, direction, mean, mean_dot, &
            variance, variance_dot, status)
        h = 2.0e-6_dp
        call oracle_predict(model%parameters(), x, components, y, x_test + h*direction, &
            test_components, 0.07_dp, 1.0e-10_dp, mean_plus, variance_plus)
        call oracle_predict(model%parameters(), x, components, y, x_test - h*direction, &
            test_components, 0.07_dp, 1.0e-10_dp, mean_minus, variance_minus)
        if (.not. status_ok(status) .or. maxval(abs(mean_dot - (mean_plus - mean_minus)/ &
            (2.0_dp*h))) > 3.0e-6_dp .or. maxval(abs(variance_dot - &
            (variance_plus - variance_minus)/(2.0_dp*h))) > 3.0e-6_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [derivative GP query_input_jvp] finite difference ", &
                maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h))), &
                maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h)))
            failures = failures + 1
        end if

        mean_bar(:, 1) = [0.35_dp, -0.2_dp]
        variance_bar = [0.25_dp, -0.15_dp]
        call model%predict_input_vjp(x_test, test_components, mean_bar, variance_bar, &
            x_bar, status)
        do i = 1, size(x_test, 1)
            fd_bar(i, 1) = 0.0_dp
            call oracle_predict(model%parameters(), x, components, y, &
                x_test + h*unit_matrix_column(2, 1, i), test_components, 0.07_dp, &
                1.0e-10_dp, mean_plus, variance_plus)
            objective_plus = sum(mean_bar*mean_plus) + sum(variance_bar*variance_plus)
            call oracle_predict(model%parameters(), x, components, y, &
                x_test - h*unit_matrix_column(2, 1, i), test_components, 0.07_dp, &
                1.0e-10_dp, mean_minus, variance_minus)
            objective_minus = sum(mean_bar*mean_minus) + sum(variance_bar*variance_minus)
            fd_bar(i, 1) = (objective_plus - objective_minus)/(2.0_dp*h)
        end do
        if (.not. status_ok(status) .or. maxval(abs(x_bar - fd_bar)) > 5.0e-6_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [derivative GP query_input_vjp] finite difference ", &
                maxval(abs(x_bar - fd_bar))
            failures = failures + 1
        end if
        adjoint_left = sum(mean_bar*mean_dot) + sum(variance_bar*variance_dot)
        adjoint_right = sum(x_bar*direction)
        if (.not. status_ok(status) .or. abs(adjoint_left - adjoint_right) > 2.0e-7_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [derivative GP query products] adjoint identity ", &
                abs(adjoint_left - adjoint_right)
            failures = failures + 1
        end if
    end subroutine test_query_input_products

    subroutine test_periodic_rational_query_products(failures)
        !! Independent finite-difference and adjoint checks for the exact
        !! third-input products of the smooth built-in leaves.
        integer, intent(inout) :: failures
        type(kernel_t) :: periodic, rational_quadratic, matern32, matern52
        type(fortnum_status_t) :: status

        matern32 = make_matern32_kernel(1, 1.3_dp, 0.8_dp, status)
        call check_smooth_query_kernel(matern32, "matern32", failures)
        matern52 = make_matern52_kernel(1, 1.3_dp, 0.8_dp, status)
        call check_smooth_query_kernel(matern52, "matern52", failures)
        periodic = make_periodic_kernel(1, 1.3_dp, 0.8_dp, 2.1_dp, status)
        call check_smooth_query_kernel(periodic, "periodic", failures)
        rational_quadratic = make_rational_quadratic_kernel(1, 1.3_dp, 0.8_dp, &
            1.7_dp, status)
        call check_smooth_query_kernel(rational_quadratic, "rational-quadratic", failures)
    end subroutine test_periodic_rational_query_products

    subroutine check_smooth_query_kernel(kernel, name, failures)
        type(kernel_t), intent(in) :: kernel
        character(len=*), intent(in) :: name
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), x_test(2, 1), direction(2, 1)
        real(dp) :: mean(2, 1), mean_dot(2, 1), variance(2), variance_dot(2)
        real(dp) :: mean_plus(2, 1), mean_minus(2, 1), variance_plus(2), variance_minus(2)
        real(dp) :: mean_bar(2, 1), variance_bar(2), x_bar(2, 1), fd_bar(2, 1)
        real(dp) :: objective_plus, objective_minus, h, error, lhs, rhs
        integer :: i

        x(:, 1) = [0.0_dp, 0.45_dp, 1.05_dp]
        y(:, 1) = [1.2_dp, -0.3_dp, 0.8_dp]
        x_test(:, 1) = [0.25_dp, 0.8_dp]
        direction(:, 1) = [0.07_dp, -0.11_dp]
        call model%fit(x, [0, 1, 0], y, kernel, 0.07_dp, status, jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a,a)') "FAIL [", trim(name)//" query] fit"
            failures = failures + 1
            return
        end if
        call model%predict_input_jvp(x_test, [1, 0], direction, mean, mean_dot, &
            variance, variance_dot, status)
        h = 1.0e-5_dp
        call model%predict(x_test + h*direction, [1, 0], mean_plus, variance_plus, status)
        call model%predict(x_test - h*direction, [1, 0], mean_minus, variance_minus, status)
        error = max(maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h))), &
            maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h))))
        if (.not. status_ok(status) .or. error > 2.0e-5_dp) then
            write (error_unit, '(a,a,es12.4)') "FAIL [", trim(name)//" query JVP] ", error
            failures = failures + 1
            return
        end if
        mean_bar(:, 1) = [0.35_dp, -0.2_dp]
        variance_bar = [0.25_dp, -0.15_dp]
        call model%predict_input_vjp(x_test, [1, 0], mean_bar, variance_bar, x_bar, status)
        do i = 1, size(x_test, 1)
            call model%predict(x_test + h*unit_matrix_column(2, 1, i), [1, 0], &
                mean_plus, variance_plus, status)
            objective_plus = sum(mean_bar*mean_plus) + sum(variance_bar*variance_plus)
            call model%predict(x_test - h*unit_matrix_column(2, 1, i), [1, 0], &
                mean_minus, variance_minus, status)
            objective_minus = sum(mean_bar*mean_minus) + sum(variance_bar*variance_minus)
            fd_bar(i, 1) = (objective_plus - objective_minus)/(2.0_dp*h)
        end do
        lhs = sum(mean_bar*mean_dot) + sum(variance_bar*variance_dot)
        rhs = sum(x_bar*direction)
        if (.not. status_ok(status) .or. maxval(abs(x_bar - fd_bar)) > 3.0e-5_dp .or. &
            abs(lhs - rhs) > 3.0e-7_dp) then
            write (error_unit, '(a,a,2es12.4)') "FAIL [", trim(name)//" query VJP] ", &
                maxval(abs(x_bar - fd_bar)), abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine check_smooth_query_kernel

    subroutine test_user_formula_observations(failures)
        !! A validated formula must support mixed value/derivative GP states,
        !! including analytic kernel-parameter products.
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_formula_t) :: formula
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x_train(3, 1), y_train(3, 1), x_query(2, 1)
        real(dp) :: mean(2, 1), mean_dot(2, 1), variance(2), variance_dot(2)
        real(dp) :: mean_plus(2, 1), mean_minus(2, 1), variance_plus(2), variance_minus(2)
        real(dp) :: direction(2), input_direction(2, 1), theta(2), h, error

        call formula%reset()
        call formula%push_squared_distance()
        call formula%divide_by_constant(-2.0_dp*0.8_dp*0.8_dp)
        call formula%exponential()
        call formula%validate(status)
        kernel = make_user_kernel(1, 1.2_dp, formula, status)
        x_train(:, 1) = [0.0_dp, 0.55_dp, 1.15_dp]
        y_train(:, 1) = [0.7_dp, -0.2_dp, 0.9_dp]
        call model%fit(x_train, [0, 1, 0], y_train, kernel, 0.05_dp, status, &
            jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [user formula GP] mixed fit"
            failures = failures + 1
            return
        end if

        x_query(:, 1) = [0.2_dp, 0.9_dp]
        call model%predict(x_query, [0, 1], mean, variance, status)
        direction = [0.13_dp, -0.17_dp]
        call model%predict_jvp(x_query, [0, 1], direction, mean, mean_dot, &
            variance, variance_dot, status)
        theta = model%parameters()
        h = 2.0e-5_dp
        call model%set_parameters(theta + h*direction, status)
        call model%predict(x_query, [0, 1], mean_plus, variance_plus, status)
        call model%set_parameters(theta - h*direction, status)
        call model%predict(x_query, [0, 1], mean_minus, variance_minus, status)
        call model%set_parameters(theta, status)
        error = max(maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h))), &
            maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h))))
        if (.not. status_ok(status) .or. error > 2.0e-5_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [user formula GP] parameter JVP finite difference ", error
            failures = failures + 1
        end if

        input_direction(:, 1) = [0.11_dp, -0.09_dp]
        call model%predict_input_jvp(x_query, [0, 1], input_direction, mean, mean_dot, &
            variance, variance_dot, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
            write (error_unit, '(a,i0)') &
                "FAIL [user formula GP] query product refusal code=", status%code
            failures = failures + 1
        end if

        call formula%reset()
        call formula%push_distance()
        call formula%negate()
        call formula%exponential()
        call formula%validate(status)
        kernel = make_user_kernel(1, 1.1_dp, formula, status)
        call model%fit(x_train, [0, 0, 0], y_train, kernel, 0.05_dp, status, &
            jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [user distance formula GP] value-only fit at coincidence"
            failures = failures + 1
            return
        end if
        call model%predict_jvp(x_query, [0, 0], direction, mean, mean_dot, &
            variance, variance_dot, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [user distance formula GP] value-only parameter JVP"
            failures = failures + 1
        end if
    end subroutine test_user_formula_observations

    subroutine test_parameter_guards(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), y(2, 1), parameters(3), value

        x(:, 1) = [0.0_dp, 1.0_dp]
        y(:, 1) = [0.4_dp, -0.2_dp]
        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call model%fit(x, [0, 1], y, kernel, 0.1_dp, status)
        parameters = model%parameters()
        call model%set_parameters(parameters(:2), status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [derivative GP] short parameter vector accepted"
            failures = failures + 1
        end if
        call model%set_parameters([parameters(1), parameters(2), huge(1.0_dp)], status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [derivative GP] nonfinite noise accepted"
            failures = failures + 1
        end if
        call model%log_marginal_likelihood(value, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [derivative GP] failed state after refusal"
            failures = failures + 1
        end if
    end subroutine test_parameter_guards

    function oracle_gradient(theta, x, components, y, noise, jitter) result(gradient)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), noise, jitter
        integer, intent(in) :: components(:)
        real(dp) :: gradient(size(theta))
        real(dp) :: plus(size(theta)), minus(size(theta)), step
        integer :: i

        do i = 1, size(theta)
            step = 2.0e-5_dp*max(1.0_dp, abs(theta(i)))
            plus = theta
            minus = theta
            plus(i) = plus(i) + step
            minus(i) = minus(i) - step
            gradient(i) = (oracle_lml(plus, x, components, y, noise, jitter) - &
                oracle_lml(minus, x, components, y, noise, jitter))/(2.0_dp*step)
        end do
    end function oracle_gradient

    real(dp) function oracle_lml(theta, x, components, y, noise, jitter) result(value)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), noise, jitter
        integer, intent(in) :: components(:)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: status
        real(dp), allocatable :: covariance(:, :), alpha(:, :)
        real(dp) :: logdet
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = oracle_covariance(x(i, 1), components(i), &
                    x(j, 1), components(j), exp(theta(1)), exp(theta(2)))
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(3)) + jitter
        end do
        call factor%factorize(covariance, status)
        allocate(alpha, source=y)
        call factor%solve(alpha, status)
        call factor%log_determinant(logdet, status)
        value = -0.5_dp*sum(y*alpha) - 0.5_dp*logdet - &
            0.5_dp*real(size(x, 1), dp)*log(2.0_dp*acos(-1.0_dp))
    end function oracle_lml

    subroutine oracle_predict(theta, x, components, y, x_test, test_components, noise, &
            jitter, mean, variance)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), x_test(:, :), noise, jitter
        integer, intent(in) :: components(:), test_components(:)
        real(dp), intent(out) :: mean(:, :), variance(:)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: status
        real(dp), allocatable :: covariance(:, :), cross(:, :), alpha(:, :), work(:, :)
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)))
        allocate(cross(size(x, 1), size(x_test, 1)))
        allocate(alpha, source=y)
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = oracle_covariance(x(i, 1), components(i), x(j, 1), &
                    components(j), exp(theta(1)), exp(theta(2)))
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(3)) + jitter
        end do
        call factor%factorize(covariance, status)
        call factor%solve(alpha, status)
        do j = 1, size(x_test, 1)
            do i = 1, size(x, 1)
                cross(i, j) = oracle_covariance(x(i, 1), components(i), x_test(j, 1), &
                    test_components(j), exp(theta(1)), exp(theta(2)))
            end do
        end do
        mean = matmul(transpose(cross), alpha)
        allocate(work, source=cross)
        call factor%solve(work, status)
        do j = 1, size(x_test, 1)
            variance(j) = oracle_covariance(x_test(j, 1), test_components(j), &
                x_test(j, 1), test_components(j), exp(theta(1)), exp(theta(2))) - &
                dot_product(cross(:, j), work(:, j))
        end do
    end subroutine oracle_predict

    subroutine oracle_joint_covariance(theta, x, components, y, x_test, test_components, &
            noise, jitter, covariance_out)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), x_test(:, :), noise, jitter
        integer, intent(in) :: components(:), test_components(:)
        real(dp), intent(out) :: covariance_out(:, :)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: status
        real(dp), allocatable :: covariance(:, :), cross(:, :), work(:, :)
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)))
        allocate(cross(size(x, 1), size(x_test, 1)))
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = oracle_covariance(x(i, 1), components(i), x(j, 1), &
                    components(j), exp(theta(1)), exp(theta(2)))
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(3)) + jitter
        end do
        call factor%factorize(covariance, status)
        do j = 1, size(x_test, 1)
            do i = 1, size(x, 1)
                cross(i, j) = oracle_covariance(x(i, 1), components(i), x_test(j, 1), &
                    test_components(j), exp(theta(1)), exp(theta(2)))
            end do
        end do
        allocate(work, source=cross)
        call factor%solve(work, status)
        do j = 1, size(x_test, 1)
            do i = 1, size(x_test, 1)
                covariance_out(i, j) = oracle_covariance(x_test(i, 1), test_components(i), &
                    x_test(j, 1), test_components(j), exp(theta(1)), exp(theta(2))) - &
                    dot_product(cross(:, i), work(:, j))
            end do
        end do
        covariance_out = 0.5_dp*(covariance_out + transpose(covariance_out))
    end subroutine oracle_joint_covariance

    real(dp) function oracle_prediction_objective(theta, x, components, y, x_test, &
            test_components, noise, jitter, mean_bar, variance_bar) result(value)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), x_test(:, :), noise, jitter
        integer, intent(in) :: components(:), test_components(:)
        real(dp), intent(in) :: mean_bar(:, :), variance_bar(:)
        real(dp) :: mean(size(x_test, 1), size(y, 2)), variance(size(x_test, 1))

        call oracle_predict(theta, x, components, y, x_test, test_components, noise, jitter, &
            mean, variance)
        value = sum(mean_bar*mean) + sum(variance_bar*variance)
    end function oracle_prediction_objective

    real(dp) function oracle_lml_matern32(theta, x, components, y, noise, jitter) result(value)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), noise, jitter
        integer, intent(in) :: components(:)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: status
        real(dp), allocatable :: covariance(:, :), alpha(:, :)
        real(dp) :: logdet
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = oracle_matern32_covariance(x(i, 1), components(i), &
                    x(j, 1), components(j), exp(theta(1)), exp(theta(2)))
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(3)) + jitter
        end do
        call factor%factorize(covariance, status)
        allocate(alpha, source=y)
        call factor%solve(alpha, status)
        call factor%log_determinant(logdet, status)
        value = -0.5_dp*sum(y*alpha) - 0.5_dp*logdet - &
            0.5_dp*real(size(x, 1), dp)*log(2.0_dp*acos(-1.0_dp))
    end function oracle_lml_matern32

    real(dp) function oracle_covariance(x1, component1, x2, component2, &
            variance, lengthscale) result(value)
        real(dp), intent(in) :: x1, x2, variance, lengthscale
        integer, intent(in) :: component1, component2
        real(dp) :: delta, inv_l2, kernel_value

        delta = x1 - x2
        inv_l2 = 1.0_dp/(lengthscale*lengthscale)
        kernel_value = variance*exp(-0.5_dp*delta*delta*inv_l2)
        if (component1 == 0 .and. component2 == 0) then
            value = kernel_value
        else if (component1 > 0 .and. component2 == 0) then
            value = -kernel_value*delta*inv_l2
        else if (component1 == 0 .and. component2 > 0) then
            value = kernel_value*delta*inv_l2
        else
            value = kernel_value*(inv_l2 - delta*delta*inv_l2*inv_l2)
        end if
    end function oracle_covariance

    real(dp) function oracle_matern32_covariance(x1, component1, x2, component2, &
            variance, lengthscale) result(value)
        real(dp), intent(in) :: x1, x2, variance, lengthscale
        integer, intent(in) :: component1, component2
        real(dp) :: delta, distance, z, a, exponential, f, fp, fpp, scale, coefficient

        delta = x1 - x2
        distance = abs(delta)
        z = distance/lengthscale
        a = sqrt(3.0_dp)
        exponential = exp(-a*z)
        f = variance*(1.0_dp + a*z)*exponential
        fp = -3.0_dp*variance*z*exponential/lengthscale
        fpp = 3.0_dp*variance*exponential*(a*z - 1.0_dp)/(lengthscale*lengthscale)
        if (component1 == 0 .and. component2 == 0) then
            value = f
        else if (component1 > 0 .and. component2 == 0) then
            value = fp*delta/max(distance, tiny(1.0_dp))
        else if (component1 == 0 .and. component2 > 0) then
            value = -fp*delta/max(distance, tiny(1.0_dp))
        else if (distance == 0.0_dp) then
            value = -fpp
        else
            scale = fp/distance
            coefficient = (fpp - scale)/(distance*distance)
            value = -(scale + coefficient*delta*delta)
        end if
    end function oracle_matern32_covariance

    function unit_vector(n, position) result(vector)
        integer, intent(in) :: n, position
        real(dp) :: vector(n)

        vector = 0.0_dp
        vector(position) = 1.0_dp
    end function unit_vector

    function unit_matrix_column(n, m, position) result(matrix)
        integer, intent(in) :: n, m, position
        real(dp) :: matrix(n, m)

        matrix = 0.0_dp
        matrix(position, 1) = 1.0_dp
    end function unit_matrix_column

end program test_derivative_gp_products
