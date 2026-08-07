program test_derivative_gp_products
    !! Independent likelihood/product checks for mixed value/derivative GPs.
    !! The oracle assembles the covariance from the RBF partials directly;
    !! it does not call the production derivative-covariance helper.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, kernel_multiply, make_linear_kernel, &
        make_matern32_kernel, make_rbf_kernel
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_likelihood_products(failures)
    call test_matern_parameter_products(failures)
    call test_product_parameter_products(failures)
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

end program test_derivative_gp_products
