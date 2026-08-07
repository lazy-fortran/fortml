program test_derivative_gp_spectral_mixture
    !! Independent dense-oracle checks for spectral-mixture derivative GPs.
    !! The covariance oracle below re-derives value, first, and mixed second
    !! lag derivatives directly; it does not call the production derivative
    !! covariance or spectral helper routines.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_spectral_mixture_kernel
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_spectral_derivative_products(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " spectral-mixture derivative GP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS spectral-mixture derivative GP dense oracles"

contains

    subroutine test_spectral_derivative_products(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: weights(2), means(2, 2), scales(2, 2)
        real(dp) :: x_train(5, 2), y_train(5, 1), x_query(3, 2)
        real(dp) :: mean(3, 1), variance(3), mean_ref(3, 1), variance_ref(3)
        real(dp) :: posterior(3, 3), posterior_ref(3, 3)
        real(dp) :: theta(11), gradient(11), fd_gradient(11), direction(11)
        real(dp) :: mean_dot(3, 1), variance_dot(3), mean_plus(3, 1), mean_minus(3, 1)
        real(dp) :: variance_plus(3), variance_minus(3), mean_bar(3, 1), variance_bar(3)
        real(dp) :: x_bar(3, 2), fd_bar(3, 2), objective_plus, objective_minus
        real(dp) :: h, h_input, lhs, rhs, value, expected, hvp(11)
        integer :: components(5), query_components(3), i, j

        weights = [1.15_dp, 0.63_dp]
        means = reshape([0.21_dp, -0.37_dp, 0.48_dp, 0.16_dp], shape(means))
        scales = reshape([0.31_dp, 0.57_dp, 0.22_dp, 0.44_dp], shape(scales))
        x_train = reshape([ &
            -0.7_dp, 0.2_dp, 0.35_dp, -0.4_dp, 0.9_dp, 0.65_dp, &
            -0.1_dp, 1.1_dp, 0.55_dp, -0.85_dp], shape(x_train))
        y_train(:, 1) = [0.8_dp, -0.25_dp, 0.45_dp, 1.1_dp, -0.6_dp]
        components = [0, 1, 2, 0, 1]
        x_query = reshape([0.15_dp, -0.25_dp, 0.75_dp, 0.4_dp, -0.45_dp, 0.95_dp], &
            shape(x_query))
        query_components = [2, 0, 1]
        kernel = make_spectral_mixture_kernel(2, 2, weights, means, scales, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [spectral derivative GP] constructor"
            failures = failures + 1
            return
        end if
        call model%fit(x_train, components, y_train, kernel, 0.055_dp, status, &
            jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [spectral derivative GP] fit"
            failures = failures + 1
            return
        end if
        theta = model%parameters()

        call model%log_marginal_likelihood(value, status)
        expected = oracle_lml(theta, x_train, components, y_train, 0.055_dp, 1.0e-10_dp)
        if (.not. status_ok(status) .or. abs(value - expected) > 3.0e-10_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [spectral derivative GP] independent likelihood oracle ", &
                abs(value - expected)
            failures = failures + 1
        end if

        call model%predict(x_query, query_components, mean, variance, status)
        call oracle_predict(theta, x_train, components, y_train, x_query, query_components, &
            0.055_dp, 1.0e-10_dp, mean_ref, variance_ref)
        if (.not. status_ok(status) .or. maxval(abs(mean - mean_ref)) > 3.0e-10_dp .or. &
            maxval(abs(variance - variance_ref)) > 3.0e-10_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [spectral derivative GP] independent prediction oracle ", &
                maxval(abs(mean - mean_ref)), maxval(abs(variance - variance_ref))
            failures = failures + 1
        end if

        call model%joint_covariance(x_query, query_components, posterior, status)
        call oracle_joint_covariance(theta, x_train, components, x_query, query_components, &
            0.055_dp, 1.0e-10_dp, posterior_ref)
        if (.not. status_ok(status) .or. maxval(abs(posterior - posterior_ref)) > 4.0e-10_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [spectral derivative GP] independent joint oracle ", &
                maxval(abs(posterior - posterior_ref))
            failures = failures + 1
        end if

        call model%hyperparameter_gradient(gradient, status)
        h = 3.0e-6_dp
        do i = 1, size(theta)
            fd_gradient(i) = (oracle_lml(theta + h*unit_vector(size(theta), i), x_train, &
                components, y_train, 0.055_dp, 1.0e-10_dp) - oracle_lml(theta - &
                h*unit_vector(size(theta), i), x_train, components, y_train, 0.055_dp, &
                1.0e-10_dp))/(2.0_dp*h)
        end do
        if (.not. status_ok(status) .or. maxval(abs(gradient - fd_gradient)) > 3.0e-6_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [spectral derivative GP] hyperparameter gradient oracle ", &
                maxval(abs(gradient - fd_gradient))
            failures = failures + 1
        end if

        direction = [0.13_dp, -0.08_dp, 0.11_dp, -0.05_dp, 0.07_dp, -0.04_dp, &
            0.09_dp, -0.06_dp, 0.12_dp, -0.03_dp, 0.17_dp]
        call model%hyperparameter_hvp(direction, hvp, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
            write (error_unit, '(a,i0)') &
                "FAIL [spectral derivative GP] mixed HVP refusal code=", status%code
            failures = failures + 1
        end if

        direction(1:6) = [0.07_dp, -0.11_dp, 0.05_dp, 0.09_dp, -0.08_dp, 0.06_dp]
        direction(7:11) = 0.0_dp
        call model%predict_input_jvp(x_query, query_components, reshape(direction(1:6), [3, 2]), &
            mean, mean_dot, variance, variance_dot, status)
        h_input = 2.0e-6_dp
        call model%predict(x_query + h_input*reshape(direction(1:6), [3, 2]), query_components, &
            mean_plus, variance_plus, status)
        call model%predict(x_query - h_input*reshape(direction(1:6), [3, 2]), query_components, &
            mean_minus, variance_minus, status)
        if (.not. status_ok(status) .or. maxval(abs(mean_dot - (mean_plus - mean_minus)/ &
            (2.0_dp*h_input))) > 4.0e-6_dp .or. maxval(abs(variance_dot - &
            (variance_plus - variance_minus)/(2.0_dp*h_input))) > 4.0e-6_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [spectral derivative GP] query JVP finite difference ", &
                maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h_input))), &
                maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h_input)))
            failures = failures + 1
        end if

        mean_bar(:, 1) = [0.35_dp, -0.2_dp, 0.17_dp]
        variance_bar = [0.25_dp, -0.15_dp, 0.11_dp]
        call model%predict_input_vjp(x_query, query_components, mean_bar, variance_bar, &
            x_bar, status)
        do i = 1, size(x_query, 1)
            do j = 1, size(x_query, 2)
                call model%predict(x_query + h_input*unit_matrix([size(x_query, 1), &
                    size(x_query, 2)], i, j), &
                    query_components, mean_plus, variance_plus, status)
                objective_plus = sum(mean_bar*mean_plus) + sum(variance_bar*variance_plus)
                call model%predict(x_query - h_input*unit_matrix([size(x_query, 1), &
                    size(x_query, 2)], i, j), &
                    query_components, mean_minus, variance_minus, status)
                objective_minus = sum(mean_bar*mean_minus) + sum(variance_bar*variance_minus)
                fd_bar(i, j) = (objective_plus - objective_minus)/(2.0_dp*h_input)
            end do
        end do
        lhs = sum(mean_bar*mean_dot) + sum(variance_bar*variance_dot)
        rhs = sum(x_bar*reshape(direction(1:6), [3, 2]))
        if (.not. status_ok(status) .or. maxval(abs(x_bar - fd_bar)) > 5.0e-6_dp .or. &
            abs(lhs - rhs) > 2.0e-7_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [spectral derivative GP] query VJP oracle/adjoint ", &
                maxval(abs(x_bar - fd_bar)), abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_spectral_derivative_products

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
                covariance(i, j) = oracle_covariance(theta, x(i, :), components(i), &
                    x(j, :), components(j))
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(size(theta))) + jitter
        end do
        call factor%factorize(covariance, status)
        allocate(alpha, source=y)
        call factor%solve(alpha, status)
        call factor%log_determinant(logdet, status)
        value = -0.5_dp*sum(y*alpha) - 0.5_dp*logdet - &
            0.5_dp*real(size(x, 1), dp)*log(2.0_dp*acos(-1.0_dp))
    end function oracle_lml

    subroutine oracle_predict(theta, x, components, y, x_query, query_components, noise, &
            jitter, mean, variance)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), x_query(:, :), noise, jitter
        integer, intent(in) :: components(:), query_components(:)
        real(dp), intent(out) :: mean(:, :), variance(:)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: status
        real(dp), allocatable :: covariance(:, :), cross(:, :), alpha(:, :), work(:, :)
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)), cross(size(x, 1), size(x_query, 1)))
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
        call factor%factorize(covariance, status)
        call factor%solve(alpha, status)
        do j = 1, size(x_query, 1)
            do i = 1, size(x, 1)
                cross(i, j) = oracle_covariance(theta, x(i, :), components(i), &
                    x_query(j, :), query_components(j))
            end do
        end do
        mean = matmul(transpose(cross), alpha)
        allocate(work, source=cross)
        call factor%solve(work, status)
        do j = 1, size(x_query, 1)
            variance(j) = oracle_covariance(theta, x_query(j, :), query_components(j), &
                x_query(j, :), query_components(j)) - dot_product(cross(:, j), work(:, j))
        end do
    end subroutine oracle_predict

    subroutine oracle_joint_covariance(theta, x, components, query, query_components, noise, &
            jitter, covariance_out)
        real(dp), intent(in) :: theta(:), x(:, :), query(:, :), noise, jitter
        integer, intent(in) :: components(:), query_components(:)
        real(dp), intent(out) :: covariance_out(:, :)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: status
        real(dp), allocatable :: covariance(:, :), cross(:, :), work(:, :)
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)), cross(size(x, 1), size(query, 1)))
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = oracle_covariance(theta, x(i, :), components(i), &
                    x(j, :), components(j))
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(size(theta))) + jitter
        end do
        call factor%factorize(covariance, status)
        do j = 1, size(query, 1)
            do i = 1, size(x, 1)
                cross(i, j) = oracle_covariance(theta, x(i, :), components(i), query(j, :), &
                    query_components(j))
            end do
        end do
        allocate(work, source=cross)
        call factor%solve(work, status)
        do j = 1, size(query, 1)
            do i = 1, size(query, 1)
                covariance_out(i, j) = oracle_covariance(theta, query(i, :), query_components(i), &
                    query(j, :), query_components(j)) - dot_product(cross(:, i), work(:, j))
            end do
        end do
        covariance_out = 0.5_dp*(covariance_out + transpose(covariance_out))
    end subroutine oracle_joint_covariance

    real(dp) function oracle_covariance(theta, x1, component1, x2, component2) result(value)
        real(dp), intent(in) :: theta(:), x1(:), x2(:)
        integer, intent(in) :: component1, component2
        real(dp) :: tau(size(x1)), f(size(x1)), f1(size(x1)), f2(size(x1))
        real(dp) :: a, phase, scale, mean, weight, component
        real(dp) :: l1, l2, e, c, c1, c2
        integer :: d, q, block, base, i

        d = size(x1)
        tau = x1 - x2
        a = 2.0_dp*acos(-1.0_dp)
        block = 1 + 2*d
            value = 0.0_dp
            do q = 1, (size(theta) - 1)/block
                base = (q - 1)*block
                weight = exp(theta(base + 1))
                do i = 1, d
                    scale = exp(theta(base + 1 + i))
                    mean = theta(base + 1 + d + i)
                    phase = a*tau(i)*mean
                    l1 = -a*a*tau(i)*scale*scale
                    l2 = -a*a*scale*scale
                    e = exp(-0.5_dp*a*a*tau(i)*tau(i)*scale*scale)
                    c = cos(phase)
                    c1 = -a*mean*sin(phase)
                    c2 = -a*a*mean*mean*cos(phase)
                    f(i) = e*c
                    f1(i) = e*(l1*c + c1)
                    f2(i) = e*((l2 + l1*l1)*c + 2.0_dp*l1*c1 + c2)
                end do
                component = product(f)
                if (component1 == 0 .and. component2 == 0) then
                    value = value + weight*component
                else if (component1 > 0 .and. component2 == 0) then
                    value = value + weight*f1(component1)*product_except(f, component1)
                else if (component1 == 0 .and. component2 > 0) then
                    value = value - weight*f1(component2)*product_except(f, component2)
                else if (component1 == component2) then
                    value = value - weight*f2(component1)*product_except(f, component1)
                else
                    value = value - weight*f1(component1)*f1(component2)* &
                        product_except_two(f, component1, component2)
                end if
            end do
        end function oracle_covariance

        pure function product_except(values, skip) result(product_value)
            real(dp), intent(in) :: values(:)
            integer, intent(in) :: skip
            real(dp) :: product_value
            integer :: i

            product_value = 1.0_dp
            do i = 1, size(values)
                if (i /= skip) product_value = product_value*values(i)
            end do
        end function product_except

        pure function product_except_two(values, skip1, skip2) result(product_value)
            real(dp), intent(in) :: values(:)
            integer, intent(in) :: skip1, skip2
            real(dp) :: product_value
            integer :: i

            product_value = 1.0_dp
            do i = 1, size(values)
                if (i /= skip1 .and. i /= skip2) product_value = product_value*values(i)
            end do
        end function product_except_two

        pure function unit_vector(n, index) result(vector)
            integer, intent(in) :: n, index
            real(dp) :: vector(n)

            vector = 0.0_dp
            vector(index) = 1.0_dp
        end function unit_vector

        pure function unit_matrix(shape1, row, col) result(matrix)
            integer, intent(in) :: shape1(:), row, col
            real(dp) :: matrix(shape1(1), shape1(2))

            matrix = 0.0_dp
            matrix(row, col) = 1.0_dp
        end function unit_matrix

    end program test_derivative_gp_spectral_mixture
