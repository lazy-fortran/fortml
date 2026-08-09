program test_second_derivative_gp
    !! Independent RBF order-two GP oracle and typed device-boundary checks.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel, make_matern32_kernel, &
        make_matern52_kernel
    use fortml_second_derivative_gaussian_process, only: second_derivative_gp_t
    implicit none

    type(second_derivative_gp_t) :: model, matern_model, bad_model
    type(kernel_t) :: rbf, matern, matern52
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 1), y(4), query(4, 1), direction(4)
    real(dp) :: mean(4), variance(4), mean_dot(4), variance_dot(4)
    real(dp) :: mean_plus(4), variance_plus(4), mean_minus(4), variance_minus(4)
    real(dp) :: covariance(4, 4), covariance_expected(4, 4)
    real(dp) :: mean_expected(4), variance_expected(4), prior
    real(dp) :: mean_bar(4), variance_bar(4), query_bar(4)
    real(dp), allocatable :: gram(:, :), alpha(:), cross(:, :), work(:, :)
    real(dp), allocatable :: matern_parameters(:), matern_gradient(:), matern_hvp(:)
    integer :: orders(4), query_orders(4), bad_orders(4), failures, i, j
    real(dp) :: h, lhs, rhs, max_error, matern_objective
    real(dp) :: matern_gradient_fd(3), matern_hvp_fd(3), matern_direction(3)
    real(dp) :: matern_h_gradient, matern_h_hvp

    x(:, 1) = [-1.1_dp, -0.25_dp, 0.45_dp, 1.2_dp]
    orders = [0, 1, 2, 0]
    y = [0.7_dp, -0.2_dp, 1.1_dp, -0.45_dp]
    query(:, 1) = [-0.8_dp, -0.1_dp, 0.65_dp, 1.0_dp]
    query_orders = [0, 1, 2, 0]
    direction = [0.23_dp, -0.17_dp, 0.11_dp, -0.19_dp]
    failures = 0

    rbf = make_rbf_kernel(1, 1.6_dp, 0.75_dp, status)
    call check(status_ok(status), "RBF kernel construction", failures)
    call model%fit(x, orders, y, rbf, 0.035_dp, status, 1.0e-10_dp)
    call check(status_ok(status) .and. model%fitted() .and. model%observation_count() == 4, &
        "order-two RBF fit", failures)
    call check(model%parameter_count() == 3 .and. all(ieee_is_finite(model%parameters())), &
        "hyperparameter metadata", failures)

    call independent_setup(x, orders, y, 1.6_dp, 0.75_dp, 0.035_dp, gram, alpha)
    allocate(cross(4, 4), work(4, 4))
    do j = 1, 4
        do i = 1, 4
            cross(i, j) = oracle_cov(1.6_dp, 0.75_dp, x(i, 1), orders(i), &
                query(j, 1), query_orders(j))
        end do
    end do
    mean_expected = matmul(transpose(cross), alpha)
    work = cross
    do j = 1, 4
        call oracle_solve(gram, work(:, j))
    end do
    do j = 1, 4
        prior = oracle_cov(1.6_dp, 0.75_dp, query(j, 1), query_orders(j), &
            query(j, 1), query_orders(j))
        variance_expected(j) = prior - dot_product(cross(:, j), work(:, j))
    end do
    call model%predict(query, query_orders, mean, variance, status)
    max_error = max(maxval(abs(mean - mean_expected)), maxval(abs(variance - variance_expected)))
    call check(status_ok(status) .and. max_error < 3.0e-10_dp, &
        "independent covariance/prediction oracle", failures)

    call model%joint_covariance(query, query_orders, covariance, status)
    do j = 1, 4
        do i = 1, 4
            covariance_expected(i, j) = oracle_cov(1.6_dp, 0.75_dp, query(i, 1), &
                query_orders(i), query(j, 1), query_orders(j)) - dot_product(cross(:, i), work(:, j))
        end do
    end do
    covariance_expected = 0.5_dp*(covariance_expected + transpose(covariance_expected))
    call check(status_ok(status) .and. maxval(abs(covariance - covariance_expected)) < 3.0e-10_dp, &
        "independent joint covariance oracle", failures)

    h = 2.0e-5_dp
    call model%predict_input_jvp(query, query_orders, direction, mean, mean_dot, variance, &
        variance_dot, status)
    call model%predict(query + h*spread(direction, 2, 1), query_orders, mean_plus, variance_plus, status)
    call model%predict(query - h*spread(direction, 2, 1), query_orders, mean_minus, variance_minus, status)
    call check(status_ok(status) .and. maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h))) < 2.0e-6_dp &
        .and. maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h))) < 2.0e-6_dp, &
        "input JVP central difference", failures)

    mean_bar = [0.3_dp, -0.5_dp, 0.2_dp, 0.4_dp]
    variance_bar = [-0.2_dp, 0.6_dp, -0.1_dp, 0.25_dp]
    call model%predict_input_vjp(query, query_orders, mean_bar, variance_bar, query_bar, status)
    lhs = dot_product(query_bar, direction)
    rhs = dot_product(mean_bar, mean_dot) + dot_product(variance_bar, variance_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-6_dp, "input VJP duality", failures)

    matern52 = make_matern52_kernel(1, 1.6_dp, 0.75_dp, status)
    call check(status_ok(status), "Matern-5/2 kernel construction", failures)
    call matern_model%fit(x, orders, y, matern52, 0.035_dp, status, 1.0e-10_dp)
    call check(status_ok(status) .and. matern_model%fitted(), "order-two Matern-5/2 fit", failures)
    matern_parameters = matern_model%parameters()
    allocate(matern_gradient(3), matern_hvp(3))
    call matern_model%log_marginal_likelihood(matern_objective, status)
    call matern_model%hyperparameter_gradient(matern_gradient, status)
    call check(status_ok(status) .and. all(ieee_is_finite(matern_gradient)), &
        "Matern-5/2 analytic likelihood gradient", failures)
    matern_h_gradient = 2.0e-5_dp
    do i = 1, 3
        matern_gradient_fd(i) = (matern_oracle_lml(matern_parameters + matern_h_gradient* &
            unit_vector(3, i), x, orders, y, 0.035_dp) - matern_oracle_lml(matern_parameters - &
            matern_h_gradient*unit_vector(3, i), x, orders, y, 0.035_dp))/(2.0_dp*matern_h_gradient)
    end do
    call check(maxval(abs(matern_gradient - matern_gradient_fd)) < 3.0e-5_dp, &
        "Matern-5/2 likelihood gradient finite difference", failures)
    matern_direction = [0.13_dp, -0.09_dp, 0.07_dp]
    call matern_model%hyperparameter_hvp(matern_direction, matern_hvp, status)
    matern_h_hvp = 2.0e-3_dp
    do i = 1, 3
        matern_hvp_fd(i) = matern_mixed_second_difference(matern_parameters, matern_direction, &
            unit_vector(3, i), x, orders, y, 0.035_dp, matern_h_hvp)
    end do
    call check(status_ok(status) .and. maxval(abs(matern_hvp - matern_hvp_fd)) < 5.0e-4_dp, &
        "Matern-5/2 likelihood HVP finite difference", failures)
    deallocate(gram, alpha)
    call matern_independent_setup(x, orders, y, 1.6_dp, 0.75_dp, 0.035_dp, gram, alpha)
    do j = 1, 4
        do i = 1, 4
            cross(i, j) = matern52_oracle_cov(1.6_dp, 0.75_dp, x(i, 1), orders(i), &
                query(j, 1), query_orders(j))
        end do
    end do
    mean_expected = matmul(transpose(cross), alpha)
    work = cross
    do j = 1, 4
        call oracle_solve(gram, work(:, j))
    end do
    do j = 1, 4
        prior = matern52_oracle_cov(1.6_dp, 0.75_dp, query(j, 1), query_orders(j), &
            query(j, 1), query_orders(j))
        variance_expected(j) = prior - dot_product(cross(:, j), work(:, j))
    end do
    call matern_model%predict(query, query_orders, mean, variance, status)
    max_error = max(maxval(abs(mean - mean_expected)), maxval(abs(variance - variance_expected)))
    call check(status_ok(status) .and. max_error < 5.0e-10_dp, &
        "independent Matern-5/2 prediction oracle", failures)

    h = 2.0e-5_dp
    call matern_model%predict_input_jvp(query, query_orders, direction, mean, mean_dot, variance, &
        variance_dot, status)
    call matern_model%predict(query + h*spread(direction, 2, 1), query_orders, mean_plus, variance_plus, status)
    call matern_model%predict(query - h*spread(direction, 2, 1), query_orders, mean_minus, variance_minus, status)
    call check(status_ok(status) .and. maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h))) < 5.0e-5_dp &
        .and. maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h))) < 5.0e-5_dp, &
        "Matern-5/2 input JVP central difference", failures)

    call matern_model%predict_input_vjp(query, query_orders, mean_bar, variance_bar, query_bar, status)
    lhs = dot_product(query_bar, direction)
    rhs = dot_product(mean_bar, mean_dot) + dot_product(variance_bar, variance_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 8.0e-5_dp, &
        "Matern-5/2 input VJP duality", failures)

    query(1, 1) = x(3, 1)
    query_orders(1) = 2
    call matern_model%predict_input_jvp(query, query_orders, direction, mean, mean_dot, variance, &
        variance_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "Matern-5/2 coincident fifth-derivative refusal", failures)
    query(1, 1) = -0.8_dp
    query_orders(1) = 0

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, query, query_orders, mean, variance, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "typed CUDA prediction refusal", failures)
    call model%joint_covariance_device(cuda, query, query_orders, covariance, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "typed CUDA covariance refusal", failures)
    call model%predict_input_jvp_device(cuda, query, query_orders, direction, mean, mean_dot, &
        variance, variance_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "typed CUDA input JVP refusal", failures)
    call model%predict_input_vjp_device(cuda, query, query_orders, mean_bar, variance_bar, &
        query_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "typed CUDA input VJP refusal", failures)
    call check(.not. model%device_supported(FORTML_DEVICE_CUDA), "CUDA capability metadata", failures)

    matern = make_matern32_kernel(1, 1.6_dp, 0.75_dp, status)
    call bad_model%fit(x, orders, y, matern, 0.035_dp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "non-RBF typed refusal", failures)
    bad_orders = [0, 1, 4, 0]
    call bad_model%fit(x, bad_orders, y, rbf, 0.035_dp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "order-four generated-kernel refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL second-derivative GP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS second-derivative GP independent behavioral oracle"

contains

    subroutine independent_setup(x, orders, y, variance, lengthscale, noise, gram, alpha)
        real(dp), intent(in) :: x(:, :), y(:), variance, lengthscale, noise
        integer, intent(in) :: orders(:)
        real(dp), allocatable, intent(out) :: gram(:, :), alpha(:)
        integer :: i, j, n

        n = size(x, 1)
        allocate(gram(n, n), alpha(n))
        do j = 1, n
            do i = 1, n
                gram(i, j) = oracle_cov(variance, lengthscale, x(i, 1), orders(i), &
                    x(j, 1), orders(j))
            end do
            gram(j, j) = gram(j, j) + noise + 1.0e-10_dp
        end do
        alpha = y
        call oracle_solve(gram, alpha)
    end subroutine independent_setup

    subroutine matern_independent_setup(x, orders, y, variance, lengthscale, noise, gram, alpha)
        real(dp), intent(in) :: x(:, :), y(:), variance, lengthscale, noise
        integer, intent(in) :: orders(:)
        real(dp), allocatable, intent(out) :: gram(:, :), alpha(:)
        integer :: i, j, n

        n = size(x, 1)
        allocate(gram(n, n), alpha(n))
        do j = 1, n
            do i = 1, n
                gram(i, j) = matern52_oracle_cov(variance, lengthscale, x(i, 1), orders(i), &
                    x(j, 1), orders(j))
            end do
            gram(j, j) = gram(j, j) + noise + 1.0e-10_dp
        end do
        alpha = y
        call oracle_solve(gram, alpha)
    end subroutine matern_independent_setup

    real(dp) function matern_oracle_lml(theta, x, orders, y, noise) result(value)
        real(dp), intent(in) :: theta(:), x(:, :), y(:), noise
        integer, intent(in) :: orders(:)
        real(dp), allocatable :: gram(:, :), alpha(:), lower(:, :)
        real(dp) :: total, logdet
        integer :: i, j, k, n

        n = size(y)
        allocate(gram(n, n), alpha(n), lower(n, n))
        do j = 1, n
            do i = 1, n
                gram(i, j) = matern52_oracle_cov(exp(theta(1)), exp(theta(2)), x(i, 1), &
                    orders(i), x(j, 1), orders(j))
            end do
            gram(j, j) = gram(j, j) + exp(theta(3)) + 1.0e-10_dp
        end do
        lower = 0.0_dp
        do i = 1, n
            do j = 1, i
                total = gram(i, j)
                do k = 1, j - 1
                    total = total - lower(i, k)*lower(j, k)
                end do
                if (i == j) then
                    lower(i, j) = sqrt(total)
                else
                    lower(i, j) = total/lower(j, j)
                end if
            end do
        end do
        alpha = y
        call oracle_solve(gram, alpha)
        logdet = 0.0_dp
        do i = 1, n
            logdet = logdet + 2.0_dp*log(lower(i, i))
        end do
        value = -0.5_dp*dot_product(y, alpha) - 0.5_dp*logdet - &
            0.5_dp*real(n, dp)*log(2.0_dp*acos(-1.0_dp))
    end function matern_oracle_lml

    real(dp) function matern_mixed_second_difference(theta, direction, probe, x, orders, y, &
            noise, step) result(value)
        real(dp), intent(in) :: theta(:), direction(:), probe(:), x(:, :), y(:), noise, step
        integer, intent(in) :: orders(:)
        real(dp) :: pp(size(theta)), pm(size(theta)), mp(size(theta)), mm(size(theta))

        pp = theta + step*direction + step*probe
        pm = theta + step*direction - step*probe
        mp = theta - step*direction + step*probe
        mm = theta - step*direction - step*probe
        value = (matern_oracle_lml(pp, x, orders, y, noise) - matern_oracle_lml(pm, x, orders, y, noise) - &
            matern_oracle_lml(mp, x, orders, y, noise) + matern_oracle_lml(mm, x, orders, y, noise))/ &
            (4.0_dp*step*step)
    end function matern_mixed_second_difference

    pure function unit_vector(n, index) result(vector)
        integer, intent(in) :: n, index
        real(dp) :: vector(n)

        vector = 0.0_dp
        vector(index) = 1.0_dp
    end function unit_vector

    subroutine oracle_solve(matrix, rhs)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), intent(inout) :: rhs(:)
        real(dp), allocatable :: lower(:, :), work(:)
        integer :: i, j, k, n
        real(dp) :: total

        n = size(rhs)
        allocate(lower(n, n), work(n))
        lower = 0.0_dp
        do i = 1, n
            do j = 1, i
                total = matrix(i, j)
                do k = 1, j - 1
                    total = total - lower(i, k)*lower(j, k)
                end do
                if (i == j) then
                    lower(i, j) = sqrt(total)
                else
                    lower(i, j) = total/lower(j, j)
                end if
            end do
        end do
        do i = 1, n
            total = rhs(i)
            do k = 1, i - 1
                total = total - lower(i, k)*work(k)
            end do
            work(i) = total/lower(i, i)
        end do
        do i = n, 1, -1
            total = work(i)
            do k = i + 1, n
                total = total - lower(k, i)*rhs(k)
            end do
            rhs(i) = total/lower(i, i)
        end do
    end subroutine oracle_solve

    pure real(dp) function oracle_cov(variance, lengthscale, x1, order1, x2, order2) result(value)
        real(dp), intent(in) :: variance, lengthscale, x1, x2
        integer, intent(in) :: order1, order2
        real(dp) :: d, base

        d = x1 - x2
        base = variance*exp(-0.5_dp*d*d/(lengthscale*lengthscale))
        value = (-1.0_dp)**order2*oracle_distance_derivative(base, d, lengthscale, order1 + order2)
    end function oracle_cov

    pure real(dp) function matern52_oracle_cov(variance, lengthscale, x1, order1, x2, order2) result(value)
        real(dp), intent(in) :: variance, lengthscale, x1, x2
        integer, intent(in) :: order1, order2
        real(dp), parameter :: root_five = 2.2360679774997896964_dp
        real(dp) :: tau, radius, base, radial_derivative
        integer :: total_order, sign_tau

        tau = x1 - x2
        radius = abs(tau)/lengthscale
        base = variance*exp(-root_five*radius)
        total_order = order1 + order2
        select case (total_order)
        case (0)
            radial_derivative = 1.0_dp + root_five*radius + (5.0_dp/3.0_dp)*radius*radius
        case (1)
            radial_derivative = -(5.0_dp/3.0_dp)*radius*(1.0_dp + root_five*radius)
        case (2)
            radial_derivative = (5.0_dp/3.0_dp)*(5.0_dp*radius*radius - root_five*radius - 1.0_dp)
        case (3)
            radial_derivative = (25.0_dp/3.0_dp)*radius*(3.0_dp - root_five*radius)
        case (4)
            radial_derivative = (25.0_dp/3.0_dp)*(3.0_dp - 5.0_dp*root_five*radius + 5.0_dp*radius*radius)
        case default
            radial_derivative = 0.0_dp
        end select
        sign_tau = 1
        if (tau < 0.0_dp) sign_tau = -1
        if (mod(total_order, 2) == 1) radial_derivative = sign_tau*radial_derivative
        value = base*radial_derivative/lengthscale**total_order
        if (mod(order2, 2) == 1) value = -value
    end function matern52_oracle_cov

    pure real(dp) function oracle_distance_derivative(base, d, lengthscale, order) result(value)
        real(dp), intent(in) :: base, d, lengthscale
        integer, intent(in) :: order
        real(dp) :: inv2, inv4, inv6, inv8, inv10

        inv2 = 1.0_dp/(lengthscale*lengthscale)
        inv4 = inv2*inv2
        inv6 = inv4*inv2
        inv8 = inv4*inv4
        inv10 = inv8*inv2
        select case (order)
        case (0)
            value = base
        case (1)
            value = -d*inv2*base
        case (2)
            value = (d*d*inv4 - inv2)*base
        case (3)
            value = (3.0_dp*d*inv4 - d*d*d*inv6)*base
        case (4)
            value = (d**4*inv8 - 6.0_dp*d*d*inv6 + 3.0_dp*inv4)*base
        case (5)
            value = (-d**5*inv10 + 10.0_dp*d**3*inv8 - 15.0_dp*d*inv6)*base
        case default
            value = 0.0_dp
        end select
    end function oracle_distance_derivative

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [second-derivative-gp] "//description
        end if
    end subroutine check

end program test_second_derivative_gp
