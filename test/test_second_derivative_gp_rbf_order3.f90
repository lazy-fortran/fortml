program test_second_derivative_gp_rbf_order3
    !! Independent behavioral oracle for the smooth RBF order-three lane.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel, make_matern52_kernel
    use fortml_second_derivative_gaussian_process, only: second_derivative_gp_t
    implicit none

    type(second_derivative_gp_t) :: model, probe, matern
    type(kernel_t) :: rbf, matern52
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(5, 1), y(5), query(4, 1), direction(4)
    real(dp) :: mean(4), variance(4), mean_dot(4), variance_dot(4)
    real(dp) :: mean_plus(4), variance_plus(4), mean_minus(4), variance_minus(4)
    real(dp) :: mean_bar(4), variance_bar(4), query_bar(4)
    real(dp), allocatable :: gram(:, :), alpha(:), cross(:, :), work(:, :)
    real(dp), allocatable :: parameters(:), gradient(:), hvp(:), gradient_plus(:), gradient_minus(:)
    real(dp) :: covariance(4, 4), covariance_expected(4, 4)
    real(dp) :: mean_expected(4), variance_expected(4)
    real(dp) :: objective, objective_plus, objective_minus, prior
    integer :: orders(5), query_orders(4), failures, i, j
    real(dp) :: h, max_error, lhs, rhs

    x(:, 1) = [-1.2_dp, -0.45_dp, 0.15_dp, 0.82_dp, 1.35_dp]
    y = [0.73_dp, -0.24_dp, 0.91_dp, -0.38_dp, 0.19_dp]
    orders = [0, 1, 2, 3, 0]
    query(:, 1) = [-0.91_dp, -0.17_dp, 0.57_dp, 1.08_dp]
    query_orders = [0, 1, 2, 3]
    direction = [0.17_dp, -0.11_dp, 0.08_dp, -0.14_dp]
    mean_bar = [0.31_dp, -0.42_dp, 0.16_dp, 0.27_dp]
    variance_bar = [-0.12_dp, 0.34_dp, -0.21_dp, 0.18_dp]
    failures = 0

    rbf = make_rbf_kernel(1, 1.35_dp, 0.79_dp, status)
    call check(status_ok(status), "RBF constructor", failures)
    call model%fit(x, orders, y, rbf, 0.041_dp, status, 1.0e-10_dp)
    call check(status_ok(status) .and. model%fitted(), "RBF order-three fit", failures)
    call check(model%parameter_count() == 3 .and. all(ieee_is_finite(model%parameters())), &
        "RBF parameter metadata", failures)

    call independent_setup(x, orders, y, 1.35_dp, 0.79_dp, 0.041_dp, gram, alpha)
    allocate(cross(5, 4), work(5, 4))
    do j = 1, 4
        do i = 1, 5
            cross(i, j) = oracle_cov(1.35_dp, 0.79_dp, x(i, 1), orders(i), query(j, 1), &
                query_orders(j))
        end do
    end do
    mean_expected = matmul(transpose(cross), alpha)
    work = cross
    do j = 1, 4
        call oracle_solve(gram, work(:, j))
    end do
    do j = 1, 4
        prior = oracle_cov(1.35_dp, 0.79_dp, query(j, 1), query_orders(j), query(j, 1), &
            query_orders(j))
        variance_expected(j) = prior - dot_product(cross(:, j), work(:, j))
    end do
    call model%predict(query, query_orders, mean, variance, status)
    max_error = max(maxval(abs(mean - mean_expected)), maxval(abs(variance - variance_expected)))
    call check(status_ok(status) .and. max_error < 2.0e-9_dp, &
        "order-three independent prediction oracle", failures)

    call model%joint_covariance(query, query_orders, covariance, status)
    do j = 1, 4
        do i = 1, 4
            covariance_expected(i, j) = oracle_cov(1.35_dp, 0.79_dp, query(i, 1), &
                query_orders(i), query(j, 1), query_orders(j)) - dot_product(work(:, i), cross(:, j))
        end do
    end do
    covariance_expected = 0.5_dp*(covariance_expected + transpose(covariance_expected))
    call check(status_ok(status) .and. maxval(abs(covariance - covariance_expected)) < 2.0e-9_dp, &
        "order-three latent covariance oracle", failures)

    h = 2.0e-5_dp
    call model%predict_input_jvp(query, query_orders, direction, mean, mean_dot, variance, &
        variance_dot, status)
    call model%predict(query + h*spread(direction, 2, 1), query_orders, mean_plus, variance_plus, status)
    call model%predict(query - h*spread(direction, 2, 1), query_orders, mean_minus, variance_minus, status)
    call check(status_ok(status) .and. maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h))) < &
        4.0e-6_dp .and. maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h))) < &
        4.0e-6_dp, "order-three input JVP finite difference", failures)

    call model%predict_input_vjp(query, query_orders, mean_bar, variance_bar, query_bar, status)
    lhs = dot_product(query_bar, direction)
    rhs = dot_product(mean_bar, mean_dot) + dot_product(variance_bar, variance_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 5.0e-6_dp, &
        "order-three input VJP adjoint identity", failures)

    allocate(parameters(model%parameter_count()), gradient(model%parameter_count()), &
        hvp(model%parameter_count()), gradient_plus(model%parameter_count()), &
        gradient_minus(model%parameter_count()))
    parameters = model%parameters()
    call model%log_marginal_likelihood(objective, status)
    call model%hyperparameter_gradient(gradient, status)
    call check(status_ok(status) .and. ieee_is_finite(objective) .and. all(ieee_is_finite(gradient)), &
        "RBF likelihood and analytic gradient", failures)
    do i = 1, model%parameter_count()
        parameters(i) = parameters(i) + h
        call probe%fit(x, orders, y, rbf, 0.041_dp, status, 1.0e-10_dp)
        call probe%set_parameters(parameters, status)
        call probe%log_marginal_likelihood(objective_plus, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call probe%set_parameters(parameters, status)
        call probe%log_marginal_likelihood(objective_minus, status)
        parameters(i) = parameters(i) + h
        call check(status_ok(status) .and. abs(gradient(i) - &
            (objective_plus - objective_minus)/(2.0_dp*h)) < 2.0e-5_dp, &
            "RBF likelihood gradient finite difference", failures)
    end do
    parameters = model%parameters()
    direction = [0.17_dp, -0.11_dp, 0.08_dp, -0.14_dp]
    call model%hyperparameter_hvp([0.07_dp, -0.04_dp, 0.09_dp], hvp, status)
    parameters = model%parameters() + h*[0.07_dp, -0.04_dp, 0.09_dp]
    call probe%fit(x, orders, y, rbf, 0.041_dp, status, 1.0e-10_dp)
    call probe%set_parameters(parameters, status)
    call probe%hyperparameter_gradient(gradient_plus, status)
    parameters = model%parameters() - h*[0.07_dp, -0.04_dp, 0.09_dp]
    call probe%set_parameters(parameters, status)
    call probe%hyperparameter_gradient(gradient_minus, status)
    call check(status_ok(status) .and. maxval(abs(hvp - (gradient_plus - gradient_minus)/(2.0_dp*h))) < &
        2.0e-4_dp, "RBF likelihood HVP finite difference", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, query, query_orders, mean, variance, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "typed CUDA order-three refusal", failures)

    matern52 = make_matern52_kernel(1, 1.35_dp, 0.79_dp, status)
    call matern%fit(x(:4, :), orders(:4), y(:4), matern52, 0.041_dp, status, 1.0e-10_dp)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "Matern order-three refusal", failures)
    call model%fit(x, [0, 1, 2, 4, 0], y, rbf, 0.041_dp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "RBF order-four refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL RBF order-three GP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS RBF order-three GP independent behavioral oracle"

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
        real(dp) :: difference, base

        difference = x1 - x2
        base = variance*exp(-0.5_dp*difference*difference/(lengthscale*lengthscale))
        value = (-1.0_dp)**order2*oracle_distance_derivative(base, difference, lengthscale, &
            order1 + order2)
    end function oracle_cov

    pure real(dp) function oracle_distance_derivative(base, difference, lengthscale, order) &
            result(value)
        real(dp), intent(in) :: base, difference, lengthscale
        integer, intent(in) :: order
        real(dp) :: inv2, inv4, inv6, inv8, inv10, inv12

        inv2 = lengthscale**(-2)
        inv4 = inv2*inv2
        inv6 = inv4*inv2
        inv8 = inv4*inv4
        inv10 = inv8*inv2
        inv12 = inv10*inv2
        select case (order)
        case (0)
            value = base
        case (1)
            value = -difference*inv2*base
        case (2)
            value = (difference**2*inv4 - inv2)*base
        case (3)
            value = (3.0_dp*difference*inv4 - difference**3*inv6)*base
        case (4)
            value = (difference**4*inv8 - 6.0_dp*difference**2*inv6 + 3.0_dp*inv4)*base
        case (5)
            value = (-difference**5*inv10 + 10.0_dp*difference**3*inv8 - &
                15.0_dp*difference*inv6)*base
        case (6)
            value = (difference**6*inv12 - 15.0_dp*difference**4*inv10 + &
                45.0_dp*difference**2*inv8 - 15.0_dp*inv6)*base
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
            write (error_unit, '(a)') "  FAIL [second-derivative-gp-order3] "//description
        end if
    end subroutine check

end program test_second_derivative_gp_rbf_order3
