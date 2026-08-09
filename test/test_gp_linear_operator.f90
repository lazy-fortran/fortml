program test_gp_linear_operator
    !! Independent dense oracle for registered first-order GP operators.
    !! The oracle below uses closed-form one-dimensional RBF value, gradient,
    !! and mixed-Hessian blocks; it does not call the production covariance
    !! helper.  It also checks the operator-coefficient adjoint identity and
    !! the explicit CUDA refusal boundary.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_operator_gaussian_process, only: &
        linear_differential_operator_registry_t, gp_linear_operator_regression_t
    implicit none

    type(linear_differential_operator_registry_t) :: train_ops, query_ops
    type(linear_differential_operator_registry_t) :: plus_ops, minus_ops
    type(gp_linear_operator_regression_t) :: model
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(3, 1), y(3, 1), query(3, 1)
    real(dp) :: mean(3, 1), variance(3), covariance(3, 3)
    real(dp) :: oracle_mean(3, 1), oracle_variance(3), oracle_covariance_matrix(3, 3)
    real(dp) :: direction(2, 3), mean_dot(3, 1), variance_dot(3)
    real(dp) :: fd_mean_dot(3, 1), fd_variance_dot(3), h
    real(dp) :: mean_bar(3, 1), variance_bar(3), operator_bar(2, 3)
    real(dp) :: lhs, rhs
    integer :: failures

    failures = 0
    x(:, 1) = [-0.45_dp, 0.15_dp, 0.9_dp]
    y(:, 1) = [0.6_dp, -0.25_dp, 0.8_dp]
    query(:, 1) = [-0.1_dp, 0.35_dp, 0.65_dp]
    call train_ops%initialize(1, 3, status)
    call check(status_ok(status), "training operator registry initializes", failures)
    call train_ops%set_operator(1, "value", [1.0_dp, 0.0_dp], status)
    call train_ops%set_operator(2, "gradient", [0.0_dp, 1.0_dp], status)
    call train_ops%set_operator(3, "robin", [0.7_dp, -0.4_dp], status)
    call check(status_ok(status), "training operators register by name", failures)
    call query_ops%initialize(1, 3, status)
    call query_ops%set_operator(1, "value", [1.0_dp, 0.0_dp], status)
    call query_ops%set_operator(2, "gradient", [0.0_dp, 1.0_dp], status)
    call query_ops%set_operator(3, "robin", [0.7_dp, -0.4_dp], status)
    kernel = make_rbf_kernel(1, 1.25_dp, 0.82_dp, status)
    call model%fit(x, train_ops, y, kernel, 0.06_dp, status, jitter=1.0e-11_dp)
    call check(status_ok(status), "registered-operator GP fit", failures)
    call check(model%observation_count() == 3 .and. model%parameter_count() == 3, &
        "operator GP state metadata", failures)

    call model%predict(query, query_ops, mean, variance, status)
    call oracle_predict(x, train_ops%coefficients, y, query, query_ops%coefficients, &
        1.25_dp, 0.82_dp, 0.06_dp, 1.0e-11_dp, oracle_mean, oracle_variance)
    call check(status_ok(status) .and. maxval(abs(mean - oracle_mean)) < 2.0e-11_dp, &
        "operator posterior means match independent oracle", failures)
    call check(status_ok(status) .and. maxval(abs(variance - oracle_variance)) < 2.0e-11_dp, &
        "operator posterior variances match independent oracle", failures)

    call model%joint_covariance(query, query_ops, covariance, status)
    call oracle_joint(x, train_ops%coefficients, query, query_ops%coefficients, &
        1.25_dp, 0.82_dp, 0.06_dp, 1.0e-11_dp, oracle_covariance_matrix)
    call check(status_ok(status) .and. maxval(abs(covariance - oracle_covariance_matrix)) < 2.0e-11_dp, &
        "operator posterior covariance matches independent oracle", failures)

    direction = reshape([0.09_dp, -0.05_dp, 0.04_dp, 0.03_dp, -0.08_dp, 0.06_dp], [2, 3])
    call model%predict_operator_jvp(query, query_ops, direction, mean, mean_dot, variance, &
        variance_dot, status)
    h = 2.0e-6_dp
    plus_ops = query_ops
    minus_ops = query_ops
    plus_ops%coefficients = query_ops%coefficients + h*direction
    minus_ops%coefficients = query_ops%coefficients - h*direction
    call model%predict(query, plus_ops, oracle_mean, oracle_variance, status)
    call model%predict(query, minus_ops, fd_mean_dot, fd_variance_dot, status)
    fd_mean_dot = (oracle_mean - fd_mean_dot)/(2.0_dp*h)
    fd_variance_dot = (oracle_variance - fd_variance_dot)/(2.0_dp*h)
    call check(status_ok(status) .and. maxval(abs(mean_dot - fd_mean_dot)) < 3.0e-7_dp, &
        "operator coefficient JVP mean finite difference", failures)
    call check(status_ok(status) .and. maxval(abs(variance_dot - fd_variance_dot)) < 3.0e-7_dp, &
        "operator coefficient JVP variance finite difference", failures)

    mean_bar(:, 1) = [0.3_dp, -0.2_dp, 0.4_dp]
    variance_bar = [0.25_dp, -0.1_dp, 0.18_dp]
    call model%predict_operator_vjp(query, query_ops, mean_bar, variance_bar, operator_bar, status)
    lhs = sum(mean_bar*mean_dot) + dot_product(variance_bar, variance_dot)
    rhs = sum(operator_bar*direction)
    call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-10_dp, &
        "operator coefficient JVP/VJP adjoint identity", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    mean = 1234.0_dp
    variance = 5678.0_dp
    call model%predict_device(cuda, query, query_ops, mean, variance, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "operator GP CUDA prediction is typed refusal", failures)
    call check(all(mean == 1234.0_dp) .and. all(variance == 5678.0_dp), &
        "operator GP CUDA prediction refusal leaves outputs untouched", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL registered operator GP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS registered linear-operator GP independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [operator-gp] "//description
        end if
    end subroutine check

    subroutine oracle_predict(x_train, train_weights, y_train, x_query, query_weights, &
            variance, lengthscale, noise, jitter, mean, posterior_variance)
        real(dp), intent(in) :: x_train(:, :), train_weights(:, :), y_train(:, :)
        real(dp), intent(in) :: x_query(:, :), query_weights(:, :)
        real(dp), intent(in) :: variance, lengthscale, noise, jitter
        real(dp), intent(out) :: mean(:, :), posterior_variance(:)
        real(dp) :: matrix(size(x_train, 1), size(x_train, 1))
        real(dp) :: cross(size(x_train, 1), size(x_query, 1))
        real(dp) :: rhs(size(x_train, 1), 1), solved(size(x_train, 1), 1)
        real(dp) :: prior, work(size(x_train, 1), 1)
        integer :: i, j

        do j = 1, size(x_train, 1)
            do i = 1, size(x_train, 1)
                matrix(i, j) = oracle_covariance(x_train(i, 1), train_weights(:, i), &
                    x_train(j, 1), train_weights(:, j), variance, lengthscale)
            end do
            matrix(j, j) = matrix(j, j) + noise + jitter
        end do
        rhs(:, 1) = y_train(:, 1)
        call oracle_spd_solve(matrix, rhs, solved)
        do j = 1, size(x_query, 1)
            do i = 1, size(x_train, 1)
                cross(i, j) = oracle_covariance(x_train(i, 1), train_weights(:, i), &
                    x_query(j, 1), query_weights(:, j), variance, lengthscale)
            end do
            mean(j, 1) = dot_product(cross(:, j), solved(:, 1))
            work(:, 1) = cross(:, j)
            call oracle_spd_solve(matrix, work, work)
            prior = oracle_covariance(x_query(j, 1), query_weights(:, j), x_query(j, 1), &
                query_weights(:, j), variance, lengthscale)
            posterior_variance(j) = prior - dot_product(cross(:, j), work(:, 1))
        end do
    end subroutine oracle_predict

    subroutine oracle_joint(x_train, train_weights, x_query, query_weights, variance, &
            lengthscale, noise, jitter, covariance)
        real(dp), intent(in) :: x_train(:, :), train_weights(:, :), x_query(:, :), query_weights(:, :)
        real(dp), intent(in) :: variance, lengthscale, noise, jitter
        real(dp), intent(out) :: covariance(:, :)
        real(dp) :: matrix(size(x_train, 1), size(x_train, 1))
        real(dp) :: cross(size(x_train, 1), size(x_query, 1))
        real(dp) :: prior(size(x_query, 1), size(x_query, 1))
        real(dp) :: work(size(x_train, 1), size(x_query, 1))
        integer :: i, j

        do j = 1, size(x_train, 1)
            do i = 1, size(x_train, 1)
                matrix(i, j) = oracle_covariance(x_train(i, 1), train_weights(:, i), &
                    x_train(j, 1), train_weights(:, j), variance, lengthscale)
            end do
            matrix(j, j) = matrix(j, j) + noise + jitter
        end do
        do j = 1, size(x_query, 1)
            do i = 1, size(x_train, 1)
                cross(i, j) = oracle_covariance(x_train(i, 1), train_weights(:, i), &
                    x_query(j, 1), query_weights(:, j), variance, lengthscale)
            end do
            do i = 1, size(x_query, 1)
                prior(i, j) = oracle_covariance(x_query(i, 1), query_weights(:, i), &
                    x_query(j, 1), query_weights(:, j), variance, lengthscale)
            end do
        end do
        work = cross
        call oracle_spd_solve(matrix, work, work)
        covariance = prior - matmul(transpose(cross), work)
    end subroutine oracle_joint

    real(dp) function oracle_covariance(x1, weights1, x2, weights2, variance, lengthscale)
        real(dp), intent(in) :: x1, weights1(:), x2, weights2(:), variance, lengthscale
        real(dp) :: difference, kernel_value, grad1, grad2, mixed
        difference = x1 - x2
        kernel_value = variance*exp(-0.5_dp*difference*difference/(lengthscale*lengthscale))
        grad1 = -kernel_value*difference/(lengthscale*lengthscale)
        grad2 = -grad1
        mixed = kernel_value*(1.0_dp/(lengthscale*lengthscale) - &
            difference*difference/(lengthscale**4))
        oracle_covariance = weights1(1)*weights2(1)*kernel_value + &
            weights1(2)*weights2(1)*grad1 + weights1(1)*weights2(2)*grad2 + &
            weights1(2)*weights2(2)*mixed
    end function oracle_covariance

    subroutine oracle_spd_solve(matrix, rhs, solution)
        real(dp), intent(in) :: matrix(:, :), rhs(:, :)
        real(dp), intent(out) :: solution(:, :)
        real(dp) :: lower(size(matrix, 1), size(matrix, 1))
        real(dp) :: work(size(rhs, 1), size(rhs, 2))
        integer :: i, j, k, n, r

        n = size(matrix, 1)
        lower = 0.0_dp
        do i = 1, n
            do j = 1, i
                lower(i, j) = matrix(i, j) - sum(lower(i, :j - 1)*lower(j, :j - 1))
                if (i == j) then
                    lower(i, j) = sqrt(lower(i, j))
                else
                    lower(i, j) = lower(i, j)/lower(j, j)
                end if
            end do
        end do
        work = rhs
        do r = 1, size(rhs, 2)
            do i = 1, n
                work(i, r) = (work(i, r) - sum(lower(i, :i - 1)*work(:i - 1, r)))/lower(i, i)
            end do
            do i = n, 1, -1
                work(i, r) = (work(i, r) - sum(lower(i + 1:, i)*work(i + 1:, r)))/lower(i, i)
            end do
        end do
        solution = work
    end subroutine oracle_spd_solve

end program test_gp_linear_operator
