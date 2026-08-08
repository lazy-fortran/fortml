program test_multi_output_gp_products
    !! Independent finite-difference and adjoint oracles for multi-output GP
    !! query and packed fitted-parameter products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_multi_output_gp, only: multi_output_gp_t
    implicit none
    integer, parameter :: n = 4, m = 3, d = 1, p = 2, rank = 1
    real(dp), parameter :: eps = 2.0e-6_dp
    real(dp) :: x(n, d), y(n, p), query(m, d), query_direction(m, d)
    real(dp) :: weights(p, rank), independent(p), noise
    type(multi_output_gp_t) :: model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    real(dp) :: mean(m, p), mean_dot(m, p), mean_plus(m, p), mean_minus(m, p)
    real(dp) :: mean_bar(m, p), query_bar(m, d)
    real(dp) :: covariance(n*p, n*p), covariance_dot(n*p, n*p)
    real(dp) :: covariance_plus(n*p, n*p), covariance_minus(n*p, n*p)
    real(dp) :: covariance_bar(n*p, n*p)
    real(dp), allocatable :: theta(:), theta_plus(:), theta_minus(:), parameter_bar(:)
    real(dp) :: product_left, product_right, max_error
    type(fortml_device_t) :: cuda
    integer :: failures, i, nparameters, parameter_index

    do i = 1, n
        x(i, 1) = -0.7_dp + 0.4_dp*real(i - 1, dp)
        y(i, 1) = sin(1.1_dp*x(i, 1))
        y(i, 2) = cos(0.8_dp*x(i, 1)) - 0.1_dp
    end do
    do i = 1, m
        query(i, 1) = -0.5_dp + 0.35_dp*real(i - 1, dp)
        query_direction(i, 1) = 0.2_dp - 0.07_dp*real(i - 1, dp)
    end do
    mean_bar = reshape([0.3_dp, -0.2_dp, 0.7_dp, -0.4_dp, 0.1_dp, 0.6_dp], [m, p])
    weights(:, 1) = [0.8_dp, -0.45_dp]
    independent = [0.25_dp, 0.35_dp]
    noise = 0.12_dp
    kernel = make_rbf_kernel(d, 1.2_dp, 0.65_dp, status)
    call model%initialize(kernel, weights, independent, noise, status)
    call model%fit(x, y, status)
    failures = 0
    call check(status_ok(status), "fit", failures)

    call model%predict_input_jvp(query, query_direction, mean, mean_dot, status)
    call check(status_ok(status), "input JVP status", failures)
    call model%predict_input_vjp(query, mean_bar, query_bar, status)
    call check(status_ok(status), "input VJP status", failures)
    call model%predict(query, mean_plus, status)
    do i = 1, m
        query(i, 1) = query(i, 1) + eps*query_direction(i, 1)
    end do
    call model%predict(query, mean_plus, status)
    do i = 1, m
        query(i, 1) = query(i, 1) - 2.0_dp*eps*query_direction(i, 1)
    end do
    call model%predict(query, mean_minus, status)
    query(:, 1) = query(:, 1) + eps*query_direction(:, 1)
    max_error = maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*eps)))
    call check(max_error < 3.0e-5_dp, "input JVP finite difference", failures)
    ! Recompute the adjoint identity with a second deterministic direction.
    query_direction(:, 1) = [0.11_dp, -0.23_dp, 0.31_dp]
    call model%predict_input_jvp(query, query_direction, mean, mean_dot, status)
    product_left = sum(mean_bar*mean_dot)
    product_right = sum(query_bar*query_direction)
    call check(abs(product_left - product_right) < 3.0e-10_dp, &
        "input VJP adjoint", failures)

    nparameters = model%parameter_count()
    allocate(theta(nparameters), theta_plus(nparameters), theta_minus(nparameters), &
        parameter_bar(nparameters))
    theta = model%parameters()
    theta_plus = theta
    theta_minus = theta
    do parameter_index = 1, nparameters
        theta_plus = theta
        theta_minus = theta
        theta_plus(parameter_index) = theta_plus(parameter_index) + eps
        theta_minus(parameter_index) = theta_minus(parameter_index) - eps
        call finite_difference_prediction(theta_plus, mean_plus, status)
        call finite_difference_prediction(theta_minus, mean_minus, status)
        call model%predict_parameter_jvp(query, unit_direction(theta, parameter_index), &
            mean, mean_dot, status)
        call check(status_ok(status), "parameter JVP status", failures)
        max_error = maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*eps)))
        call check(max_error < 2.0e-4_dp, "parameter JVP finite difference", failures)
    end do
    mean_bar = reshape([0.4_dp, -0.1_dp, 0.6_dp, 0.2_dp, -0.3_dp, 0.5_dp], [m, p])
    call model%predict_parameter_vjp(query, mean_bar, parameter_bar, status)
    call check(status_ok(status), "parameter VJP status", failures)
    call model%predict_parameter_jvp(query, theta_direction(theta), mean, mean_dot, status)
    product_left = sum(mean_bar*mean_dot)
    product_right = dot_product(parameter_bar, theta_direction(theta))
    call check(abs(product_left - product_right) < 2.0e-9_dp, &
        "parameter VJP adjoint", failures)

    covariance_bar = 0.0_dp
    do i = 1, n*p
        do parameter_index = 1, n*p
            covariance_bar(i, parameter_index) = &
                0.11_dp*sin(0.23_dp*real(i + 2*parameter_index, dp))
        end do
    end do
    call model%joint_covariance_parameter_jvp(x, theta_direction(theta), covariance, &
        covariance_dot, status)
    call check(status_ok(status), "covariance parameter JVP status", failures)
    call finite_difference_covariance(theta + eps*theta_direction(theta), covariance_plus, status)
    call finite_difference_covariance(theta - eps*theta_direction(theta), covariance_minus, status)
    max_error = maxval(abs(covariance_dot - (covariance_plus - covariance_minus)/(2.0_dp*eps)))
    call check(max_error < 2.0e-8_dp, "covariance parameter JVP finite difference", failures)
    call model%joint_covariance_parameter_vjp(x, covariance_bar, parameter_bar, status)
    call check(status_ok(status), "covariance parameter VJP status", failures)
    product_left = sum(covariance_bar*covariance_dot)
    product_right = dot_product(parameter_bar, theta_direction(theta))
    call check(abs(product_left - product_right) < 2.0e-9_dp, &
        "covariance parameter VJP adjoint", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    covariance = 4321.0_dp
    covariance_dot = 8765.0_dp
    call model%predict_input_jvp_device(cuda, query, query_direction, mean, mean_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA input JVP refusal", failures)
    call model%predict_parameter_vjp_device(cuda, query, mean_bar, parameter_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA parameter VJP refusal", failures)
    call model%joint_covariance_parameter_jvp_device(cuda, x, theta_direction(theta), &
        covariance, covariance_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA covariance JVP refusal", failures)
    call check(all(covariance == 4321.0_dp) .and. all(covariance_dot == 8765.0_dp), &
        "CUDA covariance JVP refusal leaves outputs untouched", failures)
    parameter_bar = 2468.0_dp
    call model%joint_covariance_parameter_vjp_device(cuda, x, covariance_bar, parameter_bar, &
        status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA covariance VJP refusal", failures)
    call check(all(parameter_bar == 2468.0_dp), &
        "CUDA covariance VJP refusal leaves output untouched", failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " multi-output product test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures
        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//label//"]"
            failures = failures + 1
        end if
    end subroutine check

    function unit_direction(theta, index) result(direction)
        real(dp), intent(in) :: theta(:)
        integer, intent(in) :: index
        real(dp) :: direction(size(theta))
        direction = 0.0_dp
        direction(index) = 1.0_dp
    end function unit_direction

    function theta_direction(theta) result(direction)
        real(dp), intent(in) :: theta(:)
        real(dp) :: direction(size(theta))
        integer :: j
        do j = 1, size(theta)
            direction(j) = 0.07_dp*sin(0.31_dp*real(j, dp))
        end do
    end function theta_direction

    subroutine finite_difference_prediction(parameters, value, status)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: candidate_kernel
        real(dp) :: candidate_weights(p, rank), candidate_independent(p), candidate_noise
        real(dp) :: ktrain(n, n), kcross(n, m), coreg(p, p)
        real(dp) :: matrix(n*p, n*p), rhs(n*p), alpha(n*p)
        integer :: kc, index, a, b, ell, i, j

        kc = kernel%parameter_count()
        candidate_kernel = kernel
        call candidate_kernel%set_parameters(parameters(:kc), status)
        if (.not. status_ok(status)) return
        candidate_noise = exp(parameters(kc + 1))
        index = kc + 2
        do a = 1, p
            do ell = 1, rank
                candidate_weights(a, ell) = parameters(index)
                index = index + 1
            end do
        end do
        candidate_independent = parameters(index:index + p - 1)
        call candidate_kernel%matrix(x, x, ktrain, status)
        if (.not. status_ok(status)) return
        call candidate_kernel%matrix(x, query, kcross, status)
        if (.not. status_ok(status)) return
        do b = 1, p
            do a = 1, p
                coreg(a, b) = sum(candidate_weights(a, :)*candidate_weights(b, :))
            end do
            coreg(b, b) = coreg(b, b) + candidate_independent(b)
        end do
        matrix = 0.0_dp
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, n
                        matrix((a - 1)*n + i, (b - 1)*n + j) = coreg(a, b)*ktrain(i, j)
                    end do
                end do
            end do
        end do
        do a = 1, p
            do i = 1, n
                matrix((a - 1)*n + i, (a - 1)*n + i) = &
                    matrix((a - 1)*n + i, (a - 1)*n + i) + candidate_noise
                rhs((a - 1)*n + i) = y(i, a)
            end do
        end do
        call dense_solve(matrix, rhs, alpha)
        value = 0.0_dp
        do a = 1, p
            do i = 1, m
                do b = 1, p
                    do j = 1, n
                        value(i, a) = value(i, a) + coreg(a, b)*kcross(j, i)* &
                            alpha((b - 1)*n + j)
                    end do
                end do
            end do
        end do
    end subroutine finite_difference_prediction

    subroutine finite_difference_covariance(parameters, value, status)
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: candidate_kernel
        real(dp) :: candidate_weights(p, rank), candidate_independent(p)
        real(dp) :: ktrain(n, n), coreg(p, p)
        integer :: kc, index, a, b, ell, i, j

        value = 0.0_dp
        kc = kernel%parameter_count()
        if (size(parameters) /= model%parameter_count() .or. &
            any(shape(value) /= [n*p, n*p])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "covariance oracle shape is invalid")
            return
        end if
        candidate_kernel = kernel
        call candidate_kernel%set_parameters(parameters(:kc), status)
        if (.not. status_ok(status)) return
        index = kc + 2
        do a = 1, p
            do ell = 1, rank
                candidate_weights(a, ell) = parameters(index)
                index = index + 1
            end do
        end do
        candidate_independent = parameters(index:index + p - 1)
        call candidate_kernel%matrix(x, x, ktrain, status)
        if (.not. status_ok(status)) return
        do b = 1, p
            do a = 1, p
                coreg(a, b) = sum(candidate_weights(a, :)*candidate_weights(b, :))
            end do
            coreg(b, b) = coreg(b, b) + candidate_independent(b)
        end do
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, n
                        value((a - 1)*n + i, (b - 1)*n + j) = coreg(a, b)*ktrain(i, j)
                    end do
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine finite_difference_covariance

    subroutine dense_solve(matrix, rhs, solution)
        real(dp), intent(in) :: matrix(:, :), rhs(:)
        real(dp), intent(out) :: solution(:)
        real(dp), allocatable :: a(:, :), b(:)
        real(dp) :: factor, swap
        integer :: i, j, k, pivot, dimension

        dimension = size(rhs)
        allocate(a, source=matrix)
        allocate(b, source=rhs)
        do k = 1, dimension - 1
            pivot = k
            do i = k + 1, dimension
                if (abs(a(i, k)) > abs(a(pivot, k))) pivot = i
            end do
            if (pivot /= k) then
                do j = k, dimension
                    swap = a(k, j)
                    a(k, j) = a(pivot, j)
                    a(pivot, j) = swap
                end do
                swap = b(k)
                b(k) = b(pivot)
                b(pivot) = swap
            end if
            do i = k + 1, dimension
                factor = a(i, k)/a(k, k)
                a(i, k:) = a(i, k:) - factor*a(k, k:)
                b(i) = b(i) - factor*b(k)
            end do
        end do
        solution(dimension) = b(dimension)/a(dimension, dimension)
        do i = dimension - 1, 1, -1
            solution(i) = (b(i) - sum(a(i, i + 1:)*solution(i + 1:)))/a(i, i)
        end do
    end subroutine dense_solve

end program test_multi_output_gp_products
