program test_derivative_gp_kernel_matrix
    !! Independent finite-difference oracle for the derivative-observation
    !! kernel catalog.  This test checks both scalar input derivatives and the
    !! mixed value/first-derivative GP prediction path for every smooth leaf
    !! family that advertises the shared derivative-observation contract.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortnum_kinds, only: dp
    use fortml_kernels, only: kernel_t, make_rbf_kernel, make_rbf_ard_kernel, &
        make_matern32_kernel, make_matern52_kernel, make_periodic_kernel, &
        make_local_periodic_kernel, make_rational_quadratic_kernel, &
        make_cosine_kernel, make_polynomial_kernel, make_spectral_mixture_kernel, &
        make_linear_kernel, make_constant_kernel, make_change_point_kernel, &
        kernel_add, kernel_multiply
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    implicit none

    integer, parameter :: n_features = 2, n_kernels = 14
    real(dp), parameter :: eps = 2.0e-5_dp
    type(kernel_t) :: kernels(n_kernels)
    character(len=32) :: names(n_kernels)
    type(fortnum_status_t) :: status
    integer :: failures, i
    real(dp) :: max_error

    failures = 0
    max_error = 0.0_dp
    call make_catalog(kernels, names, status)
    call check(status_ok(status), "kernel catalog construction", failures)
    if (.not. status_ok(status)) error stop 1

    do i = 1, n_kernels
        call check_kernel(kernels(i), trim(names(i)), max_error, failures)
    end do

    if (failures /= 0) then
        write (error_unit, '(a,i0,a,es12.4)') &
            "FAIL derivative GP kernel matrix: ", failures, " max_error=", max_error
        error stop 1
    end if
    write (*, '(a,i0,a,es12.4)') &
        "PASS derivative GP kernel matrix; kernels=", n_kernels, &
        " max_error=", max_error

contains

    subroutine make_catalog(kernels, names, status)
        type(kernel_t), intent(out) :: kernels(:)
        character(len=*), intent(out) :: names(:)
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: left, right
        real(dp) :: weights(2), means(2, 2), scales(2, 2)

        names = [character(len=32) :: "rbf", "ard_rbf", "matern32", "matern52", &
            "periodic", "local_periodic", "rational_quadratic", "cosine", &
            "polynomial", "spectral_mixture", "linear", "constant", &
            "change_point", "sum_product"]
        kernels(1) = make_rbf_kernel(n_features, 1.3_dp, 0.8_dp, status)
        if (.not. status_ok(status)) return
        kernels(2) = make_rbf_ard_kernel(n_features, 1.3_dp, [0.75_dp, 1.25_dp], status)
        if (.not. status_ok(status)) return
        kernels(3) = make_matern32_kernel(n_features, 1.3_dp, 0.8_dp, status)
        if (.not. status_ok(status)) return
        kernels(4) = make_matern52_kernel(n_features, 1.3_dp, 0.8_dp, status)
        if (.not. status_ok(status)) return
        kernels(5) = make_periodic_kernel(n_features, 1.3_dp, 0.8_dp, 2.1_dp, status)
        if (.not. status_ok(status)) return
        kernels(6) = make_local_periodic_kernel(n_features, 1.3_dp, 0.8_dp, &
            0.9_dp, 2.1_dp, status)
        if (.not. status_ok(status)) return
        kernels(7) = make_rational_quadratic_kernel(n_features, 1.3_dp, 0.8_dp, &
            1.7_dp, status)
        if (.not. status_ok(status)) return
        kernels(8) = make_cosine_kernel(n_features, 1.3_dp, 0.8_dp, status)
        if (.not. status_ok(status)) return
        kernels(9) = make_polynomial_kernel(n_features, 1.3_dp, 0.4_dp, 1.5_dp, &
            2.2_dp, status)
        if (.not. status_ok(status)) return
        weights = [1.15_dp, 0.63_dp]
        means = reshape([0.21_dp, -0.37_dp, 0.48_dp, 0.16_dp], shape(means))
        scales = reshape([0.31_dp, 0.57_dp, 0.22_dp, 0.44_dp], shape(scales))
        kernels(10) = make_spectral_mixture_kernel(n_features, 2, weights, means, scales, status)
        if (.not. status_ok(status)) return
        kernels(11) = make_linear_kernel(n_features, 1.3_dp, status)
        if (.not. status_ok(status)) return
        kernels(12) = make_constant_kernel(n_features, 1.3_dp, status)
        if (.not. status_ok(status)) return
        left = make_rbf_kernel(n_features, 1.1_dp, 0.7_dp, status)
        if (.not. status_ok(status)) return
        right = make_constant_kernel(n_features, 0.4_dp, status)
        if (.not. status_ok(status)) return
        kernels(13) = make_change_point_kernel(left, right, 1, 0.15_dp, 0.8_dp, status)
        if (.not. status_ok(status)) return
        kernels(14) = kernel_add(left, kernel_multiply(left, right, status), status)
    end subroutine make_catalog

    subroutine check_kernel(kernel, name, max_error, failures)
        type(kernel_t), intent(in) :: kernel
        character(len=*), intent(in) :: name
        real(dp), intent(inout) :: max_error
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(fortnum_status_t) :: local_status
        real(dp) :: x1(n_features), x2(n_features), value
        real(dp) :: gradient_x1(n_features), gradient_x2(n_features)
        real(dp) :: mixed_hessian(n_features, n_features)
        real(dp) :: plus, minus, expected, error
        real(dp) :: x_train(3, n_features), y_train(3, 1)
        real(dp) :: query(2, n_features), mean(2, 1), variance(2)
        integer :: components(3), query_components(2), i, j

        x1 = [-0.73_dp, 0.41_dp]
        x2 = [0.37_dp, -0.29_dp]
        call kernel%input_derivatives(x1, x2, value, gradient_x1, gradient_x2, &
            mixed_hessian, local_status)
        call check(status_ok(local_status), trim(name)//" analytic input derivatives", failures)
        if (.not. status_ok(local_status)) return
        call check(ieee_is_finite(value) .and. all(ieee_is_finite(gradient_x1)) .and. &
            all(ieee_is_finite(gradient_x2)) .and. all(ieee_is_finite(mixed_hessian)), &
            trim(name)//" finite derivatives", failures)

        do i = 1, n_features
            plus = x1(i)
            x1(i) = plus + eps
            expected = kernel%value(x1, x2)
            x1(i) = plus - eps
            minus = kernel%value(x1, x2)
            x1(i) = plus
            error = abs((expected - minus)/(2.0_dp*eps) - gradient_x1(i))
            max_error = max(max_error, error)
            call check(error < 3.0e-4_dp, trim(name)//" x1 finite-difference gradient", failures)
        end do
        do j = 1, n_features
            plus = x2(j)
            x2(j) = plus + eps
            expected = kernel%value(x1, x2)
            x2(j) = plus - eps
            minus = kernel%value(x1, x2)
            x2(j) = plus
            error = abs((expected - minus)/(2.0_dp*eps) - gradient_x2(j))
            max_error = max(max_error, error)
            call check(error < 3.0e-4_dp, trim(name)//" x2 finite-difference gradient", failures)
        end do
        do i = 1, n_features
            do j = 1, n_features
                plus = x1(i)
                x1(i) = plus + eps
                x2(j) = x2(j) + eps
                expected = kernel%value(x1, x2)
                x2(j) = x2(j) - 2.0_dp*eps
                expected = expected - kernel%value(x1, x2)
                x2(j) = x2(j) + eps
                x1(i) = plus - eps
                x2(j) = x2(j) + eps
                minus = kernel%value(x1, x2)
                x2(j) = x2(j) - 2.0_dp*eps
                minus = minus - kernel%value(x1, x2)
                x2(j) = x2(j) + eps
                x1(i) = plus
                error = abs((expected - minus)/(4.0_dp*eps*eps) - mixed_hessian(i, j))
                max_error = max(max_error, error)
                call check(error < 4.0e-2_dp, trim(name)//" mixed Hessian", failures)
            end do
        end do

        x_train = reshape([-0.73_dp, 0.41_dp, 0.12_dp, -0.18_dp, 0.77_dp, 0.26_dp], &
            shape(x_train))
        y_train(:, 1) = [0.2_dp, -0.1_dp, 0.4_dp]
        components = [0, 1, 0]
        query = reshape([-0.21_dp, 0.31_dp, 0.53_dp, -0.11_dp], shape(query))
        query_components = [1, 0]
        call model%fit(x_train, components, y_train, kernel, 0.05_dp, local_status)
        call check(status_ok(local_status), trim(name)//" mixed-observation fit", failures)
        if (.not. status_ok(local_status)) return
        call model%predict(query, query_components, mean, variance, local_status)
        call check(status_ok(local_status), trim(name)//" mixed-observation prediction", failures)
        call check(all(ieee_is_finite(mean)) .and. all(ieee_is_finite(variance)), &
            trim(name)//" finite mixed-observation prediction", failures)
    end subroutine check_kernel

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [derivative-gp-kernel-matrix] "//description
        end if
    end subroutine check

end program test_derivative_gp_kernel_matrix
