program test_kernel_operator_device
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernel_operator, only: kernel_operator_t
    use fortml_kernels, only: kernel_add, kernel_multiply, kernel_t, &
        make_constant_kernel, make_linear_kernel, make_matern32_kernel, &
        make_rbf_kernel, make_white_noise_kernel
    use fortnum_krylov, only: KRYLOV_OK
    use fortnum_status, only: FORTNUM_OK, fortnum_status_t
    implicit none

    integer, parameter :: n_samples = 7, n_features = 3, n_rhs = 2
    real(dp), parameter :: variance = 1.3_dp, lengthscale = 0.75_dp
    real(dp), parameter :: diagonal_shift = 0.04_dp
    real(dp) :: points(n_samples, n_features), input(n_samples)
    real(dp) :: output(n_samples), expected(n_samples)
    real(dp) :: inputs(n_samples, n_rhs), outputs(n_samples, n_rhs)
    real(dp) :: expected_matrix(n_samples, n_rhs)
    real(dp) :: composite_expected(n_samples)
    real(dp) :: composite_expected_matrix(n_samples, n_rhs)
    real(dp) :: right_hand_side(n_samples), target_solution(n_samples)
    real(dp) :: solution(n_samples), residual_norm
    real(dp) :: multi_solution(n_samples, n_rhs)
    real(dp) :: multi_residual_norm(n_rhs)
    integer :: multi_info(n_rhs), multi_iterations(n_rhs)
    type(kernel_operator_t) :: rbf_generic
    type(kernel_operator_t) :: composite_operator
    type(kernel_operator_t) :: mixed_operator
    type(kernel_t) :: matern_kernel, linear_kernel, constant_kernel
    type(kernel_t) :: white_noise_kernel, sum_kernel, product_kernel
    type(fortnum_status_t) :: status
    real(dp) :: distance, matern_value, mixed_expected(n_samples)
    integer :: cg_info, cg_iterations, column, feature, i, j, nfail

    nfail = 0
    do feature = 1, n_features
        do i = 1, n_samples
            points(i, feature) = sin( &
                0.17_dp*real(i + 2*feature, dp)) + &
                0.03_dp*cos(0.11_dp*real(i*feature, dp))
        end do
    end do
    input = [0.4_dp, -0.7_dp, 1.1_dp, 0.2_dp, -0.3_dp, 0.8_dp, -1.0_dp]
    do column = 1, n_rhs
        inputs(:, column) = input + 0.2_dp*real(column, dp)
    end do

    call rbf_generic%initialize( &
        points, make_rbf_kernel(n_features, variance, lengthscale, status), &
        diagonal_shift, status, 4)
    call require(status%code == FORTNUM_OK, &
        "generic RBF operator initializes", nfail)
    call require(rbf_generic%device_supported(), &
        "generic RBF operator advertises its fused device backend", nfail)

    do i = 1, n_samples
        expected(i) = diagonal_shift*input(i)
        do j = 1, n_samples
            expected(i) = expected(i) + variance*exp( &
                -0.5_dp*sum((points(i, :) - points(j, :))**2)/ &
                (lengthscale*lengthscale))*input(j)
        end do
    end do
    do column = 1, n_rhs
        do i = 1, n_samples
            expected_matrix(i, column) = diagonal_shift*inputs(i, column)
            do j = 1, n_samples
                expected_matrix(i, column) = expected_matrix(i, column) + &
                    variance*exp( &
                    -0.5_dp*sum((points(i, :) - points(j, :))**2)/ &
                    (lengthscale*lengthscale))*inputs(j, column)
            end do
        end do
    end do

    call rbf_generic%enter_data(status)
    call require(status%code == FORTNUM_OK, &
        "generic RBF points enter persistent device data", nfail)
    !$acc data copyin(input) copyout(output)
    call rbf_generic%matvec_device(input, output, status)
    !$acc end data
    call require(status%code == FORTNUM_OK, &
        "generic RBF device vector product succeeds", nfail)
    call require(maxval(abs(output - expected)) < 3.0e-13_dp, &
        "generic RBF device vector product matches direct oracle", nfail)

    !$acc data copyin(inputs) copyout(outputs)
    call rbf_generic%matmat_device(inputs, outputs, status)
    !$acc end data
    call require(status%code == FORTNUM_OK, &
        "generic RBF device matrix product succeeds", nfail)
    call require(maxval(abs(outputs - expected_matrix)) < 3.0e-13_dp, &
        "generic RBF device matrix product matches direct oracle", nfail)
    call rbf_generic%exit_data(status)
    call require(status%code == FORTNUM_OK, &
        "generic RBF points exit persistent device data", nfail)

    call require(maxval(abs(rbf_generic%diagonal() - &
        (variance + diagonal_shift))) < 3.0e-14_dp, &
        "generic RBF diagonal remains correct", nfail)

    call composite_operator%initialize( &
        points, kernel_add( &
        make_rbf_kernel(n_features, variance, lengthscale, status), &
        make_constant_kernel(n_features, 0.2_dp, status), status), &
        diagonal_shift, status, 4)
    call require(status%code == FORTNUM_OK, &
        "composite host operator initializes", nfail)
    call require(composite_operator%device_supported(), &
        "composite kernel advertises static device lowering", nfail)
    do i = 1, n_samples
        composite_expected(i) = expected(i) + 0.2_dp*sum(input)
    end do
    do column = 1, n_rhs
        composite_expected_matrix(:, column) = expected_matrix(:, column) + &
            0.2_dp*sum(inputs(:, column))
    end do
    call composite_operator%enter_data(status)
    call require(status%code == FORTNUM_OK, &
        "composite kernel enters persistent device data", nfail)
    !$acc data copyin(input) copyout(output)
    call composite_operator%matvec_device(input, output, status)
    !$acc end data
    call require(status%code == FORTNUM_OK, &
        "composite kernel device vector product succeeds", nfail)
    call require(maxval(abs(output - composite_expected)) < 3.0e-13_dp, &
        "composite kernel vector product matches direct oracle", nfail)
    !$acc data copyin(inputs) copyout(outputs)
    call composite_operator%matmat_device(inputs, outputs, status)
    !$acc end data
    call require(status%code == FORTNUM_OK, &
        "composite kernel device matrix product succeeds", nfail)
    call require(maxval(abs(outputs - composite_expected_matrix)) < 3.0e-13_dp, &
        "composite kernel matrix product matches direct oracle", nfail)
    target_solution = input
    right_hand_side = composite_expected
    solution = 0.0_dp
    call composite_operator%solve_cg_device( &
        right_hand_side, solution, 1.0e-12_dp, 50, cg_info, cg_iterations, &
        residual_norm)
    call require(cg_info == KRYLOV_OK, &
        "generic device CG converges", nfail)
    call require(cg_iterations > 0 .and. cg_iterations <= 50, &
        "generic device CG reports bounded iterations", nfail)
    call require(maxval(abs(solution - target_solution)) < 3.0e-11_dp, &
        "generic device CG matches the independent constructed solution", nfail)
    call require(residual_norm < 3.0e-11_dp, &
        "generic device CG reports a true residual", nfail)
    multi_solution = 0.0_dp
    call composite_operator%solve_cg_multi_device( &
        composite_expected_matrix, multi_solution, 1.0e-12_dp, 50, &
        multi_info, multi_iterations, multi_residual_norm)
    call require(all(multi_info == KRYLOV_OK), &
        "generic device multi-RHS CG converges", nfail)
    call require(all(multi_iterations > 0) .and. &
        all(multi_iterations <= 50), &
        "generic device multi-RHS CG reports bounded iterations", nfail)
    call require(maxval(abs(multi_solution - inputs)) < 3.0e-11_dp, &
        "generic device multi-RHS CG matches the independent solutions", nfail)
    call require(maxval(multi_residual_norm) < 3.0e-11_dp, &
        "generic device multi-RHS CG reports true residuals", nfail)
    call composite_operator%exit_data(status)
    call require(status%code == FORTNUM_OK, &
        "composite kernel exits persistent device data", nfail)

    matern_kernel = make_matern32_kernel( &
        n_features, 0.9_dp, 0.6_dp, status)
    linear_kernel = make_linear_kernel(n_features, 0.25_dp, status)
    constant_kernel = make_constant_kernel(n_features, 0.3_dp, status)
    white_noise_kernel = make_white_noise_kernel(n_features, 0.1_dp, status)
    sum_kernel = kernel_add(matern_kernel, linear_kernel, status)
    product_kernel = kernel_multiply(sum_kernel, constant_kernel, status)
    sum_kernel = kernel_add(product_kernel, white_noise_kernel, status)
    call mixed_operator%initialize(points, sum_kernel, diagonal_shift, status, 4)
    call require(status%code == FORTNUM_OK, &
        "nested Matérn/linear/white-noise operator initializes", nfail)
    call require(mixed_operator%device_supported(), &
        "nested common kernel expression advertises device lowering", nfail)
    do i = 1, n_samples
        mixed_expected(i) = diagonal_shift*input(i)
        do j = 1, n_samples
            distance = sqrt(sum((points(i, :) - points(j, :))**2))/0.6_dp
            matern_value = 0.9_dp*(1.0_dp + sqrt(3.0_dp)*distance)*exp( &
                -sqrt(3.0_dp)*distance)
            mixed_expected(i) = mixed_expected(i) + ( &
                0.3_dp*(matern_value + 0.25_dp*sum( &
                points(i, :)*points(j, :))) + &
                merge(0.1_dp, 0.0_dp, i == j))*input(j)
        end do
    end do
    call mixed_operator%enter_data(status)
    call require(status%code == FORTNUM_OK, &
        "nested common kernel enters persistent device data", nfail)
    !$acc data copyin(input) copyout(output)
    call mixed_operator%matvec_device(input, output, status)
    !$acc end data
    call require(status%code == FORTNUM_OK, &
        "nested common kernel device product succeeds", nfail)
    call require(maxval(abs(output - mixed_expected)) < 4.0e-13_dp, &
        "nested common kernel product matches direct oracle", nfail)
    call mixed_operator%exit_data(status)
    call require(status%code == FORTNUM_OK, &
        "nested common kernel exits persistent device data", nfail)

    if (nfail > 0) then
        write (error_unit, '(a,i0)') "FAIL: generic kernel device checks: ", nfail
        error stop 1
    end if
    write (*, '(a)') "PASS: generic kernel device behavioral tests"

contains

    subroutine require(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL: "//description
            failures = failures + 1
        end if
    end subroutine require

end program test_kernel_operator_device
