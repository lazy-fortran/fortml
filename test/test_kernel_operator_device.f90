program test_kernel_operator_device
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernel_operator, only: kernel_operator_t
    use fortml_kernels, only: kernel_add, make_constant_kernel, &
        make_rbf_kernel
    use fortnum_status, only: FORTNUM_DOMAIN_ERROR, FORTNUM_OK, &
        fortnum_status_t
    implicit none

    integer, parameter :: n_samples = 7, n_features = 3, n_rhs = 2
    real(dp), parameter :: variance = 1.3_dp, lengthscale = 0.75_dp
    real(dp), parameter :: diagonal_shift = 0.04_dp
    real(dp) :: points(n_samples, n_features), input(n_samples)
    real(dp) :: output(n_samples), expected(n_samples)
    real(dp) :: inputs(n_samples, n_rhs), outputs(n_samples, n_rhs)
    real(dp) :: expected_matrix(n_samples, n_rhs)
    type(kernel_operator_t) :: rbf_generic
    type(kernel_operator_t) :: composite_operator
    type(fortnum_status_t) :: status
    integer :: column, feature, i, j, nfail

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
    call require(.not. composite_operator%device_supported(), &
        "composite kernel stays on the validated host backend", nfail)
    call composite_operator%enter_data(status)
    call require(status%code == FORTNUM_DOMAIN_ERROR, &
        "composite kernel rejects unsupported device lowering", nfail)

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
