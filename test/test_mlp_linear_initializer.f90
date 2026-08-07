program test_mlp_linear_initializer
    !! Independent affine oracle for the public two-layer linear initializer.
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp, only: mlp_t, MLP_LINEAR, MLP_TANH
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_affine_initializer(failures)
    call test_refusals_and_transaction(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " MLP linear-initializer test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_affine_initializer(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: weight1(2, 3), bias1(3), weight2(3, 2), bias2(2)
        real(dp) :: x(4, 2), y(4, 2), expected(4, 2), theta(17), packed(17)
        real(dp) :: updated_weight1(2, 3), updated_bias1(3)
        real(dp) :: updated_weight2(3, 2), updated_bias2(2)

        weight1 = reshape([ &
            0.4_dp, -0.7_dp, 0.2_dp, 0.9_dp, -0.1_dp, 0.3_dp], shape(weight1))
        bias1 = [0.15_dp, -0.25_dp, 0.35_dp]
        weight2 = reshape([ &
            -0.6_dp, 0.8_dp, 0.5_dp, 0.2_dp, -0.4_dp, 0.7_dp], shape(weight2))
        bias2 = [-0.45_dp, 0.65_dp]
        x = reshape([ &
            -1.0_dp, 0.5_dp, 0.3_dp, 1.2_dp, &
             0.7_dp, -0.9_dp, 1.1_dp, -0.2_dp], shape(x))

        call model%initialize_linear(weight1, bias1, weight2, bias2, status)
        call check(status_ok(status), "linear initializer status", failures)
        call check(model%hidden_activation == MLP_LINEAR .and. &
            model%output_activation == MLP_LINEAR, &
            "linear initializer activations", failures)
        call model%predict(x, y, status)
        expected = matmul(matmul(x, weight1) + spread(bias1, dim=1, ncopies=size(x, 1)), &
            weight2) + spread(bias2, dim=1, ncopies=size(x, 1))
        call check(status_ok(status) .and. maxval(abs(y - expected)) < 2.0e-14_dp, &
            "affine prediction oracle", failures)

        theta = model%parameters()
        packed = [reshape(weight1, [size(weight1)]), bias1, &
            reshape(weight2, [size(weight2)]), bias2]
        call check(size(theta) == 17 .and. maxval(abs(theta - packed)) < 2.0e-14_dp, &
            "column-major packed layout", failures)

        updated_weight1 = 0.5_dp*weight1
        updated_bias1 = bias1 + 0.4_dp
        updated_weight2 = -0.3_dp*weight2
        updated_bias2 = bias2 - 0.2_dp
        call model%set_linear_parameters(updated_weight1, updated_bias1, &
            updated_weight2, updated_bias2, status)
        call check(status_ok(status), "linear setter status", failures)
        theta = model%parameters()
        packed = [reshape(updated_weight1, [size(updated_weight1)]), updated_bias1, &
            reshape(updated_weight2, [size(updated_weight2)]), updated_bias2]
        call check(maxval(abs(theta - packed)) < 2.0e-14_dp, &
            "linear setter packed state", failures)
    end subroutine test_affine_initializer

    subroutine test_refusals_and_transaction(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model, uninitialized, nonlinear
        type(fortnum_status_t) :: status
        real(dp) :: weight1(2, 3), bias1(3), weight2(3, 2), bias2(2)
        real(dp) :: bad_weight1(2, 3), bad_weight2(2, 2), before(17), after(17)

        weight1 = 0.2_dp
        bias1 = [-0.1_dp, 0.0_dp, 0.1_dp]
        weight2 = -0.3_dp
        bias2 = [0.4_dp, -0.5_dp]
        call model%initialize_linear(weight1, bias1, weight2, bias2, status)
        before = model%parameters()

        bad_weight1 = weight1
        bad_weight1(1, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
        call model%set_linear_parameters(bad_weight1, bias1, weight2, bias2, status)
        after = model%parameters()
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "nonfinite setter refusal", failures)
        call check(maxval(abs(after - before)) == 0.0_dp, &
            "nonfinite setter transaction", failures)

        bad_weight2 = weight2(1:2, :)
        call model%set_linear_parameters(weight1, bias1, bad_weight2, bias2, status)
        after = model%parameters()
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "dimension setter refusal", failures)
        call check(maxval(abs(after - before)) == 0.0_dp, &
            "dimension setter transaction", failures)

        call nonlinear%initialize([2, 3, 2], status, hidden_activation=MLP_TANH, &
            output_activation=MLP_LINEAR)
        call nonlinear%set_linear_parameters(weight1, bias1, weight2, bias2, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "nonlinear topology setter refusal", failures)

        call uninitialized%set_linear_parameters(weight1, bias1, weight2, bias2, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "uninitialized setter refusal", failures)

        bad_weight1 = weight1
        bad_weight1(1, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
        call uninitialized%initialize_linear(bad_weight1, bias1, weight2, bias2, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "nonfinite initializer refusal", failures)
    end subroutine test_refusals_and_transaction

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [" // trim(label) // "]"
        end if
    end subroutine check

end program test_mlp_linear_initializer
