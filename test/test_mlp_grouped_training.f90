program test_mlp_grouped_training
    !! Independent analytic checks for group-wise MLP hyperparameter products.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_grouped_training, only: mlp_parameter_group_t, &
        mlp_grouped_training_objective_t
    use fortopt_objective, only: objective_t
    implicit none

    type(mlp_t), target :: model
    type(mlp_t) :: bad_model
    type(mlp_parameter_group_t) :: groups(2)
    type(mlp_grouped_training_objective_t) :: objective
    type(objective_t) :: fortopt_objective
    type(fortnum_status_t) :: status
    real(dp) :: x(3, 1), target(3, 1), parameters(4), direction(4)
    real(dp) :: gradient(4), product(4), expected(4), expected_hvp(4)
    real(dp) :: value, tangent, value_plus, value_minus, h
    integer :: failures, i

    failures = 0
    x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
    target(:, 1) = 0.8_dp*x(:, 1) + 0.2_dp
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call check(status_ok(status), "linear model initialization", failures)
    call model%set_parameters([0.4_dp, -0.3_dp], status)
    call groups(1)%initialize("weight", 1, 1, -1.0_dp, status)
    call check(status_ok(status), "weight group initialization", failures)
    call groups(2)%initialize("bias", 2, 2, -2.0_dp, status)
    call check(status_ok(status), "bias group initialization", failures)
    call objective%initialize(bad_model, x, target, groups, status)
    call check(.not. status_ok(status), "uninitialized model refusal", failures)
    call objective%initialize(model, x, target, groups, status)
    call check(status_ok(status) .and. objective%initialized(), &
        "grouped objective initialization", failures)
    call check(objective%parameter_count() == 4 .and. objective%group_count() == 2, &
        "packed grouped parameter layout", failures)
    call check(trim(objective%group_name(1)) == "weight", "weight group name", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. ieee_is_finite(value), &
        "grouped value/gradient status", failures)
    expected = [ &
        -0.2666666666666667_dp + exp(-1.0_dp)*0.4_dp, &
        -0.5_dp + exp(-2.0_dp)*(-0.3_dp), &
        0.5_dp*exp(-1.0_dp)*0.4_dp*0.4_dp, &
        0.5_dp*exp(-2.0_dp)*(-0.3_dp)*(-0.3_dp)]
    call check(maxval(abs(gradient - expected)) < 2.0e-12_dp, &
        "grouped gradient agrees with closed-form ridge oracle", failures)

    direction = [0.17_dp, -0.23_dp, 0.31_dp, -0.27_dp]
    call objective%hvp(parameters, direction, product, status)
    call check(status_ok(status), "grouped HVP status", failures)
    ! The linear-network data Hessian is X^T X / n with no cross term here.
    expected_hvp = [ &
        (2.0_dp/3.0_dp)*direction(1) + exp(-1.0_dp)* &
            (direction(1) + 0.4_dp*direction(3)), &
        direction(2) + exp(-2.0_dp)* &
            (direction(2) - 0.3_dp*direction(4)), &
        exp(-1.0_dp)*(0.4_dp*direction(1) + &
            0.5_dp*0.4_dp*0.4_dp*direction(3)), &
        exp(-2.0_dp)*(-0.3_dp*direction(2) + &
            0.5_dp*(-0.3_dp)*(-0.3_dp)*direction(4))]
    call check(maxval(abs(product - expected_hvp)) < 2.0e-11_dp, &
        "grouped HVP mixed-block oracle", failures)

    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status), "grouped JVP status", failures)
    h = 2.0e-6_dp
    call objective%value_gradient(parameters + h*direction, value_plus, gradient, status)
    call objective%value_gradient(parameters - h*direction, value_minus, gradient, status)
    call check(abs(tangent - (value_plus - value_minus)/(2.0_dp*h)) < 2.0e-8_dp, &
        "grouped JVP central-difference oracle", failures)

    call objective%vjp(parameters, 1.7_dp, product, status)
    call check(status_ok(status) .and. maxval(abs(product - 1.7_dp*expected)) < 2.0e-12_dp, &
        "grouped VJP scalar cotangent", failures)
    call objective%fortopt(fortopt_objective, status)
    call check(status_ok(status), "FortOpt grouped context adapter", failures)
    call fortopt_objective%value_gradient(parameters, value, product, status)
    call check(status_ok(status), "FortOpt grouped callback", failures)

    call objective%initialize(model, x, target, groups, status, &
        device_kind=FORTML_DEVICE_CUDA)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA grouped objective typed refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS grouped MLP training independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL: "//description
        end if
    end subroutine check

end program test_mlp_grouped_training
