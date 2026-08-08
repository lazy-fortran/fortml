program test_mlp_weighted_validation_hypergradient
    !! Independent weighted-validation oracle for the SGD trajectory adapter.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_sgd_momentum_hypergradient, only: &
        mlp_sgd_momentum_hypergradient_objective_t, &
        mlp_sgd_momentum_hypergradient_options_t, &
        MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT
    implicit none

    type(mlp_t), target :: model
    type(mlp_sgd_momentum_hypergradient_objective_t) :: objective
    type(mlp_sgd_momentum_hypergradient_options_t) :: options
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: validation_weight(3)
    real(dp) :: parameters(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: hvp_product(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: hvp_plus(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: hvp_minus(MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT)
    real(dp) :: value, value_plus, value_minus, tangent, h
    integer :: i, failures

    failures = 0
    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.65_dp*train_x(:, 1) - 0.1_dp
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.65_dp*validation_x(:, 1) - 0.1_dp
    validation_weight = [1.0_dp, 2.0_dp, 4.0_dp]

    call model%initialize([1, 1], status, hidden_activation=MLP_LINEAR, &
        output_activation=MLP_LINEAR)
    call model%set_parameters([0.13_dp, -0.08_dp], status)
    options%steps = 4
    options%learning_rate = 0.11_dp
    options%l2 = 0.06_dp
    options%momentum = 0.29_dp
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_momentum = 0.05_dp
    options%upper_momentum = 0.8_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status, validation_weight)
    call check(status_ok(status), "weighted validation initialization", failures)
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "weighted value/gradient", failures)
    call check(abs(value - 1.4998256378050192e-2_dp) < 2.0e-12_dp, &
        "weighted validation value independent oracle", failures)
    call check(maxval(abs(gradient - [-6.10401804644091e-2_dp, &
        1.65088250231663e-3_dp, -8.02196233914802e-2_dp])) < 2.0e-10_dp, &
        "weighted validation gradient independent oracle", failures)

    h = 2.0e-6_dp
    do i = 1, MLP_SGD_MOMENTUM_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(status_ok(status), "weighted finite-difference evaluation", failures)
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 3.0e-6_dp, &
            "weighted validation hypergradient central difference", failures)
    end do

    direction = [0.23_dp, -0.17_dp, 0.11_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call objective%value_gradient(parameters+h*direction, value_plus, vjp_gradient, status)
    call objective%value_gradient(parameters-h*direction, value_minus, vjp_gradient, status)
    call check(status_ok(status) .and. abs(tangent - (value_plus-value_minus)/(2.0_dp*h)) &
        < 3.0e-6_dp, "weighted validation JVP central difference", failures)
    call check(abs(tangent - (-2.31440501021786e-2_dp)) < 2.0e-10_dp, &
        "weighted validation JVP independent oracle", failures)
    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. maxval(abs(vjp_gradient-1.7_dp*gradient)) < 2.0e-12_dp, &
        "weighted validation scalar VJP", failures)

    call objective%hvp(parameters, direction, hvp_product, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "non-uniform validation HVP typed refusal", failures)

    validation_weight = [1.0_dp, 1.0_dp, 1.0_dp]
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status, validation_weight)
    call check(status_ok(status), "uniform validation reinitialization", failures)
    parameters = objective%parameters()
    h = 2.0e-6_dp
    call objective%hvp(parameters, direction, hvp_product, status)
    call objective%value_gradient(parameters+h*direction, value_plus, hvp_plus, status)
    call objective%value_gradient(parameters-h*direction, value_minus, hvp_minus, status)
    call check(status_ok(status) .and. maxval(abs(hvp_product - &
        (hvp_plus-hvp_minus)/(2.0_dp*h))) < 4.0e-6_dp, &
        "uniform validation HVP central-difference oracle", failures)
    call check(maxval(abs(hvp_product - [3.79430071650692e-2_dp, &
        -8.46766233519869e-4_dp, 3.37074129459075e-2_dp])) < 4.0e-6_dp, &
        "uniform validation HVP independent oracle", failures)

    validation_weight = [1.0_dp, -1.0_dp, 1.0_dp]
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status, validation_weight)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "negative validation weight refusal", failures)
    validation_weight = [1.0_dp, 0.0_dp, 0.0_dp]
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status, validation_weight)
    call check(status_ok(status), "zero-supported validation weights", failures)
    options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status, validation_weight)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA weighted validation typed refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP weighted validation hypergradient independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

end program test_mlp_weighted_validation_hypergradient
