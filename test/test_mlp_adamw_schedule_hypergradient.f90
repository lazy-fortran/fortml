program test_mlp_adamw_schedule_hypergradient
    !! Independent finite-difference and adjoint checks for scheduled AdamW.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: MLP_OPTIMIZER_ADAM
    use fortml_mlp_schedules, only: make_mlp_schedule_cosine_decay, &
        make_mlp_schedule_exponential_decay
    use fortml_mlp_adamw_schedule_hypergradient, only: &
        mlp_adamw_schedule_hypergradient_objective_t, &
        mlp_adamw_schedule_hypergradient_options_t, &
        mlp_adamw_schedule_hypergradient_result_t, &
        mlp_adamw_schedule_hypergradient_metadata_t, &
        mlp_optimize_adamw_schedule_hyperparameters, &
        MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT
    implicit none

    type(mlp_t), target :: model
    type(mlp_adamw_schedule_hypergradient_objective_t) :: objective
    type(mlp_adamw_schedule_hypergradient_options_t) :: options, bad_options
    type(mlp_adamw_schedule_hypergradient_result_t) :: result
    type(mlp_adamw_schedule_hypergradient_metadata_t) :: metadata
    type(fortnum_status_t) :: status
    real(dp) :: train_x(6, 1), train_target(6, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: hvp(MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: value, value_plus, value_minus, tangent, h
    integer :: i, failures

    failures = 0
    train_x(:, 1) = [-2.5_dp, -1.5_dp, -0.5_dp, 0.5_dp, 1.5_dp, 2.5_dp]
    train_target(:, 1) = 0.6_dp*train_x(:, 1) - 0.35_dp
    validation_x(:, 1) = [-1.75_dp, 0.25_dp, 1.9_dp]
    validation_target(:, 1) = 0.6_dp*validation_x(:, 1) - 0.35_dp

    call model%initialize([1, 1], status, initialization_seed=41)
    call model%set_parameters([0.12_dp, -0.08_dp], status)
    options%steps = 6
    options%schedule = make_mlp_schedule_cosine_decay(8, 0.2_dp)
    options%base_rate = 0.05_dp
    options%l2 = 0.03_dp
    options%weight_decay = 0.04_dp
    options%beta1 = 0.7_dp
    options%beta2 = 0.9_dp
    options%epsilon = 0.02_dp
    options%lower_log_base_rate = -5.0_dp
    options%upper_log_base_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_log_weight_decay = -5.0_dp
    options%upper_log_weight_decay = 0.0_dp
    options%lower_log_epsilon = -5.0_dp
    options%upper_log_epsilon = 0.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "scheduled AdamW initialization", failures)
    metadata = objective%metadata()
    call check(metadata%parameter_count == MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT .and. &
        metadata%inner_steps == 6, "scheduled AdamW packed metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "scheduled AdamW value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_ADAMW_SCHEDULE_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 8.0e-6_dp, &
            "scheduled AdamW hypergradient central difference", failures)
    end do

    direction = [0.21_dp, -0.17_dp, 0.23_dp, -0.19_dp, 0.13_dp, 0.11_dp, 0.07_dp, -0.05_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status), "scheduled AdamW forward JVP", failures)
    call objective%value_gradient(parameters+h*direction, value_plus, vjp_gradient, status)
    call objective%value_gradient(parameters-h*direction, value_minus, vjp_gradient, status)
    call check(abs(tangent-(value_plus-value_minus)/(2.0_dp*h)) < 1.0e-5_dp, &
        "scheduled AdamW forward JVP central difference", failures)
    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call check(status_ok(status), "scheduled AdamW scalar VJP", failures)
    call check(maxval(abs(vjp_gradient-1.7_dp*gradient)) < 2.0e-12_dp, &
        "scheduled AdamW scalar VJP adjoint", failures)
    call objective%hvp(parameters, direction, hvp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "scheduled AdamW outer HVP typed refusal", failures)

    options%schedule = make_mlp_schedule_exponential_decay(1, 0.8_dp)
    options%steps = 4
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "exponential scheduled AdamW initialization", failures)
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    parameters(8) = parameters(8) + h
    call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
    parameters(8) = parameters(8) - 2.0_dp*h
    call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
    call check(abs(gradient(8) - (value_plus-value_minus)/(2.0_dp*h)) < 8.0e-6_dp, &
        "exponential AdamW decay-factor derivative", failures)

    bad_options = options
    bad_options%optimizer = MLP_OPTIMIZER_ADAM
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "non-AdamW optimizer refusal", failures)
    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA AdamW hypergradient refusal", failures)

    options%steps = 3
    options%max_iterations = 80
    options%gradient_tolerance = 1.0e-5_dp
    call mlp_optimize_adamw_schedule_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    call check(status_ok(status), "scheduled AdamW FortOpt L-BFGS-B solve", failures)
    call check(result%converged .and. result%base_rate > 0.0_dp .and. &
        result%weight_decay > 0.0_dp .and. result%beta1 > 0.0_dp .and. &
        result%beta1 < 1.0_dp .and. result%beta2 > 0.0_dp .and. result%beta2 < 1.0_dp, &
        "scheduled AdamW FortOpt result", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP scheduled AdamW hypergradient independent behavioral oracles"

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

end program test_mlp_adamw_schedule_hypergradient
