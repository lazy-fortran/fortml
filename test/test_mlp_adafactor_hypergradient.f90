program test_mlp_adafactor_hypergradient
    !! Independent finite-difference and adjoint checks for the smooth active
    !! branch of vector Adafactor trajectory hypergradients.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_adagrad_hypergradient, only: &
        mlp_adagrad_hypergradient_options_t
    use fortml_mlp_adagrad_hypergradient, only: MLP_ADAGRAD_HYPERPARAMETER_COUNT
    use fortml_mlp_adagrad_hypergradient, only: MLP_ADAGRAD_LOG_LEARNING_RATE
    use fortml_mlp_adafactor_hypergradient, only: &
        mlp_adafactor_hypergradient_objective_t, mlp_adafactor_hypergradient_options_t, &
        mlp_adafactor_hypergradient_metadata_t, mlp_adafactor_hypergradient_result_t, &
        mlp_optimize_adafactor_hyperparameters, MLP_ADAFACTOR_HYPERPARAMETER_COUNT, &
        MLP_ADAFACTOR_LOG_LEARNING_RATE, MLP_ADAFACTOR_LOG_L2, MLP_ADAFACTOR_DECAY, &
        MLP_ADAFACTOR_LOG_EPSILON, MLP_ADAFACTOR_LOG_CLIP_THRESHOLD
    use fortopt_objective, only: objective_t
    implicit none

    type(mlp_t), target :: model
    type(mlp_adafactor_hypergradient_objective_t) :: objective
    type(mlp_adafactor_hypergradient_options_t) :: options, bad_options
    type(mlp_adafactor_hypergradient_metadata_t) :: metadata
    type(mlp_adafactor_hypergradient_result_t) :: result
    type(objective_t) :: fortopt_objective
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_ADAFACTOR_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_ADAFACTOR_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_ADAFACTOR_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_ADAFACTOR_HYPERPARAMETER_COUNT)
    real(dp) :: value, value_plus, value_minus, tangent, h
    integer :: i, failures

    failures = 0
    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.7_dp*train_x(:, 1) - 0.2_dp
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.7_dp*validation_x(:, 1) - 0.2_dp
    call model%initialize([1, 1], status, initialization_seed=23)
    call model%set_parameters([0.15_dp, -0.1_dp], status)
    options%steps = 4
    options%learning_rate = 0.12_dp
    options%l2 = 0.07_dp
    options%decay = 0.75_dp
    options%epsilon = 0.03_dp
    options%clip_threshold = 0.2_dp
    options%lower_log_learning_rate = -4.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_decay = 0.1_dp
    options%upper_decay = 0.95_dp
    options%lower_log_epsilon = -5.0_dp
    options%upper_log_epsilon = 0.0_dp
    options%lower_log_clip_threshold = -5.0_dp
    options%upper_log_clip_threshold = 0.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "Adafactor hypergradient initialization", failures)
    metadata = objective%metadata()
    call check(objective%parameter_count() == MLP_ADAFACTOR_HYPERPARAMETER_COUNT .and. &
        metadata%log_learning_rate_index == MLP_ADAFACTOR_LOG_LEARNING_RATE .and. &
        metadata%log_l2_index == MLP_ADAFACTOR_LOG_L2 .and. &
        metadata%decay_index == MLP_ADAFACTOR_DECAY .and. &
        metadata%log_epsilon_index == MLP_ADAFACTOR_LOG_EPSILON .and. &
        metadata%log_clip_threshold_index == MLP_ADAFACTOR_LOG_CLIP_THRESHOLD .and. &
        metadata%inner_steps == 4, "Adafactor packed metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "Adafactor value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_ADAFACTOR_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(status_ok(status), "Adafactor central-difference evaluations", failures)
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 3.0e-6_dp, &
            "Adafactor hypergradient central difference", failures)
    end do
    direction = [0.31_dp, -0.27_dp, 0.19_dp, 0.13_dp, -0.22_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call objective%value_gradient(parameters+h*direction, value_plus, vjp_gradient, status)
    call objective%value_gradient(parameters-h*direction, value_minus, vjp_gradient, status)
    call check(status_ok(status), "Adafactor forward JVP", failures)
    call check(abs(tangent-(value_plus-value_minus)/(2.0_dp*h)) < 5.0e-6_dp, &
        "Adafactor forward JVP central difference", failures)
    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(maxval(abs(vjp_gradient-1.7_dp*gradient)) < 2.0e-12_dp, &
        "Adafactor scalar VJP adjoint", failures)
    call objective%fortopt(fortopt_objective, status)
    call check(status_ok(status), "Adafactor FortOpt context adapter", failures)

    bad_options = options
    bad_options%relative_step = .true.
    bad_options%learning_rate = 1.2_dp
    bad_options%upper_log_learning_rate = 1.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status_ok(status), "relative-step Adafactor initialization", failures)
    metadata = objective%metadata()
    call check(metadata%relative_step, "relative-step metadata", failures)
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "relative-step Adafactor value/gradient", failures)
    h = 2.0e-6_dp
    parameters(MLP_ADAFACTOR_LOG_LEARNING_RATE) = &
        parameters(MLP_ADAFACTOR_LOG_LEARNING_RATE) + h
    call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
    parameters(MLP_ADAFACTOR_LOG_LEARNING_RATE) = &
        parameters(MLP_ADAFACTOR_LOG_LEARNING_RATE) - 2.0_dp*h
    call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
    parameters(MLP_ADAFACTOR_LOG_LEARNING_RATE) = &
        parameters(MLP_ADAFACTOR_LOG_LEARNING_RATE) + h
    call check(status_ok(status) .and. &
        abs(gradient(MLP_ADAFACTOR_LOG_LEARNING_RATE) - &
        (value_plus-value_minus)/(2.0_dp*h)) < 4.0e-6_dp, &
        "relative-step active branch central difference", failures)

    bad_options = options
    bad_options%scale_parameter = .true.
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status_ok(status), "parameter-scale Adafactor initialization", failures)
    metadata = objective%metadata()
    call check(metadata%scale_parameter, "parameter-scale metadata", failures)
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "parameter-scale Adafactor value/gradient", failures)
    do i = 1, MLP_ADAFACTOR_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(status_ok(status), "parameter-scale central-difference evaluations", failures)
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 6.0e-6_dp, &
            "parameter-scale Adafactor central difference", failures)
    end do

    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA Adafactor hypergradient refusal", failures)
    call mlp_optimize_adafactor_hyperparameters(model, train_x, train_target, validation_x, &
        validation_target, bad_options, result, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA Adafactor optimizer refusal", failures)

    ! Retain an import-level guard against accidentally confusing this packed
    ! layout with another optimizer's outer vector in client code.
    call check(MLP_ADAGRAD_HYPERPARAMETER_COUNT /= MLP_ADAFACTOR_HYPERPARAMETER_COUNT .and. &
        MLP_ADAGRAD_LOG_LEARNING_RATE == 1, "optimizer layout distinction", failures)
    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP Adafactor hypergradient independent behavioral oracles"

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

end program test_mlp_adafactor_hypergradient
