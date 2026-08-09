program test_mlp_clip_hypergradient
    !! Independent production-trainer oracle for clipping-threshold products.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_train, mlp_loss_value_gradient, MLP_OPTIMIZER_SGD
    use fortml_mlp_clip_hypergradient, only: &
        mlp_clip_hypergradient_objective_t, mlp_clip_hypergradient_options_t, &
        mlp_clip_hypergradient_result_t, mlp_clip_hypergradient_metadata_t, &
        mlp_optimize_clip_hyperparameters, MLP_CLIP_HYPERPARAMETER_COUNT, &
        MLP_CLIP_LOG_LEARNING_RATE, MLP_CLIP_LOG_L2, MLP_CLIP_LOG_NORM
    implicit none

    integer, parameter :: steps = 4
    type(mlp_t), target :: model
    type(mlp_clip_hypergradient_objective_t) :: objective
    type(mlp_clip_hypergradient_options_t) :: options, bad_options
    type(mlp_clip_hypergradient_result_t) :: result
    type(mlp_clip_hypergradient_metadata_t) :: metadata
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_CLIP_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_CLIP_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_CLIP_HYPERPARAMETER_COUNT)
    real(dp) :: parameter_bar(MLP_CLIP_HYPERPARAMETER_COUNT)
    real(dp) :: product(MLP_CLIP_HYPERPARAMETER_COUNT)
    real(dp) :: value, expected_value, plus_value, minus_value, tangent
    real(dp) :: raw_gradient(2), l2_gradient, raw_norm, h
    integer :: i, failures

    failures = 0
    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.7_dp*train_x(:, 1) - 0.2_dp
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.7_dp*validation_x(:, 1) - 0.2_dp
    call initialize_reference_model(model, status)
    call check(status_ok(status), "clip model initialization", failures)

    options%steps = steps
    options%learning_rate = 0.12_dp
    options%l2 = 0.07_dp
    options%gradient_clip_norm = 0.30_dp
    options%lower_log_learning_rate = log(0.03_dp)
    options%upper_log_learning_rate = log(0.25_dp)
    options%lower_log_l2 = log(0.01_dp)
    options%upper_log_l2 = log(0.20_dp)
    options%lower_log_clip_norm = log(0.05_dp)
    options%upper_log_clip_norm = log(0.40_dp)
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "clip hypergradient initialization", failures)
    metadata = objective%metadata()
    call check(objective%parameter_count() == MLP_CLIP_HYPERPARAMETER_COUNT .and. &
        metadata%log_learning_rate_index == MLP_CLIP_LOG_LEARNING_RATE .and. &
        metadata%log_l2_index == MLP_CLIP_LOG_L2 .and. &
        metadata%log_clip_norm_index == MLP_CLIP_LOG_NORM .and. &
        metadata%inner_steps == steps, "clip packed metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "clip exact value/gradient", failures)
    call production_value(parameters, expected_value, status)
    call check(status_ok(status) .and. abs(value-expected_value) < 2.0e-13_dp, &
        "clip objective matches production trainer", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_CLIP_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call production_value(parameters, plus_value, status)
        call check(status_ok(status), "clip production plus branch", failures)
        parameters(i) = parameters(i) - 2.0_dp*h
        call production_value(parameters, minus_value, status)
        call check(status_ok(status), "clip production minus branch", failures)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i)-(plus_value-minus_value)/(2.0_dp*h)) < 3.0e-6_dp, &
            "clip gradient matches production finite difference", failures)
    end do
    direction = [0.23_dp, -0.17_dp, 0.31_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call production_value(parameters+h*direction, plus_value, status)
    call production_value(parameters-h*direction, minus_value, status)
    call check(status_ok(status) .and. &
        abs(tangent-(plus_value-minus_value)/(2.0_dp*h)) < 3.0e-6_dp, &
        "clip JVP matches production finite difference", failures)
    call objective%vjp(parameters, 1.7_dp, parameter_bar, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. &
        maxval(abs(parameter_bar-1.7_dp*gradient)) < 2.0e-12_dp, &
        "clip scalar VJP adjoint", failures)

    ! An inactive threshold cannot affect the fixed trajectory.
    options%gradient_clip_norm = 10.0_dp
    options%upper_log_clip_norm = log(20.0_dp)
    call initialize_reference_model(model, status)
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. abs(gradient(MLP_CLIP_LOG_NORM)) < 1.0e-14_dp, &
        "inactive clip threshold has zero derivative", failures)

    ! The exact active-set boundary is deliberately outside the derivative API.
    call initialize_reference_model(model, status)
    call mlp_loss_value_gradient(model, train_x, train_target, options%l2, &
        value, raw_gradient, l2_gradient, status)
    raw_norm = sqrt(sum(raw_gradient*raw_gradient))
    options%gradient_clip_norm = raw_norm
    options%upper_log_clip_norm = log(20.0_dp)
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "clip active-set kink refusal", failures)

    options%gradient_clip_norm = 0.30_dp
    options%upper_log_clip_norm = log(0.40_dp)
    call initialize_reference_model(model, status)
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    parameters = objective%parameters()
    call objective%hvp(parameters, direction, product, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(product == 0.0_dp), &
        "clip outer HVP typed refusal", failures)
    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "clip CUDA trajectory refusal", failures)

    options%max_iterations = 300
    options%gradient_tolerance = 5.0e-2_dp
    options%objective_tolerance = 1.0e-8_dp
    call mlp_optimize_clip_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    if (.not. status_ok(status)) then
        write (*, '(a,i0,2a)') "clip FortOpt status=", status%code, " message=", &
            trim(status%msg)
    end if
    call check(status_ok(status) .and. result%converged .and. &
        result%learning_rate >= exp(options%lower_log_learning_rate) .and. &
        result%learning_rate <= exp(options%upper_log_learning_rate) .and. &
        result%l2 >= exp(options%lower_log_l2) .and. &
        result%l2 <= exp(options%upper_log_l2) .and. &
        result%gradient_clip_norm >= exp(options%lower_log_clip_norm) .and. &
        result%gradient_clip_norm <= exp(options%upper_log_clip_norm), &
        "clip FortOpt L-BFGS-B result", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP clip hypergradient cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP clip hypergradient independent production oracles"

contains

    subroutine initialize_reference_model(candidate, local_status)
        type(mlp_t), intent(out) :: candidate
        type(fortnum_status_t), intent(out) :: local_status

        call candidate%initialize([1, 1], local_status, hidden_activation=MLP_LINEAR, &
            output_activation=MLP_LINEAR)
        if (.not. status_ok(local_status)) return
        call candidate%set_parameters([0.15_dp, -0.1_dp], local_status)
    end subroutine initialize_reference_model

    subroutine production_value(packed, production_loss, local_status)
        real(dp), intent(in) :: packed(:)
        real(dp), intent(out) :: production_loss
        type(fortnum_status_t), intent(out) :: local_status
        type(mlp_t) :: production_model
        type(mlp_training_options_t) :: training_options
        type(mlp_training_state_t) :: training_state
        real(dp) :: validation_gradient(2), validation_l2_gradient

        production_loss = huge(1.0_dp)
        call initialize_reference_model(production_model, local_status)
        if (.not. status_ok(local_status)) return
        training_options%max_epochs = steps
        training_options%batch_size = 0
        training_options%shuffle = .false.
        training_options%optimizer = MLP_OPTIMIZER_SGD
        training_options%learning_rate = exp(packed(MLP_CLIP_LOG_LEARNING_RATE))
        training_options%l2 = exp(packed(MLP_CLIP_LOG_L2))
        training_options%gradient_clip_norm = exp(packed(MLP_CLIP_LOG_NORM))
        training_options%momentum = 0.0_dp
        training_options%tolerance = 0.0_dp
        call mlp_train(production_model, train_x, train_target, local_status, &
            training_options, training_state)
        if (.not. status_ok(local_status)) return
        call mlp_loss_value_gradient(production_model, validation_x, validation_target, &
            0.0_dp, production_loss, validation_gradient, validation_l2_gradient, &
            local_status)
    end subroutine production_value

    subroutine check(condition, label, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            write (*, '(a)') "FAIL: "//label
            failure_count = failure_count + 1
        end if
    end subroutine check

end program test_mlp_clip_hypergradient
