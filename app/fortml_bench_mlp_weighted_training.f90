program fortml_bench_mlp_weighted_training
    !! Release workload and independent oracle summary for weighted MLP training.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_loss_value_gradient, mlp_train, MLP_OPTIMIZER_SGD
    implicit none

    type(mlp_t) :: model
    type(mlp_training_options_t) :: options
    type(mlp_training_state_t) :: state
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 1), target(4, 1), weight(4), theta(2), prediction(4), residual(4)
    real(dp) :: gradient(2), expected_gradient(2), value, expected_value, l2_gradient
    real(dp) :: mass, loss_error, gradient_error, parameter_error
    real(dp) :: validation_x(2, 1), validation_target(2, 1), validation_weight(2)
    real(dp) :: expected_theta(2), invalid_theta(2), invalid_status

    x(:, 1) = [-1.0_dp, -0.25_dp, 0.75_dp, 1.5_dp]
    target(:, 1) = [0.4_dp, -0.1_dp, 0.8_dp, 1.2_dp]
    weight = [1.0_dp, 0.0_dp, 2.0_dp, 0.5_dp]
    theta = [0.3_dp, -0.2_dp]
    validation_x(:, 1) = [-0.5_dp, 1.0_dp]
    validation_target(:, 1) = [0.0_dp, 0.9_dp]
    validation_weight = [0.25_dp, 1.5_dp]
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    if (.not. status_ok(status)) error stop "weighted benchmark initialization failed"
    call model%set_parameters(theta, status)
    call mlp_loss_value_gradient(model, x, target, 0.07_dp, value, gradient, &
        l2_gradient, status, sample_weight=weight)
    if (.not. status_ok(status)) error stop "weighted benchmark loss failed"
    prediction = x(:, 1)*theta(1) + theta(2)
    residual = prediction - target(:, 1)
    mass = sum(weight)
    expected_value = 0.5_dp*sum(weight*residual**2)/mass + &
        0.5_dp*0.07_dp*sum(theta**2)
    expected_gradient(1) = sum(weight*residual*x(:, 1))/mass + 0.07_dp*theta(1)
    expected_gradient(2) = sum(weight*residual)/mass + 0.07_dp*theta(2)
    loss_error = abs(value - expected_value)
    gradient_error = maxval(abs(gradient - expected_gradient))

    options%optimizer = MLP_OPTIMIZER_SGD
    options%max_epochs = 1
    options%batch_size = 2
    options%accumulation_steps = 2
    options%learning_rate = 0.025_dp
    options%tolerance = 0.0_dp
    options%restore_best = .false.
    expected_theta = theta - options%learning_rate * &
        [sum(weight*residual*x(:, 1))/mass, sum(weight*residual)/mass]
    call mlp_train(model, x, target, status, options, state, &
        validation_x=validation_x, validation_target=validation_target, &
        sample_weight=weight, validation_weight=validation_weight)
    if (.not. status_ok(status)) error stop "weighted benchmark training failed"
    parameter_error = maxval(abs(model%parameters() - expected_theta))

    invalid_theta = model%parameters()
    call mlp_train(model, x, target, status, sample_weight=[1.0_dp, -1.0_dp, 1.0_dp, 1.0_dp])
    if (status_ok(status) .or. maxval(abs(model%parameters() - invalid_theta)) > 1.0e-14_dp) then
        error stop "weighted benchmark transactional refusal failed"
    end if
    invalid_status = real(status%code, dp)

    write (*, '(a)') "metric,value"
    write (*, '(a,",",es24.16)') "loss_error", loss_error
    write (*, '(a,",",es24.16)') "gradient_error", gradient_error
    write (*, '(a,",",es24.16)') "parameter_error", parameter_error
    write (*, '(a,",",es24.16)') "validation_loss", state%final_validation_loss
    write (*, '(a,",",es24.16)') "invalid_status", invalid_status
end program fortml_bench_mlp_weighted_training
