program test_mlp_regressor
    !! Independent behavioral checks for the public MLP regressor facade.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp_regressor, only: mlp_regressor_t, mlp_regressor_options_t, &
        mlp_regressor_state_t
    implicit none

    type(mlp_regressor_t) :: model, lbfgs_model
    type(mlp_regressor_options_t) :: options, lbfgs_options
    type(mlp_regressor_state_t) :: lbfgs_state
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 1), target(6, 1), prediction(6, 1), prediction_plus(6, 1)
    real(dp) :: prediction_minus(6, 1), prediction_dot(6, 1), x_dot(6, 1)
    real(dp), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
    real(dp), allocatable :: gradient(:), hvp(:)
    real(dp) :: value, l2_gradient, l2_hvp, h
    logical :: ok

    x(:, 1) = [-1.5_dp, -1.0_dp, -0.25_dp, 0.5_dp, 1.0_dp, 1.75_dp]
    target(:, 1) = 0.7_dp*x(:, 1) - 0.15_dp
    allocate(options%layer_sizes(3))
    options%layer_sizes = [1, 3, 1]
    options%training%max_epochs = 20
    options%training%learning_rate = 0.03_dp
    options%training%tolerance = 0.0_dp
    options%training%restore_best = .false.
    call model%fit(x, target, status, options)
    ok = status_ok(status) .and. model%fitted()
    if (.not. ok) error stop "MLP regressor training fit failed"
    allocate(theta(model%parameter_count()), theta_dot(model%parameter_count()))
    allocate(theta_plus(model%parameter_count()), theta_minus(model%parameter_count()))
    allocate(gradient(model%parameter_count()), hvp(model%parameter_count()))
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "MLP regressor prediction failed"
    theta = model%parameters()
    theta_dot = 0.03_dp
    x_dot = 0.0_dp
    call model%predict_jvp(x, theta_dot, x_dot, prediction, prediction_dot, status)
    if (.not. status_ok(status)) error stop "MLP regressor JVP failed"
    h = 1.0e-6_dp
    call model%set_parameters(theta + h*theta_dot, status)
    call model%predict(x, prediction_plus, status)
    call model%set_parameters(theta - h*theta_dot, status)
    call model%predict(x, prediction_minus, status)
    call model%set_parameters(theta, status)
    if (maxval(abs(prediction_dot - (prediction_plus - prediction_minus)/(2.0_dp*h))) > 2.0e-7_dp) then
        error stop "MLP regressor JVP oracle failed"
    end if
    call model%loss_gradient(x, target, 0.02_dp, value, gradient, l2_gradient, status)
    if (.not. status_ok(status)) error stop "MLP regressor loss gradient failed"
    call model%loss_hvp(x, target, 0.02_dp, theta_dot, -0.04_dp, hvp, l2_hvp, status)
    if (.not. status_ok(status) .or. any(.not. (hvp == hvp))) error stop "MLP regressor HVP failed"

    allocate(lbfgs_options%layer_sizes(2))
    lbfgs_options%layer_sizes = [1, 1]
    lbfgs_options%use_lbfgsb = .true.
    lbfgs_options%lbfgsb%max_iterations = 100
    lbfgs_options%lbfgsb%gradient_tolerance = 1.0e-7_dp
    lbfgs_options%lbfgsb%step_tolerance = 1.0e-12_dp
    lbfgs_options%lbfgsb%objective_tolerance = 1.0e-12_dp
    call lbfgs_model%fit(x, target, status, lbfgs_options)
    lbfgs_state = lbfgs_model%state()
    if (.not. status_ok(status) .or. .not. lbfgs_state%converged) then
        error stop "MLP regressor L-BFGS-B fit failed"
    end if
    write (*, '(a)') "PASS MLP regressor independent behavioral oracle"
end program test_mlp_regressor
