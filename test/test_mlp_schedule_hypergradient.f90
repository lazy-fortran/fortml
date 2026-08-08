program test_mlp_schedule_hypergradient
    !! Independent finite-difference and adjoint checks for scheduled MLP
    !! trajectory products, plus FortOpt and device-boundary contracts.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_schedules, only: make_mlp_schedule_cosine_decay, &
        make_mlp_schedule_constant, make_mlp_schedule_one_cycle
    use fortml_mlp_schedule_hypergradient, only: &
        mlp_schedule_hypergradient_objective_t, &
        mlp_schedule_hypergradient_options_t, &
        mlp_schedule_hypergradient_result_t, &
        mlp_schedule_hypergradient_metadata_t, &
        mlp_optimize_schedule_hyperparameters, &
        MLP_SCHEDULE_HYPERPARAMETER_COUNT, MLP_SCHEDULE_LOG_BASE_RATE, &
        MLP_SCHEDULE_LOG_L2, MLP_SCHEDULE_LOGIT_MIN_FRACTION, &
        MLP_SCHEDULE_LOGIT_DECAY_FACTOR, MLP_SCHEDULE_LOG_PEAK_FRACTION, &
        MLP_SCHEDULE_LOG_FINAL_FRACTION
    implicit none

    type(mlp_t), target :: model, nonlinear_model
    type(mlp_schedule_hypergradient_objective_t) :: objective
    type(mlp_schedule_hypergradient_options_t) :: options, bad_options
    type(mlp_schedule_hypergradient_result_t) :: result
    type(mlp_schedule_hypergradient_metadata_t) :: metadata
    type(objective_t) :: fortopt_objective
    type(fortnum_status_t) :: status
    real(dp) :: train_x(5, 1), train_target(5, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: hvp_product(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: hvp_product_2(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: hessian_fd(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: direction_2(MLP_SCHEDULE_HYPERPARAMETER_COUNT)
    real(dp) :: value, value_plus, value_minus, tangent, h, output_bar
    integer :: i, failures

    failures = 0
    train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    train_target(:, 1) = 0.7_dp*train_x(:, 1) - 0.2_dp
    validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
    validation_target(:, 1) = 0.7_dp*validation_x(:, 1) - 0.2_dp

    call model%initialize([1, 1], status, initialization_seed=23)
    call model%set_parameters([0.15_dp, -0.1_dp], status)
    options%steps = 4
    options%base_rate = 0.12_dp
    options%l2 = 0.07_dp
    options%schedule = make_mlp_schedule_cosine_decay(6, 0.2_dp)
    options%lower_log_base_rate = -4.0_dp
    options%upper_log_base_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "schedule hypergradient initialization", failures)
    call check(objective%is_initialized(), "initialized predicate", failures)
    call check(objective%parameter_count() == MLP_SCHEDULE_HYPERPARAMETER_COUNT, &
        "packed parameter count", failures)
    metadata = objective%metadata()
    call check(metadata%inner_steps == 4 .and. metadata%schedule_kind == &
        options%schedule%kind, "schedule metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "reverse value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_SCHEDULE_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        call check(status_ok(status), "plus finite-difference evaluation", failures)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        call check(status_ok(status), "minus finite-difference evaluation", failures)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < &
            3.0e-7_dp, "schedule hypergradient central difference", failures)
    end do

    direction = [0.31_dp, -0.27_dp, 0.18_dp, -0.22_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status), "forward schedule JVP", failures)
    call objective%value_gradient(parameters+h*direction, value_plus, &
        vjp_gradient, status)
    call objective%value_gradient(parameters-h*direction, value_minus, &
        vjp_gradient, status)
    call check(abs(tangent-(value_plus-value_minus)/(2.0_dp*h)) < 4.0e-7_dp, &
        "schedule JVP central difference", failures)

    output_bar = 1.7_dp
    call objective%vjp(parameters, output_bar, vjp_gradient, status)
    call check(status_ok(status), "reverse schedule VJP", failures)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(maxval(abs(vjp_gradient-output_bar*gradient)) < 2.0e-12_dp, &
        "schedule VJP scalar adjoint", failures)

    call objective%hvp(parameters, direction, hvp_product, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "nonconstant affine schedule HVP typed refusal", failures)

    ! A constant-rate trajectory of one affine layer has an exact outer HVP.
    options%schedule = make_mlp_schedule_constant()
    options%steps = 4
    options%base_rate = 0.12_dp
    options%l2 = 0.07_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "constant affine HVP initialization", failures)
    parameters = objective%parameters()
    direction = [0.31_dp, -0.27_dp, 0.18_dp, -0.22_dp]
    call objective%hvp(parameters, direction, hvp_product, status)
    call check(status_ok(status), "constant affine HVP status", failures)
    call objective%value_gradient(parameters+h*direction, value_plus, &
        vjp_gradient, status)
    call check(status_ok(status), "constant affine HVP plus gradient", failures)
    call objective%value_gradient(parameters-h*direction, value_minus, &
        gradient, status)
    call check(status_ok(status), "constant affine HVP minus gradient", failures)
    hessian_fd = (vjp_gradient-gradient)/(2.0_dp*h)
    call check(maxval(abs(hvp_product-hessian_fd)) < 2.0e-7_dp, &
        "constant affine HVP central difference", failures)
    direction_2 = [-0.13_dp, 0.22_dp, -0.19_dp, 0.11_dp]
    call objective%hvp(parameters, direction_2, hvp_product_2, status)
    call check(status_ok(status), "constant affine second HVP status", failures)
    call check(abs(dot_product(direction_2, hvp_product)- &
        dot_product(direction, hvp_product_2)) < 3.0e-10_dp, &
        "constant affine HVP Hessian symmetry", failures)
    call objective%fortopt(fortopt_objective, status)
    call check(status_ok(status), "constant affine FortOpt context adapter", failures)
    call fortopt_objective%value_gradient(parameters, value, vjp_gradient, status)
    call check(status_ok(status), "constant affine FortOpt callback", failures)

    call objective%hvp(parameters, direction, hvp_product, status)
    call check(status_ok(status), "constant affine repeated HVP", failures)

    ! One-cycle uses logarithmic peak/final-rate coordinates in the same
    ! four-vector, with exact JVP/VJP products through both schedule phases.
    options%schedule = make_mlp_schedule_one_cycle(2, 8, 1.8_dp, 0.08_dp)
    options%steps = 5
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "one-cycle hypergradient initialization", failures)
    metadata = objective%metadata()
    call check(metadata%one_cycle_coordinates, "one-cycle coordinate metadata", failures)
    parameters = objective%parameters()
    call check(abs(parameters(MLP_SCHEDULE_LOG_PEAK_FRACTION)-log(1.8_dp)) < 1.0e-14_dp .and. &
        abs(parameters(MLP_SCHEDULE_LOG_FINAL_FRACTION)-log(0.08_dp)) < 1.0e-14_dp, &
        "one-cycle packed logarithms", failures)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "one-cycle value/gradient", failures)
    do i = 1, MLP_SCHEDULE_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(status_ok(status) .and. abs(gradient(i) - &
            (value_plus-value_minus)/(2.0_dp*h)) < 4.0e-7_dp, &
            "one-cycle hypergradient central difference", failures)
    end do
    direction = [0.17_dp, -0.13_dp, 0.21_dp, -0.19_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call objective%value_gradient(parameters+h*direction, value_plus, vjp_gradient, status)
    call objective%value_gradient(parameters-h*direction, value_minus, vjp_gradient, status)
    call check(status_ok(status) .and. abs(tangent - &
        (value_plus-value_minus)/(2.0_dp*h)) < 5.0e-7_dp, &
        "one-cycle JVP central difference", failures)
    call objective%vjp(parameters, output_bar, vjp_gradient, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. maxval(abs(vjp_gradient-output_bar*gradient)) < &
        2.0e-12_dp, "one-cycle scalar VJP adjoint", failures)
    call objective%fortopt(fortopt_objective, status)
    call check(status_ok(status), "one-cycle FortOpt context adapter", failures)

    call objective%fortopt(fortopt_objective, status)
    call check(status_ok(status), "FortOpt schedule context adapter", failures)
    call fortopt_objective%value_gradient(parameters, value, vjp_gradient, status)
    call check(status_ok(status), "FortOpt schedule callback", failures)

    ! Nonlinear networks retain the explicit third-derivative refusal even
    ! under a constant schedule.
    call nonlinear_model%initialize([1, 2, 1], status, initialization_seed=29)
    call nonlinear_model%set_parameters([0.03_dp, -0.04_dp, 0.02_dp, 0.01_dp, &
        -0.02_dp, 0.05_dp, -0.01_dp], status)
    options%schedule = make_mlp_schedule_constant()
    options%steps = 4
    call objective%initialize(nonlinear_model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "nonlinear HVP initialization", failures)
    parameters = objective%parameters()
    call objective%hvp(parameters, direction, hvp_product, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "nonlinear schedule HVP typed refusal", failures)

    ! Nonconstant affine schedules retain the rate-second-product refusal.
    options%schedule = make_mlp_schedule_cosine_decay(6, 0.2_dp)
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "nonconstant affine HVP initialization", failures)
    parameters = objective%parameters()
    call objective%hvp(parameters, direction, hvp_product, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "nonconstant affine schedule HVP typed refusal", failures)

    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA schedule hypergradient refusal", failures)

    ! A constant schedule makes the schedule fields inactive and gives a
    ! stable two-variable L-BFGS-B smoke solve for the production adapter.
    options%schedule = make_mlp_schedule_constant()
    options%steps = 3
    options%max_iterations = 120
    options%gradient_tolerance = 1.0e-5_dp
    call mlp_optimize_schedule_hyperparameters(model, train_x, train_target, &
        validation_x, validation_target, options, result, status)
    call check(status_ok(status), "FortOpt scheduled hyperparameter solve", failures)
    call check(result%converged .and. result%base_rate > 0.0_dp .and. &
        result%l2 > 0.0_dp, "FortOpt scheduled result", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP schedule hypergradient independent behavioral oracles"

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

end program test_mlp_schedule_hypergradient
