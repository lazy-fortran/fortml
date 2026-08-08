program test_mlp_radam_hypergradient
    !! Independent finite-difference and adjoint checks for RAdam trajectory products.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: MLP_OPTIMIZER_ADAM
    use fortml_mlp_radam_hypergradient, only: &
        mlp_radam_hypergradient_objective_t, mlp_radam_hypergradient_options_t, &
        mlp_radam_hypergradient_metadata_t, mlp_radam_hypergradient_result_t, &
        mlp_optimize_radam_hyperparameters, MLP_RADAM_HYPERPARAMETER_COUNT, &
        MLP_RADAM_LOG_LEARNING_RATE, MLP_RADAM_LOG_L2, MLP_RADAM_LOGIT_BETA1, &
        MLP_RADAM_LOGIT_BETA2, MLP_RADAM_LOG_EPSILON
    implicit none

    type(mlp_t), target :: model
    type(mlp_radam_hypergradient_objective_t) :: objective
    type(mlp_radam_hypergradient_options_t) :: options, bad_options
    type(mlp_radam_hypergradient_result_t) :: result
    type(mlp_radam_hypergradient_metadata_t) :: metadata
    type(fortnum_status_t) :: status
    real(dp) :: train_x(6, 1), train_target(6, 1)
    real(dp) :: validation_x(3, 1), validation_target(3, 1)
    real(dp) :: parameters(MLP_RADAM_HYPERPARAMETER_COUNT)
    real(dp) :: direction(MLP_RADAM_HYPERPARAMETER_COUNT)
    real(dp) :: gradient(MLP_RADAM_HYPERPARAMETER_COUNT)
    real(dp) :: vjp_gradient(MLP_RADAM_HYPERPARAMETER_COUNT)
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
    options%learning_rate = 0.05_dp
    options%l2 = 0.03_dp
    options%beta1 = 0.7_dp
    options%beta2 = 0.9_dp
    options%epsilon = 0.02_dp
    options%lower_log_learning_rate = -5.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -5.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_log_epsilon = -5.0_dp
    options%upper_log_epsilon = 0.0_dp
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, options, status)
    call check(status_ok(status), "RAdam hypergradient initialization", failures)
    metadata = objective%metadata()
    call check(metadata%parameter_count == MLP_RADAM_HYPERPARAMETER_COUNT .and. &
        metadata%inner_steps == 6, "RAdam packed metadata", failures)

    parameters = objective%parameters()
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "RAdam value/gradient", failures)
    h = 2.0e-6_dp
    do i = 1, MLP_RADAM_HYPERPARAMETER_COUNT
        parameters(i) = parameters(i) + h
        call objective%value_gradient(parameters, value_plus, vjp_gradient, status)
        parameters(i) = parameters(i) - 2.0_dp*h
        call objective%value_gradient(parameters, value_minus, vjp_gradient, status)
        parameters(i) = parameters(i) + h
        call check(abs(gradient(i) - (value_plus-value_minus)/(2.0_dp*h)) < 4.0e-6_dp, &
            "RAdam hypergradient central difference", failures)
    end do

    direction = [0.21_dp, -0.17_dp, 0.23_dp, -0.19_dp, 0.13_dp]
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status), "RAdam forward JVP", failures)
    call objective%value_gradient(parameters+h*direction, value_plus, vjp_gradient, status)
    call objective%value_gradient(parameters-h*direction, value_minus, vjp_gradient, status)
    call check(abs(tangent-(value_plus-value_minus)/(2.0_dp*h)) < 6.0e-6_dp, &
        "RAdam forward JVP central difference", failures)
    call objective%vjp(parameters, 1.7_dp, vjp_gradient, status)
    call check(status_ok(status), "RAdam scalar VJP", failures)
    call check(maxval(abs(vjp_gradient-1.7_dp*gradient)) < 2.0e-12_dp, &
        "RAdam scalar VJP adjoint", failures)

    bad_options = options
    bad_options%optimizer = MLP_OPTIMIZER_ADAM
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "non-RAdam optimizer refusal", failures)
    bad_options = options
    bad_options%device_kind = FORTML_DEVICE_CUDA
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA RAdam hypergradient refusal", failures)

    options%steps = 3
    options%max_iterations = 80
    options%gradient_tolerance = 1.0e-5_dp
    call mlp_optimize_radam_hyperparameters(model, train_x, train_target, validation_x, &
        validation_target, options, result, status)
    call check(status_ok(status), "RAdam FortOpt L-BFGS-B solve", failures)
    call check(result%converged .and. result%learning_rate > 0.0_dp .and. &
        result%beta1 > 0.0_dp .and. result%beta1 < 1.0_dp .and. &
        result%beta2 > 0.0_dp .and. result%beta2 < 1.0_dp, &
        "RAdam FortOpt result", failures)

    bad_options = options
    bad_options%rho_boundary_tolerance = huge(1.0_dp)
    call objective%initialize(model, train_x, train_target, validation_x, &
        validation_target, bad_options, status)
    call objective%value_gradient(objective%parameters(), value, gradient, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "RAdam rho branch nonsmooth refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS MLP RAdam hypergradient independent behavioral oracles"

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

end program test_mlp_radam_hypergradient
