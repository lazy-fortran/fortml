program test_mlp_rmsprop_weighted_hypergradient
    !! Weighted RMSprop trajectory products against a separate scalar affine
    !! recurrence and central differences of that recurrence.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortopt_objective, only: objective_t
    use fortml_mlp_hypergradient, only: &
        mlp_rmsprop_hypergradient_objective_t, &
        mlp_rmsprop_hypergradient_options_t, mlp_rmsprop_hypergradient_result_t, &
        mlp_rmsprop_hypergradient_metadata_t, mlp_optimize_rmsprop_hyperparameters, &
        MLP_RMSPROP_HYPERPARAMETER_COUNT
    implicit none

    integer :: failures

    failures = 0
    call test_weighted_products(failures)
    call test_weight_contracts(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " weighted RMSprop hypergradient test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: weighted RMSprop trajectory hypergradients"

contains

    subroutine fixture(train_x, train_target, train_weight, validation_x, &
            validation_target, validation_weight, options)
        real(dp), intent(out) :: train_x(5, 1), train_target(5, 1), train_weight(5)
        real(dp), intent(out) :: validation_x(3, 1), validation_target(3, 1)
        real(dp), intent(out) :: validation_weight(3)
        type(mlp_rmsprop_hypergradient_options_t), intent(out) :: options

        train_x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
        train_target(:, 1) = 0.7_dp*train_x(:, 1) - 0.2_dp
        train_weight = [0.25_dp, 1.5_dp, 0.0_dp, 2.0_dp, 0.75_dp]
        validation_x(:, 1) = [-1.5_dp, 0.5_dp, 1.75_dp]
        validation_target(:, 1) = 0.7_dp*validation_x(:, 1) - 0.2_dp
        validation_weight = [2.0_dp, 0.5_dp, 1.25_dp]
        options%steps = 4
        options%learning_rate = 0.12_dp
        options%l2 = 0.07_dp
        options%rmsprop_decay = 0.78_dp
        options%epsilon = 0.03_dp
        options%momentum = 0.21_dp
        options%centered = .true.
        options%lower_log_learning_rate = -4.0_dp
        options%upper_log_learning_rate = 0.0_dp
        options%lower_log_l2 = -5.0_dp
        options%upper_log_l2 = 0.0_dp
        options%lower_decay = 0.2_dp
        options%upper_decay = 0.95_dp
        options%lower_log_epsilon = -5.0_dp
        options%upper_log_epsilon = 0.0_dp
        options%lower_momentum = 0.01_dp
        options%upper_momentum = 0.8_dp
    end subroutine fixture

    subroutine test_weighted_products(failures)
        integer, intent(inout) :: failures
        type(mlp_t), target :: model
        type(mlp_rmsprop_hypergradient_objective_t), target :: adapter
        type(mlp_rmsprop_hypergradient_options_t) :: options
        type(mlp_rmsprop_hypergradient_metadata_t) :: metadata
        type(mlp_rmsprop_hypergradient_result_t) :: result
        type(objective_t) :: objective
        type(fortnum_status_t) :: status
        real(dp) :: train_x(5, 1), train_target(5, 1), train_weight(5)
        real(dp) :: validation_x(3, 1), validation_target(3, 1)
        real(dp) :: validation_weight(3), parameters(5), direction(5)
        real(dp) :: gradient(5), reference_gradient(5), product(5)
        real(dp) :: reference_product(5), parameter_bar(5), objective_gradient(5)
        real(dp) :: value, tangent, reference_value, objective_value, value_bar
        real(dp) :: h_gradient, h_direction, initial_objective

        call fixture(train_x, train_target, train_weight, validation_x, &
            validation_target, validation_weight, options)
        call model%initialize([1, 1], status, initialization_seed=23)
        call check(status_ok(status), "weighted fixture model initializes", failures)
        call model%set_parameters([0.15_dp, -0.1_dp], status)
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status, train_weight, validation_weight)
        call check(status_ok(status), "weighted adapter initializes", failures)
        metadata = adapter%metadata()
        call check(metadata%weighted_training .and. metadata%weighted_validation, &
            "weight-presence metadata", failures)
        call check(abs(metadata%training_weight_mass - sum(train_weight)) < 1.0e-14_dp &
            .and. abs(metadata%validation_weight_mass - sum(validation_weight)) &
            < 1.0e-14_dp, "weight-mass metadata", failures)

        parameters = adapter%parameters()
        direction = [0.31_dp, -0.27_dp, 0.17_dp, -0.13_dp, 0.19_dp]
        h_gradient = 2.0e-6_dp
        h_direction = 2.0e-4_dp
        call adapter%value_gradient(parameters, value, gradient, status)
        call check(status_ok(status), "weighted value-gradient status", failures)
        reference_value = oracle_value(parameters, train_x(:, 1), train_target(:, 1), &
            train_weight, validation_x(:, 1), validation_target(:, 1), &
            validation_weight, options%steps, options%centered)
        call check(abs(value - reference_value) < 2.0e-13_dp, &
            "weighted affine trajectory value oracle", failures)
        reference_gradient = oracle_gradient(parameters, train_x(:, 1), &
            train_target(:, 1), train_weight, validation_x(:, 1), &
            validation_target(:, 1), validation_weight, options%steps, &
            options%centered, h_gradient)
        call check(maxval(abs(gradient - reference_gradient)) < 3.0e-7_dp, &
            "weighted five-coordinate gradient oracle", failures)

        call adapter%jvp(parameters, direction, value, tangent, status)
        call check(status_ok(status), "weighted JVP status", failures)
        call check(abs(tangent - dot_product(reference_gradient, direction)) &
            < 4.0e-7_dp, "weighted JVP oracle", failures)
        value_bar = -1.7_dp
        call adapter%vjp(parameters, value_bar, parameter_bar, status)
        call check(status_ok(status), "weighted VJP status", failures)
        call check(abs(dot_product(parameter_bar, direction) - value_bar*tangent) &
            < 3.0e-12_dp, "weighted JVP/VJP adjoint", failures)

        call adapter%hvp(parameters, direction, product, status)
        call check(status_ok(status), "weighted outer HVP status", failures)
        reference_product = (oracle_gradient(parameters + h_direction*direction, &
            train_x(:, 1), train_target(:, 1), train_weight, validation_x(:, 1), &
            validation_target(:, 1), validation_weight, options%steps, &
            options%centered, h_gradient) - oracle_gradient(parameters &
            - h_direction*direction, train_x(:, 1), train_target(:, 1), &
            train_weight, validation_x(:, 1), validation_target(:, 1), &
            validation_weight, options%steps, options%centered, h_gradient)) &
            /(2.0_dp*h_direction)
        call check(maxval(abs(product - reference_product)) < 6.0e-5_dp, &
            "weighted outer HVP oracle", failures)

        call adapter%fortopt(objective, status)
        call check(status_ok(status), "weighted FortOpt context initializes", failures)
        call objective%value_gradient(parameters, objective_value, objective_gradient, &
            status)
        call check(status_ok(status), "weighted FortOpt context evaluates", failures)
        call check(abs(objective_value - value) < 2.0e-13_dp .and. &
            maxval(abs(objective_gradient - gradient)) < 2.0e-13_dp, &
            "weighted FortOpt context matches adapter", failures)

        initial_objective = value
        call model%set_parameters([0.15_dp, -0.1_dp], status)
        options%max_iterations = 80
        options%gradient_tolerance = 1.0e-5_dp
        call mlp_optimize_rmsprop_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status, train_weight, &
            validation_weight)
        call check(status_ok(status), "weighted FortOpt L-BFGS-B status", failures)
        call check(result%converged .and. result%objective <= initial_objective, &
            "weighted FortOpt objective decreases", failures)
    end subroutine test_weighted_products

    subroutine test_weight_contracts(failures)
        integer, intent(inout) :: failures
        type(mlp_t), target :: model
        type(mlp_rmsprop_hypergradient_objective_t) :: adapter
        type(mlp_rmsprop_hypergradient_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: train_x(5, 1), train_target(5, 1), train_weight(5)
        real(dp) :: validation_x(3, 1), validation_target(3, 1)
        real(dp) :: validation_weight(3), before(2), after(2)

        call fixture(train_x, train_target, train_weight, validation_x, &
            validation_target, validation_weight, options)
        call model%initialize([1, 1], status, initialization_seed=23)
        call model%set_parameters([0.15_dp, -0.1_dp], status)
        before = model%parameters()
        train_weight = 0.0_dp
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status, train_weight, validation_weight)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "zero-mass training weights are refused", failures)
        call check(.not. adapter%is_initialized(), &
            "failed weighted adapter remains uninitialized", failures)
        after = model%parameters()
        call check(maxval(abs(after - before)) < 1.0e-14_dp, &
            "malformed weights preserve model parameters", failures)

        call fixture(train_x, train_target, train_weight, validation_x, &
            validation_target, validation_weight, options)
        options%device_kind = FORTML_DEVICE_CUDA
        call adapter%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status, train_weight, validation_weight)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "weighted CUDA trajectory is a typed refusal", failures)
    end subroutine test_weight_contracts

    real(dp) function oracle_value(parameters, train_x, train_target, train_weight, &
            validation_x, validation_target, validation_weight, steps, &
            centered) result(value)
        real(dp), intent(in) :: parameters(5), train_x(:), train_target(:)
        real(dp), intent(in) :: train_weight(:), validation_x(:), validation_target(:)
        real(dp), intent(in) :: validation_weight(:)
        integer, intent(in) :: steps
        logical, intent(in) :: centered
        real(dp) :: theta(2), gradient(2), square_average(2), gradient_average(2)
        real(dp) :: momentum_buffer(2), residual(size(train_x)), validation_residual
        real(dp) :: learning_rate, l2, decay, epsilon, momentum, variance(2)
        real(dp) :: train_mass, validation_mass
        integer :: step, i

        learning_rate = exp(parameters(1))
        l2 = exp(parameters(2))
        decay = parameters(3)
        epsilon = exp(parameters(4))
        momentum = parameters(5)
        theta = [0.15_dp, -0.1_dp]
        square_average = 0.0_dp
        gradient_average = 0.0_dp
        momentum_buffer = 0.0_dp
        train_mass = sum(train_weight)
        do step = 1, steps
            residual = theta(1)*train_x + theta(2) - train_target
            gradient(1) = sum(train_weight*residual*train_x)/train_mass + l2*theta(1)
            gradient(2) = sum(train_weight*residual)/train_mass + l2*theta(2)
            square_average = decay*square_average + &
                (1.0_dp - decay)*gradient*gradient
            if (centered) then
                gradient_average = decay*gradient_average + &
                    (1.0_dp - decay)*gradient
                variance = square_average - gradient_average*gradient_average
            else
                variance = square_average
            end if
            momentum_buffer = momentum*momentum_buffer + &
                gradient/(sqrt(max(variance, 0.0_dp)) + epsilon)
            theta = theta - learning_rate*momentum_buffer
        end do
        validation_mass = sum(validation_weight)
        value = 0.0_dp
        do i = 1, size(validation_x)
            validation_residual = theta(1)*validation_x(i) + theta(2) &
                - validation_target(i)
            value = value + 0.5_dp*validation_weight(i)*validation_residual**2
        end do
        value = value/validation_mass
    end function oracle_value

    function oracle_gradient(parameters, train_x, train_target, train_weight, &
            validation_x, validation_target, validation_weight, steps, centered, &
            h) result(gradient)
        real(dp), intent(in) :: parameters(5), train_x(:), train_target(:)
        real(dp), intent(in) :: train_weight(:), validation_x(:), validation_target(:)
        real(dp), intent(in) :: validation_weight(:), h
        integer, intent(in) :: steps
        logical, intent(in) :: centered
        real(dp) :: gradient(5), plus(5), minus(5)
        integer :: i

        do i = 1, size(parameters)
            plus = parameters
            minus = parameters
            plus(i) = plus(i) + h
            minus(i) = minus(i) - h
            gradient(i) = (oracle_value(plus, train_x, train_target, train_weight, &
                validation_x, validation_target, validation_weight, steps, centered) &
                - oracle_value(minus, train_x, train_target, train_weight, &
                validation_x, validation_target, validation_weight, steps, centered)) &
                /(2.0_dp*h)
        end do
    end function oracle_gradient

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: "//description
        end if
    end subroutine check

end program test_mlp_rmsprop_weighted_hypergradient
