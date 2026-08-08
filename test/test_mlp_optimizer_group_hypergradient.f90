program test_mlp_optimizer_group_hypergradient
    !! Independent oracle for optimizer-group trajectory products.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_optimizer_group_t, mlp_training_options_t, &
        mlp_training_state_t, mlp_train, MLP_OPTIMIZER_SGD
    use fortml_mlp_optimizer_group_hypergradient, only: &
        mlp_optimizer_group_hypergradient_options_t, &
        mlp_optimizer_group_hypergradient_objective_t, &
        mlp_optimizer_group_hypergradient_result_t, &
        mlp_optimize_optimizer_group_hyperparameters
    implicit none

    integer :: failures

    failures = 0
    call test_products_and_independent_fd(failures)
    call test_trainer_parity(failures)
    call test_clipped_trajectory_and_boundary(failures)
    call test_fortopt_and_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP optimizer-group hypergradient cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP optimizer-group hypergradient independent behavioral oracles"

contains

    subroutine fixture(model, options, train_x, train_target, validation_x, validation_target, &
            status)
        type(mlp_t), intent(out) :: model
        type(mlp_optimizer_group_hypergradient_options_t), intent(out) :: options
        real(dp), intent(out) :: train_x(4, 1), train_target(4, 1)
        real(dp), intent(out) :: validation_x(3, 1), validation_target(3, 1)
        type(fortnum_status_t), intent(out) :: status
        type(mlp_optimizer_group_t) :: weight_group, bias_group

        train_x(:, 1) = [-1.0_dp, -0.2_dp, 0.7_dp, 1.5_dp]
        train_target(:, 1) = 0.6_dp*train_x(:, 1) - 0.15_dp
        validation_x(:, 1) = [-0.8_dp, 0.4_dp, 1.2_dp]
        validation_target(:, 1) = 0.6_dp*validation_x(:, 1) - 0.15_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        if (.not. status_ok(status)) return
        call model%set_parameters([0.25_dp, 0.1_dp], status)
        if (.not. status_ok(status)) return
        call weight_group%initialize("weight", 1, 1, 0.7_dp, status)
        if (.not. status_ok(status)) return
        call bias_group%initialize("bias", 2, 2, 1.3_dp, status)
        if (.not. status_ok(status)) return
        options%steps = 4
        options%learning_rate = 0.08_dp
        options%l2 = 0.03_dp
        options%lower_log_learning_rate = -6.0_dp
        options%upper_log_learning_rate = 0.0_dp
        options%lower_log_l2 = -7.0_dp
        options%upper_log_l2 = 0.0_dp
        options%lower_log_multiplier = -4.0_dp
        options%upper_log_multiplier = 4.0_dp
        allocate(options%groups(2))
        options%groups = [weight_group, bias_group]
    end subroutine fixture

    subroutine test_products_and_independent_fd(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_optimizer_group_hypergradient_options_t) :: options
        type(mlp_optimizer_group_hypergradient_objective_t) :: objective
        type(fortnum_status_t) :: status
        real(dp) :: train_x(4, 1), train_target(4, 1)
        real(dp) :: validation_x(3, 1), validation_target(3, 1)
        real(dp), allocatable :: parameters(:), gradient(:), direction(:)
        real(dp), allocatable :: plus(:), minus(:)
        real(dp) :: value, tangent, value_plus, value_minus, fd, scale
        integer :: i

        call fixture(model, options, train_x, train_target, validation_x, validation_target, status)
        call objective%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        call check(status_ok(status) .and. objective%parameter_count() == 4, &
            "optimizer-group objective setup", failures)
        parameters = objective%parameters()
        allocate(gradient(size(parameters)), direction(size(parameters)), &
            plus(size(parameters)), minus(size(parameters)))
        call objective%value_gradient(parameters, value, gradient, status)
        call check(status_ok(status) .and. ieee_is_finite(value) .and. &
            all(ieee_is_finite(gradient)), "optimizer-group value/gradient", failures)
        direction = [0.17_dp, -0.13_dp, 0.11_dp, -0.09_dp]
        call objective%jvp(parameters, direction, value, tangent, status)
        call check(status_ok(status) .and. abs(tangent-dot_product(gradient, direction)) < 1.0e-11_dp, &
            "optimizer-group JVP contraction", failures)
        do i = 1, size(parameters)
            plus = parameters
            minus = parameters
            scale = 2.0e-6_dp
            plus(i) = plus(i) + scale
            minus(i) = minus(i) - scale
            call objective%jvp(plus, direction*0.0_dp, value_plus, tangent, status)
            call check(status_ok(status), "optimizer-group plus FD branch", failures)
            call objective%jvp(minus, direction*0.0_dp, value_minus, tangent, status)
            call check(status_ok(status), "optimizer-group minus FD branch", failures)
            fd = (value_plus-value_minus)/(2.0_dp*scale)
            call check(abs(fd-gradient(i)) < 2.0e-8_dp, &
                "optimizer-group central-difference product", failures)
        end do
        call objective%vjp(parameters, 1.4_dp, direction, status)
        call check(status_ok(status) .and. maxval(abs(direction-1.4_dp*gradient)) < 1.0e-11_dp, &
            "optimizer-group scalar VJP", failures)
    end subroutine test_products_and_independent_fd

    subroutine test_trainer_parity(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: objective_model, trainer_model
        type(mlp_optimizer_group_hypergradient_options_t) :: options
        type(mlp_training_options_t) :: training_options
        type(mlp_optimizer_group_hypergradient_objective_t) :: objective
        type(fortnum_status_t) :: status
        real(dp) :: train_x(4, 1), train_target(4, 1)
        real(dp) :: validation_x(3, 1), validation_target(3, 1)
        real(dp) :: value, gradient(4), expected(2)

        call fixture(objective_model, options, train_x, train_target, validation_x, &
            validation_target, status)
        call objective%initialize(objective_model, train_x, train_target, validation_x, &
            validation_target, options, status)
        call objective%value_gradient(objective%parameters(), value, gradient, status)
        expected = objective_model%parameters()
        call trainer_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call trainer_model%set_parameters([0.25_dp, 0.1_dp], status)
        training_options%max_epochs = options%steps
        training_options%learning_rate = options%learning_rate
        training_options%l2 = options%l2
        training_options%optimizer = MLP_OPTIMIZER_SGD
        training_options%tolerance = 0.0_dp
        training_options%restore_best = .false.
        allocate(training_options%optimizer_groups(2))
        training_options%optimizer_groups = options%groups
        call mlp_train(trainer_model, train_x, train_target, status, training_options)
        call check(status_ok(status) .and. maxval(abs(expected-trainer_model%parameters())) < 1.0e-13_dp, &
            "optimizer-group trajectory matches production trainer", failures)
    end subroutine test_trainer_parity

    subroutine test_clipped_trajectory_and_boundary(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model, trainer_model
        type(mlp_optimizer_group_hypergradient_options_t) :: options
        type(mlp_training_options_t) :: training_options
        type(mlp_training_state_t) :: training_state
        type(mlp_optimizer_group_hypergradient_objective_t) :: objective
        type(fortnum_status_t) :: status
        real(dp) :: train_x(4, 1), train_target(4, 1)
        real(dp) :: validation_x(3, 1), validation_target(3, 1)
        real(dp), allocatable :: parameters(:), gradient(:), plus(:), minus(:)
        real(dp) :: value, value_plus, value_minus, fd, h, clip, boundary_clip
        integer :: i

        call fixture(model, options, train_x, train_target, validation_x, &
            validation_target, status)
        options%gradient_clip_norm = 0.1_dp
        call objective%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        call check(status_ok(status), "clipped optimizer-group setup", failures)
        clip = options%gradient_clip_norm
        parameters = objective%parameters()
        allocate(gradient(size(parameters)), plus(size(parameters)), minus(size(parameters)))
        call objective%value_gradient(parameters, value, gradient, status)
        call check(status_ok(status) .and. abs(value-clipped_oracle(parameters, clip)) < &
            2.0e-13_dp, "clipped trajectory matches independent oracle", failures)
        h = 2.0e-6_dp
        do i = 1, size(parameters)
            plus = parameters
            minus = parameters
            plus(i) = plus(i) + h
            minus(i) = minus(i) - h
            value_plus = clipped_oracle(plus, clip)
            value_minus = clipped_oracle(minus, clip)
            fd = (value_plus-value_minus)/(2.0_dp*h)
            call check(abs(fd-gradient(i)) < 2.0e-8_dp, &
                "clipped trajectory derivative oracle", failures)
        end do

        call trainer_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call trainer_model%set_parameters([0.25_dp, 0.1_dp], status)
        training_options%max_epochs = options%steps
        training_options%learning_rate = options%learning_rate
        training_options%l2 = options%l2
        training_options%gradient_clip_norm = options%gradient_clip_norm
        training_options%optimizer = MLP_OPTIMIZER_SGD
        training_options%tolerance = 0.0_dp
        training_options%restore_best = .false.
        allocate(training_options%optimizer_groups(2))
        training_options%optimizer_groups = options%groups
        call mlp_train(trainer_model, train_x, train_target, status, training_options, &
            training_state)
        call check(status_ok(status) .and. training_state%gradient_clipped_updates == &
            options%steps .and. abs(clipped_oracle(parameters, clip)-value) < &
            2.0e-13_dp .and. maxval(abs(trainer_model%parameters()- &
            clipped_parameters(parameters, clip))) < 2.0e-13_dp, &
            "clipped trajectory matches production trainer", failures)

        call fixture(model, options, train_x, train_target, validation_x, &
            validation_target, status)
        boundary_clip = sqrt(0.26075_dp**2 + 0.1655_dp**2)
        options%gradient_clip_norm = boundary_clip
        call objective%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        parameters = objective%parameters()
        call objective%value_gradient(parameters, value, gradient, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(gradient == 0.0_dp), &
            "clipping active-set boundary refusal", failures)
    end subroutine test_clipped_trajectory_and_boundary

    function clipped_oracle(parameters, clip) result(value)
        real(dp), intent(in) :: parameters(:), clip
        real(dp) :: value
        real(dp) :: theta(2), gradient(2), residual(4), validation_residual(3)
        real(dp), parameter :: x(4) = [-1.0_dp, -0.2_dp, 0.7_dp, 1.5_dp]
        real(dp), parameter :: target(4) = [-0.75_dp, -0.27_dp, 0.27_dp, 0.75_dp]
        real(dp), parameter :: validation_x(3) = [-0.8_dp, 0.4_dp, 1.2_dp]
        real(dp), parameter :: validation_target(3) = [-0.63_dp, 0.09_dp, 0.57_dp]
        real(dp) :: learning_rate, l2, scales(2), norm
        integer :: step

        learning_rate = exp(parameters(1))
        l2 = exp(parameters(2))
        scales = exp(parameters(3:4))
        theta = [0.25_dp, 0.1_dp]
        do step = 1, 4
            residual = x*theta(1) + theta(2) - target
            gradient = [sum(residual*x)/4.0_dp + l2*theta(1), &
                sum(residual)/4.0_dp + l2*theta(2)]
            norm = sqrt(sum(gradient*gradient))
            if (clip > 0.0_dp .and. norm > clip) gradient = gradient*clip/norm
            theta = theta-learning_rate*scales*gradient
        end do
        validation_residual = validation_x*theta(1) + theta(2) - validation_target
        value = 0.5_dp*sum(validation_residual*validation_residual)/3.0_dp
    end function clipped_oracle

    function clipped_parameters(parameters, clip) result(theta)
        real(dp), intent(in) :: parameters(:), clip
        real(dp) :: theta(2), gradient(2), residual(4), norm
        real(dp), parameter :: x(4) = [-1.0_dp, -0.2_dp, 0.7_dp, 1.5_dp]
        real(dp), parameter :: target(4) = [-0.75_dp, -0.27_dp, 0.27_dp, 0.75_dp]
        real(dp) :: learning_rate, l2, scales(2)
        integer :: step

        learning_rate = exp(parameters(1))
        l2 = exp(parameters(2))
        scales = exp(parameters(3:4))
        theta = [0.25_dp, 0.1_dp]
        do step = 1, 4
            residual = x*theta(1) + theta(2) - target
            gradient = [sum(residual*x)/4.0_dp + l2*theta(1), &
                sum(residual)/4.0_dp + l2*theta(2)]
            norm = sqrt(sum(gradient*gradient))
            if (clip > 0.0_dp .and. norm > clip) gradient = gradient*clip/norm
            theta = theta-learning_rate*scales*gradient
        end do
    end function clipped_parameters

    subroutine test_fortopt_and_refusals(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_optimizer_group_hypergradient_options_t) :: options
        type(mlp_optimizer_group_hypergradient_objective_t) :: objective
        type(mlp_optimizer_group_hypergradient_result_t) :: result
        type(fortnum_status_t) :: status
        real(dp) :: train_x(4, 1), train_target(4, 1)
        real(dp) :: validation_x(3, 1), validation_target(3, 1)
        real(dp) :: value, gradient(4), direction(4), product(4)
        type(mlp_optimizer_group_t) :: overlap

        call fixture(model, options, train_x, train_target, validation_x, validation_target, status)
        options%gradient_clip_norm = 0.1_dp
        options%max_iterations = 12
        options%gradient_tolerance = 1.0e-4_dp
        call mlp_optimize_optimizer_group_hyperparameters(model, train_x, train_target, &
            validation_x, validation_target, options, result, status)
        call check(status_ok(status) .or. status%code == FORTNUM_CONVERGENCE_ERROR, &
            "optimizer-group FortOpt adapter returns a defined result", failures)
        call check(ieee_is_finite(result%objective) .and. size(result%multiplier) == 2 .and. &
            all(result%multiplier > 0.0_dp), "optimizer-group physical result", failures)

        call fixture(model, options, train_x, train_target, validation_x, validation_target, status)
        options%device_kind = FORTML_DEVICE_CUDA
        call objective%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "optimizer-group CUDA refusal", failures)

        ! The inner trajectory uses an exact MLP HVP, but an outer HVP would
        ! require third network derivatives.  The public method must refuse
        ! explicitly rather than silently finite-differencing the objective.
        call fixture(model, options, train_x, train_target, validation_x, validation_target, status)
        call objective%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        direction = [0.2_dp, -0.1_dp, 0.07_dp, -0.05_dp]
        product = 1.0_dp
        call objective%hvp(objective%parameters(), direction, product, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(product == 0.0_dp), &
            "optimizer-group outer HVP typed refusal", failures)
        product = 1.0_dp
        call objective%hvp(objective%parameters(), direction(:3), product, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR .and. all(product == 0.0_dp), &
            "optimizer-group HVP shape refusal", failures)

        call fixture(model, options, train_x, train_target, validation_x, validation_target, status)
        call overlap%initialize("overlap", 1, 2, 1.0_dp, status)
        options%groups(2) = overlap
        call objective%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        call check(.not. status_ok(status), "optimizer-group overlap refusal", failures)
    end subroutine test_fortopt_and_refusals

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_mlp_optimizer_group_hypergradient
