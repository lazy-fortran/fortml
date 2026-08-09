program test_mlp_training
    !! Independent behavioral checks for deterministic MLP training products.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortopt_objective, only: objective_t
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, &
        mlp_training_state_t, mlp_training_objective_t, &
        mlp_training_checkpoint_t, &
        mlp_loss_diagnostics_t, MLP_REDUCTION_MEAN, MLP_REDUCTION_SUM, &
        MLP_OPTIMIZER_SGD, &
        MLP_EVENT_TRAIN_BEGIN, MLP_EVENT_UPDATE, MLP_EVENT_VALIDATION, &
        MLP_EVENT_EPOCH_END, MLP_EVENT_CHECKPOINT, MLP_EVENT_TRAIN_END, &
        MLP_EVENT_UPDATE_SKIPPED, MLP_PRECISION_FP32, &
        mlp_loss_value_gradient, mlp_loss_hvp, mlp_train
    implicit none

    integer :: failures
    integer :: event_counts(7)
    logical :: event_fail

    failures = 0
    call test_first_adam_step(failures)
    call test_sgd_step_and_checkpoint(failures)
    call test_sgd_option_refusals(failures)
    call test_reproducible_minibatch_and_callback(failures)
    call test_typed_event_callback(failures)
    call test_loss_scale_skip_event(failures)
    call test_resume_checkpoint(failures)
    call test_l2_hyperparameter_product(failures)
    call test_loss_hvp_oracle(failures)
    call test_nonlinear_loss_hvp(failures)
    call test_optimizer_objective_adapter(failures)
    call test_weighted_reductions_and_diagnostics(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP training cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP training independent behavioral oracles"

contains

    subroutine test_typed_event_callback(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(mlp_training_checkpoint_t) :: checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1)

        x(:, 1) = [-1.0_dp, -0.25_dp, 0.25_dp, 1.0_dp]
        target(:, 1) = 0.5_dp*x(:, 1) + 0.2_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        options%max_epochs = 2
        options%batch_size = 2
        options%learning_rate = 0.02_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        options%event_callback => record_training_event
        event_counts = 0
        call mlp_train(model, x, target, status, options, state, &
            validation_x=x, validation_target=target, checkpoint=checkpoint)
        call check(status_ok(status), "typed event callback status", failures)
        call check(event_counts(MLP_EVENT_TRAIN_BEGIN) == 1 .and. &
            event_counts(MLP_EVENT_UPDATE) == state%updates .and. &
            event_counts(MLP_EVENT_UPDATE_SKIPPED) == 0 .and. &
            event_counts(MLP_EVENT_VALIDATION) == state%epochs .and. &
            event_counts(MLP_EVENT_EPOCH_END) == state%epochs .and. &
            event_counts(MLP_EVENT_CHECKPOINT) > state%epochs .and. &
            event_counts(MLP_EVENT_TRAIN_END) == 1 .and. checkpoint%valid(), &
            "typed event sequence and checkpoint", failures)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        event_counts = 0
        event_fail = .true.
        call mlp_train(model, x, target, status, options, state, &
            validation_x=x, validation_target=target)
        call check(.not. status_ok(status) .and. &
            event_counts(MLP_EVENT_VALIDATION) == 1 .and. &
            event_counts(MLP_EVENT_TRAIN_END) == 0, &
            "typed event failure propagation", failures)
        event_fail = .false.
    end subroutine test_typed_event_callback

    subroutine test_loss_scale_skip_event(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2, 1), before(2)

        ! The FP32 forward/gradient boundary remains finite, while the
        ! deliberately huge loss scale overflows the binary64 scaled vector.
        ! This exercises the production skip branch rather than the direct
        ! loss-scale helper and has an independent zero-update oracle.
        x(:, 1) = [1.0_dp, -1.0_dp]
        target(:, 1) = [1.0e38_dp, -1.0e38_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        before = model%parameters()
        options%optimizer = MLP_OPTIMIZER_SGD
        options%precision_kind = MLP_PRECISION_FP32
        options%max_epochs = 1
        options%learning_rate = 0.1_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call options%loss_scale%initialize(status, enabled=.true., &
            initial_scale=1.0e300_dp, maximum_scale=1.0e300_dp, &
            backoff_factor=0.5_dp, growth_interval=1000)
        options%event_callback => record_training_event
        event_counts = 0
        event_fail = .false.
        call mlp_train(model, x, target, status, options, state)
        call check(status_ok(status) .and. state%updates == 0 .and. &
            state%loss_scale%overflow_count == 1 .and. &
            state%loss_scale%skipped_updates == 1 .and. &
            event_counts(MLP_EVENT_UPDATE_SKIPPED) == 1 .and. &
            maxval(abs(model%parameters() - before)) == 0.0_dp, &
            "loss-scale overflow emits skip event without optimizer mutation", failures)
    end subroutine test_loss_scale_skip_event

    subroutine record_training_event(event, epoch, update, loss, &
            validation_loss, gradient_norm, learning_rate, stop, status)
        integer, intent(in) :: event, epoch, update
        real(dp), intent(in) :: loss, validation_loss, gradient_norm, learning_rate
        logical, intent(out) :: stop
        type(fortnum_status_t), intent(out) :: status

        if (event >= 1 .and. event <= 7) event_counts(event) = &
            event_counts(event) + 1
        stop = .false.
        if (event_fail .and. event == MLP_EVENT_VALIDATION) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "test event failure")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine record_training_event

    subroutine test_resume_checkpoint(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: full_model, resumed_model
        type(mlp_training_options_t) :: full_options, split_options
        type(mlp_training_state_t) :: full_state, resumed_state
        type(mlp_training_checkpoint_t) :: full_checkpoint, resumed_checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), target(8, 1), full_theta(7), resumed_theta(7)

        x(:, 1) = [-2.0_dp, -1.5_dp, -1.0_dp, -0.25_dp, 0.25_dp, 1.0_dp, &
            1.5_dp, 2.0_dp]
        target(:, 1) = 0.8_dp*x(:, 1) - 0.35_dp
        call full_model%initialize([1, 2, 1], status, initialization_seed=41)
        call resumed_model%initialize([1, 2, 1], status, initialization_seed=41)
        full_options%max_epochs = 8
        full_options%batch_size = 2
        full_options%accumulation_steps = 2
        full_options%shuffle = .true.
        full_options%shuffle_seed = 991
        full_options%learning_rate = 0.012_dp
        full_options%tolerance = 0.0_dp
        full_options%restore_best = .false.
        full_options%learning_rate_schedule => deterministic_decay
        split_options = full_options
        split_options%max_epochs = 3

        call mlp_train(full_model, x, target, status, full_options, full_state, &
            checkpoint=full_checkpoint)
        call check(status_ok(status), "uninterrupted checkpoint status", failures)
        call check(full_checkpoint%valid() .and. full_checkpoint%epoch == 8 .and. &
            full_checkpoint%updates == full_state%updates .and. &
            full_checkpoint%adam_step_count == full_state%updates, &
            "uninterrupted checkpoint metadata", failures)

        call mlp_train(resumed_model, x, target, status, split_options, &
            checkpoint=resumed_checkpoint)
        call check(status_ok(status), "partial checkpoint status", failures)
        call check(resumed_checkpoint%valid() .and. resumed_checkpoint%epoch == 3 .and. &
            resumed_checkpoint%iterator_position == size(x, 1) + 1 .and. &
            size(resumed_checkpoint%first_moment) == resumed_model%parameter_count(), &
            "partial checkpoint state", failures)
        call mlp_train(resumed_model, x, target, status, full_options, resumed_state, &
            checkpoint=resumed_checkpoint)
        call check(status_ok(status), "resumed checkpoint status", failures)
        full_theta = full_model%parameters()
        resumed_theta = resumed_model%parameters()
        call check(resumed_state%epochs == full_state%epochs .and. &
            resumed_state%updates == full_state%updates .and. &
            maxval(abs(full_theta - resumed_theta)) < 1.0e-14_dp .and. &
            maxval(abs(full_state%loss_history - resumed_state%loss_history)) < &
            1.0e-14_dp .and. &
            maxval(abs(full_checkpoint%parameters - resumed_checkpoint%parameters)) < &
            1.0e-14_dp .and. &
            maxval(abs(full_checkpoint%first_moment - &
            resumed_checkpoint%first_moment)) < 1.0e-14_dp .and. &
            maxval(abs(full_checkpoint%second_moment - &
            resumed_checkpoint%second_moment)) < 1.0e-14_dp .and. &
            all(full_checkpoint%iterator_order == resumed_checkpoint%iterator_order), &
            "uninterrupted/resumed Adam trajectory", failures)
        split_options%batch_size = 1
        call mlp_train(resumed_model, x, target, status, split_options, &
            checkpoint=resumed_checkpoint)
        call check(.not. status_ok(status), "incompatible checkpoint refusal", failures)
    end subroutine test_resume_checkpoint

    subroutine deterministic_decay(epoch, update, base_rate, rate)
        integer, intent(in) :: epoch, update
        real(dp), intent(in) :: base_rate
        real(dp), intent(out) :: rate

        rate = base_rate/(1.0_dp + 0.025_dp*real(epoch - 1, dp)) &
            + 1.0e-5_dp*real(mod(update, 3), dp)
    end subroutine deterministic_decay

    subroutine test_first_adam_step(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2)

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        theta = 0.0_dp
        call model%set_parameters(theta, status)
        options%max_epochs = 1
        options%learning_rate = 0.1_dp
        options%tolerance = 0.0_dp
        call mlp_train(model, x, target, status, options, state)
        theta = model%parameters()
        ! For the zero-initialized affine model, dL/dw=-2/3 and dL/db=0.
        ! Adam's first bias-corrected step is therefore exactly +learning_rate
        ! for w and zero for b, an oracle independent of the implementation.
        call check(status_ok(status), "first Adam status", failures)
        call check(state%updates == 1 .and. state%epochs == 1, &
            "one full-batch update", failures)
        call check(abs(theta(1) - options%learning_rate*(2.0_dp/3.0_dp)/ &
            (2.0_dp/3.0_dp + options%epsilon)) < 2.0e-14_dp .and. &
            abs(theta(2)) < 2.0e-14_dp, "first Adam step oracle", failures)
        call check(size(state%loss_history) == 1 .and. &
            state%final_loss < state%initial_loss, "loss history and decrease", &
            failures)
    end subroutine test_first_adam_step

    subroutine test_sgd_step_and_checkpoint(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: one_step_model, full_model, resumed_model
        type(mlp_training_options_t) :: one_step_options, full_options, split_options
        type(mlp_training_state_t) :: one_step_state, full_state, resumed_state
        type(mlp_training_checkpoint_t) :: full_checkpoint, resumed_checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2), full_theta(2), resumed_theta(2)

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        call one_step_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        theta = 0.0_dp
        call one_step_model%set_parameters(theta, status)
        one_step_options%optimizer = MLP_OPTIMIZER_SGD
        one_step_options%momentum = 0.8_dp
        one_step_options%max_epochs = 1
        one_step_options%learning_rate = 0.1_dp
        one_step_options%tolerance = 0.0_dp
        one_step_options%restore_best = .false.
        call mlp_train(one_step_model, x, target, status, one_step_options, &
            one_step_state)
        theta = one_step_model%parameters()
        call check(status_ok(status), "first SGD status", failures)
        call check(one_step_state%updates == 1 .and. &
            maxval(abs(theta - [1.0_dp/15.0_dp, 0.0_dp])) < 1.0e-14_dp, &
            "first SGD step independent oracle", failures)

        call full_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call resumed_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call full_model%set_parameters([0.0_dp, 0.0_dp], status)
        call resumed_model%set_parameters([0.0_dp, 0.0_dp], status)
        full_options = one_step_options
        full_options%max_epochs = 2
        split_options = full_options
        split_options%max_epochs = 1
        call mlp_train(full_model, x, target, status, full_options, full_state, &
            checkpoint=full_checkpoint)
        call mlp_train(resumed_model, x, target, status, split_options, &
            checkpoint=resumed_checkpoint)
        call check(status_ok(status) .and. resumed_checkpoint%valid() .and. &
            resumed_checkpoint%optimizer == MLP_OPTIMIZER_SGD .and. &
            maxval(abs(resumed_checkpoint%first_moment - &
            [-2.0_dp/3.0_dp, 0.0_dp])) < 1.0e-14_dp, &
            "SGD checkpoint velocity", failures)
        call mlp_train(resumed_model, x, target, status, full_options, resumed_state, &
            checkpoint=resumed_checkpoint)
        full_theta = full_model%parameters()
        resumed_theta = resumed_model%parameters()
        call check(status_ok(status) .and. resumed_state%updates == full_state%updates &
            .and. maxval(abs(full_theta - resumed_theta)) < 1.0e-14_dp .and. &
            maxval(abs(full_state%loss_history - resumed_state%loss_history)) < &
            1.0e-14_dp, "resumed SGD trajectory", failures)
    end subroutine test_sgd_step_and_checkpoint

    subroutine test_sgd_option_refusals(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2, 1)

        x(:, 1) = [-1.0_dp, 1.0_dp]
        target(:, 1) = [-1.0_dp, 1.0_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        options%optimizer = 99
        options%max_epochs = 1
        call mlp_train(model, x, target, status, options, state)
        call check(.not. status_ok(status), "invalid optimizer refusal", failures)
        options%optimizer = MLP_OPTIMIZER_SGD
        options%nesterov = .true.
        options%momentum = 0.0_dp
        call mlp_train(model, x, target, status, options, state)
        call check(.not. status_ok(status), "zero-momentum Nesterov refusal", failures)
    end subroutine test_sgd_option_refusals

    subroutine test_reproducible_minibatch_and_callback(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: first_model, second_model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: first_state, second_state
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), target(6, 1), first_theta(3), second_theta(3)

        x(:, 1) = [-2.0_dp, -1.0_dp, -0.5_dp, 0.5_dp, 1.0_dp, 2.0_dp]
        target(:, 1) = 0.75_dp*x(:, 1) - 0.2_dp
        call first_model%initialize([1, 2, 1], status, &
            output_activation=MLP_LINEAR, initialization_seed=31)
        call second_model%initialize([1, 2, 1], status, &
            output_activation=MLP_LINEAR, initialization_seed=31)
        options%max_epochs = 12
        options%batch_size = 2
        options%shuffle = .true.
        options%shuffle_seed = 123
        options%learning_rate = 0.02_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        options%callback => stop_at_three
        call mlp_train(first_model, x, target, status, options, first_state)
        call mlp_train(second_model, x, target, status, options, second_state)
        first_theta = first_model%parameters()
        second_theta = second_model%parameters()
        call check(status_ok(status), "mini-batch status", failures)
        call check(first_state%early_stopped .and. first_state%epochs == 3, &
            "callback early stop", failures)
        call check(first_state%updates == 9, "mini-batch update count", failures)
        call check(second_state%epochs == first_state%epochs .and. &
            maxval(abs(first_theta - second_theta)) < 1.0e-14_dp .and. &
            maxval(abs(first_state%loss_history - second_state%loss_history)) < &
            1.0e-14_dp, "reproducible shuffled training", failures)
    end subroutine test_reproducible_minibatch_and_callback

    subroutine test_l2_hyperparameter_product(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2), gradient(2)
        real(dp) :: value, l2_bar, plus, minus, h
        real(dp) :: gradient_plus(2), gradient_minus(2), ignored

        x(:, 1) = [-1.0_dp, 0.0_dp, 2.0_dp]
        target(:, 1) = [0.5_dp, -0.25_dp, 1.5_dp]
        theta = [0.3_dp, -0.2_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters(theta, status)
        call mlp_loss_value_gradient(model, x, target, 0.4_dp, value, &
            gradient, l2_bar, status)
        h = 1.0e-6_dp
        call mlp_loss_value_gradient(model, x, target, 0.4_dp + h, plus, &
            gradient_plus, ignored, status)
        call mlp_loss_value_gradient(model, x, target, 0.4_dp - h, minus, &
            gradient_minus, ignored, status)
        call check(status_ok(status), "loss product status", failures)
        call check(abs(l2_bar - 0.5_dp*sum(theta*theta)) < 1.0e-14_dp, &
            "analytic L2 hyperparameter derivative", failures)
        call check(abs(l2_bar - (plus - minus)/(2.0_dp*h)) < 2.0e-10_dp, &
            "finite-difference L2 derivative", failures)
    end subroutine test_l2_hyperparameter_product

    subroutine test_loss_hvp_oracle(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2), dtheta(2)
        real(dp) :: gradient_plus(2), gradient_minus(2), parameter_hvp(2)
        real(dp) :: value, l2_bar, ignored, l2_hvp, finite_l2_hvp
        real(dp) :: h, l2, l2_direction

        x(:, 1) = [-1.0_dp, 0.5_dp, 2.0_dp]
        target(:, 1) = [0.5_dp, -0.25_dp, 1.5_dp]
        theta = [0.3_dp, -0.2_dp]
        dtheta = [-0.4_dp, 0.7_dp]
        l2 = 0.4_dp
        l2_direction = -0.15_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters(theta, status)
        call mlp_loss_hvp(model, x, target, l2, dtheta, l2_direction, &
            parameter_hvp, l2_hvp, status)
        h = 1.0e-6_dp
        call model%set_parameters(theta + h*dtheta, status)
        call mlp_loss_value_gradient(model, x, target, l2 + h*l2_direction, &
            value, gradient_plus, l2_bar, status)
        call model%set_parameters(theta - h*dtheta, status)
        call mlp_loss_value_gradient(model, x, target, l2 - h*l2_direction, &
            value, gradient_minus, ignored, status)
        call model%set_parameters(theta, status)
        finite_l2_hvp = (0.5_dp*sum((theta + h*dtheta)**2) - &
            0.5_dp*sum((theta - h*dtheta)**2))/(2.0_dp*h)
        call check(status_ok(status), "loss HVP status", failures)
        call check(maxval(abs(parameter_hvp - &
            (gradient_plus - gradient_minus)/(2.0_dp*h))) < 2.0e-8_dp, &
            "joint parameter HVP finite difference", failures)
        call check(abs(l2_hvp - finite_l2_hvp) < 2.0e-10_dp, &
            "mixed L2 HVP finite difference", failures)
    end subroutine test_loss_hvp_oracle

    subroutine test_nonlinear_loss_hvp(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(7), dtheta(7)
        real(dp) :: parameter_hvp(7), gradient_plus(7), gradient_minus(7)
        real(dp) :: value, l2_bar, ignored, l2_hvp
        real(dp) :: h, l2, l2_direction

        x(:, 1) = [-0.8_dp, 0.2_dp, 1.1_dp]
        target(:, 1) = [0.4_dp, -0.1_dp, 0.9_dp]
        theta = [0.2_dp, -0.3_dp, 0.1_dp, 0.05_dp, 0.7_dp, -0.4_dp, 0.2_dp]
        dtheta = [-0.1_dp, 0.15_dp, 0.07_dp, -0.12_dp, 0.2_dp, 0.09_dp, -0.18_dp]
        l2 = 0.25_dp
        l2_direction = 0.11_dp
        call model%initialize([1, 2, 1], status, initialization_seed=9)
        call model%set_parameters(theta, status)
        call mlp_loss_hvp(model, x, target, l2, dtheta, l2_direction, &
            parameter_hvp, l2_hvp, status)
        h = 1.0e-6_dp
        call model%set_parameters(theta + h*dtheta, status)
        call mlp_loss_value_gradient(model, x, target, l2 + h*l2_direction, &
            value, gradient_plus, l2_bar, status)
        call model%set_parameters(theta - h*dtheta, status)
        call mlp_loss_value_gradient(model, x, target, l2 - h*l2_direction, &
            value, gradient_minus, ignored, status)
        call model%set_parameters(theta, status)
        call check(status_ok(status), "nonlinear HVP status", failures)
        call check(maxval(abs(parameter_hvp - &
            (gradient_plus - gradient_minus)/(2.0_dp*h))) < 3.0e-8_dp, &
            "nonlinear joint HVP finite difference", failures)
        call check(abs(l2_hvp - dot_product(theta, dtheta)) < 2.0e-10_dp, &
            "nonlinear mixed L2 HVP", failures)
    end subroutine test_nonlinear_loss_hvp

    subroutine test_optimizer_objective_adapter(failures)
        integer, intent(inout) :: failures
        type(mlp_t), target :: model
        type(mlp_training_objective_t), target :: adapter
        type(objective_t) :: fortopt_objective
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), point(3), direction(3)
        real(dp) :: gradient(3), gradient_plus(3), gradient_minus(3)
        real(dp) :: product(3), finite_gradient(3)
        real(dp) :: value, plus, minus, fortopt_value
        real(dp) :: fortopt_gradient(3), h
        real(dp) :: tangent, vjp_gradient(3), output_bar
        integer :: i

        x(:, 1) = [-1.0_dp, 0.5_dp, 2.0_dp]
        target(:, 1) = [0.5_dp, -0.25_dp, 1.5_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.3_dp, -0.2_dp], status)
        call adapter%initialize(model, x, target, 0.4_dp, status, &
            optimize_l2=.true.)
        point = adapter%parameters()
        direction = [-0.4_dp, 0.7_dp, -0.15_dp]
        call adapter%value_gradient(point, value, gradient, status)
        h = 1.0e-6_dp
        do i = 1, size(point)
            call adapter%value_gradient(point + unit_direction(size(point), i)*h, &
                plus, gradient_plus, status)
            call adapter%value_gradient(point - unit_direction(size(point), i)*h, &
                minus, gradient_minus, status)
            finite_gradient(i) = (plus - minus)/(2.0_dp*h)
        end do
        call adapter%value_gradient(point, value, gradient, status)
        call adapter%hvp(point, direction, product, status)
        call adapter%value_gradient(point + h*direction, plus, gradient_plus, &
            status)
        call adapter%value_gradient(point - h*direction, minus, gradient_minus, &
            status)
        call adapter%value_gradient(point, value, gradient, status)
        call adapter%fortopt(fortopt_objective, status)
        call fortopt_objective%value_gradient(point, fortopt_value, &
            fortopt_gradient, status)
        call adapter%jvp(point, direction, value, tangent, status)
        call adapter%value_gradient(point + h*direction, plus, gradient_plus, &
            status)
        call adapter%value_gradient(point - h*direction, minus, gradient_minus, &
            status)
        output_bar = 1.7_dp
        call adapter%vjp(point, output_bar, vjp_gradient, status)
        call adapter%value_gradient(point, value, gradient, status)
        call check(status_ok(status), "optimizer objective adapter status", failures)
        call check(maxval(abs(gradient - finite_gradient)) < 3.0e-9_dp, &
            "optimizer objective gradient finite difference", failures)
        call check(maxval(abs(product - &
            (gradient_plus - gradient_minus)/(2.0_dp*h))) < 3.0e-8_dp, &
            "optimizer objective HVP finite difference", failures)
        call check(abs(fortopt_value - value) < 1.0e-14_dp .and. &
            maxval(abs(fortopt_gradient - gradient)) < 1.0e-14_dp, &
            "FortOpt objective adapter", failures)
        call check(abs(tangent - (plus - minus)/(2.0_dp*h)) < 3.0e-8_dp, &
            "optimizer objective JVP finite difference", failures)
        call check(maxval(abs(vjp_gradient - output_bar*gradient)) < &
            1.0e-14_dp, "optimizer objective scalar VJP", failures)
    end subroutine test_optimizer_objective_adapter

    subroutine test_weighted_reductions_and_diagnostics(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_loss_diagnostics_t) :: diagnostics
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), weights(3), theta(2)
        real(dp) :: gradient(2), gradient_mean(2), gradient_plus(2), gradient_minus(2)
        real(dp) :: mean_value, sum_value, plus, minus, ignored, l2
        real(dp) :: h, expected_mean, expected_sum, direct_data, direct_reg

        x(:, 1) = [-1.0_dp, 0.5_dp, 2.0_dp]
        target(:, 1) = [0.5_dp, -0.25_dp, 1.5_dp]
        weights = [1.0_dp, 2.0_dp, 0.5_dp]
        theta = [0.3_dp, -0.2_dp]
        l2 = 0.4_dp
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters(theta, status)
        call mlp_loss_value_gradient(model, x, target, l2, mean_value, gradient, &
            ignored, status, sample_weight=weights, reduction=MLP_REDUCTION_MEAN, &
            diagnostics=diagnostics)
        call check(status_ok(status), "weighted mean status", failures)
        gradient_mean = gradient
        direct_data = 0.5_dp*sum(weights*(theta(1)*x(:, 1) + theta(2) - &
            target(:, 1))**2)/sum(weights)
        direct_reg = 0.5_dp*l2*sum(theta*theta)
        expected_mean = direct_data + direct_reg
        call check(abs(mean_value - expected_mean) < 1.0e-14_dp .and. &
            abs(diagnostics%data_loss - direct_data) < 1.0e-14_dp .and. &
            abs(diagnostics%regularization_loss - direct_reg) < 1.0e-14_dp .and. &
            abs(diagnostics%weight_mass - sum(weights)) < 1.0e-14_dp .and. &
            diagnostics%sample_count == 3, "named loss diagnostics", failures)

        call mlp_loss_value_gradient(model, x, target, l2, sum_value, gradient, &
            ignored, status, sample_weight=weights, reduction=MLP_REDUCTION_SUM)
        expected_sum = (mean_value - 0.5_dp*l2*sum(theta*theta))*sum(weights) + &
            0.5_dp*l2*sum(theta*theta)
        call check(abs(sum_value - expected_sum) < 1.0e-14_dp, &
            "weighted sum/mean reduction relation", failures)

        h = 1.0e-6_dp
        call model%set_parameters(theta + h*[1.0_dp, 0.0_dp], status)
        call mlp_loss_value_gradient(model, x, target, l2, plus, gradient_plus, &
            ignored, status, sample_weight=weights, reduction=MLP_REDUCTION_MEAN)
        call model%set_parameters(theta - h*[1.0_dp, 0.0_dp], status)
        call mlp_loss_value_gradient(model, x, target, l2, minus, gradient_minus, &
            ignored, status, sample_weight=weights, reduction=MLP_REDUCTION_MEAN)
        call check(abs(gradient_mean(1) - (plus - minus)/(2.0_dp*h)) < 2.0e-8_dp, &
            "weighted gradient first-coordinate oracle", failures)
        call model%set_parameters(theta + h*[0.0_dp, 1.0_dp], status)
        call mlp_loss_value_gradient(model, x, target, l2, plus, gradient_plus, &
            ignored, status, sample_weight=weights, reduction=MLP_REDUCTION_MEAN)
        call model%set_parameters(theta - h*[0.0_dp, 1.0_dp], status)
        call mlp_loss_value_gradient(model, x, target, l2, minus, gradient_minus, &
            ignored, status, sample_weight=weights, reduction=MLP_REDUCTION_MEAN)
        call model%set_parameters(theta, status)
        call check(abs(gradient_mean(2) - (plus - minus)/(2.0_dp*h)) < 2.0e-8_dp, &
            "weighted gradient second-coordinate oracle", failures)

        weights = 0.0_dp
        call mlp_loss_value_gradient(model, x, target, l2, mean_value, gradient, &
            ignored, status, sample_weight=weights)
        call check(.not. status_ok(status), "zero-support weight refusal", failures)
        weights = [1.0_dp, -0.5_dp, 1.0_dp]
        call mlp_loss_value_gradient(model, x, target, l2, mean_value, gradient, &
            ignored, status, sample_weight=weights)
        call check(.not. status_ok(status), "negative weight refusal", failures)
        call mlp_loss_value_gradient(model, x, target, l2, mean_value, gradient, &
            ignored, status, reduction=99)
        call check(.not. status_ok(status), "invalid reduction refusal", failures)
    end subroutine test_weighted_reductions_and_diagnostics

    function unit_direction(n, index) result(direction)
        integer, intent(in) :: n, index
        real(dp) :: direction(n)

        direction = 0.0_dp
        direction(index) = 1.0_dp
    end function unit_direction

    subroutine stop_at_three(epoch, loss, gradient_norm, stop)
        integer, intent(in) :: epoch
        real(dp), intent(in) :: loss, gradient_norm
        logical, intent(out) :: stop

        stop = epoch >= 3
        if (loss < 0.0_dp .or. gradient_norm < 0.0_dp) stop = .false.
    end subroutine stop_at_three

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

end program test_mlp_training
