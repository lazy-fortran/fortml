program test_mlp_radam
    !! Independent RAdam recurrence, checkpoint, and device-boundary oracles.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_radam, only: radam_t
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_training_checkpoint_t, mlp_train, MLP_OPTIMIZER_RADAM
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load, &
        MLP_CHECKPOINT_SCHEMA_VERSION
    implicit none

    integer :: failures

    failures = 0
    call test_recurrence(failures)
    call test_checkpoint_resume(failures)
    call test_option_refusals(failures)
    call test_device_refusal(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP RAdam cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP RAdam independent behavioral oracles"

contains

    subroutine test_recurrence(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), expected(2), first(2), second(2)
        real(dp) :: gradient(2), bias1, bias2, rho_inf, rho_t, rectification
        integer :: step

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.0_dp, 0.0_dp], status)
        options%optimizer = MLP_OPTIMIZER_RADAM
        options%max_epochs = 8
        options%learning_rate = 0.15_dp
        options%beta1 = 0.7_dp
        options%beta2 = 0.8_dp
        options%epsilon = 0.03_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, x, target, status, options, state)
        expected = [0.0_dp, 0.0_dp]
        first = 0.0_dp
        second = 0.0_dp
        rho_inf = 2.0_dp/(1.0_dp - options%beta2) - 1.0_dp
        do step = 1, 8
            gradient = [2.0_dp*(expected(1) - 1.0_dp)/3.0_dp, expected(2)]
            first = options%beta1*first + (1.0_dp - options%beta1)*gradient
            second = options%beta2*second + (1.0_dp - options%beta2)*gradient**2
            bias1 = 1.0_dp - options%beta1**step
            bias2 = 1.0_dp - options%beta2**step
            rho_t = rho_inf - 2.0_dp*real(step, dp)*options%beta2**step/bias2
            if (rho_t > 4.0_dp) then
                rectification = sqrt((rho_t - 4.0_dp)*(rho_t - 2.0_dp)*rho_inf/ &
                    ((rho_inf - 4.0_dp)*(rho_inf - 2.0_dp)*rho_t))
                expected = expected - options%learning_rate*rectification*(first/bias1)/ &
                    (sqrt(second/bias2) + options%epsilon)
            else
                expected = expected - options%learning_rate*(first/bias1)
            end if
        end do
        call check(status_ok(status) .and. state%updates == 8, &
            "RAdam MLP update count", failures)
        call check(maxval(abs(model%parameters() - expected)) < 4.0e-13_dp, &
            "RAdam MLP recurrence matches independent oracle", failures)
    end subroutine test_recurrence

    subroutine test_checkpoint_resume(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: full_model, resumed_model
        type(mlp_training_options_t) :: full_options, split_options
        type(mlp_training_state_t) :: full_state, resumed_state
        type(mlp_training_checkpoint_t) :: full_checkpoint, resumed_checkpoint, loaded_checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1), full_theta(2), resumed_theta(2)

        x(:, 1) = [-1.5_dp, -0.5_dp, 0.5_dp, 1.5_dp]
        target(:, 1) = 0.4_dp*x(:, 1) + 0.2_dp
        call full_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call resumed_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call full_model%set_parameters([0.0_dp, 0.0_dp], status)
        call resumed_model%set_parameters([0.0_dp, 0.0_dp], status)
        full_options%optimizer = MLP_OPTIMIZER_RADAM
        full_options%max_epochs = 7
        full_options%learning_rate = 0.08_dp
        full_options%beta1 = 0.6_dp
        full_options%beta2 = 0.85_dp
        full_options%epsilon = 1.0e-4_dp
        full_options%tolerance = 0.0_dp
        full_options%restore_best = .false.
        split_options = full_options
        split_options%max_epochs = 3
        call mlp_train(full_model, x, target, status, full_options, full_state, &
            checkpoint=full_checkpoint)
        call mlp_train(resumed_model, x, target, status, split_options, &
            checkpoint=resumed_checkpoint)
        call check(status_ok(status) .and. resumed_checkpoint%valid() .and. &
            resumed_checkpoint%optimizer == MLP_OPTIMIZER_RADAM .and. &
            resumed_checkpoint%format_version == 10 .and. &
            resumed_checkpoint%adam_step_count == 3 .and. &
            MLP_CHECKPOINT_SCHEMA_VERSION == 10, &
            "RAdam checkpoint metadata and format bump", failures)
        call mlp_checkpoint_save(resumed_checkpoint, "test_mlp_radam_checkpoint.txt", status)
        call mlp_checkpoint_load(loaded_checkpoint, "test_mlp_radam_checkpoint.txt", status)
        call check(status_ok(status) .and. loaded_checkpoint%valid() .and. &
            loaded_checkpoint%format_version == 10, &
            "RAdam formatted checkpoint round trip", failures)
        call mlp_train(resumed_model, x, target, status, full_options, resumed_state, &
            checkpoint=loaded_checkpoint)
        full_theta = full_model%parameters()
        resumed_theta = resumed_model%parameters()
        call check(status_ok(status) .and. resumed_state%updates == full_state%updates .and. &
            maxval(abs(full_theta - resumed_theta)) < 4.0e-13_dp .and. &
            maxval(abs(full_checkpoint%first_moment - loaded_checkpoint%first_moment)) < &
            4.0e-13_dp .and. maxval(abs(full_checkpoint%second_moment - &
            loaded_checkpoint%second_moment)) < 4.0e-13_dp, &
            "RAdam checkpoint continuation matches uninterrupted trajectory", failures)
    end subroutine test_checkpoint_resume

    subroutine test_option_refusals(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_training_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2, 1)

        x(:, 1) = [-1.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        options%optimizer = MLP_OPTIMIZER_RADAM
        options%max_epochs = 1
        options%beta1 = 1.0_dp
        call mlp_train(model, x, target, status, options, state)
        call check(.not. status_ok(status), "unit beta1 refusal", failures)
        options%beta1 = 0.9_dp
        options%beta2 = 1.0_dp
        call mlp_train(model, x, target, status, options, state)
        call check(.not. status_ok(status), "unit beta2 refusal", failures)
        options%beta2 = 0.999_dp
        options%epsilon = 0.0_dp
        call mlp_train(model, x, target, status, options, state)
        call check(.not. status_ok(status), "zero epsilon refusal", failures)
    end subroutine test_option_refusals

    subroutine test_device_refusal(failures)
        integer, intent(inout) :: failures
        type(radam_t) :: optimizer
        type(fortml_device_t) :: device
        type(fortnum_status_t) :: status
        real(dp) :: x(2), gradient(2), before(2)

        call optimizer%initialize(2, status, learning_rate=0.1_dp)
        x = [1.0_dp, -2.0_dp]
        gradient = [0.3_dp, -0.4_dp]
        before = x
        call optimizer%step_device(device, x, gradient, status)
        call check(status%code /= FORTNUM_NOT_IMPLEMENTED, &
            "invalid device is distinct from CUDA refusal", failures)
        device%kind = FORTML_DEVICE_CPU
        call optimizer%step_device(device, x, gradient, status)
        call check(status_ok(status) .and. optimizer%device_supported(FORTML_DEVICE_CPU), &
            "RAdam CPU device dispatch", failures)
        before = x
        device%kind = FORTML_DEVICE_CUDA
        call optimizer%step_device(device, x, gradient, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            maxval(abs(x - before)) == 0.0_dp .and. &
            .not. optimizer%device_supported(FORTML_DEVICE_CUDA), &
            "RAdam CUDA typed refusal preserves parameters", failures)
    end subroutine test_device_refusal

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL [mlp-radam] "//description
        end if
    end subroutine check

end program test_mlp_radam
