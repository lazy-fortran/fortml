program test_mlp_optimizer_groups
    !! Behavioral oracles for deterministic MLP optimizer-group updates.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, &
        mlp_training_checkpoint_t, mlp_optimizer_group_t, &
        mlp_train, MLP_OPTIMIZER_SGD
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load
    implicit none

    integer :: failures

    failures = 0
    call test_one_step(failures)
    call test_invalid_groups(failures)
    call test_checkpoint_roundtrip(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP optimizer-group behavioral cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP optimizer-group behavioral oracles"

contains

    subroutine test_one_step(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_optimizer_group_t) :: group
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2, 1), theta0(2), gradient(2), expected(2)
        integer :: layers(2)

        layers = [1, 1]
        x(:, 1) = [1.0_dp, 2.0_dp]
        target(:, 1) = [2.0_dp, 4.0_dp]
        call model%initialize(layers, status, output_activation=MLP_LINEAR)
        call model%set_parameters([0.5_dp, 0.25_dp], status)
        theta0 = model%parameters()
        ! Independent 0.5*MSE oracle for predictions [0.75, 1.25]:
        ! dL/dtheta = [-3.375, -2.0] for the mean reduction.
        gradient = [-3.375_dp, -2.0_dp]
        call group%initialize("weights", 1, 1, 2.0_dp, status)
        allocate(options%optimizer_groups(1))
        options%optimizer_groups(1) = group
        options%max_epochs = 1
        options%learning_rate = 0.1_dp
        options%optimizer = MLP_OPTIMIZER_SGD
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, x, target, status, options)
        call check(status_ok(status), "one-step grouped training status", failures)
        expected = theta0 - 0.1_dp*gradient
        expected(1) = theta0(1) - 0.2_dp*gradient(1)
        call check(maxval(abs(model%parameters() - expected)) < 5.0e-14_dp, &
            "group multiplier scales only selected update", failures)
    end subroutine test_one_step

    subroutine test_invalid_groups(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_optimizer_group_t) :: first, second
        type(fortnum_status_t) :: status
        real(dp) :: x(1, 1), target(1, 1)
        integer :: layers(2)

        layers = [1, 1]
        x = 1.0_dp
        target = 0.0_dp
        call model%initialize(layers, status, output_activation=MLP_LINEAR)
        call first%initialize("first", 1, 2, 1.0_dp, status)
        call second%initialize("overlap", 2, 2, 1.0_dp, status)
        allocate(options%optimizer_groups(2))
        options%optimizer_groups = [first, second]
        call mlp_train(model, x, target, status, options)
        call check(.not. status_ok(status), "overlapping groups are refused", failures)

        call second%initialize("out-of-range", 3, 3, 1.0_dp, status)
        options%optimizer_groups(2) = second
        call mlp_train(model, x, target, status, options)
        call check(.not. status_ok(status), "out-of-range groups are refused", failures)
    end subroutine test_invalid_groups

    subroutine test_checkpoint_roundtrip(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_training_options_t) :: options
        type(mlp_optimizer_group_t) :: group
        type(mlp_training_checkpoint_t) :: checkpoint, loaded
        type(fortnum_status_t) :: status
        integer :: layers(2), unit
        character(*), parameter :: path = "test_mlp_optimizer_groups_checkpoint.txt"

        layers = [1, 1]
        call model%initialize(layers, status, output_activation=MLP_LINEAR)
        call group%initialize("bias", 2, 2, 0.5_dp, status)
        allocate(options%optimizer_groups(1))
        options%optimizer_groups(1) = group
        options%max_epochs = 1
        options%learning_rate = 0.05_dp
        options%optimizer = MLP_OPTIMIZER_SGD
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, reshape([1.0_dp], [1, 1]), reshape([2.0_dp], [1, 1]), &
            status, options, checkpoint=checkpoint)
        call check(status_ok(status) .and. checkpoint%valid() .and. &
            checkpoint%n_optimizer_groups == 1, &
            "group metadata is captured", failures)
        call mlp_checkpoint_save(checkpoint, path, status)
        call check(status_ok(status), "group checkpoint save", failures)
        call mlp_checkpoint_load(loaded, path, status)
        call check(status_ok(status) .and. loaded%valid() .and. &
            loaded%n_optimizer_groups == 1 .and. &
            loaded%optimizer_group_first(1) == 2 .and. &
            loaded%optimizer_group_last(1) == 2 .and. &
            loaded%optimizer_group_learning_rate_multiplier(1) == 0.5_dp, &
            "group metadata round-trips", failures)
        open(newunit=unit, file=path, status="old", action="read")
        close(unit, status="delete")
    end subroutine test_checkpoint_roundtrip

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_mlp_optimizer_groups
