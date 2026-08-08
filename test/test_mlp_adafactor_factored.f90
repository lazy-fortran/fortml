program test_mlp_adafactor_factored
    !! Independent row/column Adafactor oracle plus MLP integration contract.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_training_checkpoint_t, mlp_train, MLP_OPTIMIZER_ADAFACTOR
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load
    use fortml_adafactor_factored, only: adafactor_factored_t, &
        adafactor_block_spec_t
    implicit none

    integer :: failures

    failures = 0
    call test_factored_recurrence(failures)
    call test_mlp_integration_and_checkpoint(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL factored Adafactor cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS factored Adafactor independent behavioral oracles"

contains

    subroutine test_factored_recurrence(failures)
        integer, intent(inout) :: failures
        type(adafactor_factored_t) :: optimizer
        type(adafactor_block_spec_t) :: blocks(2)
        type(fortnum_status_t) :: status
        real(dp) :: x(6), expected(6), gradient(6)
        real(dp), allocatable :: dense(:)
        real(dp) :: row(2), column(2), vector_moment(2), dense_expected(6)
        real(dp) :: decay, epsilon, clip_threshold, rate, update_rms, clip_scale
        integer :: step, i, j, index

        blocks(1) = adafactor_block_spec_t(first=1, last=4, rows=2, columns=2, factored=.true.)
        blocks(2) = adafactor_block_spec_t(first=5, last=6, rows=2, columns=1, factored=.false.)
        decay = 0.6_dp
        epsilon = 0.03_dp
        clip_threshold = 0.8_dp
        rate = 0.2_dp
        call optimizer%initialize(6, blocks, status, learning_rate=rate, decay=decay, &
            epsilon=epsilon, clip_threshold=clip_threshold)
        x = [0.5_dp, -0.4_dp, 0.3_dp, -0.2_dp, 0.1_dp, -0.05_dp]
        expected = x
        gradient = [0.7_dp, -0.2_dp, 0.4_dp, -0.6_dp, 0.3_dp, -0.5_dp]
        row = 0.0_dp
        column = 0.0_dp
        vector_moment = 0.0_dp
        do step = 1, 3
            row = decay*row
            column = decay*column
            do j = 1, 2
                do i = 1, 2
                    index = (j - 1)*2 + i
                    row(i) = row(i) + (1.0_dp-decay)*gradient(index)**2/2.0_dp
                    column(j) = column(j) + (1.0_dp-decay)*gradient(index)**2/2.0_dp
                end do
            end do
            vector_moment = decay*vector_moment + (1.0_dp-decay)*gradient(5:6)**2
            dense_expected(1:4) = 0.0_dp
            do j = 1, 2
                do i = 1, 2
                    index = (j - 1)*2 + i
                    dense_expected(index) = row(i)*column(j)/max(sum(row)/2.0_dp, epsilon)
                end do
            end do
            dense_expected(5:6) = vector_moment
            update_rms = sqrt(sum(dense_expected)/6.0_dp)
            clip_scale = max(1.0_dp, update_rms/clip_threshold)
            expected = expected - rate*gradient/clip_scale/(sqrt(dense_expected)+epsilon)
            call optimizer%step(x, gradient, status)
        end do
        call optimizer%dense_second_moment(dense, status)
        call check(status_ok(status), "factored recurrence status", failures)
        call check(maxval(abs(x-expected)) < 3.0e-14_dp, &
            "matrix/vector recurrence matches independent oracle", failures)
        call check(maxval(abs(dense-dense_expected)) < 3.0e-14_dp, &
            "dense factor reconstruction matches independent oracle", failures)
        call check(optimizer%device_supported(FORTML_DEVICE_CPU) .and. &
            .not. optimizer%device_supported(FORTML_DEVICE_CUDA), &
            "CPU support and typed CUDA capability", failures)
    end subroutine test_factored_recurrence

    subroutine test_mlp_integration_and_checkpoint(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model, full_model, resumed_model, loaded_model
        type(mlp_training_options_t) :: options, full_options, split_options
        type(mlp_training_state_t) :: state, full_state, split_state
        type(mlp_training_checkpoint_t) :: checkpoint, full_checkpoint, split_checkpoint, loaded_checkpoint
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), target(4, 1)
        real(dp), allocatable :: initial_parameters(:), full_parameters(:), resumed_parameters(:), loaded_parameters(:)
        character(*), parameter :: path = "test_mlp_adafactor_factored_checkpoint.txt"

        x = reshape([ -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, &
            0.5_dp, -0.5_dp, 1.5_dp, -1.5_dp ], shape(x))
        target(:, 1) = [0.2_dp, -0.4_dp, 0.8_dp, 1.1_dp]
        call model%initialize([2, 3, 1], status, output_activation=MLP_LINEAR)
        options%optimizer = MLP_OPTIMIZER_ADAFACTOR
        options%adafactor_factored = .true.
        options%max_epochs = 2
        options%learning_rate = 0.03_dp
        options%adafactor_decay = 0.8_dp
        options%adafactor_clip_threshold = 1.0_dp
        options%epsilon = 1.0e-3_dp
        options%tolerance = 0.0_dp
        options%restore_best = .false.
        call mlp_train(model, x, target, status, options, state)
        call check(status_ok(status) .and. state%updates == 2, &
            "MLP trainer integrates factored weight blocks", failures)

        full_options = options
        full_options%max_epochs = 6
        full_options%batch_size = 2
        full_options%shuffle = .true.
        full_options%shuffle_seed = 31
        full_options%tolerance = 0.0_dp
        split_options = full_options
        split_options%max_epochs = 3
        call full_model%initialize([2, 3, 1], status, output_activation=MLP_LINEAR)
        call resumed_model%initialize([2, 3, 1], status, output_activation=MLP_LINEAR)
        call loaded_model%initialize([2, 3, 1], status, output_activation=MLP_LINEAR)
        initial_parameters = full_model%parameters()
        call resumed_model%set_parameters(initial_parameters, status)
        call loaded_model%set_parameters(initial_parameters, status)
        call mlp_train(full_model, x, target, status, full_options, full_state, &
            checkpoint=full_checkpoint)
        call check(status_ok(status) .and. full_checkpoint%valid() .and. &
            full_checkpoint%adafactor_factored, &
            "factored full checkpoint capture", failures)
        call mlp_train(resumed_model, x, target, status, split_options, split_state, &
            checkpoint=split_checkpoint)
        call check(status_ok(status) .and. split_checkpoint%valid() .and. &
            split_checkpoint%n_adafactor_blocks == 4, &
            "factored split checkpoint capture", failures)
        call mlp_checkpoint_save(split_checkpoint, path, status)
        call check(status_ok(status), "factored checkpoint formatted save", failures)
        call mlp_checkpoint_load(loaded_checkpoint, path, status)
        call check(status_ok(status) .and. loaded_checkpoint%valid() .and. &
            loaded_checkpoint%adafactor_factored .and. &
            maxval(abs(loaded_checkpoint%adafactor_row_moment - &
            split_checkpoint%adafactor_row_moment)) < 1.0e-15_dp, &
            "factored checkpoint formatted load", failures)
        call mlp_train(resumed_model, x, target, status, full_options, split_state, &
            checkpoint=split_checkpoint)
        call check(status_ok(status), "factored native resume", failures)
        call mlp_train(loaded_model, x, target, status, full_options, split_state, &
            checkpoint=loaded_checkpoint)
        call check(status_ok(status), "factored formatted resume", failures)
        full_parameters = full_model%parameters()
        resumed_parameters = resumed_model%parameters()
        loaded_parameters = loaded_model%parameters()
        call check(maxval(abs(full_parameters - resumed_parameters)) < 2.0e-14_dp .and. &
            maxval(abs(full_parameters - loaded_parameters)) < 2.0e-14_dp, &
            "factored uninterrupted/native/formatted trajectories agree", failures)
        call execute_command_line("rm -f " // path)
    end subroutine test_mlp_integration_and_checkpoint

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL [factored-adafactor] "//description
        end if
    end subroutine check

end program test_mlp_adafactor_factored
