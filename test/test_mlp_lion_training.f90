program test_mlp_lion_training
    !! Independent recurrence and resume oracle for production Lion training.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortnum_status, only: FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, mlp_training_state_t, &
        mlp_training_checkpoint_t, mlp_train, MLP_OPTIMIZER_LION
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load
    use fortml_mlp_lion_hypergradient, only: &
        mlp_lion_hypergradient_options_t, mlp_lion_hypergradient_objective_t
    use fortml_device, only: FORTML_DEVICE_CUDA
    implicit none

    type(mlp_t) :: full_model, split_model
    type(mlp_training_options_t) :: full_options, split_options
    type(mlp_training_state_t) :: full_state, split_state
    type(mlp_training_checkpoint_t) :: full_checkpoint, split_checkpoint
    type(mlp_training_checkpoint_t) :: loaded_checkpoint
    type(mlp_lion_hypergradient_options_t) :: hyper_options
    type(mlp_lion_hypergradient_objective_t) :: hyper_objective
    type(fortnum_status_t) :: status
    real(dp) :: x(2, 1), target(2, 1), expected(2), expected_momentum(2)
    real(dp) :: gradient(2), interpolated(2), update(2), prediction(2)
    real(dp) :: expected_ema(2), theta(2), ema_decay, rate, beta1, beta2, wd
    integer :: epoch, failures

    failures = 0
    x(:, 1) = [1.0_dp, 2.0_dp]
    target(:, 1) = [0.5_dp, 1.5_dp]
    rate = 0.05_dp
    beta1 = 0.8_dp
    beta2 = 0.9_dp
    wd = 0.1_dp
    ema_decay = 0.5_dp

    call full_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call check(status_ok(status), "full model initialize", failures)
    call split_model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call check(status_ok(status), "split model initialize", failures)
    call full_model%set_parameters([0.0_dp, 0.0_dp], status)
    call split_model%set_parameters([0.0_dp, 0.0_dp], status)

    full_options%optimizer = MLP_OPTIMIZER_LION
    full_options%max_epochs = 3
    full_options%learning_rate = rate
    full_options%beta1 = beta1
    full_options%beta2 = beta2
    full_options%weight_decay = wd
    full_options%ema_decay = ema_decay
    full_options%tolerance = 0.0_dp
    full_options%restore_best = .false.
    split_options = full_options
    split_options%max_epochs = 1

    call mlp_train(full_model, x, target, status, full_options, full_state, &
        checkpoint=full_checkpoint)
    call check(status_ok(status), "full Lion training", failures)
    call check(full_state%updates == 3 .and. full_checkpoint%valid(), &
        "full Lion state/checkpoint", failures)

    call mlp_train(split_model, x, target, status, split_options, split_state, &
        checkpoint=split_checkpoint)
    call check(status_ok(status) .and. split_checkpoint%valid(), &
        "split Lion checkpoint", failures)
    call mlp_train(split_model, x, target, status, full_options, split_state, &
        checkpoint=split_checkpoint)
    call check(status_ok(status), "resumed Lion training", failures)
    call check(maxval(abs(full_model%parameters()-split_model%parameters())) < 2.0e-14_dp, &
        "Lion checkpoint trajectory", failures)
    call check(maxval(abs(full_checkpoint%first_moment-split_checkpoint%first_moment)) < &
        2.0e-14_dp, "Lion momentum checkpoint trajectory", failures)

    expected = 0.0_dp
    expected_momentum = 0.0_dp
    expected_ema = expected
    do epoch = 1, 3
        prediction = expected(1)*x(:, 1) + expected(2)
        gradient(1) = sum((prediction-target(:, 1))*x(:, 1))/real(size(x, 1), dp)
        gradient(2) = sum(prediction-target(:, 1))/real(size(x, 1), dp)
        interpolated = beta1*expected_momentum + (1.0_dp-beta1)*gradient
        update = 0.0_dp
        where (interpolated > 0.0_dp) update = 1.0_dp
        where (interpolated < 0.0_dp) update = -1.0_dp
        expected = expected - rate*(update + wd*expected)
        expected_momentum = beta2*expected_momentum + (1.0_dp-beta2)*gradient
        expected_ema = ema_decay*expected_ema + (1.0_dp-ema_decay)*expected
    end do
    theta = full_model%parameters()
    call check(maxval(abs(theta-expected)) < 2.0e-14_dp, &
        "independent Lion recurrence", failures)
    call check(maxval(abs(full_state%ema_parameters-expected_ema)) < 2.0e-14_dp .and. &
        all(ieee_is_finite(expected_ema)), "Lion EMA recurrence", failures)
    call check(maxval(abs(full_checkpoint%first_moment-expected_momentum)) < 2.0e-14_dp, &
        "independent Lion momentum recurrence", failures)
    call mlp_checkpoint_save(full_checkpoint, "test_mlp_lion_checkpoint.txt", status)
    call check(status_ok(status), "Lion text checkpoint save", failures)
    call mlp_checkpoint_load(loaded_checkpoint, "test_mlp_lion_checkpoint.txt", status)
    call check(status_ok(status) .and. loaded_checkpoint%valid() .and. &
        maxval(abs(loaded_checkpoint%first_moment-expected_momentum)) < 2.0e-14_dp, &
        "Lion text checkpoint round trip", failures)
    call remove_file("test_mlp_lion_checkpoint.txt")

    hyper_options%device_kind = FORTML_DEVICE_CUDA
    call hyper_objective%initialize(full_model, x, target, x, target, &
        hyper_options, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "Lion CUDA derivative refusal", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP Lion cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP Lion independent behavioral oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

    subroutine remove_file(path)
        character(*), intent(in) :: path
        logical :: exists
        integer :: unit, ios

        inquire(file=path, exist=exists)
        if (.not. exists) return
        open(newunit=unit, file=path, status="old", iostat=ios)
        if (ios == 0) close(unit, status="delete")
    end subroutine remove_file

end program test_mlp_lion_training
