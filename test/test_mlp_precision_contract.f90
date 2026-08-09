program test_mlp_precision_contract
    !! Independent oracles for the MLP training precision capabilities.
    use, intrinsic :: iso_fortran_env, only: real32
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_train, mlp_training_options_t, &
        mlp_training_state_t, mlp_training_checkpoint_t, MLP_OPTIMIZER_SGD, &
        MLP_PRECISION_FP64, MLP_PRECISION_FP32, MLP_PRECISION_FP16, &
        MLP_PRECISION_BF16, mlp_precision_name
    implicit none

    integer :: failures, kind, epoch
    type(mlp_t) :: reference, candidate, fp32_model
    type(mlp_training_options_t) :: options
    type(mlp_training_state_t) :: reference_state, candidate_state
    type(mlp_training_checkpoint_t) :: checkpoint
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 1), target(4, 1), expected(2), theta_before(2), residual(4)
    real(dp) :: gradient(2)

    failures = 0
    x(:, 1) = [-1.5_dp, -0.5_dp, 0.5_dp, 1.5_dp]
    target(:, 1) = 0.4_dp*x(:, 1) + 0.2_dp
    expected = [0.1_dp, -0.2_dp]
    options%max_epochs = 3
    options%learning_rate = 0.05_dp
    options%optimizer = MLP_OPTIMIZER_SGD
    options%tolerance = 0.0_dp
    options%restore_best = .false.
    options%precision_kind = MLP_PRECISION_FP64

    call reference%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call reference%set_parameters(expected, status)
    call mlp_train(reference, x, target, status, options, reference_state, &
        checkpoint=checkpoint)
    call check(status_ok(status) .and. checkpoint%valid() .and. &
        checkpoint%precision_kind == MLP_PRECISION_FP64 .and. &
        reference_state%precision_kind == MLP_PRECISION_FP64, &
        "FP64 precision state is explicit", failures)
    do epoch = 1, options%max_epochs
        residual = x(:, 1)*expected(1) + expected(2) - target(:, 1)
        gradient = [sum(residual*x(:, 1))/real(size(x, 1), dp), &
            sum(residual)/real(size(x, 1), dp)]
        expected = expected - options%learning_rate*gradient
    end do
    call check(all(ieee_is_finite(expected)) .and. &
        maxval(abs(reference%parameters() - expected)) < 2.0e-14_dp, &
        "FP64 trajectory matches independent linear-MSE recurrence", failures)
    call check(trim(mlp_precision_name(MLP_PRECISION_FP64)) == "fp64" .and. &
        trim(mlp_precision_name(MLP_PRECISION_FP32)) == "fp32" .and. &
        trim(mlp_precision_name(MLP_PRECISION_FP16)) == "fp16" .and. &
        trim(mlp_precision_name(MLP_PRECISION_BF16)) == "bf16", &
        "precision names are stable", failures)

    call test_fp32_master_recurrence(fp32_model, x, target, failures)

    do kind = MLP_PRECISION_FP16, MLP_PRECISION_BF16
        call candidate%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call candidate%set_parameters([0.1_dp, -0.2_dp], status)
        theta_before = candidate%parameters()
        options%precision_kind = kind
        call mlp_train(candidate, x, target, status, options, candidate_state)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            maxval(abs(candidate%parameters() - theta_before)) == 0.0_dp .and. &
            candidate_state%precision_kind == kind .and. candidate_state%updates == 0, &
            "unsupported precision returns typed refusal without mutation", failures)
    end do

    call candidate%initialize([1, 1], status, output_activation=MLP_LINEAR)
    options%precision_kind = 99
    call mlp_train(candidate, x, target, status, options, candidate_state)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "unknown precision is a domain error", failures)

    if (failures /= 0) then
        write (*, '(a,i0)') "FAIL MLP precision contract cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP precision capability independent oracle"

contains

    subroutine test_fp32_master_recurrence(model, x, target, failures)
        type(mlp_t), intent(out) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        integer, intent(inout) :: failures
        type(mlp_training_options_t) :: fp32_options
        type(mlp_training_state_t) :: fp32_state
        type(mlp_training_checkpoint_t) :: fp32_checkpoint
        type(mlp_t) :: full_model, split_model
        type(mlp_training_options_t) :: split_options
        type(mlp_training_state_t) :: full_state, split_state
        type(mlp_training_checkpoint_t) :: split_checkpoint
        type(fortnum_status_t) :: local_status
        real(dp) :: initial(2), expected(2), forward_theta(2), residual(size(x, 1))
        real(dp) :: gradient_expected(2), x32(size(x, 1)), target32(size(target, 1))
        real(dp) :: actual(2), rounded_actual(2)
        integer :: step

        initial = [0.123456789012345_dp, -0.234567890123456_dp]
        x32 = real(real(x(:, 1), real32), dp)
        target32 = real(real(target(:, 1), real32), dp)
        expected = initial
        forward_theta = real(real(expected, real32), dp)
        do step = 1, 3
            residual = x32*forward_theta(1) + forward_theta(2) - target32
            gradient_expected = [sum(residual*x32)/real(size(x, 1), dp), &
                sum(residual)/real(size(x, 1), dp)]
            gradient_expected = real(real(gradient_expected, real32), dp)
            expected = expected - 0.05_dp*gradient_expected
            forward_theta = real(real(expected, real32), dp)
        end do

        call model%initialize([1, 1], local_status, output_activation=MLP_LINEAR)
        call model%set_parameters(initial, local_status)
        fp32_options%max_epochs = 3
        fp32_options%learning_rate = 0.05_dp
        fp32_options%optimizer = MLP_OPTIMIZER_SGD
        fp32_options%tolerance = 0.0_dp
        fp32_options%restore_best = .false.
        fp32_options%precision_kind = MLP_PRECISION_FP32
        call fp32_options%loss_scale%initialize(local_status, enabled=.true., &
            initial_scale=2.0_dp, growth_interval=100)
        call mlp_train(model, x, target, local_status, fp32_options, fp32_state, &
            checkpoint=fp32_checkpoint)
        actual = model%parameters()
        rounded_actual = real(real(actual, real32), dp)
        call check(status_ok(local_status) .and. fp32_state%updates == 3 .and. &
            fp32_state%precision_kind == MLP_PRECISION_FP32 .and. &
            fp32_checkpoint%valid() .and. fp32_checkpoint%precision_kind == &
            MLP_PRECISION_FP32, "FP32 master-weight training completes", failures)
        call check(maxval(abs(actual - expected)) < 2.0e-14_dp, &
            "FP32 trajectory matches independently rounded recurrence", failures)
        call check(maxval(abs(fp32_checkpoint%parameters - expected)) < 2.0e-14_dp, &
            "FP32 checkpoint stores the binary64 master vector", failures)
        call check(maxval(abs(expected - rounded_actual)) > 0.0_dp, &
            "FP32 master retains precision beyond the forward boundary", failures)

        call full_model%initialize([1, 1], local_status, output_activation=MLP_LINEAR)
        call full_model%set_parameters(initial, local_status)
        fp32_options%max_epochs = 3
        call mlp_train(full_model, x, target, local_status, fp32_options, full_state)
        call split_model%initialize([1, 1], local_status, output_activation=MLP_LINEAR)
        call split_model%set_parameters(initial, local_status)
        split_options = fp32_options
        split_options%max_epochs = 1
        call mlp_train(split_model, x, target, local_status, split_options, split_state, &
            checkpoint=split_checkpoint)
        split_options%max_epochs = 3
        call mlp_train(split_model, x, target, local_status, split_options, split_state, &
            checkpoint=split_checkpoint)
        call check(status_ok(local_status) .and. split_checkpoint%valid() .and. &
            maxval(abs(split_model%parameters() - full_model%parameters())) < 2.0e-14_dp, &
            "FP32 checkpoint resume matches uninterrupted master trajectory", failures)
    end subroutine test_fp32_master_recurrence

    subroutine check(condition, label, count)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: count

        if (.not. condition) then
            count = count + 1
            write (*, '(a)') "FAIL: "//trim(label)
        end if
    end subroutine check

end program test_mlp_precision_contract
