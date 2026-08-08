program test_mlp_precision_contract
    !! Independent oracle for the MLP training precision capability boundary.
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
    type(mlp_t) :: reference, candidate
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

    do kind = MLP_PRECISION_FP32, MLP_PRECISION_BF16
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
