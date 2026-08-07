program fortml_bench_roc_auc
    !! Correctness-gated binary and multiclass ROC-AUC workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_classification_metrics, only: classification_roc_auc, &
        classification_roc_auc_ovr, classification_roc_auc_device, classification_pr_auc, &
        classification_pr_auc_ovr, classification_pr_auc_device
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    real(dp) :: binary_scores(4), binary_value, binary_error, pr_binary_value, pr_binary_error
    real(dp) :: multiclass_scores(6, 3), multiclass_value, multiclass_error
    real(dp) :: pr_multiclass_value, pr_multiclass_error
    integer :: binary_labels(4), multiclass_labels(6), classes(3), repetitions, i
    integer(int64) :: started, finished, rate
    real(dp) :: binary_seconds, multiclass_seconds, pr_binary_seconds, pr_multiclass_seconds
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda

    binary_scores = [0.9_dp, 0.8_dp, 0.8_dp, 0.2_dp]
    binary_labels = [42, 42, -7, -7]
    multiclass_labels = [-2, -2, 4, 4, 9, 9]
    classes = [-2, 4, 9]
    multiclass_scores(:, 1) = [0.9_dp, 0.7_dp, 0.8_dp, 0.2_dp, 0.1_dp, 0.3_dp]
    multiclass_scores(:, 2) = [0.05_dp, 0.1_dp, 0.8_dp, 0.7_dp, 0.1_dp, 0.2_dp]
    multiclass_scores(:, 3) = [0.05_dp, 0.1_dp, 0.1_dp, 0.1_dp, 0.8_dp, 0.7_dp]

    call classification_roc_auc(binary_scores, binary_labels, 42, binary_value, status)
    if (.not. status_ok(status)) error stop "ROC-AUC binary benchmark failed"
    binary_error = abs(binary_value - 0.875_dp)
    call classification_roc_auc_ovr(multiclass_scores, multiclass_labels, classes, &
        multiclass_value, status)
    if (.not. status_ok(status)) error stop "ROC-AUC OVR benchmark failed"
    multiclass_error = abs(multiclass_value - (0.875_dp + 2.0_dp)/3.0_dp)
    call classification_pr_auc(binary_scores, binary_labels, 42, pr_binary_value, status)
    if (.not. status_ok(status)) error stop "PR-AUC binary benchmark failed"
    pr_binary_error = abs(pr_binary_value - 5.0_dp/6.0_dp)
    call classification_pr_auc_ovr(multiclass_scores, multiclass_labels, classes, &
        pr_multiclass_value, status)
    if (.not. status_ok(status)) error stop "PR-AUC OVR benchmark failed"
    pr_multiclass_error = abs(pr_multiclass_value - 17.0_dp/18.0_dp)

    call system_clock(started, rate)
    do repetitions = 1, 1024
        call classification_roc_auc(binary_scores, binary_labels, 42, binary_value, status)
    end do
    call system_clock(finished)
    binary_seconds = real(finished - started, dp)/real(rate, dp)/1024.0_dp
    call system_clock(started, rate)
    do i = 1, 1024
        call classification_roc_auc_ovr(multiclass_scores, multiclass_labels, classes, &
            multiclass_value, status)
    end do
    call system_clock(finished)
    multiclass_seconds = real(finished - started, dp)/real(rate, dp)/1024.0_dp
    call system_clock(started, rate)
    do repetitions = 1, 1024
        call classification_pr_auc(binary_scores, binary_labels, 42, pr_binary_value, status)
    end do
    call system_clock(finished)
    pr_binary_seconds = real(finished - started, dp)/real(rate, dp)/1024.0_dp
    call system_clock(started, rate)
    do i = 1, 1024
        call classification_pr_auc_ovr(multiclass_scores, multiclass_labels, classes, &
            pr_multiclass_value, status)
    end do
    call system_clock(finished)
    pr_multiclass_seconds = real(finished - started, dp)/real(rate, dp)/1024.0_dp
    write (*, '(a,es24.16)') "roc_auc_binary,", binary_value
    write (*, '(a,es24.16)') "roc_auc_binary_error,", binary_error
    write (*, '(a,es24.16)') "roc_auc_binary_seconds,", binary_seconds
    write (*, '(a,es24.16)') "roc_auc_ovr,", multiclass_value
    write (*, '(a,es24.16)') "roc_auc_ovr_error,", multiclass_error
    write (*, '(a,es24.16)') "roc_auc_ovr_seconds,", multiclass_seconds
    write (*, '(a,es24.16)') "pr_auc_binary,", pr_binary_value
    write (*, '(a,es24.16)') "pr_auc_binary_error,", pr_binary_error
    write (*, '(a,es24.16)') "pr_auc_binary_seconds,", pr_binary_seconds
    write (*, '(a,es24.16)') "pr_auc_ovr,", pr_multiclass_value
    write (*, '(a,es24.16)') "pr_auc_ovr_error,", pr_multiclass_error
    write (*, '(a,es24.16)') "pr_auc_ovr_seconds,", pr_multiclass_seconds

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call classification_roc_auc_device(cuda, binary_scores, binary_labels, 42, &
        binary_value, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "ROC-AUC CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "roc_auc_cuda,unavailable"
    call classification_pr_auc_device(cuda, binary_scores, binary_labels, 42, &
        binary_value, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "PR-AUC CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "pr_auc_cuda,unavailable"
end program fortml_bench_roc_auc
