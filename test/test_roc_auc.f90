program test_roc_auc
    !! Independent pairwise-oracle checks for ROC-AUC metrics.
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_classification_metrics, only: classification_roc_auc, &
        classification_roc_auc_ovr, classification_roc_auc_device, &
        classification_roc_auc_ovr_device
    implicit none

    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: scores(4), value, weighted_value, nan_value
    real(real64) :: multiclass_scores(6, 3), macro, per_class(3), cuda_value
    real(real64) :: weights(4), cuda_scores(4), cuda_multiclass(6, 3)
    integer :: labels(4), multiclass_labels(6), classes(3), failures

    failures = 0
    scores = [0.9_real64, 0.8_real64, 0.8_real64, 0.2_real64]
    labels = [42, 42, -7, -7]
    weights = [1.0_real64, 2.0_real64, 1.0_real64, 1.0_real64]
    call classification_roc_auc(scores, labels, 42, value, status)
    call check(status_ok(status) .and. abs(value - 0.875_real64) < 1.0e-14_real64, &
        "binary half-credit tie oracle", failures)
    call classification_roc_auc(scores, labels, 42, weighted_value, status, weights)
    call check(status_ok(status) .and. abs(weighted_value - 5.0_real64/6.0_real64) &
        < 1.0e-14_real64, "weighted pairwise oracle", failures)

    multiclass_labels = [-2, -2, 4, 4, 9, 9]
    classes = [-2, 4, 9]
    multiclass_scores(:, 1) = [0.9_real64, 0.7_real64, 0.8_real64, 0.2_real64, 0.1_real64, 0.3_real64]
    multiclass_scores(:, 2) = [0.05_real64, 0.1_real64, 0.8_real64, 0.7_real64, 0.1_real64, 0.2_real64]
    multiclass_scores(:, 3) = [0.05_real64, 0.1_real64, 0.1_real64, 0.1_real64, 0.8_real64, 0.7_real64]
    call classification_roc_auc_ovr(multiclass_scores, multiclass_labels, classes, &
        macro, status, per_class=per_class)
    call check(status_ok(status) .and. &
        maxval(abs(per_class - [0.875_real64, 1.0_real64, 1.0_real64])) < 1.0e-14_real64 .and. &
        abs(macro - (0.875_real64 + 2.0_real64)/3.0_real64) < 1.0e-14_real64, &
        "one-vs-rest macro and per-class oracle", failures)

    call classification_roc_auc([1.0_real64, 2.0_real64], [1, 1], 1, value, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "single-class refusal", failures)
    call classification_roc_auc([1.0_real64, 2.0_real64, 3.0_real64], [1, 2, 3], 1, &
        value, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "more-than-two-class refusal", failures)
    nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
    call classification_roc_auc([1.0_real64, nan_value], [1, 2], 1, value, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "nonfinite-score refusal", failures)
    call classification_roc_auc_ovr(multiclass_scores, [1, 1, 1, 4, 9, 9], classes, macro, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "OVR degenerate-class refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda_value = -17.0_real64
    cuda_scores = scores
    cuda_multiclass = multiclass_scores
    call classification_roc_auc_device(cuda, cuda_scores, labels, 42, cuda_value, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. cuda_value == -17.0_real64, &
        "binary CUDA refusal preserves output", failures)
    call classification_roc_auc_ovr_device(cuda, cuda_multiclass, multiclass_labels, &
        classes, cuda_value, status, per_class=per_class)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. cuda_value == -17.0_real64, &
        "OVR CUDA refusal preserves output", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL ROC-AUC cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS ROC-AUC independent pairwise oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [ROC-AUC] "//description
        end if
    end subroutine check

end program test_roc_auc
