program test_pr_multilabel_metrics
    !! Independent hand-derived oracles for PR-AUC and multilabel metrics.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_classification_metrics, only: &
        classification_pr_auc, classification_pr_auc_ovr, classification_pr_auc_device, &
        classification_multilabel_jaccard, classification_multilabel_hamming_loss, &
        CLASSIFICATION_AVERAGE_MICRO, CLASSIFICATION_AVERAGE_MACRO, &
        CLASSIFICATION_AVERAGE_SAMPLES, CLASSIFICATION_ZERO_DIVISION_ONE
    implicit none

    integer :: labels(4, 3), predictions(4, 3), failures
    integer :: binary_labels(4), classes(3), multiclass_labels(6)
    integer :: all_zero(1, 1), bad_predictions(4, 3)
    real(dp) :: value, weighted, scores(4), weights(4)
    real(dp) :: multiclass_scores(6, 3), per_class(3), cuda_value
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda

    failures = 0
    labels = 0
    labels(:, 1) = [1, 1, 0, 0]
    labels(:, 2) = [0, 1, 1, 0]
    labels(:, 3) = [1, 0, 0, 0]
    predictions = 0
    predictions(:, 1) = [1, 1, 0, 0]
    predictions(:, 2) = [1, 0, 1, 0]
    predictions(:, 3) = [0, 0, 1, 0]
    weights = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]

    call classification_multilabel_jaccard(labels, predictions, value, status, &
        CLASSIFICATION_AVERAGE_MICRO)
    call check(status_ok(status) .and. abs(value - 3.0_dp/7.0_dp) < 1.0e-14_dp, &
        "micro Jaccard intersection/union oracle", failures)
    call classification_multilabel_jaccard(labels, predictions, value, status, &
        CLASSIFICATION_AVERAGE_MACRO)
    call check(status_ok(status) .and. abs(value - 4.0_dp/9.0_dp) < 1.0e-14_dp, &
        "macro Jaccard per-label oracle", failures)
    call classification_multilabel_jaccard(labels, predictions, value, status, &
        CLASSIFICATION_AVERAGE_SAMPLES, weights)
    call check(status_ok(status) .and. abs(value - 17.0_dp/60.0_dp) < 1.0e-14_dp, &
        "weighted samples Jaccard oracle", failures)

    call classification_multilabel_hamming_loss(labels, predictions, value, status)
    call check(status_ok(status) .and. abs(value - 1.0_dp/3.0_dp) < 1.0e-14_dp, &
        "micro Hamming oracle", failures)
    call classification_multilabel_hamming_loss(labels, predictions, value, status, &
        weights, CLASSIFICATION_AVERAGE_SAMPLES)
    call check(status_ok(status) .and. abs(value - 7.0_dp/30.0_dp) < 1.0e-14_dp, &
        "weighted samples Hamming oracle", failures)

    all_zero = 0
    call classification_multilabel_jaccard(all_zero, all_zero, value, status, &
        CLASSIFICATION_AVERAGE_SAMPLES, zero_division=CLASSIFICATION_ZERO_DIVISION_ONE)
    call check(status_ok(status) .and. abs(value - 1.0_dp) < 1.0e-14_dp, &
        "explicit empty-support Jaccard policy", failures)
    bad_predictions = predictions
    bad_predictions(1, 1) = 2
    call classification_multilabel_jaccard(labels, bad_predictions, value, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "nonbinary indicator refusal", failures)
    call classification_multilabel_hamming_loss(labels, predictions, value, status, &
        [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero sample-weight refusal", failures)

    scores = [0.9_dp, 0.8_dp, 0.7_dp, 0.6_dp]
    binary_labels = [1, 0, 1, 0]
    call classification_pr_auc(scores, binary_labels, 1, value, status)
    call check(status_ok(status) .and. abs(value - 5.0_dp/6.0_dp) < 1.0e-14_dp, &
        "binary average-precision step oracle", failures)
    call classification_pr_auc(scores, binary_labels, 1, weighted, status, weights)
    call check(status_ok(status) .and. abs(weighted - 3.0_dp/4.0_dp) < 1.0e-14_dp, &
        "weighted average-precision oracle", failures)
    call classification_pr_auc([0.8_dp, 0.8_dp], [1, 0], 1, value, status)
    call check(status_ok(status) .and. abs(value - 0.5_dp) < 1.0e-14_dp, &
        "tied-score group oracle", failures)
    call classification_pr_auc([0.1_dp, 0.2_dp], [1, 1], 1, value, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "single-class PR-AUC refusal", failures)

    classes = [1, 2, 3]
    multiclass_labels = [1, 2, 3, 1, 2, 3]
    multiclass_scores(:, 1) = [0.9_dp, 0.1_dp, 0.2_dp, 0.8_dp, 0.3_dp, 0.4_dp]
    multiclass_scores(:, 2) = [0.1_dp, 0.8_dp, 0.85_dp, 0.3_dp, 0.9_dp, 0.4_dp]
    multiclass_scores(:, 3) = [0.1_dp, 0.2_dp, 0.9_dp, 0.3_dp, 0.4_dp, 0.8_dp]
    call classification_pr_auc_ovr(multiclass_scores, multiclass_labels, classes, &
        value, status, per_class=per_class)
    call check(status_ok(status) .and. maxval(abs(per_class - [1.0_dp, 5.0_dp/6.0_dp, &
        1.0_dp])) < 1.0e-14_dp .and. abs(value - 17.0_dp/18.0_dp) < 1.0e-14_dp, &
        "one-vs-rest PR-AUC oracle", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda_value = -17.0_dp
    call classification_pr_auc_device(cuda, scores, binary_labels, 1, cuda_value, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. cuda_value == -17.0_dp, &
        "binary CUDA PR-AUC refusal preserves output", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS PR-AUC and multilabel independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL: "//description
        end if
    end subroutine check

end program test_pr_multilabel_metrics
