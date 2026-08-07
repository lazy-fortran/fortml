program fortml_bench_multilabel_metrics
    !! Correctness-gated workload for multilabel averages and ROC-AUC metrics.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_classification_metrics, only: &
        classification_multilabel_precision_recall_f1, &
        classification_multilabel_probability_metrics, classification_roc_auc, &
        classification_roc_auc_ovr, CLASSIFICATION_AVERAGE_MICRO, &
        CLASSIFICATION_AVERAGE_MACRO, CLASSIFICATION_AVERAGE_SAMPLES
    implicit none

    integer, parameter :: n_samples = 3, n_labels = 3, repetitions = 256
    integer :: labels(n_samples, n_labels), predictions(n_samples, n_labels)
    integer :: binary_labels(4), classes(3), multiclass_labels(6), repetition
    real(real64) :: probabilities(n_samples, n_labels), precision, recall, f1
    real(real64) :: scores(4), multiclass_scores(6, 3), roc_value, roc_macro
    real(real64) :: elapsed
    integer :: clock_start, clock_end, clock_rate
    type(fortnum_status_t) :: status

    labels = reshape([1, 0, 0, 0, 1, 0, 0, 0, 0], shape(labels))
    predictions = reshape([1, 0, 0, 0, 0, 0, 1, 0, 0], shape(predictions))
    probabilities = reshape([0.5_real64, 0.1_real64, 0.2_real64, &
        0.2_real64, 0.5_real64, 0.1_real64, 0.8_real64, 0.2_real64, 0.3_real64], &
        shape(probabilities))
    call classification_multilabel_precision_recall_f1(labels, predictions, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_MICRO)
    if (.not. status_ok(status)) error stop 1
    write (*, '(a,3(",",es24.16))') "multilabel_micro", precision, recall, f1
    call classification_multilabel_precision_recall_f1(labels, predictions, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_MACRO)
    if (.not. status_ok(status)) error stop 1
    write (*, '(a,3(",",es24.16))') "multilabel_macro", precision, recall, f1
    call classification_multilabel_precision_recall_f1(labels, predictions, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_SAMPLES)
    if (.not. status_ok(status)) error stop 1
    write (*, '(a,3(",",es24.16))') "multilabel_samples", precision, recall, f1
    call classification_multilabel_probability_metrics(probabilities, labels, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_MICRO, threshold=0.5_real64)
    if (.not. status_ok(status)) error stop 1
    write (*, '(a,3(",",es24.16))') "multilabel_threshold", precision, recall, f1

    scores = [0.9_real64, 0.8_real64, 0.8_real64, 0.2_real64]
    binary_labels = [42, 42, -7, -7]
    call classification_roc_auc(scores, binary_labels, 42, roc_value, status)
    if (.not. status_ok(status)) error stop 1
    write (*, '(a,",",es24.16)') "roc_binary", roc_value
    classes = [-2, 4, 9]
    multiclass_labels = [-2, -2, 4, 4, 9, 9]
    multiclass_scores(:, 1) = [0.9_real64, 0.7_real64, 0.8_real64, 0.2_real64, 0.1_real64, 0.3_real64]
    multiclass_scores(:, 2) = [0.05_real64, 0.1_real64, 0.8_real64, 0.7_real64, 0.1_real64, 0.2_real64]
    multiclass_scores(:, 3) = [0.05_real64, 0.1_real64, 0.1_real64, 0.1_real64, 0.8_real64, 0.7_real64]
    call classification_roc_auc_ovr(multiclass_scores, multiclass_labels, classes, &
        roc_macro, status)
    if (.not. status_ok(status)) error stop 1
    write (*, '(a,",",es24.16)') "roc_ovr_macro", roc_macro

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call classification_multilabel_precision_recall_f1(labels, predictions, &
            precision, recall, f1, status, CLASSIFICATION_AVERAGE_MICRO)
        if (.not. status_ok(status)) error stop 1
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, real64)/real(max(1, clock_rate), real64) &
        /real(repetitions, real64)
    write (*, '(a,",",es24.16)') "multilabel_metric_seconds", elapsed
end program fortml_bench_multilabel_metrics
