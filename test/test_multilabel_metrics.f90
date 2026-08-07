program test_multilabel_metrics
    !! Independent hand-derived checks for multilabel averaging semantics.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_classification_metrics, only: &
        classification_multilabel_precision_recall_f1, &
        classification_multilabel_probability_metrics, &
        CLASSIFICATION_AVERAGE_MICRO, CLASSIFICATION_AVERAGE_MACRO, &
        CLASSIFICATION_AVERAGE_SAMPLES, CLASSIFICATION_ZERO_DIVISION_ONE
    implicit none

    integer :: labels(3, 3), predictions(3, 3), failures
    real(dp) :: probabilities(3, 3), weights(3)
    real(dp) :: precision, recall, f1
    type(fortnum_status_t) :: status

    failures = 0
    labels = reshape([1, 0, 0, 0, 1, 0, 0, 0, 0], shape(labels))
    predictions = reshape([1, 0, 0, 0, 0, 0, 1, 0, 0], shape(predictions))

    call classification_multilabel_precision_recall_f1(labels, predictions, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_MICRO)
    call check(status_ok(status), "micro status", failures)
    call check(abs(precision - 0.5_dp) < 1.0e-14_dp .and. &
        abs(recall - 0.5_dp) < 1.0e-14_dp .and. abs(f1 - 0.5_dp) < 1.0e-14_dp, &
        "micro TP/FP/FN oracle", failures)

    call classification_multilabel_precision_recall_f1(labels, predictions, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_MACRO)
    call check(status_ok(status), "macro status", failures)
    call check(abs(precision - 1.0_dp/3.0_dp) < 1.0e-14_dp .and. &
        abs(recall - 1.0_dp/3.0_dp) < 1.0e-14_dp .and. &
        abs(f1 - 1.0_dp/3.0_dp) < 1.0e-14_dp, &
        "macro zero-support oracle", failures)

    weights = [1.0_dp, 2.0_dp, 3.0_dp]
    call classification_multilabel_precision_recall_f1(labels, predictions, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_SAMPLES, weights)
    call check(status_ok(status), "weighted samples status", failures)
    call check(abs(precision - 1.0_dp/12.0_dp) < 1.0e-14_dp .and. &
        abs(recall - 1.0_dp/6.0_dp) < 1.0e-14_dp .and. &
        abs(f1 - 1.0_dp/9.0_dp) < 1.0e-14_dp, &
        "weighted samples averaging oracle", failures)

    call classification_multilabel_precision_recall_f1(labels, predictions, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_SAMPLES, &
        zero_division=CLASSIFICATION_ZERO_DIVISION_ONE)
    call check(status_ok(status), "zero-division one status", failures)
    call check(abs(precision - 5.0_dp/6.0_dp) < 1.0e-14_dp .and. &
        abs(recall - 2.0_dp/3.0_dp) < 1.0e-14_dp .and. &
        abs(f1 - 5.0_dp/9.0_dp) < 1.0e-14_dp, &
        "zero-division one oracle", failures)

    probabilities = reshape([0.5_dp, 0.1_dp, 0.2_dp, &
        0.2_dp, 0.5_dp, 0.1_dp, 0.8_dp, 0.2_dp, 0.3_dp], shape(probabilities))
    call classification_multilabel_probability_metrics(probabilities, labels, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_MICRO, threshold=0.5_dp)
    call check(status_ok(status), "probability threshold status", failures)
    call check(abs(precision - 2.0_dp/3.0_dp) < 1.0e-14_dp .and. &
        abs(recall - 1.0_dp) < 1.0e-14_dp .and. abs(f1 - 0.8_dp) < 1.0e-14_dp, &
        "probability >= threshold tie oracle", failures)

    call classification_multilabel_probability_metrics(probabilities, labels, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_MICRO, threshold=1.5_dp)
    call check(.not. status_ok(status), "invalid threshold refusal", failures)
    predictions(1, 1) = 2
    call classification_multilabel_precision_recall_f1(labels, predictions, &
        precision, recall, f1, status, CLASSIFICATION_AVERAGE_MICRO)
    call check(.not. status_ok(status), "nonbinary indicator refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS multilabel metric independent behavioral oracles"

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

end program test_multilabel_metrics
