program test_classification_metrics
    use fortml_classification_metrics, only: classification_accuracy, &
        classification_balanced_accuracy, classification_confusion_matrix, &
        classification_precision_recall_f1, classification_log_loss, &
        classification_top_k_accuracy, classification_brier_score, &
        classification_binary_matthews, classification_calibration_error, &
        classification_maximum_calibration_error, classification_reliability_diagram
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_metric_oracles(failures)
    call test_weighted_log_loss(failures)
    call test_extended_metrics(failures)
    call test_calibration_oracle(failures)
    call test_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL classification metric cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS classification metric independent behavioral oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

    subroutine test_metric_oracles(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: labels(4) = [0, 0, 1, 1]
        integer, parameter :: predictions(4) = [0, 1, 1, 1]
        integer, parameter :: classes(2) = [0, 1]
        integer :: matrix(2, 2)
        real(dp) :: accuracy, balanced, precision(2), recall(2), f1(2)

        call classification_accuracy(labels, predictions, accuracy, status)
        call check(status_ok(status), "accuracy status", failures)
        call check(abs(accuracy - 0.75_dp) < 1.0e-14_dp, &
            "analytic accuracy", failures)
        call classification_balanced_accuracy(labels, predictions, classes, &
            balanced, status)
        call check(status_ok(status), "balanced accuracy status", failures)
        call check(abs(balanced - 0.75_dp) < 1.0e-14_dp, &
            "analytic balanced accuracy", failures)
        call classification_confusion_matrix(labels, predictions, classes, &
            matrix, status)
        call check(status_ok(status), "confusion matrix status", failures)
        call check(all(matrix == reshape([1, 0, 1, 2], [2, 2])), &
            "analytic confusion matrix", failures)
        call classification_precision_recall_f1(labels, predictions, classes, &
            precision, recall, f1, status)
        call check(status_ok(status), "precision recall status", failures)
        call check(maxval(abs(precision - [1.0_dp, 2.0_dp/3.0_dp])) < 1.0e-14_dp, &
            "analytic precision", failures)
        call check(maxval(abs(recall - [0.5_dp, 1.0_dp])) < 1.0e-14_dp, &
            "analytic recall", failures)
        call check(maxval(abs(f1 - [2.0_dp/3.0_dp, 0.8_dp])) < 1.0e-14_dp, &
            "analytic F1", failures)
    end subroutine test_metric_oracles

    subroutine test_weighted_log_loss(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: labels(4) = [0, 0, 1, 1]
        integer, parameter :: classes(2) = [0, 1]
        real(dp), parameter :: probabilities(4, 2) = reshape([ &
            0.8_dp, 0.4_dp, 0.1_dp, 0.2_dp, 0.2_dp, 0.6_dp, 0.9_dp, 0.8_dp], &
            [4, 2])
        real(dp), parameter :: weights(4) = [1.0_dp, 2.0_dp, 1.0_dp, 1.0_dp]
        real(dp) :: value, expected

        expected = -(log(0.8_dp) + 2.0_dp*log(0.4_dp) + log(0.9_dp) + &
            log(0.8_dp))/5.0_dp
        call classification_log_loss(probabilities, labels, classes, value, &
            status, sample_weight=weights)
        call check(status_ok(status), "weighted log-loss status", failures)
        call check(abs(value - expected) < 1.0e-14_dp, &
            "weighted log-loss oracle", failures)
        call classification_accuracy(labels, [0, 1, 1, 1], value, status, &
            sample_weight=weights)
        call check(status_ok(status), "weighted accuracy status", failures)
        call check(abs(value - 0.6_dp) < 1.0e-14_dp, &
            "weighted accuracy oracle", failures)
    end subroutine test_weighted_log_loss

    subroutine test_extended_metrics(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: labels(4) = [0, 1, 2, 2]
        integer, parameter :: predictions(4) = [0, 2, 1, 2]
        integer, parameter :: classes(3) = [0, 1, 2]
        real(dp), parameter :: probabilities(4, 3) = reshape([ &
            0.70_dp, 0.10_dp, 0.20_dp, 0.20_dp, &
            0.20_dp, 0.60_dp, 0.30_dp, 0.50_dp, &
            0.10_dp, 0.30_dp, 0.50_dp, 0.30_dp], [4, 3])
        real(dp) :: top1, top2, brier, matthews, expected_brier

        call classification_top_k_accuracy(probabilities, labels, classes, 1, &
            top1, status)
        call check(status_ok(status), "top-1 accuracy status", failures)
        call check(abs(top1 - 0.75_dp) < 1.0e-14_dp, &
            "top-1 accuracy oracle", failures)
        call classification_top_k_accuracy(probabilities, labels, classes, 2, &
            top2, status)
        call check(status_ok(status), "top-2 accuracy status", failures)
        call check(abs(top2 - 1.0_dp) < 1.0e-14_dp, &
            "top-2 accuracy oracle", failures)

        expected_brier = ((0.3_dp**2 + 0.2_dp**2 + 0.1_dp**2) + &
            (0.1_dp**2 + 0.4_dp**2 + 0.3_dp**2) + &
            (0.2_dp**2 + 0.3_dp**2 + 0.5_dp**2) + &
            (0.2_dp**2 + 0.5_dp**2 + 0.7_dp**2))/4.0_dp
        call classification_brier_score(probabilities, labels, classes, brier, status)
        call check(status_ok(status), "Brier score status", failures)
        call check(abs(brier - expected_brier) < 1.0e-14_dp, &
            "Brier score oracle", failures)

        call classification_binary_matthews([0, 0, 1, 1], [0, 1, 1, 1], &
            [0, 1], matthews, status)
        call check(status_ok(status), "binary Matthews status", failures)
        call check(abs(matthews - 0.5773502691896258_dp) < 1.0e-14_dp, &
            "binary Matthews oracle", failures)
    end subroutine test_extended_metrics

    subroutine test_calibration_oracle(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, parameter :: labels(4) = [0, 1, 2, 2]
        integer, parameter :: classes(3) = [0, 1, 2]
        real(dp), parameter :: probabilities(4, 3) = reshape([ &
            0.70_dp, 0.10_dp, 0.20_dp, 0.20_dp, &
            0.20_dp, 0.60_dp, 0.30_dp, 0.50_dp, &
            0.10_dp, 0.30_dp, 0.50_dp, 0.30_dp], [4, 3])
        real(dp), parameter :: weights(4) = [1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp]
        real(dp) :: expected_error, maximum_error
        real(dp) :: mean_confidence(2), mean_accuracy(2), bin_weight(2)

        call classification_calibration_error(probabilities, labels, classes, 2, &
            expected_error, status)
        call check(status_ok(status), "calibration error status", failures)
        call check(abs(expected_error - 0.175_dp) < 1.0e-14_dp, &
            "calibration error hand oracle", failures)
        call classification_maximum_calibration_error(probabilities, labels, &
            classes, 2, maximum_error, status)
        call check(status_ok(status), "maximum calibration error status", failures)
        call check(abs(maximum_error - 0.175_dp) < 1.0e-14_dp, &
            "maximum calibration error hand oracle", failures)

        call classification_calibration_error(probabilities, labels, classes, 2, &
            expected_error, status, sample_weight=weights)
        call check(status_ok(status), "weighted calibration error status", failures)
        call check(abs(expected_error - 0.1_dp) < 1.0e-14_dp, &
            "weighted calibration error hand oracle", failures)

        call classification_reliability_diagram(probabilities, labels, classes, 2, &
            mean_confidence, mean_accuracy, bin_weight, status)
        call check(status_ok(status), "reliability diagram status", failures)
        call check(all(abs(mean_confidence - [0.0_dp, 0.575_dp]) < 1.0e-14_dp), &
            "reliability confidence hand oracle", failures)
        call check(all(abs(mean_accuracy - [0.0_dp, 0.75_dp]) < 1.0e-14_dp), &
            "reliability accuracy hand oracle", failures)
        call check(all(abs(bin_weight - [0.0_dp, 4.0_dp]) < 1.0e-14_dp), &
            "reliability mass hand oracle", failures)

        call classification_reliability_diagram(probabilities, labels, classes, 2, &
            mean_confidence, mean_accuracy, bin_weight, status, sample_weight=weights)
        call check(status_ok(status), "weighted reliability diagram status", failures)
        call check(abs(mean_confidence(2) - (3.4_dp/6.0_dp)) < 1.0e-14_dp, &
            "weighted reliability confidence oracle", failures)
        call check(abs(mean_accuracy(2) - (4.0_dp/6.0_dp)) < 1.0e-14_dp, &
            "weighted reliability accuracy oracle", failures)
        call check(abs(bin_weight(2) - 6.0_dp) < 1.0e-14_dp, &
            "weighted reliability mass oracle", failures)

        call classification_calibration_error(reshape([0.5_dp, 1.0_dp, &
            0.5_dp, 0.0_dp, 0.0_dp, 0.0_dp], [2, 3]), [1, 0], classes, 2, &
            expected_error, status)
        call check(status_ok(status), "tie calibration error status", failures)
        call check(abs(expected_error - 0.25_dp) < 1.0e-14_dp, &
            "tie and confidence-one calibration oracle", failures)
    end subroutine test_calibration_oracle

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer :: matrix(2, 2)
        real(dp) :: value

        call classification_accuracy([1, 2], [1], value, status)
        call check(.not. status_ok(status), "mismatched label refusal", failures)
        call classification_confusion_matrix([1, 2], [1, 2], [1, 1], &
            matrix, status)
        call check(.not. status_ok(status), "duplicate class refusal", failures)
        call classification_log_loss(reshape([0.5_dp, -0.5_dp], [1, 2]), [1], &
            [0, 1], value, status)
        call check(.not. status_ok(status), "negative probability refusal", failures)
        call classification_top_k_accuracy(reshape([0.5_dp, 0.5_dp], [1, 2]), [1], &
            [0, 1], 0, value, status)
        call check(.not. status_ok(status), "invalid top-k refusal", failures)
        call classification_binary_matthews([0, 1], [0, 1], [0, 1, 2], value, status)
        call check(.not. status_ok(status), "nonbinary Matthews refusal", failures)
        call classification_calibration_error(reshape([0.5_dp, 0.5_dp], [1, 2]), &
            [1], [0, 1], 0, value, status)
        call check(.not. status_ok(status), "invalid calibration bin refusal", failures)
        call classification_calibration_error(reshape([0.5_dp, 0.5_dp], [1, 2]), &
            [2], [0, 1], 2, value, status)
        call check(.not. status_ok(status), "unknown calibration label refusal", failures)
        call classification_reliability_diagram(reshape([0.5_dp, 0.5_dp], [1, 2]), &
            [0], [0, 1], 2, [0.0_dp], mean_accuracy, bin_weight, status)
        call check(.not. status_ok(status), "reliability output shape refusal", failures)
    end subroutine test_refusals

end program test_classification_metrics
