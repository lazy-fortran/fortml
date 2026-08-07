program test_classification_metrics
    use fortml_classification_metrics, only: classification_accuracy, &
        classification_balanced_accuracy, classification_confusion_matrix, &
        classification_precision_recall_f1, classification_log_loss
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_metric_oracles(failures)
    call test_weighted_log_loss(failures)
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
    end subroutine test_refusals

end program test_classification_metrics
