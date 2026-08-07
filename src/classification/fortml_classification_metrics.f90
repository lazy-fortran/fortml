module fortml_classification_metrics
    !! Shared classification metrics with explicit class-label semantics.
    !!
    !! Samples are rows.  Class labels need not be contiguous or zero based;
    !! the caller supplies their deterministic class order for multiclass
    !! metrics.  Every routine refuses shape, label, weight, and nonfinite
    !! errors instead of silently changing the sample axis.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    public :: classification_accuracy
    public :: classification_balanced_accuracy
    public :: classification_confusion_matrix
    public :: classification_precision_recall_f1
    public :: classification_log_loss
    public :: classification_top_k_accuracy
    public :: classification_brier_score
    public :: classification_binary_matthews
    public :: classification_calibration_error
    public :: classification_maximum_calibration_error
    public :: classification_multilabel_precision_recall_f1
    public :: classification_multilabel_probability_metrics
    public :: classification_multilabel_jaccard
    public :: classification_multilabel_hamming_loss
    public :: classification_multilabel_hamming

    integer, parameter, public :: CLASSIFICATION_AVERAGE_MICRO = 1
    integer, parameter, public :: CLASSIFICATION_AVERAGE_MACRO = 2
    integer, parameter, public :: CLASSIFICATION_AVERAGE_SAMPLES = 3
    integer, parameter, public :: CLASSIFICATION_ZERO_DIVISION_ZERO = 0
    integer, parameter, public :: CLASSIFICATION_ZERO_DIVISION_ONE = 1
    public :: classification_roc_auc
    public :: classification_roc_auc_ovr
    public :: classification_roc_auc_device
    public :: classification_roc_auc_ovr_device
    public :: classification_pr_auc
    public :: classification_pr_auc_ovr
    public :: classification_pr_auc_device
    public :: classification_pr_auc_ovr_device

contains

    subroutine classification_accuracy(labels, predictions, value, status, &
            sample_weight)
        integer, intent(in) :: labels(:), predictions(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator
        integer :: i

        value = 0.0_dp
        call validate_label_vectors(labels, predictions, status, &
            "classification accuracy")
        if (status%code /= FORTNUM_OK) return
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "classification accuracy")
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            if (labels(i) == predictions(i)) then
                if (present(sample_weight)) then
                    value = value + sample_weight(i)
                else
                    value = value + 1.0_dp
                end if
            end if
        end do
        value = value/denominator
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_accuracy

    subroutine classification_balanced_accuracy(labels, predictions, classes, &
            value, status)
        integer, intent(in) :: labels(:), predictions(:), classes(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, support
        real(dp) :: recall

        value = 0.0_dp
        call validate_label_vectors(labels, predictions, status, &
            "balanced accuracy")
        if (status%code /= FORTNUM_OK) return
        call validate_classes(classes, status, "balanced accuracy")
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(classes)
            support = 0
            recall = 0.0_dp
            do j = 1, size(labels)
                if (labels(j) == classes(i)) then
                    support = support + 1
                    if (predictions(j) == classes(i)) recall = recall + 1.0_dp
                end if
            end do
            if (support > 0) value = value + recall/real(support, dp)
        end do
        value = value/real(size(classes), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_balanced_accuracy

    subroutine classification_confusion_matrix(labels, predictions, classes, &
            matrix, status)
        integer, intent(in) :: labels(:), predictions(:), classes(:)
        integer, intent(out) :: matrix(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, true_class, predicted_class

        matrix = 0
        if (any(shape(matrix) /= [size(classes), size(classes)])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "confusion matrix: output shape is invalid")
            return
        end if
        call validate_label_vectors(labels, predictions, status, &
            "confusion matrix")
        if (status%code /= FORTNUM_OK) return
        call validate_classes(classes, status, "confusion matrix")
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            true_class = class_position(labels(i), classes)
            predicted_class = class_position(predictions(i), classes)
            if (true_class == 0 .or. predicted_class == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "confusion matrix: labels must belong to classes")
                return
            end if
            matrix(true_class, predicted_class) = &
                matrix(true_class, predicted_class) + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_confusion_matrix

    subroutine classification_precision_recall_f1(labels, predictions, classes, &
            precision, recall, f1, status)
        integer, intent(in) :: labels(:), predictions(:), classes(:)
        real(dp), intent(out) :: precision(:), recall(:), f1(:)
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: matrix(:, :)
        integer :: i, true_positive, false_positive, false_negative

        precision = 0.0_dp
        recall = 0.0_dp
        f1 = 0.0_dp
        if (size(precision) /= size(classes) .or. size(recall) /= size(classes) &
            .or. size(f1) /= size(classes)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "precision/recall/F1: output shape is invalid")
            return
        end if
        allocate(matrix(size(classes), size(classes)))
        call classification_confusion_matrix(labels, predictions, classes, &
            matrix, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(classes)
            true_positive = matrix(i, i)
            false_positive = sum(matrix(:, i)) - true_positive
            false_negative = sum(matrix(i, :)) - true_positive
            if (true_positive + false_positive > 0) precision(i) = &
                real(true_positive, dp)/real(true_positive + false_positive, dp)
            if (true_positive + false_negative > 0) recall(i) = &
                real(true_positive, dp)/real(true_positive + false_negative, dp)
            if (precision(i) + recall(i) > 0.0_dp) f1(i) = &
                2.0_dp*precision(i)*recall(i)/(precision(i) + recall(i))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_precision_recall_f1

    subroutine classification_log_loss(probabilities, labels, classes, value, &
            status, sample_weight)
        real(dp), intent(in) :: probabilities(:, :)
        integer, intent(in) :: labels(:), classes(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), parameter :: tiny_probability = 1.0e-15_dp
        real(dp) :: denominator, row_sum, weight
        integer :: i, class_index

        value = 0.0_dp
        if (size(probabilities, 1) /= size(labels) .or. &
            size(probabilities, 2) /= size(classes) .or. &
            size(labels) < 1 .or. size(classes) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classification log loss: input shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classification log loss: probabilities must be finite")
            return
        end if
        if (any(probabilities < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classification log loss: probabilities must be nonnegative")
            return
        end if
        call validate_classes(classes, status, "classification log loss")
        if (status%code /= FORTNUM_OK) return
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "classification log loss")
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            class_index = class_position(labels(i), classes)
            if (class_index == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "classification log loss: label is not in classes")
                return
            end if
            row_sum = sum(probabilities(i, :))
            if (.not. ieee_is_finite(row_sum) .or. row_sum <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "classification log loss: each row needs positive mass")
                return
            end if
            weight = 1.0_dp
            if (present(sample_weight)) weight = sample_weight(i)
            value = value - weight*log(max(probabilities(i, class_index)/ &
                row_sum, tiny_probability))
        end do
        value = value/denominator
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_log_loss

    subroutine classification_top_k_accuracy(probabilities, labels, classes, k, &
            value, status, sample_weight)
        !! Fraction of rows whose true class is among the deterministic top-k.
        real(dp), intent(in) :: probabilities(:, :)
        integer, intent(in) :: labels(:), classes(:), k
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator, weight, target_probability
        integer :: i, j, class_index, rank

        value = 0.0_dp
        call validate_probability_inputs(probabilities, labels, classes, status, &
            "top-k accuracy")
        if (status%code /= FORTNUM_OK) return
        if (k < 1 .or. k > size(classes)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "top-k accuracy: k is outside the class range")
            return
        end if
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "top-k accuracy")
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            class_index = class_position(labels(i), classes)
            if (class_index == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "top-k accuracy: label is not in classes")
                return
            end if
            target_probability = probabilities(i, class_index)
            rank = 1
            do j = 1, size(classes)
                if (probabilities(i, j) > target_probability) rank = rank + 1
                if (probabilities(i, j) == target_probability .and. &
                    j < class_index) rank = rank + 1
            end do
            weight = 1.0_dp
            if (present(sample_weight)) weight = sample_weight(i)
            if (rank <= k) value = value + weight
        end do
        value = value/denominator
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_top_k_accuracy

    subroutine classification_brier_score(probabilities, labels, classes, value, &
            status, sample_weight)
        !! Multiclass Brier score after row normalization of positive masses.
        real(dp), intent(in) :: probabilities(:, :)
        integer, intent(in) :: labels(:), classes(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator, row_sum, target, weight
        integer :: i, j, class_index

        value = 0.0_dp
        call validate_probability_inputs(probabilities, labels, classes, status, &
            "Brier score")
        if (status%code /= FORTNUM_OK) return
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "Brier score")
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            class_index = class_position(labels(i), classes)
            if (class_index == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Brier score: label is not in classes")
                return
            end if
            row_sum = sum(probabilities(i, :))
            weight = 1.0_dp
            if (present(sample_weight)) weight = sample_weight(i)
            do j = 1, size(classes)
                target = 0.0_dp
                if (j == class_index) target = 1.0_dp
                value = value + weight*(probabilities(i, j)/row_sum - target)**2
            end do
        end do
        value = value/denominator
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_brier_score

    subroutine classification_binary_matthews(labels, predictions, classes, value, &
            status, sample_weight)
        !! Binary Matthews correlation coefficient with optional sample weights.
        integer, intent(in) :: labels(:), predictions(:), classes(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator, tp, tn, fp, fn, weight, determinant
        integer :: i

        value = 0.0_dp
        call validate_label_vectors(labels, predictions, status, &
            "binary Matthews correlation")
        if (status%code /= FORTNUM_OK) return
        if (size(classes) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary Matthews correlation: exactly two classes are required")
            return
        end if
        call validate_classes(classes, status, "binary Matthews correlation")
        if (status%code /= FORTNUM_OK) return
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "binary Matthews correlation")
        if (status%code /= FORTNUM_OK) return
        tp = 0.0_dp
        tn = 0.0_dp
        fp = 0.0_dp
        fn = 0.0_dp
        do i = 1, size(labels)
            weight = 1.0_dp
            if (present(sample_weight)) weight = sample_weight(i)
            if (labels(i) == classes(2) .and. predictions(i) == classes(2)) then
                tp = tp + weight
            else if (labels(i) == classes(1) .and. predictions(i) == classes(1)) then
                tn = tn + weight
            else if (labels(i) == classes(1) .and. predictions(i) == classes(2)) then
                fp = fp + weight
            else if (labels(i) == classes(2) .and. predictions(i) == classes(1)) then
                fn = fn + weight
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "binary Matthews correlation: label is not in classes")
                return
            end if
        end do
        determinant = tp*tn - fp*fn
        denominator = sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        if (denominator > 0.0_dp) value = determinant/denominator
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_binary_matthews

    subroutine classification_multilabel_precision_recall_f1(labels, predictions, &
            precision, recall, f1, status, average, sample_weight, zero_division)
        !! Precision, recall, and F1 for a binary indicator matrix.
        !!
        !! `average` selects micro (global TP/FP/FN), macro (the unweighted
        !! mean of per-label scores), or samples (the sample-weighted mean of
        !! per-row scores).  Indicator entries must be exactly zero or one.
        !! A zero denominator is assigned `zero_division` (zero by default,
        !! one when explicitly requested), matching scikit-learn's explicit
        !! zero-division policy without warnings or hidden NaNs.
        integer, intent(in) :: labels(:, :), predictions(:, :)
        real(dp), intent(out) :: precision, recall, f1
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in) :: average
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: zero_division
        integer :: zero_value, i, j
        real(dp) :: weight, denominator, tp, fp, fn, row_tp, row_fp, row_fn
        real(dp) :: label_precision, label_recall, label_f1

        precision = 0.0_dp
        recall = 0.0_dp
        f1 = 0.0_dp
        zero_value = CLASSIFICATION_ZERO_DIVISION_ZERO
        if (present(zero_division)) zero_value = zero_division
        if (average < CLASSIFICATION_AVERAGE_MICRO .or. &
                average > CLASSIFICATION_AVERAGE_SAMPLES .or. &
                (zero_value /= CLASSIFICATION_ZERO_DIVISION_ZERO .and. &
                zero_value /= CLASSIFICATION_ZERO_DIVISION_ONE)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel metrics: average or zero-division policy is invalid")
            return
        end if
        if (size(labels, 1) < 1 .or. size(labels, 2) < 1 .or. &
                any(shape(predictions) /= shape(labels)) .or. &
                any((labels /= 0) .and. (labels /= 1)) .or. &
                any((predictions /= 0) .and. (predictions /= 1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel metrics: indicators must be nonempty binary matrices")
            return
        end if
        call validate_weights(size(labels, 1), sample_weight, denominator, status, &
            "multilabel metrics")
        if (status%code /= FORTNUM_OK) return

        select case (average)
        case (CLASSIFICATION_AVERAGE_MICRO)
            tp = 0.0_dp
            fp = 0.0_dp
            fn = 0.0_dp
            do i = 1, size(labels, 1)
                weight = 1.0_dp
                if (present(sample_weight)) weight = sample_weight(i)
                do j = 1, size(labels, 2)
                    if (labels(i, j) == 1 .and. predictions(i, j) == 1) then
                        tp = tp + weight
                    else if (labels(i, j) == 0 .and. predictions(i, j) == 1) then
                        fp = fp + weight
                    else if (labels(i, j) == 1 .and. predictions(i, j) == 0) then
                        fn = fn + weight
                    end if
                end do
            end do
            precision = safe_multilabel_ratio(tp, tp + fp, zero_value)
            recall = safe_multilabel_ratio(tp, tp + fn, zero_value)
            f1 = safe_multilabel_ratio(2.0_dp*tp, 2.0_dp*tp + fp + fn, zero_value)
        case (CLASSIFICATION_AVERAGE_MACRO)
            do j = 1, size(labels, 2)
                tp = 0.0_dp
                fp = 0.0_dp
                fn = 0.0_dp
                do i = 1, size(labels, 1)
                    weight = 1.0_dp
                    if (present(sample_weight)) weight = sample_weight(i)
                    if (labels(i, j) == 1 .and. predictions(i, j) == 1) then
                        tp = tp + weight
                    else if (labels(i, j) == 0 .and. predictions(i, j) == 1) then
                        fp = fp + weight
                    else if (labels(i, j) == 1 .and. predictions(i, j) == 0) then
                        fn = fn + weight
                    end if
                end do
                precision = precision + safe_multilabel_ratio(tp, tp + fp, zero_value)
                recall = recall + safe_multilabel_ratio(tp, tp + fn, zero_value)
                f1 = f1 + safe_multilabel_ratio(2.0_dp*tp, 2.0_dp*tp + fp + fn, zero_value)
            end do
            precision = precision/real(size(labels, 2), dp)
            recall = recall/real(size(labels, 2), dp)
            f1 = f1/real(size(labels, 2), dp)
        case (CLASSIFICATION_AVERAGE_SAMPLES)
            do i = 1, size(labels, 1)
                weight = 1.0_dp
                if (present(sample_weight)) weight = sample_weight(i)
                row_tp = 0.0_dp
                row_fp = 0.0_dp
                row_fn = 0.0_dp
                do j = 1, size(labels, 2)
                    if (labels(i, j) == 1 .and. predictions(i, j) == 1) then
                        row_tp = row_tp + 1.0_dp
                    else if (labels(i, j) == 0 .and. predictions(i, j) == 1) then
                        row_fp = row_fp + 1.0_dp
                    else if (labels(i, j) == 1 .and. predictions(i, j) == 0) then
                        row_fn = row_fn + 1.0_dp
                    end if
                end do
                label_precision = safe_multilabel_ratio(row_tp, row_tp + row_fp, zero_value)
                label_recall = safe_multilabel_ratio(row_tp, row_tp + row_fn, zero_value)
                label_f1 = safe_multilabel_ratio(2.0_dp*row_tp, &
                    2.0_dp*row_tp + row_fp + row_fn, zero_value)
                precision = precision + weight*label_precision/denominator
                recall = recall + weight*label_recall/denominator
                f1 = f1 + weight*label_f1/denominator
            end do
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_multilabel_precision_recall_f1

    subroutine classification_multilabel_probability_metrics(probabilities, labels, &
            precision, recall, f1, status, average, threshold, sample_weight, &
            zero_division)
        !! Threshold probabilities (`>= threshold` is positive), then evaluate
        !! the indicator-matrix metric contract above.  The default threshold
        !! is 0.5 and must be finite in the closed unit interval.
        real(dp), intent(in) :: probabilities(:, :)
        integer, intent(in) :: labels(:, :)
        real(dp), intent(out) :: precision, recall, f1
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in) :: average
        real(dp), intent(in), optional :: threshold
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: zero_division
        integer, allocatable :: predictions(:, :)
        real(dp) :: cutoff
        integer :: i, j

        precision = 0.0_dp
        recall = 0.0_dp
        f1 = 0.0_dp
        cutoff = 0.5_dp
        if (present(threshold)) cutoff = threshold
        if (.not. ieee_is_finite(cutoff) .or. cutoff < 0.0_dp .or. cutoff > 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel probability metrics: threshold must be in [0,1]")
            return
        end if
        if (size(probabilities, 1) < 1 .or. size(probabilities, 2) < 1 .or. &
                any(shape(probabilities) /= shape(labels)) .or. &
                any(.not. ieee_is_finite(probabilities)) .or. &
                any(probabilities < 0.0_dp) .or. any(probabilities > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel probability metrics: probabilities or shapes are invalid")
            return
        end if
        allocate(predictions(size(labels, 1), size(labels, 2)))
        do i = 1, size(labels, 1)
            do j = 1, size(labels, 2)
                predictions(i, j) = merge(1, 0, probabilities(i, j) >= cutoff)
            end do
        end do
        call classification_multilabel_precision_recall_f1(labels, predictions, &
            precision, recall, f1, status, average, sample_weight, zero_division)
    end subroutine classification_multilabel_probability_metrics

    subroutine classification_multilabel_jaccard(labels, predictions, value, &
            status, average, sample_weight, zero_division)
        !! Jaccard similarity for binary indicator matrices.
        !!
        !! `average` has the same values as the precision/recall contract:
        !! micro aggregates intersections and unions globally, macro averages
        !! one score per label, and samples averages one score per row.  A
        !! zero union is assigned `zero_division` (zero by default), making
        !! empty-target and empty-prediction rows explicit and deterministic.
        integer, intent(in) :: labels(:, :), predictions(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: average
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: zero_division
        integer :: average_value, zero_value, i, j
        real(dp) :: denominator, weight, intersection, union, row_intersection
        real(dp) :: row_union

        value = 0.0_dp
        average_value = CLASSIFICATION_AVERAGE_MACRO
        if (present(average)) average_value = average
        zero_value = CLASSIFICATION_ZERO_DIVISION_ZERO
        if (present(zero_division)) zero_value = zero_division
        if (average_value < CLASSIFICATION_AVERAGE_MICRO .or. &
                average_value > CLASSIFICATION_AVERAGE_SAMPLES .or. &
                (zero_value /= CLASSIFICATION_ZERO_DIVISION_ZERO .and. &
                zero_value /= CLASSIFICATION_ZERO_DIVISION_ONE)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel Jaccard: average or zero-division policy is invalid")
            return
        end if
        call validate_multilabel_inputs(labels, predictions, status, &
            "multilabel Jaccard")
        if (status%code /= FORTNUM_OK) return
        call validate_weights(size(labels, 1), sample_weight, denominator, status, &
            "multilabel Jaccard")
        if (status%code /= FORTNUM_OK) return

        select case (average_value)
        case (CLASSIFICATION_AVERAGE_MICRO)
            intersection = 0.0_dp
            union = 0.0_dp
            do i = 1, size(labels, 1)
                weight = 1.0_dp
                if (present(sample_weight)) weight = sample_weight(i)
                do j = 1, size(labels, 2)
                    if (labels(i, j) == 1 .and. predictions(i, j) == 1) &
                        intersection = intersection + weight
                    if (labels(i, j) == 1 .or. predictions(i, j) == 1) &
                        union = union + weight
                end do
            end do
            value = safe_multilabel_ratio(intersection, union, zero_value)
        case (CLASSIFICATION_AVERAGE_MACRO)
            do j = 1, size(labels, 2)
                intersection = 0.0_dp
                union = 0.0_dp
                do i = 1, size(labels, 1)
                    weight = 1.0_dp
                    if (present(sample_weight)) weight = sample_weight(i)
                    if (labels(i, j) == 1 .and. predictions(i, j) == 1) &
                        intersection = intersection + weight
                    if (labels(i, j) == 1 .or. predictions(i, j) == 1) &
                        union = union + weight
                end do
                value = value + safe_multilabel_ratio(intersection, union, zero_value)
            end do
            value = value/real(size(labels, 2), dp)
        case (CLASSIFICATION_AVERAGE_SAMPLES)
            do i = 1, size(labels, 1)
                weight = 1.0_dp
                if (present(sample_weight)) weight = sample_weight(i)
                row_intersection = 0.0_dp
                row_union = 0.0_dp
                do j = 1, size(labels, 2)
                    if (labels(i, j) == 1 .and. predictions(i, j) == 1) &
                        row_intersection = row_intersection + 1.0_dp
                    if (labels(i, j) == 1 .or. predictions(i, j) == 1) &
                        row_union = row_union + 1.0_dp
                end do
                value = value + weight*safe_multilabel_ratio(row_intersection, &
                    row_union, zero_value)/denominator
            end do
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_multilabel_jaccard

    subroutine classification_multilabel_hamming_loss(labels, predictions, value, &
            status, sample_weight, average)
        !! Hamming loss for binary indicator matrices.
        !!
        !! The default `micro` contract is the weighted fraction of incorrect
        !! indicators.  `macro` averages per-label fractions and `samples`
        !! averages per-row fractions.  All forms use row sample weights.
        integer, intent(in) :: labels(:, :), predictions(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: average
        integer :: average_value, i, j
        real(dp) :: denominator, weight, errors, row_errors

        value = 0.0_dp
        average_value = CLASSIFICATION_AVERAGE_MICRO
        if (present(average)) average_value = average
        if (average_value < CLASSIFICATION_AVERAGE_MICRO .or. &
                average_value > CLASSIFICATION_AVERAGE_SAMPLES) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel Hamming loss: average policy is invalid")
            return
        end if
        call validate_multilabel_inputs(labels, predictions, status, &
            "multilabel Hamming loss")
        if (status%code /= FORTNUM_OK) return
        call validate_weights(size(labels, 1), sample_weight, denominator, status, &
            "multilabel Hamming loss")
        if (status%code /= FORTNUM_OK) return

        select case (average_value)
        case (CLASSIFICATION_AVERAGE_MICRO)
            errors = 0.0_dp
            do i = 1, size(labels, 1)
                weight = 1.0_dp
                if (present(sample_weight)) weight = sample_weight(i)
                do j = 1, size(labels, 2)
                    if (labels(i, j) /= predictions(i, j)) errors = errors + weight
                end do
            end do
            value = errors/(denominator*real(size(labels, 2), dp))
        case (CLASSIFICATION_AVERAGE_MACRO)
            do j = 1, size(labels, 2)
                errors = 0.0_dp
                do i = 1, size(labels, 1)
                    weight = 1.0_dp
                    if (present(sample_weight)) weight = sample_weight(i)
                    if (labels(i, j) /= predictions(i, j)) errors = errors + weight
                end do
                value = value + errors/denominator
            end do
            value = value/real(size(labels, 2), dp)
        case (CLASSIFICATION_AVERAGE_SAMPLES)
            do i = 1, size(labels, 1)
                weight = 1.0_dp
                if (present(sample_weight)) weight = sample_weight(i)
                row_errors = 0.0_dp
                do j = 1, size(labels, 2)
                    if (labels(i, j) /= predictions(i, j)) &
                        row_errors = row_errors + 1.0_dp
                end do
                value = value + weight*row_errors/ &
                    (denominator*real(size(labels, 2), dp))
            end do
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_multilabel_hamming_loss

    subroutine classification_multilabel_hamming(labels, predictions, value, &
            status, sample_weight)
        !! Short alias for `classification_multilabel_hamming_loss`.
        integer, intent(in) :: labels(:, :), predictions(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)

        if (present(sample_weight)) then
            call classification_multilabel_hamming_loss(labels, predictions, value, &
                status, sample_weight=sample_weight)
        else
            call classification_multilabel_hamming_loss(labels, predictions, value, status)
        end if
    end subroutine classification_multilabel_hamming

    subroutine classification_roc_auc(scores, labels, positive_class, value, &
            status, sample_weight)
        !! Weighted binary ROC area with deterministic half-credit ties.
        !!
        !! `positive_class` selects the positive label; all remaining rows must
        !! carry one single negative label.  A score tie contributes exactly
        !! one half of a correctly ordered pair.  Missing positive/negative
        !! support is a typed domain error rather than a NaN or warning.
        real(dp), intent(in) :: scores(:)
        integer, intent(in) :: labels(:), positive_class
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: weights(:)
        real(dp) :: denominator
        integer :: i, negative_class
        logical :: negative_seen

        value = 0.0_dp
        if (size(scores) < 1 .or. size(labels) /= size(scores) .or. &
            any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ROC AUC: scores and labels have invalid shapes or scores are nonfinite")
            return
        end if
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "ROC AUC")
        if (status%code /= FORTNUM_OK) return
        allocate(weights(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) weights = sample_weight
        negative_class = 0
        negative_seen = .false.
        do i = 1, size(labels)
            if (labels(i) /= positive_class) then
                if (.not. negative_seen) then
                    negative_class = labels(i)
                    negative_seen = .true.
                else if (labels(i) /= negative_class) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "ROC AUC: binary labels contain more than two classes")
                    return
                end if
            end if
        end do
        if (.not. negative_seen .or. sum(weights, mask=labels == positive_class) <= 0.0_dp .or. &
            sum(weights, mask=labels /= positive_class) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ROC AUC: positive and negative support are both required")
            return
        end if
        call roc_auc_pair(scores, labels == positive_class, weights, value)
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_roc_auc

    subroutine classification_roc_auc_ovr(scores, labels, classes, value, status, &
            sample_weight, per_class)
        !! Macro one-vs-rest ROC area for multiclass score columns.
        !!
        !! `scores(:,j)` is the score for `classes(j)`.  Each class must have
        !! positive weighted support and at least one weighted negative row;
        !! this strict policy makes the macro average deterministic and avoids
        !! silently returning the warning/NaN convention used by some APIs.
        real(dp), intent(in) :: scores(:, :)
        integer, intent(in) :: labels(:), classes(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(out), optional :: per_class(:)
        real(dp), allocatable :: weights(:)
        real(dp) :: denominator, positive_mass, negative_mass, auc
        integer :: i, j, class_index

        value = 0.0_dp
        if (present(per_class)) per_class = 0.0_dp
        if (size(scores, 1) < 1 .or. size(scores, 2) /= size(classes) .or. &
            size(classes) < 2 .or. size(labels) /= size(scores, 1) .or. &
            any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ROC AUC OVR: scores, labels, or class shapes are invalid")
            return
        end if
        if (present(per_class)) then
            if (size(per_class) /= size(classes)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ROC AUC OVR: per-class output shape is invalid")
                return
            end if
        end if
        call validate_classes(classes, status, "ROC AUC OVR")
        if (status%code /= FORTNUM_OK) return
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "ROC AUC OVR")
        if (status%code /= FORTNUM_OK) return
        allocate(weights(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) weights = sample_weight
        do i = 1, size(labels)
            class_index = class_position(labels(i), classes)
            if (class_index == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ROC AUC OVR: labels must belong to classes")
                return
            end if
        end do
        do j = 1, size(classes)
            positive_mass = sum(weights, mask=labels == classes(j))
            negative_mass = sum(weights, mask=labels /= classes(j))
            if (positive_mass <= 0.0_dp .or. negative_mass <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ROC AUC OVR: every class needs positive and negative support")
                return
            end if
            call roc_auc_pair(scores(:, j), labels == classes(j), weights, auc)
            if (present(per_class)) per_class(j) = auc
            value = value + auc
        end do
        if (present(per_class)) then
            value = value/real(size(classes), dp)
        else
            value = value/real(size(classes), dp)
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_roc_auc_ovr

    subroutine classification_roc_auc_device(device, scores, labels, positive_class, &
            value, status, sample_weight)
        !! Device dispatch for binary ROC AUC; CUDA is explicitly unavailable.
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: scores(:)
        integer, intent(in) :: labels(:), positive_class
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ROC AUC device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            if (present(sample_weight)) then
                call classification_roc_auc(scores, labels, positive_class, value, &
                    status, sample_weight)
            else
                call classification_roc_auc(scores, labels, positive_class, value, status)
            end if
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ROC AUC device: no resident CUDA reduction is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ROC AUC device: device kind is invalid")
        end select
    end subroutine classification_roc_auc_device

    subroutine classification_roc_auc_ovr_device(device, scores, labels, classes, value, &
            status, sample_weight, per_class)
        !! Device dispatch for macro one-vs-rest ROC AUC.
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: scores(:, :)
        integer, intent(in) :: labels(:), classes(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(out), optional :: per_class(:)

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ROC AUC OVR device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            if (present(sample_weight)) then
                if (present(per_class)) then
                    call classification_roc_auc_ovr(scores, labels, classes, value, &
                        status, sample_weight, per_class)
                else
                    call classification_roc_auc_ovr(scores, labels, classes, value, &
                        status, sample_weight=sample_weight)
                end if
            else if (present(per_class)) then
                call classification_roc_auc_ovr(scores, labels, classes, value, &
                    status, per_class=per_class)
            else
                call classification_roc_auc_ovr(scores, labels, classes, value, status)
            end if
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ROC AUC OVR device: no resident CUDA reduction is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ROC AUC OVR device: device kind is invalid")
        end select
    end subroutine classification_roc_auc_ovr_device

    subroutine classification_pr_auc(scores, labels, positive_class, value, &
            status, sample_weight)
        !! Binary precision-recall area using average-precision step semantics.
        !!
        !! Scores are processed in descending threshold order.  Equal-score
        !! rows are consumed as one threshold group, avoiding arbitrary input
        !! order dependence.  The area is the sum of precision at each group
        !! times its recall increment, matching the usual weighted average
        !! precision definition.  Both positive and negative weighted support
        !! are required; otherwise a domain status is returned.
        real(dp), intent(in) :: scores(:)
        integer, intent(in) :: labels(:), positive_class
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: weights(:)
        real(dp) :: denominator, positive_mass, negative_mass
        integer :: negative_class

        value = 0.0_dp
        if (size(scores) < 1 .or. size(labels) /= size(scores) .or. &
                any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PR AUC: scores and labels have invalid shapes or scores are nonfinite")
            return
        end if
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "PR AUC")
        if (status%code /= FORTNUM_OK) return
        allocate(weights(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) weights = sample_weight
        call validate_binary_ranking_labels(labels, positive_class, negative_class, &
            status, "PR AUC")
        if (status%code /= FORTNUM_OK) return
        positive_mass = sum(weights, mask=labels == positive_class)
        negative_mass = sum(weights, mask=labels == negative_class)
        if (positive_mass <= 0.0_dp .or. negative_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PR AUC: positive and negative support must have positive weight")
            return
        end if
        call pr_auc_average_precision(scores, labels == positive_class, weights, value)
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_pr_auc

    subroutine classification_pr_auc_ovr(scores, labels, classes, value, status, &
            sample_weight, per_class)
        !! Macro one-vs-rest weighted average precision.
        real(dp), intent(in) :: scores(:, :)
        integer, intent(in) :: labels(:), classes(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(out), optional :: per_class(:)
        real(dp), allocatable :: weights(:)
        real(dp) :: denominator, positive_mass, negative_mass, ap
        integer :: i, j, class_index

        value = 0.0_dp
        if (present(per_class)) per_class = 0.0_dp
        if (size(scores, 1) < 1 .or. size(scores, 2) /= size(classes) .or. &
                size(classes) < 2 .or. size(labels) /= size(scores, 1) .or. &
                any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PR AUC OVR: scores, labels, or class shapes are invalid")
            return
        end if
        if (present(per_class)) then
            if (size(per_class) /= size(classes)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "PR AUC OVR: per-class output shape is invalid")
                return
            end if
        end if
        call validate_classes(classes, status, "PR AUC OVR")
        if (status%code /= FORTNUM_OK) return
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "PR AUC OVR")
        if (status%code /= FORTNUM_OK) return
        allocate(weights(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) weights = sample_weight
        do i = 1, size(labels)
            class_index = class_position(labels(i), classes)
            if (class_index == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "PR AUC OVR: labels must belong to classes")
                return
            end if
        end do
        do j = 1, size(classes)
            positive_mass = sum(weights, mask=labels == classes(j))
            negative_mass = sum(weights, mask=labels /= classes(j))
            if (positive_mass <= 0.0_dp .or. negative_mass <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "PR AUC OVR: every class needs positive and negative support")
                return
            end if
            call pr_auc_average_precision(scores(:, j), labels == classes(j), &
                weights, ap)
            if (present(per_class)) per_class(j) = ap
            value = value + ap
        end do
        value = value/real(size(classes), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_pr_auc_ovr

    subroutine classification_pr_auc_device(device, scores, labels, positive_class, &
            value, status, sample_weight)
        !! Device dispatch for binary PR AUC; CUDA has no linked ranking kernel.
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: scores(:)
        integer, intent(in) :: labels(:), positive_class
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PR AUC device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            if (present(sample_weight)) then
                call classification_pr_auc(scores, labels, positive_class, value, &
                    status, sample_weight)
            else
                call classification_pr_auc(scores, labels, positive_class, value, status)
            end if
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "PR AUC device: no resident ranking reduction is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PR AUC device: device kind is invalid")
        end select
    end subroutine classification_pr_auc_device

    subroutine classification_pr_auc_ovr_device(device, scores, labels, classes, value, &
            status, sample_weight, per_class)
        !! Device dispatch for macro one-vs-rest PR AUC.
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: scores(:, :)
        integer, intent(in) :: labels(:), classes(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(out), optional :: per_class(:)

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PR AUC OVR device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            if (present(sample_weight)) then
                if (present(per_class)) then
                    call classification_pr_auc_ovr(scores, labels, classes, value, &
                        status, sample_weight, per_class)
                else
                    call classification_pr_auc_ovr(scores, labels, classes, value, &
                        status, sample_weight=sample_weight)
                end if
            else if (present(per_class)) then
                call classification_pr_auc_ovr(scores, labels, classes, value, status, &
                    per_class=per_class)
            else
                call classification_pr_auc_ovr(scores, labels, classes, value, status)
            end if
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "PR AUC OVR device: no resident ranking reduction is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PR AUC OVR device: device kind is invalid")
        end select
    end subroutine classification_pr_auc_ovr_device

    subroutine roc_auc_pair(scores, positive, weights, value)
        real(dp), intent(in) :: scores(:), weights(:)
        logical, intent(in) :: positive(:)
        real(dp), intent(out) :: value
        real(dp) :: positive_mass, negative_mass, concordance
        integer :: i, j

        positive_mass = sum(weights, mask=positive)
        negative_mass = sum(weights, mask=.not. positive)
        concordance = 0.0_dp
        do i = 1, size(scores)
            if (.not. positive(i)) cycle
            do j = 1, size(scores)
                if (positive(j)) cycle
                if (scores(i) > scores(j)) then
                    concordance = concordance + weights(i)*weights(j)
                else if (scores(i) == scores(j)) then
                    concordance = concordance + 0.5_dp*weights(i)*weights(j)
                end if
            end do
        end do
        value = concordance/(positive_mass*negative_mass)
    end subroutine roc_auc_pair

    subroutine classification_calibration_error(probabilities, labels, classes, &
            bins, value, status, sample_weight)
        !! Weighted expected calibration error of multiclass probabilities.
        !!
        !! Each row contributes its normalized maximum class probability as
        !! confidence and the deterministic first-maximum class as its
        !! prediction.  Equal-width confidence bins use a right-closed final
        !! bin, so confidence one is always included.  Empty bins contribute
        !! zero.  The result is the sample-weighted mean absolute gap between
        !! bin accuracy and bin confidence.
        real(dp), intent(in) :: probabilities(:, :)
        integer, intent(in) :: labels(:), classes(:), bins
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: maximum_error

        call calibration_errors(probabilities, labels, classes, bins, value, &
            maximum_error, status, sample_weight)
    end subroutine classification_calibration_error

    subroutine classification_maximum_calibration_error(probabilities, labels, &
            classes, bins, value, status, sample_weight)
        !! Maximum per-bin absolute confidence/accuracy gap.
        real(dp), intent(in) :: probabilities(:, :)
        integer, intent(in) :: labels(:), classes(:), bins
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: expected_error

        call calibration_errors(probabilities, labels, classes, bins, &
            expected_error, value, status, sample_weight)
    end subroutine classification_maximum_calibration_error

    subroutine calibration_errors(probabilities, labels, classes, bins, expected, &
            maximum, status, sample_weight)
        real(dp), intent(in) :: probabilities(:, :)
        integer, intent(in) :: labels(:), classes(:), bins
        real(dp), intent(out) :: expected, maximum
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: bin_weight(:), confidence_sum(:), correct_sum(:)
        real(dp) :: denominator, row_sum, confidence, weight, gap
        integer :: i, j, bin, predicted, class_index

        expected = 0.0_dp
        maximum = 0.0_dp
        call validate_probability_inputs(probabilities, labels, classes, status, &
            "calibration error")
        if (status%code /= FORTNUM_OK) return
        if (bins < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibration error: bin count must be positive")
            return
        end if
        call validate_weights(size(labels), sample_weight, denominator, status, &
            "calibration error")
        if (status%code /= FORTNUM_OK) return
        allocate(bin_weight(bins), confidence_sum(bins), correct_sum(bins))
        bin_weight = 0.0_dp
        confidence_sum = 0.0_dp
        correct_sum = 0.0_dp
        do i = 1, size(labels)
            class_index = class_position(labels(i), classes)
            if (class_index == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibration error: label is not in classes")
                return
            end if
            row_sum = sum(probabilities(i, :))
            confidence = maxval(probabilities(i, :))/row_sum
            predicted = 1
            do j = 2, size(classes)
                if (probabilities(i, j) > probabilities(i, predicted)) then
                    predicted = j
                end if
            end do
            bin = int(confidence*real(bins, dp)) + 1
            if (bin > bins) bin = bins
            weight = 1.0_dp
            if (present(sample_weight)) weight = sample_weight(i)
            bin_weight(bin) = bin_weight(bin) + weight
            confidence_sum(bin) = confidence_sum(bin) + weight*confidence
            if (predicted == class_index) correct_sum(bin) = &
                correct_sum(bin) + weight
        end do
        do bin = 1, bins
            if (bin_weight(bin) > 0.0_dp) then
                gap = abs(correct_sum(bin)/bin_weight(bin) - &
                    confidence_sum(bin)/bin_weight(bin))
                expected = expected + bin_weight(bin)/denominator*gap
                maximum = max(maximum, gap)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine calibration_errors

    subroutine validate_multilabel_inputs(labels, predictions, status, context)
        integer, intent(in) :: labels(:, :), predictions(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        if (size(labels, 1) < 1 .or. size(labels, 2) < 1 .or. &
                any(shape(predictions) /= shape(labels))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": indicator matrices must be nonempty and have equal shape")
            return
        end if
        if (any((labels /= 0) .and. (labels /= 1)) .or. &
                any((predictions /= 0) .and. (predictions /= 1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": indicators must be exactly zero or one")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_multilabel_inputs

    subroutine validate_binary_ranking_labels(labels, positive_class, negative_class, &
            status, context)
        integer, intent(in) :: labels(:), positive_class
        integer, intent(out) :: negative_class
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context
        integer :: i
        logical :: negative_seen

        negative_class = 0
        negative_seen = .false.
        do i = 1, size(labels)
            if (labels(i) == positive_class) cycle
            if (.not. negative_seen) then
                negative_class = labels(i)
                negative_seen = .true.
            else if (labels(i) /= negative_class) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    trim(context)//": labels must contain exactly two classes")
                return
            end if
        end do
        if (.not. negative_seen) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": positive and negative classes are both required")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_binary_ranking_labels

    subroutine pr_auc_average_precision(scores, positive, weights, value)
        real(dp), intent(in) :: scores(:), weights(:)
        logical, intent(in) :: positive(:)
        real(dp), intent(out) :: value
        logical, allocatable :: consumed(:)
        real(dp) :: threshold, previous_true_mass, true_mass, predicted_mass
        real(dp) :: positive_mass, recall_increment, precision
        integer :: i, n_consumed

        value = 0.0_dp
        positive_mass = sum(weights, mask=positive)
        previous_true_mass = 0.0_dp
        allocate(consumed(size(scores)))
        consumed = .false.
        n_consumed = 0
        do while (n_consumed < size(scores))
            threshold = -huge(1.0_dp)
            do i = 1, size(scores)
                if (.not. consumed(i)) threshold = max(threshold, scores(i))
            end do
            do i = 1, size(scores)
                if (.not. consumed(i) .and. scores(i) == threshold) then
                    consumed(i) = .true.
                    n_consumed = n_consumed + 1
                end if
            end do
            true_mass = sum(weights, mask=positive .and. scores >= threshold)
            predicted_mass = sum(weights, mask=scores >= threshold)
            recall_increment = (true_mass - previous_true_mass)/positive_mass
            if (predicted_mass > 0.0_dp .and. recall_increment > 0.0_dp) then
                precision = true_mass/predicted_mass
                value = value + recall_increment*precision
            end if
            previous_true_mass = true_mass
        end do
    end subroutine pr_auc_average_precision

    subroutine validate_probability_inputs(probabilities, labels, classes, status, &
            context)
        real(dp), intent(in) :: probabilities(:, :)
        integer, intent(in) :: labels(:), classes(:)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context
        real(dp) :: row_sum
        integer :: i

        if (size(probabilities, 1) /= size(labels) .or. &
            size(probabilities, 2) /= size(classes) .or. &
            size(labels) < 1 .or. size(classes) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": input shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(probabilities)) .or. &
            any(probabilities < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": probabilities must be finite and nonnegative")
            return
        end if
        call validate_classes(classes, status, context)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            row_sum = sum(probabilities(i, :))
            if (.not. ieee_is_finite(row_sum) .or. row_sum <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    trim(context)//": each probability row needs positive mass")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_probability_inputs

    subroutine validate_label_vectors(labels, predictions, status, context)
        integer, intent(in) :: labels(:), predictions(:)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        if (size(labels) < 1 .or. size(predictions) /= size(labels)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": label vectors have invalid shapes")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_label_vectors

    subroutine validate_classes(classes, status, context)
        integer, intent(in) :: classes(:)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context
        integer :: i, j

        if (size(classes) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": classes must not be empty")
            return
        end if
        do i = 1, size(classes)
            do j = i + 1, size(classes)
                if (classes(i) == classes(j)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        trim(context)//": classes must be unique")
                    return
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_classes

    subroutine validate_weights(n, sample_weight, denominator, status, context)
        integer, intent(in) :: n
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(out) :: denominator
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        denominator = real(n, dp)
        if (.not. present(sample_weight)) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (size(sample_weight) /= n .or. any(.not. ieee_is_finite(sample_weight)) &
            .or. any(sample_weight < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": weights must be finite and nonnegative")
            return
        end if
        denominator = sum(sample_weight)
        if (.not. ieee_is_finite(denominator) .or. denominator <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": weights must have positive mass")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_weights

    real(dp) function safe_multilabel_ratio(numerator, denominator, zero_value) result(value)
        real(dp), intent(in) :: numerator, denominator
        integer, intent(in) :: zero_value

        if (denominator > 0.0_dp) then
            value = numerator/denominator
        else
            value = real(zero_value, dp)
        end if
    end function safe_multilabel_ratio

    integer function class_position(label, classes) result(position)
        integer, intent(in) :: label
        integer, intent(in) :: classes(:)
        integer :: i

        position = 0
        do i = 1, size(classes)
            if (classes(i) == label) then
                position = i
                return
            end if
        end do
    end function class_position

end module fortml_classification_metrics
