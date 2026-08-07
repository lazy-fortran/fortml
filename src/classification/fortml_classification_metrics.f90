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
        FORTNUM_DOMAIN_ERROR
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
