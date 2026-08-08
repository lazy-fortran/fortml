program test_multiclass_isotonic_calibration
    !! Independent weighted one-vs-rest PAVA and refusal checks.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_probability_calibration, only: &
        multiclass_probability_calibrator_t, probability_calibration_options_t, &
        probability_calibration_state_t, CALIBRATION_ISOTONIC, CALIBRATION_SIGMOID
    implicit none

    type(multiclass_probability_calibrator_t) :: model, repeated, invalid, unfitted
    type(probability_calibration_options_t) :: options
    type(probability_calibration_state_t) :: state
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: scores(9, 3), weights(9), probabilities(9, 3)
    real(dp) :: expected(9, 3), repeated_probabilities(9, 3)
    real(dp) :: scores_dot(9, 3), probabilities_dot(9, 3)
    real(dp) :: probabilities_bar(9, 3), scores_bar(9, 3)
    real(dp) :: empty(0)
    integer :: labels(9), predicted(9), classes(3), expected_labels(9), failures
    real(dp), allocatable :: parameters(:)

    scores(1, :) = [3.0_dp, 0.0_dp, -2.0_dp]
    scores(2, :) = [2.0_dp, 1.0_dp, -1.0_dp]
    scores(3, :) = [1.0_dp, 2.0_dp, -1.0_dp]
    scores(4, :) = [0.0_dp, 3.0_dp, -1.0_dp]
    scores(5, :) = [-1.0_dp, 2.0_dp, 1.0_dp]
    scores(6, :) = [-2.0_dp, 1.0_dp, 3.0_dp]
    scores(7, :) = [1.0_dp, -1.0_dp, 2.0_dp]
    scores(8, :) = [2.0_dp, -2.0_dp, 1.0_dp]
    scores(9, :) = [-1.0_dp, -2.0_dp, 2.0_dp]
    labels = [7, 7, 42, 42, 42, 99, 99, 7, 99]
    weights = [1.0_dp, 0.5_dp, 1.0_dp, 2.0_dp, 1.0_dp, 1.5_dp, 0.75_dp, 1.25_dp, 0.8_dp]
    scores_dot = reshape([ &
        0.1_dp, -0.2_dp, 0.3_dp, -0.2_dp, 0.4_dp, -0.1_dp, &
        0.3_dp, 0.1_dp, -0.2_dp, -0.1_dp, 0.2_dp, 0.4_dp, &
        0.2_dp, -0.3_dp, 0.1_dp, -0.4_dp, 0.2_dp, 0.3_dp, &
        0.1_dp, 0.2_dp, -0.2_dp, -0.2_dp, 0.1_dp, 0.4_dp, &
        -0.3_dp, 0.2_dp, 0.1_dp], shape(scores_dot), order=[2, 1])
    probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp, 0.4_dp, -0.3_dp, &
        0.1_dp, 0.5_dp, -0.2_dp, 0.3_dp, -0.4_dp, 0.2_dp, &
        0.5_dp, -0.1_dp, 0.6_dp, -0.2_dp, 0.2_dp, -0.4_dp, &
        0.3_dp, -0.2_dp, 0.4_dp, -0.1_dp, 0.2_dp, 0.5_dp, &
        -0.2_dp, 0.3_dp, 0.1_dp], shape(probabilities_bar), order=[2, 1])
    failures = 0
    options = probability_calibration_options_t(method=CALIBRATION_ISOTONIC)

    call model%fit(scores, labels, status, options=options, sample_weight=weights, state=state)
    call check(status_ok(status) .and. model%fitted() .and. state%converged, &
        "multiclass isotonic fit status", failures)
    classes = model%classes()
    call check(all(classes == [7, 42, 99]) .and. model%method() == CALIBRATION_ISOTONIC, &
        "sorted classes and isotonic method metadata", failures)
    call check(state%knot_count > 0 .and. model%parameter_count() == 0, &
        "isotonic knot and parameter metadata", failures)
    parameters = model%parameters()
    call check(size(parameters) == 0, "isotonic has no smooth parameters", failures)

    expected = reference_isotonic(scores, labels, weights, classes)
    call model%predict_proba(scores, probabilities, status)
    call check(status_ok(status) .and. maxval(abs(probabilities-expected)) < 2.0e-13_dp, &
        "weighted one-vs-rest PAVA probability oracle", failures)
    call check(maxval(abs(sum(probabilities, dim=2)-1.0_dp)) < 2.0e-14_dp .and. &
        minval(probabilities) >= 0.0_dp .and. maxval(probabilities) <= 1.0_dp, &
        "isotonic simplex and bounds", failures)
    call model%predict(scores, predicted, status)
    expected_labels = classes(maxloc(expected, dim=2))
    call check(status_ok(status) .and. all(predicted == expected_labels), &
        "isotonic label prediction follows calibrated argmax", failures)

    call repeated%fit(scores, labels, status, options=options, sample_weight=weights)
    call repeated%predict_proba(scores, repeated_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(repeated_probabilities-probabilities)) < &
        2.0e-14_dp, "deterministic weighted isotonic calibration", failures)

    call model%predict_proba_jvp(scores, scores_dot, probabilities, probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. maxval(abs(probabilities_dot)) == 0.0_dp, &
        "isotonic score derivative refusal", failures)
    call model%predict_proba_vjp(scores, probabilities_bar, scores_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. maxval(abs(scores_bar)) == 0.0_dp, &
        "isotonic score adjoint refusal", failures)
    call model%predict_proba_parameter_jvp(scores, empty, probabilities, probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "isotonic parameter JVP refusal", failures)
    call model%predict_proba_parameter_vjp(scores, probabilities_bar, empty, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "isotonic parameter VJP refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, scores, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), "isotonic CUDA refusal", failures)

    call invalid%fit(scores, labels, status, options= &
        probability_calibration_options_t(method=CALIBRATION_SIGMOID))
    call check(status_ok(status) .and. invalid%method() == CALIBRATION_SIGMOID .and. &
        invalid%parameter_count() == 6, "multiclass Platt policy fit", failures)
    call unfitted%predict_proba(scores, probabilities, status)
    call check(.not. status_ok(status), "unfitted isotonic prediction refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL multiclass isotonic cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS multiclass isotonic independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [multiclass-isotonic] "//description
        end if
    end subroutine check

    function reference_isotonic(logits, labels, sample_weight, classes) result(probabilities)
        real(dp), intent(in) :: logits(:, :), sample_weight(:)
        integer, intent(in) :: labels(:), classes(:)
        real(dp) :: probabilities(size(logits, 1), size(logits, 2))
        real(dp) :: raw(size(logits, 1), size(logits, 2)), normalizer
        real(dp) :: knots(size(logits, 1)), values(size(logits, 1))
        integer :: i, j, knot_count

        do i = 1, size(logits, 1)
            raw(i, :) = exp(logits(i, :)-maxval(logits(i, :)))
            raw(i, :) = raw(i, :)/sum(raw(i, :))
        end do
        do j = 1, size(logits, 2)
            call reference_column(raw(:, j), labels, sample_weight, classes(j), knots, values, knot_count)
            do i = 1, size(logits, 1)
                probabilities(i, j) = reference_value(knots, values, knot_count, raw(i, j))
            end do
        end do
        do i = 1, size(logits, 1)
            normalizer = sum(probabilities(i, :))
            probabilities(i, :) = probabilities(i, :)/normalizer
        end do
    end function reference_isotonic

    subroutine reference_column(scores, labels, sample_weight, class_label, knots, values, knot_count)
        real(dp), intent(in) :: scores(:), sample_weight(:)
        integer, intent(in) :: labels(:), class_label
        real(dp), intent(out) :: knots(:), values(:)
        integer, intent(out) :: knot_count
        real(dp) :: x(size(scores)), w(size(scores))
        real(dp) :: y(size(scores)), block_x(size(scores)), block_w(size(scores))
        integer :: order(size(scores)), i, j, key, m, n_blocks
        real(dp) :: previous_mean, current_mean

        knots = 0.0_dp
        values = 0.0_dp
        order = [(i, i=1,size(scores))]
        do i = 2, size(scores)
            key = order(i)
            j = i-1
            do while (j >= 1)
                if (scores(order(j)) <= scores(key)) exit
                order(j+1) = order(j)
                j = j-1
            end do
            order(j+1) = key
        end do
        m = 0
        do i = 1, size(scores)
            if (sample_weight(order(i)) <= 0.0_dp) cycle
            if (m == 0) then
                m = m+1
                x(m) = scores(order(i))
                w(m) = sample_weight(order(i))
                y(m) = w(m)*merge(1.0_dp, 0.0_dp, labels(order(i)) == class_label)
            else if (scores(order(i)) /= x(m)) then
                m = m+1
                x(m) = scores(order(i))
                w(m) = sample_weight(order(i))
                y(m) = w(m)*merge(1.0_dp, 0.0_dp, labels(order(i)) == class_label)
            else
                w(m) = w(m)+sample_weight(order(i))
                y(m) = y(m)+sample_weight(order(i))* &
                    merge(1.0_dp, 0.0_dp, labels(order(i)) == class_label)
            end if
        end do
        n_blocks = 0
        do i = 1, m
            n_blocks = n_blocks+1
            block_x(n_blocks) = x(i)
            block_w(n_blocks) = w(i)
            y(n_blocks) = y(i)
            do while (n_blocks >= 2)
                previous_mean = y(n_blocks-1)/block_w(n_blocks-1)
                current_mean = y(n_blocks)/block_w(n_blocks)
                if (previous_mean <= current_mean) exit
                block_x(n_blocks-1) = (block_w(n_blocks-1)*block_x(n_blocks-1)+ &
                    block_w(n_blocks)*block_x(n_blocks))/ &
                    (block_w(n_blocks-1)+block_w(n_blocks))
                y(n_blocks-1) = y(n_blocks-1)+y(n_blocks)
                block_w(n_blocks-1) = block_w(n_blocks-1)+block_w(n_blocks)
                n_blocks = n_blocks-1
            end do
        end do
        knot_count = n_blocks
        knots(:knot_count) = block_x(:knot_count)
        values(:knot_count) = y(:knot_count)/block_w(:knot_count)
    end subroutine reference_column

    real(dp) function reference_value(knots, values, count, score) result(value)
        real(dp), intent(in) :: knots(:), values(:), score
        integer, intent(in) :: count
        integer :: i
        real(dp) :: fraction

        if (count == 1 .or. score <= knots(1)) then
            value = values(1)
        else if (score >= knots(count)) then
            value = values(count)
        else
            i = 1
            do while (i < count-1)
                if (score <= knots(i+1)) exit
                i = i+1
            end do
            fraction = (score-knots(i))/(knots(i+1)-knots(i))
            value = values(i)+fraction*(values(i+1)-values(i))
        end if
    end function reference_value

end program test_multiclass_isotonic_calibration
