program test_ovo_logistic_partial_fit
    !! Independent oracle for weighted OVO replay, arbitrary labels, rollback,
    !! and the explicit accelerator boundary.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_ovo_logistic_classifier, only: ovo_logistic_classifier_t
    use fortml_classification_state, only: classification_state_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(ovo_logistic_classifier_t) :: incremental, reference, before
    type(classification_state_t) :: metadata
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 2), first_x(4, 2), second_x(5, 2), query(3, 2)
    real(dp) :: p_incremental(3, 3), p_reference(3, 3), p_before(3, 3)
    real(dp) :: first_weight(4), second_weight(5), all_weight(9)
    integer :: labels(9), first_labels(4), second_labels(5), bad_labels(3)
    integer :: classes(3), classes_bad(3), failures

    x = reshape([ &
        -2.0_dp, 2.0_dp, -1.8_dp, 2.1_dp, -2.1_dp, 1.8_dp, &
         2.0_dp, 2.0_dp,  1.8_dp, 2.1_dp,  2.1_dp, 1.8_dp, &
         0.0_dp,-2.0_dp,  0.2_dp,-1.8_dp, -0.2_dp,-2.1_dp], shape(x))
    labels = [42, 42, 42, -7, -7, -7, 10, 10, 10]
    first_x = x(:4, :)
    first_labels = labels(:4)
    second_x = x(5:, :)
    second_labels = labels(5:)
    query = reshape([-2.0_dp, 2.0_dp, 2.0_dp, 2.0_dp, 0.0_dp, -2.0_dp], &
        shape(query))
    first_weight = [0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp]
    second_weight = [0.75_dp, 1.25_dp, 0.8_dp, 1.4_dp, 2.1_dp]
    all_weight = [first_weight, second_weight]
    classes = [-7, 10, 42]
    classes_bad = [-7, 11, 42]
    bad_labels = [42, 999, -7]
    failures = 0

    call reference%fit(x, labels, status, l2=0.1_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp, sample_weight=all_weight, &
        class_weight=[1.5_dp, 0.8_dp, 2.0_dp])
    call check(status_ok(status) .and. reference%fitted(), &
        "weighted one-shot reference fit", failures)

    call incremental%partial_fit(first_x, first_labels, status, classes=classes, &
        l2=0.1_dp, max_iterations=1000, tolerance=1.0e-7_dp, &
        sample_weight=first_weight, class_weight=[1.5_dp, 0.8_dp, 2.0_dp])
    metadata = incremental%metadata()
    call check(status_ok(status) .and. .not. incremental%fitted() .and. &
        metadata%initialized() .and. metadata%class_count() == 3 .and. &
        metadata%sample_count() == 4 .and. metadata%batch_count() == 1 .and. &
        all(metadata%classes() == classes), &
        "deferred weighted first batch metadata", failures)
    call incremental%partial_fit(second_x, second_labels, status, &
        sample_weight=second_weight)
    metadata = incremental%metadata()
    call check(status_ok(status) .and. incremental%fitted() .and. &
        metadata%sample_count() == 9 .and. metadata%batch_count() == 2, &
        "second batch completes every pair", failures)

    call incremental%predict_proba(query, p_incremental, status)
    call reference%predict_proba(query, p_reference, status)
    call check(status_ok(status) .and. maxval(abs(p_incremental - p_reference)) < &
        3.0e-7_dp .and. maxval(abs(sum(p_incremental, dim=2) - 1.0_dp)) < &
        2.0e-14_dp, "weighted replay matches one-shot probability oracle", failures)

    before = incremental
    p_before = p_incremental
    call incremental%partial_fit(query, bad_labels, status)
    call check(.not. status_ok(status), "unknown class is refused", failures)
    call incremental%predict_proba(query, p_incremental, status)
    metadata = incremental%metadata()
    call check(status_ok(status) .and. maxval(abs(p_incremental - p_before)) == 0.0_dp .and. &
        metadata%sample_count() == 9 .and. metadata%batch_count() == 2, &
        "unknown-class rollback preserves predictions and metadata", failures)
    call before%predict_proba(query, p_before, status)
    call check(status_ok(status) .and. maxval(abs(p_before - p_incremental)) == 0.0_dp, &
        "transactional copy remains independent oracle", failures)

    call incremental%partial_fit(query, [42, -7, 10], status, &
        sample_weight=[1.0_dp, -1.0_dp, 1.0_dp])
    call check(.not. status_ok(status), "negative sample weight is refused", failures)
    call incremental%partial_fit(query, [42, -7, 10], status, classes=classes_bad)
    call check(.not. status_ok(status), "changed class vocabulary is refused", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call incremental%predict_proba_device(cuda, query, p_incremental, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. incremental%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA path is an explicit typed refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL OVO partial-fit cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS OVO logistic partial-fit independent oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [ovo-partial] "//name
        end if
    end subroutine check

end program test_ovo_logistic_partial_fit
