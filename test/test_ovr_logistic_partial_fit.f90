program test_ovr_logistic_partial_fit
    !! Independent oracle for sorted labels, replayable partial fitting, and rollback.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_ovr_logistic_classifier, only: ovr_logistic_classifier_t
    use fortml_classification_state, only: classification_state_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(ovr_logistic_classifier_t) :: incremental, replay, reference, before
    type(classification_state_t) :: metadata
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 2), first_x(4, 2), second_x(5, 2), query(3, 2)
    real(dp) :: p_incremental(3, 3), p_reference(3, 3), p_before(3, 3)
    real(dp) :: x_dot(3, 2), p_dot(3, 3), p_plus(3, 3), p_minus(3, 3)
    integer :: labels(9), first_labels(4), second_labels(5), query_labels(3)
    integer :: classes(3), classes_bad(3), failures
    real(dp) :: error_value

    x = reshape([ &
        -2.0_dp, 2.0_dp, -1.8_dp, 2.1_dp, -2.1_dp, 1.8_dp, &
        2.0_dp, 2.0_dp, 1.8_dp, 2.1_dp, 2.1_dp, 1.8_dp, &
        0.0_dp, -2.0_dp, 0.2_dp, -1.8_dp, -0.2_dp, -2.1_dp], shape(x))
    labels = [42, 42, 42, -7, -7, -7, 10, 10, 10]
    first_x = x(:4, :)
    first_labels = labels(:4)
    second_x = x(5:, :)
    second_labels = labels(5:)
    query = reshape([-2.0_dp, 2.0_dp, 2.0_dp, 2.0_dp, 0.0_dp, -2.0_dp], &
        shape(query))
    x_dot = reshape([0.1_dp, -0.2_dp, -0.2_dp, 0.1_dp, 0.2_dp, 0.1_dp], &
        shape(x_dot))
    classes = [-7, 10, 42]
    classes_bad = [-7, 11, 42]
    failures = 0

    call reference%fit(x, labels, status, l2=0.1_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp)
    call check(status_ok(status), "reference fit", failures)
    call incremental%partial_fit(first_x, first_labels, status, classes=classes, &
        l2=0.1_dp, max_iterations=1000, tolerance=1.0e-7_dp)
    metadata = incremental%metadata()
    call check(status_ok(status) .and. .not. incremental%fitted() .and. &
        metadata%initialized() .and. metadata%class_count() == 3 .and. &
        metadata%sample_count() == 4 .and. metadata%batch_count() == 1, &
        "metadata and deferred first batch", failures)
    call incremental%partial_fit(second_x, second_labels, status)
    metadata = incremental%metadata()
    call check(status_ok(status) .and. incremental%fitted() .and. &
        metadata%sample_count() == 9 .and. metadata%batch_count() == 2, &
        "second batch completes all classes", failures)

    call incremental%predict_proba(query, p_incremental, status)
    call reference%predict_proba(query, p_reference, status)
    error_value = maxval(abs(p_incremental - p_reference))
    call check(status_ok(status) .and. error_value < 3.0e-7_dp, &
        "deterministic replay matches one-shot fit", failures)

    before = incremental
    p_before = p_incremental
    query_labels = [42, 999, -7]
    call incremental%partial_fit(query, query_labels, status)
    call check(.not. status_ok(status), "unknown class is refused", failures)
    call incremental%predict_proba(query, p_incremental, status)
    metadata = incremental%metadata()
    call check(status_ok(status) .and. maxval(abs(p_incremental - p_before)) == 0.0_dp .and. &
        metadata%sample_count() == 9 .and. metadata%batch_count() == 2, &
        "malformed batch leaves model and metadata unchanged", failures)
    call before%predict_proba(query, p_before, status)
    call check(status_ok(status) .and. maxval(abs(p_before - p_incremental)) == 0.0_dp, &
        "transactional copy oracle", failures)

    call incremental%predict_proba_jvp(query, x_dot, p_incremental, p_dot, status)
    call incremental%predict_proba(query + 1.0e-5_dp*x_dot, p_plus, status)
    call incremental%predict_proba(query - 1.0e-5_dp*x_dot, p_minus, status)
    call check(status_ok(status) .and. maxval(abs(p_dot - &
        (p_plus - p_minus)/(2.0e-5_dp))) < 3.0e-6_dp, &
        "fixed-state input JVP survives partial fit", failures)

    replay = reference
    call replay%partial_fit(first_x, first_labels, status, classes=classes_bad)
    call check(.not. status_ok(status), "replay refuses changed class metadata", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL OVR partial-fit cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS OVR logistic partial-fit independent oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [ovr-partial] "//name
        end if
    end subroutine check

end program test_ovr_logistic_partial_fit
