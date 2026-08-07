program test_cuda_forest_api
    !! Ordinary-build contract test for the Fortran resident-forest wrapper.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_cuda_forest_api, only: cuda_forest_plan_t
    implicit none

    type(cuda_forest_plan_t) :: plan
    type(fortnum_status_t) :: status
    integer :: tree_offset(3), node_feature(6), node_left(6), node_right(6), &
        class_label(2), labels(2), failures
    real(real64) :: node_threshold(6), node_probability(12), query_x(2, 1), &
        probabilities(2, 2)

    failures = 0
    tree_offset = [0, 3, 6]
    node_feature = [0, -1, -1, 0, -1, -1]
    node_left = [1, -1, -1, 4, -1, -1]
    node_right = [2, -1, -1, 5, -1, -1]
    node_threshold = [0.0_real64, 0.0_real64, 0.0_real64, 1.0_real64, &
        0.0_real64, 0.0_real64]
    node_probability = [ &
        1.0_real64, 0.0_real64, 0.0_real64, 1.0_real64, 0.0_real64, 1.0_real64, &
        1.0_real64, 0.0_real64, 0.0_real64, 1.0_real64, 0.0_real64, 1.0_real64]
    class_label = [-7, 11]
    query_x = reshape([-1.0_real64, 2.0_real64], shape(query_x))
    probabilities = -31.0_real64
    labels = -17

    call plan%create(tree_offset, node_feature, node_left, node_right, &
        node_threshold, node_probability, class_label, -1, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. .not. plan%fitted(), &
        "invalid device index is rejected", failures)
    call plan%create(tree_offset, node_feature, node_left, node_right, &
        node_threshold, node_probability, class_label, 0, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. .not. plan%fitted(), &
        "ordinary build retains typed native-unavailable refusal", failures)
    call plan%predict_proba(query_x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(probabilities == -31.0_real64), &
        "probability refusal preserves output", failures)
    call plan%predict(query_x, labels, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(labels == -17), &
        "label refusal preserves output", failures)
    call plan%destroy(status)
    call check(.not. plan%fitted(), "destroy clears wrapper state", failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL CUDA forest API cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS CUDA forest Fortran API refusal oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [CUDA forest API] "//description
        end if
    end subroutine check

end program test_cuda_forest_api
