program test_ranking_metrics
    !! Independent hand oracle for grouped NDCG and device boundaries.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_ranking_metrics, only: ranking_ndcg, ranking_ndcg_device
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: relevance(6), scores(6), value, expected
    real(dp) :: weights(6)
    integer :: group(6), failures

    relevance = [3.0_dp, 2.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    scores = [0.8_dp, 0.4_dp, 0.1_dp, 0.9_dp, 0.2_dp, 0.7_dp]
    group = [11, 11, 11, 42, 42, 42]
    weights = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
    expected = 0.5_dp*(1.0_dp + (3.0_dp/(log(3.0_dp)/log(2.0_dp)) + &
        1.0_dp/(log(4.0_dp)/log(2.0_dp)))/(3.0_dp + &
        1.0_dp/(log(3.0_dp)/log(2.0_dp))))
    failures = 0

    call ranking_ndcg(relevance, scores, group, value, status)
    call check(status_ok(status) .and. abs(value - expected) < 2.0e-14_dp, &
        "macro NDCG hand oracle", failures)
    call ranking_ndcg(relevance, scores, group, value, status, k=1)
    call check(status_ok(status) .and. abs(value - 0.5_dp) < 2.0e-14_dp, &
        "per-query cutoff", failures)
    weights = [2.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 3.0_dp, 1.0_dp]
    call ranking_ndcg(relevance, scores, group, value, status, &
        sample_weight=weights)
    call check(status_ok(status) .and. value >= 0.0_dp .and. value <= 1.0_dp, &
        "weighted bounded NDCG", failures)

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call ranking_ndcg_device(cpu, relevance, scores, group, value, status)
    call check(status_ok(status), "CPU device dispatch", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call ranking_ndcg_device(cuda, relevance, scores, group, value, status)
    call check(.not. status_ok(status), "typed CUDA refusal", failures)
    relevance = 0.0_dp
    call ranking_ndcg(relevance, scores, group, value, status)
    call check(.not. status_ok(status), "zero-ideal refusal", failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL ranking metric cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS grouped ranking metric independent oracle"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [ranking-metric] "//name
        end if
    end subroutine check

end program test_ranking_metrics
