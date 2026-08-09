program fortml_bench_ranking_metrics
    !! Release workload for grouped NDCG and the explicit CUDA boundary.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_ranking_metrics, only: ranking_ndcg, ranking_ndcg_device
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    real(dp) :: relevance(6), scores(6), value, cpu_value, expected, elapsed
    integer :: group(6)
    integer(int64) :: tick_start, tick_end, ticks_per_second
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    relevance = [3.0_dp, 2.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
    scores = [0.8_dp, 0.4_dp, 0.1_dp, 0.9_dp, 0.2_dp, 0.7_dp]
    group = [11, 11, 11, 42, 42, 42]
    expected = 0.5_dp*(1.0_dp + (3.0_dp/(log(3.0_dp)/log(2.0_dp)) + &
        1.0_dp/(log(4.0_dp)/log(2.0_dp)))/(3.0_dp + &
        1.0_dp/(log(3.0_dp)/log(2.0_dp))))
    call system_clock(tick_start, ticks_per_second)
    call ranking_ndcg(relevance, scores, group, value, status)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop "ranking NDCG failed"
    cpu_value = value
    elapsed = real(tick_end - tick_start, dp)/real(ticks_per_second, dp)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call ranking_ndcg_device(cuda, relevance, scores, group, value, status)
    write (*, '(a,",",es24.16,",",es24.16,",",es24.16,",",i0)') &
        "ranking_ndcg", cpu_value, abs(cpu_value - expected), elapsed, status%code
end program fortml_bench_ranking_metrics
