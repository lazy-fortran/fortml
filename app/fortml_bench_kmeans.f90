program fortml_bench_kmeans
    !! Release workload for deterministic dense k-means clustering.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_kmeans, only: kmeans_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 240, n_features = 2, n_clusters = 3
    integer, parameter :: repetitions = 8
    real(dp) :: x(n_samples, n_features), distances(n_samples, n_clusters)
    real(dp) :: elapsed_fit, elapsed_transform
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: repetition
    type(kmeans_t) :: model
    type(fortnum_status_t) :: status

    call fixture(x)
    call model%fit(x, status, n_clusters=n_clusters, max_iter=100, tolerance=1.0e-8_dp, &
        initialization_seed=7)
    if (.not. status_ok(status)) error stop "kmeans benchmark fit failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%fit(x, status, n_clusters=n_clusters, max_iter=100, tolerance=1.0e-8_dp, &
            initialization_seed=7)
        if (.not. status_ok(status)) error stop "kmeans benchmark fit timing failed"
    end do
    call system_clock(clock_end)
    elapsed_fit = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%transform(x, distances, status)
        if (.not. status_ok(status)) error stop "kmeans benchmark transform timing failed"
    end do
    call system_clock(clock_end)
    elapsed_transform = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "kmeans_fit_seconds_per_operation,", elapsed_fit
    write (*, '(a,es24.16)') "kmeans_transform_seconds_per_operation,", elapsed_transform
    write (*, '(a,es24.16)') "kmeans_inertia,", model%inertia()

contains

    subroutine fixture(values)
        real(dp), intent(out) :: values(:, :)
        integer :: i
        real(dp) :: offset

        do i = 1, 80
            offset = real(i-1, dp)/80.0_dp
            values(i, :) = [-0.2_dp+0.4_dp*offset, 0.1_dp*sin(offset)]
            values(80+i, :) = [5.0_dp+0.4_dp*offset, 5.1_dp*cos(offset)]
            values(160+i, :) = [10.0_dp-0.4_dp*offset, 0.2_dp+0.1_dp*sin(offset)]
        end do
    end subroutine fixture

end program fortml_bench_kmeans
