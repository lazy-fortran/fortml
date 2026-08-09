program fortml_bench_quantile_transformer
    !! Release workload for the uniform empirical quantile transformer.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_preprocessing, only: quantile_transformer_t
    implicit none

    type(quantile_transformer_t) :: transformer
    type(fortnum_status_t) :: status
    real(dp), allocatable :: x(:, :), query(:, :), transformed(:, :), restored(:, :)
    integer(int64) :: tick_start, tick_end, ticks_per_second
    real(dp) :: elapsed, checksum, roundtrip_error
    integer :: i

    allocate(x(128, 2), query(64, 2), transformed(64, 2), restored(64, 2))
    do i = 1, 128
        x(i, 1) = real(i - 1, dp)/127.0_dp
        x(i, 2) = 10.0_dp*x(i, 1) + 0.25_dp*sin(real(i, dp))
    end do
    do i = 1, 64
        query(i, 1) = (real(i, dp) - 0.5_dp)/64.0_dp
        query(i, 2) = 10.0_dp*query(i, 1) + 0.25_dp*sin(real(i, dp) + 0.5_dp)
    end do

    call transformer%fit(x, status, n_quantiles=64)
    if (.not. status_ok(status)) error stop "quantile transformer benchmark fit failed"
    call system_clock(tick_start, ticks_per_second)
    call transformer%transform(query, transformed, status)
    call transformer%inverse_transform(transformed, restored, status)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop "quantile transformer benchmark product failed"
    elapsed = real(tick_end - tick_start, dp)/real(ticks_per_second, dp)
    checksum = sum(transformed)
    roundtrip_error = maxval(abs(restored - query))
    write (*, '(a,",pass,checksum,",es24.16,",",es24.16)') &
        "quantile_transformer", checksum, elapsed
    write (*, '(a,",pass,roundtrip_max_abs_error,",es24.16,",",es24.16)') &
        "quantile_transformer", roundtrip_error, elapsed
    write (*, '(a,",unavailable,typed_uniform_only,",i0)') &
        "quantile_transformer,cuda", 3
end program fortml_bench_quantile_transformer
