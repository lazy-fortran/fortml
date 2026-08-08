program fortml_bench_robust_scaler
    !! Release workload for the dense median/IQR preprocessing transformer.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_preprocessing, only: robust_scaler_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 96, n_features = 5, repetitions = 256
    real(dp) :: x(n_samples, n_features), x_dot(n_samples, n_features)
    real(dp) :: transformed(n_samples, n_features), transformed_dot(n_samples, n_features)
    real(dp) :: elapsed, value_checksum, tangent_checksum
    integer :: i, j, repetition, clock_start, clock_end, clock_rate
    type(robust_scaler_t) :: scaler
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_samples
            x(i, j) = sin(0.031_dp*real(i, dp) + 0.17_dp*real(j, dp)) + &
                0.01_dp*real(mod(i*j, 13), dp)
            x_dot(i, j) = cos(0.023_dp*real(i + 2*j, dp))
        end do
    end do
    call scaler%fit(x, status)
    if (.not. status_ok(status)) error stop "robust scaler fit failed"
    call scaler%transform(x, transformed, status)
    if (.not. status_ok(status)) error stop "robust scaler transform failed"
    call scaler%transform_jvp(x_dot, transformed_dot, status)
    if (.not. status_ok(status)) error stop "robust scaler JVP failed"
    value_checksum = sum(transformed)
    tangent_checksum = sum(transformed_dot)

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call scaler%transform(x, transformed, status)
        if (.not. status_ok(status)) error stop "robust scaler timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/ &
        real(repetitions, dp)
    write (*, '(a,i0,a,i0,a,3(es24.16,a))') "robust_scaler,", n_samples, ",", &
        n_features, ",", elapsed, ",", value_checksum, ",", tangent_checksum, ""
end program fortml_bench_robust_scaler
