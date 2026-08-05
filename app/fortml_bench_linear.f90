program fortml_bench_linear
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_linear_regression, only: linear_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 512
    integer, parameter :: n_features = 16
    integer, parameter :: n_outputs = 4
    integer, parameter :: repetitions = 8
    real(dp) :: x(n_samples, n_features), y(n_samples, n_outputs)
    real(dp) :: prediction(n_samples, n_outputs)
    real(dp) :: coefficients(n_features + 1, n_outputs)
    real(dp) :: residual, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, k
    type(linear_regression_t) :: model
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_samples
            x(i, j) = sin(0.013_dp*real(i, dp) + 0.071_dp*real(j, dp)) + &
                cos(0.009_dp*real(i*j, dp))
        end do
    end do
    do k = 1, n_outputs
        coefficients(:, k) = 0.05_dp*real(k, dp)
        do j = 1, n_features
            coefficients(j + 1, k) = sin(0.17_dp*real(j*k, dp))
        end do
    end do
    y = spread(coefficients(1, :), dim=1, ncopies=n_samples)
    do j = 1, n_features
        y = y + spread(x(:, j), dim=2, ncopies=n_outputs)* &
            spread(coefficients(j + 1, :), dim=1, ncopies=n_samples)
    end do

    call model%fit(x, y, status)
    if (.not. status_ok(status)) error stop "benchmark fit correctness setup failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "benchmark predict correctness setup failed"
    residual = maxval(abs(prediction - y))
    if (residual > 1.0e-9_dp) error stop "benchmark correctness oracle failed"

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%fit(x, y, status, ridge=1.0e-10_dp)
        call model%predict(x, prediction, status)
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16)') &
        "linear_regression,", n_samples, ",", n_features, ",", n_outputs, ",", &
        repetitions, ",", elapsed/real(repetitions, dp)

end program fortml_bench_linear
