program fortml_bench_chebyshev
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_basis, only: basis_map_t, make_chebyshev_basis
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 4096, n_inputs = 3, degree = 8
    integer, parameter :: repetitions = 16
    real(dp) :: x(n_samples, n_inputs), x_dot(n_samples, n_inputs)
    real(dp), allocatable :: phi(:, :), phi_dot(:, :), u(:, :)
    real(dp), allocatable :: theta_dot(:), theta_bar(:), x_bar(:, :)
    real(dp), allocatable :: theta_hvp(:), x_hvp(:, :)
    real(dp) :: elapsed, checksum
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, repetition, n_features
    type(basis_map_t) :: basis
    type(fortnum_status_t) :: status

    do j = 1, n_inputs
        do i = 1, n_samples
            x(i, j) = 0.8_dp*sin(0.003_dp*real(i, dp) + 0.19_dp*real(j, dp))
            x_dot(i, j) = 0.2_dp*cos(0.005_dp*real(i + 3*j, dp))
        end do
    end do
    basis = make_chebyshev_basis(n_inputs, degree, status, include_intercept=.true.)
    if (.not. status_ok(status)) error stop "Chebyshev benchmark construction failed"
    n_features = basis%feature_count()
    allocate(phi(n_samples, n_features), phi_dot(n_samples, n_features))
    allocate(u(n_samples, n_features), theta_dot(basis%parameter_count()))
    allocate(theta_bar(size(theta_dot)), x_bar(n_samples, n_inputs))
    allocate(theta_hvp(size(theta_dot)), x_hvp(n_samples, n_inputs))
    do j = 1, n_features
        do i = 1, n_samples
            u(i, j) = 0.11_dp*sin(0.009_dp*real(i + j, dp))
        end do
    end do
    theta_dot = 0.0_dp

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call basis%evaluate(x, phi, status)
        if (.not. status_ok(status)) error stop "Chebyshev value failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    checksum = sum(phi)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "chebyshev_value,", n_samples, ",", n_inputs, ",", degree, ",", &
        elapsed/real(repetitions, dp), ",", checksum

    call basis%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
    if (.not. status_ok(status)) error stop "Chebyshev JVP failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call basis%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        if (.not. status_ok(status)) error stop "Chebyshev JVP timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "chebyshev_jvp,", n_samples, ",", n_inputs, ",", degree, ",", &
        elapsed/real(repetitions, dp), ",", sum(phi_dot)

    call basis%vjp(x, u, theta_bar, x_bar, status)
    if (.not. status_ok(status)) error stop "Chebyshev VJP failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call basis%vjp(x, u, theta_bar, x_bar, status)
        if (.not. status_ok(status)) error stop "Chebyshev VJP timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "chebyshev_vjp,", n_samples, ",", n_inputs, ",", degree, ",", &
        elapsed/real(repetitions, dp), ",", sum(theta_bar), ",", sum(x_bar)

    call basis%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
    if (.not. status_ok(status)) error stop "Chebyshev HVP failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call basis%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        if (.not. status_ok(status)) error stop "Chebyshev HVP timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
        "chebyshev_hvp,", n_samples, ",", n_inputs, ",", degree, ",", &
        elapsed/real(repetitions, dp), ",", sum(theta_hvp), ",", sum(x_hvp)
end program fortml_bench_chebyshev
