program fortml_bench_elastic_net
    !! Complete-array release benchmark for weighted elastic-net regression.
    !!
    !! Each output line is
    !!   elastic_net,<workload>,<one-based-index>,<value>,<seconds>
    !! with arrays flattened in Fortran (column-major) order.  The companion
    !! Python harness checks every element against an independent coordinate
    !! descent oracle before retaining timing rows.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_elastic_net_regression, only: elastic_net_regression_t
    implicit none

    integer, parameter :: n_samples = 96
    integer, parameter :: n_features = 6
    integer, parameter :: n_outputs = 3
    integer, parameter :: repetitions = 24
    real(dp), parameter :: alpha = 0.21_dp
    real(dp), parameter :: l1_ratio = 0.43_dp
    real(dp) :: x(n_samples, n_features), y(n_samples, n_outputs)
    real(dp) :: x_dot(n_samples, n_features), u(n_samples, n_outputs)
    real(dp) :: prediction(n_samples, n_outputs), prediction_vector(n_samples)
    real(dp) :: prediction_dot(n_samples, n_outputs)
    real(dp) :: theta_dot((n_features+1)*n_outputs), theta_bar((n_features+1)*n_outputs)
    real(dp) :: x_bar(n_samples, n_features), coefficients(n_features+1, n_outputs)
    real(dp) :: target_vector(n_samples), vector_prediction(n_samples)
    real(dp) :: elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, k, index
    type(elastic_net_regression_t) :: model, vector_model
    type(fortnum_status_t) :: status

    call make_fixture(x, y, x_dot, u, theta_dot)
    target_vector = y(:, 1)

    call model%fit(x, y, status, alpha=alpha, l1_ratio=l1_ratio, &
        sample_weight=weights_fixture(), max_iterations=5000, tolerance=1.0e-12_dp)
    if (.not. status_ok(status)) error stop "elastic-net benchmark fit failed"
    coefficients = model%coefficients()
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "elastic-net benchmark prediction failed"
    call model%jvp(x, theta_dot, x_dot, prediction, prediction_dot, status)
    if (.not. status_ok(status)) error stop "elastic-net benchmark JVP failed"
    call model%vjp(x, u, theta_bar, x_bar, status)
    if (.not. status_ok(status)) error stop "elastic-net benchmark VJP failed"

    call vector_model%fit(x, target_vector, status, alpha=alpha, l1_ratio=l1_ratio, &
        sample_weight=weights_fixture(), max_iterations=5000, tolerance=1.0e-12_dp)
    if (.not. status_ok(status)) error stop "elastic-net benchmark vector fit failed"
    call vector_model%predict(x, vector_prediction, status)
    if (.not. status_ok(status)) error stop "elastic-net benchmark vector prediction failed"

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%fit(x, y, status, alpha=alpha, l1_ratio=l1_ratio, &
            sample_weight=weights_fixture(), max_iterations=5000, tolerance=1.0e-12_dp)
        if (.not. status_ok(status)) error stop "elastic-net timed matrix fit failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_matrix("fit_matrix", coefficients, elapsed)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call vector_model%fit(x, target_vector, status, alpha=alpha, l1_ratio=l1_ratio, &
            sample_weight=weights_fixture(), max_iterations=5000, tolerance=1.0e-12_dp)
        if (.not. status_ok(status)) error stop "elastic-net timed vector fit failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_matrix("fit_vector", vector_model%coefficients(), elapsed)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "elastic-net timed matrix prediction failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_matrix("predict_matrix", prediction, elapsed)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call vector_model%predict(x, vector_prediction, status)
        if (.not. status_ok(status)) error stop "elastic-net timed vector prediction failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_vector("predict_vector", vector_prediction, elapsed)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%jvp(x, theta_dot, x_dot, prediction, prediction_dot, status)
        if (.not. status_ok(status)) error stop "elastic-net timed JVP failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_matrix("predict_jvp", prediction_dot, elapsed)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%vjp(x, u, theta_bar, x_bar, status)
        if (.not. status_ok(status)) error stop "elastic-net timed VJP failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_vector("predict_vjp_theta", theta_bar, elapsed)
    call emit_matrix("predict_vjp_x", x_bar, elapsed)

contains

    function weights_fixture() result(weights)
        real(dp) :: weights(n_samples)
        integer :: row

        do row = 1, n_samples
            weights(row) = 0.4_dp + 0.06_dp*(1.0_dp + sin(0.17_dp*real(row, dp)))
        end do
    end function weights_fixture

    subroutine make_fixture(x, y, x_dot, u, theta_dot)
        real(dp), intent(out) :: x(:, :), y(:, :), x_dot(:, :), u(:, :), theta_dot(:)
        real(dp) :: latent(n_samples, n_outputs)
        integer :: row, column, output

        do column = 1, n_features
            do row = 1, n_samples
                x(row, column) = sin(0.071_dp*real(row*column, dp)) + &
                    0.03_dp*cos(0.11_dp*real(row, dp) + real(column, dp))
                x_dot(row, column) = 0.07_dp*cos(0.037_dp*real(row, dp)* &
                    real(column+1, dp))
            end do
        end do
        do row = 1, n_samples
            latent(row, 1) = 0.7_dp + 0.8_dp*x(row, 1) - 0.2_dp*x(row, 2) + 0.1_dp*x(row, 3)
            latent(row, 2) = -0.3_dp + 0.4_dp*x(row, 3) + 0.9_dp*x(row, 4) - 0.25_dp*x(row, 5)
            latent(row, 3) = 0.2_dp - 0.5_dp*x(row, 2) + 0.3_dp*x(row, 5) + 0.6_dp*x(row, 6)
            y(row, 1) = latent(row, 1) + 0.02_dp*sin(0.13_dp*real(row, dp))
            y(row, 2) = latent(row, 2) + 0.02_dp*cos(0.09_dp*real(row, dp))
            y(row, 3) = latent(row, 3) + 0.02_dp*sin(0.05_dp*real(row, dp) + 0.3_dp)
            u(row, 1) = sin(0.041_dp*real(row, dp))
            u(row, 2) = cos(0.053_dp*real(row, dp))
            u(row, 3) = sin(0.067_dp*real(row, dp) + 0.2_dp)
        end do
        do output = 1, n_outputs
            do column = 1, n_features+1
                index = column + (output-1)*(n_features+1)
                theta_dot(index) = -0.17_dp + 0.40_dp*real(index-1, dp)/ &
                    real((n_features+1)*n_outputs-1, dp)
            end do
        end do
    end subroutine make_fixture

    subroutine emit_vector(name, values, seconds)
        character(len=*), intent(in) :: name
        real(dp), intent(in) :: values(:)
        real(dp), intent(in) :: seconds
        integer :: element

        do element = 1, size(values)
            write (*, '(a,a,a,i0,a,es24.16,a,es24.16)') &
                "elastic_net,", trim(name), ",", element, ",", &
                values(element), ",", seconds
        end do
    end subroutine emit_vector

    subroutine emit_matrix(name, values, seconds)
        character(len=*), intent(in) :: name
        real(dp), intent(in) :: values(:, :)
        real(dp), intent(in) :: seconds
        integer :: row, column, element

        element = 0
        do column = 1, size(values, 2)
            do row = 1, size(values, 1)
                element = element+1
                write (*, '(a,a,a,i0,a,es24.16,a,es24.16)') &
                    "elastic_net,", trim(name), ",", element, ",", &
                    values(row, column), ",", seconds
            end do
        end do
    end subroutine emit_matrix

end program fortml_bench_elastic_net
