program fortml_bench_positive_linear_regression
    !! Correctness-gated release workload for positive linear regression.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_positive_linear_regression, only: positive_linear_regression_t
    implicit none
    integer, parameter :: dp = real64
    integer, parameter :: n_samples = 64, n_features = 3, n_outputs = 2
    integer, parameter :: n_parameters = (n_features+1)*n_outputs
    integer, parameter :: repetitions = 12
    type(positive_linear_regression_t) :: model, vector_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(n_samples, n_features), y(n_samples, n_outputs)
    real(dp) :: weights(n_samples), x_dot(n_samples, n_features)
    real(dp) :: u(n_samples, n_outputs), theta_dot(n_parameters)
    real(dp) :: prediction(n_samples, n_outputs), prediction_dot(n_samples, n_outputs)
    real(dp) :: vector_prediction(n_samples), theta_bar(n_parameters)
    real(dp) :: x_bar(n_samples, n_features), coefficients(n_features+1, n_outputs)
    real(dp) :: elapsed
    real(dp), allocatable :: vector_coefficients(:, :)
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, k

    call make_fixture(x, y, weights, x_dot, u, theta_dot)
    call model%fit(x, y, status, sample_weight=weights, max_iterations=20000, &
        tolerance=1.0e-11_dp)
    if (.not. status_ok(status)) error stop "positive linear benchmark fit failed"
    coefficients = model%coefficients()
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "positive linear benchmark prediction failed"
    call model%jvp(x, theta_dot, x_dot, prediction, prediction_dot, status)
    if (.not. status_ok(status)) error stop "positive linear benchmark JVP failed"
    call model%vjp(x, u, theta_bar, x_bar, status)
    if (.not. status_ok(status)) error stop "positive linear benchmark VJP failed"
    call vector_model%fit(x, y(:, 1), status, sample_weight=weights, &
        max_iterations=20000, tolerance=1.0e-11_dp)
    if (.not. status_ok(status)) error stop "positive linear benchmark vector fit failed"
    vector_coefficients = vector_model%coefficients()
    call vector_model%predict(x, vector_prediction, status)
    if (.not. status_ok(status)) error stop "positive linear benchmark vector prediction failed"

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%fit(x, y, status, sample_weight=weights, max_iterations=20000, &
            tolerance=1.0e-11_dp)
        if (.not. status_ok(status)) error stop "positive linear timed fit failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_matrix("fit_matrix", coefficients, elapsed)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "positive linear timed prediction failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_matrix("predict_matrix", prediction, elapsed)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%jvp(x, theta_dot, x_dot, prediction, prediction_dot, status)
        if (.not. status_ok(status)) error stop "positive linear timed JVP failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_matrix("predict_jvp", prediction_dot, elapsed)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%vjp(x, u, theta_bar, x_bar, status)
        if (.not. status_ok(status)) error stop "positive linear timed VJP failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    call emit_vector("predict_vjp_theta", theta_bar, elapsed)
    call emit_matrix("predict_vjp_x", x_bar, elapsed)
    call emit_matrix("fit_vector", vector_coefficients, elapsed)
    call emit_vector("predict_vector", vector_prediction, elapsed)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop &
        "positive linear CUDA contract changed"
    write (*, '(a)') "positive_linear,cuda_status,unavailable"

contains

    subroutine make_fixture(x, y, weights, x_dot, u, theta_dot)
        real(dp), intent(out) :: x(:, :), y(:, :), weights(:), x_dot(:, :), u(:, :)
        real(dp), intent(out) :: theta_dot(:)
        integer :: row, column, index

        do column = 1, n_features
            do row = 1, n_samples
                x(row, column) = sin(0.071_dp*real(row*column, dp)) + &
                    0.03_dp*cos(0.11_dp*real(row, dp)+real(column, dp))
                x_dot(row, column) = 0.07_dp*cos(0.037_dp*real(row, dp)* &
                    real(column+1, dp))
            end do
        end do
        do row = 1, n_samples
            y(row, 1) = 0.3_dp + 0.9_dp*x(row, 1) + 0.4_dp*x(row, 2) + &
                0.2_dp*x(row, 3) + 0.02_dp*sin(0.13_dp*real(row, dp))
            y(row, 2) = 0.5_dp + 0.3_dp*x(row, 1) + 0.7_dp*x(row, 2) + &
                0.6_dp*x(row, 3) + 0.02_dp*cos(0.09_dp*real(row, dp))
            weights(row) = 0.5_dp + real(mod(row, 5), dp)/5.0_dp
            u(row, 1) = sin(0.041_dp*real(row, dp))
            u(row, 2) = cos(0.053_dp*real(row, dp))
        end do
        do index = 1, n_parameters
            theta_dot(index) = -0.04_dp + 0.08_dp*real(index-1, dp)/ &
                real(n_parameters-1, dp)
        end do
    end subroutine make_fixture

    subroutine emit_vector(name, values, seconds)
        character(*), intent(in) :: name
        real(dp), intent(in) :: values(:), seconds
        integer :: element

        do element = 1, size(values)
            write (*, '(a,a,a,i0,a,es24.16,a,es24.16)') &
                "positive_linear,", trim(name), ",", element, ",", &
                values(element), ",", seconds
        end do
    end subroutine emit_vector

    subroutine emit_matrix(name, values, seconds)
        character(*), intent(in) :: name
        real(dp), intent(in) :: values(:, :), seconds
        integer :: row, column, element

        element = 0
        do column = 1, size(values, 2)
            do row = 1, size(values, 1)
                element = element+1
                write (*, '(a,a,a,i0,a,es24.16,a,es24.16)') &
                    "positive_linear,", trim(name), ",", element, ",", &
                    values(row, column), ",", seconds
            end do
        end do
    end subroutine emit_matrix

end program fortml_bench_positive_linear_regression
