program fortml_bench_mlp
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 512
    integer, parameter :: n_features = 16
    integer, parameter :: n_hidden = 32
    integer, parameter :: n_outputs = 4
    integer, parameter :: repetitions = 32
    integer, parameter :: n_parameters = n_features*n_hidden + n_hidden + &
        n_hidden*n_outputs + n_outputs
    real(dp) :: x(n_samples, n_features), target(n_samples, n_outputs)
    real(dp) :: prediction(n_samples, n_outputs), u(n_samples, n_outputs)
    real(dp) :: theta(n_parameters), direction(n_parameters)
    real(dp) :: parameter_bar(n_parameters), x_bar(n_samples, n_features)
    real(dp) :: y_plus(n_samples, n_outputs), y_minus(n_samples, n_outputs)
    real(dp) :: dx(n_samples, n_features), dy(n_samples, n_outputs)
    real(dp) :: residual, adjoint_error, finite_difference_error, elapsed
    real(dp) :: h
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, k
    type(mlp_t) :: model
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_samples
            x(i, j) = sin(0.013_dp*real(i, dp) + 0.071_dp*real(j, dp)) + &
                cos(0.009_dp*real(i*j, dp))
            dx(i, j) = cos(0.017_dp*real(i + j, dp))
        end do
    end do
    do i = 1, n_parameters
        theta(i) = 0.08_dp*sin(0.37_dp*real(i, dp))
        direction(i) = 0.03_dp*cos(0.23_dp*real(i, dp))
    end do
    do k = 1, n_outputs
        do i = 1, n_samples
            u(i, k) = 0.2_dp*sin(0.011_dp*real(i*k, dp))
        end do
    end do

    call model%initialize([n_features, n_hidden, n_outputs], status)
    call model%set_parameters(theta, status)
    call reference_predict(theta, x, target)
    call model%predict(x, prediction, status)
    residual = maxval(abs(prediction - target))
    if (.not. status_ok(status) .or. residual > 2.0e-13_dp) then
        error stop "MLP benchmark forward oracle failed"
    end if

    call model%jvp(x, direction, dx, prediction, dy, status)
    h = 1.0e-6_dp
    call reference_predict(theta + h*direction, x + h*dx, y_plus)
    call reference_predict(theta - h*direction, x - h*dx, y_minus)
    finite_difference_error = maxval(abs(dy - (y_plus - y_minus)/(2.0_dp*h)))
    call model%vjp(x, u, parameter_bar, x_bar, status)
    adjoint_error = abs(sum(u*dy) - sum(parameter_bar*direction) - sum(x_bar*dx))
    if (.not. status_ok(status) .or. finite_difference_error > 3.0e-9_dp .or. &
        adjoint_error > 2.0e-10_dp) then
        error stop "MLP benchmark product oracle failed"
    end if

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%predict(x, prediction, status)
        call model%vjp(x, u, parameter_bar, x_bar, status)
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16)') &
        "mlp,", n_samples, ",", n_features, ",", n_outputs, ",", &
        repetitions, ",", elapsed/real(repetitions, dp)

contains

    subroutine reference_predict(parameters, input, output)
        real(dp), intent(in) :: parameters(:), input(:, :)
        real(dp), intent(out) :: output(:, :)
        real(dp) :: weight_1(n_features, n_hidden), bias_1(n_hidden)
        real(dp) :: weight_2(n_hidden, n_outputs), bias_2(n_outputs)
        real(dp) :: hidden(n_hidden), preactivation(n_hidden)
        integer :: i, j, position

        position = 1
        weight_1 = reshape(parameters(position:position + n_features*n_hidden - 1), &
            shape(weight_1))
        position = position + n_features*n_hidden
        bias_1 = parameters(position:position + n_hidden - 1)
        position = position + n_hidden
        weight_2 = reshape(parameters(position:position + n_hidden*n_outputs - 1), &
            shape(weight_2))
        position = position + n_hidden*n_outputs
        bias_2 = parameters(position:position + n_outputs - 1)

        do i = 1, size(input, 1)
            do j = 1, n_hidden
                preactivation(j) = bias_1(j) + sum(input(i, :)*weight_1(:, j))
                hidden(j) = tanh(preactivation(j))
            end do
            do j = 1, n_outputs
                output(i, j) = bias_2(j) + sum(hidden*weight_2(:, j))
            end do
        end do
    end subroutine reference_predict

end program fortml_bench_mlp
