program fortml_bench_linear_conditioning
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_linear_regression, only: linear_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 16
    integer, parameter :: n_features = 3
    integer, parameter :: n_outputs = 1
    integer, parameter :: n_cases = 5
    integer, parameter :: repetitions = 32
    real(dp), parameter :: epsilons(n_cases) = [ &
        1.0_dp, 1.0e-4_dp, 1.0e-8_dp, 1.0e-12_dp, 1.0e-14_dp]
    real(dp), parameter :: true_coef(4) = [1.25_dp, 2.0_dp, -3.0_dp, 0.75_dp]
    real(dp) :: x(n_samples, n_features), y(n_samples, n_outputs)
    real(dp) :: prediction(n_samples, n_outputs)
    real(dp) :: elapsed, residual
    real(dp) :: epsilon
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: case_number, i, repetition
    type(linear_regression_t) :: model
    type(fortnum_status_t) :: status

    do case_number = 1, n_cases
        epsilon = epsilons(case_number)
        call make_case(epsilon, x, y)
        call model%fit(x, y, status)
        if (.not. status_ok(status)) error stop "conditioning setup fit failed"

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            call model%fit(x, y, status)
            if (.not. status_ok(status)) error stop "conditioning timed fit failed"
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp) / &
            real(clock_rate, dp) / real(repetitions, dp)

        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "conditioning prediction failed"
        residual = maxval(abs(prediction - y))
        write (*, '(a,",",es24.16,",",es24.16,",",es24.16,",",es24.16,",",es24.16,",",es24.16)') &
            "linear_conditioning", epsilon, elapsed, model%coef(:, 1)
        write (*, '(a,",",es24.16)') "linear_conditioning_residual", residual
    end do

contains

    subroutine make_case(epsilon, x, y)
        real(dp), intent(in) :: epsilon
        real(dp), intent(out) :: x(:, :), y(:, :)
        real(dp) :: t, alternating
        integer :: i

        do i = 1, size(x, 1)
            t = real(i, dp) - 8.5_dp
            alternating = 1.0_dp
            if (mod(i, 2) /= 0) alternating = -1.0_dp
            x(i, 1) = t
            x(i, 2) = t + epsilon*alternating
            x(i, 3) = t*t
            y(i, 1) = true_coef(1) + true_coef(2)*x(i, 1) + &
                true_coef(3)*x(i, 2) + true_coef(4)*x(i, 3)
        end do
    end subroutine make_case

end program fortml_bench_linear_conditioning
