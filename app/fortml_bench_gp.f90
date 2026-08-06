program fortml_bench_gp
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortnum_linalg, only: dense_solve
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 128
    integer, parameter :: n_features = 4
    integer, parameter :: n_outputs = 2
    integer, parameter :: n_test = 32
    integer, parameter :: repetitions = 4
    integer, parameter :: fit_repetitions = 8
    integer, parameter :: predict_repetitions = 32
    real(dp), parameter :: signal_variance = 1.4_dp
    real(dp), parameter :: lengthscale = 0.9_dp
    real(dp), parameter :: noise_variance = 0.08_dp
    real(dp), parameter :: jitter = 1.0e-10_dp
    real(dp) :: x(n_samples, n_features), y(n_samples, n_outputs)
    real(dp) :: x_test(n_test, n_features), prediction(n_test, n_outputs)
    real(dp) :: variance(n_test), reference_mean(n_test, n_outputs)
    real(dp) :: reference_variance(n_test)
    real(dp) :: covariance(n_samples, n_samples), cross(n_samples, n_test)
    real(dp) :: reference_rhs(n_samples, n_outputs)
    real(dp) :: reference_alpha(n_samples, n_outputs)
    real(dp) :: reference_cross(n_samples, n_test)
    real(dp) :: residual, variance_residual, elapsed, fit_elapsed, predict_elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, k, info
    type(kernel_t) :: kernel
    type(gp_regression_t) :: model
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_samples
            x(i, j) = sin(0.013_dp*real(i + 3*j, dp)) + &
                0.1_dp*cos(0.017_dp*real(i*j, dp))
        end do
        do i = 1, n_test
            x_test(i, j) = sin(0.019_dp*real(i + 2*j, dp)) + &
                0.1_dp*cos(0.011_dp*real(i*j, dp))
        end do
    end do
    do k = 1, n_outputs
        do i = 1, n_samples
            y(i, k) = sin(0.021_dp*real(i*k, dp)) + &
                0.3_dp*cos(0.007_dp*real(i + 2*k, dp))
        end do
    end do

    kernel = make_rbf_kernel(n_features, signal_variance, lengthscale, status)
    call model%fit(x, y, kernel, noise_variance, status, jitter=jitter)
    call model%predict(x_test, prediction, variance, status)
    call reference_gp(x, y, x_test, reference_mean, reference_variance)
    residual = maxval(abs(prediction - reference_mean))
    variance_residual = maxval(abs(variance - reference_variance))
    if (.not. status_ok(status) .or. residual > 2.0e-10_dp .or. &
        variance_residual > 2.0e-10_dp) then
        error stop "GP benchmark correctness oracle failed"
    end if
    call write_oracle(prediction, variance)
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do k = 1, fit_repetitions
        call model%fit(x, y, kernel, noise_variance, status, jitter=jitter)
    end do
    call system_clock(clock_end)
    fit_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)

    call system_clock(clock_start, clock_rate)
    do k = 1, predict_repetitions
        call model%predict(x_test, prediction, variance, status)
    end do
    call system_clock(clock_end)
    predict_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%fit(x, y, kernel, noise_variance, status, jitter=jitter)
        call model%predict(x_test, prediction, variance, status)
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16)') &
        "gp_fit,", n_samples, ",", n_features, ",", n_outputs, ",", &
        fit_repetitions, ",", fit_elapsed/real(fit_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16)') &
        "gp_predict,", n_samples, ",", n_features, ",", n_outputs, ",", &
        predict_repetitions, ",", predict_elapsed/real(predict_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16)') &
        "gp,", n_samples, ",", n_features, ",", n_outputs, ",", &
        repetitions, ",", elapsed/real(repetitions, dp)

contains

    subroutine reference_gp(input, targets, test_input, mean, latent_variance)
        real(dp), intent(in) :: input(:, :), targets(:, :), test_input(:, :)
        real(dp), intent(out) :: mean(:, :), latent_variance(:)
        integer :: row, test_row
        real(dp) :: squared_distance, prior

        do row = 1, size(input, 1)
            do test_row = 1, size(test_input, 1)
                squared_distance = sum((input(row, :) - test_input(test_row, :))**2)
                cross(row, test_row) = signal_variance*exp(-0.5_dp* &
                    squared_distance/(lengthscale*lengthscale))
            end do
        end do
        do row = 1, size(input, 1)
            do test_row = 1, size(input, 1)
                squared_distance = sum((input(row, :) - input(test_row, :))**2)
                covariance(row, test_row) = signal_variance*exp(-0.5_dp* &
                    squared_distance/(lengthscale*lengthscale))
            end do
            covariance(row, row) = covariance(row, row) + noise_variance + jitter
        end do
        reference_rhs = targets
        call dense_solve(covariance, reference_rhs, reference_alpha, info)
        call dense_solve(covariance, cross, reference_cross, info)
        mean = matmul(transpose(cross), reference_alpha)
        prior = signal_variance
        latent_variance = prior - sum(cross*reference_cross, dim=1)
    end subroutine reference_gp

    subroutine write_oracle(mean, latent_variance)
        real(dp), intent(in) :: mean(:, :), latent_variance(:)
        character(len=1024) :: path
        integer :: column, row, unit, environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE", path, &
            status=environment_status)
        if (environment_status /= 0 .or. len_trim(path) == 0) return

        open (newunit=unit, file=trim(path), status="replace", action="write")
        write (unit, '(a)') "quantity,row,column,value"
        do column = 1, size(mean, 2)
            do row = 1, size(mean, 1)
                write (unit, '(a,i0,a,i0,a,es26.17e3)') &
                    "mean,", row, ",", column, ",", mean(row, column)
            end do
        end do
        do row = 1, size(latent_variance)
            write (unit, '(a,i0,a,es26.17e3)') &
                "variance,", row, ",1,", latent_variance(row)
        end do
        close (unit)
    end subroutine write_oracle

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=environment_status)
        oracle_only_requested = environment_status == 0 .and. &
            trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_gp
