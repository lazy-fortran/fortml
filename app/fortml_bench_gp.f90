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
    real(dp), parameter :: signal_variance = 1.4_dp
    real(dp), parameter :: lengthscale = 0.9_dp
    real(dp), parameter :: noise_variance = 0.08_dp
    real(dp), parameter :: jitter = 1.0e-10_dp
    real(dp) :: x(n_samples, n_features), y(n_samples, n_outputs)
    real(dp) :: x_test(n_test, n_features), prediction(n_test, n_outputs)
    real(dp) :: variance(n_test), reference_mean(n_test, n_outputs)
    real(dp) :: reference_variance(n_test)
    real(dp) :: covariance(n_samples, n_samples), cross(n_samples, n_test)
    real(dp) :: reference_rhs(n_samples, n_outputs), reference_alpha(n_samples, n_outputs)
    real(dp) :: reference_cross(n_samples, n_test)
    real(dp) :: residual, variance_residual, elapsed
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

    call system_clock(clock_start, clock_rate)
    do k = 1, repetitions
        call model%fit(x, y, kernel, noise_variance, status, jitter=jitter)
        call model%predict(x_test, prediction, variance, status)
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
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

end program fortml_bench_gp
