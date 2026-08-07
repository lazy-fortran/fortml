program fortml_bench_glm_regression
    !! Release workload for weighted Poisson and Gamma log-link GLMs.
    !!
    !! The fixture is deterministic and the objective values are reported so
    !! an external NumPy harness can verify the same weighted deviance.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_glm_regression, only: glm_regression_t, GLM_FAMILY_POISSON, &
        GLM_FAMILY_GAMMA
    implicit none

    integer, parameter :: n_samples = 256, n_features = 3
    real(dp) :: x(n_samples, n_features), poisson_target(n_samples)
    real(dp) :: gamma_target(n_samples), weights(n_samples)
    type(glm_regression_t) :: model

    call make_fixture(x, poisson_target, gamma_target, weights)
    call benchmark_family(model, x, poisson_target, weights, GLM_FAMILY_POISSON)
    call benchmark_family(model, x, gamma_target, weights, GLM_FAMILY_GAMMA)

contains

    subroutine make_fixture(x, poisson_target, gamma_target, weights)
        real(dp), intent(out) :: x(:, :), poisson_target(:), gamma_target(:), weights(:)
        integer :: i
        real(dp) :: eta
        do i = 1, size(x, 1)
            x(i, 1) = -2.0_dp + 4.0_dp*real(i-1, dp)/real(size(x, 1)-1, dp)
            x(i, 2) = sin(0.17_dp*real(i, dp))
            x(i, 3) = cos(0.11_dp*real(i, dp))
            eta = 0.25_dp + 0.55_dp*x(i, 1) - 0.2_dp*x(i, 2) + 0.15_dp*x(i, 3)
            poisson_target(i) = exp(eta)
            gamma_target(i) = exp(eta + 0.08_dp*sin(0.07_dp*real(i, dp)))
            weights(i) = 0.75_dp + real(mod(i, 5), dp)/5.0_dp
        end do
    end subroutine make_fixture

    subroutine benchmark_family(model, x, target, weights, family)
        type(glm_regression_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:), weights(:)
        integer, intent(in) :: family
        real(dp) :: prediction(size(target)), value
        real(dp), allocatable :: theta(:), gradient(:)
        real(dp) :: elapsed
        integer(int64) :: start_tick, end_tick, clock_rate
        type(fortnum_status_t) :: status
        integer :: repetition, repeats

        repeats = 4
        call system_clock(start_tick, clock_rate)
        do repetition = 1, repeats
            call model%fit(x, target, status, family=family, alpha=0.05_dp, &
                sample_weight=weights, max_iterations=500, tolerance=1.0e-8_dp)
            if (.not. status_ok(status)) error stop "GLM benchmark fit failed"
        end do
        call system_clock(end_tick)
        elapsed = real(end_tick-start_tick, dp)/real(clock_rate, dp)/real(repeats, dp)
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "GLM benchmark prediction failed"
        theta = model%parameters()
        allocate(gradient(size(theta)))
        call model%objective_value_gradient(x, target, theta, value, gradient, status, &
            family=family, alpha=0.05_dp, sample_weight=weights)
        if (.not. status_ok(status)) error stop "GLM benchmark objective failed"
        write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') &
            "glm_family,", family, ",n_samples,", size(x, 1), ",fit_seconds,", &
            elapsed, ",mean_prediction,", sum(prediction)/real(size(prediction), dp), &
            ",objective,", value
    end subroutine benchmark_family

end program fortml_bench_glm_regression
