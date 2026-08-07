program fortml_bench_gp_hyperparameter_training
    !! Release application for exact-GP seeded hyperparameter multistart.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_gp_training, only: gp_hyperparameter_options_t, &
        gp_hyperparameter_result_t, gp_optimize_hyperparameters
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 10
    integer, parameter :: n_features = 1
    integer, parameter :: n_outputs = 1
    integer, parameter :: starts = 4
    real(dp) :: x(n_samples, n_features), y(n_samples, n_outputs)
    real(dp), allocatable :: parameters(:), gradient(:)
    real(dp) :: elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i
    type(kernel_t) :: kernel
    type(gp_regression_t) :: model
    type(gp_hyperparameter_options_t) :: options
    type(gp_hyperparameter_result_t) :: result
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        x(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(n_samples - 1, dp)
        y(i, 1) = sin(2.0_dp*x(i, 1)) + 0.15_dp*cos(3.0_dp*x(i, 1))
    end do
    kernel = make_rbf_kernel(n_features, 1.6_dp, 0.35_dp, status)
    call model%fit(x, y, kernel, 0.08_dp, status, jitter=1.0e-10_dp)
    if (.not. status_ok(status)) error stop "GP multistart benchmark fit failed"

    options%max_iterations = 120
    options%max_line_search = 40
    options%gradient_tolerance = 2.0e-5_dp
    options%lower_bound = -8.0_dp
    options%upper_bound = 8.0_dp
    options%starts = starts
    options%seed = 20260807_int64
    options%include_current = .true.
    call system_clock(clock_start, clock_rate)
    call gp_optimize_hyperparameters(model, options, result, status)
    call system_clock(clock_end)
    if (.not. status_ok(status) .or. .not. result%converged) then
        error stop "GP multistart benchmark optimization failed"
    end if
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    parameters = model%parameters()
    allocate(gradient(size(parameters)))
    call model%hyperparameter_gradient(gradient, status)
    if (.not. status_ok(status)) error stop "GP multistart benchmark gradient failed"
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,3(es24.16,a))') &
        "gp_exact_multistart,", result%start_count, ",", &
        result%successful_starts, ",", result%best_start, ",", &
        result%objective_evaluations, ",", elapsed, ",", &
        result%negative_log_marginal_likelihood, ",", result%gradient_norm, ",", &
        parameters(1), ",", parameters(2), ",", parameters(3), ","

    cuda%selected = .true.
    cuda%available = .true.
    cuda%kind = FORTML_DEVICE_CUDA
    call gp_optimize_hyperparameters(model, options, result, status, cuda)
    write (*, '(a,i0)') "gp_exact_cuda_refusal,", status%code
end program fortml_bench_gp_hyperparameter_training
