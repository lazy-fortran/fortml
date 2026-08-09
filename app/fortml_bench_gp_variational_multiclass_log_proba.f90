program fortml_bench_gp_variational_multiclass_log_proba
    !! Release workload for stable variational-GP multiclass log probabilities.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_gp_variational_multiclass_classification, only: &
        gp_variational_multiclass_classification_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 64, n_classes = 3, n_features = 2
    integer, parameter :: n_inducing = 4, repetitions = 12
    real(dp) :: x(n_samples, n_features), inducing(n_inducing, n_features)
    real(dp) :: direction(14*n_classes), log_probabilities(n_samples, n_classes)
    real(dp) :: log_probabilities_dot(n_samples, n_classes), seconds, sum_error
    integer :: classes(n_classes), i, j, code
    integer(int64) :: clock_start, clock_end, clock_rate
    type(gp_variational_multiclass_classification_t) :: model
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        x(i, 1) = -1.2_dp + 2.4_dp*real(i - 1, dp)/real(n_samples - 1, dp)
        x(i, 2) = sin(x(i, 1))
    end do
    do i = 1, n_inducing
        inducing(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(n_inducing - 1, dp)
        inducing(i, 2) = sin(inducing(i, 1))
    end do
    classes = [30, 10, 20]
    direction = 0.0_dp
    do i = 1, size(direction)
        direction(i) = 0.003_dp*real(mod(i, 7) - 3, dp)
    end do

    kernel = make_rbf_kernel(n_features, 1.3_dp, 0.9_dp, status)
    if (.not. status_ok(status)) error stop "kernel construction failed"
    call model%initialize(inducing, classes, kernel, 16, 20260809, status)
    if (.not. status_ok(status)) error stop "variational GP multiclass initialization failed"
    call model%predict_log_proba(x, log_probabilities, status)
    if (.not. status_ok(status)) error stop "log-probability warmup failed"
    call model%predict_log_proba_parameter_jvp(x, direction(1:model%parameter_count()), &
        log_probabilities, log_probabilities_dot, status)
    if (.not. status_ok(status)) error stop "log-probability JVP warmup failed"
    call system_clock(clock_start, clock_rate)
    do i = 1, repetitions
        call model%predict_log_proba(x, log_probabilities, status)
        if (.not. status_ok(status)) error stop "log-probability prediction failed"
    end do
    call system_clock(clock_end)
    seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    sum_error = maxval(abs(sum(exp(log_probabilities), dim=2) - 1.0_dp))
    write (*, '(a,es24.16,a,es24.16,a,es24.16)') &
        "gp_variational_multiclass_log_proba,cpu,seconds,", seconds, ",simplex_error,", &
        sum_error, ",log_probability_sum,", sum(log_probabilities)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_log_proba_device(cuda, x, log_probabilities, status)
    code = status%code
    write (*, '(a,i0)') "gp_variational_multiclass_log_proba,device,cuda,refused,", code
end program fortml_bench_gp_variational_multiclass_log_proba
