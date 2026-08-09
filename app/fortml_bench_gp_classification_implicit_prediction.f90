program fortml_bench_gp_classification_implicit_prediction
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
    implicit none

    integer, parameter :: repetitions = 32
    real(dp) :: x(8, 1), weights(8), query(3, 1), direction(2)
    real(dp) :: mean(3), mean_dot(3), variance(3), variance_dot(3)
    real(dp) :: probabilities(3, 2), probabilities_dot(3, 2), seconds
    integer :: labels(8), likelihood, repetition, i, cuda_code
    integer(int64) :: tick_start, tick_end, tick_rate
    type(gp_classification_t) :: model
    type(gp_classification_options_t) :: options
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    x(:, 1) = [-1.7_dp, -1.15_dp, -0.62_dp, -0.18_dp, &
        0.14_dp, 0.55_dp, 1.08_dp, 1.63_dp]
    labels = [-3, -3, -3, -3, 9, 9, 9, 9]
    weights = [0.45_dp, 1.4_dp, 0.0_dp, 0.8_dp, &
        1.7_dp, 0.6_dp, 1.25_dp, 0.9_dp]
    query(:, 1) = [-0.85_dp, 0.05_dp, 0.92_dp]
    direction = [0.19_dp, -0.14_dp]
    options%max_iterations = 120
    options%tolerance = 1.0e-11_dp
    options%jitter = 1.0e-7_dp
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.

    do likelihood = GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
        options%likelihood = likelihood
        kernel = make_rbf_kernel(1, 1.35_dp, 0.72_dp, status)
        if (.not. status_ok(status)) error stop "implicit prediction kernel failed"
        call model%fit(x, labels, kernel, status, options, sample_weight=weights)
        if (.not. status_ok(status)) error stop "implicit prediction fit failed"
        call model%predict_latent_hyperparameter_jvp(query, direction, mean, &
            mean_dot, variance, variance_dot, status)
        if (.not. status_ok(status)) error stop "implicit latent product failed"
        call model%predict_proba_hyperparameter_jvp(query, direction, probabilities, &
            probabilities_dot, status)
        if (.not. status_ok(status)) error stop "implicit probability product failed"
        call system_clock(tick_start, tick_rate)
        do repetition = 1, repetitions
            call model%predict_proba_hyperparameter_jvp(query, direction, &
                probabilities, probabilities_dot, status)
            if (.not. status_ok(status)) error stop "implicit product timing failed"
        end do
        call system_clock(tick_end)
        seconds = real(tick_end - tick_start, dp)/real(tick_rate, dp)/ &
            real(repetitions, dp)
        do i = 1, size(query, 1)
            write (*, '(a,i0,a,i0,a,es24.16)') &
                "gp_classification_implicit,", likelihood, ",mean_dot,", i, ",", mean_dot(i)
            write (*, '(a,i0,a,i0,a,es24.16)') &
                "gp_classification_implicit,", likelihood, ",variance_dot,", i, ",", &
                variance_dot(i)
            write (*, '(a,i0,a,i0,a,es24.16)') &
                "gp_classification_implicit,", likelihood, ",probability_dot,", i, ",", &
                probabilities_dot(i, 2)
        end do
        write (*, '(a,i0,a,es24.16)') &
            "gp_classification_implicit,", likelihood, ",seconds,", seconds
        probabilities = -1.0_dp
        probabilities_dot = -1.0_dp
        call model%predict_proba_hyperparameter_jvp_device(cuda, query, direction, &
            probabilities, probabilities_dot, status)
        cuda_code = status%code
        write (*, '(a,i0,a,i0)') &
            "gp_classification_implicit,", likelihood, ",cuda_status,", cuda_code
    end do
end program fortml_bench_gp_classification_implicit_prediction
