program fortml_bench_gp_categorical_likelihood
    !! Release probe for categorical variational-GP likelihood hyperparameters.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_variational_categorical_classification, only: &
        gp_variational_categorical_classification_t, &
        gp_variational_likelihood_options_t, gp_variational_likelihood_state_t
    implicit none

    type(gp_variational_categorical_classification_t) :: model
    type(gp_variational_likelihood_options_t) :: options
    type(gp_variational_likelihood_state_t) :: fit_state
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(6, 1), inducing(3, 1), probabilities(6, 3), probabilities_dot(6, 3)
    real(dp) :: means(6, 3), variances(6, 3)
    real(dp) :: probabilities_bar(6, 3), parameter_bar(1), value, tangent
    real(dp) :: fit_seconds, t0, t1
    real(dp), allocatable :: packed(:)
    integer :: labels(6), classes(3), clock_rate, clock_start, clock_stop, cuda_code, i, j

    x(:, 1) = [-1.2_dp, -0.7_dp, -0.15_dp, 0.25_dp, 0.8_dp, 1.3_dp]
    inducing(:, 1) = [-0.9_dp, 0.0_dp, 0.85_dp]
    labels = [30, 10, 20, 10, 30, 20]
    classes = [30, 10, 20]
    probabilities_bar = reshape([ &
        0.17_dp, -0.09_dp, 0.04_dp, 0.12_dp, -0.06_dp, 0.08_dp, &
        -0.05_dp, 0.11_dp, -0.08_dp, 0.07_dp, 0.03_dp, -0.02_dp, &
        0.02_dp, 0.06_dp, 0.09_dp, -0.04_dp, 0.05_dp, 0.01_dp], [6, 3])
    kernel = make_rbf_kernel(1, 1.2_dp, 0.8_dp, status)
    if (.not. status_ok(status)) error stop "kernel constructor failed"
    call model%initialize(inducing, classes, kernel, 8, 20260808, status)
    if (.not. status_ok(status)) error stop "categorical initialization failed"
    packed = model%parameters()
    packed(1:3) = [0.65_dp, -0.25_dp, 0.4_dp]
    packed(10:12) = [-0.35_dp, 0.2_dp, 0.55_dp]
    packed(19:21) = [0.25_dp, 0.45_dp, -0.3_dp]
    call model%set_parameters(packed, status)
    if (.not. status_ok(status)) error stop "categorical state setup failed"
    options%max_iterations = 80
    options%gradient_tolerance = 1.0e-6_dp
    call system_clock(count_rate=clock_rate)
    call system_clock(count=clock_start)
    call model%fit_likelihood(x, labels, status, options, fit_state)
    call system_clock(count=clock_stop)
    if (.not. status_ok(status)) error stop "categorical likelihood fit failed"
    fit_seconds = real(clock_stop - clock_start, dp)/real(clock_rate, dp)
    call model%predict_latent(x, means, variances, status)
    if (.not. status_ok(status)) error stop "categorical latent prediction failed"
    call model%elbo_likelihood_parameter_gradient(x, labels, value, parameter_bar, status)
    if (.not. status_ok(status)) error stop "categorical likelihood gradient failed"
    call model%elbo_likelihood_parameter_jvp(x, labels, [1.0_dp], value, tangent, status)
    if (.not. status_ok(status)) error stop "categorical likelihood JVP failed"
    call model%predict_proba_likelihood_parameter_jvp(x, [1.0_dp], probabilities, &
        probabilities_dot, status)
    if (.not. status_ok(status)) error stop "categorical likelihood probability JVP failed"
    call model%predict_proba_likelihood_parameter_vjp(x, probabilities_bar, parameter_bar, status)
    if (.not. status_ok(status)) error stop "categorical likelihood probability VJP failed"
    write (*, '(a,es24.16)') "gp_categorical_likelihood_scale,", model%likelihood_scale()
    write (*, '(a,es24.16)') "gp_categorical_likelihood_fit_seconds,", fit_seconds
    write (*, '(a,i0)') "gp_categorical_likelihood_iterations,", fit_state%iterations
    write (*, '(a,es24.16)') "gp_categorical_likelihood_elbo,", value
    write (*, '(a,es24.16)') "gp_categorical_likelihood_gradient,", parameter_bar(1)
    write (*, '(a,es24.16)') "gp_categorical_likelihood_jvp,", tangent
    do i = 1, size(x, 1)
        do j = 1, 3
            write (*, '(a,i0,a,i0,a,es24.16,a,es24.16)') "gp_categorical_likelihood_latent,", &
                i, ",", j, ",", means(i, j), ",", variances(i, j)
            write (*, '(a,i0,a,i0,a,es24.16,a,es24.16)') "gp_categorical_likelihood_probability,", &
                i, ",", j, ",", probabilities(i, j), ",", probabilities_dot(i, j)
        end do
    end do
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_likelihood_parameter_jvp_device(cuda, x, [1.0_dp], probabilities, &
        probabilities_dot, status)
    cuda_code = status%code
    write (*, '(a,i0)') "gp_categorical_likelihood_cuda_jvp,", cuda_code
end program fortml_bench_gp_categorical_likelihood
