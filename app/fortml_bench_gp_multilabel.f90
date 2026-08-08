program fortml_bench_gp_multilabel
    !! Correctness-gated release probe for multilabel Laplace GP heads.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_multilabel_classification, only: &
        gp_multilabel_classification_t, gp_multilabel_classification_options_t
    implicit none
    type(gp_multilabel_classification_t) :: model
    type(gp_multilabel_classification_options_t) :: options
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(10, 1), query(5, 1), query_dot(5, 1)
    real(dp) :: probabilities(5, 2), base_probabilities(5, 2)
    real(dp) :: probabilities_dot(5, 2), input_dot(5, 2), x_bar(5, 1)
    real(dp) :: probabilities_bar(5, 2), t0, t1, fit_seconds, predict_seconds
    real(dp), allocatable :: direction(:), parameter_bar(:)
    integer :: indicators(10, 2), predicted(5, 2), clock_rate, clock_start, clock_stop
    integer :: i, j, cuda_code

    x(:, 1) = [-2.0_dp, -1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, &
        0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp]
    indicators(:, 1) = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    indicators(:, 2) = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    query(:, 1) = [-1.7_dp, -0.4_dp, 0.0_dp, 0.7_dp, 1.7_dp]
    query_dot(:, 1) = [0.2_dp, -0.3_dp, 0.1_dp, 0.4_dp, -0.2_dp]
    probabilities_bar = reshape([0.2_dp, -0.4_dp, 0.7_dp, -0.1_dp, 0.5_dp, &
        -0.3_dp, -0.6_dp, 0.1_dp, 0.2_dp, 0.8_dp], shape(probabilities_bar))
    kernel = make_rbf_kernel(1, 1.3_dp, 0.75_dp, status)
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    options%jitter = 1.0e-7_dp
    call system_clock(count_rate=clock_rate)
    call system_clock(count=clock_start)
    call model%fit(x, indicators, kernel, status, options, &
        sample_weight=[1.0_dp, 0.9_dp, 1.1_dp, 1.0_dp, 0.8_dp, 1.2_dp, &
        1.0_dp, 1.1_dp, 0.9_dp, 1.0_dp])
    call system_clock(count=clock_stop)
    fit_seconds = real(clock_stop-clock_start, dp)/real(clock_rate, dp)
    if (.not. status_ok(status)) error stop "multilabel GP benchmark fit failed"
    call system_clock(count=clock_start)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predicted, status)
    call system_clock(count=clock_stop)
    predict_seconds = real(clock_stop-clock_start, dp)/real(clock_rate, dp)
    if (.not. status_ok(status)) error stop "multilabel GP benchmark prediction failed"
    base_probabilities = probabilities
    allocate(direction(model%parameter_count()), parameter_bar(model%parameter_count()))
    direction = [0.13_dp, -0.09_dp, 0.07_dp, -0.05_dp]
    call model%predict_proba_jvp(query, query_dot, probabilities, probabilities_dot, status)
    if (.not. status_ok(status)) error stop "multilabel GP input JVP failed"
    input_dot = probabilities_dot
    call model%predict_proba_vjp(query, probabilities_bar, x_bar, status)
    if (.not. status_ok(status)) error stop "multilabel GP input VJP failed"
    call model%predict_proba_parameter_jvp(query, direction, probabilities, &
        probabilities_dot, status)
    if (.not. status_ok(status)) error stop "multilabel GP parameter JVP failed"
    call model%predict_proba_parameter_vjp(query, probabilities_bar, parameter_bar, status)
    if (.not. status_ok(status)) error stop "multilabel GP parameter VJP failed"
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, query, probabilities, status)
    cuda_code = status%code

    write (*, '(a,i0)') "gp_multilabel_parameter_count,", model%parameter_count()
    write (*, '(a,es24.16)') "gp_multilabel_fit_seconds,", fit_seconds
    write (*, '(a,es24.16)') "gp_multilabel_predict_seconds,", predict_seconds
    do i = 1, size(query, 1)
        do j = 1, 2
            write (*, '(a,2(i0,a),es24.16)') "gp_multilabel_probability,", i, ",", j, ",", base_probabilities(i, j)
            write (*, '(a,2(i0,a),es24.16)') "gp_multilabel_probability_jvp,", i, ",", j, ",", input_dot(i, j)
            write (*, '(a,2(i0,a),i0)') "gp_multilabel_prediction,", i, ",", j, ",", predicted(i, j)
        end do
        write (*, '(a,i0,a,es24.16)') "gp_multilabel_input_vjp,", i, ",", x_bar(i, 1)
    end do
    write (*, '(a,i0)') "gp_multilabel_cuda,", cuda_code
end program fortml_bench_gp_multilabel
