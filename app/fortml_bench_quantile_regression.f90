program fortml_bench_quantile_regression
    !! Release smoke workload for weighted multi-output quantile regression.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_quantile_regression, only: quantile_regression_t, &
        quantile_training_objective_t, quantile_lbfgsb_options_t, &
        quantile_lbfgsb_result_t, quantile_optimize_lbfgsb
    implicit none
    integer, parameter :: dp = real64
    type(quantile_regression_t), target :: model
    type(quantile_training_objective_t) :: objective
    type(quantile_lbfgsb_options_t) :: options
    type(quantile_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(7, 2), target(7, 2), levels(2), weights(7), prediction(7, 2)
    real(dp) :: probe_value, probe_gradient(6), probe_parameters(6)
    integer(int64) :: started, finished, rate
    real(dp) :: elapsed
    integer :: i

    levels = [0.25_dp, 0.75_dp]
    x = reshape([ &
        -1.2_dp, -0.8_dp, -0.3_dp, 0.1_dp, 0.6_dp, 1.0_dp, 1.5_dp, &
        0.7_dp, -0.5_dp, 1.1_dp, -1.3_dp, 0.4_dp, 1.7_dp, -0.9_dp], &
        shape(x))
    target(:, 1) = [0.3_dp, 0.8_dp, -0.2_dp, 1.1_dp, 0.1_dp, 1.8_dp, 0.7_dp]
    target(:, 2) = [-0.4_dp, 0.5_dp, 1.4_dp, -0.8_dp, 1.1_dp, 0.2_dp, 1.7_dp]
    weights = [0.6_dp, 1.0_dp, 1.3_dp, 0.8_dp, 1.2_dp, 0.9_dp, 0.7_dp]
    options%l2 = 1.0e-3_dp
    options%fit_smoothing = 1.0e-1_dp
    options%max_iterations = 300
    options%max_line_search = 100
    call model%initialize(2, 2, levels, status)
    if (.not. status_ok(status)) error stop "quantile release initialization failed"
    call system_clock(started, rate)
    call quantile_optimize_lbfgsb(model, x, target, options, result, status, &
        sample_weight=weights)
    if (.not. status_ok(status)) then
        write (*, '(a,i0,1x,a)') "quantile_release_fit_status,", status%code, &
            trim(status%msg)
        error stop "quantile release fit failed"
    end if
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "quantile release prediction failed"
    call system_clock(finished)
    elapsed = real(finished-started, dp)/real(rate, dp)
    write (*, '(a,es24.16)') "quantile_fit_predict_seconds,", elapsed
    write (*, '(a,es24.16)') "quantile_exact_objective,", result%objective
    write (*, '(a,es24.16)') "quantile_smoothed_gradient_norm,", result%gradient_norm
    write (*, '(a,es24.16)') "quantile_exact_gradient_norm,", result%exact_gradient_norm
    write (*, '(a,es24.16)') "quantile_prediction_mean,", &
        sum(prediction)/real(size(prediction), dp)
    call model%set_parameters([0.1_dp, -0.2_dp, 1.0_dp, 0.3_dp, 0.4_dp, 0.8_dp], status)
    if (.not. status_ok(status)) error stop "quantile release probe state failed"
    call objective%initialize(model, x, target, options%l2, status, &
        sample_weight=weights)
    if (.not. status_ok(status)) error stop "quantile release objective failed"
    probe_parameters = objective%parameters()
    call objective%value_gradient(probe_parameters, probe_value, probe_gradient, status)
    if (.not. status_ok(status)) error stop "quantile release probe products failed"
    write (*, '(a,es24.16)') "quantile_probe_value,", probe_value
    write (*, '(a,es24.16)') "quantile_probe_gradient_norm,", &
        sqrt(sum(probe_gradient**2))
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop &
        "quantile CUDA contract changed"
    write (*, '(a)') "quantile_cuda,unavailable"
end program fortml_bench_quantile_regression
