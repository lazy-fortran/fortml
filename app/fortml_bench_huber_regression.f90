program fortml_bench_huber_regression
    !! Release smoke workload for weighted Huber regression products.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_huber_regression, only: huber_regression_t, &
        huber_training_objective_t, huber_lbfgsb_options_t, &
        huber_lbfgsb_result_t, huber_optimize_lbfgsb
    implicit none
    integer, parameter :: dp = real64
    type(huber_regression_t), target :: model
    type(huber_training_objective_t) :: objective
    type(huber_lbfgsb_options_t) :: options
    type(huber_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(8, 1), y(8, 1), weights(8), prediction(8, 1)
    real(dp) :: probe_value, probe_gradient(2), probe_parameters(2)
    integer(int64) :: started, finished, rate
    real(dp) :: elapsed

    x(:, 1) = [-2.0_dp, -1.5_dp, -1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    y(:, 1) = [-3.7_dp, -2.5_dp, -1.6_dp, -0.9_dp, 0.2_dp, 1.0_dp, 2.2_dp, 3.0_dp]
    weights = [0.5_dp, 0.7_dp, 1.0_dp, 1.2_dp, 1.5_dp, 1.1_dp, 0.9_dp, 0.8_dp]
    options%delta = 1.0_dp
    options%l2 = 1.0e-3_dp
    options%max_iterations = 180
    options%gradient_tolerance = 1.0e-8_dp
    call model%initialize(1, 1, status)
    if (.not. status_ok(status)) error stop "Huber release initialization failed"
    call system_clock(started, rate)
    call huber_optimize_lbfgsb(model, x, y, options, result, status, &
        sample_weight=weights)
    call system_clock(finished)
    if (.not. status_ok(status)) error stop "Huber release fit failed"
    elapsed = real(finished-started, dp)/real(rate, dp)
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "Huber release prediction failed"
    write (*, '(a,es24.16)') "huber_fit_seconds,", elapsed
    write (*, '(a,es24.16)') "huber_objective,", result%objective
    write (*, '(a,es24.16)') "huber_gradient_norm,", result%gradient_norm
    write (*, '(a,es24.16)') "huber_prediction_mean,", sum(prediction)/real(size(prediction), dp)
    call model%set_parameters([0.5_dp, 1.1_dp], status)
    if (.not. status_ok(status)) error stop "Huber release probe state failed"
    call objective%initialize(model, x, y, options%delta, options%l2, status)
    if (.not. status_ok(status)) error stop "Huber release objective failed"
    probe_parameters = objective%parameters()
    call objective%value_gradient(probe_parameters, probe_value, probe_gradient, status)
    if (.not. status_ok(status)) error stop "Huber release probe products failed"
    write (*, '(a,es24.16)') "huber_probe_value,", probe_value
    write (*, '(a,es24.16)') "huber_probe_gradient_norm,", sqrt(sum(probe_gradient**2))
    write (*, '(a,2es24.16)') "huber_probe_parameters,", probe_parameters
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop "Huber CUDA contract changed"
    write (*, '(a)') "huber_cuda,unavailable"
end program fortml_bench_huber_regression
