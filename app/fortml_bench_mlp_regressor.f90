program fortml_bench_mlp_regressor
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp_regressor, only: mlp_regressor_t, mlp_regressor_options_t
    implicit none
    type(mlp_regressor_t) :: model, lbfgs_model
    type(mlp_regressor_options_t) :: options, lbfgs_options
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), target(8, 1), prediction(8, 1), mse
    integer :: i, clock_start, clock_end, clock_rate
    real(dp) :: elapsed

    x(:, 1) = [-2.0_dp, -1.5_dp, -1.0_dp, -0.25_dp, 0.25_dp, 1.0_dp, 1.5_dp, 2.0_dp]
    target(:, 1) = 0.8_dp*x(:, 1) - 0.35_dp
    allocate(options%layer_sizes(3)); options%layer_sizes = [1, 3, 1]
    options%training%max_epochs = 24
    options%training%learning_rate = 0.02_dp
    options%training%tolerance = 0.0_dp
    options%training%restore_best = .false.
    call system_clock(clock_start, clock_rate)
    call model%fit(x, target, status, options)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "MLP regressor benchmark fit failed"
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "MLP regressor benchmark prediction failed"
    mse = sum((prediction-target)**2)/real(size(target), dp)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "mlp_regressor_train", size(x, 1), model%parameter_count(), elapsed, mse

    allocate(lbfgs_options%layer_sizes(2)); lbfgs_options%layer_sizes = [1, 1]
    lbfgs_options%use_lbfgsb = .true.
    lbfgs_options%lbfgsb%max_iterations = 100
    lbfgs_options%lbfgsb%gradient_tolerance = 1.0e-7_dp
    call system_clock(clock_start)
    call lbfgs_model%fit(x, target, status, lbfgs_options)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "MLP regressor benchmark L-BFGS-B failed"
    call lbfgs_model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "MLP regressor benchmark L-BFGS-B prediction failed"
    mse = sum((prediction-target)**2)/real(size(target), dp)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    write (*, '(a,",",i0,",",i0,",",es24.16,",",es24.16)') &
        "mlp_regressor_lbfgsb", size(x, 1), lbfgs_model%parameter_count(), elapsed, mse
end program fortml_bench_mlp_regressor
