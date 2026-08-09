program fortml_bench_xgboost_multiclass_log_proba
    !! Release workload for stable multiclass XGBoost log probabilities and
    !! fixed-tree input/leaf-coordinate products.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_options_t
    use fortml_xgboost_multiclass, only: xgboost_multiclass_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(xgboost_multiclass_t) :: model
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(9, 1), query(3, 1), query_dot(3, 1)
    real(dp) :: probabilities(3, 3), log_probabilities(3, 3)
    real(dp) :: log_dot(3, 3), log_bar(3, 3), log_plus(3, 3), log_minus(3, 3)
    real(dp) :: x_bar(3, 1)
    real(dp), allocatable :: direction(:), parameter_bar(:), parameters(:)
    real(dp) :: start_time, finish_time, h, input_error, input_duality
    real(dp) :: parameter_duality, roundtrip_error
    integer :: labels(9), cuda_status

    x(:, 1) = [-4.0_dp, -3.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, &
        2.0_dp, 3.0_dp, 4.0_dp]
    labels = [-8, -8, -8, 2, 2, 2, 11, 11, 11]
    query(:, 1) = [-2.3_dp, 0.17_dp, 2.4_dp]
    query_dot(:, 1) = [0.17_dp, -0.13_dp, 0.23_dp]
    log_bar = reshape([0.2_dp, -0.1_dp, 0.4_dp, -0.3_dp, 0.1_dp, -0.2_dp, &
        0.5_dp, -0.4_dp, 0.3_dp], shape(log_bar))
    options%n_estimators = 4
    options%max_depth = 1
    options%learning_rate = 0.4_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.0_dp

    call cpu_time(start_time)
    call model%fit(x, labels, status, options)
    call cpu_time(finish_time)
    if (.not. status_ok(status)) error stop "multiclass log-probability fit failed"
    write (*, '(a,1x,es24.16)') "xgb_mc_log_fit_seconds", finish_time - start_time

    call model%predict_proba(query, probabilities, status)
    call model%predict_log_proba(query, log_probabilities, status)
    if (.not. status_ok(status)) error stop "multiclass log-probability prediction failed"
    roundtrip_error = maxval(abs(exp(log_probabilities) - probabilities))
    write (*, '(a,1x,es24.16)') "xgb_mc_log_roundtrip_error", roundtrip_error

    h = 1.0e-6_dp
    call model%predict_log_proba_jvp(query, query_dot, log_probabilities, log_dot, status)
    if (.not. status_ok(status)) error stop "multiclass log-probability input JVP failed"
    call model%predict_log_proba(query + h*query_dot, log_plus, status)
    call model%predict_log_proba(query - h*query_dot, log_minus, status)
    input_error = maxval(abs(log_dot - (log_plus - log_minus)/(2.0_dp*h)))
    write (*, '(a,1x,es24.16)') "xgb_mc_log_input_jvp_error", input_error

    call model%predict_log_proba_vjp(query, log_bar, x_bar, status)
    if (.not. status_ok(status)) error stop "multiclass log-probability input VJP failed"
    input_duality = abs(sum(x_bar*query_dot) - sum(log_bar*log_dot))
    write (*, '(a,1x,es24.16)') "xgb_mc_log_input_vjp_error", input_duality

    allocate(direction(model%parameter_count()), parameter_bar(model%parameter_count()))
    parameters = model%parameters(status)
    if (.not. status_ok(status) .or. size(parameters) /= size(direction)) then
        error stop "multiclass log-probability parameter metadata failed"
    end if
    direction = 0.013_dp
    direction(1::2) = -0.021_dp
    call model%predict_log_proba_parameter_jvp(query, direction, log_probabilities, log_dot, &
        status)
    call model%predict_log_proba_parameter_vjp(query, log_bar, parameter_bar, status)
    if (.not. status_ok(status)) error stop "multiclass log-probability parameter products failed"
    parameter_duality = abs(dot_product(parameter_bar, direction) - sum(log_bar*log_dot))
    write (*, '(a,1x,es24.16)') "xgb_mc_log_parameter_vjp_error", parameter_duality

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_log_proba_device(cuda, query, log_probabilities, status)
    cuda_status = status%code
    write (*, '(a,1x,i0)') "xgb_mc_log_cuda_status", cuda_status

end program fortml_bench_xgboost_multiclass_log_proba
