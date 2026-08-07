program fortml_bench_xgboost_derivatives
    !! Release workload for fitted-tree query derivative products.
    !!
    !! A fitted XGBoost tree is piecewise constant.  Away from split surfaces
    !! its input JVP and VJP are therefore exactly zero; on a learned surface
    !! the public API must refuse the classical derivative.  The benchmark
    !! emits complete query values so the Python harness can check those
    !! claims with an independent central-difference oracle.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 8, n_features = 1, n_query = 6
    real(dp), parameter :: epsilon = 1.0e-6_dp
    real(dp) :: x(n_samples, n_features), target(n_samples)
    real(dp) :: query(n_query, n_features), query_plus(n_query, n_features)
    real(dp) :: query_minus(n_query, n_features), tangent(n_query, n_features)
    real(dp) :: prediction(n_query), prediction_plus(n_query), prediction_minus(n_query)
    real(dp) :: jvp_prediction(n_query), tangent_prediction(n_query), cotangent(n_query)
    real(dp) :: adjoint(n_query, n_features)
    real(dp) :: boundary(1, n_features), boundary_tangent(1, n_features)
    real(dp) :: boundary_prediction(1), boundary_tangent_prediction(1)
    real(dp) :: boundary_cotangent(1), boundary_adjoint(1, n_features)
    real(dp) :: cuda_prediction(n_query)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: elapsed_jvp, elapsed_vjp
    integer :: i
    type(xgboost_t) :: model
    type(xgboost_options_t) :: options
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp, 7.0_dp]
    target = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
    query(:, 1) = [0.25_dp, 1.25_dp, 2.25_dp, 4.25_dp, 5.25_dp, 6.25_dp]
    tangent(:, 1) = 1.0_dp
    cotangent = [1.0_dp, -0.5_dp, 0.25_dp, 0.75_dp, -1.25_dp, 0.5_dp]
    query_plus = query + epsilon*tangent
    query_minus = query - epsilon*tangent

    options%n_estimators = 1
    options%max_depth = 1
    options%min_samples_leaf = 1
    options%learning_rate = 1.0_dp
    options%l2 = 0.0_dp
    options%min_child_weight = 0.0_dp
    call model%fit_regression(x, target, status, options)
    if (.not. status_ok(status)) error stop "XGBoost derivative fixture fit failed"
    call model%predict(query, prediction, status)
    if (.not. status_ok(status)) error stop "XGBoost derivative fixture prediction failed"
    call model%predict(query_plus, prediction_plus, status)
    if (.not. status_ok(status)) error stop "XGBoost derivative plus prediction failed"
    call model%predict(query_minus, prediction_minus, status)
    if (.not. status_ok(status)) error stop "XGBoost derivative minus prediction failed"

    call system_clock(clock_start, clock_rate)
    do i = 1, 256
        call model%predict_jvp(query, tangent, jvp_prediction, tangent_prediction, status)
        if (.not. status_ok(status)) error stop "XGBoost derivative JVP failed"
    end do
    call system_clock(clock_end)
    elapsed_jvp = real(clock_end - clock_start, dp)/real(clock_rate, dp)/256.0_dp
    call model%predict_jvp(query, tangent, jvp_prediction, tangent_prediction, status)
    if (.not. status_ok(status)) error stop "XGBoost derivative JVP reference failed"

    call system_clock(clock_start, clock_rate)
    do i = 1, 256
        call model%predict_vjp(query, cotangent, adjoint, status)
        if (.not. status_ok(status)) error stop "XGBoost derivative VJP failed"
    end do
    call system_clock(clock_end)
    elapsed_vjp = real(clock_end - clock_start, dp)/real(clock_rate, dp)/256.0_dp
    call model%predict_vjp(query, cotangent, adjoint, status)
    if (.not. status_ok(status)) error stop "XGBoost derivative VJP reference failed"

    boundary(1, 1) = 3.5_dp
    boundary_tangent(1, 1) = 1.0_dp
    boundary_cotangent(1) = 1.0_dp
    call model%predict_jvp(boundary, boundary_tangent, boundary_prediction, &
        boundary_tangent_prediction, status)
    write (*, '(a,i0)') "xgb_derivative_jvp_boundary_status ", status%code
    call model%predict_vjp(boundary, boundary_cotangent, boundary_adjoint, status)
    write (*, '(a,i0)') "xgb_derivative_vjp_boundary_status ", status%code

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda_prediction = -37.0_dp
    call model%predict_device(cuda, query, cuda_prediction, status)
    write (*, '(a,i0)') "xgb_derivative_cuda_status ", status%code

    write (*, '(a,*(1x,es24.16))') "xgb_derivative_query", query(:, 1)
    write (*, '(a,*(1x,es24.16))') "xgb_derivative_prediction", prediction
    write (*, '(a,*(1x,es24.16))') "xgb_derivative_prediction_plus", prediction_plus
    write (*, '(a,*(1x,es24.16))') "xgb_derivative_prediction_minus", prediction_minus
    write (*, '(a,*(1x,es24.16))') "xgb_derivative_jvp", tangent_prediction
    write (*, '(a,*(1x,es24.16))') "xgb_derivative_vjp", adjoint(:, 1)
    write (*, '(a,es24.16)') "xgb_derivative_jvp_seconds ", elapsed_jvp
    write (*, '(a,es24.16)') "xgb_derivative_vjp_seconds ", elapsed_vjp
end program fortml_bench_xgboost_derivatives
