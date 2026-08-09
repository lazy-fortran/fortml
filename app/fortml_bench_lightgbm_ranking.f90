program fortml_bench_lightgbm_ranking
    !! Release workload for LightGBM query-weighted rank:pairwise training.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    implicit none

    real(real64) :: x(2,1), target(2), prediction(2)
    integer :: group(2)
    type(lightgbm_t) :: model
    type(lightgbm_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: started, finished, oracle_error

    x(:,1) = [0.0_real64, 1.0_real64]
    target = [0.0_real64, 1.0_real64]
    group = [17, 17]
    options = lightgbm_options_t()
    options%n_estimators = 1
    options%num_leaves = 2
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 1.0_real64
    options%l2 = 0.0_real64

    call cpu_time(started)
    call model%fit_ranking(x, target, group, status, options)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "LightGBM ranking fit failed"
    write (*, '(a,es24.16)') "lightgbm_ranking_fit_seconds,", &
        max(0.0_real64, finished-started)

    call cpu_time(started)
    call model%predict(x, prediction, status)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "LightGBM ranking prediction failed"
    oracle_error = max(abs(prediction(1)+2.0_real64), &
        abs(prediction(2)-2.0_real64))
    write (*, '(a,es24.16)') "lightgbm_ranking_predict_seconds,", &
        max(0.0_real64, finished-started)
    write (*, '(a,2(es24.16,:,a))') "lightgbm_ranking_prediction,", &
        prediction(1), ",", prediction(2)
    write (*, '(a,es24.16)') "lightgbm_ranking_oracle_error,", oracle_error
    write (*, '(a,a)') "lightgbm_ranking_objective,", trim(model%objective_name())

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    write (*, '(a,i0)') "lightgbm_ranking_cuda_status,", status%code
end program fortml_bench_lightgbm_ranking
