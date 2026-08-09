program fortml_bench_lightgbm
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    implicit none

    integer, parameter :: n = 192, d = 3
    real(real64) :: x(n, d), target(n), labels(n), weight(n), prediction(n), &
        host_prediction(n), margin(n)
    real(real64) :: staged(n, 8), warm_staged(n, 8), contributions(n, 9), sliced_prediction(n)
    real(real64) :: sx(6, 1), sy(6), expected(6), tiny_prediction(6)
    type(lightgbm_t) :: regression, binary, tiny, prefix, restored, warm
    type(lightgbm_options_t) :: options, tiny_options, warm_options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: started, finished, mse, accuracy, oracle_error, product_error
    integer :: i, snapshot_unit
    character(*), parameter :: snapshot_path = "fortml_bench_lightgbm_snapshot.txt"

    do i = 1, n
        x(i, 1) = -1.0_real64 + 2.0_real64*real(i-1, real64)/real(n-1, real64)
        x(i, 2) = sin(0.09_real64*real(i, real64))
        x(i, 3) = cos(0.04_real64*real(i, real64))
        target(i) = merge(1.5_real64 + 0.25_real64*x(i, 2), &
            -0.7_real64 + 0.12_real64*x(i, 3), x(i, 1) >= 0.08_real64)
        labels(i) = merge(1.0_real64, 0.0_real64, x(i, 1) + 0.2_real64*x(i, 2) >= 0.0_real64)
        weight(i) = 1.0_real64 + real(mod(i, 4), real64)
    end do
    options = lightgbm_options_t()
    options%n_estimators = 8
    options%num_leaves = 8
    options%min_data_in_leaf = 3
    options%max_bin = 16
    options%learning_rate = 0.2_real64
    options%l2 = 1.0_real64
    call cpu_time(started)
    call regression%fit_regression(x, target, status, options, weight)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "lightgbm regression fit failed"
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16)') "lightgbm_fit,", n, ",", d, ",", &
        real(regression%estimator_count(), real64), ",", max(0.0_real64, finished-started)
    call cpu_time(started)
    call regression%predict(x, prediction, status)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "lightgbm regression predict failed"
    host_prediction = prediction
    mse = sum(weight*(prediction-target)**2)/sum(weight)
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16)') "lightgbm_predict,", n, ",", d, ",", &
        real(regression%tree_depth(1), real64), ",", max(0.0_real64, finished-started), ",", mse

    call cpu_time(started)
    call regression%predict_staged(x, staged, status)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "lightgbm staged prediction failed"
    product_error = maxval(abs(staged(:, regression%estimator_count())-prediction))
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') "lightgbm_staged,", n, ",", d, ",", &
        regression%estimator_count(), ",", max(0.0_real64, finished-started), ",", product_error

    warm_options = options
    warm_options%n_estimators = 4
    call warm%fit_regression(x, target, status, warm_options, weight)
    if (status%code /= FORTNUM_OK) error stop "lightgbm warm prefix fit failed"
    call cpu_time(started)
    call warm%fit_warm_start(x, target, status, options, weight)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "lightgbm warm continuation failed"
    call warm%predict_staged(x, warm_staged, status)
    if (status%code /= FORTNUM_OK) error stop "lightgbm warm staged prediction failed"
    product_error = maxval(abs(warm_staged-staged))
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') "lightgbm_warm_start,", n, ",", d, ",", &
        warm%estimator_count(), ",", max(0.0_real64, finished-started), ",", product_error
    warm_options%n_estimators = 4
    call warm%fit_warm_start(x, target, status, warm_options, weight)
    write (*, '(a,i0)') "lightgbm_warm_start_invalid,", status%code

    call regression%predict_margin(x, margin, status)
    if (status%code /= FORTNUM_OK) error stop "lightgbm margin prediction failed"
    call cpu_time(started)
    call regression%predict_contributions(x, contributions, status)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "lightgbm contribution prediction failed"
    product_error = maxval(abs(sum(contributions, dim=2)-margin))
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') "lightgbm_contributions,", n, ",", d, ",", &
        regression%estimator_count(), ",", max(0.0_real64, finished-started), ",", product_error

    call cpu_time(started)
    call regression%slice(4, prefix, status)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "lightgbm prefix slice failed"
    call prefix%predict(x, sliced_prediction, status)
    if (status%code /= FORTNUM_OK) error stop "lightgbm prefix prediction failed"
    product_error = maxval(abs(sliced_prediction-staged(:, 4)))
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') "lightgbm_slice,", n, ",", d, ",", &
        prefix%estimator_count(), ",", max(0.0_real64, finished-started), ",", product_error

    call cpu_time(started)
    call regression%save_text(snapshot_path, status)
    if (status%code /= FORTNUM_OK) error stop "lightgbm text save failed"
    call restored%load_text(snapshot_path, status)
    if (status%code /= FORTNUM_OK) error stop "lightgbm text load failed"
    call restored%predict(x, sliced_prediction, status)
    call cpu_time(finished)
    if (status%code /= FORTNUM_OK) error stop "lightgbm restored prediction failed"
    product_error = maxval(abs(sliced_prediction-prediction))
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') "lightgbm_persistence,", n, ",", d, ",", &
        restored%estimator_count(), ",", max(0.0_real64, finished-started), ",", product_error
    open(newunit=snapshot_unit, file=snapshot_path, status="old", position="append", &
        action="write")
    write(snapshot_unit, '(a)') "unexpected_record 1"
    close(snapshot_unit)
    call restored%load_text(snapshot_path, status)
    write (*, '(a,i0)') "lightgbm_persistence_invalid,", status%code
    open(newunit=snapshot_unit, file=snapshot_path, status="old", action="read")
    close(snapshot_unit, status="delete")

    call binary%fit_binary(x, labels, status, options, weight)
    if (status%code /= FORTNUM_OK) error stop "lightgbm binary fit failed"
    call binary%predict(x, prediction, status)
    if (status%code /= FORTNUM_OK) error stop "lightgbm binary predict failed"
    accuracy = real(count((prediction >= 0.5_real64) .eqv. (labels >= 0.5_real64)), real64)/real(n, real64)
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16)') "lightgbm_binary,", n, ",", d, ",", &
        real(binary%estimator_count(), real64), ",", accuracy

    sx(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64, 5.0_real64]
    sy = [0.0_real64, 0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64, 100.0_real64]
    expected = [0.0_real64, 0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64, 100.0_real64]
    tiny_options = lightgbm_options_t()
    tiny_options%n_estimators = 1
    tiny_options%num_leaves = 3
    tiny_options%min_data_in_leaf = 1
    tiny_options%max_bin = 16
    tiny_options%learning_rate = 1.0_real64
    tiny_options%l2 = 0.0_real64
    call tiny%fit_regression(sx, sy, status, tiny_options)
    call tiny%predict(sx, tiny_prediction, status)
    if (status%code /= FORTNUM_OK) error stop "lightgbm oracle fit failed"
    oracle_error = maxval(abs(tiny_prediction-expected))
    write (*, '(a,es24.16)') "lightgbm_oracle,", oracle_error

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call regression%predict_device(cuda, x, prediction, status)
    if (status%code == FORTNUM_OK) then
        product_error = maxval(abs(prediction-host_prediction))
    else
        product_error = -1.0_real64
    end if
    write (*, '(a,i0,a,es24.16)') "lightgbm_cuda_result,", status%code, ",", product_error
    write (*, '(a,i0)') "lightgbm_cuda,", status%code
end program fortml_bench_lightgbm
