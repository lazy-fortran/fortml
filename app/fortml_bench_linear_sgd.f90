program fortml_bench_linear_sgd
    !! Correctness-gated release workload for deterministic linear SGD.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_linear_sgd, only: linear_sgd_options_t, linear_sgd_regression_t, &
        linear_sgd_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    real(dp) :: x(8, 2), y(8), prediction(8), probabilities(8, 2)
    integer :: labels(8)
    type(linear_sgd_options_t) :: options
    type(linear_sgd_regression_t) :: regression
    type(linear_sgd_classifier_t) :: classifier
    type(fortnum_status_t) :: status
    integer :: i

    do i = 1, size(x, 1)
        x(i, :) = [real(i-1, dp), real(mod(i, 3), dp)]
        y(i) = 0.5_dp + 1.5_dp*x(i, 1) - 0.25_dp*x(i, 2)
        labels(i) = merge(1, -1, y(i) > 5.0_dp)
    end do
    options%epochs = 64
    options%batch_size = 2
    options%learning_rate = 0.03_dp
    options%shuffle = .true.
    options%shuffle_seed = 41
    options%average = .true.
    call regression%fit(x, y, status, options)
    if (.not. status_ok(status)) error stop "linear SGD regression benchmark failed"
    call regression%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "linear SGD regression prediction failed"
    write (*, '(a,i0,a,i0,a,es24.16)') "linear_sgd_regression,", size(x, 1), ",", &
        regression%update_count(), ",", sum((prediction-y)**2)/real(size(y), dp)

    options%epochs = 96
    options%batch_size = 2
    options%average = .false.
    call classifier%fit(x, labels, status, options, classes=[-1, 1])
    if (.not. status_ok(status)) error stop "linear SGD classifier benchmark failed"
    call classifier%predict_proba(x, probabilities, status)
    if (.not. status_ok(status)) error stop "linear SGD classifier prediction failed"
    write (*, '(a,i0,a,i0,a,es24.16)') "linear_sgd_classifier,", size(x, 1), ",", &
        classifier%update_count(), ",", sum(probabilities(:, 2))/real(size(x, 1), dp)
end program fortml_bench_linear_sgd
