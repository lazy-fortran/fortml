program fortml_bench_classifiers
    !! Correctness-gated multinomial and neural classifier workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_softmax_regression, only: softmax_regression_t
    use fortml_mlp_classifier, only: mlp_classifier_t, mlp_classifier_options_t, &
        mlp_classifier_state_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 192, n_features = 6, n_classes = 3
    integer, parameter :: n_hidden = 12, n_softmax_iterations = 1000
    real(dp), parameter :: softmax_l2 = 5.0e-2_dp
    real(dp), parameter :: mlp_l2 = 1.0e-3_dp
    real(dp), parameter :: mlp_learning_rate = 3.0e-2_dp
    integer, parameter :: mlp_epochs = 80, prediction_repetitions = 16
    integer, parameter :: class_labels(3) = [-7, 3, 11]
    real(dp) :: x(n_samples, n_features), softmax_probabilities(n_samples, n_classes)
    real(dp) :: mlp_probabilities(n_samples, n_classes)
    integer :: labels(n_samples), softmax_labels(n_samples), mlp_labels(n_samples)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(softmax_regression_t) :: softmax
    type(mlp_classifier_t) :: mlp
    type(mlp_classifier_options_t) :: mlp_options
    type(mlp_classifier_state_t) :: mlp_state

    call get_environment_variable("FORTML_BENCH_CLASSIFIER_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_CLASSIFIER_ORACLE is required"
    end if
    call make_fixture(x, labels)

    call system_clock(clock_start, clock_rate)
    call softmax%fit(x, labels, status, l2=softmax_l2, max_iterations=&
        n_softmax_iterations, tolerance=1.0e-6_dp)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "softmax benchmark fit failed"
    fit_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    call softmax%predict_proba(x, softmax_probabilities, status)
    call softmax%predict(x, softmax_labels, status)
    if (.not. status_ok(status)) error stop "softmax benchmark prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call softmax%predict_proba(x, softmax_probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "softmax_fit,", n_samples, ",", &
        n_features, ",", n_classes, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "softmax_predict,", n_samples, ",", &
        n_features, ",", n_classes, ",", predict_seconds

    mlp_options%max_epochs = mlp_epochs
    mlp_options%batch_size = 0
    mlp_options%shuffle = .false.
    mlp_options%learning_rate = mlp_learning_rate
    mlp_options%l2 = mlp_l2
    mlp_options%initialization_seed = 23
    mlp_options%tolerance = 0.0_dp
    call system_clock(clock_start, clock_rate)
    call mlp%fit(x, labels, status, hidden_layer_sizes=[n_hidden], &
        options=mlp_options, state=mlp_state)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "MLP benchmark fit failed"
    fit_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    call mlp%predict_proba(x, mlp_probabilities, status)
    call mlp%predict(x, mlp_labels, status)
    if (.not. status_ok(status)) error stop "MLP benchmark prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call mlp%predict_proba(x, mlp_probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16)') "mlp_classifier_fit,", &
        n_samples, ",", n_features, ",", n_hidden, ",", n_classes, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16)') "mlp_classifier_predict,", &
        n_samples, ",", n_features, ",", n_hidden, ",", n_classes, ",", predict_seconds

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_samples
        write (unit, '(a,i0,a,i0,a,i0)') "label,", i, ",1,", labels(i)
        write (unit, '(a,i0,a,i0,a,i0)') "softmax_prediction,", i, ",1,", &
            softmax_labels(i)
        write (unit, '(a,i0,a,i0,a,i0)') "mlp_prediction,", i, ",1,", mlp_labels(i)
        do j = 1, n_classes
            write (unit, '(a,i0,a,i0,a,es24.16)') "softmax_probability,", i, &
                ",", j, ",", softmax_probabilities(i, j)
            write (unit, '(a,i0,a,i0,a,es24.16)') "mlp_probability,", i, ",", &
                j, ",", mlp_probabilities(i, j)
        end do
    end do
    close (unit)

contains

    subroutine make_fixture(x, labels)
        real(dp), intent(out) :: x(:, :)
        integer, intent(out) :: labels(:)
        real(dp) :: phase, score(3), bias(3)
        integer :: i, j, class_index

        do i = 1, size(x, 1)
            phase = real(i, dp)
            do j = 1, size(x, 2)
                x(i, j) = sin(0.017_dp*phase + 0.071_dp*real(j, dp)) + &
                    0.2_dp*cos(0.009_dp*phase*real(j, dp))
            end do
            score = [0.4_dp*x(i, 1) - 0.2_dp*x(i, 2) + 0.1_dp*x(i, 3), &
                -0.1_dp*x(i, 1) + 0.5_dp*x(i, 2) - 0.2_dp*x(i, 4), &
                0.2_dp*x(i, 3) + 0.3_dp*x(i, 5) - 0.4_dp*x(i, 6)]
            bias = [0.3_dp*sin(0.11_dp*phase), 0.3_dp*cos(0.11_dp*phase), &
                0.3_dp*sin(0.13_dp*phase + 1.0_dp)]
            class_index = maxloc(score + bias, dim=1)
            labels(i) = class_labels(class_index)
        end do
    end subroutine make_fixture

end program fortml_bench_classifiers
