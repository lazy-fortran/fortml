program fortml_bench_ovo_classifier
    !! Correctness-gated one-vs-one logistic classifier workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_ovo_logistic_classifier, only: ovo_logistic_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 256, n_features = 6, n_classes = 4
    integer, parameter :: n_pairs = 6, prediction_repetitions = 64
    integer, parameter :: class_labels(4) = [-7, 3, 11, 42]
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, n_classes)
    integer :: labels(n_samples), predicted(n_samples)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(ovo_logistic_classifier_t) :: model

    call get_environment_variable("FORTML_BENCH_OVO_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_OVO_ORACLE is required"
    end if
    call make_fixture(x, labels)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, l2=5.0e-2_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "OVO benchmark fit failed"
    fit_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    if (.not. status_ok(status)) error stop "OVO benchmark prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16)') "ovo_logistic_fit,", &
        n_samples, ",", n_features, ",", n_classes, ",", n_pairs, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16)') "ovo_logistic_predict,", &
        n_samples, ",", n_features, ",", n_classes, ",", n_pairs, ",", predict_seconds

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_samples
        write (unit, '(a,i0,a,i0,a,i0)') "label,", i, ",1,", labels(i)
        write (unit, '(a,i0,a,i0,a,i0)') "prediction,", i, ",1,", predicted(i)
        do j = 1, n_classes
            write (unit, '(a,i0,a,i0,a,es24.16)') "probability,", i, ",", &
                j, ",", probabilities(i, j)
        end do
    end do
    close (unit)

contains

    subroutine make_fixture(x, labels)
        real(dp), intent(out) :: x(:, :)
        integer, intent(out) :: labels(:)
        real(dp) :: phase, score(n_classes), bias(n_classes)
        integer :: i, j, class_index

        do i = 1, size(x, 1)
            phase = real(i, dp)
            do j = 1, size(x, 2)
                x(i, j) = sin(0.017_dp*phase + 0.071_dp*real(j, dp)) + &
                    0.2_dp*cos(0.009_dp*phase*real(j, dp))
            end do
            score = [ &
                0.4_dp*x(i, 1) - 0.2_dp*x(i, 2) + 0.1_dp*x(i, 3), &
                -0.1_dp*x(i, 1) + 0.5_dp*x(i, 2) - 0.2_dp*x(i, 4), &
                0.2_dp*x(i, 3) + 0.3_dp*x(i, 5) - 0.4_dp*x(i, 6), &
                -0.3_dp*x(i, 1) + 0.2_dp*x(i, 4) + 0.4_dp*x(i, 6)]
            bias = [0.3_dp*sin(0.11_dp*phase), 0.3_dp*cos(0.11_dp*phase), &
                0.3_dp*sin(0.13_dp*phase + 1.0_dp), &
                0.2_dp*cos(0.07_dp*phase + 0.4_dp)]
            class_index = maxloc(score + bias, dim=1)
            labels(i) = class_labels(class_index)
        end do
    end subroutine make_fixture

end program fortml_bench_ovo_classifier
