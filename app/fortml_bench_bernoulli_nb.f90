program fortml_bench_bernoulli_nb
    !! Release workload for the relaxed Bernoulli Naive Bayes classifier.
    !!
    !! The Python harness owns the independent behavioral oracle.  This app
    !! emits complete output arrays when FORTML_BENCH_BERNOULLI_ORACLE is set
    !! and reports fit, prediction, and input-JVP timings.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_bernoulli_naive_bayes, only: bernoulli_naive_bayes_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 512, n_features = 8, n_classes = 3
    integer, parameter :: fit_repetitions = 16, prediction_repetitions = 128
    integer, parameter :: jvp_repetitions = 128
    real(dp), parameter :: alpha = 0.5_dp
    real(dp) :: x(n_samples, n_features), x_dot(n_samples, n_features)
    real(dp) :: log_probabilities(n_samples, n_classes)
    real(dp) :: probabilities(n_samples, n_classes)
    real(dp) :: log_probabilities_dot(n_samples, n_classes)
    integer :: labels(n_samples), predicted(n_samples)
    integer :: i, j, k, bit, class_index
    real(dp) :: score(3), fit_elapsed, prediction_elapsed, jvp_elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    type(bernoulli_naive_bayes_t) :: model
    type(fortnum_status_t) :: status

    call make_fixture(x, labels, x_dot)
    call model%fit(x, labels, status, alpha=alpha)
    if (.not. status_ok(status)) error stop "BernoulliNB benchmark fit failed"
    call model%predict_log_proba(x, log_probabilities, status)
    if (.not. status_ok(status)) error stop "BernoulliNB benchmark log prediction failed"
    call model%predict_proba(x, probabilities, status)
    if (.not. status_ok(status)) error stop "BernoulliNB benchmark prediction failed"
    call model%predict(x, predicted, status)
    if (.not. status_ok(status)) error stop "BernoulliNB benchmark labels failed"
    call model%predict_log_proba_jvp(x, x_dot, log_probabilities, &
        log_probabilities_dot, status)
    if (.not. status_ok(status)) error stop "BernoulliNB benchmark JVP failed"
    if (maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) > 2.0e-14_dp) then
        error stop "BernoulliNB benchmark probability oracle failed"
    end if
    call write_oracle(log_probabilities, probabilities, predicted)
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do k = 1, fit_repetitions
        call model%fit(x, labels, status, alpha=alpha)
        if (.not. status_ok(status)) error stop "BernoulliNB timed fit failed"
    end do
    call system_clock(clock_end)
    fit_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(fit_repetitions, dp)

    call system_clock(clock_start, clock_rate)
    do k = 1, prediction_repetitions
        call model%predict_log_proba(x, log_probabilities, status)
        if (.not. status_ok(status)) error stop "BernoulliNB timed prediction failed"
    end do
    call system_clock(clock_end)
    prediction_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)

    call system_clock(clock_start, clock_rate)
    do k = 1, jvp_repetitions
        call model%predict_log_proba_jvp(x, x_dot, log_probabilities, &
            log_probabilities_dot, status)
        if (.not. status_ok(status)) error stop "BernoulliNB timed JVP failed"
    end do
    call system_clock(clock_end)
    jvp_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(jvp_repetitions, dp)

    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "bernoulli_nb_fit,", n_samples, ",", n_features, ",", n_classes, ",", &
        fit_elapsed
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "bernoulli_nb_predict,", n_samples, ",", n_features, ",", n_classes, ",", &
        prediction_elapsed
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "bernoulli_nb_jvp,", n_samples, ",", n_features, ",", n_classes, ",", &
        jvp_elapsed

contains

    subroutine make_fixture(features, targets, tangent)
        real(dp), intent(out) :: features(:, :), tangent(:, :)
        integer, intent(out) :: targets(:)
        integer :: row, column, value
        integer, parameter :: class_labels(3) = [-7, 3, 11]
        real(dp) :: phase, score(3)

        do column = 1, n_features
            do row = 1, n_samples
                value = mod(7*row + 11*column + row*column, 17)
                bit = merge(1, 0, value >= 8)
                features(row, column) = 0.1_dp + 0.8_dp*real(bit, dp)
                tangent(row, column) = 0.03_dp*sin(0.013_dp*real(row, dp) &
                    + 0.17_dp*real(column, dp))
            end do
        end do
        do row = 1, n_samples
            phase = real(row, dp)
            score(1) = 1.7_dp*features(row, 1) - 0.6_dp*features(row, 2) &
                + 0.2_dp*sin(0.07_dp*phase)
            score(2) = 1.4_dp*features(row, 3) + 0.8_dp*features(row, 4) &
                - 0.4_dp*features(row, 5)
            score(3) = 1.2_dp*features(row, 6) - 0.7_dp*features(row, 7) &
                + 0.3_dp*features(row, 8) + 0.1_dp*cos(0.11_dp*phase)
            class_index = maxloc(score, dim=1)
            targets(row) = class_labels(class_index)
        end do
    end subroutine make_fixture

    subroutine write_oracle(log_probability, probability, prediction)
        real(dp), intent(in) :: log_probability(:, :), probability(:, :)
        integer, intent(in) :: prediction(:)
        character(len=1024) :: path
        integer :: i, j, unit, environment_status

        call get_environment_variable("FORTML_BENCH_BERNOULLI_ORACLE", path, &
            status=environment_status)
        if (environment_status /= 0 .or. len_trim(path) == 0) return
        open (newunit=unit, file=trim(path), status="replace", action="write")
        write (unit, '(a)') "quantity,row,column,value"
        do i = 1, size(prediction)
            write (unit, '(a,i0,a,i0,a,i0)') &
                "prediction,", i, ",1,", prediction(i)
            do j = 1, size(probability, 2)
                write (unit, '(a,i0,a,i0,a,es26.17e3)') &
                    "log_probability,", i, ",", j, ",", log_probability(i, j)
                write (unit, '(a,i0,a,i0,a,es26.17e3)') &
                    "probability,", i, ",", j, ",", probability(i, j)
            end do
        end do
        close (unit)
    end subroutine write_oracle

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=environment_status)
        oracle_only_requested = environment_status == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_bernoulli_nb
