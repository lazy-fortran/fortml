program fortml_bench_discriminant_analysis
    !! Correctness-gated weighted LDA/QDA workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_discriminant_analysis, only: lda_classifier_t, qda_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 120, n_features = 3, n_query = 9, n_classes = 3
    integer, parameter :: prediction_repetitions = 128
    real(dp) :: x(n_samples, n_features), query(n_query, n_features)
    real(dp) :: weights(n_samples), probabilities(n_query, n_classes)
    real(dp) :: means(n_features, n_classes)
    real(dp) :: covariance(n_features, n_features)
    real(dp) :: tangent(n_query, n_features), tangent_output(n_query, n_classes)
    integer :: labels(n_samples), predicted(n_query), classes(n_classes)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds, jvp_seconds
    character(len=1024) :: output_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(lda_classifier_t) :: lda
    type(qda_classifier_t) :: qda

    call get_environment_variable("FORTML_BENCH_DISCRIMINANT_OUTPUT", output_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(output_path) == 0) then
        error stop "FORTML_BENCH_DISCRIMINANT_OUTPUT is required"
    end if
    call make_fixture(x, labels, weights, query, tangent)
    open (newunit=unit, file=trim(output_path), status="replace", action="write")
    write (unit, '(a)') "model,quantity,row,column,value"

    call time_lda(lda, x, labels, weights, query, tangent, fit_seconds, &
        predict_seconds, jvp_seconds, probabilities, predicted, classes, status)
    if (.not. status_ok(status)) error stop "LDA benchmark failed"
    call write_outputs(unit, "lda", probabilities, predicted, classes)
    means = lda%means()
    do i = 1, n_classes
        do j = 1, n_features
            write (unit, '(a,",mean,",i0,",",i0,",",es24.16)') "lda", i, j, means(j, i)
        end do
    end do
    covariance = lda%covariance()
    do i = 1, n_features
        do j = 1, n_features
            write (unit, '(a,",covariance,",i0,",",i0,",",es24.16)') "lda", i, j, covariance(i, j)
        end do
    end do
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
        "lda_fit_predict,", n_samples, ",", n_features, ",", fit_seconds, ",", &
        predict_seconds, ",", jvp_seconds, ",", sum(probabilities)

    call time_qda(qda, x, labels, weights, query, tangent, fit_seconds, &
        predict_seconds, jvp_seconds, probabilities, predicted, classes, status)
    if (.not. status_ok(status)) error stop "QDA benchmark failed"
    call write_outputs(unit, "qda", probabilities, predicted, classes)
    write (*, '(a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
        "qda_fit_predict,", n_samples, ",", n_features, ",", fit_seconds, ",", &
        predict_seconds, ",", jvp_seconds, ",", sum(probabilities)
    close (unit)

contains

    subroutine time_lda(model, x, labels, weights, query, tangent, fit_seconds, &
            predict_seconds, jvp_seconds, probabilities, predicted, classes, status)
        type(lda_classifier_t), intent(out) :: model
        real(dp), intent(in) :: x(:, :), weights(:), query(:, :), tangent(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: fit_seconds, predict_seconds, jvp_seconds
        real(dp), intent(out) :: probabilities(:, :)
        integer, intent(out) :: predicted(:), classes(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: tangent_output(size(query, 1), size(classes))
        integer :: repetition

        call system_clock(clock_start, clock_rate)
        call model%fit(x, labels, status, reg_param=0.03_dp, sample_weight=weights)
        call system_clock(clock_end)
        fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
        if (.not. status_ok(status)) return
        call model%predict_proba(query, probabilities, status)
        call model%predict(query, predicted, status)
        classes = model%classes()
        if (.not. status_ok(status)) return
        call system_clock(clock_start, clock_rate)
        do repetition = 1, prediction_repetitions
            call model%predict_proba(query, probabilities, status)
        end do
        call system_clock(clock_end)
        predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
            /real(prediction_repetitions, dp)
        call model%predict_proba_jvp(query, tangent, probabilities, tangent_output, status)
        call system_clock(clock_start, clock_rate)
        do repetition = 1, prediction_repetitions
            call model%predict_proba_jvp(query, tangent, probabilities, tangent_output, status)
        end do
        call system_clock(clock_end)
        jvp_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
            /real(prediction_repetitions, dp)
    end subroutine time_lda

    subroutine time_qda(model, x, labels, weights, query, tangent, fit_seconds, &
            predict_seconds, jvp_seconds, probabilities, predicted, classes, status)
        type(qda_classifier_t), intent(out) :: model
        real(dp), intent(in) :: x(:, :), weights(:), query(:, :), tangent(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: fit_seconds, predict_seconds, jvp_seconds
        real(dp), intent(out) :: probabilities(:, :)
        integer, intent(out) :: predicted(:), classes(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: tangent_output(size(query, 1), size(classes))
        integer :: repetition

        call system_clock(clock_start, clock_rate)
        call model%fit(x, labels, status, reg_param=0.03_dp, sample_weight=weights)
        call system_clock(clock_end)
        fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
        if (.not. status_ok(status)) return
        call model%predict_proba(query, probabilities, status)
        call model%predict(query, predicted, status)
        classes = model%classes()
        if (.not. status_ok(status)) return
        call system_clock(clock_start, clock_rate)
        do repetition = 1, prediction_repetitions
            call model%predict_proba(query, probabilities, status)
        end do
        call system_clock(clock_end)
        predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
            /real(prediction_repetitions, dp)
        call model%predict_proba_jvp(query, tangent, probabilities, tangent_output, status)
        call system_clock(clock_start, clock_rate)
        do repetition = 1, prediction_repetitions
            call model%predict_proba_jvp(query, tangent, probabilities, tangent_output, status)
        end do
        call system_clock(clock_end)
        jvp_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
            /real(prediction_repetitions, dp)
    end subroutine time_qda

    subroutine write_outputs(unit, name, probabilities, predicted, classes)
        integer, intent(in) :: unit
        character(*), intent(in) :: name
        real(dp), intent(in) :: probabilities(:, :)
        integer, intent(in) :: predicted(:), classes(:)
        integer :: i, j

        do i = 1, size(predicted)
            write (unit, '(a,",prediction,",i0,",1,",i0)') trim(name), i, predicted(i)
            do j = 1, size(probabilities, 2)
                write (unit, '(a,",probability,",i0,",",i0,",",es24.16)') trim(name), &
                    i, j, probabilities(i, j)
            end do
        end do
        do j = 1, size(classes)
            write (unit, '(a,",class,",i0,",1,",i0)') trim(name), j, classes(j)
        end do
    end subroutine write_outputs

    subroutine make_fixture(x, labels, weights, query, tangent)
        real(dp), intent(out) :: x(:, :), weights(:), query(:, :), tangent(:, :)
        integer, intent(out) :: labels(:)
        real(dp) :: phase, score
        integer :: i, j

        do i = 1, size(x, 1)
            phase = real(i, dp)
            do j = 1, size(x, 2)
                x(i, j) = sin(0.031_dp*phase + 0.19_dp*real(j, dp)) + &
                    0.11_dp*cos(0.017_dp*phase*real(j, dp))
            end do
            score = 0.8_dp*x(i, 1) - 0.45_dp*x(i, 2) + 0.25_dp*x(i, 3)
            if (score < -0.25_dp) then
                labels(i) = -17
            else if (score < 0.28_dp) then
                labels(i) = 4
            else
                labels(i) = 23
            end if
            weights(i) = 0.7_dp + 0.6_dp*real(mod(i, 7), dp)/6.0_dp
        end do
        do i = 1, size(query, 1)
            query(i, :) = [ -1.1_dp + 2.2_dp*real(i-1, dp)/real(size(query, 1)-1, dp), &
                sin(0.4_dp*real(i, dp)), cos(0.3_dp*real(i, dp)) ]
            tangent(i, :) = [ 0.1_dp*cos(0.2_dp*real(i, dp)), &
                -0.1_dp*sin(0.3_dp*real(i, dp)), 0.04_dp ]
        end do
    end subroutine make_fixture

end program fortml_bench_discriminant_analysis
