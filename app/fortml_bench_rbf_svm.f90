program fortml_bench_rbf_svm
    !! Correctness-gated dense RBF binary SVM workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_rbf_svm_classifier, only: rbf_svm_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 36, n_features = 2
    integer, parameter :: prediction_repetitions = 64
    real(dp) :: x(n_samples, n_features), scores(n_samples)
    real(dp) :: sample_weight(n_samples), coefficients(n_samples)
    integer :: labels(n_samples), predicted(n_samples), classes(2)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(rbf_svm_classifier_t) :: model

    call get_environment_variable("FORTML_BENCH_RBF_SVM_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_RBF_SVM_ORACLE is required"
    end if
    call make_fixture(x, labels, sample_weight)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, c=2.0_dp, gamma=0.6_dp, &
        max_iterations=10000, tolerance=1.0e-9_dp, sample_weight=sample_weight)
    call system_clock(clock_end)
    if (.not. status_ok(status)) then
        write (*, '(a)') "RBF SVM benchmark fit status: "//trim(status%msg)
        error stop "RBF SVM benchmark fit failed"
    end if
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%decision_function(x, scores, status)
    call model%predict(x, predicted, status)
    coefficients = model%coefficients()
    classes = model%classes()
    if (.not. status_ok(status)) error stop "RBF SVM benchmark prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%decision_function(x, scores, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16)') "rbf_svm_fit,", n_samples, ",", &
        n_features, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,es24.16)') "rbf_svm_predict,", n_samples, ",", &
        n_features, ",", predict_seconds

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_samples
        write (unit, '(a,i0,a,i0)') "label,", i, ",1,", labels(i)
        write (unit, '(a,i0,a,es24.16)') "score,", i, ",1,", scores(i)
        write (unit, '(a,i0,a,i0)') "prediction,", i, ",1,", predicted(i)
        write (unit, '(a,i0,a,es24.16)') "coefficient,", i, ",1,", coefficients(i)
    end do
    write (unit, '(a,i0,a,es24.16)') "gamma,1,1,", model%gamma()
    write (unit, '(a,i0,a,es24.16)') "intercept,1,1,", model%intercept()
    do j = 1, 2
        write (unit, '(a,i0,a,i0)') "class,", j, ",1,", classes(j)
    end do
    close (unit)

contains

    subroutine make_fixture(x, labels, sample_weight)
        real(dp), intent(out) :: x(:, :), sample_weight(:)
        integer, intent(out) :: labels(:)
        real(dp) :: phase
        integer :: i

        do i = 1, size(x, 1)
            phase = real(i, dp)
            if (i <= size(x, 1)/2) then
                x(i, 1) = -1.0_dp + 0.05_dp*sin(0.17_dp*phase)
                x(i, 2) = 0.2_dp*cos(0.13_dp*phase)
                labels(i) = -12
            else
                x(i, 1) = 1.0_dp + 0.05_dp*sin(0.17_dp*phase)
                x(i, 2) = -0.2_dp*cos(0.13_dp*phase)
                labels(i) = 37
            end if
            sample_weight(i) = 0.75_dp + 0.5_dp*real(mod(i, 7), dp)/6.0_dp
        end do
    end subroutine make_fixture

end program fortml_bench_rbf_svm
