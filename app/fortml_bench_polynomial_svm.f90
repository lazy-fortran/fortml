program fortml_bench_polynomial_svm
    !! Correctness-gated dense polynomial binary SVM workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_polynomial_svm_classifier, only: polynomial_svm_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 8, n_features = 2
    integer, parameter :: prediction_repetitions = 64
    real(dp) :: x(n_samples, n_features), scores(n_samples), probabilities(n_samples, 2)
    real(dp) :: sample_weight(n_samples), coefficients(n_samples)
    integer :: labels(n_samples), predicted(n_samples), classes(2)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(polynomial_svm_classifier_t) :: model

    call get_environment_variable("FORTML_BENCH_POLYNOMIAL_SVM_ORACLE", oracle_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) error stop "FORTML_BENCH_POLYNOMIAL_SVM_ORACLE is required"
    call make_fixture(x, labels, sample_weight)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, c=3.0_dp, gamma=0.4_dp, degree=2, coef0=1.0_dp, &
        max_iterations=50000, tolerance=1.0e-6_dp, sample_weight=sample_weight)
    call system_clock(clock_end)
    if (.not. status_ok(status)) then
        write (*, '(a)') "polynomial SVM benchmark fit status: "//trim(status%msg)
        error stop "polynomial SVM benchmark fit failed"
    end if
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%decision_function(x, scores, status)
    call model%predict(x, predicted, status)
    call model%predict_proba(x, probabilities, status)
    coefficients = model%coefficients(); classes = model%classes()
    if (.not. status_ok(status)) error stop "polynomial SVM benchmark prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%decision_function(x, scores, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16)') "polynomial_svm_fit,", n_samples, ",", n_features, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,es24.16)') "polynomial_svm_predict,", n_samples, ",", n_features, ",", predict_seconds

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_samples
        write (unit, '(a,i0,a,i0)') "label,", i, ",1,", labels(i)
        write (unit, '(a,i0,a,es24.16)') "score,", i, ",1,", scores(i)
        write (unit, '(a,i0,a,i0)') "prediction,", i, ",1,", predicted(i)
        write (unit, '(a,i0,a,es24.16)') "coefficient,", i, ",1,", coefficients(i)
    end do
    write (unit, '(a,es24.16)') "gamma,1,1,", model%gamma()
    write (unit, '(a,es24.16)') "coef0,1,1,", model%coef0()
    write (unit, '(a,i0)') "degree,1,1,", model%degree()
    write (unit, '(a,es24.16)') "intercept,1,1,", model%intercept()
    do j = 1, 2
        write (unit, '(a,i0,a,i0)') "class,", j, ",1,", classes(j)
    end do
    close (unit)

contains
    subroutine make_fixture(x, labels, sample_weight)
        real(dp), intent(out) :: x(:, :), sample_weight(:)
        integer, intent(out) :: labels(:)
        real(dp), parameter :: first_x(4) = [-1.0_dp, -0.8_dp, 1.0_dp, 0.8_dp]
        real(dp), parameter :: first_y(4) = [-1.0_dp, -1.1_dp, 1.0_dp, 1.1_dp]
        real(dp), parameter :: second_y(4) = [1.0_dp, 1.1_dp, -1.0_dp, -1.1_dp]
        integer :: i
        do i = 1, size(x, 1)
            if (i <= 4) then
                x(i, 1) = first_x(i); x(i, 2) = first_y(i)
                labels(i) = -12
            else
                x(i, 1) = first_x(i-4); x(i, 2) = second_y(i-4)
                labels(i) = 37
            end if
            sample_weight(i) = 1.0_dp
        end do
    end subroutine make_fixture
end program fortml_bench_polynomial_svm
