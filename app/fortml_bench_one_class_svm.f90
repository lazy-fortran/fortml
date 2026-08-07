program fortml_bench_one_class_svm
    !! Release workload for the dense RBF one-class SVM.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_one_class_svm, only: one_class_svm_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 48, n_features = 2, n_query = 4
    integer, parameter :: prediction_repetitions = 64
    real(dp), parameter :: nu_value = 0.5_dp, gamma_value = 0.8_dp
    real(dp), parameter :: fit_tolerance = 1.0e-10_dp
    real(dp) :: x(n_samples, n_features), query(n_query, n_features)
    real(dp) :: scores(n_query)
    integer :: labels(n_query)
    real(dp), allocatable :: weights(:)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: output_path
    integer :: environment_status, unit, i, repetition
    type(fortnum_status_t) :: status
    type(one_class_svm_t) :: model

    call get_environment_variable("FORTML_BENCH_ONE_CLASS_SVM_OUTPUT", &
        output_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(output_path) == 0) then
        error stop "FORTML_BENCH_ONE_CLASS_SVM_OUTPUT is required"
    end if

    call make_fixture(x, query)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, status, nu=nu_value, gamma=gamma_value, &
        max_iterations=5000, tolerance=fit_tolerance)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "one-class SVM fit failed"
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)

    call model%decision_function(query, scores, status)
    if (.not. status_ok(status)) error stop "one-class SVM prediction failed"
    call model%predict(query, labels, status)
    if (.not. status_ok(status)) error stop "one-class SVM labels failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%decision_function(query, scores, status)
        if (.not. status_ok(status)) error stop "one-class SVM timing failed"
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)

    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "one_class_svm_fit,", n_samples, ",", n_features, ",", n_query, ",", &
        fit_seconds
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "one_class_svm_predict,", n_samples, ",", n_features, ",", n_query, ",", &
        predict_seconds

    weights = model%support_weights()
    open (newunit=unit, file=trim(output_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_samples
        write (unit, '(a,i0,a,es24.16)') "weight,", i, ",1,", weights(i)
    end do
    write (unit, '(a,i0,a,es24.16)') "offset,", 1, ",1,", model%offset()
    do i = 1, n_query
        write (unit, '(a,i0,a,es24.16)') "score,", i, ",1,", scores(i)
        write (unit, '(a,i0,a,i0)') "prediction,", i, ",1,", labels(i)
    end do
    close (unit)

contains

    subroutine make_fixture(train, points)
        real(dp), intent(out) :: train(:, :), points(:, :)
        real(dp) :: angle, radius
        integer :: i

        do i = 1, size(train, 1)
            angle = 2.0_dp*acos(-1.0_dp)*real(i-1, dp)/real(size(train, 1), dp)
            radius = 1.0_dp + 0.08_dp*sin(3.0_dp*angle)
            train(i, :) = [radius*cos(angle), radius*sin(angle)]
        end do
        points(1, :) = [1.0_dp, 0.0_dp]
        points(2, :) = [0.0_dp, 0.0_dp]
        points(3, :) = [1.8_dp, 1.2_dp]
        points(4, :) = [-1.1_dp, 0.2_dp]
    end subroutine make_fixture

end program fortml_bench_one_class_svm
