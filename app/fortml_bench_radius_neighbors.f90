program fortml_bench_radius_neighbors
    !! Release workload for dense radius-neighbor classification.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_radius_neighbors_classifier, only: radius_neighbors_classifier_t, &
        RADIUS_WEIGHTS_DISTANCE
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 96, n_features = 2, n_query = 6, n_classes = 3
    integer, parameter :: prediction_repetitions = 256
    real(dp) :: x(n_samples, n_features), query(n_query, n_features)
    real(dp) :: probabilities(n_query, n_classes), sample_weight(n_samples)
    integer :: labels(n_samples), predicted(n_query), classes(n_classes)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: output_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(radius_neighbors_classifier_t) :: model

    call get_environment_variable("FORTML_BENCH_RADIUS_OUTPUT", output_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(output_path) == 0) then
        error stop "FORTML_BENCH_RADIUS_OUTPUT is required"
    end if
    call make_fixture(x, labels, sample_weight, query)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, radius=0.38_dp, &
        weights=RADIUS_WEIGHTS_DISTANCE, sample_weight=sample_weight, &
        outlier_label=3)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "radius fit failed"
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predicted, status)
    classes = model%classes()
    if (.not. status_ok(status)) error stop "radius prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(query, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    write (*, '(a,i0,a,i0,a,es24.16)') "radius_neighbors_fit,", n_samples, ",", &
        n_features, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,es24.16)') "radius_neighbors_predict,", n_query, ",", &
        n_features, ",", predict_seconds

    open (newunit=unit, file=trim(output_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_query
        write (unit, '(a,i0,a,i0,a,i0)') "prediction,", i, ",1,", predicted(i)
        do j = 1, n_classes
            write (unit, '(a,i0,a,i0,a,es24.16)') "probability,", i, ",", j, ",", &
                probabilities(i, j)
        end do
    end do
    do j = 1, n_classes
        write (unit, '(a,i0,a,i0)') "class,", j, ",", classes(j)
    end do
    close (unit)

contains

    subroutine make_fixture(x, labels, sample_weight, query)
        real(dp), intent(out) :: x(:, :), sample_weight(:), query(:, :)
        integer, intent(out) :: labels(:)
        integer :: i
        real(dp) :: a, b
        do i = 1, size(x, 1)
            a = -1.2_dp + 2.4_dp*real(i-1, dp)/real(size(x, 1)-1, dp)
            b = sin(0.19_dp*real(i, dp))
            x(i, :) = [a, b]
            if (a < -0.35_dp) then
                labels(i) = 3
            else if (a < 0.42_dp) then
                labels(i) = 11
            else
                labels(i) = 17
            end if
            sample_weight(i) = 0.8_dp + 0.4_dp*real(mod(i, 5), dp)/4.0_dp
        end do
        query(1, :) = [-1.05_dp, 0.0_dp]
        query(2, :) = [-0.15_dp, 0.35_dp]
        query(3, :) = [0.25_dp, -0.45_dp]
        query(4, :) = [0.85_dp, 0.15_dp]
        query(5, :) = [1.18_dp, -0.2_dp]
        query(6, :) = [0.0_dp, 1.4_dp]
    end subroutine make_fixture

end program fortml_bench_radius_neighbors
