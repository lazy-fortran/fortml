program fortml_bench_knn
    !! Release workload for deterministic uniform and inverse-distance kNN.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_knn_classifier, only: knn_classifier_t, KNN_WEIGHTS_UNIFORM, &
        KNN_WEIGHTS_DISTANCE
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_train = 96, n_query = 48, n_features = 3
    integer, parameter :: n_classes = 3, n_neighbors = 7, repetitions = 32
    real(dp) :: x(n_train, n_features), query(n_query, n_features)
    real(dp) :: probabilities(n_query, n_classes), elapsed, accuracy
    real(dp) :: probability_norm, prediction_sum
    integer :: labels(n_train), query_labels(n_query), predicted(n_query)
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, repetition
    type(knn_classifier_t) :: model
    type(fortnum_status_t) :: status

    call make_fixture(x, labels, query, query_labels)
    call model%fit(x, labels, status, n_neighbors=n_neighbors, &
        weights=KNN_WEIGHTS_UNIFORM)
    if (.not. status_ok(status)) error stop "kNN uniform fit failed"
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predicted, status)
    if (.not. status_ok(status)) error stop "kNN uniform prediction failed"
    accuracy = real(count(predicted == query_labels), dp)/real(n_query, dp)
    probability_norm = sum(probabilities*probabilities)
    prediction_sum = sum(real(predicted, dp))
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%predict_proba(query, probabilities, status)
        if (.not. status_ok(status)) error stop "kNN uniform timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
        "knn_uniform,", n_query, ",", n_features, ",", n_neighbors, ",", &
        elapsed, ",", accuracy, ",", sum(probabilities), ",", probability_norm, ",", &
        prediction_sum

    call model%fit(x, labels, status, n_neighbors=n_neighbors, &
        weights=KNN_WEIGHTS_DISTANCE)
    if (.not. status_ok(status)) error stop "kNN distance fit failed"
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predicted, status)
    if (.not. status_ok(status)) error stop "kNN distance prediction failed"
    accuracy = real(count(predicted == query_labels), dp)/real(n_query, dp)
    probability_norm = sum(probabilities*probabilities)
    prediction_sum = sum(real(predicted, dp))
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%predict_proba(query, probabilities, status)
        if (.not. status_ok(status)) error stop "kNN distance timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
        "knn_distance,", n_query, ",", n_features, ",", n_neighbors, ",", &
        elapsed, ",", accuracy, ",", sum(probabilities), ",", probability_norm, ",", &
        prediction_sum

contains

    subroutine make_fixture(train, train_labels, points, point_labels)
        real(dp), intent(out) :: train(:, :), points(:, :)
        integer, intent(out) :: train_labels(:), point_labels(:)
        integer :: i, j, class_index

        do i = 1, size(train, 1)
            class_index = 1 + mod(i - 1, n_classes)
            train_labels(i) = 10*class_index - 7
            do j = 1, size(train, 2)
                train(i, j) = 1.7_dp*real(class_index - 2, dp) + &
                    0.13_dp*sin(0.17_dp*real(i*j, dp)) + &
                    0.04_dp*cos(0.11_dp*real(i + 2*j, dp))
            end do
        end do
        do i = 1, size(points, 1)
            class_index = 1 + mod(i + 1, n_classes)
            point_labels(i) = 10*class_index - 7
            do j = 1, size(points, 2)
                points(i, j) = 1.7_dp*real(class_index - 2, dp) + &
                    0.09_dp*cos(0.19_dp*real(i*j, dp)) + &
                    0.02_dp*sin(0.07_dp*real(i + j, dp))
            end do
        end do
    end subroutine make_fixture

end program fortml_bench_knn
