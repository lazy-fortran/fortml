program fortml_bench_naive_bayes_partial_fit
    !! Release workload for the transactional partial-fit NB family.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_bernoulli_naive_bayes, only: bernoulli_naive_bayes_t
    use fortml_multinomial_naive_bayes, only: multinomial_naive_bayes_t
    use fortml_complement_naive_bayes, only: complement_naive_bayes_t
    use fortml_categorical_naive_bayes, only: categorical_naive_bayes_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    real(dp) :: xb(6, 2), xm(6, 2), query(2, 2)
    integer :: xc(6, 2), labels(6), classes(3)
    type(bernoulli_naive_bayes_t) :: bernoulli
    type(multinomial_naive_bayes_t) :: multinomial
    type(complement_naive_bayes_t) :: complement
    type(categorical_naive_bayes_t) :: categorical
    type(fortml_device_t) :: cuda

    xb = reshape([1.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, &
                  1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp], shape(xb))
    xm = reshape([8.0_dp, 0.0_dp, 10.0_dp, 2.0_dp, 4.0_dp, 6.0_dp, &
                  4.0_dp, 0.0_dp, 6.0_dp, 2.0_dp, 1.0_dp, 3.0_dp], shape(xm))
    xc = reshape([1, 2, 1, 1, 2, 3, 2, 1, 2, 3, 1, 3], shape(xc))
    labels = [9, -3, 9, -3, 4, 4]; classes = [-3, 4, 9]
    query = reshape([1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp], shape(query))
    cuda%kind = FORTML_DEVICE_CUDA; cuda%selected = .true.; cuda%available = .true.

    call time_bernoulli(bernoulli, xb, labels, classes, cuda)
    call time_multinomial(multinomial, xm, labels, classes, query, cuda)
    call time_complement(complement, xm, labels, classes, query, cuda)
    call time_categorical(categorical, xc, labels, classes, cuda)

contains

    subroutine time_bernoulli(model, x, y, known_classes, device)
        type(bernoulli_naive_bayes_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: y(:), known_classes(:)
        type(fortml_device_t), intent(in) :: device
        integer(int64) :: start, finish, rate
        real(dp) :: seconds
        type(fortnum_status_t) :: local_status
        call model%partial_fit(x(:3, :), y(:3), local_status, classes=known_classes)
        if (.not. status_ok(local_status)) error stop "Bernoulli prefix failed"
        call system_clock(start, rate)
        call model%partial_fit(x(4:, :), y(4:), local_status)
        call system_clock(finish)
        seconds = real(finish-start, dp)/real(rate, dp)
        if (.not. status_ok(local_status)) error stop "Bernoulli suffix failed"
        call model%partial_fit_device(device, x(1:1, :), [9], local_status)
        write (*, '(a,",",es24.16,",",i0,",",i0,",",i0)') &
            "bernoulli", seconds, model%batch_count(), 0, local_status%code
    end subroutine time_bernoulli

    subroutine time_multinomial(model, x, y, known_classes, q, device)
        type(multinomial_naive_bayes_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), q(:, :)
        integer, intent(in) :: y(:), known_classes(:)
        type(fortml_device_t), intent(in) :: device
        integer(int64) :: start, finish, rate
        real(dp) :: seconds, probabilities(2, 3)
        type(fortnum_status_t) :: local_status
        call model%partial_fit(x(:3, :), y(:3), local_status, classes=known_classes)
        if (.not. status_ok(local_status)) error stop "Multinomial prefix failed"
        call system_clock(start, rate)
        call model%partial_fit(x(4:, :), y(4:), local_status)
        call system_clock(finish)
        seconds = real(finish-start, dp)/real(rate, dp)
        if (.not. status_ok(local_status)) error stop "Multinomial suffix failed"
        call model%predict_proba(q, probabilities, local_status)
        call model%partial_fit_device(device, x(1:1, :), [9], local_status)
        write (*, '(a,",",es24.16,",",i0,",",i0,",",i0)') &
            "multinomial", seconds, model%batch_count(), 0, local_status%code
    end subroutine time_multinomial

    subroutine time_complement(model, x, y, known_classes, q, device)
        type(complement_naive_bayes_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), q(:, :)
        integer, intent(in) :: y(:), known_classes(:)
        type(fortml_device_t), intent(in) :: device
        integer(int64) :: start, finish, rate
        real(dp) :: seconds, probabilities(2, 3)
        type(fortnum_status_t) :: local_status
        call model%partial_fit(x(:3, :), y(:3), local_status, classes=known_classes)
        if (.not. status_ok(local_status)) error stop "Complement prefix failed"
        call system_clock(start, rate)
        call model%partial_fit(x(4:, :), y(4:), local_status)
        call system_clock(finish)
        seconds = real(finish-start, dp)/real(rate, dp)
        if (.not. status_ok(local_status)) error stop "Complement suffix failed"
        call model%predict_proba(q, probabilities, local_status)
        call model%partial_fit_device(device, x(1:1, :), [9], local_status)
        write (*, '(a,",",es24.16,",",i0,",",i0,",",i0)') &
            "complement", seconds, model%batch_count(), 0, local_status%code
    end subroutine time_complement

    subroutine time_categorical(model, x, y, known_classes, device)
        type(categorical_naive_bayes_t), intent(inout) :: model
        integer, intent(in) :: x(:, :), y(:), known_classes(:)
        type(fortml_device_t), intent(in) :: device
        integer(int64) :: start, finish, rate
        real(dp) :: seconds
        type(fortnum_status_t) :: local_status
        call model%partial_fit(x(:3, :), y(:3), local_status, classes=known_classes)
        if (.not. status_ok(local_status)) error stop "Categorical prefix failed"
        call system_clock(start, rate)
        call model%partial_fit(x(4:, :), y(4:), local_status)
        call system_clock(finish)
        seconds = real(finish-start, dp)/real(rate, dp)
        if (.not. status_ok(local_status)) error stop "Categorical suffix failed"
        call model%partial_fit_device(device, x(1:1, :), [9], local_status)
        write (*, '(a,",",es24.16,",",i0,",",i0,",",i0)') &
            "categorical", seconds, model%batch_count(), 0, local_status%code
    end subroutine time_categorical

end program fortml_bench_naive_bayes_partial_fit
