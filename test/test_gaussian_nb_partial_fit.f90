program test_gaussian_nb_partial_fit
    !! Independent sufficient-statistic/replay oracle for GaussianNB streams.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_gaussian_naive_bayes, only: gaussian_naive_bayes_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gaussian_naive_bayes_t) :: streamed, reference, before, device_model, invalid
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 2), first_x(3, 2), second_x(3, 2), query(2, 2)
    real(dp) :: probabilities(2, 3), reference_probabilities(2, 3)
    real(dp) :: before_probabilities(2, 3)
    real(dp) :: expected_mean(2, 3), expected_variance(2, 3)
    real(dp) :: expected_prior(3), means(2, 3), variances(2, 3), prior(3)
    integer :: labels(6), first_labels(3), second_labels(3), classes(3)
    integer :: bad_labels(2), failures
    integer :: invalid_labels(3)
    real(dp) :: smoothing_before

    x(:, 1) = [8.0_dp, 0.0_dp, 10.0_dp, 2.0_dp, 4.0_dp, 6.0_dp]
    x(:, 2) = [4.0_dp, 0.0_dp, 6.0_dp, 2.0_dp, 1.0_dp, 3.0_dp]
    labels = [9, -3, 9, -3, 4, 4]
    first_x = x(:3, :)
    first_labels = labels(:3)
    second_x = x(4:, :)
    second_labels = labels(4:)
    classes = [-3, 4, 9]
    query(1, :) = [1.0_dp, 1.0_dp]
    query(2, :) = [5.0_dp, 2.0_dp]
    failures = 0

    smoothing_before = invalid%var_smoothing_value()
    invalid_labels = [9, 999, 9]
    call invalid%partial_fit(first_x, invalid_labels, status, classes=classes, &
        var_smoothing=0.0_dp)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. &
        .not. invalid%partial_fit_initialized() .and. invalid%sample_count() == 0 .and. &
        invalid%var_smoothing_value() == smoothing_before, &
        "invalid first batch leaves stream configuration untouched", failures)

    call streamed%partial_fit(first_x, first_labels, status, classes=classes, &
        var_smoothing=0.0_dp)
    call check(status_ok(status) .and. streamed%partial_fit_initialized() .and. &
        .not. streamed%fitted() .and. streamed%sample_count() == 3 .and. &
        streamed%batch_count() == 1 .and. all(streamed%classes() == classes), &
        "first one-class batch is deferred with stable class metadata", failures)

    call streamed%partial_fit(second_x, second_labels, status)
    call check(status_ok(status) .and. streamed%fitted() .and. &
        streamed%sample_count() == 6 .and. streamed%batch_count() == 2, &
        "second batch completes all declared classes", failures)

    ! Independent weighted-moment oracle (population variance, no smoothing).
    expected_mean = reshape([1.0_dp, 1.0_dp, 5.0_dp, 2.0_dp, 9.0_dp, 5.0_dp], &
        shape(expected_mean))
    expected_variance = 1.0_dp
    expected_prior = 1.0_dp/3.0_dp
    means = streamed%means()
    variances = streamed%variances()
    prior = streamed%class_prior()
    call check(maxval(abs(means - expected_mean)) < 2.0e-14_dp .and. &
        maxval(abs(variances - expected_variance)) < 2.0e-14_dp .and. &
        maxval(abs(prior - expected_prior)) < 2.0e-14_dp, &
        "streamed moments and priors match analytic oracle", failures)

    call reference%fit(x, labels, status, var_smoothing=0.0_dp)
    call streamed%predict_proba(query, probabilities, status)
    call reference%predict_proba(query, reference_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(probabilities - &
        reference_probabilities)) < 2.0e-14_dp, &
        "streamed prediction matches independent one-shot replay", failures)

    before = streamed
    before_probabilities = probabilities
    bad_labels = [9, 999]
    call streamed%partial_fit(query, bad_labels, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. &
        streamed%sample_count() == 6 .and. streamed%batch_count() == 2, &
        "unknown labels are rejected", failures)
    call streamed%predict_proba(query, probabilities, status)
    call before%predict_proba(query, before_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(probabilities - &
        before_probabilities)) == 0.0_dp, &
        "malformed batch leaves fitted state unchanged", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call check(status_ok(status), "CPU device selection", failures)
    call device_model%partial_fit_device(cpu, first_x, first_labels, status, &
        classes=classes, var_smoothing=0.0_dp)
    call check(status_ok(status) .and. .not. device_model%fitted(), &
        "CPU device path uses exact replay", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call device_model%partial_fit_device(cuda, second_x, second_labels, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        device_model%sample_count() == 3, &
        "CUDA path is a typed resident-state refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GaussianNB partial-fit cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GaussianNB partial-fit independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gaussian-nb-partial] "//description
        end if
    end subroutine check

end program test_gaussian_nb_partial_fit
