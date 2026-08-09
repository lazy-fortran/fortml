program test_naive_bayes_partial_fit
    !! Independent stream, rollback, and device-boundary oracle for all NB variants.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_bernoulli_naive_bayes, only: bernoulli_naive_bayes_t
    use fortml_multinomial_naive_bayes, only: multinomial_naive_bayes_t
    use fortml_complement_naive_bayes, only: complement_naive_bayes_t
    use fortml_categorical_naive_bayes, only: categorical_naive_bayes_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call check_bernoulli(failures)
    call check_multinomial(failures)
    call check_complement(failures)
    call check_categorical(failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL Naive Bayes partial-fit cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS all Naive Bayes partial-fit independent oracles"

contains

    subroutine check_bernoulli(failures)
        integer, intent(inout) :: failures
        type(bernoulli_naive_bayes_t) :: stream, reference, before
        type(fortml_device_t) :: cuda
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 2), query(2, 2), p(2, 3), p_ref(2, 3), p_before(2, 3)
        real(dp), allocatable :: fp(:, :), prior(:)
        real(dp) :: expected_fp(2, 3)
        integer :: labels(6), classes(3), bad(2), bad_code

        x = reshape([1.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, &
                     1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp], shape(x))
        labels = [9, -3, 9, -3, 4, 4]; classes = [-3, 4, 9]
        query = reshape([1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp], shape(query))
        bad = [9, 99]
        call stream%partial_fit(x(:3, :), labels(:3), status, classes=classes, alpha=1.0_dp)
        call check(status_ok(status) .and. .not. stream%fitted() .and. &
            stream%sample_count() == 3, "Bernoulli deferred first batch", failures)
        call stream%partial_fit(x(4:, :), labels(4:), status)
        call check(status_ok(status) .and. stream%fitted() .and. stream%batch_count() == 2, &
            "Bernoulli completed stream", failures)
        fp = stream%feature_probabilities(); prior = stream%class_prior()
        expected_fp = reshape([0.25_dp, 0.5_dp, 0.5_dp, 0.75_dp, 0.75_dp, 0.5_dp], &
            shape(expected_fp))
        call check(maxval(abs(fp - expected_fp)) < 2.0e-14_dp .and. &
            maxval(abs(prior - 1.0_dp/3.0_dp)) < 2.0e-14_dp, &
            "Bernoulli sufficient-statistic oracle", failures)
        call reference%fit(x, labels, status, alpha=1.0_dp)
        call stream%predict_proba(query, p, status); call reference%predict_proba(query, p_ref, status)
        call check(status_ok(status) .and. maxval(abs(p - p_ref)) < 2.0e-14_dp, &
            "Bernoulli replay equals one-shot probability", failures)
        before = stream; call before%predict_proba(query, p_before, status)
        call stream%partial_fit(query, bad, status)
        bad_code = status%code
        call stream%predict_proba(query, p, status)
        call check(bad_code == FORTNUM_DOMAIN_ERROR .and. stream%sample_count() == 6 .and. &
            maxval(abs(p - p_before)) == 0.0_dp, "Bernoulli rollback", failures)
        cuda%kind = FORTML_DEVICE_CUDA; cuda%selected = .true.; cuda%available = .true.
        call stream%partial_fit_device(cuda, query, [9, 4], status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. stream%sample_count() == 6, &
            "Bernoulli CUDA refusal", failures)
    end subroutine check_bernoulli

    subroutine check_multinomial(failures)
        integer, intent(inout) :: failures
        type(multinomial_naive_bayes_t) :: stream, reference
        type(fortml_device_t) :: cuda
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 2), query(2, 2), p(2, 3), p_ref(2, 3)
        real(dp), allocatable :: fp(:, :), counts(:, :)
        integer :: labels(6), classes(3)

        x = reshape([8.0_dp, 0.0_dp, 10.0_dp, 2.0_dp, 4.0_dp, 6.0_dp, &
                     4.0_dp, 0.0_dp, 6.0_dp, 2.0_dp, 1.0_dp, 3.0_dp], shape(x))
        labels = [9, -3, 9, -3, 4, 4]; classes = [-3, 4, 9]
        query = reshape([1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp], shape(query))
        call stream%partial_fit(x(:3, :), labels(:3), status, classes=classes, alpha=1.0_dp)
        call stream%partial_fit(x(4:, :), labels(4:), status)
        call check(status_ok(status) .and. stream%fitted() .and. stream%batch_count() == 2, &
            "Multinomial completed stream", failures)
        counts = stream%feature_counts(); fp = stream%feature_probabilities()
        call check(maxval(abs(counts - reshape([2.0_dp, 2.0_dp, 10.0_dp, 4.0_dp, &
            18.0_dp, 10.0_dp], shape(counts)))) < 2.0e-14_dp, &
            "Multinomial count oracle", failures)
        call reference%fit(x, labels, status, alpha=1.0_dp)
        call stream%predict_proba(query, p, status); call reference%predict_proba(query, p_ref, status)
        call check(status_ok(status) .and. maxval(abs(p - p_ref)) < 2.0e-14_dp, &
            "Multinomial replay equals one-shot probability", failures)
        cuda%kind = FORTML_DEVICE_CUDA; cuda%selected = .true.; cuda%available = .true.
        call stream%partial_fit_device(cuda, query, [9, 4], status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. stream%sample_count() == 6, &
            "Multinomial CUDA refusal", failures)
    end subroutine check_multinomial

    subroutine check_complement(failures)
        integer, intent(inout) :: failures
        type(complement_naive_bayes_t) :: stream, reference
        type(fortml_device_t) :: cuda
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 2), query(2, 2), p(2, 3), p_ref(2, 3)
        real(dp), allocatable :: counts(:, :), cp(:, :), weights(:, :)
        integer :: labels(6), classes(3), bad(2), bad_code

        x = reshape([8.0_dp, 0.0_dp, 10.0_dp, 2.0_dp, 4.0_dp, 6.0_dp, &
                     4.0_dp, 0.0_dp, 6.0_dp, 2.0_dp, 1.0_dp, 3.0_dp], shape(x))
        labels = [9, -3, 9, -3, 4, 4]; classes = [-3, 4, 9]
        query = reshape([1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp], shape(query)); bad = [9, 999]
        call stream%partial_fit(x(:3, :), labels(:3), status, classes=classes, alpha=1.0_dp)
        call stream%partial_fit(x(4:, :), labels(4:), status, norm=.false.)
        call check(status_ok(status) .and. stream%fitted() .and. stream%batch_count() == 2, &
            "Complement completed stream", failures)
        counts = stream%feature_counts(); cp = stream%feature_probabilities(); weights = stream%feature_weights()
        call check(maxval(abs(counts - reshape([2.0_dp, 2.0_dp, 10.0_dp, 4.0_dp, &
            18.0_dp, 10.0_dp], shape(counts)))) < 2.0e-14_dp .and. &
            all(cp > 0.0_dp) .and. all(weights > 0.0_dp), &
            "Complement sufficient-statistic oracle", failures)
        call reference%fit(x, labels, status, alpha=1.0_dp, norm=.false.)
        call stream%predict_proba(query, p, status); call reference%predict_proba(query, p_ref, status)
        call check(status_ok(status) .and. maxval(abs(p - p_ref)) < 2.0e-14_dp, &
            "Complement replay equals one-shot probability", failures)
        call stream%partial_fit(query, bad, status)
        bad_code = status%code
        call check(bad_code == FORTNUM_DOMAIN_ERROR .and. stream%sample_count() == 6, &
            "Complement unknown-label rollback", failures)
        cuda%kind = FORTML_DEVICE_CUDA; cuda%selected = .true.; cuda%available = .true.
        call stream%partial_fit_device(cuda, query, [9, 4], status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. stream%sample_count() == 6, &
            "Complement CUDA refusal", failures)
    end subroutine check_complement

    subroutine check_categorical(failures)
        integer, intent(inout) :: failures
        type(categorical_naive_bayes_t) :: stream, reference
        type(fortml_device_t) :: cuda
        type(fortnum_status_t) :: status
        integer :: x(6, 2), query(2, 2), labels(6), classes(3), bad(2), bad_code
        real(dp) :: p(2, 3), p_ref(2, 3), prior(3)

        x = reshape([1, 2, 1, 1, 2, 3, 2, 1, 2, 3, 1, 3], shape(x))
        labels = [9, -3, 9, -3, 4, 4]; classes = [-3, 4, 9]
        query = reshape([1, 1, 2, 3], shape(query)); bad = [9, 99]
        call stream%partial_fit(x(:3, :), labels(:3), status, classes=classes, alpha=1.0_dp)
        call check(status_ok(status) .and. .not. stream%fitted(), &
            "Categorical deferred first batch", failures)
        call stream%partial_fit(x(4:, :), labels(4:), status)
        prior = stream%class_prior()
        call check(status_ok(status) .and. stream%fitted() .and. stream%batch_count() == 2 .and. &
            maxval(abs(prior - 1.0_dp/3.0_dp)) < 2.0e-14_dp, &
            "Categorical completed stream and prior oracle", failures)
        call reference%fit(x, labels, status, alpha=1.0_dp)
        call stream%predict_proba(query, p, status); call reference%predict_proba(query, p_ref, status)
        call check(status_ok(status) .and. maxval(abs(p - p_ref)) < 2.0e-14_dp, &
            "Categorical replay equals one-shot probability", failures)
        call stream%partial_fit(query, bad, status)
        bad_code = status%code
        call check(bad_code == FORTNUM_DOMAIN_ERROR .and. stream%sample_count() == 6, &
            "Categorical unknown-label rollback", failures)
        cuda%kind = FORTML_DEVICE_CUDA; cuda%selected = .true.; cuda%available = .true.
        call stream%partial_fit_device(cuda, query, [9, 4], status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. stream%sample_count() == 6, &
            "Categorical CUDA refusal", failures)
    end subroutine check_categorical

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [nb-partial] "//description
        end if
    end subroutine check

end program test_naive_bayes_partial_fit
