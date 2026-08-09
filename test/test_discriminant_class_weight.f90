program test_discriminant_class_weight
    !! Independent oracle for class-weighted LDA/QDA fitting.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_discriminant_analysis, only: lda_classifier_t, qda_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    type(lda_classifier_t) :: lda, lda_reference
    type(qda_classifier_t) :: qda
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 1), query(3, 1), probabilities(3, 2), probability_reference(3, 2)
    real(dp) :: means(1, 2), covariance(1, 1), priors(2), counts(2)
    real(dp) :: qda_covariance(1, 1, 2), expected_probability
    integer :: labels(6), classes(2), predictions(3), failures

    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
    labels = [7, 7, 7, -2, -2, -2]
    query(:, 1) = [0.0_dp, 1.0_dp, 4.0_dp]
    failures = 0

    ! The classes are sorted as [-2, 7], so these factors make the
    ! effective per-row weights [2,2,2,0.5,0.5,0.5].
    call lda%fit(x, labels, status, reg_param=0.0_dp, class_weight=[0.5_dp, 2.0_dp])
    call check(status_ok(status) .and. lda%fitted(), "class-weighted LDA fit", failures)
    classes = lda%classes()
    call check(all(classes == [-2, 7]), "class weights use sorted labels", failures)
    counts = lda%weighted_class_counts()
    call check(maxval(abs(counts - [1.5_dp, 6.0_dp])) < 2.0e-14_dp, &
        "class-weighted effective counts", failures)
    priors = lda%class_prior()
    call check(maxval(abs(priors - [0.2_dp, 0.8_dp])) < 2.0e-14_dp, &
        "class-weighted empirical priors", failures)
    means = lda%means()
    call check(maxval(abs(means - reshape([4.0_dp, 1.0_dp], shape(means)))) < &
        2.0e-14_dp, "class-weighted means", failures)
    covariance = lda%covariance()
    call check(abs(covariance(1, 1) - 2.0_dp/3.0_dp) < 2.0e-14_dp, &
        "class-weighted pooled covariance", failures)

    call lda%predict_proba(query, probabilities, status)
    call lda%predict(query, predictions, status)
    expected_probability = 1.0_dp/(1.0_dp + 4.0_dp*exp(11.25_dp - &
        4.5_dp*query(1, 1)))
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < &
        2.0e-14_dp .and. abs(probabilities(1, 1) - expected_probability) < 2.0e-14_dp &
        .and. all(predictions == [7, 7, -2]), &
        "class-weighted LDA probability oracle", failures)

    ! Class weights must be exactly equivalent to multiplying sample weights.
    call lda_reference%fit(x, labels, status, reg_param=0.0_dp, &
        sample_weight=[2.0_dp, 2.0_dp, 2.0_dp, 0.5_dp, 0.5_dp, 0.5_dp])
    call check(status_ok(status), "reference weighted LDA fit", failures)
    call lda_reference%predict_proba(query, probability_reference, status)
    call check(status_ok(status), "reference weighted LDA prediction", failures)
    call lda%predict_proba(query, probabilities, status)
    call check(maxval(abs(probabilities - probability_reference)) < 2.0e-14_dp, &
        "class-weight/sample-weight equivalence", failures)

    call qda%fit(x, labels, status, reg_param=0.0_dp, class_weight=[0.5_dp, 2.0_dp])
    call check(status_ok(status) .and. qda%fitted(), "class-weighted QDA fit", failures)
    qda_covariance = qda%covariances()
    call check(maxval(abs(qda_covariance(1, 1, :) - 2.0_dp/3.0_dp)) < 2.0e-14_dp, &
        "class-weighted QDA covariance", failures)
    call qda%predict(query, predictions, status)
    call check(status_ok(status) .and. all(predictions == [7, 7, -2]), &
        "class-weighted QDA prediction", failures)

    call lda%fit(x, labels, status, class_weight=[1.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "class-weight shape refusal", failures)
    call qda%fit(x, labels, status, class_weight=[0.0_dp, 1.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "nonpositive class-weight refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL discriminant class-weight cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS class-weighted LDA/QDA independent oracle"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [discriminant class-weight] "//name
        end if
    end subroutine check

end program test_discriminant_class_weight
