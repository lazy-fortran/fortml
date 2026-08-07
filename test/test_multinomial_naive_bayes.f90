program test_multinomial_naive_bayes
    !! Independent analytic and differential checks for MultinomialNB.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_multinomial_naive_bayes, only: multinomial_naive_bayes_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    type(multinomial_naive_bayes_t) :: model, weighted_model, unfitted
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 3), query(3, 3), query_dot(3, 3)
    real(dp) :: log_probabilities(3, 3), log_plus(3, 3), log_minus(3, 3)
    real(dp) :: log_dot(3, 3), log_bar(3, 3), x_bar(3, 3)
    real(dp) :: parameter_log_dot(3, 3), parameter_bar(12)
    real(dp) :: theta_dot(12), theta(12), prior(3), q(3, 3), counts(3, 3)
    real(dp) :: probabilities(3, 3), nan_query(3, 3), nan_value
    real(dp) :: expected_q(3, 3), expected_counts(3, 3), expected_log(3)
    real(dp) :: expected_prior(3), normalizer, lhs, rhs, step
    real(dp), allocatable :: parameters(:)
    integer :: labels(6), classes(3), prediction(3), expected_prediction(3)
    integer :: failures, i, j, c

    x(1, :) = [1.0_dp, 1.0_dp, 0.0_dp]
    x(2, :) = [0.0_dp, 1.0_dp, 0.0_dp]
    x(3, :) = [1.0_dp, 0.0_dp, 1.0_dp]
    x(4, :) = [0.0_dp, 0.0_dp, 1.0_dp]
    x(5, :) = [1.0_dp, 1.0_dp, 1.0_dp]
    x(6, :) = [0.0_dp, 0.0_dp, 0.0_dp]
    labels = [7, -2, 7, -2, 4, 4]
    query(1, :) = [0.2_dp, 0.8_dp, 0.4_dp]
    query(2, :) = [1.3_dp, 0.2_dp, 0.7_dp]
    query(3, :) = [0.6_dp, 1.1_dp, 1.4_dp]
    query_dot(1, :) = [0.1_dp, -0.2_dp, 0.3_dp]
    query_dot(2, :) = [-0.2_dp, 0.1_dp, -0.1_dp]
    query_dot(3, :) = [0.3_dp, 0.2_dp, -0.2_dp]
    failures = 0

    call model%fit(x, labels, status, alpha=1.0_dp)
    call check(status_ok(status) .and. model%fitted(), &
        "fit and fitted state", failures)
    classes = model%classes()
    call check(all(classes == [-2, 4, 7]) .and. model%class_count() == 3 .and. &
        model%feature_count() == 3, "sorted arbitrary class metadata", failures)
    q = model%feature_probabilities()
    expected_q = reshape([ &
        1.0_dp/5.0_dp, 2.0_dp/5.0_dp, 2.0_dp/5.0_dp, &
        1.0_dp/3.0_dp, 1.0_dp/3.0_dp, 1.0_dp/3.0_dp, &
        3.0_dp/7.0_dp, 2.0_dp/7.0_dp, 2.0_dp/7.0_dp], shape(expected_q))
    call check(maxval(abs(q - expected_q)) < 2.0e-14_dp, &
        "analytic smoothed feature probabilities", failures)
    counts = model%feature_counts()
    expected_counts = reshape([0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, &
        2.0_dp, 1.0_dp, 1.0_dp], shape(expected_counts))
    call check(maxval(abs(counts - expected_counts)) < 2.0e-14_dp, &
        "weighted feature-count oracle", failures)
    prior = model%class_prior()
    call check(maxval(abs(prior - 1.0_dp/3.0_dp)) < 2.0e-14_dp .and. &
        abs(sum(prior) - 1.0_dp) < 2.0e-14_dp, "empirical class priors", failures)

    call model%predict_log_proba(query, log_probabilities, status)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, prediction, status)
    do i = 1, 3
        expected_log = log(prior)
        do c = 1, 3
            do j = 1, 3
                expected_log(c) = expected_log(c) + query(i, j)*log(expected_q(j, c))
            end do
        end do
        normalizer = sum(exp(expected_log - maxval(expected_log)))
        expected_log = expected_log - maxval(expected_log) - log(normalizer)
        expected_prediction(i) = classes(maxloc(expected_log, dim=1))
        call check(maxval(abs(log_probabilities(i, :) - expected_log)) < 2.0e-14_dp, &
            "analytic log-probability oracle", failures)
    end do
    call check(status_ok(status) .and. all(prediction == expected_prediction) .and. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp .and. &
        maxval(abs(probabilities - exp(log_probabilities))) < 2.0e-14_dp, &
        "stable probabilities and predictions", failures)

    step = 1.0e-6_dp
    call model%predict_log_proba_jvp(query, query_dot, log_probabilities, log_dot, status)
    call model%predict_log_proba(query + step*query_dot, log_plus, status)
    call model%predict_log_proba(query - step*query_dot, log_minus, status)
    call check(status_ok(status) .and. maxval(abs(log_dot - &
        (log_plus - log_minus)/(2.0_dp*step))) < 3.0e-7_dp, &
        "input log-probability JVP finite difference", failures)
    log_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.3_dp, -0.4_dp, 0.5_dp, -0.2_dp, &
        0.6_dp, 0.1_dp, -0.3_dp], shape(log_bar))
    call model%predict_log_proba_vjp(query, log_bar, x_bar, status)
    lhs = sum(log_bar*log_dot)
    rhs = sum(x_bar*query_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-7_dp, &
        "input log-probability VJP adjoint", failures)

    parameters = model%parameters()
    call check(size(parameters) == model%parameter_count() .and. &
        model%parameter_count() == 12, "packed parameter metadata", failures)
    theta = parameters
    theta_dot = [(0.001_dp*real(i, dp), i=1,size(theta_dot))]
    call model%predict_log_proba_parameter_jvp(query, theta_dot, log_probabilities, &
        parameter_log_dot, status)
    call model%set_parameters(theta + step*theta_dot, status)
    call model%predict_log_proba(query, log_plus, status)
    call model%set_parameters(theta - step*theta_dot, status)
    call model%predict_log_proba(query, log_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(parameter_log_dot - &
        (log_plus - log_minus)/(2.0_dp*step))) < 4.0e-7_dp, &
        "parameter log-probability JVP finite difference", failures)
    parameter_bar = [(0.01_dp*real(i, dp), i=1,size(parameter_bar))]
    call model%predict_log_proba_parameter_vjp(query, log_bar, parameter_bar, status)
    lhs = sum(log_bar*parameter_log_dot)
    rhs = sum(parameter_bar*theta_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 4.0e-7_dp, &
        "parameter log-probability VJP adjoint", failures)

    call weighted_model%fit(x, labels, status, alpha=1.0_dp, &
        sample_weight=[1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp, 1.0_dp, 1.0_dp], &
        class_weight=[1.0_dp, 2.0_dp, 3.0_dp])
    q = weighted_model%feature_probabilities()
    expected_q = reshape([ &
        1.0_dp/7.0_dp, 3.0_dp/7.0_dp, 3.0_dp/7.0_dp, &
        1.0_dp/3.0_dp, 1.0_dp/3.0_dp, 1.0_dp/3.0_dp, &
        7.0_dp/15.0_dp, 4.0_dp/15.0_dp, 4.0_dp/15.0_dp], shape(expected_q))
    expected_prior = [4.0_dp, 4.0_dp, 6.0_dp]/14.0_dp
    call check(status_ok(status) .and. maxval(abs(q - expected_q)) < 2.0e-14_dp .and. &
        maxval(abs(weighted_model%class_prior() - expected_prior)) < 2.0e-14_dp .and. &
        maxval(abs(weighted_model%weighted_class_counts() - [4.0_dp, 4.0_dp, 6.0_dp])) &
        < 2.0e-14_dp, "sample and class weight oracle", failures)

    call model%fit(x, labels, status, alpha=0.0_dp)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "zero alpha refusal for differentiable model", failures)
    call model%fit(x, labels, status, alpha=-1.0_dp)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "negative alpha refusal", failures)
    call model%fit(x, labels, status, sample_weight=[1.0_dp, 0.0_dp, 0.0_dp, &
        0.0_dp, 0.0_dp, 0.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero class mass refusal", failures)
    call model%fit(x, labels, status, class_weight=[1.0_dp, 2.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, "class-weight shape refusal", failures)
    call model%fit(x, labels, status, priors=[1.0_dp, 0.0_dp, 1.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, "nonpositive prior refusal", failures)
    nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
    nan_query = query
    nan_query(1, 1) = nan_value
    call weighted_model%predict_log_proba(nan_query, log_probabilities, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "nonfinite prediction-input refusal", failures)
    theta = weighted_model%parameters()
    theta(1) = 0.0_dp
    call weighted_model%set_parameters(theta, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "boundary packed probability refusal", failures)
    call unfitted%predict_proba(query, probabilities, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "unfitted prediction refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL MultinomialNB cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS Multinomial Naive Bayes independent analytic oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [multinomial-nb] "//name
        end if
    end subroutine check

end program test_multinomial_naive_bayes
