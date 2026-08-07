program test_gaussian_naive_bayes
    !! Independent analytic checks for weighted Gaussian Naive Bayes.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_gaussian_naive_bayes, only: gaussian_naive_bayes_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    type(gaussian_naive_bayes_t) :: model, weighted_model, unfitted
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 2), query(3, 2), query_dot(3, 2)
    real(dp) :: log_probabilities(3, 3), log_plus(3, 3), log_minus(3, 3)
    real(dp) :: log_dot(3, 3), log_bar(3, 3), x_bar(3, 2)
    real(dp) :: probabilities_one(1, 3)
    real(dp) :: parameter_log_dot(3, 3), parameter_bar(15)
    real(dp) :: expected_prior(3), expected_mean(2, 3), expected_variance(2, 3)
    real(dp) :: weighted_counts(3), prior(3), theta_dot(15), theta(15)
    real(dp), allocatable :: means(:, :), variances(:, :), parameters(:)
    integer :: labels(6), classes(3), prediction(3), failures, i
    real(dp) :: lhs, rhs, step

    x(:, 1) = [8.0_dp, 0.0_dp, 10.0_dp, 2.0_dp, 4.0_dp, 6.0_dp]
    x(:, 2) = [4.0_dp, 0.0_dp, 6.0_dp, 2.0_dp, 1.0_dp, 3.0_dp]
    labels = [9, -3, 9, -3, 4, 4]
    query(1, :) = [1.0_dp, 1.0_dp]
    query(2, :) = [5.0_dp, 2.0_dp]
    query(3, :) = [9.0_dp, 5.0_dp]
    query_dot(1, :) = [0.1_dp, -0.2_dp]
    query_dot(2, :) = [-0.2_dp, 0.1_dp]
    query_dot(3, :) = [0.3_dp, 0.2_dp]
    failures = 0

    call model%fit(x, labels, status, var_smoothing=0.0_dp)
    call check(status_ok(status) .and. model%fitted(), &
        "fit and fitted state", failures)
    classes = model%classes()
    call check(status_ok(status) .and. all(classes == [-3, 4, 9]) .and. &
        model%class_count() == 3 .and. model%feature_count() == 2, &
        "sorted arbitrary class metadata", failures)
    means = model%means()
    variances = model%variances()
    expected_mean = reshape([1.0_dp, 1.0_dp, 5.0_dp, 2.0_dp, 9.0_dp, 5.0_dp], &
        shape(expected_mean))
    expected_variance = 1.0_dp
    call check(maxval(abs(means - expected_mean)) < 2.0e-14_dp .and. &
        maxval(abs(variances - expected_variance)) < 2.0e-14_dp, &
        "weighted moment oracle", failures)
    prior = model%class_prior()
    call check(maxval(abs(prior - 1.0_dp/3.0_dp)) < 2.0e-14_dp .and. &
        abs(sum(prior) - 1.0_dp) < 2.0e-14_dp, "empirical class priors", failures)

    call model%predict_log_proba(query, log_probabilities, status)
    call model%predict(query, prediction, status)
    call check(status_ok(status) .and. all(prediction == [-3, 4, 9]) .and. &
        maxval(abs(sum(exp(log_probabilities), dim=2) - 1.0_dp)) < 2.0e-14_dp .and. &
        all(exp(log_probabilities) > 0.0_dp), &
        "stable log-probability and predictions", failures)

    call model%predict_log_proba_jvp(query, query_dot, log_probabilities, log_dot, status)
    step = 1.0e-6_dp
    call model%predict_log_proba(query + step*query_dot, log_plus, status)
    call model%predict_log_proba(query - step*query_dot, log_minus, status)
    call check(status_ok(status) .and. maxval(abs(log_dot - &
        (log_plus - log_minus)/(2.0_dp*step))) < 3.0e-7_dp, &
        "input log-probability JVP finite difference", failures)
    log_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.3_dp, &
        -0.4_dp, 0.5_dp, -0.2_dp, &
        0.6_dp, 0.1_dp, -0.3_dp], shape(log_bar))
    call model%predict_log_proba_vjp(query, log_bar, x_bar, status)
    lhs = sum(log_bar*log_dot)
    rhs = sum(x_bar*query_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-7_dp, &
        "input log-probability VJP adjoint", failures)

    parameters = model%parameters()
    call check(size(parameters) == model%parameter_count() .and. &
        model%parameter_count() == 15, "packed parameter metadata", failures)
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

    call weighted_model%fit(x, labels, status, var_smoothing=0.0_dp, &
        sample_weight=[1.0_dp, 2.0_dp, 1.0_dp, 2.0_dp, 1.0_dp, 1.0_dp], &
        class_weight=[1.0_dp, 2.0_dp, 3.0_dp])
    weighted_counts = weighted_model%weighted_class_counts()
    prior = weighted_model%class_prior()
    expected_prior = [4.0_dp, 4.0_dp, 6.0_dp]/14.0_dp
    call check(status_ok(status) .and. maxval(abs(weighted_counts - [4.0_dp, 4.0_dp, &
        6.0_dp])) < 2.0e-14_dp .and. maxval(abs(prior - expected_prior)) < &
        2.0e-14_dp, "sample and class weight oracle", failures)

    call model%fit(x, labels, status, var_smoothing=0.0_dp, &
        priors=[2.0_dp, 3.0_dp, 5.0_dp])
    prior = model%class_prior()
    call check(status_ok(status) .and. maxval(abs(prior - [0.2_dp, 0.3_dp, 0.5_dp])) < &
        2.0e-14_dp, "explicit normalized priors", failures)
    theta = model%parameters()
    theta(1:6) = 0.0_dp
    theta(7:12) = 1.0_dp
    theta(13:15) = [1.0_dp, 2.0_dp, 3.0_dp]
    call model%set_parameters(theta, status)
    call model%predict_proba(reshape([0.0_dp, 0.0_dp], [1, 2]), &
        probabilities_one, status)
    call check(status_ok(status), "valid packed parameter update", failures)

    call unfitted%predict_proba(query, log_probabilities, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "unfitted prediction refusal", failures)
    call model%fit(x, labels, status, var_smoothing=-1.0_dp)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "negative smoothing refusal", failures)
    call model%fit(x, labels, status, class_weight=[1.0_dp, 2.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, "class-weight shape refusal", failures)
    call model%fit(x, labels, status, sample_weight=[1.0_dp, 0.0_dp, 0.0_dp, &
        0.0_dp, 0.0_dp, 0.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero class mass refusal", failures)
    call model%fit(x, labels, status, priors=[1.0_dp, 0.0_dp, 1.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, "nonpositive prior refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GaussianNB cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS Gaussian Naive Bayes independent analytic oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gaussian-nb] "//name
        end if
    end subroutine check

end program test_gaussian_naive_bayes
