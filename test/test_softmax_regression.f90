program test_softmax_regression
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_softmax_regression, only: softmax_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(softmax_regression_t) :: model
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 2), probabilities(6, 3), log_probabilities(6, 3)
    real(dp) :: log_probabilities_dot(6, 3), scores(6, 3), expected(6, 3)
    real(dp) :: x_dot(6, 2), theta(9), theta_dot(9), logp_plus(6, 3), logp_minus(6, 3)
    integer :: labels(6), predicted(6), classes(3), i
    integer :: failures

    failures = 0
    x = reshape([ &
        2.0_dp, 1.0_dp,-2.0_dp,-1.0_dp, 0.0_dp, 0.0_dp, &
        0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 2.0_dp, 1.0_dp], shape(x))
    labels = [-2, -2, 7, 7, 42, 42]
    call model%fit(x, labels, status, l2=0.1_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp)
    if (.not. status_ok(status)) then
        write (error_unit, '(a)') "FAIL [softmax] fit: "//trim(status%msg)
        error stop 1
    end if
    call model%decision_function(x, scores, status)
    call model%predict_proba(x, probabilities, status)
    call model%predict_log_proba(x, log_probabilities, status)
    call model%predict(x, predicted, status)
    classes = model%classes()
    if (.not. status_ok(status) .or. any(classes /= [-2, 7, 42]) .or. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) > 1.0e-14_dp .or. &
        maxval(abs(exp(log_probabilities) - probabilities)) > 1.0e-14_dp .or. &
        real(count(predicted == labels), dp)/real(size(labels), dp) < 0.99_dp .or. &
        any(shape(scores) /= [6, 3])) then
        write (error_unit, '(a)') "FAIL [softmax] independent probability/class oracle"
        failures = failures + 1
    end if

    x(:, :) = 0.0_dp
    expected(:, 1) = 0.25_dp
    expected(:, 2) = 0.25_dp
    expected(:, 3) = 0.5_dp
    call model%fit(x, labels, status, l2=0.0_dp, sample_weight=[ &
        1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 3.0_dp], &
        max_iterations=1000, tolerance=1.0e-8_dp)
    call model%predict_proba(x, probabilities, status)
    if (.not. status_ok(status) .or. maxval(abs(probabilities - expected)) > &
        2.0e-7_dp) then
        write (error_unit, '(a)') "FAIL [softmax] weighted probability oracle"
        failures = failures + 1
    end if

    x(:, :) = 0.0_dp
    expected(:, 1) = 1.0_dp/7.0_dp
    expected(:, 2) = 2.0_dp/7.0_dp
    expected(:, 3) = 4.0_dp/7.0_dp
    call model%fit(x, labels, status, l2=0.0_dp, class_weight=[ &
        1.0_dp, 2.0_dp, 4.0_dp], max_iterations=1000, tolerance=1.0e-8_dp)
    call model%predict_proba(x, probabilities, status)
    if (.not. status_ok(status) .or. maxval(abs(probabilities - expected)) > &
        2.0e-7_dp) then
        write (error_unit, '(a)') "FAIL [softmax] class-weighted probability oracle"
        failures = failures + 1
    end if
    call model%fit(x, labels, status, sample_weight=[0.0_dp, 0.0_dp, 0.0_dp, &
        0.0_dp, 0.0_dp, 0.0_dp])
    if (status_ok(status)) then
        write (error_unit, '(a)') "FAIL [softmax] zero sample-weight refusal"
        failures = failures + 1
    end if
    call model%fit(x, labels, status, class_weight=[1.0_dp, 2.0_dp])
    if (status_ok(status)) then
        write (error_unit, '(a)') "FAIL [softmax] class-weight shape refusal"
        failures = failures + 1
    end if
    call model%fit(x, labels, status, class_weight=[1.0_dp, 2.0_dp, 0.0_dp])
    if (status_ok(status)) then
        write (error_unit, '(a)') "FAIL [softmax] nonpositive class-weight refusal"
        failures = failures + 1
    end if

    call model%fit(x, [1, 1, 1, 1, 1, 1], status)
    if (status_ok(status)) then
        write (error_unit, '(a)') "FAIL [softmax] one-class refusal"
        failures = failures + 1
    end if

    ! Finite-difference and adjoint checks for the full log-softmax product.
    x = reshape([ &
        -1.0_dp, 0.5_dp, 0.2_dp, 1.0_dp, 0.7_dp, -0.4_dp, &
        0.3_dp, -0.8_dp, 1.1_dp, 0.2_dp, -0.6_dp, 0.9_dp], shape(x))
    labels = [-2, 7, 42, -2, 7, 42]
    call model%fit(x, labels, status, l2=0.2_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp)
    if (.not. status_ok(status)) then
        write (error_unit, '(a)') "FAIL [softmax] product fit"
        failures = failures + 1
    else
        x_dot = 0.0_dp
        theta = model%parameters()
        theta_dot = [(0.01_dp*real(i, dp), i=1,9)]
        call model%predict_log_proba_jvp(x, theta_dot, x_dot, log_probabilities, &
            log_probabilities_dot, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [softmax] log-probability JVP"
            failures = failures + 1
        end if
        call model%set_parameters(theta + 1.0e-6_dp*theta_dot, status)
        call model%predict_log_proba(x, logp_plus, status)
        call model%set_parameters(theta - 1.0e-6_dp*theta_dot, status)
        call model%predict_log_proba(x, logp_minus, status)
        call model%set_parameters(theta, status)
        if (maxval(abs(log_probabilities_dot - (logp_plus - logp_minus)/2.0e-6_dp)) > &
            2.0e-6_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [softmax] log-probability finite difference: ", &
                maxval(abs(log_probabilities_dot - (logp_plus - logp_minus)/2.0e-6_dp))
            failures = failures + 1
        end if
    end if
    if (failures /= 0) error stop 1
    write (*, '(a)') "PASS"

end program test_softmax_regression
