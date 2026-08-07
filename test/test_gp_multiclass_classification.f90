program test_gp_multiclass_classification
    !! Independent behavior checks for the one-vs-rest GP classifier.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t, &
        gp_multiclass_classification_options_t, &
        gp_multiclass_classification_state_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(gp_multiclass_classification_t) :: model, repeat_model, probit_model, unfitted
    type(gp_multiclass_classification_options_t) :: options, probit_options
    type(gp_multiclass_classification_state_t) :: state, repeat_state
    type(fortnum_status_t) :: status
    type(kernel_t) :: kernel
    real(dp) :: x(9, 2), query(6, 2), query_dot(6, 2), query_plus(6, 2)
    real(dp) :: query_minus(6, 2), probabilities(6, 3), probabilities_dot(6, 3)
    real(dp) :: probabilities_plus(6, 3), probabilities_minus(6, 3)
    real(dp) :: repeat_probabilities(6, 3), probit_probabilities(6, 3)
    integer :: labels(9), predicted(9), query_predicted(6), classes(3)
    integer :: failures

    x(1, :) = [-0.1_dp, 1.9_dp]
    x(2, :) = [0.1_dp, 2.1_dp]
    x(3, :) = [0.2_dp, 1.8_dp]
    x(4, :) = [-0.1_dp, -0.1_dp]
    x(5, :) = [0.1_dp, 0.2_dp]
    x(6, :) = [0.3_dp, 0.0_dp]
    x(7, :) = [1.9_dp, 0.0_dp]
    x(8, :) = [2.1_dp, 0.2_dp]
    x(9, :) = [1.8_dp, 0.3_dp]
    labels = [42, 42, 42, -7, -7, -7, 10, 10, 10]
    query(1, :) = [0.0_dp, 2.0_dp]
    query(2, :) = [0.0_dp, 0.0_dp]
    query(3, :) = [2.0_dp, 0.0_dp]
    query(4, :) = [0.2_dp, 1.0_dp]
    query(5, :) = [1.0_dp, 0.2_dp]
    query(6, :) = [0.5_dp, 0.5_dp]
    query_dot = 0.0_dp
    query_dot(:, 1) = [0.2_dp, -0.1_dp, 0.3_dp, 0.0_dp, -0.2_dp, 0.1_dp]
    query_dot(:, 2) = [-0.1_dp, 0.2_dp, 0.1_dp, -0.3_dp, 0.2_dp, 0.0_dp]
    failures = 0

    kernel = make_rbf_kernel(2, 1.5_dp, 0.55_dp, status)
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    options%jitter = 1.0e-7_dp
    call model%fit(x, labels, kernel, status, options, state)
    call check(status_ok(status) .and. state%converged .and. model%fitted(), &
        "one-vs-rest fit", failures)
    classes = model%classes()
    call check(all(classes == [-7, 10, 42]) .and. model%class_count() == 3 .and. &
        model%feature_count() == 2, "sorted class metadata", failures)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, query_predicted, status)
    call check(status_ok(status) .and. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp .and. &
        all(probabilities >= 0.0_dp), "normalized probability simplex", failures)
    call check(query_predicted(1) == 42 .and. query_predicted(2) == -7 .and. &
        query_predicted(3) == 10, "class ranking at separated centers", failures)

    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. count(predicted == labels) >= 8, &
        "one-vs-rest training behavior", failures)

    call model%predict_proba_jvp(query, query_dot, probabilities, &
        probabilities_dot, status)
    query_plus = query + 1.0e-5_dp*query_dot
    query_minus = query - 1.0e-5_dp*query_dot
    call model%predict_proba(query_plus, probabilities_plus, status)
    call model%predict_proba(query_minus, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0e-5_dp))) < 3.0e-6_dp, &
        "normalized probability JVP finite difference", failures)

    call repeat_model%fit(x, labels, kernel, status, options, repeat_state)
    call repeat_model%predict_proba(query, repeat_probabilities, status)
    call check(status_ok(status) .and. repeat_state%converged .and. &
        maxval(abs(repeat_probabilities - probabilities)) < 2.0e-14_dp, &
        "deterministic one-vs-rest fit", failures)

    probit_options = options
    probit_options%likelihood = 2
    call probit_model%fit(x, labels, kernel, status, probit_options)
    call probit_model%predict_proba(query, probit_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(sum(probit_probabilities, dim=2) - &
        1.0_dp)) < 2.0e-14_dp, "probit one-vs-rest fit", failures)

    call unfitted%predict_proba(query, probabilities, status)
    call check(.not. status_ok(status), "unfitted prediction refusal", failures)
    call model%predict(query, predicted(:5), status)
    call check(.not. status_ok(status), "prediction label shape refusal", failures)
    call model%fit(x, [1, 1, 1, 1, 1, 1, 1, 1, 1], kernel, status, options)
    call check(.not. status_ok(status), "one-class fit refusal", failures)
    options%likelihood = 99
    call model%fit(x, labels, kernel, status, options)
    call check(.not. status_ok(status), "invalid likelihood refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL GP multiclass classification cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP multiclass classification independent oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-multiclass] "//name
        end if
    end subroutine check

end program test_gp_multiclass_classification
