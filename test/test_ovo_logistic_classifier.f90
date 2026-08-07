program test_ovo_logistic_classifier
    !! Independent behavior checks for one-vs-one logistic classification.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_ovo_logistic_classifier, only: ovo_logistic_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(ovo_logistic_classifier_t) :: model, weighted_model, unfitted
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 2), query(4, 2), query_dot(4, 2)
    real(dp) :: probabilities(4, 3), probabilities_dot(4, 3)
    real(dp) :: probabilities_plus(4, 3), probabilities_minus(4, 3)
    real(dp) :: probabilities_bar(4, 3), x_bar(4, 2)
    real(dp) :: weighted_x(3, 1), weighted_query(1, 1), weighted_probabilities(1, 2)
    real(dp) :: parameter_probabilities_dot(4, 3)
    real(dp), allocatable :: parameters(:), parameters_dot(:), parameters_plus(:)
    real(dp), allocatable :: parameters_minus(:), parameters_bar(:)
    integer :: labels(9), weighted_labels(3), classes(3), pairs(2, 3)
    integer :: expected_pairs(2, 3), predicted(4), zero_predicted(4)
    integer :: failures, i
    real(dp) :: lhs, rhs

    x(1, :) = [-2.0_dp, 2.0_dp]
    x(2, :) = [-1.8_dp, 2.1_dp]
    x(3, :) = [-2.1_dp, 1.8_dp]
    x(4, :) = [2.0_dp, 2.0_dp]
    x(5, :) = [1.8_dp, 2.1_dp]
    x(6, :) = [2.1_dp, 1.8_dp]
    x(7, :) = [0.0_dp, -2.0_dp]
    x(8, :) = [0.2_dp, -1.8_dp]
    x(9, :) = [-0.2_dp, -2.1_dp]
    labels = [42, 42, 42, -7, -7, -7, 10, 10, 10]
    query(1, :) = [-2.0_dp, 2.0_dp]
    query(2, :) = [2.0_dp, 2.0_dp]
    query(3, :) = [0.0_dp, -2.0_dp]
    query(4, :) = [0.0_dp, 0.0_dp]
    query_dot(1, :) = [0.1_dp, -0.2_dp]
    query_dot(2, :) = [-0.2_dp, 0.1_dp]
    query_dot(3, :) = [0.2_dp, 0.1_dp]
    query_dot(4, :) = [-0.1_dp, 0.3_dp]
    failures = 0

    call model%fit(x, labels, status, l2=0.1_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp)
    call check(status_ok(status) .and. model%fitted(), &
        "fit and fitted state", failures)
    classes = model%classes()
    expected_pairs = reshape([-7, 10, -7, 42, 10, 42], shape(expected_pairs))
    pairs = model%pair_classes()
    call check(status_ok(status) .and. all(classes == [-7, 10, 42]) .and. &
        model%class_count() == 3 .and. model%feature_count() == 2 .and. &
        model%pair_count() == 3 .and. all(pairs == expected_pairs), &
        "sorted classes and deterministic pair metadata", failures)

    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predicted, status)
    call check(status_ok(status) .and. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp .and. &
        all(probabilities >= 0.0_dp), "pairwise vote probability simplex", failures)
    call check(status_ok(status) .and. all(predicted(:3) == [42, -7, 10]), &
        "separable center predictions", failures)

    call model%predict_proba_jvp(query, query_dot, probabilities, &
        probabilities_dot, status)
    call model%predict_proba(query + 1.0e-5_dp*query_dot, probabilities_plus, status)
    call model%predict_proba(query - 1.0e-5_dp*query_dot, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0e-5_dp))) < 3.0e-6_dp, &
        "input probability JVP finite difference", failures)

    probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp,  0.3_dp, &
       -0.4_dp,  0.5_dp, -0.2_dp, &
        0.6_dp,  0.1_dp, -0.3_dp, &
       -0.2_dp,  0.7_dp,  0.4_dp], shape(probabilities_bar))
    call model%predict_proba_vjp(query, probabilities_bar, x_bar, status)
    lhs = sum(probabilities_bar*probabilities_dot)
    rhs = sum(x_bar*query_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-6_dp, &
        "input probability VJP adjoint identity", failures)

    parameters = model%parameters()
    allocate(parameters_dot(size(parameters)), parameters_plus(size(parameters)), &
        parameters_minus(size(parameters)), parameters_bar(size(parameters)))
    parameters_dot = [(0.01_dp*real(i, dp), i=1,size(parameters))]
    call model%predict_proba_parameter_jvp(query, parameters_dot, probabilities, &
        parameter_probabilities_dot, status)
    parameters_plus = parameters + 1.0e-5_dp*parameters_dot
    call model%set_parameters(parameters_plus, status)
    call model%predict_proba(query, probabilities_plus, status)
    parameters_minus = parameters - 1.0e-5_dp*parameters_dot
    call model%set_parameters(parameters_minus, status)
    call model%predict_proba(query, probabilities_minus, status)
    call model%set_parameters(parameters, status)
    call check(status_ok(status) .and. maxval(abs(parameter_probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0e-5_dp))) < 3.0e-6_dp, &
        "parameter probability JVP finite difference", failures)
    call model%predict_proba_parameter_vjp(query, probabilities_bar, &
        parameters_bar, status)
    lhs = sum(probabilities_bar*parameter_probabilities_dot)
    rhs = sum(parameters_bar*parameters_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-6_dp, &
        "parameter probability VJP adjoint identity", failures)
    call check(model%parameter_count() == model%pair_count()*3 .and. &
        size(parameters) == model%parameter_count(), "packed pair parameters", failures)

    parameters = 0.0_dp
    call model%set_parameters(parameters, status)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, zero_predicted, status)
    call check(status_ok(status) .and. maxval(abs(probabilities - 1.0_dp/3.0_dp)) < &
        2.0e-14_dp .and. all(zero_predicted == -7), &
        "zero-parameter equal vote probabilities and first tie", failures)

    weighted_x = 0.0_dp
    weighted_query = 0.0_dp
    weighted_labels = [-7, -7, 42]
    call weighted_model%fit(weighted_x, weighted_labels, status, l2=0.0_dp, &
        class_weight=[1.0_dp, 3.0_dp], max_iterations=1000, tolerance=1.0e-9_dp)
    call weighted_model%predict_proba(weighted_query, weighted_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(weighted_probabilities - &
        reshape([0.4_dp, 0.6_dp], [1, 2]))) < 2.0e-7_dp, &
        "class-weighted pair probability oracle", failures)

    call unfitted%predict_proba(query, probabilities, status)
    call check(.not. status_ok(status), "unfitted prediction refusal", failures)
    call model%fit(x, labels, status, class_weight=[1.0_dp, 2.0_dp])
    call check(.not. status_ok(status), "class-weight shape refusal", failures)
    call model%fit(x, labels, status, class_weight=[1.0_dp, 2.0_dp, 0.0_dp])
    call check(.not. status_ok(status), "nonpositive class-weight refusal", failures)
    x(1, 1) = ieee_value(x(1, 1), ieee_quiet_nan)
    call model%fit(x, labels, status)
    call check(.not. status_ok(status), "nonfinite fit-input refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL OVO logistic classification cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS OVO logistic classifier independent oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [ovo-logistic] "//name
        end if
    end subroutine check

end program test_ovo_logistic_classifier
