program test_gp_ordinal_classification
    !! Independent ordinal GP behavior and derivative-product oracle.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_ordinal_classification, only: &
        gp_ordinal_classification_t, gp_ordinal_classification_options_t, &
        gp_ordinal_classification_state_t
    implicit none

    type(gp_ordinal_classification_t) :: model, model_plus, model_minus
    type(gp_ordinal_classification_options_t) :: options
    type(gp_ordinal_classification_state_t) :: state
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 1), query(5, 1), query_plus(5, 1), query_minus(5, 1)
    real(dp) :: query_dot(5, 1), probabilities(5, 3), probabilities_dot(5, 3)
    real(dp) :: probabilities_plus(5, 3), probabilities_minus(5, 3)
    real(dp) :: probabilities_bar(5, 3), query_bar(5, 1)
    real(dp), allocatable :: parameters(:), direction(:), parameters_plus(:), parameters_minus(:)
    real(dp) :: thresholds(2), expected_thresholds(2), h, lhs, rhs
    integer :: labels(9), predicted(5), classes(3), failures, i

    x(:, 1) = [-1.5_dp, -1.1_dp, -0.8_dp, -0.2_dp, 0.0_dp, 0.3_dp, &
        0.8_dp, 1.1_dp, 1.5_dp]
    labels = [-4, -4, -4, 7, 7, 7, 19, 19, 19]
    query(:, 1) = [-1.25_dp, -0.5_dp, 0.0_dp, 0.6_dp, 1.3_dp]
    query_dot(:, 1) = [0.2_dp, -0.3_dp, 0.1_dp, 0.5_dp, -0.2_dp]
    failures = 0
    kernel = make_rbf_kernel(1, 1.2_dp, 0.7_dp, status)
    options%noise_variance = 0.04_dp
    options%jitter = 1.0e-8_dp
    call model%fit(x, labels, kernel, status, options, state)
    call check(status_ok(status) .and. state%converged .and. model%fitted(), &
        "ordinal GP fit", failures)
    classes = model%classes()
    thresholds = model%thresholds()
    expected_thresholds = [1.5_dp, 2.5_dp]
    call check(all(classes == [-4, 7, 19]) .and. all(abs(thresholds - &
        expected_thresholds) < 1.0e-14_dp), "ordered class metadata", failures)
    call model%predict_proba(query, probabilities, status)
    call check(status_ok(status) .and. all(probabilities >= 0.0_dp) .and. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp, &
        "probability simplex", failures)
    call model%predict(query, predicted, status)
    call check(status_ok(status) .and. all(predicted == [-4, -4, 7, 19, 19]), &
        "ordered hard prediction", failures)

    ! Parameter JVP is checked against a separately evaluated central difference.
    allocate(parameters(model%parameter_count()), direction(model%parameter_count()), &
        parameters_plus(model%parameter_count()), parameters_minus(model%parameter_count()))
    parameters = model%parameters()
    direction = [0.11_dp, -0.07_dp, 0.04_dp]
    call model%predict_proba_parameter_jvp(query, direction, probabilities, &
        probabilities_dot, status)
    call model_plus%fit(x, labels, kernel, status, options)
    call model_minus%fit(x, labels, kernel, status, options)
    h = 1.0e-5_dp
    parameters_plus = parameters + h*direction
    parameters_minus = parameters - h*direction
    call model_plus%set_parameters(parameters_plus, status)
    call model_minus%set_parameters(parameters_minus, status)
    call model_plus%predict_proba(query, probabilities_plus, status)
    call model_minus%predict_proba(query, probabilities_minus, status)
    call check(maxval(abs(probabilities_dot - (probabilities_plus - probabilities_minus)/ &
        (2.0_dp*h))) < 4.0e-6_dp, "probability parameter JVP finite difference", failures)
    probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.4_dp, -0.3_dp, 0.1_dp, &
        -0.2_dp, 0.5_dp, -0.4_dp, 0.3_dp, -0.1_dp, &
        0.1_dp, -0.2_dp, 0.3_dp, 0.4_dp, -0.5_dp], shape(probabilities_bar))
    call model%predict_proba_parameter_vjp(query, probabilities_bar, parameters_plus, status)
    lhs = dot_product(parameters_plus, direction)
    rhs = sum(probabilities_bar*probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 4.0e-6_dp, &
        "probability parameter VJP duality", failures)

    ! Query-input JVP and VJP use an independent central-difference oracle.
    call model%predict_proba_input_jvp(query, query_dot, probabilities, probabilities_dot, status)
    query_plus = query + h*query_dot
    query_minus = query - h*query_dot
    call model%predict_proba(query_plus, probabilities_plus, status)
    call model%predict_proba(query_minus, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 2.0e-5_dp, &
        "probability input JVP finite difference", failures)
    call model%predict_proba_input_vjp(query, probabilities_bar, query_bar, status)
    lhs = sum(query_bar*query_dot)
    rhs = sum(probabilities_bar*probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-5_dp, &
        "probability input VJP duality", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, query, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA prediction refusal", failures)
    call model%predict_proba_parameter_vjp_device(cuda, query, probabilities_bar, &
        parameters_plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA parameter-product refusal", failures)
    call model%predict_proba_input_vjp_device(cuda, query, probabilities_bar, query_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA input-product refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL ordinal GP cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS ordinal GP independent behavioral and derivative oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-ordinal] "//description
        end if
    end subroutine check

end program test_gp_ordinal_classification
