program test_polynomial_svm_classifier
    !! Independent polynomial-kernel score, derivative, and device oracles.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_polynomial_svm_classifier, only: polynomial_svm_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(polynomial_svm_classifier_t) :: model, invalid_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(real64) :: x(8, 2), query(4, 2), query_dot(4, 2)
    real(real64) :: scores(4), scores_dot(4), scores_plus(4), scores_minus(4)
    real(real64) :: probabilities(4, 2), probabilities_dot(4, 2)
    real(real64) :: probabilities_plus(4, 2), probabilities_minus(4, 2)
    real(real64), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
    real(real64), allocatable :: theta_bar(:), coefficients(:)
    real(real64) :: x_bar(4, 2), scores_bar(4), probability_bar(4, 2), expected(4)
    real(real64) :: eps, dot, base
    integer :: labels(8), predictions(4), classes(2), failures, i, j

    x = reshape([ &
        -1.0_real64, -1.0_real64,  -0.8_real64, -1.1_real64, &
         1.0_real64,  1.0_real64,   0.8_real64,  1.1_real64, &
        -1.0_real64,  1.0_real64,  -0.8_real64,  1.1_real64, &
         1.0_real64, -1.0_real64,   0.8_real64, -1.1_real64], [8, 2])
    labels = [-4, -4, -4, -4, 9, 9, 9, 9]
    query = reshape([ &
        -0.9_real64, -0.9_real64, 0.9_real64, 0.9_real64, &
        -0.9_real64,  0.9_real64, 0.9_real64, -0.9_real64], [4, 2])
    query_dot = reshape([0.1_real64, -0.2_real64, 0.2_real64, 0.1_real64, &
        -0.1_real64, 0.3_real64, 0.05_real64, -0.1_real64], [4, 2])
    scores_bar = [0.3_real64, -0.4_real64, 0.7_real64, 0.2_real64]
    probability_bar(:, 1) = [0.2_real64, -0.1_real64, 0.4_real64, -0.3_real64]
    probability_bar(:, 2) = [-0.3_real64, 0.5_real64, 0.6_real64, 0.1_real64]
    failures = 0
    eps = 2.0e-6_real64

    call model%fit(x, labels, status, c=3.0_real64, gamma=0.4_real64, degree=2, &
        coef0=1.0_real64, max_iterations=2000, tolerance=1.0e-8_real64)
    call check(status_ok(status) .and. model%fitted(), "polynomial SVM fit", failures)
    classes = model%classes()
    call check(all(classes == [-4, 9]) .and. model%degree() == 2 .and. &
        abs(model%coef0()-1.0_real64) < 1.0e-14_real64, &
        "sorted classes and polynomial metadata", failures)
    call model%decision_function(query, scores, status)
    call check(status_ok(status), "decision function", failures)
    coefficients = model%coefficients()
    expected = model%intercept()
    do i = 1, size(query, 1)
        do j = 1, size(x, 1)
            dot = dot_product(query(i, :), x(j, :))
            base = model%gamma()*dot + model%coef0()
            expected(i) = expected(i) + coefficients(j)*base**model%degree()
        end do
    end do
    call check(maxval(abs(scores-expected)) < 2.0e-12_real64, &
        "explicit polynomial score oracle", failures)
    call model%predict(query, predictions, status)
    call check(status_ok(status) .and. all(predictions == [-4, -4, 9, 9]), &
        "score threshold and arbitrary labels", failures)
    call model%predict_proba(query, probabilities, status)
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2)-1.0_real64)) < &
        2.0e-14_real64 .and. all(probabilities > 0.0_real64), &
        "sigmoid probabilities", failures)

    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)), theta_bar(size(theta)))
    do i = 1, size(theta); theta_dot(i) = 0.01_real64*real(i, real64); end do
    call model%decision_function_jvp(query, theta_dot, query_dot, scores, scores_dot, status)
    call check(status_ok(status), "decision JVP", failures)
    theta_plus = theta + eps*theta_dot; theta_minus = theta - eps*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%decision_function(query+eps*query_dot, scores_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%decision_function(query-eps*query_dot, scores_minus, status)
    call model%set_parameters(theta, status)
    call check(maxval(abs(scores_dot-(scores_plus-scores_minus)/(2.0_real64*eps))) < 2.0e-5_real64, &
        "decision JVP finite-difference oracle", failures)
    call model%decision_function_vjp(query, scores_bar, theta_bar, x_bar, status)
    call check(status_ok(status) .and. abs(sum(scores_bar*scores_dot) - &
        (sum(theta_bar*theta_dot)+sum(x_bar*query_dot))) < 2.0e-10_real64, &
        "decision VJP adjoint oracle", failures)

    call model%predict_proba_jvp(query, theta_dot, query_dot, probabilities, probabilities_dot, status)
    call check(status_ok(status), "probability JVP", failures)
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(query+eps*query_dot, probabilities_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(query-eps*query_dot, probabilities_minus, status)
    call model%set_parameters(theta, status)
    call check(maxval(abs(probabilities_dot-(probabilities_plus-probabilities_minus)/(2.0_real64*eps))) < 2.0e-5_real64, &
        "probability JVP finite-difference oracle", failures)
    call model%predict_proba_vjp(query, probability_bar, theta_bar, x_bar, status)
    call check(status_ok(status), "probability VJP", failures)

    cuda%kind = FORTML_DEVICE_CUDA; cuda%selected = .true.; cuda%available = .true.
    call model%decision_function_device(cuda, query, scores, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA score refusal", failures)
    call model%predict_device(cuda, query, predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA prediction refusal", failures)
    call model%predict_proba_device(cuda, query, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA probability refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%decision_function_device(cpu, query, scores, status)
    call check(status_ok(status), "CPU device dispatch", failures)

    call invalid_model%fit(x, labels, status, degree=0)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid degree refusal", failures)
    call invalid_model%fit(x, labels, status, coef0=-1.0_real64)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "negative coef0 refusal", failures)
    call invalid_model%fit(x, [1,1,1,1,1,1,1,1], status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "single-class refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL polynomial SVM cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS polynomial SVM independent behavioral oracles"
contains
    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [polynomial-svm] "//description
        end if
    end subroutine check
end program test_polynomial_svm_classifier
