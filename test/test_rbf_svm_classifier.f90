program test_rbf_svm_classifier
    !! Independent dense-kernel and derivative oracles for binary RBF SVM.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_rbf_svm_classifier, only: rbf_svm_classifier_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(rbf_svm_classifier_t) :: model, invalid_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(real64) :: x(6, 1), query(3, 1), query_dot(3, 1)
    real(real64) :: scores(3), scores_dot(3), scores_plus(3), scores_minus(3)
    real(real64) :: probabilities(3, 2), probabilities_dot(3, 2)
    real(real64) :: probabilities_plus(3, 2), probabilities_minus(3, 2)
    real(real64), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
    real(real64), allocatable :: theta_bar(:)
    real(real64) :: x_bar(3, 1), scores_bar(3), probability_bar(3, 2)
    real(real64) :: expected(3), d2, eps
    real(real64), allocatable :: coefficients(:)
    real(real64) :: zero_weights(6)
    integer :: labels(6), predictions(3), classes(2), failures, i, j

    x(:, 1) = [-2.0_real64, -1.0_real64, -0.4_real64, 0.4_real64, &
        1.0_real64, 2.0_real64]
    labels = [-3, -3, -3, 7, 7, 7]
    query(:, 1) = [-1.4_real64, 0.0_real64, 1.4_real64]
    query_dot(:, 1) = [0.2_real64, -0.3_real64, 0.1_real64]
    scores_bar = [0.3_real64, -0.4_real64, 0.7_real64]
    probability_bar(:, 1) = [0.2_real64, -0.1_real64, 0.4_real64]
    probability_bar(:, 2) = [-0.3_real64, 0.5_real64, 0.6_real64]
    failures = 0
    eps = 2.0e-6_real64
    zero_weights = 0.0_real64

    call model%fit(x, labels, status, c=3.0_real64, gamma=0.7_real64, &
        max_iterations=1000, tolerance=1.0e-9_real64, &
        sample_weight=[1.0_real64, 2.0_real64, 1.0_real64, 1.0_real64, &
            2.0_real64, 1.0_real64])
    call check(status_ok(status) .and. model%fitted(), &
        "weighted RBF SVM fit", failures)
    classes = model%classes()
    call check(all(classes == [-3, 7]), "sorted arbitrary class labels", failures)
    call model%decision_function(query, scores, status)
    call check(status_ok(status), "decision function", failures)
    coefficients = model%coefficients()
    expected = model%intercept()
    do i = 1, size(query, 1)
        do j = 1, size(x, 1)
            d2 = (query(i, 1) - x(j, 1))**2
            expected(i) = expected(i) + coefficients(j)*exp(-model%gamma()*d2)
        end do
    end do
    call check(maxval(abs(scores - expected)) < 2.0e-12_real64, &
        "dense RBF score oracle", failures)
    call model%predict(query, predictions, status)
    call check(status_ok(status) .and. all(predictions == [-3, 7, 7]), &
        "score threshold and arbitrary labels", failures)
    call model%predict_proba(query, probabilities, status)
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2) - 1.0_real64)) < &
        2.0e-14_real64 .and. all(probabilities > 0.0_real64), &
        "uncalibrated sigmoid probabilities", failures)

    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)), &
        theta_bar(size(theta)))
    do i = 1, size(theta)
        theta_dot(i) = 0.01_real64*real(i, real64)
    end do
    call model%decision_function_jvp(query, theta_dot, query_dot, scores, scores_dot, status)
    call check(status_ok(status), "decision JVP", failures)
    theta_plus = theta + eps*theta_dot
    theta_minus = theta - eps*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%decision_function(query + eps*query_dot, scores_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%decision_function(query - eps*query_dot, scores_minus, status)
    call model%set_parameters(theta, status)
    call check(maxval(abs(scores_dot - (scores_plus-scores_minus)/(2.0_real64*eps))) < &
        2.0e-6_real64, "decision JVP finite-difference oracle", failures)
    call model%decision_function_vjp(query, scores_bar, theta_bar, x_bar, status)
    call check(status_ok(status) .and. abs(sum(scores_bar*scores_dot) - &
        (sum(theta_bar*theta_dot) + sum(x_bar*query_dot))) < 2.0e-10_real64, &
        "decision VJP adjoint oracle", failures)

    call model%predict_proba_jvp(query, theta_dot, query_dot, probabilities, &
        probabilities_dot, status)
    call check(status_ok(status), "probability JVP", failures)
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(query + eps*query_dot, probabilities_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(query - eps*query_dot, probabilities_minus, status)
    call model%set_parameters(theta, status)
    call check(maxval(abs(probabilities_dot - &
        (probabilities_plus-probabilities_minus)/(2.0_real64*eps))) < 2.0e-6_real64, &
        "probability JVP finite-difference oracle", failures)
    call model%predict_proba_vjp(query, probability_bar, theta_bar, x_bar, status)
    call check(status_ok(status), "probability VJP", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%decision_function_device(cuda, query, scores, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA score refusal", failures)
    call model%predict_device(cuda, query, predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA prediction refusal", failures)
    call model%predict_proba_device(cuda, query, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA probability refusal", failures)
    call model%decision_function_jvp_device(cuda, query, theta_dot, query_dot, scores, &
        scores_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA decision JVP refusal", failures)
    call model%decision_function_vjp_device(cuda, query, scores_bar, theta_bar, x_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA decision VJP refusal", failures)
    call model%predict_proba_jvp_device(cuda, query, theta_dot, query_dot, probabilities, &
        probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA probability JVP refusal", failures)
    call model%predict_proba_vjp_device(cuda, query, probability_bar, theta_bar, x_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA probability VJP refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%decision_function_jvp_device(cpu, query, theta_dot, query_dot, scores, &
        scores_dot, status)
    call check(status_ok(status), "CPU decision JVP dispatch", failures)
    call model%predict_proba_jvp_device(cpu, query, theta_dot, query_dot, probabilities, &
        probabilities_dot, status)
    call check(status_ok(status), "CPU probability JVP dispatch", failures)
    call model%decision_function_vjp_device(cpu, query, scores_bar, theta_bar, x_bar, status)
    call check(status_ok(status), "CPU decision VJP dispatch", failures)
    call model%predict_proba_vjp_device(cpu, query, probability_bar, theta_bar, x_bar, status)
    call check(status_ok(status), "CPU probability VJP dispatch", failures)
    call model%predict_device(cpu, query, predictions, status)
    call check(status_ok(status) .and. all(predictions == [-3, 7, 7]), &
        "CPU device prediction", failures)

    call invalid_model%fit(x, labels, status, gamma=-1.0_real64)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid gamma refusal", failures)
    call invalid_model%fit(x, [1, 1, 1, 1, 1, 1], status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "single-class refusal", failures)
    call invalid_model%fit(x, labels, status, sample_weight=zero_weights)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero-mass weight refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL RBF SVM cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS RBF SVM independent behavioral oracles"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [rbf-svm] "//description
        end if
    end subroutine check

end program test_rbf_svm_classifier
