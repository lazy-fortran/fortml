program test_one_class_svm
    !! Independent RBF and dual-constraint checks for the one-class SVM.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_one_class_svm, only: one_class_svm_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(one_class_svm_t) :: model, strict_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(real64) :: x(2, 1), query(3, 1), scores(3), scores_dot(3)
    real(real64) :: scores_plus(3), scores_minus(3)
    real(real64) :: x_dot(3, 1), scores_bar(3), x_bar(3, 1), weights(2)
    real(real64) :: expected_rho, expected_query_score, eps
    integer :: labels(3), failures

    x(:, 1) = [0.0_real64, 1.0_real64]
    query(:, 1) = [0.25_real64, 0.75_real64, 2.0_real64]
    x_dot(:, 1) = [0.1_real64, -0.2_real64, 0.3_real64]
    scores_bar = [0.2_real64, -0.4_real64, 0.7_real64]
    failures = 0

    call model%fit(x, status, nu=0.5_real64, gamma=1.0_real64, &
        max_iterations=1000, tolerance=1.0e-12_real64)
    call check(status_ok(status) .and. model%fitted(), "RBF one-class fit", failures)
    weights = model%support_weights()
    call check(size(weights) == 2 .and. maxval(abs(weights - [0.5_real64, &
        0.5_real64])) < 1.0e-10_real64, "capped-simplex dual oracle", failures)
    call check(abs(sum(weights) - 1.0_real64) < 1.0e-12_real64 .and. &
        model%support_vector_count() == 2, "dual mass and support count", failures)

    expected_rho = 0.5_real64*(0.5_real64*(1.0_real64 + exp(-1.0_real64)) + &
        0.5_real64*(1.0_real64 + exp(-1.0_real64)))
    call check(abs(model%offset() - expected_rho) < 1.0e-10_real64, &
        "offset oracle", failures)
    call model%decision_function(query, scores, status)
    expected_query_score = 0.5_real64*(exp(-0.0625_real64) + &
        exp(-0.5625_real64)) - expected_rho
    call check(status_ok(status) .and. abs(scores(1) - expected_query_score) < &
        1.0e-10_real64, "RBF decision oracle", failures)
    call model%predict(query, labels, status)
    call check(status_ok(status) .and. all(labels == [1, 1, -1]), &
        "inlier/outlier prediction", failures)

    call model%decision_function_jvp(query, x_dot, scores, scores_dot, status)
    eps = 1.0e-6_real64
    call model%decision_function(query + eps*x_dot, scores_plus, status)
    call model%decision_function(query - eps*x_dot, scores_minus, status)
    call check(status_ok(status) .and. maxval(abs(scores_dot - &
        (scores_plus - scores_minus)/(2.0_real64*eps))) < 1.0e-7_real64, &
        "RBF decision JVP finite-difference oracle", failures)

    call model%decision_function_vjp(query, scores_bar, x_bar, status)
    call check(status_ok(status) .and. abs(sum(scores_bar*scores_dot) - &
        sum(x_bar*x_dot)) < 1.0e-10_real64, "decision VJP adjoint", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%decision_function_device(cuda, query, scores, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA decision refusal", failures)
    call model%predict_device(cuda, query, labels, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA prediction refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_device(cpu, query, labels, status)
    call check(status_ok(status) .and. all(labels == [1, 1, -1]), &
        "CPU device prediction", failures)

    call strict_model%fit(x, status, nu=0.0_real64)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid nu refusal", failures)
    call strict_model%fit(x, status, gamma=-1.0_real64)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid gamma refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL one-class SVM cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS one-class SVM independent behavioral oracles"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [one-class-svm] "//description
        end if
    end subroutine check

end program test_one_class_svm
