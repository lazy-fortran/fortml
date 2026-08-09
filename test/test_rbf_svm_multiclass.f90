program test_rbf_svm_multiclass
    !! Independent behavioral and fixed-state derivative oracles for OVR RBF SVM.
    use, intrinsic :: iso_fortran_env, only: error_unit, real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_rbf_svm_multiclass, only: rbf_svm_multiclass_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(rbf_svm_multiclass_t) :: model, invalid_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(9, 1), query(3, 1), query_dot(3, 1)
    real(real64) :: probabilities(3, 3), probabilities_dot(3, 3)
    real(real64) :: probabilities_plus(3, 3), probabilities_minus(3, 3)
    real(real64) :: probabilities_bar(3, 3), query_bar(3, 1)
    real(real64), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
    real(real64), allocatable :: theta_bar(:)
    integer :: labels(9), predictions(3), classes(3), failures, i
    real(real64) :: eps

    x(:, 1) = [-3.0_real64, -2.5_real64, -2.0_real64, 0.0_real64, 0.5_real64, &
        1.0_real64, 2.0_real64, 2.5_real64, 3.0_real64]
    labels = [7, 7, 7, -3, -3, -3, 42, 42, 42]
    query(:, 1) = [-2.2_real64, 0.4_real64, 2.2_real64]
    query_dot(:, 1) = [0.2_real64, -0.1_real64, 0.3_real64]
    probabilities_bar = reshape([0.4_real64, -0.2_real64, 0.3_real64, &
        -0.1_real64, 0.6_real64, -0.5_real64, 0.2_real64, 0.1_real64, &
        -0.3_real64], shape(probabilities_bar))
    failures = 0
    eps = 2.0e-6_real64

    call model%fit(x, labels, status, c=2.0_real64, gamma=0.6_real64, &
        max_iterations=2000, tolerance=1.0e-7_real64)
    call check(status_ok(status) .and. model%fitted(), &
        "transactional OVR fit", failures)
    classes = model%classes()
    call check(all(classes == [-3, 7, 42]), "sorted arbitrary labels", failures)
    call check(model%class_count() == 3 .and. model%feature_count() == 1, &
        "class and feature metadata", failures)
    call model%predict_proba(query, probabilities, status)
    call check(status_ok(status) .and. all(ieee_is_finite(probabilities)), &
        "finite multiclass probabilities", failures)
    call check(maxval(abs(sum(probabilities, dim=2) - 1.0_real64)) < 2.0e-14_real64, &
        "probability simplex normalization", failures)
    call model%predict(query, predictions, status)
    call check(status_ok(status), "multiclass label prediction", failures)
    do i = 1, size(predictions)
        call check(any(predictions(i) == classes), "prediction uses fitted labels", failures)
    end do

    call model%predict_proba_jvp(query, query_dot, probabilities, probabilities_dot, status)
    call check(status_ok(status), "input probability JVP status", failures)
    call model%predict_proba(query + eps*query_dot, probabilities_plus, status)
    call model%predict_proba(query - eps*query_dot, probabilities_minus, status)
    call check(maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_real64*eps))) < 3.0e-6_real64, &
        "input probability JVP finite-difference oracle", failures)
    call model%predict_proba_vjp(query, probabilities_bar, query_bar, status)
    call check(status_ok(status) .and. abs(sum(probabilities_bar*probabilities_dot) - &
        sum(query_bar*query_dot)) < 2.0e-9_real64, &
        "input probability VJP adjoint oracle", failures)

    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)), &
        theta_bar(size(theta)))
    do i = 1, size(theta)
        theta_dot(i) = 0.003_real64*real(i, real64)
    end do
    call model%predict_proba_parameter_jvp(query, theta_dot, probabilities, &
        probabilities_dot, status)
    call check(status_ok(status), "parameter probability JVP status", failures)
    theta_plus = theta + eps*theta_dot
    theta_minus = theta - eps*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(query, probabilities_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(query, probabilities_minus, status)
    call model%set_parameters(theta, status)
    call check(maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_real64*eps))) < 3.0e-6_real64, &
        "parameter probability JVP finite-difference oracle", failures)
    call model%predict_proba_parameter_vjp(query, probabilities_bar, theta_bar, status)
    call check(status_ok(status) .and. abs(sum(probabilities_bar*probabilities_dot) - &
        sum(theta_bar*theta_dot)) < 2.0e-9_real64, &
        "parameter probability VJP adjoint oracle", failures)

    call model%predict_proba(query, probabilities, status)
    call invalid_model%fit(x, labels, status, gamma=-1.0_real64)
    call check(.not. status_ok(status), "invalid option refusal", failures)
    call invalid_model%fit(x, [1, 1, 1, 1, 1, 1, 1, 1, 1], status)
    call check(.not. status_ok(status), "single-class refusal", failures)
    call model%predict_proba(query, probabilities_plus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities - probabilities_plus)) < &
        2.0e-14_real64, "refused fit leaves model unchanged", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, query, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA probability refusal", failures)
    call model%predict_device(cuda, query, predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA prediction refusal", failures)
    call model%predict_proba_parameter_jvp_device(cuda, query, theta_dot, probabilities, &
        probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA parameter JVP refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL RBF SVM multiclass cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS RBF SVM multiclass independent behavioral oracles"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [rbf-svm-multiclass] "//description
        end if
    end subroutine check

end program test_rbf_svm_multiclass
