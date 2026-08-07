program test_gp_variational_multiclass_classification
    !! Independent behavioral oracles for OVR variational GP classification.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_variational_multiclass_classification, only: &
        gp_variational_multiclass_classification_t
    use fortml_gp_variational_classification, only: &
        gp_variational_classification_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_variational_multiclass_classification_t) :: model, duplicate_model
    type(gp_variational_classification_t) :: binary
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    real(dp) :: x(6, 1), inducing(2, 1), value, tangent, plus, minus, h
    real(dp) :: gradient(15), direction(15), reference(15)
    real(dp) :: probabilities(6, 3), probabilities_dot(6, 3)
    real(dp) :: probabilities_plus(6, 3), probabilities_minus(6, 3)
    real(dp) :: probabilities_bar(6, 3), parameter_bar(15), parameter_bar_fd(15)
    real(dp) :: objective_plus, objective_minus
    real(dp), allocatable :: lambda(:), shifted(:)
    integer :: labels(6), requested_classes(3), sorted_classes(3), predicted(6)
    integer :: binary_labels(6), failures, i, j
    real(dp) :: expected_sum, binary_value

    x(:, 1) = [-1.1_dp, -0.55_dp, -0.05_dp, 0.30_dp, 0.80_dp, 1.25_dp]
    inducing(:, 1) = [-0.8_dp, 0.75_dp]
    labels = [30, 10, 20, 10, 30, 20]
    requested_classes = [30, 10, 20]
    sorted_classes = [10, 20, 30]
    failures = 0

    kernel = make_rbf_kernel(1, 1.3_dp, 0.9_dp, status)
    call check(status_ok(status), "kernel constructor", failures)
    call model%initialize(inducing, requested_classes, kernel, 16, 20260807, status)
    call check(status_ok(status), "multiclass initialization", failures)
    call check(model%initialized(), "initialized flag", failures)
    call check(model%class_count() == 3 .and. model%feature_count() == 1, &
        "class and feature counts", failures)
    call check(all(model%classes() == sorted_classes), "sorted class labels", failures)
    call check(model%parameter_count() == 15, "packed per-class parameter count", failures)

    ! The OVR objective is the sum of three independent binary objectives.
    expected_sum = 0.0_dp
    do j = 1, 3
        binary_labels = merge(1, 0, labels == sorted_classes(j))
        call binary%initialize(inducing, kernel, 16, 20260807 + j - 1, status)
        call check(status_ok(status), "binary oracle initialization", failures)
        call binary%elbo(x, binary_labels, binary_value, status)
        call check(status_ok(status), "binary oracle ELBO", failures)
        expected_sum = expected_sum + binary_value
    end do
    call model%elbo(x, labels, value, status)
    call check(status_ok(status), "multiclass ELBO", failures)
    call check(abs(value - expected_sum) < 2.0e-12_dp, &
        "OVR ELBO sum oracle", failures)

    lambda = model%parameters()
    lambda = lambda + [ &
        0.10_dp, -0.06_dp, 0.03_dp, 0.02_dp, -0.01_dp, &
        -0.08_dp, 0.05_dp, -0.02_dp, 0.01_dp, 0.03_dp, &
        0.04_dp, 0.07_dp, -0.01_dp, -0.02_dp, 0.02_dp]
    call model%set_parameters(lambda, status)
    call check(status_ok(status), "set packed per-class parameters", failures)
    call model%elbo_gradient(x, labels, value, gradient, status)
    call check(status_ok(status), "multiclass ELBO gradient", failures)
    h = 2.0e-6_dp
    allocate(shifted(size(lambda)))
    do i = 1, size(lambda)
        shifted = lambda
        shifted(i) = shifted(i) + h
        call model%set_parameters(shifted, status)
        call model%elbo(x, labels, plus, status)
        shifted(i) = lambda(i) - h
        call model%set_parameters(shifted, status)
        call model%elbo(x, labels, minus, status)
        reference(i) = (plus - minus)/(2.0_dp*h)
    end do
    call model%set_parameters(lambda, status)
    call check(maxval(abs(gradient - reference)) < 5.0e-5_dp, &
        "packed multiclass gradient finite-difference oracle", failures)

    direction = [ &
        0.04_dp, -0.03_dp, 0.02_dp, 0.01_dp, -0.02_dp, &
        -0.02_dp, 0.05_dp, -0.01_dp, 0.03_dp, 0.02_dp, &
        0.01_dp, 0.02_dp, -0.04_dp, 0.01_dp, 0.03_dp]
    call model%elbo_jvp(x, labels, direction, value, tangent, status)
    call check(status_ok(status), "multiclass ELBO JVP", failures)
    call model%set_parameters(lambda + h*direction, status)
    call model%elbo(x, labels, plus, status)
    call model%set_parameters(lambda - h*direction, status)
    call model%elbo(x, labels, minus, status)
    call model%set_parameters(lambda, status)
    call check(abs(tangent - (plus - minus)/(2.0_dp*h)) < 5.0e-5_dp, &
        "multiclass ELBO JVP directional oracle", failures)

    call model%predict_proba(x, probabilities, status)
    call check(status_ok(status), "multiclass probability prediction", failures)
    call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-13_dp, &
        "probability simplex normalization", failures)
    call check(all(probabilities >= 0.0_dp) .and. all(probabilities <= 1.0_dp), &
        "probability bounds", failures)
    call model%predict(x, predicted, status)
    call check(status_ok(status), "multiclass label prediction", failures)
    do i = 1, size(predicted)
        call check(any(predicted(i) == sorted_classes), "predicted label class set", failures)
    end do

    call model%predict_proba_parameter_jvp(x, direction, probabilities, &
        probabilities_dot, status)
    call check(status_ok(status), "probability parameter JVP", failures)
    call model%set_parameters(lambda + h*direction, status)
    call model%predict_proba(x, probabilities_plus, status)
    call model%set_parameters(lambda - h*direction, status)
    call model%predict_proba(x, probabilities_minus, status)
    call model%set_parameters(lambda, status)
    call check(maxval(abs(probabilities_dot - (probabilities_plus - probabilities_minus)/ &
        (2.0_dp*h))) < 5.0e-5_dp, "probability JVP finite-difference oracle", failures)

    probabilities_bar(:, 1) = [0.17_dp, -0.09_dp, 0.04_dp, 0.12_dp, -0.06_dp, 0.08_dp]
    probabilities_bar(:, 2) = [-0.05_dp, 0.11_dp, -0.08_dp, 0.07_dp, 0.03_dp, -0.02_dp]
    probabilities_bar(:, 3) = [0.02_dp, 0.06_dp, 0.09_dp, -0.04_dp, 0.05_dp, 0.01_dp]
    call model%predict_proba_parameter_vjp(x, probabilities_bar, parameter_bar, status)
    call check(status_ok(status), "multiclass probability parameter VJP", failures)
    do i = 1, size(parameter_bar)
        shifted = lambda
        shifted(i) = shifted(i) + h
        call model%set_parameters(shifted, status)
        call model%predict_proba(x, probabilities_plus, status)
        objective_plus = sum(probabilities_plus*probabilities_bar)
        shifted(i) = lambda(i) - h
        call model%set_parameters(shifted, status)
        call model%predict_proba(x, probabilities_minus, status)
        objective_minus = sum(probabilities_minus*probabilities_bar)
        parameter_bar_fd(i) = (objective_plus - objective_minus)/(2.0_dp*h)
    end do
    call model%set_parameters(lambda, status)
    call check(maxval(abs(parameter_bar - parameter_bar_fd)) < 6.0e-5_dp, &
        "multiclass probability VJP finite-difference oracle", failures)
    call check(abs(dot_product(parameter_bar, direction) - sum(probabilities_dot* &
        probabilities_bar)) < 6.0e-5_dp, &
        "multiclass probability JVP/VJP dot-product identity", failures)
    call model%predict_proba_parameter_vjp(x(1:5, :), probabilities_bar, parameter_bar, status)
    call check(.not. status_ok(status), "multiclass malformed VJP cotangent refusal", failures)

    device%kind = FORTML_DEVICE_CPU
    device%selected = .true.
    device%available = .true.
    call model%predict_proba_device(device, x, probabilities_plus, status)
    call check(status_ok(status), "CPU device prediction dispatch", failures)
    call model%predict_proba_parameter_vjp_device(device, x, probabilities_bar, parameter_bar, status)
    call check(status_ok(status), "CPU multiclass probability VJP dispatch", failures)
    device%kind = FORTML_DEVICE_CUDA
    call model%predict_proba_device(device, x, probabilities_plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA resident multiclass refusal", failures)
    call model%predict_proba_parameter_vjp_device(device, x, probabilities_bar, parameter_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA multiclass probability VJP refusal", failures)
    call model%elbo_device(device, x, labels, value, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA resident multiclass ELBO refusal", failures)

    call model%elbo(x, [30, 10, 99, 10, 30, 20], value, status)
    call check(.not. status_ok(status), "unknown label refusal", failures)
    call duplicate_model%initialize(inducing, [10, 10, 20], kernel, 16, 3, status)
    call check(.not. status_ok(status), "duplicate class refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL variational GP multiclass cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS variational GP multiclass independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') &
                "  FAIL [variational-gp-multiclass] "//description
        end if
    end subroutine check

end program test_gp_variational_multiclass_classification
