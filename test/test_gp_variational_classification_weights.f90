program test_gp_variational_classification_weights
    !! Independent weighted-ELBO and device-boundary oracles.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_variational_classification, only: &
        gp_variational_classification_t
    use fortml_gp_variational_multiclass_classification, only: &
        gp_variational_multiclass_classification_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_variational_classification_t) :: model
    type(gp_variational_multiclass_classification_t) :: multiclass
    type(gp_variational_classification_t) :: binary
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    real(dp) :: x(5, 1), inducing(3, 1), weights(5), doubled(5)
    real(dp), allocatable :: parameters(:), shifted(:), gradient(:)
    real(dp) :: value, tangent, plus, minus, expected, divergence, h
    real(dp) :: weighted_value, weighted_expected, unweighted_value, unweighted_expected
    real(dp) :: multiclass_value, binary_value
    real(dp) :: direction(9), reference(9)
    integer :: labels(5), multiclass_labels(5), binary_labels(5)
    integer :: classes(3), failures, i, class_index

    x(:, 1) = [-1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp]
    inducing(:, 1) = [-0.8_dp, 0.1_dp, 0.9_dp]
    labels = [0, 0, 0, 1, 1]
    weights = [0.5_dp, 1.5_dp, 0.0_dp, 2.0_dp, 3.0_dp]
    doubled = 2.0_dp
    failures = 0
    kernel = make_rbf_kernel(1, 1.3_dp, 0.9_dp, status)
    call check(status_ok(status), "kernel constructor", failures)
    call model%initialize(inducing, kernel, 20, 20260808, status)
    call check(status_ok(status), "weighted variational initialization", failures)
    parameters = model%parameters()
    call check(size(parameters) == 9, "packed parameter count", failures)
    allocate(gradient(size(parameters)), shifted(size(parameters)))

    call model%elbo(x, labels, unweighted_value, status, &
        expected_log_likelihood=unweighted_expected, kl_value=divergence)
    call check(status_ok(status), "unweighted ELBO", failures)
    call model%elbo(x, labels, weighted_value, status, &
        expected_log_likelihood=weighted_expected, sample_weight=doubled)
    call check(status_ok(status), "uniform weighted ELBO", failures)
    call check(abs(weighted_expected - 2.0_dp*unweighted_expected) < 2.0e-12_dp, &
        "uniform weights scale likelihood", failures)
    call check(abs(weighted_value - (2.0_dp*unweighted_expected - divergence)) < 2.0e-12_dp, &
        "uniform weights preserve unscaled KL", failures)

    parameters = parameters + [0.12_dp, -0.08_dp, 0.04_dp, 0.05_dp, -0.03_dp, &
        0.02_dp, 0.06_dp, -0.05_dp, 0.03_dp]
    call model%set_parameters(parameters, status)
    call check(status_ok(status), "perturbed variational vector", failures)
    call model%elbo_gradient(x, labels, value, gradient, status, sample_weight=weights)
    call check(status_ok(status), "weighted ELBO gradient", failures)
    h = 2.0e-6_dp
    do i = 1, size(parameters)
        shifted = parameters
        shifted(i) = shifted(i) + h
        call model%set_parameters(shifted, status)
        call model%elbo(x, labels, plus, status, sample_weight=weights)
        shifted(i) = parameters(i) - h
        call model%set_parameters(shifted, status)
        call model%elbo(x, labels, minus, status, sample_weight=weights)
        reference(i) = (plus - minus)/(2.0_dp*h)
    end do
    call model%set_parameters(parameters, status)
    call check(maxval(abs(gradient - reference)) < 4.0e-5_dp, &
        "weighted ELBO gradient finite-difference oracle", failures)

    direction = [0.05_dp, -0.04_dp, 0.02_dp, 0.03_dp, -0.01_dp, 0.04_dp, &
        -0.03_dp, 0.02_dp, -0.05_dp]
    call model%elbo_jvp(x, labels, direction, value, tangent, status, &
        sample_weight=weights)
    call check(status_ok(status), "weighted ELBO JVP", failures)
    call model%set_parameters(parameters + h*direction, status)
    call model%elbo(x, labels, plus, status, sample_weight=weights)
    call model%set_parameters(parameters - h*direction, status)
    call model%elbo(x, labels, minus, status, sample_weight=weights)
    call model%set_parameters(parameters, status)
    call check(abs(tangent - (plus - minus)/(2.0_dp*h)) < 4.0e-5_dp, &
        "weighted ELBO JVP finite-difference oracle", failures)

    call model%elbo(x, labels, value, status, sample_weight=[1.0_dp, 2.0_dp])
    call check(.not. status_ok(status), "sample-weight shape refusal", failures)
    call model%elbo(x, labels, value, status, sample_weight=[1.0_dp, -1.0_dp, &
        1.0_dp, 1.0_dp, 1.0_dp])
    call check(.not. status_ok(status), "negative sample-weight refusal", failures)
    call model%elbo(x, labels, value, status, sample_weight=[0.0_dp, 0.0_dp, &
        0.0_dp, 0.0_dp, 0.0_dp])
    call check(.not. status_ok(status), "zero sample-weight mass refusal", failures)

    device%kind = FORTML_DEVICE_CPU
    device%selected = .true.
    device%available = .true.
    call model%elbo_device(device, x, labels, value, status, sample_weight=weights)
    call check(status_ok(status), "weighted CPU device dispatch", failures)
    call check(abs(value - (model_value(model, x, labels, weights))) < 1.0e-12_dp, &
        "weighted CPU device value", failures)
    device%kind = FORTML_DEVICE_CUDA
    call model%elbo_device(device, x, labels, value, status, sample_weight=weights)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "weighted CUDA ELBO refusal", failures)

    ! The OVR wrapper is checked against independently initialized binary
    ! models, so this verifies both class routing and shared row weighting.
    classes = [10, 20, 30]
    multiclass_labels = [10, 10, 20, 30, 30]
    call multiclass%initialize(inducing, classes, kernel, 20, 20260808, status)
    call check(status_ok(status), "weighted OVR initialization", failures)
    call multiclass%elbo(x, multiclass_labels, multiclass_value, status, &
        sample_weight=weights)
    call check(status_ok(status), "weighted OVR ELBO", failures)
    binary_value = 0.0_dp
    do class_index = 1, size(classes)
        call binary%initialize(inducing, kernel, 20, 20260808 + class_index - 1, status)
        call check(status_ok(status), "binary OVR reference initialization", failures)
        binary_labels = 0
        where (multiclass_labels == classes(class_index)) binary_labels = 1
        call binary%elbo(x, binary_labels, value, status, sample_weight=weights)
        call check(status_ok(status), "binary OVR reference ELBO", failures)
        binary_value = binary_value + value
    end do
    call check(abs(multiclass_value - binary_value) < 3.0e-12_dp, &
        "weighted OVR composition oracle", failures)
    device%kind = FORTML_DEVICE_CPU
    call multiclass%elbo_device(device, x, multiclass_labels, value, status, &
        sample_weight=weights)
    call check(status_ok(status), "weighted OVR CPU device dispatch", failures)
    device%kind = FORTML_DEVICE_CUDA
    call multiclass%elbo_device(device, x, multiclass_labels, value, status, &
        sample_weight=weights)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "weighted OVR CUDA ELBO refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL weighted variational GP classification cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS weighted variational GP classification independent oracles"

contains

    function model_value(current, x_data, label_data, row_weights) result(value)
        type(gp_variational_classification_t), intent(inout) :: current
        real(dp), intent(in) :: x_data(:, :), row_weights(:)
        integer, intent(in) :: label_data(:)
        real(dp) :: value
        type(fortnum_status_t) :: local_status

        call current%elbo(x_data, label_data, value, local_status, &
            sample_weight=row_weights)
        if (.not. status_ok(local_status)) value = huge(1.0_dp)
    end function model_value

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [weighted-variational-gp] "//description
        end if
    end subroutine check

end program test_gp_variational_classification_weights
