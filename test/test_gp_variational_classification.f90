program test_gp_variational_classification
    !! Independent finite-difference and device-contract oracles for the
    !! inducing-point Bernoulli variational GP objective.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_variational_classification, only: &
        gp_variational_classification_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_variational_classification_t) :: model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    real(dp) :: x(4, 1), inducing(2, 1), direction(5), gradient(5), reference(5)
    real(dp), allocatable :: lambda(:), shifted(:)
    real(dp) :: value, tangent, plus, minus, expected, divergence, h
    integer :: labels(4), failures, i

    x(:, 1) = [-0.9_dp, -0.2_dp, 0.35_dp, 1.0_dp]
    inducing(:, 1) = [-0.7_dp, 0.55_dp]
    labels = [0, 0, 1, 1]
    failures = 0
    kernel = make_rbf_kernel(1, 1.4_dp, 0.8_dp, status)
    call check(status_ok(status), "kernel constructor", failures)
    call model%initialize(inducing, kernel, 24, 20260807, status)
    call check(status_ok(status), "variational initialization", failures)
    call check(model%parameter_count() == 5, "packed parameter count", failures)

    call model%elbo(x, labels, value, status, expected, divergence)
    call check(status_ok(status), "prior ELBO status", failures)
    call check(abs(divergence) < 2.0e-10_dp, "prior KL is zero", failures)
    call check(abs(value - expected + divergence) < 2.0e-13_dp, &
        "ELBO decomposition", failures)

    lambda = model%parameters()
    lambda = lambda + [0.20_dp, -0.15_dp, 0.10_dp, 0.07_dp, -0.04_dp]
    call model%set_parameters(lambda, status)
    call check(status_ok(status), "set perturbed variational vector", failures)
    call model%elbo_gradient(x, labels, value, gradient, status)
    call check(status_ok(status), "ELBO gradient status", failures)
    allocate(shifted(size(lambda)))
    h = 2.0e-6_dp
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
    call check(maxval(abs(gradient - reference)) < 3.0e-5_dp, &
        "packed ELBO gradient finite-difference oracle", failures)

    direction = [0.07_dp, -0.11_dp, 0.03_dp, 0.08_dp, -0.05_dp]
    call model%elbo_jvp(x, labels, direction, value, tangent, status)
    call check(status_ok(status), "ELBO JVP status", failures)
    call model%set_parameters(lambda + h*direction, status)
    call model%elbo(x, labels, plus, status)
    call model%set_parameters(lambda - h*direction, status)
    call model%elbo(x, labels, minus, status)
    call model%set_parameters(lambda, status)
    call check(abs(tangent - (plus - minus)/(2.0_dp*h)) < 3.0e-5_dp, &
        "ELBO JVP directional oracle", failures)

    device%kind = FORTML_DEVICE_CPU
    device%selected = .true.
    device%available = .true.
    call model%elbo_device(device, x, labels, plus, status)
    call check(status_ok(status) .and. abs(plus - value) < 1.0e-13_dp, &
        "CPU device dispatch", failures)
    ! A minibatch may change its row count between objective calls; the fixed
    ! seed must rebuild the table without changing the packed state.
    call model%elbo(x(1:2, :), labels(1:2), plus, status, scale=2.0_dp)
    call check(status_ok(status), "variable-size minibatch dispatch", failures)
    device%kind = FORTML_DEVICE_CUDA
    call model%elbo_device(device, x, labels, plus, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA resident-graph refusal", failures)

    call model%elbo(x, [0, 1, 2, 1], plus, status)
    call check(.not. status_ok(status), "non-Bernoulli label refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL variational GP classification cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS variational GP classification independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [variational-gp-classification] "//description
        end if
    end subroutine check

end program test_gp_variational_classification
