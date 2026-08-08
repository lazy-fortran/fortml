program test_gp_variational_classification_input
    !! Independent finite-difference and adjoint oracles for variational-GP
    !! query-coordinate products (the fitted inducing state is fixed).
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_gp_variational_classification, only: &
        gp_variational_classification_t, GP_VARIATIONAL_PROBIT
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_variational_classification_t) :: model, probit_model
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    real(real64) :: inducing(2, 1), query(3, 1), query_dot(3, 1)
    real(real64) :: mean(3), variance(3), mean_dot(3), variance_dot(3)
    real(real64) :: mean_plus(3), mean_minus(3), variance_plus(3), variance_minus(3)
    real(real64) :: mean_bar(3), variance_bar(3), query_bar(3, 1)
    real(real64) :: probabilities(3, 2), probabilities_dot(3, 2)
    real(real64) :: probabilities_plus(3, 2), probabilities_minus(3, 2)
    real(real64) :: probabilities_bar(3, 2), query_bar_fd(3, 1)
    real(real64) :: query_plus(3, 1), query_minus(3, 1)
    real(real64) :: objective_plus, objective_minus, h
    integer :: failures, i

    inducing(:, 1) = [-0.75_real64, 0.55_real64]
    query(:, 1) = [-1.1_real64, 0.05_real64, 1.2_real64]
    query_dot(:, 1) = [0.13_real64, -0.22_real64, 0.09_real64]
    mean_bar = [0.12_real64, -0.08_real64, 0.06_real64]
    variance_bar = [-0.04_real64, 0.07_real64, 0.03_real64]
    probabilities_bar(:, 1) = [0.14_real64, -0.05_real64, 0.08_real64]
    probabilities_bar(:, 2) = [-0.03_real64, 0.11_real64, 0.06_real64]
    h = 2.0e-6_real64
    failures = 0

    kernel = make_rbf_kernel(1, 1.4_real64, 0.8_real64, status)
    call check(status_ok(status), "RBF kernel constructor", failures)
    call model%initialize(inducing, kernel, 24, 20260808, status)
    call check(status_ok(status), "logistic variational initialization", failures)
    call model%set_parameters([0.18_real64, -0.12_real64, 0.08_real64, &
        0.06_real64, -0.05_real64], status)
    call check(status_ok(status), "set logistic variational state", failures)

    call model%predict_latent_input_jvp(query, query_dot, mean, mean_dot, &
        variance, variance_dot, status)
    call check(status_ok(status), "latent input JVP status", failures)
    call model%predict_latent(query + h*query_dot, mean_plus, variance_plus, status)
    call model%predict_latent(query - h*query_dot, mean_minus, variance_minus, status)
    call check(maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_real64*h))) < &
        3.0e-6_real64 .and. maxval(abs(variance_dot - &
        (variance_plus - variance_minus)/(2.0_real64*h))) < 3.0e-6_real64, &
        "latent input JVP finite-difference oracle", failures)
    call model%predict_latent_input_vjp(query, mean_bar, variance_bar, query_bar, status)
    call check(status_ok(status) .and. abs(dot_product(mean_bar, mean_dot) + &
        dot_product(variance_bar, variance_dot) - sum(query_bar*query_dot)) < &
        2.0e-8_real64, "latent input VJP adjoint oracle", failures)

    call model%predict_proba_input_jvp(query, query_dot, probabilities, &
        probabilities_dot, status)
    call check(status_ok(status), "probability input JVP status", failures)
    call model%predict_proba(query + h*query_dot, probabilities_plus, status)
    call model%predict_proba(query - h*query_dot, probabilities_minus, status)
    call check(maxval(abs(probabilities_dot - (probabilities_plus - probabilities_minus)/ &
        (2.0_real64*h))) < 3.0e-6_real64, &
        "probability input JVP finite-difference oracle", failures)
    call model%predict_proba_input_vjp(query, probabilities_bar, query_bar, status)
    call check(status_ok(status) .and. abs(sum(probabilities_bar*probabilities_dot) - &
        sum(query_bar*query_dot)) < 2.0e-8_real64, &
        "probability input VJP adjoint oracle", failures)

    ! The same input products must carry the probit Gaussian-CDF branch.
    call probit_model%initialize(inducing, kernel, 24, 20260808, status, &
        likelihood=GP_VARIATIONAL_PROBIT)
    call check(status_ok(status), "probit variational initialization", failures)
    call probit_model%set_parameters(model%parameters(), status)
    call probit_model%predict_proba_input_jvp(query, query_dot, probabilities, &
        probabilities_dot, status)
    call probit_model%predict_proba(query + h*query_dot, probabilities_plus, status)
    call probit_model%predict_proba(query - h*query_dot, probabilities_minus, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_real64*h))) < 3.0e-6_real64, &
        "probit probability input JVP oracle", failures)

    ! Reverse products also agree with an independently assembled scalar
    ! finite-difference objective over each query coordinate.
    call model%predict_proba_input_vjp(query, probabilities_bar, query_bar, status)
    query_bar_fd = 0.0_real64
    ! Check the VJP against central differences one coordinate at a time.
    do i = 1, 3
        query_plus = query
        query_minus = query
        query_plus(i, 1) = query_plus(i, 1) + h
        query_minus(i, 1) = query_minus(i, 1) - h
        call model%predict_proba(query_plus, probabilities_plus, status)
        objective_plus = sum(probabilities_plus*probabilities_bar)
        call model%predict_proba(query_minus, probabilities_minus, status)
        objective_minus = sum(probabilities_minus*probabilities_bar)
        query_bar_fd(i, 1) = (objective_plus - objective_minus)/(2.0_real64*h)
    end do
    call check(maxval(abs(query_bar - query_bar_fd)) < 3.0e-6_real64, &
        "probability input VJP finite-difference oracle", failures)

    device%kind = FORTML_DEVICE_CPU
    device%selected = .true.
    device%available = .true.
    call model%predict_proba_input_vjp_device(device, query, probabilities_bar, &
        query_bar, status)
    call check(status_ok(status), "CPU input VJP dispatch", failures)
    device%kind = FORTML_DEVICE_CUDA
    call model%predict_proba_input_vjp_device(device, query, probabilities_bar, &
        query_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA input VJP refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL variational GP input cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS variational GP input independent oracles"

contains

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [variational-gp-input] "//description
        end if
    end subroutine check

end program test_gp_variational_classification_input
