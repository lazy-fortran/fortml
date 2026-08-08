program test_gp_variational_kernel_products
    !! Independent finite-difference and adjoint oracles for fixed-state
    !! variational-GP kernel hyperparameter products (binary and OVR).
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_variational_classification, only: gp_variational_classification_t
    use fortml_gp_variational_multiclass_classification, only: &
        gp_variational_multiclass_classification_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel, clone_kernel
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(gp_variational_classification_t) :: binary, binary_plus, binary_minus
    type(gp_variational_multiclass_classification_t) :: multi, multi_plus, multi_minus
    type(kernel_t) :: kernel, kernel_plus, kernel_minus
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: device
    real(dp) :: x(5, 1), inducing(2, 1), direction(2), h
    real(dp) :: mean(5), mean_dot(5), variance(5), variance_dot(5)
    real(dp) :: mean_plus(5), mean_minus(5), variance_plus(5), variance_minus(5)
    real(dp) :: mean_bar(5), variance_bar(5), parameter_bar(2)
    real(dp) :: probabilities(5, 2), probabilities_dot(5, 2)
    real(dp) :: probabilities_plus(5, 2), probabilities_minus(5, 2)
    real(dp) :: probabilities_bar(5, 2), probability_parameter_bar(2)
    real(dp) :: state(5), state_multi(15), parameters(2)
    real(dp) :: margins(5, 3), margins_dot(5, 3), variances(5, 3), variances_dot(5, 3)
    real(dp) :: probabilities_multi(5, 3), probabilities_multi_dot(5, 3)
    real(dp) :: probabilities_multi_plus(5, 3), probabilities_multi_minus(5, 3)
    real(dp) :: probabilities_multi_bar(5, 3), multi_parameter_bar(2)
    real(dp) :: labels_real(5)
    integer :: labels(5), classes(3), failures

    x(:, 1) = [-1.4_dp, -0.6_dp, -0.1_dp, 0.65_dp, 1.3_dp]
    inducing(:, 1) = [-0.9_dp, 0.8_dp]
    labels = [10, 10, 20, 20, 10]
    classes = [10, 20, 30]
    direction = [0.17_dp, -0.23_dp]
    mean_bar = [0.2_dp, -0.4_dp, 0.7_dp, -0.1_dp, 0.3_dp]
    variance_bar = [-0.3_dp, 0.6_dp, -0.2_dp, 0.5_dp, -0.1_dp]
    probabilities_bar(:, 1) = mean_bar
    probabilities_bar(:, 2) = variance_bar
    probabilities_multi_bar(:, 1) = [0.2_dp, -0.4_dp, 0.7_dp, -0.1_dp, 0.3_dp]
    probabilities_multi_bar(:, 2) = [-0.3_dp, 0.6_dp, -0.2_dp, 0.5_dp, -0.1_dp]
    probabilities_multi_bar(:, 3) = [0.1_dp, 0.05_dp, -0.25_dp, 0.3_dp, -0.2_dp]
    failures = 0
    h = 2.0e-6_dp

    kernel = make_rbf_kernel(1, 1.2_dp, 0.85_dp, status)
    call check(status_ok(status), "RBF constructor", failures)
    parameters = kernel%parameters()
    call binary%initialize(inducing, kernel, 24, 20260808, status)
    call check(status_ok(status), "binary initialization", failures)
    state = binary%parameters()
    state = state + [0.12_dp, -0.07_dp, 0.03_dp, -0.02_dp, 0.05_dp]
    call binary%set_parameters(state, status)
    call check(status_ok(status), "binary state", failures)
    call binary%predict_latent_kernel_parameter_jvp(x, direction, mean, mean_dot, &
        variance, variance_dot, status)
    call check(status_ok(status), "binary kernel latent JVP", failures)

    kernel_plus = clone_kernel(kernel)
    kernel_minus = clone_kernel(kernel)
    call kernel_plus%set_parameters(parameters + h*direction, status)
    call kernel_minus%set_parameters(parameters - h*direction, status)
    call binary_plus%initialize(inducing, kernel_plus, 24, 20260808, status)
    call binary_minus%initialize(inducing, kernel_minus, 24, 20260808, status)
    call binary_plus%set_parameters(state, status)
    call binary_minus%set_parameters(state, status)
    call binary_plus%predict_latent(x, mean_plus, variance_plus, status)
    call binary_minus%predict_latent(x, mean_minus, variance_minus, status)
    call check(maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*h))) < 3.0e-6_dp, &
        "binary latent kernel JVP finite difference", failures)
    call check(maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*h))) < 3.0e-6_dp, &
        "binary variance kernel JVP finite difference", failures)

    call binary%predict_latent_kernel_parameter_vjp(x, mean_bar, variance_bar, &
        parameter_bar, status)
    call check(status_ok(status) .and. abs(dot_product(parameter_bar, direction) - &
        (dot_product(mean_bar, mean_dot) + dot_product(variance_bar, variance_dot))) < 4.0e-6_dp, &
        "binary latent kernel JVP/VJP duality", failures)
    call binary%predict_proba_kernel_parameter_jvp(x, direction, probabilities, &
        probabilities_dot, status)
    call binary%predict_proba_kernel_parameter_vjp(x, probabilities_bar, &
        probability_parameter_bar, status)
    call check(status_ok(status) .and. abs(dot_product(probability_parameter_bar, direction) - &
        sum(probabilities_bar*probabilities_dot)) < 4.0e-6_dp, &
        "binary probability kernel JVP/VJP duality", failures)
    call binary_plus%predict_proba(x, probabilities_plus, status)
    call binary_minus%predict_proba(x, probabilities_minus, status)
    call check(maxval(abs(probabilities_dot - (probabilities_plus - probabilities_minus)/ &
        (2.0_dp*h))) < 4.0e-6_dp, "binary probability kernel JVP finite difference", failures)

    device%kind = FORTML_DEVICE_CUDA
    device%selected = .true.
    device%available = .true.
    call binary%predict_proba_kernel_parameter_vjp_device(device, x, probabilities_bar, &
        probability_parameter_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "binary CUDA kernel VJP refusal", failures)

    call multi%initialize(inducing, classes, kernel, 24, 20260808, status)
    call check(status_ok(status) .and. multi%kernel_parameter_count() == 2, &
        "multiclass initialization and kernel count", failures)
    state_multi = multi%parameters()
    state_multi = state_multi + [ &
        0.11_dp, -0.06_dp, 0.03_dp, -0.02_dp, 0.05_dp, &
        -0.08_dp, 0.04_dp, -0.01_dp, 0.02_dp, -0.03_dp, &
        0.06_dp, 0.03_dp, -0.04_dp, 0.01_dp, 0.02_dp]
    call multi%set_parameters(state_multi, status)
    call check(status_ok(status), "multiclass state", failures)
    call multi%predict_latent_kernel_parameter_jvp(x, direction, margins, margins_dot, &
        variances, variances_dot, status)
    call check(status_ok(status), "multiclass latent kernel JVP", failures)
    call multi%predict_proba_kernel_parameter_jvp(x, direction, probabilities_multi, &
        probabilities_multi_dot, status)
    call check(status_ok(status), "multiclass probability kernel JVP", failures)
    call check(maxval(abs(sum(probabilities_multi, dim=2) - 1.0_dp)) < 2.0e-13_dp, &
        "multiclass probability simplex", failures)
    call multi%predict_latent_kernel_parameter_vjp(x, margins_dot, variances_dot, &
        multi_parameter_bar, status)
    call check(status_ok(status) .and. abs(dot_product(multi_parameter_bar, direction) - &
        (sum(margins_dot*margins_dot) + sum(variances_dot*variances_dot))) < 4.0e-6_dp, &
        "multiclass latent kernel product duality", failures)
    call multi%predict_proba_kernel_parameter_vjp(x, probabilities_multi_bar, &
        multi_parameter_bar, status)
    call check(status_ok(status) .and. abs(dot_product(multi_parameter_bar, direction) - &
        sum(probabilities_multi_bar*probabilities_multi_dot)) < 4.0e-6_dp, &
        "multiclass probability kernel product duality", failures)

    call multi_plus%initialize(inducing, classes, kernel_plus, 24, 20260808, status)
    call multi_minus%initialize(inducing, classes, kernel_minus, 24, 20260808, status)
    call multi_plus%set_parameters(state_multi, status)
    call multi_minus%set_parameters(state_multi, status)
    call multi_plus%predict_proba(x, probabilities_multi_plus, status)
    call multi_minus%predict_proba(x, probabilities_multi_minus, status)
    call check(maxval(abs(probabilities_multi_dot - (probabilities_multi_plus - &
        probabilities_multi_minus)/(2.0_dp*h))) < 5.0e-6_dp, &
        "multiclass probability kernel JVP finite difference", failures)
    call multi%predict_proba_kernel_parameter_vjp_device(device, x, probabilities_multi_bar, &
        multi_parameter_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "multiclass CUDA kernel VJP refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL variational GP kernel-product cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS variational GP kernel-product independent oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [variational-gp-kernel] "//description
        end if
    end subroutine check

end program test_gp_variational_kernel_products
