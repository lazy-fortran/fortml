program test_gp_classification_parameter_products
    !! Independent duality checks for fixed-state Laplace-GP parameter products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, GP_LIKELIHOOD_PROBIT
    use fortml_kernels, only: kernel_t, make_rbf_kernel, clone_kernel
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(gp_classification_t) :: logistic_model, probit_model
    type(gp_classification_options_t) :: options
    type(kernel_t) :: kernel
    type(kernel_t) :: kernel_plus, kernel_minus
    type(cholesky_factorization_t) :: factor
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), query(4, 1), direction(2)
    real(dp) :: mean(4), mean_dot(4), variance(4), variance_dot(4)
    real(dp) :: mean_bar(4), variance_bar(4), probabilities(4, 2)
    real(dp) :: probabilities_dot(4, 2), probabilities_bar(4, 2)
    real(dp) :: train_mean(8), train_variance(8), alpha(8)
    real(dp) :: train_covariance(8, 8), cross_plus(8, 4), cross_minus(8, 4)
    real(dp) :: mean_plus(4), mean_minus(4), finite_difference(4), h
    real(dp), allocatable :: parameter_bar(:), parameters(:)
    integer :: labels(8), failures

    x(:, 1) = [-1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, &
        0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    labels = [-7, -7, -7, -7, 11, 11, 11, 11]
    query(:, 1) = [-1.2_dp, -0.3_dp, 0.3_dp, 1.2_dp]
    direction = [0.17_dp, -0.23_dp]
    mean_bar = [0.2_dp, -0.4_dp, 0.7_dp, -0.1_dp]
    variance_bar = [-0.3_dp, 0.6_dp, -0.2_dp, 0.5_dp]
    probabilities_bar(:, 1) = mean_bar
    probabilities_bar(:, 2) = variance_bar
    failures = 0

    kernel = make_rbf_kernel(1, 1.4_dp, 0.7_dp, status)
    options%max_iterations = 100
    options%tolerance = 1.0e-9_dp
    options%jitter = 1.0e-7_dp
    call logistic_model%fit(x, labels, kernel, status, options)
    call check(status_ok(status), "logistic fit", failures)
    allocate(parameter_bar(logistic_model%parameter_count()))
    call logistic_model%predict_latent_parameter_jvp(query, direction, mean, mean_dot, &
        variance, variance_dot, status)
    call logistic_model%predict_latent_parameter_vjp(query, mean_bar, variance_bar, &
        parameter_bar, status)
    call check(status_ok(status) .and. abs(dot_product(parameter_bar, direction) - &
        (dot_product(mean_bar, mean_dot) + dot_product(variance_bar, variance_dot))) &
        < 3.0e-7_dp, "logistic latent parameter JVP/VJP duality", failures)

    ! Independent fixed-state finite difference: recover the fitted alpha from
    ! the public training prediction and kernel matrix, then perturb only the
    ! cross-covariance kernel blocks while keeping alpha/W fixed.
    call logistic_model%predict_latent(x, train_mean, train_variance, status)
    call kernel%matrix(x, x, train_covariance, status)
    call factor%factorize(train_covariance, status)
    alpha = train_mean
    call factor%solve(alpha, status)
    parameters = logistic_model%parameters()
    h = 1.0e-6_dp
    kernel_plus = clone_kernel(kernel)
    kernel_minus = clone_kernel(kernel)
    call kernel_plus%set_parameters(parameters + h*direction, status)
    call kernel_minus%set_parameters(parameters - h*direction, status)
    call kernel_plus%matrix(x, query, cross_plus, status)
    call kernel_minus%matrix(x, query, cross_minus, status)
    mean_plus = matmul(transpose(cross_plus), alpha)
    mean_minus = matmul(transpose(cross_minus), alpha)
    finite_difference = (mean_plus - mean_minus)/(2.0_dp*h)
    call check(status_ok(status) .and. maxval(abs(mean_dot - finite_difference)) < 2.0e-7_dp, &
        "latent parameter JVP fixed-state finite difference", failures)

    call logistic_model%predict_proba_parameter_jvp(query, direction, probabilities, &
        probabilities_dot, status)
    call logistic_model%predict_proba_parameter_vjp(query, probabilities_bar, &
        parameter_bar, status)
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) &
        < 2.0e-14_dp .and. abs(dot_product(parameter_bar, direction) - &
        sum(probabilities_bar*probabilities_dot)) < 3.0e-7_dp, &
        "logistic probability parameter JVP/VJP duality", failures)

    options%likelihood = GP_LIKELIHOOD_PROBIT
    call probit_model%fit(x, labels, kernel, status, options)
    call probit_model%predict_proba_parameter_jvp(query, direction, probabilities, &
        probabilities_dot, status)
    call probit_model%predict_proba_parameter_vjp(query, probabilities_bar, &
        parameter_bar, status)
    call check(status_ok(status) .and. abs(dot_product(parameter_bar, direction) - &
        sum(probabilities_bar*probabilities_dot)) < 4.0e-7_dp, &
        "probit probability parameter JVP/VJP duality", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL GP classification parameter-product cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP classification parameter-product independent oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-classification-parameter] "//name
        end if
    end subroutine check

end program test_gp_classification_parameter_products
