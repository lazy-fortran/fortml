program test_robust_gp_poisson_products
    !! Independent products for the robust-GP Poisson observation path.
    !!
    !! The likelihood oracle is written directly from
    !! ``y*log(rate)-rate-log_gamma(y+1)``.  Posterior and query products are
    !! checked independently by central differences and the VJP/JVP adjoint
    !! identity; no implementation helper is used to construct the oracle.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_robust_gp, only: robust_gp_t, FORTML_LIKELIHOOD_POISSON, &
        robust_poisson_log_likelihood_value, robust_poisson_log_likelihood_gradient, &
        robust_poisson_log_likelihood_jvp, robust_poisson_log_likelihood_vjp, &
        robust_poisson_log_likelihood_hvp
    use fortml_robust_gp_training, only: robust_gp_poisson_objective_t, &
        robust_gp_poisson_lbfgsb_options_t, robust_gp_poisson_lbfgsb_result_t, &
        robust_gp_poisson_optimize
    implicit none

    integer :: failures

    failures = 0
    call test_likelihood_products(failures)
    call test_posterior_products(failures)
    call test_query_products(failures)
    call test_optimizer_and_refusal(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " robust Poisson product test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: robust Poisson GP products"

contains

    subroutine test_likelihood_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: counts(4), log_rate(4), direction(4), gradient(4), product(4)
        real(dp) :: gradient_bar(4), value, tangent, value_bar, oracle, oracle_dot
        type(fortnum_status_t) :: status
        integer :: i

        counts = [0.0_dp, 1.0_dp, 3.0_dp, 2.0_dp]
        log_rate = [-0.3_dp, 0.2_dp, 0.8_dp, -0.1_dp]
        direction = [0.4_dp, -0.2_dp, 0.1_dp, 0.3_dp]
        oracle = 0.0_dp
        oracle_dot = 0.0_dp
        do i = 1, 4
            oracle = oracle + counts(i)*log_rate(i) - exp(log_rate(i)) - &
                log_gamma(counts(i) + 1.0_dp)
            oracle_dot = oracle_dot + (counts(i) - exp(log_rate(i)))*direction(i)
        end do
        call robust_poisson_log_likelihood_value(counts, log_rate, value, status)
        call check(status_ok(status), "likelihood value status", failures)
        call check(abs(value - oracle) < 2.0e-14_dp, "likelihood value oracle", failures)
        call robust_poisson_log_likelihood_gradient(counts, log_rate, gradient, status)
        call check(status_ok(status), "likelihood gradient status", failures)
        call check(maxval(abs(gradient - counts + exp(log_rate))) < 2.0e-14_dp, &
            "likelihood gradient oracle", failures)
        call robust_poisson_log_likelihood_jvp(counts, log_rate, direction, value, tangent, status)
        call check(status_ok(status), "likelihood JVP status", failures)
        call check(abs(value - oracle) < 2.0e-14_dp .and. abs(tangent - oracle_dot) < 2.0e-14_dp, &
            "likelihood JVP oracle", failures)
        value_bar = -1.7_dp
        call robust_poisson_log_likelihood_vjp(counts, log_rate, value_bar, gradient_bar, status)
        call check(status_ok(status), "likelihood VJP status", failures)
        call check(abs(dot_product(gradient_bar, direction) - value_bar*oracle_dot) < 2.0e-14_dp, &
            "likelihood VJP/JVP adjoint", failures)
        call robust_poisson_log_likelihood_hvp(counts, log_rate, direction, product, status)
        call check(status_ok(status), "likelihood HVP status", failures)
        call check(maxval(abs(product + exp(log_rate)*direction)) < 2.0e-14_dp, &
            "likelihood HVP oracle", failures)
    end subroutine test_likelihood_products

    subroutine make_fixture(model, status)
        type(robust_gp_t), intent(out) :: model
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel
        real(dp) :: x(6, 1), counts(6)
        integer :: i

        do i = 1, 6
            x(i, 1) = -1.25_dp + 0.5_dp*real(i - 1, dp)
            counts(i) = real(max(0, nint(2.0_dp*exp(0.55_dp*x(i, 1)))), dp)
        end do
        kernel = make_rbf_kernel(1, 1.2_dp, 0.9_dp, status)
        if (.not. status_ok(status)) return
        call model%fit(x, counts, kernel, FORTML_LIKELIHOOD_POISSON, status)
    end subroutine make_fixture

    subroutine test_posterior_products(failures)
        integer, intent(inout) :: failures
        type(robust_gp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: value, tangent, value_bar, gradient(6), direction(6), product(6)
        real(dp) :: plus(6), minus(6), gradient_plus(6), gradient_minus(6), fd(6)
        integer :: i

        call make_fixture(model, status)
        call check(status_ok(status), "Poisson fixture fits", failures)
        call model%log_posterior(value, status)
        call check(status_ok(status), "posterior value status", failures)
        call model%log_posterior_gradient(gradient, status)
        call check(status_ok(status), "posterior gradient status", failures)
        call check(maxval(abs(gradient)) < 2.0e-6_dp, &
            "Newton mode satisfies posterior stationarity", failures)
        direction = [0.3_dp, -0.2_dp, 0.1_dp, 0.4_dp, -0.25_dp, 0.15_dp]
        call model%log_posterior_jvp(direction, value, tangent, status)
        call check(status_ok(status), "posterior JVP status", failures)
        call check(abs(tangent - dot_product(gradient, direction)) < 2.0e-14_dp, &
            "posterior JVP oracle", failures)
        value_bar = -0.7_dp
        call model%log_posterior_vjp(value_bar, product, status)
        call check(status_ok(status), "posterior VJP status", failures)
        call check(maxval(abs(product - value_bar*gradient)) < 2.0e-14_dp, &
            "posterior VJP oracle", failures)
        call model%log_posterior_hvp(direction, product, status)
        call check(status_ok(status), "posterior HVP status", failures)
        plus = model%latent_parameters() + 1.0e-5_dp*direction
        minus = model%latent_parameters() - 1.0e-5_dp*direction
        call model%set_latent_parameters(plus, status)
        call model%log_posterior_gradient(gradient_plus, status)
        call model%set_latent_parameters(minus, status)
        call model%log_posterior_gradient(gradient_minus, status)
        call model%set_latent_parameters(0.5_dp*(plus + minus), status)
        fd = (gradient_plus - gradient_minus)/(2.0e-5_dp)
        call check(maxval(abs(product - fd)) < 3.0e-6_dp, &
            "posterior HVP finite-difference oracle", failures)
    end subroutine test_posterior_products

    subroutine test_query_products(failures)
        integer, intent(inout) :: failures
        type(robust_gp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: query(2, 1), query_dot(2, 1), mean(2), mean_dot(2), variance(2), variance_dot(2)
        real(dp) :: response(2), response_dot(2), query_bar(2, 1), x_plus(2, 1), x_minus(2, 1)
        real(dp) :: mean_plus(2), mean_minus(2), variance_plus(2), variance_minus(2)
        real(dp) :: response_bar(2), response_x_bar(2, 1), fd(2, 1)
        integer :: j

        call make_fixture(model, status)
        query(:, 1) = [-0.65_dp, 0.8_dp]
        query_dot(:, 1) = [0.2_dp, -0.35_dp]
        call model%predict_latent_jvp(query, query_dot, mean, mean_dot, variance, variance_dot, status)
        call check(status_ok(status), "latent query JVP status", failures)
        x_plus = query + 1.0e-5_dp*query_dot
        x_minus = query - 1.0e-5_dp*query_dot
        call model%predict_latent(x_plus, mean_plus, variance_plus, status)
        call model%predict_latent(x_minus, mean_minus, variance_minus, status)
        call check(maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0e-5_dp))) < 2.0e-5_dp, &
            "latent mean input JVP oracle", failures)
        call check(maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0e-5_dp))) < 2.0e-5_dp, &
            "latent variance input JVP oracle", failures)
        response_bar = [0.7_dp, -0.4_dp]
        call model%predict_response_vjp(query, response_bar, response_x_bar, status)
        call check(status_ok(status), "response query VJP status", failures)
        query_bar = 0.0_dp
        call model%predict_response_jvp(query, query_bar, response, response_dot, status)
        call check(status_ok(status), "response zero JVP status", failures)
        call model%predict_response(query, response, status)
        call model%predict_response_jvp(query, response_x_bar, response, response_dot, status)
        call check(status_ok(status), "response directional JVP status", failures)
        call check(abs(dot_product(response_bar, response_dot) - sum(response_x_bar*response_x_bar)) < 1.0e-7_dp, &
            "response VJP/JVP adjoint", failures)
        x_plus = query + 1.0e-5_dp*response_x_bar
        x_minus = query - 1.0e-5_dp*response_x_bar
        call model%predict_response(x_plus, mean_plus, status)
        call model%predict_response(x_minus, mean_minus, status)
        fd = 0.0_dp
        fd(1, 1) = dot_product(response_bar, (mean_plus - mean_minus)/(2.0e-5_dp))
        call check(abs(fd(1, 1) - sum(response_x_bar*response_x_bar)) < 1.0e-4_dp, &
            "response VJP finite-difference oracle", failures)
    end subroutine test_query_products

    subroutine test_optimizer_and_refusal(failures)
        integer, intent(inout) :: failures
        type(robust_gp_t) :: model
        type(robust_gp_poisson_objective_t) :: objective
        type(robust_gp_poisson_lbfgsb_options_t) :: options
        type(robust_gp_poisson_lbfgsb_result_t) :: result
        type(fortnum_status_t) :: status
        real(dp) :: product(6)

        call make_fixture(model, status)
        call objective%initialize(model, status)
        call check(status_ok(status), "Poisson objective initialization", failures)
        call check(objective%parameter_count() == 6, "Poisson objective parameter count", failures)
        call objective%hvp(objective%parameters(), [0.1_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp], &
            product, status)
        call check(status_ok(status), "Poisson objective HVP", failures)
        options%max_iterations = 30
        call robust_gp_poisson_optimize(model, options, result, status)
        call check(status_ok(status), "FortOpt Poisson optimization", failures)
        call check(result%converged, "FortOpt Poisson reports convergence", failures)
        options%device_kind = FORTML_DEVICE_CUDA
        call robust_gp_poisson_optimize(model, options, result, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "CUDA Poisson optimization is a typed refusal", failures)
    end subroutine test_optimizer_and_refusal

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//description//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_robust_gp_poisson_products
