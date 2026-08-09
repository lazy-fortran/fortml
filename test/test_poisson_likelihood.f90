program test_poisson_likelihood
    !! Independent oracle for Poisson GP likelihood value and derivative products.
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_poisson_likelihood, only: poisson_likelihood_t, &
        poisson_likelihood_lbfgsb_options_t, poisson_likelihood_lbfgsb_result_t, &
        poisson_log_likelihood_value, poisson_log_likelihood_value_gradient, &
        poisson_log_likelihood_jvp, poisson_log_likelihood_vjp, &
        poisson_log_likelihood_hvp
    implicit none

    integer :: failures

    failures = 0
    call test_joint_products(failures)
    call test_objective_and_optimizer(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " Poisson likelihood test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: Poisson likelihood"

contains

    subroutine fixture(observations, latents, weights)
        real(dp), intent(out) :: observations(6), latents(6), weights(6)

        observations = [0.0_dp, 1.0_dp, 2.0_dp, 4.0_dp, 3.0_dp, 5.0_dp]
        latents = log([0.6_dp, 1.0_dp, 1.4_dp, 2.8_dp, 4.1_dp, 2.0_dp])
        weights = [1.0_dp, 0.5_dp, 2.0_dp, 1.5_dp, 0.0_dp, 0.75_dp]
    end subroutine fixture

    subroutine test_joint_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: observations(6), latents(6), weights(6)
        real(dp) :: latent_gradient(6), latent_direction(6), latent_bar(6)
        real(dp) :: latent_product(6), reference_gradient(7), reference_hvp(7)
        real(dp) :: coordinates(7), direction(7), plus(7), minus(7)
        real(dp) :: value, oracle, offset_gradient, tangent, offset_bar, offset_product
        real(dp) :: value_bar, h_gradient, h_direction
        type(fortnum_status_t) :: status

        call fixture(observations, latents, weights)
        coordinates(1:6) = latents
        coordinates(7) = log(1.3_dp)
        direction = [0.3_dp, -0.2_dp, 0.1_dp, 0.45_dp, -0.4_dp, 0.25_dp, -0.35_dp]
        latent_direction = direction(1:6)
        h_gradient = 2.0e-6_dp
        h_direction = 2.0e-4_dp

        call poisson_log_likelihood_value_gradient(observations, latents, coordinates(7), &
            value, latent_gradient, offset_gradient, status, weights)
        call check(status_ok(status), "joint value/gradient status", failures)
        oracle = oracle_value(observations, coordinates, weights)
        call check(abs(value - oracle) < 2.0e-13_dp, "weighted scalar density oracle", &
            failures)
        reference_gradient = oracle_gradient(observations, coordinates, weights, h_gradient)
        call check(maxval(abs(latent_gradient - reference_gradient(1:6))) < 3.0e-8_dp, &
            "latent gradient finite-difference oracle", failures)
        call check(abs(offset_gradient - reference_gradient(7)) < 3.0e-8_dp, &
            "offset gradient finite-difference oracle", failures)

        call poisson_log_likelihood_jvp(observations, latents, coordinates(7), &
            latent_direction, direction(7), value, tangent, status, weights)
        call check(status_ok(status), "joint JVP status", failures)
        call check(abs(tangent - dot_product(reference_gradient, direction)) < 8.0e-8_dp, &
            "joint JVP oracle", failures)

        value_bar = -1.4_dp
        call poisson_log_likelihood_vjp(observations, latents, coordinates(7), value_bar, &
            latent_bar, offset_bar, status, weights)
        call check(status_ok(status), "joint VJP status", failures)
        call check(abs(dot_product(latent_bar, latent_direction) + offset_bar*direction(7) &
            - value_bar*tangent) < 4.0e-13_dp, "joint JVP/VJP adjoint", failures)

        call poisson_log_likelihood_hvp(observations, latents, coordinates(7), &
            latent_direction, direction(7), latent_product, offset_product, status, weights)
        call check(status_ok(status), "joint HVP status", failures)
        plus = coordinates + h_direction*direction
        minus = coordinates - h_direction*direction
        reference_hvp = (oracle_gradient(observations, plus, weights, h_gradient) &
            - oracle_gradient(observations, minus, weights, h_gradient)) / (2.0_dp*h_direction)
        call check(maxval(abs(latent_product - reference_hvp(1:6))) < 4.0e-5_dp, &
            "latent HVP finite-difference oracle", failures)
        call check(abs(offset_product - reference_hvp(7)) < 7.0e-5_dp, &
            "offset HVP finite-difference oracle", failures)
    end subroutine test_joint_products

    subroutine test_objective_and_optimizer(failures)
        integer, intent(inout) :: failures
        real(dp) :: observations(6), latents(6), weights(6)
        real(dp) :: parameters(1), direction(1), gradient(1), product(1)
        real(dp) :: value, tangent, value_bar, parameter_bar(1)
        real(dp) :: expected_offset
        type(poisson_likelihood_t) :: model
        type(poisson_likelihood_lbfgsb_options_t) :: options
        type(poisson_likelihood_lbfgsb_result_t) :: result
        type(fortnum_status_t) :: status

        call fixture(observations, latents, weights)
        call model%initialize(observations, latents, status, log_rate_offset=-1.5_dp, &
            sample_weight=weights)
        call check(status_ok(status), "object initialization", failures)
        parameters = [-0.4_dp]
        direction = [0.27_dp]
        call model%value_gradient(parameters, value, gradient, status)
        call check(status_ok(status), "object value/gradient", failures)
        call model%jvp(parameters, direction, value, tangent, status)
        call check(status_ok(status), "object JVP", failures)
        call check(abs(tangent - gradient(1)*direction(1)) < 2.0e-12_dp, &
            "object JVP contraction", failures)
        value_bar = -0.8_dp
        call model%vjp(parameters, value_bar, parameter_bar, status)
        call check(status_ok(status), "object VJP", failures)
        call check(abs(parameter_bar(1) - value_bar*gradient(1)) < 2.0e-12_dp, &
            "object VJP contraction", failures)
        call model%hvp(parameters, direction, product, status)
        call check(status_ok(status), "object HVP", failures)
        call check(ieee_is_finite(product(1)), "object HVP finite", failures)

        expected_offset = log(sum(weights*observations) / sum(weights*exp(latents)))
        options%max_iterations = 100
        options%gradient_tolerance = 1.0e-8_dp
        options%lower_log_rate_offset = -5.0_dp
        options%upper_log_rate_offset = 5.0_dp
        call model%optimize_lbfgsb(options, result, status)
        call check(status_ok(status), "Poisson L-BFGS-B status", failures)
        call check(result%converged, "Poisson L-BFGS-B converged", failures)
        parameters = model%parameters()
        call check(abs(parameters(1) - expected_offset) < 2.0e-5_dp, &
            "Poisson L-BFGS-B optimum", failures)
    end subroutine test_objective_and_optimizer

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        real(dp) :: observations(2), latents(2), weights(2), value
        type(poisson_likelihood_t) :: model
        type(fortnum_status_t) :: status

        observations = [1.0_dp, 2.0_dp]
        latents = [0.0_dp, 0.1_dp]
        weights = 1.0_dp
        call model%initialize(observations, latents, status, device_kind=FORTML_DEVICE_CUDA)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA initialization refusal", failures)
        call poisson_log_likelihood_value([1.0_dp, 1.5_dp], latents, 0.0_dp, value, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "fractional-count refusal", failures)
        call model%initialize(observations, latents, status, sample_weight=weights)
        call check(status_ok(status), "CPU initialization after refusal", failures)
        call model%set_parameters([ieee_quiet_nan(0.0_dp)], status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "nonfinite-parameter refusal", failures)
    end subroutine test_refusals

    real(dp) function oracle_value(observations, coordinates, weights) result(value)
        real(dp), intent(in) :: observations(:), coordinates(:), weights(:)
        integer :: i

        value = 0.0_dp
        do i = 1, size(observations)
            value = value + weights(i)*(observations(i)*(coordinates(i) + coordinates(7)) &
                - exp(coordinates(i) + coordinates(7)) &
                - log_gamma(observations(i) + 1.0_dp))
        end do
    end function oracle_value

    function oracle_gradient(observations, coordinates, weights, step) result(gradient)
        real(dp), intent(in) :: observations(:), coordinates(:), weights(:), step
        real(dp) :: gradient(size(coordinates)), plus(size(coordinates)), minus(size(coordinates))
        integer :: j

        do j = 1, size(coordinates)
            plus = coordinates
            minus = coordinates
            plus(j) = plus(j) + step
            minus(j) = minus(j) - step
            gradient(j) = (oracle_value(observations, plus, weights) &
                - oracle_value(observations, minus, weights))/(2.0_dp*step)
        end do
    end function oracle_gradient

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: " // trim(description)
        end if
    end subroutine check

end program test_poisson_likelihood
