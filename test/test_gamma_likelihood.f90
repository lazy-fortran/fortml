program test_gamma_likelihood
    !! Independent scalar-density and finite-difference oracles for Gamma GP
    !! likelihood products, shape fitting, rollback, and device dispatch.
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    use fortml_gamma_likelihood, only: gamma_likelihood_t, &
        gamma_likelihood_lbfgsb_options_t, gamma_likelihood_lbfgsb_result_t, &
        gamma_log_likelihood_value, gamma_log_likelihood_value_gradient, &
        gamma_log_likelihood_jvp, gamma_log_likelihood_vjp, &
        gamma_log_likelihood_hvp
    implicit none

    integer :: failures

    failures = 0
    call test_joint_products(failures)
    call test_objective_and_optimizer(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " Gamma likelihood test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: Gamma likelihood"

contains

    subroutine fixture(observations, latents, weights)
        real(dp), intent(out) :: observations(6), latents(6), weights(6)

        observations = [0.35_dp, 0.8_dp, 1.7_dp, 3.2_dp, 5.5_dp, 2.4_dp]
        latents = log([0.5_dp, 1.0_dp, 1.4_dp, 2.8_dp, 4.1_dp, 2.0_dp])
        weights = [1.0_dp, 0.25_dp, 2.0_dp, 1.5_dp, 0.0_dp, 0.75_dp]
    end subroutine fixture

    subroutine test_joint_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: observations(6), latents(6), weights(6)
        real(dp) :: latent_gradient(6), latent_direction(6), latent_bar(6)
        real(dp) :: latent_product(6), reference_gradient(7), reference_hvp(7)
        real(dp) :: coordinates(7), direction(7), plus(7), minus(7)
        real(dp) :: value, oracle, shape_gradient, tangent, shape_bar, shape_product
        real(dp) :: value_bar, h_gradient, h_direction
        type(fortnum_status_t) :: status

        call fixture(observations, latents, weights)
        coordinates(1:6) = latents
        coordinates(7) = log(1.7_dp)
        direction = [0.3_dp, -0.2_dp, 0.1_dp, 0.45_dp, -0.4_dp, 0.25_dp, -0.35_dp]
        latent_direction = direction(1:6)
        h_gradient = 2.0e-6_dp
        h_direction = 2.0e-4_dp

        call gamma_log_likelihood_value_gradient(observations, latents, &
            coordinates(7), value, latent_gradient, shape_gradient, status, weights)
        call check(status_ok(status), "joint value/gradient status", failures)
        oracle = oracle_value(observations, latents, coordinates(7), weights)
        call check(abs(value - oracle) < 2.0e-13_dp, "weighted scalar density oracle", &
            failures)
        reference_gradient = oracle_gradient(observations, coordinates, weights, &
            h_gradient)
        call check(maxval(abs(latent_gradient - reference_gradient(1:6))) < 3.0e-8_dp, &
            "latent gradient finite-difference oracle", failures)
        call check(abs(shape_gradient - reference_gradient(7)) < 8.0e-8_dp, &
            "shape gradient finite-difference oracle", failures)

        call gamma_log_likelihood_jvp(observations, latents, coordinates(7), &
            latent_direction, direction(7), value, tangent, status, weights)
        call check(status_ok(status), "joint JVP status", failures)
        call check(abs(tangent - dot_product(reference_gradient, direction)) &
            < 8.0e-8_dp, "joint JVP oracle", failures)

        value_bar = -1.4_dp
        call gamma_log_likelihood_vjp(observations, latents, coordinates(7), &
            value_bar, latent_bar, shape_bar, status, weights)
        call check(status_ok(status), "joint VJP status", failures)
        call check(abs(dot_product(latent_bar, latent_direction) + &
            shape_bar*direction(7) - value_bar*tangent) < 4.0e-13_dp, &
            "joint JVP/VJP adjoint", failures)

        call gamma_log_likelihood_hvp(observations, latents, coordinates(7), &
            latent_direction, direction(7), latent_product, shape_product, status, &
            weights)
        call check(status_ok(status), "joint HVP status", failures)
        plus = coordinates + h_direction*direction
        minus = coordinates - h_direction*direction
        reference_hvp = (oracle_gradient(observations, plus, weights, h_gradient) &
            - oracle_gradient(observations, minus, weights, h_gradient)) &
            /(2.0_dp*h_direction)
        call check(maxval(abs(latent_product - reference_hvp(1:6))) < 4.0e-5_dp, &
            "latent HVP finite-difference oracle", failures)
        call check(abs(shape_product - reference_hvp(7)) < 7.0e-5_dp, &
            "shape HVP finite-difference oracle", failures)

        call gamma_log_likelihood_value(observations(1:4), latents(1:4), &
            coordinates(7), oracle, status, weights(1:4))
        call check(status_ok(status), "truncated zero-weight oracle status", failures)
        call check(abs(value - oracle - weights(6)*oracle_term(observations(6), &
            latents(6), coordinates(7))) < 2.0e-13_dp, &
            "zero sample weight removes an observation", failures)
    end subroutine test_joint_products

    subroutine test_objective_and_optimizer(failures)
        integer, intent(inout) :: failures
        real(dp) :: observations(6), latents(6), weights(6)
        real(dp) :: parameters(1), initial(1), current(1), gradient(1), product(1)
        real(dp) :: value, tangent, objective_value, objective_gradient(1)
        type(gamma_likelihood_t), target :: model
        type(gamma_likelihood_lbfgsb_options_t) :: options
        type(gamma_likelihood_lbfgsb_result_t) :: result
        type(objective_t) :: objective
        type(fortnum_status_t) :: status

        call fixture(observations, latents, weights)
        call model%initialize(observations, latents, status, log(0.2_dp), weights)
        call check(status_ok(status), "objective initializes", failures)
        call check(model%initialized(), "objective reports initialized", failures)
        call check(model%parameter_count() == 1, "objective parameter count", failures)
        parameters = [log(1.7_dp)]
        call model%value_gradient(parameters, value, gradient, status)
        call check(status_ok(status), "objective value/gradient status", failures)
        call check(abs(value + oracle_value(observations, latents, parameters(1), &
            weights)) < 2.0e-13_dp, "objective is negative log likelihood", failures)
        call model%jvp(parameters, [0.3_dp], value, tangent, status)
        call check(status_ok(status), "objective JVP status", failures)
        call model%vjp(parameters, -0.6_dp, product, status)
        call check(status_ok(status), "objective VJP status", failures)
        call check(abs(product(1)*0.3_dp - (-0.6_dp)*tangent) < 2.0e-13_dp, &
            "objective adjoint identity", failures)
        call model%hvp(parameters, [-0.25_dp], product, status)
        call check(status_ok(status), "objective HVP status", failures)
        call model%fortopt(objective, status)
        call check(status_ok(status), "FortOpt context initializes", failures)
        call objective%value_gradient(parameters, objective_value, &
            objective_gradient, status)
        call check(status_ok(status), "FortOpt context evaluates", failures)
        call check(abs(objective_value - value) < 2.0e-13_dp .and. &
            abs(objective_gradient(1) - gradient(1)) < 2.0e-13_dp, &
            "FortOpt context matches object", failures)

        call model%set_parameters([log(0.2_dp)], status)
        initial = model%parameters()
        options%max_iterations = 1
        call model%optimize_lbfgsb(options, result, status)
        call check(status%code == FORTNUM_CONVERGENCE_ERROR, &
            "iteration-limit failure is typed", failures)
        current = model%parameters()
        call check(abs(current(1) - initial(1)) < 2.0e-14_dp, &
            "failed optimizer rolls back", failures)

        options%max_iterations = 100
        call model%optimize_lbfgsb(options, result, status)
        if (.not. status_ok(status)) &
            write (error_unit, '(a)') "Gamma optimizer status: "//trim(status%msg)
        call check(status_ok(status), "L-BFGS-B shape fit status", failures)
        call check(result%converged, "L-BFGS-B reports convergence", failures)
        current = model%parameters()
        call check(abs(current(1) - 3.059853981971437_dp) < 2.0e-6_dp, &
            "L-BFGS-B matches independent SciPy optimum", failures)
        call check(result%gradient_norm < 2.0e-7_dp, &
            "L-BFGS-B final gradient is small", failures)
    end subroutine test_objective_and_optimizer

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        real(dp) :: observations(6), latents(6), weights(6), initial(1), current(1)
        type(gamma_likelihood_t) :: model
        type(fortnum_status_t) :: status

        call fixture(observations, latents, weights)
        call model%initialize(observations, latents, status, log(1.5_dp), weights, &
            FORTML_DEVICE_CUDA)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "CUDA initialization is a typed refusal", failures)
        call model%initialize(observations, latents, status, log(1.5_dp), weights)
        call check(status_ok(status), "CPU fixture reinitializes", failures)
        initial = model%parameters()
        call model%set_parameters([ieee_value(0.0_dp, ieee_quiet_nan)], status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "invalid transformed shape is refused", failures)
        current = model%parameters()
        call check(abs(current(1) - initial(1)) < 2.0e-14_dp, &
            "invalid setter is transactional", failures)
        call gamma_log_likelihood_value(observations, latents, 1000.0_dp, &
            initial(1), status, weights)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "overflowing transformed shape is refused", failures)
    end subroutine test_refusals

    real(dp) function oracle_value(observations, latents, log_shape, &
            weights) result(value)
        real(dp), intent(in) :: observations(:), latents(:), log_shape, weights(:)
        integer :: i

        value = 0.0_dp
        do i = 1, size(observations)
            value = value + weights(i)*oracle_term(observations(i), latents(i), &
                log_shape)
        end do
    end function oracle_value

    real(dp) function oracle_term(observation, latent, log_shape) result(value)
        real(dp), intent(in) :: observation, latent, log_shape
        real(dp) :: shape

        shape = exp(log_shape)
        value = shape*log_shape - log_gamma(shape) + (shape - 1.0_dp)*log(observation) &
            - shape*latent - shape*observation/exp(latent)
    end function oracle_term

    function oracle_gradient(observations, coordinates, weights, h) result(gradient)
        real(dp), intent(in) :: observations(:), coordinates(:), weights(:), h
        real(dp) :: gradient(size(coordinates)), plus(size(coordinates))
        real(dp) :: minus(size(coordinates))
        integer :: i, n

        n = size(observations)
        do i = 1, size(coordinates)
            plus = coordinates
            minus = coordinates
            plus(i) = plus(i) + h
            minus(i) = minus(i) - h
            gradient(i) = (oracle_value(observations, plus(1:n), plus(n + 1), &
                weights) - oracle_value(observations, minus(1:n), minus(n + 1), &
                weights))/(2.0_dp*h)
        end do
    end function oracle_gradient

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL: "//description
        end if
    end subroutine check

end program test_gamma_likelihood
