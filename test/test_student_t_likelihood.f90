program test_student_t_likelihood
    !! Independent products for the fixed-latent Student-t likelihood.
    !!
    !! The oracle is a direct scalar density implementation in this test. It
    !! does not call FortML's product helpers: central differences of that
    !! independent density check value, gradient, and the directional HVP;
    !! the VJP is checked through the adjoint identity.
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    use fortml_student_t_likelihood, only: student_t_likelihood_t, &
        student_t_log_likelihood_value, student_t_log_likelihood_gradient, &
        student_t_log_likelihood_jvp, student_t_log_likelihood_vjp, &
        student_t_log_likelihood_hvp
    implicit none

    integer :: failures

    failures = 0
    call test_products(failures)
    call test_objective_and_transaction(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " Student-t likelihood test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: Student-t likelihood"

contains

    subroutine fixture(observations, locations, parameters)
        real(dp), intent(out) :: observations(7), locations(7), parameters(2)

        observations = [-1.2_dp, -0.25_dp, 0.15_dp, 0.9_dp, 1.7_dp, -0.8_dp, 0.45_dp]
        locations = [-0.9_dp, -0.1_dp, 0.25_dp, 0.55_dp, 1.2_dp, -0.4_dp, 0.1_dp]
        parameters = [log(0.75_dp), log(4.3_dp)]
    end subroutine fixture

    subroutine test_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: observations(7), locations(7), parameters(2), direction(2)
        real(dp) :: gradient(2), product(2), parameter_bar(2)
        real(dp) :: value, tangent, value_bar, oracle, oracle_dot
        real(dp) :: reference_gradient(2), reference_hvp(2), plus(2), minus(2)
        real(dp) :: gradient_plus(2), gradient_minus(2)
        real(dp) :: h
        integer :: i
        type(fortnum_status_t) :: status

        call fixture(observations, locations, parameters)
        direction = [0.35_dp, -0.6_dp]
        h = 2.0e-4_dp
        call student_t_log_likelihood_value(observations, locations, parameters, value, status)
        call check(status_ok(status), "likelihood value status", failures)
        oracle = oracle_value(observations, locations, parameters)
        call check(abs(value - oracle) < 3.0e-13_dp, "likelihood value oracle", failures)

        call student_t_log_likelihood_gradient(observations, locations, parameters, gradient, &
            status)
        call check(status_ok(status), "likelihood gradient status", failures)
        do i = 1, 2
            plus = parameters
            minus = parameters
            plus(i) = plus(i) + h
            minus(i) = minus(i) - h
            reference_gradient(i) = (oracle_value(observations, locations, plus) - &
                oracle_value(observations, locations, minus))/(2.0_dp*h)
        end do
        call check(maxval(abs(gradient - reference_gradient)) < 3.0e-7_dp, &
            "likelihood gradient finite-difference oracle", failures)

        call student_t_log_likelihood_jvp(observations, locations, parameters, direction, &
            value, tangent, status)
        call check(status_ok(status), "likelihood JVP status", failures)
        oracle_dot = dot_product(reference_gradient, direction)
        call check(abs(tangent - oracle_dot) < 4.0e-7_dp, "likelihood JVP oracle", failures)

        value_bar = -1.7_dp
        call student_t_log_likelihood_vjp(observations, locations, parameters, value_bar, &
            parameter_bar, status)
        call check(status_ok(status), "likelihood VJP status", failures)
        call check(abs(dot_product(parameter_bar, direction) - value_bar*tangent) < 3.0e-13_dp, &
            "likelihood VJP/JVP adjoint", failures)

        call student_t_log_likelihood_hvp(observations, locations, parameters, direction, &
            product, status)
        call check(status_ok(status), "likelihood HVP status", failures)
        plus = parameters + h*direction
        minus = parameters - h*direction
        gradient_plus = oracle_gradient(observations, locations, plus, h)
        gradient_minus = oracle_gradient(observations, locations, minus, h)
        reference_hvp = (gradient_plus - gradient_minus)/(2.0_dp*h)
        call check(maxval(abs(product - reference_hvp)) < 3.0e-5_dp, &
            "likelihood HVP finite-difference oracle", failures)
    end subroutine test_products

    subroutine test_objective_and_transaction(failures)
        integer, intent(inout) :: failures
        type(student_t_likelihood_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: observations(7), locations(7), parameters(2), initial(2), direction(2)
        real(dp) :: value, tangent, gradient(2), product(2), value_bar
        type(objective_t) :: objective
        real(dp) :: objective_value, objective_gradient(2)

        call fixture(observations, locations, parameters)
        call model%initialize(observations, locations, status, parameters)
        call check(status_ok(status), "objective fixture initializes", failures)
        call check(model%initialized(), "objective reports initialized", failures)
        call check(model%parameter_count() == 2, "objective parameter count", failures)
        initial = model%parameters()
        call model%value_gradient(parameters, value, gradient, status)
        call check(status_ok(status), "objective value/gradient status", failures)
        call student_t_log_likelihood_value(observations, locations, parameters, tangent, status)
        call check(abs(value + tangent) < 3.0e-13_dp, &
            "FortOpt objective is negative log likelihood", failures)
        call model%jvp(parameters, [0.2_dp, -0.3_dp], tangent, value_bar, status)
        call check(status_ok(status), "objective JVP status", failures)
        call model%vjp(parameters, -0.8_dp, product, status)
        call check(status_ok(status), "objective VJP status", failures)
        call check(abs(dot_product(product, [0.2_dp, -0.3_dp]) - (-0.8_dp)*value_bar) &
            < 3.0e-13_dp, "objective VJP/JVP adjoint", failures)
        direction = [0.4_dp, -0.2_dp]
        call model%hvp(parameters, direction, product, status)
        call check(status_ok(status), "objective HVP status", failures)
        call model%fortopt(objective, status)
        call check(status_ok(status), "FortOpt context initializes", failures)
        call objective%value_gradient(parameters, objective_value, objective_gradient, status)
        call check(status_ok(status), "FortOpt context evaluates", failures)
        call check(abs(objective_value - value) < 3.0e-13_dp .and. &
            maxval(abs(objective_gradient - gradient)) < 3.0e-13_dp, &
            "FortOpt context matches objective products", failures)

        call model%set_parameters([ieee_value(0.0_dp, ieee_quiet_nan), 0.0_dp], status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "invalid objective parameters are refused", failures)
        call check(maxval(abs(model%parameters() - initial)) < 1.0e-14_dp, &
            "invalid objective update is transactional", failures)
    end subroutine test_objective_and_transaction

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(student_t_likelihood_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: observations(7), locations(7), parameters(2), short_parameters(1)
        real(dp) :: overflowing_parameters(2)

        call fixture(observations, locations, parameters)
        call model%initialize(observations, locations, status, parameters, FORTML_DEVICE_CUDA)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "CUDA likelihood initialization is a typed refusal", failures)
        call model%initialize(observations, locations, status, parameters)
        call check(status_ok(status), "CPU fixture reinitializes", failures)
        short_parameters = parameters(1:1)
        call model%set_parameters(short_parameters, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "parameter count refusal", failures)
        overflowing_parameters = [parameters(1), 1000.0_dp]
        call model%set_parameters(overflowing_parameters, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "exponent overflow refusal", failures)
        call check(maxval(abs(model%parameters() - parameters)) < 1.0e-14_dp, &
            "exponent overflow update is transactional", failures)
    end subroutine test_refusals

    real(dp) function oracle_value(observations, locations, parameters) result(value)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:)
        real(dp) :: scale, nu, residual, q
        integer :: i

        scale = exp(parameters(1))
        nu = exp(parameters(2))
        value = 0.0_dp
        do i = 1, size(observations)
            residual = observations(i) - locations(i)
            q = (residual/(scale*sqrt(nu)))**2
            value = value + log_gamma(0.5_dp*(nu + 1.0_dp)) - &
                log_gamma(0.5_dp*nu) - 0.5_dp*log(nu*4.0_dp*atan(1.0_dp)) &
                - parameters(1) - 0.5_dp*(nu + 1.0_dp)*log(1.0_dp + q)
        end do
    end function oracle_value

    function oracle_gradient(observations, locations, parameters, h) result(gradient)
        real(dp), intent(in) :: observations(:), locations(:), parameters(:), h
        real(dp) :: gradient(2), plus(2), minus(2)
        integer :: i

        do i = 1, 2
            plus = parameters
            minus = parameters
            plus(i) = plus(i) + h
            minus(i) = minus(i) - h
            gradient(i) = (oracle_value(observations, locations, plus) - &
                oracle_value(observations, locations, minus))/(2.0_dp*h)
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

end program test_student_t_likelihood
