program test_glm_regression
    !! Independent behavioral checks for weighted Poisson/Gamma GLMs.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_glm_regression, only: glm_regression_t, GLM_FAMILY_POISSON, &
        GLM_FAMILY_GAMMA
    implicit none

    integer :: failures

    failures = 0
    call test_poisson_fit_and_products(failures)
    call test_gamma_fit_and_objective(failures)
    call test_domain_and_device_contract(failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GLM test(s): ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GLM regression independent behavioral oracles"

contains

    subroutine test_poisson_fit_and_products(failures)
        integer, intent(inout) :: failures
        real(real64) :: x(7, 1), y(7), weights(7), prediction(7)
        real(real64) :: design(7, 2), expected(7), prediction_matrix(7, 1)
        real(real64) :: theta_dot(2), x_dot(7, 1), y_dot(7), y_plus(7), y_minus(7)
        real(real64) :: y_dot_matrix(7, 1)
        real(real64) :: theta_bar(2), x_bar(7, 1), cotangent(7, 1)
        real(real64), allocatable :: theta(:)
        real(real64) :: h, lhs, rhs, fd_error
        type(glm_regression_t) :: model
        type(fortnum_status_t) :: status
        integer :: i

        x(:, 1) = [-2.0_real64, -1.0_real64, -0.5_real64, 0.0_real64, &
            0.5_real64, 1.0_real64, 2.0_real64]
        y = [0.4_real64, 0.7_real64, 1.0_real64, 1.4_real64, 1.8_real64, &
            2.4_real64, 3.8_real64]
        weights = [1.0_real64, 2.0_real64, 1.0_real64, 3.0_real64, 2.0_real64, &
            1.0_real64, 2.0_real64]
        call model%fit(x, y, status, family=GLM_FAMILY_POISSON, alpha=0.05_real64, &
            sample_weight=weights, max_iterations=1000, tolerance=1.0e-10_real64)
        call check(status_ok(status), "Poisson weighted L-BFGS-B fit", failures)
        call check(model%fitted() .and. model%family() == GLM_FAMILY_POISSON .and. &
            model%link() == 1 .and. model%parameter_count() == 2, &
            "Poisson metadata", failures)
        call model%predict(x, prediction, status)
        call check(status_ok(status) .and. all(prediction > 0.0_real64), &
            "Poisson positive prediction", failures)
        theta = model%parameters()
        design(:, 1) = 1.0_real64
        design(:, 2) = x(:, 1)
        do i = 1, size(x, 1)
            expected(i) = exp(design(i, 1)*theta(1) + design(i, 2)*theta(2))
        end do
        call check(maxval(abs(prediction-expected)) < 2.0e-12_real64, &
            "Poisson prediction oracle", failures)

        theta_dot = [0.13_real64, -0.21_real64]
        x_dot(:, 1) = [0.2_real64, -0.1_real64, 0.3_real64, -0.2_real64, &
            0.1_real64, -0.4_real64, 0.05_real64]
        call model%jvp(x, theta_dot, x_dot, prediction_matrix, y_dot_matrix, status)
        y_dot = y_dot_matrix(:, 1)
        call check(status_ok(status), "Poisson prediction JVP", failures)
        h = 1.0e-6_real64
        call model%set_parameters(theta+h*theta_dot, status)
        call model%predict(x+h*x_dot, y_plus, status)
        call model%set_parameters(theta-h*theta_dot, status)
        call model%predict(x-h*x_dot, y_minus, status)
        call model%set_parameters(theta, status)
        fd_error = maxval(abs(y_dot-(y_plus-y_minus)/(2.0_real64*h)))
        call check(fd_error < 2.0e-8_real64, "Poisson JVP finite-difference oracle", failures)

        cotangent(:, 1) = [0.4_real64, -0.2_real64, 0.3_real64, 0.5_real64, &
            -0.7_real64, 0.1_real64, 0.2_real64]
        call model%vjp(x, cotangent, theta_bar, x_bar, status)
        lhs = sum(cotangent(:, 1)*y_dot)
        rhs = sum(theta_bar*theta_dot)+sum(x_bar*x_dot)
        call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-11_real64, &
            "Poisson VJP adjoint oracle", failures)
    end subroutine test_poisson_fit_and_products

    subroutine test_gamma_fit_and_objective(failures)
        integer, intent(inout) :: failures
        real(real64) :: x(5, 1), y(5), weights(5), prediction(5), theta(2)
        real(real64) :: value, value_plus, gradient(2), theta_plus(2), h
        real(real64) :: alpha_gradient, dispersion_gradient, value_disp_plus
        real(real64) :: expected, mass, eta, mu
        type(glm_regression_t) :: model
        type(fortnum_status_t) :: status
        integer :: i

        x(:, 1) = [-1.0_real64, -0.25_real64, 0.0_real64, 0.75_real64, 1.5_real64]
        y = [0.75_real64, 1.2_real64, 1.7_real64, 2.1_real64, 3.2_real64]
        weights = [1.0_real64, 2.0_real64, 1.0_real64, 3.0_real64, 2.0_real64]
        call model%fit(x, y, status, family=GLM_FAMILY_GAMMA, alpha=0.1_real64, &
            sample_weight=weights, dispersion=1.7_real64, max_iterations=1000, &
            tolerance=1.0e-10_real64)
        call check(status_ok(status), "Gamma weighted L-BFGS-B fit", failures)
        call model%predict(x, prediction, status)
        call check(status_ok(status) .and. all(prediction > 0.0_real64), &
            "Gamma positive prediction", failures)
        theta = model%parameters()
        call model%objective_value_gradient(x, y, theta, value, gradient, status, &
            family=GLM_FAMILY_GAMMA, alpha=0.1_real64, sample_weight=weights, &
            dispersion=1.7_real64)
        call check(status_ok(status) .and. ieee_finite(gradient), &
            "Gamma objective value/gradient", failures)
        mass = sum(weights)
        expected = 0.0_real64
        do i = 1, size(x, 1)
            eta = theta(1)+x(i, 1)*theta(2)
            mu = exp(eta)
            expected = expected + weights(i)*(y(i)/mu+eta)/1.7_real64
        end do
        expected = expected/mass + 0.5_real64*0.1_real64*sum(theta(2:)**2)
        call check(abs(value-expected) < 2.0e-11_real64, "Gamma objective oracle", failures)
        h = 1.0e-6_real64
        theta_plus = theta
        theta_plus(2) = theta_plus(2)+h
        call model%objective_value_gradient(x, y, theta_plus, value_plus, gradient, status, &
            family=GLM_FAMILY_GAMMA, alpha=0.1_real64, sample_weight=weights, &
            dispersion=1.7_real64)
        call model%objective_value_gradient(x, y, theta, value, gradient, status, &
            family=GLM_FAMILY_GAMMA, alpha=0.1_real64, sample_weight=weights, &
            dispersion=1.7_real64, alpha_gradient=alpha_gradient, &
            dispersion_gradient=dispersion_gradient)
        call check(abs(alpha_gradient-0.5_real64*sum(theta(2:)**2)) < 2.0e-12_real64, &
            "Gamma alpha hypergradient oracle", failures)
        call model%objective_value_gradient(x, y, theta, value_disp_plus, gradient, status, &
            family=GLM_FAMILY_GAMMA, alpha=0.1_real64, sample_weight=weights, &
            dispersion=1.7_real64+h)
        call check(abs(dispersion_gradient-(value_disp_plus-value)/h) < 2.0e-5_real64, &
            "Gamma dispersion hypergradient oracle", failures)
        call check(abs(gradient(2)-(value_plus-value)/h) < 2.0e-5_real64, &
            "Gamma objective gradient finite-difference oracle", failures)
    end subroutine test_gamma_fit_and_objective

    subroutine test_domain_and_device_contract(failures)
        integer, intent(inout) :: failures
        real(real64) :: x(3, 1), y(3), prediction(3)
        type(glm_regression_t) :: model
        type(fortnum_status_t) :: status
        type(fortml_device_t) :: cuda

        x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64]
        y = [1.0_real64, 0.0_real64, 2.0_real64]
        call model%fit(x, y, status, family=GLM_FAMILY_GAMMA)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "Gamma zero-response refusal", failures)
        y = [1.0_real64, 2.0_real64, 3.0_real64]
        call model%fit(x, y, status, sample_weight=[1.0_real64, -1.0_real64, 1.0_real64])
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "negative sample-weight refusal", failures)
        cuda%kind = FORTML_DEVICE_CUDA
        cuda%selected = .true.
        cuda%available = .true.
        block
            real(real64) :: prediction_matrix(3, 1)
            call model%predict_device(cuda, x, prediction_matrix, status)
        end block
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            .not. model%device_supported(FORTML_DEVICE_CUDA), &
            "CUDA resident-kernel refusal", failures)
    end subroutine test_domain_and_device_contract

    logical function ieee_finite(values) result(ok)
        real(real64), intent(in) :: values(:)
        ok = all(values == values) .and. all(abs(values) < huge(1.0_real64))
    end function ieee_finite

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

end program test_glm_regression
