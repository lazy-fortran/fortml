program test_logistic_hyperparameter_training
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_logistic_regression, only: logistic_regression_t
    use fortml_logistic_training, only: logistic_training_objective_t, &
        logistic_lbfgsb_options_t, logistic_lbfgsb_result_t, &
        logistic_optimize_lbfgsb
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(logistic_regression_t), target :: model
    type(logistic_training_objective_t) :: objective
    type(logistic_lbfgsb_options_t) :: options
    type(logistic_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 1), parameters(3), direction(3), product(3)
    real(dp) :: gradient(3), reference_gradient(3), plus_gradient(3)
    real(dp) :: minus_gradient(3), value, reference_value, h
    integer :: labels(4), failures

    failures = 0

    x(:, 1) = [-2.0_dp, -1.0_dp, 1.0_dp, 2.0_dp]
    labels = [3, 3, 11, 11]
    call model%fit(x, labels, status, l2=0.3_dp, max_iterations=500, &
        tolerance=1.0e-8_dp)
    call check(status_ok(status), "logistic setup fit", failures)
    call objective%initialize(model, x, labels, 0.4_dp, status, optimize_l2=.true.)
    call check(status_ok(status), "objective initialization", failures)
    call check(objective%parameter_count() == 3, &
        "objective packed parameter count", failures)

    parameters = [0.25_dp, -0.15_dp, 0.7_dp]
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "objective value/gradient", failures)
    call reference_value_gradient(x, labels, parameters, reference_value, &
        reference_gradient)
    call check(abs(value - reference_value) < 2.0e-13_dp, &
        "independent logistic objective value", failures)
    call check(maxval(abs(gradient - reference_gradient)) < 2.0e-13_dp, &
        "independent logistic objective gradient", failures)

    direction = [-0.4_dp, 0.3_dp, 0.2_dp]
    call objective%hvp(parameters, direction, product, status)
    call check(status_ok(status), "objective HVP", failures)
    h = 1.0e-6_dp
    call objective%value_gradient(parameters + h*direction, value, plus_gradient, status)
    call check(status_ok(status), "HVP plus gradient", failures)
    call objective%value_gradient(parameters - h*direction, value, minus_gradient, status)
    call check(status_ok(status), "HVP minus gradient", failures)
    call check(maxval(abs(product - (plus_gradient - minus_gradient)/(2.0_dp*h))) &
        < 2.0e-7_dp, "independent logistic HVP finite difference", failures)
    call check(abs(product(3) - dot_product(parameters(:1), direction(:1))) &
        < 2.0e-13_dp, "exact L2 hyperparameter HVP block", failures)

    options%l2 = 0.4_dp
    options%max_iterations = 200
    options%gradient_tolerance = 1.0e-6_dp
    options%lower_bound = -8.0_dp
    options%upper_bound = 8.0_dp
    call logistic_optimize_lbfgsb(model, x, labels, options, result, status)
    call check(status_ok(status), "FortOpt logistic optimization", failures)
    call check(result%converged, "FortOpt logistic convergence", failures)
    call check(result%l2 == options%l2, "fixed L2 result", failures)
    call check(result%objective < huge(1.0_dp), "finite optimization objective", failures)

    options%optimize_l2 = .true.
    options%l2 = 0.5_dp
    options%l2_lower_bound = 1.0_dp
    options%l2_upper_bound = 0.1_dp
    call logistic_optimize_lbfgsb(model, x, labels, options, result, status)
    call check(.not. status_ok(status), "invalid L2 bounds refusal", failures)
    write (*, '(a)') "PASS logistic hyperparameter objective independent oracles"

contains

    subroutine reference_value_gradient(x, labels, parameters, value, gradient)
        real(dp), intent(in) :: x(:, :), parameters(:)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, gradient(:)
        real(dp) :: score, probability, residual
        integer :: i

        value = 0.0_dp
        gradient = 0.0_dp
        do i = 1, size(labels)
            score = x(i, 1)*parameters(1) + parameters(2)
            probability = 1.0_dp/(1.0_dp + exp(-score))
            if (labels(i) == 11) then
                residual = probability - 1.0_dp
                value = value + log(1.0_dp + exp(-score))
            else
                residual = probability
                value = value + log(1.0_dp + exp(score))
            end if
            gradient(1) = gradient(1) + residual*x(i, 1)
            gradient(2) = gradient(2) + residual
        end do
        value = value/real(size(labels), dp) + 0.5_dp*parameters(3)*parameters(1)**2
        gradient(:2) = gradient(:2)/real(size(labels), dp)
        gradient(1) = gradient(1) + parameters(3)*parameters(1)
        gradient(3) = 0.5_dp*parameters(1)**2
    end subroutine reference_value_gradient

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL "//trim(description)
            error stop 1
        end if
    end subroutine check

end program test_logistic_hyperparameter_training
