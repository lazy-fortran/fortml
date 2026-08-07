program test_softmax_log_hyperparameter
    !! Independent directional and adjoint oracles for transformed L2 HPO.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_softmax_regression, only: softmax_regression_t
    use fortml_softmax_training, only: softmax_training_objective_t, &
        softmax_lbfgsb_options_t, softmax_lbfgsb_result_t, &
        softmax_optimize_lbfgsb
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(softmax_regression_t), target :: model
    type(softmax_training_objective_t) :: objective
    type(softmax_lbfgsb_options_t) :: options
    type(softmax_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 2), parameters(10), direction(10)
    real(dp) :: gradient(10), plus_gradient(10), minus_gradient(10)
    real(dp) :: value, value_plus, value_minus, jvp_value, tangent
    real(dp) :: vjp_gradient(10), h, cotangent, expected_log_gradient
    integer :: labels(6), failures, n_model

    failures = 0
    x = reshape([ &
        -1.2_dp, 0.1_dp, 0.8_dp, 1.4_dp, -0.4_dp, -0.9_dp, &
        1.1_dp, -0.6_dp, 0.3_dp, 0.7_dp, -1.0_dp, 0.5_dp], shape(x))
    labels = [-3, 4, 9, 4, -3, 9]

    call model%fit(x, labels, status, l2=0.15_dp, max_iterations=500, &
        tolerance=1.0e-8_dp)
    call check(status_ok(status), "softmax setup fit", failures)
    call objective%initialize(model, x, labels, 0.4_dp, status, &
        optimize_log_l2=.true.)
    call check(status_ok(status), "log-L2 objective initialization", failures)
    n_model = model%parameter_count()
    call check(objective%parameter_count() == n_model + 1, &
        "transformed objective parameter count", failures)

    parameters = objective%parameters()
    direction = [ &
        0.07_dp, -0.08_dp, 0.03_dp, 0.05_dp, -0.06_dp, 0.04_dp, &
        -0.02_dp, 0.09_dp, -0.05_dp, 0.13_dp]
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "transformed value/gradient", failures)
    h = 2.0e-6_dp
    call objective%value_gradient(parameters + h*direction, value_plus, &
        plus_gradient, status)
    call check(status_ok(status), "transformed plus gradient", failures)
    call objective%value_gradient(parameters - h*direction, value_minus, &
        minus_gradient, status)
    call check(status_ok(status), "transformed minus gradient", failures)

    call objective%jvp(parameters, direction, jvp_value, tangent, status)
    call check(status_ok(status), "transformed JVP", failures)
    call check(abs(jvp_value - value) < 2.0e-13_dp, &
        "JVP primal value", failures)
    call check(abs(tangent - (value_plus - value_minus)/(2.0_dp*h)) < 3.0e-8_dp, &
        "transformed JVP finite difference", failures)

    cotangent = -1.7_dp
    call objective%vjp(parameters, cotangent, vjp_gradient, status)
    call check(status_ok(status), "transformed VJP", failures)
    call check(maxval(abs(vjp_gradient - cotangent*gradient)) < 2.0e-13_dp, &
        "transformed VJP adjoint oracle", failures)
    call check(abs(dot_product(gradient, direction) - tangent) < 2.0e-13_dp, &
        "JVP/VJP scalar adjoint oracle", failures)

    call objective%hvp(parameters, direction, gradient, status)
    call check(status_ok(status), "transformed HVP", failures)
    call check(maxval(abs(gradient - (plus_gradient - minus_gradient)/(2.0_dp*h))) &
        < 3.0e-7_dp, "transformed HVP finite difference", failures)
    expected_log_gradient = 0.5_dp*exp(parameters(n_model + 1))* &
        sum(parameters(:6)**2)
    call objective%value_gradient(parameters, value, plus_gradient, status)
    call check(abs(plus_gradient(n_model + 1) - expected_log_gradient) < 2.0e-13_dp, &
        "exact transformed hyperparameter gradient", failures)

    options%l2 = 0.35_dp
    options%optimize_log_l2 = .true.
    options%log_l2_lower_bound = -2.0_dp
    options%log_l2_upper_bound = 0.5_dp
    options%lower_bound = -8.0_dp
    options%upper_bound = 8.0_dp
    options%max_iterations = 300
    options%gradient_tolerance = 1.0e-6_dp
    call softmax_optimize_lbfgsb(model, x, labels, options, result, status)
    call check(status_ok(status), "bounded transformed FortOpt optimization", failures)
    call check(result%converged, "bounded transformed convergence", failures)
    call check(result%l2 >= exp(options%log_l2_lower_bound) - 1.0e-12_dp .and. &
        result%l2 <= exp(options%log_l2_upper_bound) + 1.0e-12_dp, &
        "transformed L2 bounds", failures)
    call check(abs(model%regularization() - result%l2) < 2.0e-12_dp, &
        "transformed model regularization", failures)

    options%optimize_l2 = .true.
    call softmax_optimize_lbfgsb(model, x, labels, options, result, status)
    call check(.not. status_ok(status), "direct/log L2 mode refusal", failures)

    if (failures /= 0) error stop 1
    write (*, '(a)') "PASS softmax transformed hyperparameter products"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

end program test_softmax_log_hyperparameter
