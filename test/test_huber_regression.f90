program test_huber_regression
    !! Independent finite-difference oracle for weighted Huber products.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_huber_regression, only: huber_regression_t, &
        huber_training_objective_t, huber_lbfgsb_options_t, &
        huber_lbfgsb_result_t, huber_optimize_lbfgsb
    implicit none
    integer, parameter :: dp = real64
    type(huber_regression_t), target :: model, fit_model
    type(huber_training_objective_t) :: objective
    type(huber_lbfgsb_options_t) :: options
    type(huber_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 1), targets(6, 1), weights(6)
    real(dp), allocatable :: theta(:), direction(:), gradient(:), gradient_plus(:), &
        gradient_minus(:), product(:), parameter_bar(:)
    real(dp) :: value, value_plus, value_minus, tangent, h, error
    integer :: failures, i

    failures = 0
    x(:, 1) = [-1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    targets(:, 1) = [0.1_dp, 1.6_dp, 4.4_dp, 7.0_dp, -1.5_dp, 2.4_dp]
    weights = [0.5_dp, 1.0_dp, 1.4_dp, 0.8_dp, 1.2_dp, 0.7_dp]

    call model%initialize(1, 1, status)
    call check(status_ok(status), "model initialization", failures)
    call model%set_parameters([0.5_dp, 1.1_dp], status)
    call check(status_ok(status), "model parameter state", failures)
    call objective%initialize(model, x, targets, 1.7_dp, 0.02_dp, status, &
        optimize_l2=.true., optimize_delta=.true., sample_weight=weights)
    call check(status_ok(status), "objective initialization", failures)
    theta = objective%parameters()
    allocate(direction(size(theta)), gradient(size(theta)), gradient_plus(size(theta)), &
        gradient_minus(size(theta)), product(size(theta)), parameter_bar(size(theta)))
    direction = [(0.013_dp*real(i, dp), i=1, size(theta))]
    h = 2.0e-6_dp
    call objective%value_gradient(theta, value, gradient, status)
    call check(status_ok(status), "value and gradient", failures)
    call objective%jvp(theta, direction, value_plus, tangent, status)
    call check(status_ok(status) .and. abs(tangent-dot_product(gradient, direction)) < 2.0e-11_dp, &
        "JVP contraction", failures)
    call objective%vjp(theta, 1.7_dp, parameter_bar, status)
    call check(status_ok(status) .and. maxval(abs(parameter_bar-1.7_dp*gradient)) < 2.0e-11_dp, &
        "VJP duality", failures)
    call objective%value_gradient(theta+h*direction, value_plus, gradient_plus, status)
    call objective%value_gradient(theta-h*direction, value_minus, gradient_minus, status)
    call objective%hvp(theta, direction, product, status)
    error = maxval(abs(product-(gradient_plus-gradient_minus)/(2.0_dp*h)))
    call check(status_ok(status) .and. error < 4.0e-5_dp, &
        "HVP finite-difference oracle", failures)
    call check(abs(tangent-(value_plus-value_minus)/(2.0_dp*h)) < 4.0e-5_dp, &
        "value finite-difference oracle", failures)

    options%max_iterations = 250
    options%max_line_search = 80
    options%gradient_tolerance = 1.0e-7_dp
    options%step_tolerance = 1.0e-10_dp
    options%objective_tolerance = 1.0e-10_dp
    options%lower_bound = -12.0_dp
    options%upper_bound = 12.0_dp
    options%l2 = 1.0e-3_dp
    options%delta = 1.7_dp
    call fit_model%initialize(1, 1, status)
    call huber_optimize_lbfgsb(fit_model, x, targets, options, result, status, &
        sample_weight=weights)
    call check(status_ok(status) .and. result%converged .and. &
        result%gradient_norm < 2.0e-5_dp, "bounded FortOpt Huber fit", failures)

    call objective%initialize(model, x, targets, 1.7_dp, 0.02_dp, status, &
        device_kind=FORTML_DEVICE_CUDA)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA objective refusal", failures)
    options%device_kind = FORTML_DEVICE_CUDA
    call huber_optimize_lbfgsb(fit_model, x, targets, options, result, status, &
        sample_weight=weights)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA optimizer refusal", failures)

    call model%set_parameters([0.5_dp, 0.8_dp], status)
    call objective%initialize(model, x, targets, 1.0_dp, 0.0_dp, status)
    theta = objective%parameters()
    theta(1) = 3.4_dp
    call objective%hvp(theta, direction(:size(theta)), product(:size(theta)), status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "Huber kink refusal", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL Huber-regression cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS weighted Huber-regression behavioral oracle"

contains
    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check
end program test_huber_regression
