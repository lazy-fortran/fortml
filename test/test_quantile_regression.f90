program test_quantile_regression
    !! Independent finite-difference and weighted-optimizer oracle for the
    !! multi-output linear quantile estimator.
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_quantile_regression, only: quantile_regression_t, &
        quantile_training_objective_t, quantile_lbfgsb_options_t, &
        quantile_lbfgsb_result_t, quantile_optimize_lbfgsb
    implicit none
    integer, parameter :: dp = real64
    type(quantile_regression_t), target :: model, fit_model
    type(quantile_training_objective_t) :: objective
    type(quantile_lbfgsb_options_t) :: options
    type(quantile_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    real(dp) :: x(7, 2), targets(7, 2), weights(7), levels(2)
    real(dp), allocatable :: theta(:), direction(:), gradient(:), gradient_plus(:), &
        gradient_minus(:), product(:), parameter_bar(:), prediction(:, :), &
        prediction_dot(:, :), input_dot(:, :), input_bar(:, :), output_bar(:, :)
    real(dp), allocatable :: state_before(:)
    real(dp) :: value, value_plus, value_minus, tangent, h, error, &
        contraction, prediction_contraction
    integer :: failures, i

    failures = 0
    x = reshape([ &
        -1.2_dp, -0.8_dp, -0.3_dp, 0.1_dp, 0.6_dp, 1.0_dp, 1.5_dp, &
        0.7_dp, -0.5_dp, 1.1_dp, -1.3_dp, 0.4_dp, 1.7_dp, -0.9_dp], &
        shape(x))
    targets(:, 1) = [0.3_dp, 0.8_dp, -0.2_dp, 1.1_dp, 0.1_dp, 1.8_dp, 0.7_dp]
    targets(:, 2) = [-0.4_dp, 0.5_dp, 1.4_dp, -0.8_dp, 1.1_dp, 0.2_dp, 1.7_dp]
    weights = [0.6_dp, 1.0_dp, 1.3_dp, 0.8_dp, 1.2_dp, 0.9_dp, 0.7_dp]
    levels = [0.75_dp, 0.25_dp]

    call model%initialize(2, 2, levels, status)
    call check(status_ok(status), "model initialization", failures)
    call check(maxval(abs(model%quantile_levels() - [0.25_dp, 0.75_dp])) < 1.0e-14_dp, &
        "deterministic sorted levels", failures)
    call model%set_parameters([0.4_dp, -0.7_dp, 1.1_dp, 0.25_dp, 0.6_dp, -0.3_dp], &
        status)
    call check(status_ok(status), "model parameter state", failures)
    state_before = model%parameters()
    call model%fit(x, targets, status, quantile_levels=[0.25_dp], l2=0.01_dp)
    call check(.not. status_ok(status) .and. model%fitted() .and. &
        maxval(abs(model%parameters()-state_before)) < 1.0e-14_dp, &
        "transactional malformed fit", failures)
    call objective%initialize(model, x, targets, 0.03_dp, status, &
        sample_weight=weights)
    call check(status_ok(status), "objective initialization", failures)
    theta = objective%parameters()
    allocate(direction(size(theta)), gradient(size(theta)), gradient_plus(size(theta)), &
        gradient_minus(size(theta)), product(size(theta)), parameter_bar(size(theta)))
    direction = [(0.011_dp*real(i, dp), i=1, size(theta))]
    h = 2.0e-7_dp
    call objective%value_gradient(theta, value, gradient, status)
    call check(status_ok(status), "value and gradient", failures)
    call objective%value_gradient(theta+h*direction, value_plus, gradient_plus, status)
    call objective%value_gradient(theta-h*direction, value_minus, gradient_minus, status)
    call check(status_ok(status) .and. &
        abs((value_plus-value_minus)/(2.0_dp*h)-dot_product(gradient, direction)) < 2.0e-7_dp, &
        "objective finite-difference gradient", failures)
    call objective%jvp(theta, direction, value_plus, tangent, status)
    call check(status_ok(status) .and. &
        abs(tangent-dot_product(gradient, direction)) < 2.0e-11_dp, &
        "objective JVP contraction", failures)
    call objective%vjp(theta, 1.7_dp, parameter_bar, status)
    call check(status_ok(status) .and. maxval(abs(parameter_bar-1.7_dp*gradient)) < 2.0e-11_dp, &
        "objective VJP duality", failures)
    call objective%hvp(theta, direction, product, status)
    error = maxval(abs(product-(gradient_plus-gradient_minus)/(2.0_dp*h)))
    call check(status_ok(status) .and. error < 2.0e-7_dp, &
        "piecewise objective HVP oracle", failures)

    allocate(prediction(size(x, 1), 2), prediction_dot(size(x, 1), 2), &
        input_dot(size(x, 1), 2), input_bar(size(x, 1), 2), &
        output_bar(size(x, 1), 2))
    input_dot = reshape([(0.003_dp*real(i, dp), i=1, size(x))], shape(x))
    output_bar = reshape([(0.017_dp*real(i, dp), i=1, size(x))], shape(x))
    call model%predict_jvp(x, direction, input_dot, prediction, prediction_dot, status)
    call check(status_ok(status), "prediction JVP", failures)
    call model%predict_vjp(x, output_bar, parameter_bar, input_bar, status)
    contraction = dot_product(reshape(prediction_dot, [size(prediction_dot)]), &
        reshape(output_bar, [size(output_bar)]))
    prediction_contraction = dot_product(direction, parameter_bar) + &
        dot_product(reshape(input_dot, [size(input_dot)]), &
        reshape(input_bar, [size(input_bar)]))
    call check(status_ok(status) .and. &
        abs(contraction-prediction_contraction) < 2.0e-11_dp, &
        "prediction JVP/VJP duality", failures)

    options%max_iterations = 500
    options%max_line_search = 100
    options%gradient_tolerance = 2.0e-7_dp
    options%step_tolerance = 1.0e-10_dp
    options%objective_tolerance = 1.0e-10_dp
    options%lower_bound = -15.0_dp
    options%upper_bound = 15.0_dp
    options%l2 = 1.0e-3_dp
    call fit_model%initialize(2, 2, [0.25_dp, 0.75_dp], status)
    call quantile_optimize_lbfgsb(fit_model, x, targets, options, result, status, &
        sample_weight=weights)
    call check(status_ok(status) .and. result%converged .and. &
        ieee_is_finite(result%objective) .and. ieee_is_finite(result%exact_gradient_norm), &
        "bounded FortOpt quantile fit", failures)

    call objective%initialize(model, x, targets, 0.03_dp, status, &
        device_kind=FORTML_DEVICE_CUDA)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA objective refusal", failures)
    options%device_kind = FORTML_DEVICE_CUDA
    call quantile_optimize_lbfgsb(fit_model, x, targets, options, result, status, &
        sample_weight=weights)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA optimizer refusal", failures)

    call model%initialize(1, 1, [0.5_dp], status)
    call objective%initialize(model, reshape([0.0_dp], [1, 1]), &
        reshape([0.0_dp], [1, 1]), 0.0_dp, status)
    deallocate(direction, product)
    theta = objective%parameters()
    allocate(direction(size(theta)), product(size(theta)))
    direction = 1.0_dp
    call objective%hvp(theta, direction, product, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "exact pinball kink refusal", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL quantile-regression cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS weighted multi-output quantile-regression behavioral oracle"

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
end program test_quantile_regression
