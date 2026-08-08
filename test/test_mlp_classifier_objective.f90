program test_mlp_classifier_objective
    !! Independent finite-difference and adjoint oracle for multiclass MLP
    !! cross-entropy products and the bounded FortOpt adapter.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp_classifier, only: mlp_classifier_t, &
        mlp_classifier_options_t, mlp_classifier_training_objective_t, &
        mlp_classifier_lbfgsb_options_t, mlp_classifier_lbfgsb_result_t, &
        mlp_classifier_optimize_lbfgsb
    implicit none

    integer, parameter :: dp = real64
    type(mlp_classifier_t) :: model, optimizer_model
    type(mlp_classifier_options_t) :: fit_options
    type(mlp_classifier_training_objective_t) :: objective
    type(mlp_classifier_lbfgsb_options_t) :: lbfgsb_options
    type(mlp_classifier_lbfgsb_result_t) :: lbfgsb_result
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 2), sample_weight(9), class_weight(3)
    real(dp), allocatable :: parameters(:), direction(:), gradient(:), &
        gradient_plus(:), gradient_minus(:), product(:)
    real(dp) :: value, value_plus, value_minus, tangent, tangent_left, tangent_right
    real(dp) :: h, product_error
    integer :: labels(9), failures, i

    x = reshape([ &
        -2.0_dp, -1.0_dp, -1.0_dp, -2.0_dp, 0.0_dp, 2.0_dp, &
        0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 2.0_dp, 1.0_dp, &
        2.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, 2.0_dp, 1.0_dp], shape(x))
    labels = [31, 31, -7, -7, 42, 42, 42, 31, -7]
    sample_weight = [0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp, 0.75_dp, 1.25_dp, &
        1.75_dp, 2.25_dp, 1.1_dp]
    class_weight = [1.25_dp, 0.8_dp, 1.4_dp]
    failures = 0

    fit_options%max_epochs = 8
    fit_options%learning_rate = 0.02_dp
    fit_options%initialization_seed = 29
    fit_options%restore_best = .false.
    call model%fit(x, labels, status, hidden_layer_sizes=[3], options=fit_options)
    call check(status_ok(status), "objective fixture fit", failures)
    call objective%initialize(model, x, labels, 2.0e-3_dp, status, &
        optimize_l2=.true., sample_weight=sample_weight, class_weight=class_weight)
    call check(status_ok(status), "weighted multiclass objective initialize", failures)

    parameters = objective%parameters()
    allocate(direction(size(parameters)), gradient(size(parameters)), &
        gradient_plus(size(parameters)), gradient_minus(size(parameters)), &
        product(size(parameters)))
    direction = [(0.009_dp*real(i, dp), i=1, size(direction))]
    h = 1.0e-6_dp
    call objective%value_gradient(parameters, value, gradient, status)
    call objective%jvp(parameters, direction, value_plus, tangent, status)
    tangent_left = dot_product(gradient, direction)
    call check(status_ok(status) .and. abs(tangent - tangent_left) < 1.0e-11_dp, &
        "multiclass objective JVP contraction", failures)
    call objective%vjp(parameters, 1.7_dp, product, status)
    tangent_left = dot_product(1.7_dp*gradient, direction)
    tangent_right = dot_product(product, direction)
    call check(status_ok(status) .and. abs(tangent_left - tangent_right) < 1.0e-11_dp, &
        "multiclass objective VJP duality", failures)

    call objective%value_gradient(parameters + h*direction, value_plus, &
        gradient_plus, status)
    call objective%value_gradient(parameters - h*direction, value_minus, &
        gradient_minus, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call objective%hvp(parameters, direction, product, status)
    product_error = maxval(abs(product - (gradient_plus - gradient_minus)/(2.0_dp*h)))
    call check(status_ok(status) .and. product_error < 3.0e-4_dp, &
        "multiclass objective HVP finite-difference oracle", failures)
    call check(abs(tangent - (value_plus - value_minus)/(2.0_dp*h)) < 3.0e-5_dp, &
        "multiclass objective value finite-difference oracle", failures)

    call optimizer_model%fit(x, labels, status, hidden_layer_sizes=[3], &
        options=fit_options)
    lbfgsb_options%max_iterations = 500
    lbfgsb_options%max_line_search = 80
    lbfgsb_options%gradient_tolerance = 1.0e-5_dp
    lbfgsb_options%step_tolerance = 1.0e-8_dp
    lbfgsb_options%objective_tolerance = 1.0e-8_dp
    lbfgsb_options%lower_bound = -25.0_dp
    lbfgsb_options%upper_bound = 25.0_dp
    lbfgsb_options%l2 = 1.0e-3_dp
    call mlp_classifier_optimize_lbfgsb(optimizer_model, x, labels, &
        lbfgsb_options, result=lbfgsb_result, status=status, &
        sample_weight=sample_weight, class_weight=class_weight)
    call check(status_ok(status) .and. lbfgsb_result%converged .and. &
        lbfgsb_result%objective < 1.0_dp, "multiclass bounded L-BFGS-B fit", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP multiclass-objective cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP multiclass-objective behavioral oracle"

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

end program test_mlp_classifier_objective
