program test_mlp_multilabel_objective
    !! Independent finite-difference and FortOpt oracle for the multilabel head.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortopt_objective, only: objective_t
    use fortml_mlp_multilabel_classifier, only: &
        mlp_multilabel_classifier_t, mlp_multilabel_classifier_options_t, &
        mlp_multilabel_training_objective_t, mlp_multilabel_lbfgsb_options_t, &
        mlp_multilabel_lbfgsb_result_t, mlp_multilabel_optimize_lbfgsb
    implicit none

    type(mlp_multilabel_classifier_t), target :: model, optimizer_model, log_model
    type(mlp_multilabel_classifier_options_t) :: fit_options
    type(mlp_multilabel_training_objective_t) :: objective, log_objective
    type(mlp_multilabel_lbfgsb_options_t) :: lbfgsb_options
    type(mlp_multilabel_lbfgsb_result_t) :: lbfgsb_result
    type(objective_t) :: fortopt_objective
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 2)
    integer :: indicators(8, 2), failures, i
    real(dp) :: sample_weight(8), class_weight(2, 2)
    real(dp), allocatable :: parameters(:), direction(:), gradient(:), gradient_plus(:)
    real(dp), allocatable :: gradient_minus(:), product(:), plus(:), minus(:)
    real(dp), allocatable :: initial_parameters(:)
    real(dp), allocatable :: log_parameters(:)
    real(dp) :: value, value_plus, value_minus, tangent, lhs, rhs, h

    x = reshape([ &
        -2.0_dp, -1.0_dp, -1.0_dp, -2.0_dp, &
         2.0_dp,  1.0_dp,  1.0_dp,  2.0_dp, &
        -1.0_dp,  2.0_dp, -2.0_dp,  1.0_dp, &
         1.0_dp, -2.0_dp,  2.0_dp, -1.0_dp], shape(x))
    indicators = reshape([ &
        0, 0, 0, 0, 1, 1, 1, 1, &
        1, 1, 0, 0, 0, 0, 1, 1], shape(indicators))
    sample_weight = [0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp, 0.75_dp, 1.25_dp, &
        1.75_dp, 0.9_dp]
    class_weight = reshape([1.1_dp, 0.8_dp, 0.9_dp, 1.4_dp], shape(class_weight))
    failures = 0
    h = 1.0e-6_dp

    fit_options%max_epochs = 150
    fit_options%learning_rate = 0.03_dp
    fit_options%l2 = 1.0e-4_dp
    fit_options%tolerance = 1.0e-10_dp
    fit_options%initialization_seed = 19
    fit_options%restore_best = .false.
    call model%fit(x, indicators, status, options=fit_options)
    call check(status_ok(status) .and. model%fitted(), &
        "multilabel objective fixture fit", failures)
    initial_parameters = model%parameters()

    call objective%initialize(model, x, indicators, 0.02_dp, status, &
        optimize_l2=.true., sample_weight=sample_weight, class_weight=class_weight)
    call check(status_ok(status) .and. objective%parameter_count() == &
        model%parameter_count() + 1, "direct-L2 objective initialization", failures)
    parameters = objective%parameters()
    allocate(direction(size(parameters)), gradient(size(parameters)), &
        gradient_plus(size(parameters)), gradient_minus(size(parameters)), &
        product(size(parameters)), plus(size(parameters)), minus(size(parameters)))
    direction = [(0.007_dp*real(i, dp), i=1, size(parameters))]
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. value > 0.0_dp, &
        "multilabel objective value and gradient", failures)
    plus = parameters + h*direction
    minus = parameters - h*direction
    call objective%value_gradient(plus, value_plus, gradient_plus, status)
    call objective%value_gradient(minus, value_minus, gradient_minus, status)
    call check(status_ok(status), "multilabel objective central evaluations", failures)
    call objective%jvp(parameters, direction, value, tangent, status)
    call check(status_ok(status) .and. abs(tangent - &
        (value_plus-value_minus)/(2.0_dp*h)) < 2.0e-5_dp, &
        "multilabel objective JVP finite difference", failures)
    call objective%vjp(parameters, 1.7_dp, product, status)
    lhs = sum(product*direction)
    rhs = 1.7_dp*tangent
    call check(status_ok(status) .and. abs(lhs-rhs) < 3.0e-10_dp, &
        "multilabel objective VJP duality", failures)
    call objective%hvp(parameters, direction, product, status)
    call check(status_ok(status) .and. maxval(abs(product - &
        (gradient_plus-gradient_minus)/(2.0_dp*h))) < 3.0e-4_dp, &
        "multilabel objective HVP finite difference", failures)

    call objective%fortopt(fortopt_objective, status)
    call check(status_ok(status), "multilabel FortOpt context adapter", failures)
    call fortopt_objective%value_gradient(parameters, value_plus, gradient_plus, status)
    call check(status_ok(status) .and. abs(value_plus-value) < 1.0e-12_dp, &
        "multilabel FortOpt callback", failures)

    call log_objective%initialize(model, x, indicators, 0.02_dp, status, &
        optimize_log_l2=.true., sample_weight=sample_weight, class_weight=class_weight)
    log_parameters = log_objective%parameters()
    call check(status_ok(status) .and. log_objective%parameter_count() == &
        model%parameter_count() + 1 .and. &
        abs(log_parameters(size(log_parameters))-log(0.02_dp)) < 1.0e-12_dp, &
        "positive log-L2 objective initialization", failures)
    call log_objective%value_gradient(log_parameters, value, gradient, status)
    call check(status_ok(status) .and. value > 0.0_dp, &
        "positive log-L2 value and gradient", failures)

    ! Invalid weights fail before the objective installs a model pointer or data copies.
    call objective%initialize(model, x, indicators, 0.02_dp, status, &
        sample_weight=[1.0_dp, 0.0_dp], class_weight=class_weight)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "invalid sample-weight shape refusal", failures)
    call check(maxval(abs(model%parameters()-initial_parameters)) < 1.0e-14_dp, &
        "invalid objective initialization leaves model unchanged", failures)

    call optimizer_model%fit(x, indicators, status, options=fit_options)
    call check(status_ok(status), "L-BFGS-B fixture fit", failures)
    lbfgsb_options%max_iterations = 500
    lbfgsb_options%max_line_search = 80
    lbfgsb_options%gradient_tolerance = 1.0e-6_dp
    lbfgsb_options%step_tolerance = 1.0e-9_dp
    lbfgsb_options%objective_tolerance = 1.0e-8_dp
    lbfgsb_options%l2 = 0.02_dp
    lbfgsb_options%optimize_l2 = .true.
    lbfgsb_options%l2_lower_bound = 0.02_dp
    lbfgsb_options%l2_upper_bound = 1.0_dp
    call mlp_multilabel_optimize_lbfgsb(optimizer_model, x, indicators, lbfgsb_options, &
        lbfgsb_result, status, sample_weight=sample_weight, class_weight=class_weight)
    call check(status_ok(status) .and. lbfgsb_result%converged .and. &
        lbfgsb_result%l2 >= 0.0_dp, "direct-L2 bounded L-BFGS-B", failures)

    call log_model%fit(x, indicators, status, options=fit_options)
    call check(status_ok(status), "log-L2 fixture fit", failures)
    lbfgsb_options%optimize_l2 = .false.
    lbfgsb_options%optimize_log_l2 = .true.
    lbfgsb_options%gradient_tolerance = 1.0e-2_dp
    lbfgsb_options%l2 = 0.02_dp
    lbfgsb_options%log_l2_lower_bound = -8.0_dp
    lbfgsb_options%log_l2_upper_bound = 1.0_dp
    call mlp_multilabel_optimize_lbfgsb(log_model, x, indicators, lbfgsb_options, &
        lbfgsb_result, status, sample_weight=sample_weight, class_weight=class_weight)
    call check(status_ok(status) .and. lbfgsb_result%converged .and. &
        lbfgsb_result%l2 > 0.0_dp, "positive log-L2 bounded L-BFGS-B", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP multilabel-objective cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP multilabel-objective behavioral oracle"

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

end program test_mlp_multilabel_objective
