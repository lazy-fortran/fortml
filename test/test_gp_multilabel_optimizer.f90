program test_gp_multilabel_optimizer
    !! Independent finite-difference and transactional oracle for shared GP HPO.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_multilabel_classification, only: &
        gp_multilabel_classification_t, gp_multilabel_classification_options_t, &
        gp_multilabel_training_objective_t, gp_multilabel_lbfgsb_options_t, &
        gp_multilabel_lbfgsb_result_t, gp_multilabel_optimize_lbfgsb
    implicit none

    type(gp_multilabel_classification_t), target :: model
    type(gp_multilabel_classification_options_t) :: fit_options
    type(gp_multilabel_training_objective_t) :: objective
    type(gp_multilabel_lbfgsb_options_t) :: optimizer_options
    type(gp_multilabel_lbfgsb_result_t) :: optimizer_result
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    real(dp) :: x(10, 1), value, value_plus, value_minus, tangent, lhs, rhs
    real(dp), allocatable :: parameters(:), gradient(:), gradient_plus(:), gradient_minus(:)
    real(dp), allocatable :: direction(:), saved(:), objective_gradient(:)
    real(dp) :: h
    integer :: indicators(10, 2), failures

    x(:, 1) = [-2.0_dp, -1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, &
        0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp]
    indicators(:, 1) = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    indicators(:, 2) = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    failures = 0

    kernel = make_rbf_kernel(1, 1.3_dp, 0.75_dp, status)
    fit_options%max_iterations = 100
    fit_options%tolerance = 1.0e-9_dp
    fit_options%jitter = 1.0e-7_dp
    call model%fit(x, indicators, kernel, status, fit_options)
    call check(status_ok(status), "multilabel GP fit", failures)

    parameters = model%shared_parameters()
    allocate(saved(size(parameters)), direction(size(parameters)))
    allocate(gradient(size(parameters)), gradient_plus(size(parameters)), &
        gradient_minus(size(parameters)), objective_gradient(size(parameters)))
    saved = parameters
    direction = [0.17_dp, -0.11_dp]
    call model%fixed_state_value_gradient(parameters, value, gradient, status)
    call check(status_ok(status) .and. size(parameters) == 2, &
        "shared fixed-state objective shape", failures)

    h = 1.0e-5_dp
    call model%fixed_state_value_gradient(parameters + h*direction, value_plus, &
        gradient_plus, status)
    call model%fixed_state_value_gradient(parameters - h*direction, value_minus, &
        gradient_minus, status)
    call check(status_ok(status) .and. abs(dot_product(gradient, direction) - &
        (value_plus - value_minus)/(2.0_dp*h)) < 2.0e-5_dp, &
        "shared objective directional finite difference", failures)
    call model%set_shared_parameters(parameters, status)
    call check(status_ok(status), "restore shared parameters", failures)
    call model%shared_hyperparameter_gradient(objective_gradient, status)
    call check(status_ok(status) .and. maxval(abs(objective_gradient + gradient)) < 2.0e-7_dp, &
        "shared gradient sign contract", failures)

    call objective%initialize(model, status)
    call objective%value_gradient(parameters, value, objective_gradient, status)
    call objective%jvp(parameters, direction, value_plus, tangent, status)
    call objective%vjp(parameters, 1.0_dp, gradient_plus, status)
    lhs = dot_product(gradient_plus, direction)
    rhs = tangent
    call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-7_dp, &
        "FortOpt objective JVP/VJP duality", failures)

    call model%set_shared_parameters([0.0_dp], status)
    parameters = model%shared_parameters()
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. &
        maxval(abs(parameters - saved)) < 1.0e-14_dp, &
        "invalid shared update is transactional", failures)

    optimizer_options%max_iterations = 200
    optimizer_options%max_line_search = 40
    optimizer_options%lower_bound = -1.0_dp
    optimizer_options%upper_bound = 1.0_dp
    optimizer_options%gradient_tolerance = 1.0e-3_dp
    optimizer_options%objective_tolerance = 1.0e-5_dp
    optimizer_options%step_tolerance = 1.0e-8_dp
    call gp_multilabel_optimize_lbfgsb(model, optimizer_options, optimizer_result, status)
    call check(status_ok(status) .and. optimizer_result%converged .and. &
        optimizer_result%iterations >= 0, "FortOpt shared GP optimization", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL multilabel GP optimizer cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS multilabel GP optimizer independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-multilabel-opt] "//description
        end if
    end subroutine check

end program test_gp_multilabel_optimizer
