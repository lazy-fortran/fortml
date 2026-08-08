program test_mlp_poisson_objective
    !! Independent finite-difference and adjoint oracle for the smooth
    !! Poisson MLP objective and its bounded FortOpt adapter.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_poisson, only: mlp_poisson_training_objective_t, &
        mlp_poisson_lbfgsb_options_t, mlp_poisson_lbfgsb_result_t, &
        mlp_poisson_optimize_lbfgsb
    implicit none

    integer, parameter :: dp = real64
    type(mlp_t), target :: model, optimizer_model
    type(mlp_t), target :: bad_model
    type(mlp_poisson_training_objective_t) :: objective
    type(mlp_poisson_lbfgsb_options_t) :: options
    type(mlp_poisson_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), targets(8, 1), sample_weight(8)
    real(dp), allocatable :: parameters(:), direction(:), gradient(:), gradient_plus(:), &
        gradient_minus(:), product(:)
    real(dp) :: value, value_plus, value_minus, tangent, h, error
    integer :: failures, i

    failures = 0
    x(:, 1) = [-1.5_dp, -1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp]
    targets(:, 1) = [0.45_dp, 0.62_dp, 0.85_dp, 1.05_dp, 1.35_dp, 1.80_dp, 2.30_dp, 3.10_dp]
    sample_weight = [0.5_dp, 0.8_dp, 1.0_dp, 1.2_dp, 1.5_dp, 1.1_dp, 0.9_dp, 1.3_dp]

    call model%initialize([1, 3, 1], status, hidden_activation=2, &
        output_activation=MLP_LINEAR, initialization_seed=31)
    call check(status_ok(status), "Poisson model initialization", failures)
    call model%set_parameters([0.08_dp, -0.11_dp, 0.03_dp, 0.07_dp, &
        -0.04_dp, 0.02_dp, 0.06_dp, -0.05_dp, 0.01_dp, 0.04_dp], status)
    call check(status_ok(status), "Poisson model state", failures)
    call objective%initialize(model, x, targets, 0.012_dp, status, &
        optimize_l2=.true., sample_weight=sample_weight)
    call check(status_ok(status), "Poisson objective initialization", failures)

    parameters = objective%parameters()
    allocate(direction(size(parameters)), gradient(size(parameters)), &
        gradient_plus(size(parameters)), gradient_minus(size(parameters)), &
        product(size(parameters)))
    direction = [(0.007_dp*real(i, dp), i=1, size(direction))]
    h = 2.0e-6_dp
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "Poisson value/gradient", failures)
    call objective%jvp(parameters, direction, value_plus, tangent, status)
    call check(status_ok(status) .and. abs(tangent-dot_product(gradient, direction)) < 2.0e-11_dp, &
        "Poisson JVP contraction", failures)
    call objective%vjp(parameters, 1.7_dp, product, status)
    call check(status_ok(status) .and. maxval(abs(product-1.7_dp*gradient)) < 2.0e-11_dp, &
        "Poisson VJP duality", failures)

    call objective%value_gradient(parameters+h*direction, value_plus, gradient_plus, status)
    call objective%value_gradient(parameters-h*direction, value_minus, gradient_minus, status)
    call objective%value_gradient(parameters, value, gradient, status)
    call objective%hvp(parameters, direction, product, status)
    error = maxval(abs(product-(gradient_plus-gradient_minus)/(2.0_dp*h)))
    call check(status_ok(status) .and. error < 5.0e-5_dp, &
        "Poisson HVP finite-difference oracle", failures)
    call check(abs(tangent-(value_plus-value_minus)/(2.0_dp*h)) < 3.0e-5_dp, &
        "Poisson value finite-difference oracle", failures)

    call optimizer_model%initialize([1, 1], status, hidden_activation=MLP_LINEAR, &
        output_activation=MLP_LINEAR, initialization_seed=41)
    call check(status_ok(status), "Poisson optimizer model initialization", failures)
    options%max_iterations = 160
    options%max_line_search = 80
    options%gradient_tolerance = 1.0e-7_dp
    options%step_tolerance = 1.0e-10_dp
    options%objective_tolerance = 1.0e-10_dp
    options%lower_bound = -8.0_dp
    options%upper_bound = 8.0_dp
    options%l2 = 1.0e-3_dp
    call mlp_poisson_optimize_lbfgsb(optimizer_model, x, targets, options, result, status, &
        sample_weight=sample_weight)
    call check(status_ok(status) .and. result%converged .and. &
        result%objective < 1.3_dp, "Poisson bounded L-BFGS-B fit", failures)

    call bad_model%initialize([1, 1], status, hidden_activation=MLP_LINEAR, &
        output_activation=MLP_LINEAR, initialization_seed=42)
    call objective%initialize(bad_model, x, targets, 0.0_dp, status, &
        device_kind=FORTML_DEVICE_CUDA)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA Poisson objective refusal", failures)
    options%device_kind = FORTML_DEVICE_CUDA
    call mlp_poisson_optimize_lbfgsb(bad_model, x, targets, options, result, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA Poisson optimizer refusal", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP Poisson-objective cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP Poisson-objective behavioral oracle"

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

end program test_mlp_poisson_objective
