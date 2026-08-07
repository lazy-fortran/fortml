program test_mlp_lbfgsb_training
    !! Independent closed-form and refusal checks for the MLP L-BFGS-B adapter.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_lbfgsb_options_t, &
        mlp_lbfgsb_result_t, mlp_optimize_lbfgsb
    implicit none

    integer :: failures

    failures = 0
    call test_linear_closed_form(failures)
    call test_optimized_l2_block(failures)
    call test_invalid_bounds_refusal(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP L-BFGS-B cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP L-BFGS-B independent behavioral oracles"

contains

    subroutine test_linear_closed_form(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_lbfgsb_options_t) :: options
        type(mlp_lbfgsb_result_t) :: result
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2), expected_weight

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        theta = 0.0_dp
        call model%set_parameters(theta, status)
        options%l2 = 0.1_dp
        options%max_iterations = 200
        options%gradient_tolerance = 1.0e-8_dp
        options%step_tolerance = 1.0e-14_dp
        options%objective_tolerance = 1.0e-14_dp
        call mlp_optimize_lbfgsb(model, x, target, options, result, status)
        theta = model%parameters()
        expected_weight = (2.0_dp/3.0_dp)/(2.0_dp/3.0_dp + options%l2)
        call check(status_ok(status), "closed-form optimizer status", failures)
        call check(result%converged .and. result%iterations > 0, &
            "closed-form optimizer converged", failures)
        call check(abs(theta(1) - expected_weight) < 2.0e-6_dp .and. &
            abs(theta(2)) < 2.0e-7_dp, &
            "bounded MLP optimum agrees with ridge oracle", failures)
        call check(result%l2 == options%l2 .and. result%objective < 0.1_dp, &
            "fixed L2 and objective diagnostics", failures)
    end subroutine test_linear_closed_form

    subroutine test_optimized_l2_block(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_lbfgsb_options_t) :: options
        type(mlp_lbfgsb_result_t) :: result
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1), theta(2)

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        theta = [0.2_dp, 0.1_dp]
        call model%set_parameters(theta, status)
        options%l2 = 0.4_dp
        options%optimize_l2 = .true.
        options%l2_lower_bound = 0.0_dp
        options%l2_upper_bound = 1.0_dp
        options%max_iterations = 200
        options%gradient_tolerance = 1.0e-8_dp
        options%step_tolerance = 1.0e-14_dp
        options%objective_tolerance = 1.0e-14_dp
        call mlp_optimize_lbfgsb(model, x, target, options, result, status)
        theta = model%parameters()
        call check(status_ok(status), "optimized L2 status", failures)
        call check(result%converged .and. result%l2 <= 1.0e-8_dp, &
            "optimized L2 reaches its non-negative lower bound", failures)
        call check(abs(theta(1) - 1.0_dp) < 2.0e-6_dp .and. &
            abs(theta(2)) < 2.0e-7_dp, &
            "joint parameter and L2 fit oracle", failures)
    end subroutine test_optimized_l2_block

    subroutine test_invalid_bounds_refusal(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_lbfgsb_options_t) :: options
        type(mlp_lbfgsb_result_t) :: result
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2, 1)

        x(:, 1) = [-1.0_dp, 1.0_dp]
        target(:, 1) = [0.0_dp, 1.0_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        options%lower_bound = 2.0_dp
        options%upper_bound = -2.0_dp
        call mlp_optimize_lbfgsb(model, x, target, options, result, status)
        call check(.not. status_ok(status), "invalid MLP bounds refusal", failures)
    end subroutine test_invalid_bounds_refusal

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL: "//description
        end if
    end subroutine check

end program test_mlp_lbfgsb_training
