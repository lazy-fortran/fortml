program test_mlp_optimizer_group_affine_hvp
    !! Independent oracle for the affine optimizer-group outer HVP.
    !!
    !! The reference recurrence below does not call FortML's loss, trainer, or
    !! optimizer.  It checks the second product against a central difference
    !! of the independently differentiated scalar trajectory, then verifies
    !! that a nonlinear network keeps the typed derivative boundary.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_optimizer_group_t
    use fortml_mlp_optimizer_group_hypergradient, only: &
        mlp_optimizer_group_hypergradient_options_t, &
        mlp_optimizer_group_hypergradient_objective_t
    implicit none

    type(mlp_t) :: model, nonlinear_model
    type(mlp_optimizer_group_hypergradient_options_t) :: options
    type(mlp_optimizer_group_hypergradient_objective_t) :: objective, nonlinear_objective
    type(mlp_optimizer_group_t) :: weight_group, bias_group
    type(fortnum_status_t) :: status
    real(dp) :: x(5, 1), target(5, 1), vx(4, 1), vtarget(4, 1)
    real(dp), allocatable :: p(:), plus(:), minus(:), hvp(:), gp(:), gm(:), expected(:)
    real(dp), allocatable :: nonlinear_direction(:)
    real(dp) :: value, direction(4), h_outer, h_gradient, max_error
    integer :: i, failures

    failures = 0
    x(:, 1) = [-1.2_dp, -0.4_dp, 0.3_dp, 1.0_dp, 1.8_dp]
    target(:, 1) = 0.55_dp*x(:, 1)-0.2_dp
    vx(:, 1) = [-0.9_dp, -0.1_dp, 0.8_dp, 1.4_dp]
    vtarget(:, 1) = 0.55_dp*vx(:, 1)-0.2_dp
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call check(status_ok(status), "affine model initialize", failures)
    call model%set_parameters([0.22_dp, 0.08_dp], status)
    call check(status_ok(status), "affine model parameters", failures)
    call weight_group%initialize("weight", 1, 1, 0.75_dp, status)
    call check(status_ok(status), "weight group initialize", failures)
    call bias_group%initialize("bias", 2, 2, 1.25_dp, status)
    call check(status_ok(status), "bias group initialize", failures)
    options%steps = 5
    options%learning_rate = 0.06_dp
    options%l2 = 0.025_dp
    options%lower_log_learning_rate = -8.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -8.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_log_multiplier = -4.0_dp
    options%upper_log_multiplier = 4.0_dp
    allocate(options%groups(2))
    options%groups = [weight_group, bias_group]
    call objective%initialize(model, x, target, vx, vtarget, options, status)
    call check(status_ok(status), "affine HVP objective initialize", failures)
    p = objective%parameters()
    allocate(plus(size(p)), minus(size(p)), hvp(size(p)), gp(size(p)), gm(size(p)), &
        expected(size(p)))
    direction = [0.18_dp, -0.11_dp, 0.07_dp, -0.09_dp]
    call objective%hvp(p, direction, hvp, status)
    call check(status_ok(status) .and. all(ieee_is_finite(hvp)), &
        "affine outer HVP status and finiteness", failures)
    h_outer = 2.0e-4_dp
    h_gradient = 2.0e-5_dp
    plus = p+h_outer*direction
    minus = p-h_outer*direction
    call independent_gradient(plus, x(:, 1), target(:, 1), vx(:, 1), vtarget(:, 1), &
        options%steps, h_gradient, gp)
    call independent_gradient(minus, x(:, 1), target(:, 1), vx(:, 1), vtarget(:, 1), &
        options%steps, h_gradient, gm)
    expected = (gp-gm)/(2.0_dp*h_outer)
    max_error = maxval(abs(hvp-expected))
    call check(max_error < 3.0e-6_dp, "affine HVP independent recurrence", failures)

    call nonlinear_model%initialize([1, 2, 1], status, output_activation=MLP_LINEAR)
    call check(status_ok(status), "nonlinear model initialize", failures)
    call weight_group%initialize("first", 1, 4, 0.75_dp, status)
    call bias_group%initialize("second", 5, 6, 1.25_dp, status)
    options%groups = [weight_group, bias_group]
    call nonlinear_objective%initialize(nonlinear_model, x, target, vx, vtarget, options, status)
    call check(status_ok(status), "nonlinear HVP objective initialize", failures)
    p = nonlinear_objective%parameters()
    deallocate(plus, minus)
    allocate(plus(size(p)), minus(size(p)), nonlinear_direction(size(p)))
    hvp = 1.0_dp
    nonlinear_direction = 0.0_dp
    call nonlinear_objective%hvp(p, nonlinear_direction, hvp, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(hvp == 0.0_dp), &
        "nonlinear outer HVP typed refusal", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL affine optimizer-group HVP cases: ", failures
        error stop 1
    end if
    write (*, '(a,es12.4)') "PASS affine optimizer-group HVP oracle max_error=", max_error

contains

    subroutine independent_gradient(p, x, y, vx, vy, steps, h, gradient)
        real(dp), intent(in) :: p(:), x(:), y(:), vx(:), vy(:), h
        integer, intent(in) :: steps
        real(dp), intent(out) :: gradient(:)
        real(dp) :: plus(size(p)), minus(size(p))
        integer :: j

        do j = 1, size(p)
            plus = p
            minus = p
            plus(j) = plus(j)+h
            minus(j) = minus(j)-h
            gradient(j) = (independent_value(plus, x, y, vx, vy, steps) - &
                independent_value(minus, x, y, vx, vy, steps))/(2.0_dp*h)
        end do
    end subroutine independent_gradient

    real(dp) function independent_value(p, x, y, vx, vy, steps) result(value)
        real(dp), intent(in) :: p(:), x(:), y(:), vx(:), vy(:)
        integer, intent(in) :: steps
        real(dp) :: theta(2), gradient(2), residual(size(x)), prediction(size(vx))
        real(dp) :: learning_rate, l2, effective_scales(2)
        integer :: step

        learning_rate = exp(p(1))
        l2 = exp(p(2))
        effective_scales = exp(p(3:4))
        theta = [0.22_dp, 0.08_dp]
        do step = 1, steps
            residual = theta(1)*x+theta(2)-y
            gradient(1) = sum(residual*x)/real(size(x), dp)+l2*theta(1)
            gradient(2) = sum(residual)/real(size(x), dp)+l2*theta(2)
            theta = theta-learning_rate*effective_scales*gradient
        end do
        prediction = theta(1)*vx+theta(2)
        value = 0.5_dp*sum((prediction-vy)**2)/real(size(vx), dp)
    end function independent_value

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures+1
            write (*, '(a)') "FAIL: "//trim(label)
        end if
    end subroutine check

end program test_mlp_optimizer_group_affine_hvp
