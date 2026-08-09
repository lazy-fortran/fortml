program test_mlp_adam_optimizer_group_hypergradient
    !! Independent recurrence oracle for grouped coupled-L2 Adam.
    !!
    !! The expected trajectory below is intentionally written without calling
    !! FortML's trainer, loss, or optimizer.  It checks that the grouped Adam
    !! hypergradient adapter follows the production post-update multiplier
    !! contract and that its analytic outer gradient agrees with a central
    !! difference of the same independent recurrence.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_OK, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_optimizer_group_t
    use fortml_mlp_optimizer_group_hypergradient, only: &
        mlp_optimizer_group_hypergradient_options_t, &
        mlp_optimizer_group_hypergradient_objective_t, &
        MLP_OPTIMIZER_GROUP_ADAM
    implicit none

    type(mlp_t) :: model
    type(mlp_optimizer_group_hypergradient_options_t) :: options
    type(mlp_optimizer_group_hypergradient_objective_t) :: objective
    type(mlp_optimizer_group_t) :: weight_group, bias_group
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 1), target(4, 1), vx(3, 1), vtarget(3, 1)
    real(dp), allocatable :: p(:), g(:), plus(:), minus(:)
    real(dp) :: value, value_plus, value_minus, expected, fd
    integer :: i, failures

    failures = 0
    x(:, 1) = [-1.0_dp, -0.2_dp, 0.7_dp, 1.5_dp]
    target(:, 1) = 0.6_dp*x(:, 1)-0.15_dp
    vx(:, 1) = [-0.8_dp, 0.4_dp, 1.2_dp]
    vtarget(:, 1) = 0.6_dp*vx(:, 1)-0.15_dp
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    call check(status_ok(status), "model initialize", failures)
    call model%set_parameters([0.25_dp, 0.1_dp], status)
    call check(status_ok(status), "model parameters", failures)
    call weight_group%initialize("weight", 1, 1, 0.7_dp, status)
    call check(status_ok(status), "weight group", failures)
    call bias_group%initialize("bias", 2, 2, 1.3_dp, status)
    call check(status_ok(status), "bias group", failures)
    options%steps = 4
    options%learning_rate = 0.08_dp
    options%l2 = 0.03_dp
    options%beta1 = 0.85_dp
    options%beta2 = 0.97_dp
    options%epsilon = 1.0e-7_dp
    options%optimizer = MLP_OPTIMIZER_GROUP_ADAM
    options%lower_log_learning_rate = -6.0_dp
    options%upper_log_learning_rate = 0.0_dp
    options%lower_log_l2 = -7.0_dp
    options%upper_log_l2 = 0.0_dp
    options%lower_log_multiplier = -4.0_dp
    options%upper_log_multiplier = 4.0_dp
    allocate(options%groups(2))
    options%groups = [weight_group, bias_group]
    call objective%initialize(model, x, target, vx, vtarget, options, status)
    call check(status_ok(status), "Adam group objective initialize", failures)
    p = objective%parameters()
    allocate(g(size(p)), plus(size(p)), minus(size(p)))
    call objective%value_gradient(p, value, g, status)
    expected = independent_value(p, x(:, 1), target(:, 1), vx(:, 1), vtarget(:, 1), &
        options%steps, [0.7_dp, 1.3_dp], [0.25_dp, 0.1_dp])
    call check(status_ok(status) .and. abs(value-expected) < 2.0e-13_dp, &
        "Adam group value independent recurrence", failures)
    call check(ieee_is_finite(value) .and. all(ieee_is_finite(g)), &
        "Adam group finite products", failures)
    do i = 1, size(p)
        plus = p
        minus = p
        plus(i) = plus(i)+2.0e-6_dp
        minus(i) = minus(i)-2.0e-6_dp
        call objective%value_gradient(plus, value_plus, g, status)
        call objective%value_gradient(minus, value_minus, g, status)
        fd = (value_plus-value_minus)/(4.0e-6_dp)
        call check(status_ok(status), "Adam group finite-difference status", failures)
        ! Re-evaluate the analytic gradient at the base point after the two
        ! perturbation calls so the comparison is not alias-sensitive.
        call objective%value_gradient(p, value, g, status)
        call check(status_ok(status) .and. abs(fd-g(i)) < 2.0e-7_dp, &
            "Adam group central-difference gradient", failures)
    end do
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL grouped Adam recurrence cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS grouped Adam independent recurrence oracle"

contains

    real(dp) function independent_value(p, x, y, vx, vy, steps, scales, theta0) result(value)
        real(dp), intent(in) :: p(:), x(:), y(:), vx(:), vy(:), scales(:), theta0(:)
        integer, intent(in) :: steps
        real(dp) :: theta(2), m(2), v(2), grad(2), residual(size(x))
        real(dp) :: lr, l2, beta1, beta2, eps, b1, b2, direction(2), prediction(size(vx))
        integer :: step

        lr = exp(p(1))
        l2 = exp(p(2))
        beta1 = 0.85_dp
        beta2 = 0.97_dp
        eps = 1.0e-7_dp
        theta = theta0
        m = 0.0_dp
        v = 0.0_dp
        do step = 1, steps
            residual = theta(1)*x+theta(2)-y
            grad(1) = sum(residual*x)/real(size(x), dp)+l2*theta(1)
            grad(2) = sum(residual)/real(size(x), dp)+l2*theta(2)
            m = beta1*m+(1.0_dp-beta1)*grad
            v = beta2*v+(1.0_dp-beta2)*grad*grad
            b1 = 1.0_dp-beta1**step
            b2 = 1.0_dp-beta2**step
            direction = (m/b1)/(sqrt(v/b2)+eps)
            theta = theta-lr*scales*direction
        end do
        prediction = theta(1)*vx+theta(2)
        value = 0.5_dp*sum((prediction-vy)**2)/real(size(vx), dp)
    end function independent_value

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (*, '(a)') "FAIL: "//trim(label)
            failures = failures+1
        end if
    end subroutine check

end program test_mlp_adam_optimizer_group_hypergradient
