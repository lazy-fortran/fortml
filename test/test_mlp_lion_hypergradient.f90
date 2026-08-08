program test_mlp_lion_hypergradient
    !! Independent behavioral oracle for the piecewise-smooth Lion trajectory.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, &
        FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_lion_hypergradient, only: &
        mlp_lion_hypergradient_options_t, &
        mlp_lion_hypergradient_objective_t, &
        mlp_lion_hypergradient_result_t, &
        mlp_optimize_lion_hyperparameters, &
        MLP_LION_HYPERPARAMETER_COUNT
    implicit none

    integer :: failures

    failures = 0
    call test_products_and_fd(failures)
    call test_fortopt_adapter(failures)
    call test_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP Lion hypergradient cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP Lion hypergradient independent behavioral oracles"

contains

    subroutine fixture(model, options, train_x, train_target, validation_x, validation_target, &
            status)
        type(mlp_t), intent(out) :: model
        type(mlp_lion_hypergradient_options_t), intent(out) :: options
        real(dp), intent(out) :: train_x(4, 1), train_target(4, 1)
        real(dp), intent(out) :: validation_x(3, 1), validation_target(3, 1)
        type(fortnum_status_t), intent(out) :: status

        train_x(:, 1) = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp]
        train_target(:, 1) = 0.0_dp
        validation_x(:, 1) = [1.5_dp, 2.5_dp, 3.5_dp]
        validation_target(:, 1) = [0.2_dp, 0.3_dp, 0.4_dp]
        call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
        if (.not. status_ok(status)) return
        call model%set_parameters([0.7_dp, 0.35_dp], status)
        if (.not. status_ok(status)) return
        options%steps = 3
        options%learning_rate = 1.0e-4_dp
        options%l2 = 1.0e-3_dp
        options%beta1 = 0.81_dp
        options%beta2 = 0.93_dp
        options%sign_margin = 1.0e-14_dp
        options%max_iterations = 16
        options%gradient_tolerance = 1.0e-4_dp
        options%objective_tolerance = 1.0e-14_dp
    end subroutine fixture

    subroutine test_products_and_fd(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_lion_hypergradient_options_t) :: options
        type(mlp_lion_hypergradient_objective_t) :: objective
        type(fortnum_status_t) :: status
        real(dp) :: train_x(4, 1), train_target(4, 1)
        real(dp) :: validation_x(3, 1), validation_target(3, 1)
        real(dp) :: parameters(MLP_LION_HYPERPARAMETER_COUNT)
        real(dp) :: gradient(MLP_LION_HYPERPARAMETER_COUNT)
        real(dp) :: direction(MLP_LION_HYPERPARAMETER_COUNT)
        real(dp) :: plus(MLP_LION_HYPERPARAMETER_COUNT)
        real(dp) :: minus(MLP_LION_HYPERPARAMETER_COUNT)
        real(dp) :: value, tangent, value_plus, value_minus, fd, scale
        integer :: i

        call fixture(model, options, train_x, train_target, validation_x, validation_target, status)
        call check(status_ok(status), "Lion fixture setup", failures)
        call objective%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        call check(status_ok(status), "Lion objective initialization", failures)
        parameters = objective%parameters()
        call objective%value_gradient(parameters, value, gradient, status)
        call check(status_ok(status) .and. ieee_is_finite(value) .and. &
            all(ieee_is_finite(gradient)), "Lion value and gradient", failures)
        direction = [0.11_dp, -0.07_dp, 0.13_dp, -0.09_dp]
        call objective%jvp(parameters, direction, value, tangent, status)
        call check(status_ok(status) .and. abs(tangent-dot_product(gradient, direction)) < 1.0e-12_dp, &
            "Lion JVP contraction", failures)
        do i = 1, MLP_LION_HYPERPARAMETER_COUNT
            plus = parameters
            minus = parameters
            scale = 2.0e-6_dp
            plus(i) = plus(i) + scale
            minus(i) = minus(i) - scale
            call objective%jvp(plus, direction*0.0_dp, value_plus, tangent, status)
            call check(status_ok(status), "Lion plus finite difference branch", failures)
            call objective%jvp(minus, direction*0.0_dp, value_minus, tangent, status)
            call check(status_ok(status), "Lion minus finite difference branch", failures)
            fd = (value_plus-value_minus)/(2.0_dp*scale)
            call check(abs(fd-gradient(i)) < 2.0e-8_dp, &
                "Lion central-difference hypergradient", failures)
        end do
        call objective%vjp(parameters, 1.7_dp, direction, status)
        call check(status_ok(status) .and. maxval(abs(direction-1.7_dp*gradient)) < 1.0e-12_dp, &
            "Lion scalar VJP", failures)
    end subroutine test_products_and_fd

    subroutine test_fortopt_adapter(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_lion_hypergradient_options_t) :: options
        type(mlp_lion_hypergradient_result_t) :: result
        type(fortnum_status_t) :: status
        real(dp) :: train_x(4, 1), train_target(4, 1)
        real(dp) :: validation_x(3, 1), validation_target(3, 1)

        call fixture(model, options, train_x, train_target, validation_x, validation_target, status)
        call mlp_optimize_lion_hyperparameters(model, train_x, train_target, validation_x, &
            validation_target, options, result, status)
        call check(status_ok(status) .or. status%code == FORTNUM_CONVERGENCE_ERROR, &
            "Lion FortOpt adapter returns a defined result", failures)
        call check(ieee_is_finite(result%objective) .and. ieee_is_finite(result%learning_rate) .and. &
            result%learning_rate > 0.0_dp .and. result%beta1 > 0.0_dp .and. result%beta1 < 1.0_dp .and. &
            result%beta2 > 0.0_dp .and. result%beta2 < 1.0_dp, &
            "Lion FortOpt result physical coordinates", failures)
    end subroutine test_fortopt_adapter

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_lion_hypergradient_options_t) :: options
        type(mlp_lion_hypergradient_objective_t) :: objective
        type(fortnum_status_t) :: status
        real(dp) :: train_x(4, 1), train_target(4, 1)
        real(dp) :: validation_x(3, 1), validation_target(3, 1)
        real(dp) :: value, gradient(MLP_LION_HYPERPARAMETER_COUNT)

        call fixture(model, options, train_x, train_target, validation_x, validation_target, status)
        options%device_kind = FORTML_DEVICE_CUDA
        call objective%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, "Lion CUDA refusal", failures)

        call fixture(model, options, train_x, train_target, validation_x, validation_target, status)
        options%sign_margin = 1.0_dp
        call objective%initialize(model, train_x, train_target, validation_x, &
            validation_target, options, status)
        call objective%value_gradient(objective%parameters(), value, gradient, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "Lion nondifferentiable sign-boundary refusal", failures)
    end subroutine test_refusals

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_mlp_lion_hypergradient
