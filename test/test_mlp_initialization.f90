program test_mlp_initialization
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp, only: mlp_t, MLP_TANH
    use fortopt_adam, only: adam_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(mlp_t) :: model
    type(adam_t) :: optimizer
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 2), target(4, 1), prediction(4, 1), residual(4, 1)
    real(dp) :: theta(17), gradient(17), x_bar(4, 2)
    real(dp) :: initial_loss, final_loss
    integer :: iteration

    x = reshape([ &
        -1.0_dp, -1.0_dp, 1.0_dp, 1.0_dp, &
        -1.0_dp,  1.0_dp, -1.0_dp, 1.0_dp], shape(x))
    target = reshape([0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp], shape(target))

    call model%initialize([2, 4, 1], status, hidden_activation=MLP_TANH, &
        initialization_seed=23)
    if (.not. status_ok(status)) error stop "MLP seeded initialization failed"
    theta = model%parameters()
    if (maxval(abs(theta)) <= 1.0e-12_dp .or. &
            maxval(abs(theta(1:8) - theta(1))) <= 1.0e-12_dp) then
        write (error_unit, '(a)') "FAIL [mlp-init] initializer retained symmetry"
        error stop 1
    end if

    call model%predict(x, prediction, status)
    initial_loss = squared_loss(prediction, target)
    call optimizer%initialize(size(theta), status, learning_rate=0.05_dp)
    do iteration = 1, 3000
        residual = prediction - target
        call model%vjp(x, residual, gradient, x_bar, status)
        call optimizer%step(theta, gradient, status)
        call model%set_parameters(theta, status)
        call model%predict(x, prediction, status)
    end do
    final_loss = squared_loss(prediction, target)

    if (.not. status_ok(status) .or. final_loss >= 0.05_dp .or. &
            final_loss >= 0.1_dp*initial_loss) then
        write (error_unit, '(a,2es12.4)') &
            "FAIL [mlp-init] XOR training initial/final=", initial_loss, final_loss
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    pure real(dp) function squared_loss(y, y_target) result(value)
        real(dp), intent(in) :: y(:, :), y_target(:, :)

        value = 0.5_dp*sum((y - y_target)**2)
    end function squared_loss

end program test_mlp_initialization
