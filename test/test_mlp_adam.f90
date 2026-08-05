program test_mlp_adam
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp, only: mlp_t
    use fortopt_adam, only: adam_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(mlp_t) :: model
    type(adam_t) :: optimizer
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 1), target(4, 1), prediction(4, 1), residual(4, 1)
    real(dp) :: theta(7), gradient(7), x_bar(4, 1)
    real(dp) :: initial_loss, final_loss
    integer :: iteration

    x(:, 1) = [-1.0_dp, -0.25_dp, 0.5_dp, 1.25_dp]
    target = 0.75_dp
    theta = 0.0_dp
    call model%initialize([1, 2, 1], status)
    call model%set_parameters(theta, status)
    call model%predict(x, prediction, status)
    initial_loss = squared_loss(prediction, target)
    call optimizer%initialize(size(theta), status, learning_rate=0.05_dp)

    do iteration = 1, 20
        residual = prediction - target
        call model%vjp(x, residual, gradient, x_bar, status)
        call optimizer%step(theta, gradient, status)
        call model%set_parameters(theta, status)
        call model%predict(x, prediction, status)
    end do
    final_loss = squared_loss(prediction, target)

    if (.not. status_ok(status) .or. final_loss >= initial_loss .or. &
        final_loss > 0.25_dp*initial_loss) then
        write (error_unit, '(a,2es12.4)') &
            "FAIL [mlp-adam] loss initial/final=", initial_loss, final_loss
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    pure real(dp) function squared_loss(y, y_target) result(value)
        real(dp), intent(in) :: y(:, :), y_target(:, :)

        value = 0.5_dp*sum((y - y_target)**2)
    end function squared_loss

end program test_mlp_adam
