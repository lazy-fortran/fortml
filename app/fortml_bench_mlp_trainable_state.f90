program fortml_bench_mlp_trainable_state
    !! Release application for the named MLP freeze-state contract.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    implicit none

    type(mlp_t) :: model
    type(fortnum_status_t) :: status
    real(dp), parameter :: x(2, 1) = reshape([0.25_dp, -0.75_dp], [2, 1])
    real(dp), parameter :: theta(7) = [0.4_dp, -0.2_dp, 0.1_dp, -0.3_dp, &
        0.7_dp, -0.6_dp, 0.05_dp]
    real(dp) :: y(2, 1), y_frozen(2, 1), dy(2, 1), x_bar(2, 1)
    real(dp) :: gradient(7), frozen_gradient(7), dtheta(7), u(2, 1), dx(2, 1)
    integer :: status_code, trainable_count
    real(dp) :: frozen_gradient_max, live_gradient_error, frozen_jvp_max
    real(dp) :: unfrozen_jvp_max, prediction_change

    call model%initialize([1, 2, 1], status, hidden_activation=MLP_TANH, &
        output_activation=MLP_LINEAR)
    if (status%code /= FORTNUM_OK) error stop 1
    call model%set_parameters(theta, status)
    if (status%code /= FORTNUM_OK) error stop 1
    u = 1.0_dp
    dx = 0.0_dp
    call model%predict(x, y, status)
    if (status%code /= FORTNUM_OK) error stop 1
    call model%vjp(x, u, gradient, x_bar, status)
    if (status%code /= FORTNUM_OK) error stop 1

    call model%set_trainable("layer_1.weight", .false., status)
    if (status%code /= FORTNUM_OK) error stop 1
    trainable_count = model%trainable_parameter_count()
    call model%predict(x, y_frozen, status)
    if (status%code /= FORTNUM_OK) error stop 1
    call model%vjp(x, u, frozen_gradient, x_bar, status)
    if (status%code /= FORTNUM_OK) error stop 1
    dtheta = 0.0_dp
    dtheta(1) = 1.0_dp
    call model%jvp(x, dtheta, dx, y_frozen, dy, status)
    if (status%code /= FORTNUM_OK) error stop 1
    frozen_gradient_max = maxval(abs(frozen_gradient(1:2)))
    live_gradient_error = maxval(abs(frozen_gradient(3:7) - gradient(3:7)))
    frozen_jvp_max = maxval(abs(dy))
    prediction_change = maxval(abs(y_frozen - y))

    call model%set_trainable("layer_1.weight", .true., status)
    if (status%code /= FORTNUM_OK) error stop 1
    call model%jvp(x, dtheta, dx, y_frozen, dy, status)
    if (status%code /= FORTNUM_OK) error stop 1
    unfrozen_jvp_max = maxval(abs(dy))
    status_code = status%code

    write (*, '(a,i0,6(",",es24.16))') "trainable_state", trainable_count, &
        frozen_gradient_max, live_gradient_error, frozen_jvp_max, &
        unfrozen_jvp_max, prediction_change, real(status_code, dp)
end program fortml_bench_mlp_trainable_state
