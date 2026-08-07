program fortml_bench_mlp_grouped_training
    !! Correctness-gated grouped MLP objective workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_grouped_training, only: mlp_parameter_group_t, &
        mlp_grouped_training_objective_t
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: n_samples = 128, repetitions = 128
    real(dp) :: x(n_samples, 1), target(n_samples, 1)
    real(dp) :: parameters(4), direction(4), gradient(4), product(4)
    real(dp) :: value, tangent, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, repetition
    type(mlp_t), target :: model
    type(mlp_parameter_group_t) :: groups(2)
    type(mlp_grouped_training_objective_t) :: objective
    type(fortnum_status_t) :: status

    do i = 1, n_samples
        x(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(n_samples - 1, dp)
        target(i, 1) = 0.8_dp*x(i, 1) + 0.2_dp
    end do
    call model%initialize([1, 1], status, output_activation=MLP_LINEAR)
    if (.not. status_ok(status)) error stop "grouped benchmark model initialization failed"
    call model%set_parameters([0.4_dp, -0.3_dp], status)
    if (.not. status_ok(status)) error stop "grouped benchmark parameter setup failed"
    call groups(1)%initialize("weight", 1, 1, -1.0_dp, status)
    if (.not. status_ok(status)) error stop "grouped benchmark weight group failed"
    call groups(2)%initialize("bias", 2, 2, -2.0_dp, status)
    if (.not. status_ok(status)) error stop "grouped benchmark bias group failed"
    call objective%initialize(model, x, target, groups, status)
    if (.not. status_ok(status)) error stop "grouped benchmark objective setup failed"
    parameters = objective%parameters()
    direction = [0.17_dp, -0.23_dp, 0.31_dp, -0.27_dp]
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop "grouped benchmark value product failed"
    call objective%hvp(parameters, direction, product, status)
    if (.not. status_ok(status)) error stop "grouped benchmark HVP failed"
    call objective%jvp(parameters, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "grouped benchmark JVP failed"

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call objective%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) error stop "grouped benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_grouped_value_gradient_seconds,", elapsed
    write (*, '(a,es24.16)') "mlp_grouped_value,", value
    write (*, '(a,es24.16)') "mlp_grouped_gradient_norm,", sqrt(dot_product(gradient, gradient))
    write (*, '(a,es24.16)') "mlp_grouped_hvp_norm,", sqrt(dot_product(product, product))
    write (*, '(a,es24.16)') "mlp_grouped_jvp,", tangent

    call objective%initialize(model, x, target, groups, status, device_kind=FORTML_DEVICE_CUDA)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "grouped benchmark CUDA contract changed unexpectedly"
    end if
    write (*, '(a)') "mlp_grouped_cuda,unavailable"
end program fortml_bench_mlp_grouped_training
