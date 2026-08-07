program test_cuda_dense_api
    !! Ordinary-build contract for the resident CUDA dense wrapper.
    use, intrinsic :: iso_c_binding, only: c_int64_t
    use, intrinsic :: iso_fortran_env, only: error_unit, real64
    use fortnum_status, only: fortnum_status_t, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED, FORTNUM_OK
    use fortml_cuda_dense_api, only: cuda_dense_plan_t
    use fortml_mlp, only: MLP_TANH, MLP_SIGMOID, MLP_MISH
    implicit none

    type(cuda_dense_plan_t) :: plan
    type(fortnum_status_t) :: status
    real(real64) :: weights(3, 2), bias(2), query_x(4, 3), outputs(4, 2)
    real(real64) :: query_x_dot(4, 3), weights_dot(3, 2), bias_dot(2)
    real(real64) :: outputs_dot(4, 2)
    real(real64) :: output_bar(4, 2), query_x_bar(4, 3)
    real(real64) :: weights_bar(3, 2), bias_bar(2)
    real(real64) :: target(4, 2), loss
    integer(c_int64_t) :: host_to_device_bytes, device_to_host_bytes
    integer(c_int64_t) :: resident_bytes
    integer :: failures

    failures = 0
    weights = reshape([0.5_real64, -1.0_real64, 0.25_real64, &
        -0.75_real64, 0.4_real64, 1.2_real64], shape(weights))
    bias = [-0.1_real64, 0.2_real64]
    query_x = reshape([ &
        -1.0_real64, 0.0_real64, 0.5_real64, 2.0_real64, &
        1.0_real64, -0.5_real64, 1.5_real64, -2.0_real64, &
        0.25_real64, -1.0_real64, 2.0_real64, 0.5_real64], shape(query_x))
    outputs = -31.0_real64
    query_x_dot = 0.25_real64
    weights_dot = -0.5_real64
    bias_dot = 0.125_real64
    outputs_dot = -17.0_real64
    output_bar = 0.5_real64
    query_x_bar = -23.0_real64
    weights_bar = -29.0_real64
    bias_bar = -31.0_real64
    target = 0.25_real64
    loss = -37.0_real64

    call plan%create(weights, bias, MLP_TANH, -1, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "negative device index is rejected", failures)
    call check(.not. plan%fitted(), "invalid creation leaves plan unfitted", failures)

    call plan%create(weights, bias, MLP_SIGMOID, 0, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "CPU-only sigmoid activation is refused by CUDA control plane", failures)
    call check(.not. plan%fitted(), "sigmoid refusal leaves plan unfitted", failures)
    call plan%create(weights, bias, MLP_MISH, 0, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "CPU-only Mish activation is refused by CUDA control plane", failures)
    call check(.not. plan%fitted(), "Mish refusal leaves plan unfitted", failures)

    call plan%create(weights, bias, MLP_TANH, 0, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "ordinary build exposes typed native-unavailable refusal", failures)
    call check(.not. plan%fitted(), "unavailable native plan is unfitted", failures)
    call plan%predict(query_x, outputs, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "prediction refusal is typed", failures)
    call check(all(outputs == -31.0_real64), &
        "prediction refusal preserves output", failures)
    call plan%jvp(query_x, query_x_dot, weights_dot, bias_dot, outputs, &
        outputs_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "JVP refusal is typed", failures)
    call check(all(outputs == -31.0_real64) .and. &
        all(outputs_dot == -17.0_real64), &
        "JVP refusal preserves outputs", failures)
    call plan%vjp(query_x, output_bar, query_x_bar, weights_bar, bias_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "VJP refusal is typed", failures)
    call check(all(query_x_bar == -23.0_real64) .and. &
        all(weights_bar == -29.0_real64) .and. all(bias_bar == -31.0_real64), &
        "VJP refusal preserves outputs", failures)
    call plan%train_mse(query_x, target, 0.1_real64, loss, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "resident MSE update refusal is typed", failures)
    call check(loss == -37.0_real64, &
        "MSE update refusal preserves loss output", failures)
    call plan%parameters(weights_bar, bias_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "resident parameter snapshot refusal is typed", failures)
    call check(all(weights_bar == -29.0_real64) .and. &
        all(bias_bar == -31.0_real64), &
        "parameter snapshot refusal preserves outputs", failures)
    call plan%transfer_stats(host_to_device_bytes, device_to_host_bytes, &
        resident_bytes, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "transfer statistics refusal is typed", failures)
    call check(host_to_device_bytes == 0_c_int64_t .and. &
        device_to_host_bytes == 0_c_int64_t .and. resident_bytes == 0_c_int64_t, &
        "transfer statistics refusal preserves zero counters", failures)
    call plan%destroy(status)
    call check(status%code == FORTNUM_OK .and. .not. plan%fitted(), &
        "destroy clears plan state", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL CUDA dense API cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS CUDA dense Fortran API refusal oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [CUDA dense API] "//description
        end if
    end subroutine check

end program test_cuda_dense_api
