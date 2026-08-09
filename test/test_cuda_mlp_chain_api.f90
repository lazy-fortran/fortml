program test_cuda_mlp_chain_api
    !! Ordinary-build contract for the resident CUDA MLP-chain wrapper.
    use, intrinsic :: iso_c_binding, only: c_int64_t
    use, intrinsic :: iso_fortran_env, only: error_unit, real64
    use fortnum_status, only: fortnum_status_t, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED, FORTNUM_OK
    use fortml_cuda_mlp_chain_api, only: cuda_mlp_chain_plan_t
    implicit none

    type(cuda_mlp_chain_plan_t) :: plan
    type(fortnum_status_t) :: status
    integer :: layer_sizes(4), activations(3), failures
    real(real64) :: weights(6 + 6 + 2), biases(3 + 2 + 1)
    real(real64) :: x(5, 2), y(5, 1), x_dot(5, 2), y_dot(5, 1)
    real(real64) :: weights_dot(size(weights)), biases_dot(size(biases))
    real(real64) :: output_bar(5, 1), x_bar(5, 2)
    real(real64) :: weights_bar(size(weights)), biases_bar(size(biases))
    integer(c_int64_t) :: h2d, d2h, resident

    failures = 0
    layer_sizes = [2, 3, 2, 1]
    activations = [2, 3, 1]
    weights = 0.1_real64
    biases = -0.2_real64
    x = 0.3_real64
    y = -11.0_real64
    x_dot = 0.4_real64
    y_dot = -13.0_real64
    weights_dot = 0.5_real64
    biases_dot = 0.6_real64
    output_bar = 0.7_real64
    x_bar = -17.0_real64
    weights_bar = -19.0_real64
    biases_bar = -23.0_real64

    call plan%create(layer_sizes, activations, weights, biases, -1, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "negative device index is rejected", failures)
    call check(.not. plan%fitted(), "invalid creation leaves plan unfitted", failures)

    call plan%create(layer_sizes, activations, weights, biases, 0, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "ordinary build exposes typed native-unavailable refusal", failures)
    call check(.not. plan%fitted(), "unavailable native chain is unfitted", failures)
    call plan%predict(x, y, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "prediction refusal is typed", failures)
    call check(all(y == -11.0_real64), "prediction preserves sentinel", failures)
    call plan%jvp(x, x_dot, weights_dot, biases_dot, y, y_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "JVP refusal is typed", failures)
    call check(all(y == -11.0_real64) .and. all(y_dot == -13.0_real64), &
        "JVP preserves sentinels", failures)
    call plan%vjp(x, output_bar, x_bar, weights_bar, biases_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "VJP refusal is typed", failures)
    call check(all(x_bar == -17.0_real64) .and. all(weights_bar == -19.0_real64) &
        .and. all(biases_bar == -23.0_real64), "VJP preserves sentinels", failures)
    call plan%transfer_stats(h2d, d2h, resident, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "transfer statistics refusal is typed", failures)
    call check(h2d == 0_c_int64_t .and. d2h == 0_c_int64_t .and. &
        resident == 0_c_int64_t, "statistics preserve zero sentinels", failures)
    call plan%destroy(status)
    call check(status%code == FORTNUM_OK .and. .not. plan%fitted(), &
        "destroy clears chain state", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL CUDA MLP chain API cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS CUDA MLP chain Fortran API refusal oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [CUDA MLP chain] "//description
        end if
    end subroutine check

end program test_cuda_mlp_chain_api
