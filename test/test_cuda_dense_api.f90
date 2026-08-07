program test_cuda_dense_api
    !! Ordinary-build contract for the resident CUDA dense wrapper.
    use, intrinsic :: iso_fortran_env, only: error_unit, real64
    use fortnum_status, only: fortnum_status_t, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED, FORTNUM_OK
    use fortml_cuda_dense_api, only: cuda_dense_plan_t
    use fortml_mlp, only: MLP_TANH
    implicit none

    type(cuda_dense_plan_t) :: plan
    type(fortnum_status_t) :: status
    real(real64) :: weights(3, 2), bias(2), query_x(4, 3), outputs(4, 2)
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

    call plan%create(weights, bias, MLP_TANH, -1, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "negative device index is rejected", failures)
    call check(.not. plan%fitted(), "invalid creation leaves plan unfitted", failures)

    call plan%create(weights, bias, MLP_TANH, 0, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "ordinary build exposes typed native-unavailable refusal", failures)
    call check(.not. plan%fitted(), "unavailable native plan is unfitted", failures)
    call plan%predict(query_x, outputs, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "prediction refusal is typed", failures)
    call check(all(outputs == -31.0_real64), &
        "prediction refusal preserves output", failures)
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
