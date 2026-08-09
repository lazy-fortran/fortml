program test_cuda_dense_resident_training_api
    !! Ordinary-build contract for the resident CUDA optimizer surface.
    use, intrinsic :: iso_fortran_env, only: error_unit, real64
    use fortnum_status, only: fortnum_status_t, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED, FORTNUM_OK
    use fortml_cuda_dense_api, only: cuda_dense_plan_t, CUDA_DENSE_OPT_ADAM
    use fortml_mlp, only: MLP_LINEAR
    implicit none

    type(cuda_dense_plan_t) :: plan
    type(fortnum_status_t) :: status
    real(real64) :: weights(2, 1), bias(1), query_x(3, 2), target(3, 1)
    real(real64) :: loss
    integer :: failures

    failures = 0
    weights = reshape([0.5_real64, -0.25_real64], shape(weights))
    bias = [0.1_real64]
    query_x = reshape([-1.0_real64, 0.0_real64, 2.0_real64, &
        0.5_real64, -0.75_real64, 1.25_real64], shape(query_x))
    target = reshape([0.2_real64, -0.3_real64, 0.6_real64], shape(target))
    loss = -19.0_real64

    call plan%create(weights, bias, MLP_LINEAR, -1, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "negative device index is rejected", failures)
    call plan%create(weights, bias, MLP_LINEAR, 0, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "ordinary build exposes typed native-unavailable refusal", failures)
    call check(.not. plan%fitted(), "unavailable plan is unfitted", failures)

    call plan%upload_batch(query_x, target, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "resident batch upload refusal is typed", failures)
    call plan%train_resident_mse(0.01_real64, 0.9_real64, 0.999_real64, &
        1.0e-8_real64, 0.0_real64, CUDA_DENSE_OPT_ADAM, loss, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "resident Adam refusal is typed", failures)
    call check(loss == -19.0_real64, &
        "resident optimizer refusal preserves loss", failures)
    call plan%destroy(status)
    call check(status%code == FORTNUM_OK .and. .not. plan%fitted(), &
        "destroy is idempotent for unavailable plan", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL CUDA dense resident training API cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS CUDA resident training Fortran API refusal oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') &
                "FAIL [CUDA resident training API] "//description
        end if
    end subroutine check

end program test_cuda_dense_resident_training_api
