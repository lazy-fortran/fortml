program fortml_bench_cuda_dense_training
    !! Release workload for the typed resident CUDA dense training contract.
    !! Native CUDA timing/oracle lives in test/run_cuda_dense_resident_training.sh;
    !! this ordinary-build workload records the explicit refusal status.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_cuda_dense_api, only: cuda_dense_plan_t, CUDA_DENSE_OPT_ADAM
    use fortml_mlp, only: MLP_LINEAR
    implicit none

    type(cuda_dense_plan_t) :: plan
    type(fortnum_status_t) :: status
    real(real64) :: weights(2, 1), bias(1), query_x(4, 2), target(4, 1)
    real(real64) :: loss

    weights = reshape([0.5_real64, -0.25_real64], shape(weights))
    bias = [0.1_real64]
    query_x = reshape([-1.0_real64, 0.0_real64, 2.0_real64, 0.5_real64, &
        0.75_real64, -0.5_real64, 1.25_real64, -1.5_real64], shape(query_x))
    target = reshape([0.2_real64, -0.3_real64, 0.6_real64, 0.4_real64], &
        shape(target))
    loss = -1.0_real64
    call plan%create(weights, bias, MLP_LINEAR, 0, status)
    write (*, '(a,",",i0)') "create_status", status%code
    if (status_ok(status)) then
        call plan%upload_batch(query_x, target, status)
        write (*, '(a,",",i0)') "upload_status", status%code
        call plan%train_resident_mse(0.01_real64, 0.9_real64, 0.999_real64, &
            1.0e-8_real64, 0.0_real64, CUDA_DENSE_OPT_ADAM, loss, status)
        write (*, '(a,",",i0,",",es24.16)') "adam_status_loss", status%code, loss
    end if
    call plan%destroy(status)
    write (*, '(a,",",i0)') "destroy_status", status%code
end program fortml_bench_cuda_dense_training
