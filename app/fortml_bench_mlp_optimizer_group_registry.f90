program fortml_bench_mlp_optimizer_group_registry
    !! Release workload for named optimizer-group checkpoint identity.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_training, only: mlp_training_options_t, &
        mlp_training_checkpoint_t, mlp_optimizer_group_t, mlp_train, MLP_OPTIMIZER_SGD
    use fortml_mlp_checkpoint, only: mlp_checkpoint_save, mlp_checkpoint_load
    use fortml_mlp_optimizer_group_hypergradient, only: &
        mlp_optimizer_group_hypergradient_objective_t, &
        mlp_optimizer_group_hypergradient_options_t
    implicit none

    type(mlp_t), target :: model
    type(mlp_training_options_t) :: options
    type(mlp_training_checkpoint_t) :: checkpoint, loaded
    type(mlp_optimizer_group_t) :: group
    type(mlp_optimizer_group_hypergradient_options_t) :: cuda_options
    type(mlp_optimizer_group_hypergradient_objective_t) :: cuda_objective
    type(fortnum_status_t) :: status
    real(dp) :: x(2, 1), target(2, 1)
    integer :: layers(2), cuda_status
    character(*), parameter :: path = "fortml_mlp_optimizer_group_registry_bench.txt"

    layers = [1, 1]
    x(:, 1) = [1.0_dp, 2.0_dp]
    target(:, 1) = [2.0_dp, 4.0_dp]
    call model%initialize(layers, status, output_activation=MLP_LINEAR)
    if (.not. status_ok(status)) error stop "registry benchmark model setup failed"
    call group%initialize("bias", 2, 2, 0.5_dp, status)
    if (.not. status_ok(status)) error stop "registry benchmark group setup failed"
    allocate(options%optimizer_groups(1))
    options%optimizer_groups(1) = group
    options%max_epochs = 1
    options%learning_rate = 0.05_dp
    options%optimizer = MLP_OPTIMIZER_SGD
    options%tolerance = 0.0_dp
    options%restore_best = .false.
    call mlp_train(model, x, target, status, options, checkpoint=checkpoint)
    if (.not. status_ok(status) .or. .not. checkpoint%valid()) then
        error stop "registry benchmark training failed"
    end if
    call mlp_checkpoint_save(checkpoint, path, status)
    if (.not. status_ok(status)) error stop "registry benchmark save failed"
    call mlp_checkpoint_load(loaded, path, status)
    if (.not. status_ok(status) .or. .not. loaded%valid()) then
        error stop "registry benchmark load failed"
    end if
    if (trim(loaded%optimizer_group_name(1)) /= "bias") then
        error stop "registry benchmark name replay failed"
    end if

    call group%initialize("renamed", 2, 2, 0.5_dp, status)
    options%optimizer_groups(1) = group
    options%max_epochs = 2
    call mlp_train(model, x, target, status, options, checkpoint=loaded)
    if (status_ok(status)) error stop "registry benchmark accepted name drift"
    write (*, '(a)') "mlp_optimizer_group_registry,roundtrip_name,bias"
    write (*, '(a,i0)') "mlp_optimizer_group_registry,name_drift_status,", status%code

    cuda_options%device_kind = FORTML_DEVICE_CUDA
    allocate(cuda_options%groups(1))
    cuda_options%groups(1) = group
    call cuda_objective%initialize(model, x, target, x, target, cuda_options, status)
    cuda_status = status%code
    if (cuda_status /= FORTNUM_NOT_IMPLEMENTED) then
        error stop "registry benchmark CUDA boundary changed"
    end if
    write (*, '(a,i0)') "mlp_optimizer_group_registry,cuda_status,", cuda_status

    open (unit=91, file=path, status="old", action="read")
    close (91, status="delete")
end program fortml_bench_mlp_optimizer_group_registry
