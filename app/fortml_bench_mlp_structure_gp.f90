program fortml_bench_mlp_structure_gp
    !! Structure-aware finite-feature GP initializer workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    use fortml_mlp_structure_gp, only: mlp_structure_gp_initializer_t, &
        mlp_structure_gp_metadata_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 256, n_features = 8
    integer, parameter :: n_hidden = 16, n_outputs = 2, repetitions = 8
    real(dp) :: x(n_samples, n_features), target(n_samples, n_outputs)
    real(dp) :: prediction(n_samples, n_outputs)
    real(dp), allocatable :: before(:), after(:)
    real(dp) :: mse, fit_seconds, predict_seconds, hidden_delta
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, repetition, hidden_count
    type(mlp_t) :: model
    type(mlp_structure_gp_initializer_t) :: initializer
    type(mlp_structure_gp_metadata_t) :: metadata
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_samples
            x(i, j) = sin(0.017_dp*real(i, dp) + 0.053_dp*real(j, dp)) + &
                cos(0.011_dp*real(i*j, dp))
        end do
    end do
    do j = 1, n_outputs
        do i = 1, n_samples
            target(i, j) = 0.4_dp*sin(0.013_dp*real(i*j, dp)) + &
                0.2_dp*cos(0.019_dp*real(i + j, dp))
        end do
    end do
    call model%initialize([n_features, n_hidden, n_outputs], status, &
        hidden_activation=MLP_TANH, output_activation=MLP_LINEAR, &
        initialization_seed=29)
    if (.not. status_ok(status)) error stop "structure GP benchmark model failed"
    before = model%parameters()
    hidden_count = model%parameter_count() - n_hidden*n_outputs - n_outputs
    call system_clock(clock_start, clock_rate)
    call initializer%fit_apply(model, x, target, status, 0.1_dp)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "structure GP benchmark fit failed"
    fit_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)

    after = model%parameters()
    hidden_delta = maxval(abs(after(1:hidden_count) - before(1:hidden_count)))
    call model%predict(x, prediction, status)
    if (.not. status_ok(status)) error stop "structure GP benchmark prediction failed"
    mse = sum((prediction - target)**2)/real(size(target), dp)
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%predict(x, prediction, status)
        if (.not. status_ok(status)) error stop "structure GP benchmark timing failed"
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    metadata = initializer%metadata()
    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
        "mlp_structure_gp_predict,", n_samples, ",", n_features, ",", &
        n_hidden, ",", n_outputs, ",", fit_seconds, ",", predict_seconds, ",", &
        mse, ",", hidden_delta, ",", metadata%hidden_scale
end program fortml_bench_mlp_structure_gp
