program fortml_bench_mlp_pca_initializer
    !! Deterministic PCA-seeded two-layer linear MLP workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t
    use fortml_pca, only: pca_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 512, n_features = 16
    integer, parameter :: n_components = 8, repetitions = 16
    real(dp) :: x(n_samples, n_features), reconstruction(n_samples, n_features)
    real(dp) :: reconstruction_error, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, repetition
    type(pca_t) :: pca
    type(mlp_t) :: model
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_samples
            x(i, j) = sin(0.013_dp*real(i, dp) + 0.071_dp*real(j, dp)) + &
                cos(0.009_dp*real(i*j, dp))
        end do
    end do

    call pca%fit(x, status, n_components=n_components)
    if (.not. status_ok(status)) error stop "MLP PCA benchmark PCA fit failed"
    call model%initialize_from_pca(pca, status)
    if (.not. status_ok(status)) error stop "MLP PCA benchmark initialization failed"
    call model%predict(x, reconstruction, status)
    if (.not. status_ok(status)) error stop "MLP PCA benchmark prediction failed"
    reconstruction_error = sqrt(sum((reconstruction - x)**2) / &
        real(size(x), dp))
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%predict(x, reconstruction, status)
        if (.not. status_ok(status)) error stop "MLP PCA benchmark timing failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "mlp_pca_initializer_predict,", n_samples, ",", n_features, ",", &
        n_components, ",", elapsed, ",", reconstruction_error
end program fortml_bench_mlp_pca_initializer
