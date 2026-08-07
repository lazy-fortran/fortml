program fortml_bench_pca
    !! Small deterministic dense-PCA workload for the release benchmark lane.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_pca, only: pca_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 512, n_features = 16
    integer, parameter :: n_components = 8, repetitions = 4
    real(dp) :: x(n_samples, n_features), z(n_samples, n_components)
    real(dp), allocatable :: components(:, :)
    real(dp) :: orthogonality_error, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, repetition
    type(pca_t) :: model
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_samples
            x(i, j) = sin(0.013_dp*real(i, dp) + 0.071_dp*real(j, dp)) + &
                cos(0.009_dp*real(i*j, dp))
        end do
    end do

    call model%fit(x, status, n_components=n_components)
    if (.not. status_ok(status)) error stop "PCA benchmark fit failed"
    call model%transform(x, z, status)
    if (.not. status_ok(status)) error stop "PCA benchmark transform failed"
    components = model%components()
    orthogonality_error = maxval(abs(matmul(components, transpose(components)) - &
        identity_matrix(n_components)))
    if (orthogonality_error > 1.0e-10_dp) error stop "PCA orthogonality oracle failed"

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call model%fit(x, status, n_components=n_components)
        if (.not. status_ok(status)) error stop "PCA benchmark timing fit failed"
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') &
        "pca_fit,", n_samples, ",", n_features, ",", n_components, ",", &
        elapsed, ",", orthogonality_error

contains

    function identity_matrix(n) result(identity)
        integer, intent(in) :: n
        real(dp) :: identity(n, n)
        integer :: index

        identity = 0.0_dp
        do index = 1, n
            identity(index, index) = 1.0_dp
        end do
    end function identity_matrix

end program fortml_bench_pca
