program fortml_bench_gp_reference
    !! Time FortML's exact GP against the same model in scikit-learn and
    !! GPyTorch, and print its values so accuracy is checked beside speed.
    !!
    !! The existing GP benchmarks in this tree measure FortML against itself
    !! across sizes, which catches a regression but cannot answer whether the
    !! implementation is competitive. This one is built to be paired with a
    !! Python counterpart that constructs the identical model from the same
    !! closed forms.
    !!
    !! The model is **pinned, not fitted**. Learning hyperparameters on each
    !! side would compare three optimizers and report the difference as a
    !! modelling cost. What is compared is the linear algebra every GP does:
    !! the Cholesky factorization at fit time, and the cross-covariance, solve
    !! and reduction at predict time.
    !!
    !! Fit and predict are timed separately because they scale differently --
    !! fit is cubic in the training size and predict is linear in the query
    !! count -- and a single number would hide which one a change affected.
    !!
    !! Usage:
    !!
    !!     fo exec fortml_bench_gp_reference N_TRAIN N_QUERY DIMENSION

    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    implicit none

    integer, parameter :: dp = real64
    real(dp), parameter :: LENGTHSCALE = 0.8_dp
    real(dp), parameter :: SIGNAL_VARIANCE = 1.4_dp
    real(dp), parameter :: NOISE_VARIANCE = 0.04_dp

    integer :: n_train, n_query, dimension
    character(len=32) :: argument
    integer :: length, k, j
    real(dp), allocatable :: x(:, :), y(:, :), q(:, :)
    real(dp), allocatable :: mean(:, :), variance(:)
    type(kernel_t) :: kernel
    type(gp_regression_t) :: model
    type(fortnum_status_t) :: status
    real(dp) :: t0, t1, fit_seconds, predict_seconds

    n_train = 400
    n_query = 4000
    dimension = 8
    if (command_argument_count() >= 1) then
        call get_command_argument(1, argument, length=length)
        read (argument(:length), *) n_train
    end if
    if (command_argument_count() >= 2) then
        call get_command_argument(2, argument, length=length)
        read (argument(:length), *) n_query
    end if
    if (command_argument_count() >= 3) then
        call get_command_argument(3, argument, length=length)
        read (argument(:length), *) dimension
    end if

    ! Stated closed forms, so the Python counterpart reproduces the identical
    ! inputs without arrays being shipped between them. A transcription error
    ! then shows up as a value mismatch instead of hiding.
    allocate (x(n_train, dimension), y(n_train, 1), q(n_query, dimension))
    do k = 1, n_train
        do j = 1, dimension
            x(k, j) = sin(0.29_dp*real(k, dp) + 0.41_dp*real(j, dp))
        end do
        y(k, 1) = sum(cos(1.1_dp*x(k, :))) + 0.1_dp*real(k, dp)/real(n_train, dp)
    end do
    do k = 1, n_query
        do j = 1, dimension
            q(k, j) = cos(0.17_dp*real(k, dp) + 0.23_dp*real(j, dp))
        end do
    end do

    kernel = make_rbf_kernel(dimension, SIGNAL_VARIANCE, LENGTHSCALE, status)
    if (status%code /= FORTNUM_OK) then
        print *, "kernel failed: ", trim(status%msg)
        error stop 1
    end if

    t0 = wall_seconds()
    call model%fit(x, y, kernel, NOISE_VARIANCE, status, jitter=0.0_dp)
    t1 = wall_seconds()
    fit_seconds = t1 - t0
    if (status%code /= FORTNUM_OK) then
        print *, "fit failed: ", trim(status%msg)
        error stop 1
    end if

    allocate (mean(n_query, 1), variance(n_query))
    t0 = wall_seconds()
    call model%predict(q, mean, variance, status)
    t1 = wall_seconds()
    predict_seconds = t1 - t0
    if (status%code /= FORTNUM_OK) then
        print *, "predict failed: ", trim(status%msg)
        error stop 1
    end if

    print *, "CONFIG ", n_train, n_query, dimension
    print *, "TIME fit ", fit_seconds
    print *, "TIME predict ", predict_seconds
    ! Sums rather than whole arrays: enough to catch a divergence, small
    ! enough to compare across a process boundary without shipping data.
    print *, "VALUE mean_sum ", sum(mean(:, 1))
    print *, "VALUE variance_sum ", sum(variance)
    print *, "VALUE mean_first ", mean(1, 1)
    print *, "VALUE variance_first ", variance(1)

contains

    !! Wall-clock seconds.
    !!
    !! `cpu_time` sums processor time across threads, so a multithreaded BLAS
    !! reports several times the elapsed time and a comparison against a
    !! Python reference timed with `perf_counter` becomes meaningless -- and
    !! meaningless in the direction that flatters whichever side is
    !! single-threaded. `system_clock` measures elapsed time on both.
    function wall_seconds() result(seconds)
        use, intrinsic :: iso_fortran_env, only: int64
        real(real64) :: seconds
        integer(int64) :: ticks, rate

        call system_clock(ticks, rate)
        seconds = real(ticks, real64)/real(rate, real64)
    end function wall_seconds

end program fortml_bench_gp_reference
