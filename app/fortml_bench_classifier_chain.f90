program fortml_bench_classifier_chain
    !! Correctness-gated classifier-chain logistic workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_classifier_chain, only: classifier_chain_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 192, n_features = 4, n_outputs = 3
    integer, parameter :: prediction_repetitions = 64
    integer, parameter :: clone_repetitions = 64
    integer, parameter :: parameter_count = n_outputs*n_features + &
        n_outputs*(n_outputs+1)/2
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, n_outputs)
    real(dp) :: probabilities_bar(n_samples, n_outputs), x_dot(n_samples, n_features)
    real(dp) :: theta_dot(parameter_count), theta_hvp(parameter_count)
    real(dp) :: x_hvp(n_samples, n_features)
    integer :: indicators(n_samples, n_outputs), predicted(n_samples, n_outputs)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds, hvp_seconds, clone_seconds
    real(dp) :: clone_probabilities(n_samples, n_outputs)
    real(dp), allocatable :: parameters(:)
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition
    type(fortnum_status_t) :: status
    type(classifier_chain_t) :: model, clone

    call get_environment_variable("FORTML_BENCH_CLASSIFIER_CHAIN_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_CLASSIFIER_CHAIN_ORACLE is required"
    end if
    call make_fixture(x, indicators)
    call make_directions(x_dot, probabilities_bar, theta_dot)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, indicators, status, l2=5.0e-2_dp, max_iterations=1000, &
        tolerance=1.0e-7_dp)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "classifier chain benchmark fit failed"
    parameters = model%parameters()
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    if (.not. status_ok(status)) error stop "classifier chain benchmark prediction failed"
    call model%clone(clone, status)
    if (.not. status_ok(status)) error stop "classifier chain benchmark clone failed"
    call clone%predict_proba(x, clone_probabilities, status)
    if (.not. status_ok(status)) error stop "classifier chain benchmark clone prediction failed"
    parameters = model%parameters()
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba_hvp(x, probabilities_bar, theta_dot, x_dot, &
            theta_hvp, x_hvp, status)
    end do
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "classifier chain benchmark HVP failed"
    hvp_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)
    call system_clock(clock_start, clock_rate)
    do repetition = 1, clone_repetitions
        call model%clone(clone, status)
    end do
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "classifier chain benchmark clone loop failed"
    clone_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(clone_repetitions, dp)
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "classifier_chain_fit,", &
        n_samples, ",", n_features, ",", n_outputs, ",", fit_seconds
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "classifier_chain_predict,", &
        n_samples, ",", n_features, ",", n_outputs, ",", predict_seconds
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "classifier_chain_hvp,", &
        n_samples, ",", n_features, ",", n_outputs, ",", hvp_seconds
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') "classifier_chain_clone,", &
        n_samples, ",", n_features, ",", n_outputs, ",", clone_seconds

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, size(parameters)
        write (unit, '(a,i0,a,es24.16)') "parameter,", i, ",0,", parameters(i)
    end do
    do i = 1, size(theta_hvp)
        write (unit, '(a,i0,a,es24.16)') "theta_hvp,", i, ",0,", theta_hvp(i)
    end do
    do i = 1, n_samples
        do j = 1, n_features
            write (unit, '(a,i0,a,i0,a,es24.16)') "x_hvp,", i, ",", j, ",", &
                x_hvp(i, j)
        end do
    end do
    do i = 1, n_samples
        do j = 1, n_outputs
            write (unit, '(a,i0,a,i0,a,i0)') "label,", i, ",", j, ",", indicators(i, j)
            write (unit, '(a,i0,a,i0,a,i0)') "prediction,", i, ",", j, ",", predicted(i, j)
            write (unit, '(a,i0,a,i0,a,es24.16)') "probability,", i, ",", j, ",", &
                probabilities(i, j)
            write (unit, '(a,i0,a,i0,a,es24.16)') "clone_probability,", i, ",", j, ",", &
                clone_probabilities(i, j)
        end do
    end do
    do i = 1, size(parameters)
        write (unit, '(a,i0,a,es24.16)') "parameter,", i, ",0,", parameters(i)
    end do
    close (unit)

contains

    subroutine make_fixture(x, indicators)
        real(dp), intent(out) :: x(:, :)
        integer, intent(out) :: indicators(:, :)
        real(dp) :: phase, score
        integer :: i, j

        do i = 1, size(x, 1)
            phase = real(i, dp)
            do j = 1, size(x, 2)
                x(i, j) = sin(0.021_dp*phase + 0.083_dp*real(j, dp)) + &
                    0.15_dp*cos(0.011_dp*phase*real(j, dp))
            end do
            score = 0.8_dp*x(i, 1) - 0.45_dp*x(i, 2) + 0.15_dp*sin(0.13_dp*phase)
            indicators(i, 1) = merge(1, 0, score > 0.0_dp)
            score = -0.35_dp*x(i, 1) + 0.7_dp*x(i, 3) - &
                0.1_dp*cos(0.09_dp*phase + 0.3_dp)
            indicators(i, 2) = merge(1, 0, score > 0.0_dp)
            score = 0.3_dp*x(i, 2) + 0.55_dp*x(i, 4) + &
                0.2_dp*sin(0.07_dp*phase + 0.5_dp)
            indicators(i, 3) = merge(1, 0, score > 0.0_dp)
        end do
    end subroutine make_fixture

    subroutine make_directions(x_dot, probabilities_bar, theta_dot)
        real(dp), intent(out) :: x_dot(:, :), probabilities_bar(:, :), theta_dot(:)
        integer :: i, j

        do i = 1, size(x_dot, 1)
            do j = 1, size(x_dot, 2)
                x_dot(i, j) = 0.003_dp*cos(0.017_dp*real(i*j, dp))
            end do
            do j = 1, size(probabilities_bar, 2)
                probabilities_bar(i, j) = 0.2_dp*sin(0.03_dp*real(i, dp) + &
                    0.4_dp*real(j, dp))
            end do
        end do
        do i = 1, size(theta_dot)
            theta_dot(i) = 0.01_dp*sin(0.13_dp*real(i, dp))
        end do
    end subroutine make_directions

end program fortml_bench_classifier_chain
