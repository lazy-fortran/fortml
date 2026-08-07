program fortml_bench_mlp_calibrated_classifier
    !! Complete-array calibrated neural classifier workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp_calibrated_classifier, only: &
        mlp_calibrated_classifier_t, mlp_calibrated_classifier_options_t, &
        MLP_CALIBRATION_TEMPERATURE
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 64, prediction_repetitions = 128
    real(dp) :: x(n_samples, 2), probabilities(n_samples, 2)
    integer :: labels(n_samples), predicted(n_samples)
    type(mlp_calibrated_classifier_t) :: model
    type(mlp_calibrated_classifier_options_t) :: options
    type(fortnum_status_t) :: status
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, repetition

    call get_environment_variable("FORTML_BENCH_MLP_CALIBRATED_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_MLP_CALIBRATED_ORACLE is required"
    end if
    call make_fixture(x, labels)
    options = mlp_calibrated_classifier_options_t()
    options%classifier%max_epochs = 120
    options%classifier%learning_rate = 0.04_dp
    options%classifier%initialization_seed = 23
    options%classifier%tolerance = 1.0e-8_dp
    options%calibration%method = MLP_CALIBRATION_TEMPERATURE
    options%calibration%max_iterations = 300
    options%calibration%tolerance = 1.0e-10_dp
    options%calibration%l2 = 1.0e-6_dp

    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, options=options)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "calibrated MLP benchmark fit failed"
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    if (.not. status_ok(status)) error stop "calibrated MLP benchmark prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/ &
        real(prediction_repetitions, dp)

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "method,quantity,row,column,value"
    do i = 1, n_samples
        write (unit, '(a,i0,a,es26.17e3)') "temperature,label,", i, ",1,", real(labels(i), dp)
        write (unit, '(a,i0,a,es26.17e3)') "temperature,prediction,", i, ",1,", real(predicted(i), dp)
        write (unit, '(a,i0,a,es26.17e3)') "temperature,probability,", i, ",1,", probabilities(i, 1)
        write (unit, '(a,i0,a,es26.17e3)') "temperature,probability,", i, ",2,", probabilities(i, 2)
    end do
    close (unit)
    write (*, '(a,es24.16)') "mlp_calibrated_classifier_fit,temperature,", fit_seconds
    write (*, '(a,es24.16)') "mlp_calibrated_classifier_predict,temperature,", predict_seconds

contains

    subroutine make_fixture(features, target)
        real(dp), intent(out) :: features(:, :)
        integer, intent(out) :: target(:)
        integer :: i
        real(dp) :: phase

        do i = 1, size(target)
            phase = real(i, dp)
            features(i, 1) = sin(0.19_dp*phase) + 0.02_dp*phase
            features(i, 2) = cos(0.13_dp*phase) - 0.01_dp*phase
            if (features(i, 1) + 0.7_dp*features(i, 2) > 0.0_dp) then
                target(i) = 42
            else
                target(i) = -3
            end if
        end do
    end subroutine make_fixture

end program fortml_bench_mlp_calibrated_classifier
