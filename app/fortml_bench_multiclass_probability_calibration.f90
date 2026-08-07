program fortml_bench_multiclass_probability_calibration
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_probability_calibration, only: &
        multiclass_probability_calibrator_t, probability_calibration_options_t, &
        CALIBRATION_TEMPERATURE
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 192, n_classes = 3
    integer, parameter :: fit_repetitions = 8, predict_repetitions = 128
    type(multiclass_probability_calibrator_t) :: model
    type(probability_calibration_options_t) :: options
    type(fortnum_status_t) :: status
    real(real64) :: scores(n_samples, n_classes), probabilities(n_samples, n_classes)
    integer :: labels(n_samples), predictions(n_samples), classes(n_classes)
    real(real64), allocatable :: fitted_parameters(:)
    integer :: i, j, repetition, environment_status, io_status, unit
    real(real64) :: started, finished, fit_seconds, predict_seconds
    character(len=1024) :: oracle_path

    call fixture(scores, labels)
    options = probability_calibration_options_t(method=CALIBRATION_TEMPERATURE, &
        max_iterations=500, tolerance=1.0e-11_real64, damping=1.0_real64, &
        l2=0.05_real64)
    call cpu_time(started)
    do repetition = 1, fit_repetitions
        call model%fit(scores, labels, status, options=options)
        if (.not. status_ok(status)) error stop "multiclass calibration fit failed"
    end do
    call cpu_time(finished)
    fit_seconds = (finished-started)/real(fit_repetitions, real64)
    call cpu_time(started)
    do repetition = 1, predict_repetitions
        call model%predict_proba(scores, probabilities, status)
        if (.not. status_ok(status)) error stop "multiclass calibration prediction failed"
    end do
    call cpu_time(finished)
    predict_seconds = (finished-started)/real(predict_repetitions, real64)
    call model%predict_proba(scores, probabilities, status)
    call model%predict(scores, predictions, status)
    classes = model%classes()
    fitted_parameters = model%parameters()
    if (.not. status_ok(status)) error stop "multiclass calibration output failed"
    write (*, '(a,",",a,",",es24.16)') &
        "multiclass_probability_calibration_fit", "temperature", fit_seconds
    write (*, '(a,",",a,",",es24.16)') &
        "multiclass_probability_calibration_predict", "temperature", predict_seconds

    call get_environment_variable("FORTML_BENCH_MULTICLASS_CALIBRATION_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) stop
    open (newunit=unit, file=trim(oracle_path), status="replace", action="write", &
        iostat=io_status)
    if (io_status /= 0) error stop "multiclass calibration oracle open failed"
    write (unit, '(a)') "quantity,row,column,value"
    do j = 1, n_classes
        write (unit, '(a,",",i0,",",i0,",",i0)') "class", j, 1, classes(j)
    end do
    do i = 1, n_samples
        write (unit, '(a,",",i0,",",i0,",",i0)') "label", i, 1, labels(i)
        write (unit, '(a,",",i0,",",i0,",",i0)') "prediction", i, 1, predictions(i)
        do j = 1, n_classes
            write (unit, '(a,",",i0,",",i0,",",es24.16)') &
                "probability", i, j, probabilities(i, j)
        end do
    end do
    write (unit, '(a,",",i0,",",i0,",",es24.16)') &
        "temperature", 1, 1, fitted_parameters(1)
    close (unit)

contains

    subroutine fixture(scores, labels)
        real(real64), intent(out) :: scores(:, :)
        integer, intent(out) :: labels(:)
        real(real64) :: first, second, third
        integer :: i

        do i = 1, size(labels)
            first = 1.2_real64*sin(0.031_real64*real(i, real64)) + &
                0.17_real64*cos(0.013_real64*real(i, real64))
            second = 1.1_real64*cos(0.027_real64*real(i, real64)+0.4_real64) + &
                0.12_real64*sin(0.017_real64*real(i, real64))
            third = 0.9_real64*sin(0.019_real64*real(i, real64)+1.1_real64) - &
                0.18_real64*cos(0.011_real64*real(i, real64))
            scores(i, :) = [first, second, third]
            if (first >= second .and. first >= third) then
                labels(i) = -4
            else if (second >= third) then
                labels(i) = 17
            else
                labels(i) = 91
            end if
        end do
    end subroutine fixture

end program fortml_bench_multiclass_probability_calibration
