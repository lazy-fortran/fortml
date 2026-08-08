program fortml_bench_multiclass_isotonic_calibration
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_probability_calibration, only: &
        multiclass_probability_calibrator_t, probability_calibration_options_t, &
        CALIBRATION_ISOTONIC
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: n_samples = 192, n_classes = 3
    integer, parameter :: fit_repetitions = 8, predict_repetitions = 128
    type(multiclass_probability_calibrator_t) :: model
    type(probability_calibration_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: scores(n_samples, n_classes), weights(n_samples)
    real(real64) :: probabilities(n_samples, n_classes), scores_dot(n_samples, n_classes)
    real(real64) :: probabilities_dot(n_samples, n_classes), probabilities_bar(n_samples, n_classes)
    real(real64) :: scores_bar(n_samples, n_classes)
    integer :: labels(n_samples), predictions(n_samples), classes(n_classes)
    integer :: i, j, repetition, environment_status, io_status, unit
    integer :: jvp_status, vjp_status, cuda_status
    real(real64) :: started, finished, fit_seconds, predict_seconds
    character(len=1024) :: oracle_path

    call fixture(scores, labels, weights)
    scores_dot = 0.0_real64
    scores_dot(:, 1) = 0.01_real64
    scores_dot(:, 2) = -0.02_real64
    scores_dot(:, 3) = 0.03_real64
    probabilities_bar = 0.0_real64
    options = probability_calibration_options_t(method=CALIBRATION_ISOTONIC)
    call cpu_time(started)
    do repetition = 1, fit_repetitions
        call model%fit(scores, labels, status, options=options, sample_weight=weights)
        if (.not. status_ok(status)) error stop "multiclass isotonic fit failed"
    end do
    call cpu_time(finished)
    fit_seconds = (finished-started)/real(fit_repetitions, real64)
    call cpu_time(started)
    do repetition = 1, predict_repetitions
        call model%predict_proba(scores, probabilities, status)
        if (.not. status_ok(status)) error stop "multiclass isotonic prediction failed"
    end do
    call cpu_time(finished)
    predict_seconds = (finished-started)/real(predict_repetitions, real64)
    call model%predict_proba(scores, probabilities, status)
    call model%predict(scores, predictions, status)
    classes = model%classes()
    if (.not. status_ok(status)) error stop "multiclass isotonic output failed"

    call model%predict_proba_jvp(scores, scores_dot, probabilities, probabilities_dot, status)
    jvp_status = status%code
    call model%predict_proba_vjp(scores, probabilities_bar, scores_bar, status)
    vjp_status = status%code
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, scores, probabilities, status)
    cuda_status = status%code
    if (jvp_status /= FORTNUM_NOT_IMPLEMENTED .or. vjp_status /= FORTNUM_NOT_IMPLEMENTED .or. &
        cuda_status /= FORTNUM_NOT_IMPLEMENTED) error stop "isotonic refusal contract failed"
    call model%predict_proba(scores, probabilities, status)
    if (.not. status_ok(status)) error stop "multiclass isotonic output refresh failed"

    write (*, '(a,",",a,",",es24.16)') &
        "multiclass_probability_calibration_fit", "isotonic", fit_seconds
    write (*, '(a,",",a,",",es24.16)') &
        "multiclass_probability_calibration_predict", "isotonic", predict_seconds

    call get_environment_variable("FORTML_BENCH_MULTICLASS_ISOTONIC_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) stop
    open (newunit=unit, file=trim(oracle_path), status="replace", action="write", &
        iostat=io_status)
    if (io_status /= 0) error stop "multiclass isotonic oracle open failed"
    write (unit, '(a)') "quantity,row,column,value"
    do j = 1, n_classes
        write (unit, '(a,",",i0,",",i0,",",i0)') "class", j, 1, classes(j)
    end do
    do i = 1, n_samples
        write (unit, '(a,",",i0,",",i0,",",i0)') "label", i, 1, labels(i)
        write (unit, '(a,",",i0,",",i0,",",es24.16)') "weight", i, 1, weights(i)
        write (unit, '(a,",",i0,",",i0,",",i0)') "prediction", i, 1, predictions(i)
        do j = 1, n_classes
            write (unit, '(a,",",i0,",",i0,",",es24.16)') &
                "score", i, j, scores(i, j)
            write (unit, '(a,",",i0,",",i0,",",es24.16)') &
                "probability", i, j, probabilities(i, j)
        end do
    end do
    write (unit, '(a,",",i0,",",i0,",",i0)') "jvp_status", 1, 1, jvp_status
    write (unit, '(a,",",i0,",",i0,",",i0)') "vjp_status", 1, 1, vjp_status
    write (unit, '(a,",",i0,",",i0,",",i0)') "cuda_status", 1, 1, cuda_status
    close (unit)

contains

    subroutine fixture(scores, labels, weights)
        real(real64), intent(out) :: scores(:, :), weights(:)
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
            weights(i) = 0.7_real64 + 0.05_real64*real(mod(i, 9), real64)
            if (first >= second .and. first >= third) then
                labels(i) = -4
            else if (second >= third) then
                labels(i) = 17
            else
                labels(i) = 91
            end if
        end do
    end subroutine fixture

end program fortml_bench_multiclass_isotonic_calibration
