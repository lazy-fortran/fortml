program fortml_bench_reliability_diagram
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_classification_metrics, only: classification_reliability_diagram
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 512, n_classes = 3, bins = 10
    integer, parameter :: repetitions = 128
    real(real64) :: probabilities(n_samples, n_classes), weights(n_samples)
    real(real64) :: mean_confidence(bins), mean_accuracy(bins), bin_weight(bins)
    integer :: labels(n_samples), classes(n_classes)
    integer :: i, repetition, environment_status, io_status, unit
    real(real64) :: started, finished, seconds
    type(fortnum_status_t) :: status
    character(len=1024) :: oracle_path

    classes = [-4, 17, 91]
    call fixture(probabilities, labels, weights)
    call classification_reliability_diagram(probabilities, labels, classes, bins, &
        mean_confidence, mean_accuracy, bin_weight, status, sample_weight=weights)
    if (.not. status_ok(status)) error stop "reliability diagram failed"
    call cpu_time(started)
    do repetition = 1, repetitions
        call classification_reliability_diagram(probabilities, labels, classes, bins, &
            mean_confidence, mean_accuracy, bin_weight, status, sample_weight=weights)
        if (.not. status_ok(status)) error stop "reliability diagram timing failed"
    end do
    call cpu_time(finished)
    seconds = (finished-started)/real(repetitions, real64)
    write (*, '(a,",",a,",",es24.16)') &
        "reliability_diagram", "weighted_curve", seconds

    call get_environment_variable("FORTML_BENCH_RELIABILITY_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) stop
    open (newunit=unit, file=trim(oracle_path), status="replace", action="write", &
        iostat=io_status)
    if (io_status /= 0) error stop "reliability oracle open failed"
    write (unit, '(a)') "bin,mean_confidence,mean_accuracy,bin_weight"
    do i = 1, bins
        write (unit, '(i0,",",es24.16,",",es24.16,",",es24.16)') i, &
            mean_confidence(i), mean_accuracy(i), bin_weight(i)
    end do
    close (unit)

contains

    subroutine fixture(probabilities, labels, weights)
        real(real64), intent(out) :: probabilities(:, :), weights(:)
        integer, intent(out) :: labels(:)
        real(real64) :: raw(3), total
        integer :: i

        do i = 1, size(labels)
            raw = [0.2_real64 + abs(sin(0.017_real64*real(i, real64))), &
                0.3_real64 + abs(cos(0.013_real64*real(i, real64)+0.2_real64)), &
                0.4_real64 + abs(sin(0.011_real64*real(i, real64)+0.7_real64))]
            total = sum(raw)
            probabilities(i, :) = raw/total
            labels(i) = merge(-4, merge(17, 91, mod(7*i+2, 3) == 1), &
                mod(7*i+2, 3) == 0)
            weights(i) = 0.5_real64 + real(mod(5*i+1, 7), real64)/3.0_real64
        end do
    end subroutine fixture

end program fortml_bench_reliability_diagram
