program fortml_bench_rbf_svm_multiclass
    !! Correctness-gated dense OVR multiclass RBF-SVM workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_rbf_svm_multiclass, only: rbf_svm_multiclass_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 36, n_features = 2, n_classes = 3
    integer, parameter :: prediction_repetitions = 64
    real(dp) :: x(n_samples, n_features), scores(n_samples, n_classes)
    real(dp) :: probabilities(n_samples, n_classes), sample_weight(n_samples)
    real(dp) :: cuda_probabilities(n_samples, n_classes)
    real(dp), allocatable :: parameters(:)
    integer :: labels(n_samples), predicted(n_samples), classes(n_classes)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition, cuda_code
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    type(rbf_svm_multiclass_t) :: model

    call get_environment_variable("FORTML_BENCH_RBF_SVM_MULTICLASS_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_RBF_SVM_MULTICLASS_ORACLE is required"
    end if
    call make_fixture(x, labels, sample_weight)
    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, c=2.0_dp, gamma=0.6_dp, &
        max_iterations=50000, tolerance=1.0e-6_dp, sample_weight=sample_weight)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "multiclass RBF-SVM fit failed"
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    call model%decision_function(x, scores, status)
    call model%predict_proba(x, probabilities, status)
    call model%predict(x, predicted, status)
    parameters = model%parameters()
    classes = model%classes()
    if (.not. status_ok(status)) error stop "multiclass RBF-SVM prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp) &
        /real(prediction_repetitions, dp)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, cuda_probabilities, status)
    cuda_code = status%code

    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do i = 1, n_samples
        write (unit, '(a,i0,a,i0,a,i0)') "label,", i, ",1,", labels(i)
        write (unit, '(a,i0,a,i0,a,i0)') "prediction,", i, ",1,", predicted(i)
        do j = 1, n_classes
            write (unit, '(a,i0,a,i0,a,es24.16)') "score,", i, ",", j, ",", scores(i, j)
            write (unit, '(a,i0,a,i0,a,es24.16)') "probability,", i, ",", j, ",", &
                probabilities(i, j)
        end do
    end do
    do i = 1, size(parameters)
        write (unit, '(a,i0,a,es24.16)') "parameter,", i, ",1,", parameters(i)
    end do
    do j = 1, n_classes
        write (unit, '(a,i0,a,i0)') "class,", j, ",1,", classes(j)
    end do
    write (unit, '(a,es24.16)') "fit_seconds,1,1,", fit_seconds
    write (unit, '(a,es24.16)') "predict_seconds,1,1,", predict_seconds
    write (unit, '(a,i0)') "cuda_status,1,1,", cuda_code
    close (unit)

contains

    subroutine make_fixture(x, labels, sample_weight)
        real(dp), intent(out) :: x(:, :), sample_weight(:)
        integer, intent(out) :: labels(:)
        real(dp) :: phase
        integer :: i

        do i = 1, size(x, 1)
            phase = real(i, dp)
            select case ((i - 1)/(size(x, 1)/3))
            case (0)
                x(i, 1) = -1.0_dp + 0.05_dp*sin(0.17_dp*phase)
                x(i, 2) = 0.2_dp*cos(0.13_dp*phase)
                labels(i) = -12
            case (1)
                x(i, 1) = 0.0_dp + 0.05_dp*sin(0.17_dp*phase)
                x(i, 2) = -0.2_dp*cos(0.13_dp*phase)
                labels(i) = 7
            case default
                x(i, 1) = 1.0_dp + 0.05_dp*sin(0.17_dp*phase)
                x(i, 2) = 0.2_dp*cos(0.13_dp*phase)
                labels(i) = 37
            end select
            sample_weight(i) = 0.75_dp + 0.5_dp*real(mod(i, 7), dp)/6.0_dp
        end do
    end subroutine make_fixture

end program fortml_bench_rbf_svm_multiclass
