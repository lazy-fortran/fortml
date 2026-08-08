program fortml_bench_mlp_classifier_parameter_products
    !! Fixed-input multiclass MLP probability parameter-product workload.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp_classifier, only: mlp_classifier_t, mlp_classifier_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 9, n_features = 2, n_classes = 3
    integer, parameter :: prediction_repetitions = 128
    real(dp) :: x(n_samples, n_features), probabilities(n_samples, n_classes)
    real(dp) :: probabilities_dot(n_samples, n_classes)
    real(dp) :: probabilities_bar(n_samples, n_classes)
    integer :: labels(n_samples)
    real(dp), allocatable :: theta(:), theta_dot(:), theta_bar(:)
    type(mlp_classifier_t) :: model
    type(mlp_classifier_options_t) :: options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: fit_seconds, predict_seconds, jvp_seconds, vjp_seconds
    character(len=1024) :: oracle_path
    integer :: environment_status, unit, i, j, repetition

    call get_environment_variable("FORTML_BENCH_MLP_CLASSIFIER_PARAMETER_PRODUCTS_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) then
        error stop "FORTML_BENCH_MLP_CLASSIFIER_PARAMETER_PRODUCTS_ORACLE is required"
    end if
    call make_fixture(x, labels)
    options%max_epochs = 30
    options%learning_rate = 0.03_dp
    options%initialization_seed = 29
    options%restore_best = .false.

    call system_clock(clock_start, clock_rate)
    call model%fit(x, labels, status, hidden_layer_sizes=[3], options=options)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "MLP classifier parameter-product fit failed"
    fit_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)
    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_bar(size(theta)))
    theta_dot = [(0.013_dp*real(i, dp), i=1, size(theta))]
    probabilities_bar = reshape([(0.017_dp*real(i, dp), i=1, size(probabilities_bar))], &
        shape(probabilities_bar))

    call model%predict_proba(x, probabilities, status)
    if (.not. status_ok(status)) error stop "MLP classifier probability prediction failed"
    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba(x, probabilities, status)
    end do
    call system_clock(clock_end)
    predict_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/ &
        real(prediction_repetitions, dp)

    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba_parameter_jvp(x, theta_dot, probabilities, &
            probabilities_dot, status)
    end do
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "MLP classifier parameter JVP failed"
    jvp_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/ &
        real(prediction_repetitions, dp)

    call system_clock(clock_start, clock_rate)
    do repetition = 1, prediction_repetitions
        call model%predict_proba_parameter_vjp(x, probabilities_bar, theta_bar, status)
    end do
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "MLP classifier parameter VJP failed"
    vjp_seconds = real(clock_end-clock_start, dp)/real(clock_rate, dp)/ &
        real(prediction_repetitions, dp)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_proba_parameter_jvp_device(cpu, x, theta_dot, probabilities, &
        probabilities_dot, status)
    if (.not. status_ok(status)) error stop "MLP classifier CPU parameter JVP failed"
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_parameter_jvp_device(cuda, x, theta_dot, probabilities, &
        probabilities_dot, status)
    open (newunit=unit, file=trim(oracle_path), status="replace", action="write")
    write (unit, '(a)') "quantity,row,column,value"
    do j = 1, size(theta)
        write (unit, '(a,i0,a,es26.17e3)') "parameter,", j, ",1,", theta(j)
        write (unit, '(a,i0,a,es26.17e3)') "parameter_tangent,", j, ",1,", theta_dot(j)
        write (unit, '(a,i0,a,es26.17e3)') "parameter_bar,", j, ",1,", theta_bar(j)
    end do
    do i = 1, n_samples
        do j = 1, n_classes
            write (unit, '(a,i0,a,i0,a,es26.17e3)') &
                "probability,", i, ",", j, ",", probabilities(i, j)
            write (unit, '(a,i0,a,i0,a,es26.17e3)') &
                "probability_tangent,", i, ",", j, ",", probabilities_dot(i, j)
            write (unit, '(a,i0,a,i0,a,es26.17e3)') &
                "probability_cotangent,", i, ",", j, ",", probabilities_bar(i, j)
        end do
    end do
    write (unit, '(a,i0,a,i0,a,es26.17e3)') &
        "cuda_jvp_status,", 1, ",", 1, ",", real(status%code, dp)
    close (unit)
    write (*, '(a,es24.16)') "mlp_classifier_parameter_products_fit,tanh,", fit_seconds
    write (*, '(a,es24.16)') "mlp_classifier_parameter_products_predict,tanh,", predict_seconds
    write (*, '(a,es24.16)') "mlp_classifier_parameter_products_jvp,tanh,", jvp_seconds
    write (*, '(a,es24.16)') "mlp_classifier_parameter_products_vjp,tanh,", vjp_seconds

contains

    subroutine make_fixture(features, target)
        real(dp), intent(out) :: features(:, :)
        integer, intent(out) :: target(:)

        features = reshape([ &
            -2.0_dp, -1.0_dp, -1.0_dp, -2.0_dp, 0.0_dp, 2.0_dp, &
            0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 2.0_dp, 1.0_dp, &
            2.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, 2.0_dp, 1.0_dp], shape(features))
        target = [31, 31, -7, -7, 42, 42, 42, 31, -7]
    end subroutine make_fixture

end program fortml_bench_mlp_classifier_parameter_products
