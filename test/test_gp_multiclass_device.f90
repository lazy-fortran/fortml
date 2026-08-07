program test_gp_multiclass_device
    !! Independent CPU dispatch and synthetic-CUDA refusal checks.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t, gp_multiclass_classification_options_t, &
        gp_multiclass_classification_state_t
    implicit none

    type(gp_multiclass_classification_t) :: model
    type(gp_multiclass_classification_options_t) :: options
    type(gp_multiclass_classification_state_t) :: state
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 2), query(3, 2), probabilities(3, 3), cpu_probabilities(3, 3)
    integer :: labels(9), predicted(3), cpu_predicted(3), failures

    x = reshape([ &
        -1.0_dp, 1.0_dp, -0.8_dp, 1.1_dp, -1.1_dp, 0.9_dp, &
        0.0_dp, 0.0_dp,  0.1_dp, 0.1_dp, -0.1_dp, 0.0_dp, &
        1.0_dp, 1.0_dp,  0.9_dp, 1.1_dp,  1.1_dp, 0.9_dp], shape(x))
    labels = [42, 42, 42, -7, -7, -7, 10, 10, 10]
    query = reshape([ &
        -0.9_dp, 1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp], shape(query))
    failures = 0
    kernel = make_rbf_kernel(2, 1.0_dp, 0.8_dp, status)
    options%max_iterations = 100
    options%tolerance = 1.0e-8_dp
    options%jitter = 1.0e-7_dp
    call model%fit(x, labels, kernel, status, options, state)
    call check(status_ok(status) .and. state%converged .and. model%fitted(), &
        "multiclass GP fit", failures)
    call check(model%device_supported(FORTML_DEVICE_CPU) .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "multiclass GP capability metadata", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, query, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "multiclass GP CUDA probability refusal", failures)
    call model%predict_device(cuda, query, predicted, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "multiclass GP CUDA label refusal", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call check(status_ok(status), "CPU device selection", failures)
    call model%predict_proba_device(cpu, query, cpu_probabilities, status)
    call model%predict_device(cpu, query, cpu_predicted, status)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predicted, status)
    call check(status_ok(status) .and. maxval(abs(cpu_probabilities - probabilities)) < 2.0e-14_dp .and. &
        all(cpu_predicted == predicted), "CPU dispatch matches reference", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GP multiclass device cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP multiclass device contract independent oracle"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-multiclass-device] "//description
        end if
    end subroutine check

end program test_gp_multiclass_device
