program test_xgboost_multiclass_device
    !! Independent API oracle for multiclass XGBoost device and constraint
    !! propagation.  CUDA is required to refuse explicitly until a resident
    !! tree kernel exists; CPU dispatch must equal ordinary prediction.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_xgboost, only: xgboost_options_t
    use fortml_xgboost_multiclass, only: xgboost_multiclass_t
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    implicit none

    type(xgboost_multiclass_t) :: model
    type(xgboost_options_t) :: options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 1), probabilities(6, 3), cpu_probabilities(6, 3)
    integer :: labels(6), cpu_labels(6), failures

    ! Import the options type through the binary module without exposing any
    ! implementation state from the multiclass wrapper.
    failures = 0
    x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp]
    labels = [-1, -1, 4, 4, 9, 9]
    options%n_estimators = 2
    options%max_depth = 1
    options%learning_rate = 0.5_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.0_dp
    options%monotone_constraints = [1]
    call model%fit(x, labels, status, options)
    call check(status%code == FORTNUM_OK, "fit", failures)
    call check(model%monotone_constraint(1) == 1, "constraint propagation", failures)

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call model%predict_proba(x, probabilities, status)
    call model%predict_proba_device(cpu, x, cpu_probabilities, status)
    call check(status%code == FORTNUM_OK .and. &
        maxval(abs(probabilities - cpu_probabilities)) < 2.0e-13_dp, &
        "CPU probability dispatch parity", failures)
    call model%predict(x, labels, status)
    call model%predict_device(cpu, x, cpu_labels, status)
    call check(status%code == FORTNUM_OK .and. all(labels == cpu_labels), &
        "CPU label dispatch parity", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA probability refusal", failures)
    call model%predict_device(cuda, x, labels, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA label refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " multiclass XGBoost device test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS XGBoost multiclass device/constraint oracle"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(name)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_multiclass_device
