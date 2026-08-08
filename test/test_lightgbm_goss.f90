program test_lightgbm_goss
    !! Independent hand oracle for LightGBM gradient-based one-side sampling.
    !! The fixture has equal-magnitude top gradients, so stable row ordering
    !! and the analytic `(1-a)/b` reweight produce two exact leaves (0, 10).
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(lightgbm_t) :: model, repeated
    type(lightgbm_options_t) :: options, invalid
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(4, 1), target(4), prediction(4), repeated_prediction(4)
    integer :: failures

    failures = 0
    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    target = [0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64]
    options = lightgbm_options_t()
    options%n_estimators = 1
    options%num_leaves = 2
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 1.0_real64
    options%l2 = 0.0_real64
    options%boosting_type = "goss"
    options%top_rate = 0.5_real64
    options%other_rate = 0.25_real64
    options%seed = 1729
    call model%fit_regression(x, target, status, options)
    call check(status_ok(status), "GOSS fit", failures)
    call model%predict(x, prediction, status)
    call check(status_ok(status), "GOSS prediction", failures)
    call check(maxval(abs(prediction-target)) < 1.0e-12_real64, &
        "GOSS weighted two-leaf hand oracle", failures)
    call check(trim(model%boosting_type()) == "goss" .and. &
        abs(model%top_rate()-0.5_real64) < 1.0e-14_real64 .and. &
        abs(model%other_rate()-0.25_real64) < 1.0e-14_real64, &
        "GOSS metadata", failures)

    call repeated%fit_regression(x, target, status, options)
    call repeated%predict(x, repeated_prediction, status)
    call check(status_ok(status) .and. maxval(abs(repeated_prediction-prediction)) < &
        1.0e-14_real64, "GOSS deterministic seed replay", failures)

    invalid = options
    invalid%top_rate = 0.8_real64
    invalid%other_rate = 0.25_real64
    call repeated%fit_regression(x, target, status, invalid)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid GOSS rates refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "GOSS CUDA typed refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " LightGBM GOSS test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM GOSS independent behavioral oracle"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_lightgbm_goss
