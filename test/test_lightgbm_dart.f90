program test_lightgbm_dart
    !! Independent seeded oracle for the bounded LightGBM DART policy.
    !! The hash stream is configured so round two drops tree one and round
    !! three drops tree one (with a one-tree cap), hence tree-normalisation
    !! scales are [1/4, 1/2, 1/2].  Predictions, staged products, persistence,
    !! and warm-start continuation must preserve those stored scales.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    character(*), parameter :: snapshot = "test_lightgbm_dart.txt"
    type(lightgbm_t) :: model, repeated, restored, prefix, continued
    type(lightgbm_options_t) :: options, prefix_options, invalid
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(6, 1), target(6), prediction(6), replay(6), restored_prediction(6)
    real(real64) :: staged(6, 3), contributions(6, 4), margin(6)
    real(real64) :: x_query(1, 1), x_dot(1, 1), y_query(1), y_dot(1), x_bar(1, 1)
    integer :: failures

    failures = 0
    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64, 5.0_real64]
    target = [0.0_real64, 1.0_real64, 0.0_real64, 1.0_real64, 0.0_real64, 1.0_real64]
    options = lightgbm_options_t()
    options%n_estimators = 3
    options%num_leaves = 2
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 0.5_real64
    options%l2 = 1.0_real64
    options%boosting_type = "dart"
    options%dart_drop_rate = 0.99_real64
    options%dart_skip_drop = 0.0_real64
    options%dart_max_drop = 1
    options%seed = 1729

    call model%fit_regression(x, target, status, options)
    call check(status_ok(status), "DART fit", failures)
    call check(trim(model%boosting_type()) == "dart", "DART metadata type", failures)
    call check(abs(model%dart_drop_rate()-0.99_real64) < 1.0e-14_real64 .and. &
        model%dart_max_drop() == 1, "DART metadata controls", failures)
    call check(abs(model%tree_scale(1)-0.25_real64) < 1.0e-14_real64 .and. &
        abs(model%tree_scale(2)-0.5_real64) < 1.0e-14_real64 .and. &
        abs(model%tree_scale(3)-0.5_real64) < 1.0e-14_real64, &
        "seeded dropout/tree-normalisation oracle", failures)

    call model%predict(x, prediction, status)
    call check(status_ok(status), "DART prediction", failures)
    call repeated%fit_regression(x, target, status, options)
    call repeated%predict(x, replay, status)
    call check(status_ok(status) .and. maxval(abs(replay-prediction)) < 1.0e-14_real64, &
        "DART deterministic seed replay", failures)

    call model%predict_staged_margin(x, staged, status)
    call check(status_ok(status), "DART staged margins", failures)
    call model%predict_margin(x, margin, status)
    call check(status_ok(status) .and. maxval(abs(staged(:, 3)-margin)) < 2.0e-13_real64, &
        "DART final staged margin", failures)
    call model%predict_contributions(x, contributions, status)
    call check(status_ok(status) .and. maxval(abs(sum(contributions, dim=2)-margin)) < &
        2.0e-13_real64, "DART contributions sum to margin", failures)

    call model%save_text(snapshot, status)
    call check(status_ok(status), "DART save text", failures)
    call restored%load_text(snapshot, status)
    call check(status_ok(status), "DART load text", failures)
    call restored%predict(x, restored_prediction, status)
    call check(status_ok(status) .and. maxval(abs(restored_prediction-prediction)) < &
        2.0e-13_real64 .and. abs(restored%tree_scale(1)-model%tree_scale(1)) < &
        1.0e-14_real64, "DART persistence scale/prediction", failures)

    prefix_options = options
    prefix_options%n_estimators = 2
    call prefix%fit_regression(x, target, status, prefix_options)
    call check(status_ok(status), "DART warm-start prefix", failures)
    call continued%fit_regression(x, target, status, prefix_options)
    prefix_options%n_estimators = 3
    call continued%fit_warm_start(x, target, status, prefix_options)
    call check(status_ok(status), "DART warm-start continuation", failures)
    call continued%predict(x, replay, status)
    call check(status_ok(status) .and. maxval(abs(replay-prediction)) < 2.0e-13_real64, &
        "DART warm-start equals full fit", failures)

    invalid = options
    invalid%dart_drop_rate = 1.0_real64
    call repeated%fit_regression(x, target, status, invalid)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid DART rate refusal", failures)

    x_query(1, 1) = 0.25_real64
    x_dot(1, 1) = 1.0_real64
    call model%predict_jvp(x_query, x_dot, y_query, y_dot, status)
    call check(status_ok(status) .and. abs(y_dot(1)) < 1.0e-14_real64, &
        "DART fixed-tree JVP away from split", failures)
    call model%predict_vjp(x_query, [1.0_real64], x_bar, status)
    call check(status_ok(status) .and. maxval(abs(x_bar)) < 1.0e-14_real64, &
        "DART fixed-tree VJP away from split", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "DART CUDA typed refusal", failures)

    call delete_file(snapshot)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " LightGBM DART test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM DART independent seeded oracle"

contains

    subroutine delete_file(filename)
        character(len=*), intent(in) :: filename
        integer :: unit, local_iostat

        open(newunit=unit, file=filename, status="old", action="read", iostat=local_iostat)
        if (local_iostat == 0) close(unit, status="delete")
    end subroutine delete_file

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_lightgbm_dart
