program test_xgboost_dart
    !! Independent behavioral oracle for seeded XGBoost DART state.
    use, intrinsic :: iso_fortran_env, only: real64, int64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    implicit none

    integer :: failures

    failures = 0
    call test_determinism_and_products(failures)
    call test_warm_start(failures)
    call test_schema_and_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " xgboost DART test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine fixture(x, y)
        real(real64), intent(out) :: x(:, :), y(:)
        integer :: i

        do i = 1, size(y)
            x(i, 1) = real(i - 1, real64)
            x(i, 2) = real(mod(3*i + 1, 7), real64) - 2.0_real64
            y(i) = 0.3_real64*x(i, 1) - 0.4_real64*x(i, 2) + &
                merge(0.25_real64, -0.15_real64, i > size(y)/2)
        end do
    end subroutine fixture

    subroutine dart_options(options, n_estimators)
        type(xgboost_options_t), intent(out) :: options
        integer, intent(in) :: n_estimators

        options = xgboost_options_t()
        options%n_estimators = n_estimators
        options%max_depth = 2
        options%learning_rate = 0.35_real64
        options%booster = "dart"
        options%dart_drop_rate = 0.62_real64
        options%dart_skip_drop = 0.0_real64
        options%dart_max_drop = 2
        options%subsample = 0.8_real64
        options%colsample_bytree = 0.75_real64
        options%seed = 424242_int64
    end subroutine dart_options

    subroutine test_determinism_and_products(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: first, repeat, restored
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(12, 2), y(12), query(7, 2)
        real(real64) :: p_first(7), p_repeat(7), p_restored(7), margin(7)
        real(real64) :: staged(7, 6), contributions(7, 7), product_sum(7)
        real(real64) :: staged_margin(7, 6)
        character(*), parameter :: path = "test_xgboost_dart.txt"
        integer :: i

        call fixture(x, y)
        do i = 1, size(query, 1)
            query(i, 1) = real(i - 2, real64) + 0.2_real64
            query(i, 2) = real(mod(i + 1, 5), real64) - 1.5_real64
        end do
        call dart_options(options, 6)
        call first%fit_regression(x, y, status, options)
        call check(status_ok(status) .and. first%fitted(), "DART fit", failures)
        call check(first%booster() == "dart" .and. &
            abs(first%dart_drop_rate() - options%dart_drop_rate) < 1.0e-15_real64 .and. &
            first%dart_max_drop() == 2, "DART metadata", failures)
        call repeat%fit_regression(x, y, status, options)
        call check(status_ok(status), "repeat DART fit", failures)
        call first%predict(query, p_first, status)
        call repeat%predict(query, p_repeat, status)
        call check(status_ok(status) .and. maxval(abs(p_first-p_repeat)) < 2.0e-13_real64, &
            "seed replay", failures)
        call first%predict_margin(query, margin, status)
        call check(status_ok(status), "DART margin", failures)
        call first%predict_staged(query, staged, status)
        call check(status_ok(status) .and. maxval(abs(staged(:, 6)-p_first)) < &
            2.0e-13_real64, "staged final equals prediction", failures)
        call first%predict_staged_margin(query, staged_margin, status)
        call check(status_ok(status), "staged margins", failures)
        call first%predict_contributions(query, contributions, status)
        product_sum = sum(contributions, dim=2)
        call check(status_ok(status) .and. maxval(abs(product_sum-margin)) < 2.0e-13_real64, &
            "contributions sum to DART margin", failures)

        call first%save_text(path, status)
        call check(status_ok(status), "DART save", failures)
        call restored%load_text(path, status)
        call check(status_ok(status) .and. restored%booster() == "dart", &
            "DART load metadata", failures)
        call restored%predict(query, p_restored, status)
        call check(status_ok(status) .and. maxval(abs(p_restored-p_first)) < 2.0e-13_real64, &
            "DART persistence prediction", failures)
        call execute_command_line("rm -f "//path)
    end subroutine test_determinism_and_products

    subroutine test_warm_start(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: prefix, warm, full
        type(xgboost_options_t) :: prefix_options, full_options
        type(fortnum_status_t) :: status
        real(real64) :: x(12, 2), y(12), warm_staged(12, 6), full_staged(12, 6)

        call fixture(x, y)
        call dart_options(prefix_options, 3)
        call prefix%fit_regression(x, y, status, prefix_options)
        call check(status_ok(status), "DART prefix", failures)
        full_options = prefix_options
        full_options%n_estimators = 6
        call full%fit_regression(x, y, status, full_options)
        call check(status_ok(status), "DART full", failures)
        warm = prefix
        call warm%fit_warm_start(x, y, status, full_options)
        call check(status_ok(status), "DART warm continuation", failures)
        call warm%predict_staged_margin(x, warm_staged, status)
        call full%predict_staged_margin(x, full_staged, status)
        call check(status_ok(status) .and. maxval(abs(warm_staged-full_staged)) < &
            2.0e-13_real64, "DART warm continuation matches full fit", failures)
    end subroutine test_warm_start

    subroutine test_schema_and_refusals(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        type(fortml_device_t) :: device
        real(real64) :: x(8, 2), y(8), output(8)

        call fixture(x, y)
        call dart_options(options, 3)
        options%dart_drop_rate = 1.0_real64
        call model%fit_regression(x, y, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR .and. .not. model%fitted(), &
            "invalid dropout rate refusal", failures)
        call dart_options(options, 3)
        options%dart_skip_drop = 1.0_real64
        call model%fit_regression(x, y, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR .and. .not. model%fitted(), &
            "invalid skip rate refusal", failures)
        call dart_options(options, 3)
        options%dart_max_drop = -1
        call model%fit_regression(x, y, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR .and. .not. model%fitted(), &
            "invalid drop cap refusal", failures)
        call dart_options(options, 3)
        call model%fit_regression(x, y, status, options)
        call check(status_ok(status), "DART CUDA fixture", failures)
        device%selected = .true.
        device%available = .true.
        device%kind = FORTML_DEVICE_CUDA
        call model%predict_device(device, x, output, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA typed refusal", failures)
    end subroutine test_schema_and_refusals

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [xgb dart] "//label
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_dart
