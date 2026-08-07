program test_lightgbm
    !! Independent behavioral oracle for the bounded leaf-wise histogram path.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(lightgbm_t) :: model, weighted, binary
    type(lightgbm_options_t) :: options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(real64) :: x(6, 1), target(6), prediction(6), expected(6)
    real(real64) :: wx(4, 1), wy(4), weights(4), wpred(4)
    real(real64) :: bx(4, 1), labels(4), probabilities(4, 2)
    real(real64) :: xdot(6, 1), ydot(6), xbar(6, 1), ybar(6), boundary(1, 1)
    integer :: failures

    failures = 0
    x(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64, 5.0_real64]
    target = [0.0_real64, 0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64, 100.0_real64]
    options = lightgbm_options_t()
    options%n_estimators = 1
    options%num_leaves = 3
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 1.0_real64
    options%l2 = 0.0_real64
    call model%fit_regression(x, target, status, options)
    call model%predict(x, prediction, status)
    expected = [0.0_real64, 0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64, 100.0_real64]
    call check(status_ok(status), "leaf-wise regression fit/predict", failures)
    call check(model%num_leaves() == 3 .and. model%tree_node_count(1) == 5 .and. &
        model%tree_depth(1) == 2, "best-first tree shape", failures)
    call check(maxval(abs(prediction-expected)) < 1.0e-11_real64, &
        "leaf-wise regression hand oracle", failures)
    call check(trim(model%objective_name()) == "regression", "objective metadata", failures)

    wx(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    wy = [0.0_real64, 0.0_real64, 10.0_real64, 10.0_real64]
    weights = [1.0_real64, 2.0_real64, 1.0_real64, 2.0_real64]
    options%num_leaves = 2
    options%max_bin = 4
    call weighted%fit_regression(wx, wy, status, options, weights)
    call weighted%predict(wx, wpred, status)
    call check(status_ok(status), "weighted fit/predict", failures)
    call check(maxval(abs(wpred-wy)) < 1.0e-11_real64 .and. &
        abs(weighted%base_margin()-5.0_real64) < 1.0e-12_real64, &
        "weighted leaf mean oracle", failures)

    bx(:, 1) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
    labels = [0.0_real64, 0.0_real64, 1.0_real64, 1.0_real64]
    options%num_leaves = 2
    options%l2 = 0.0_real64
    call binary%fit_binary(bx, labels, status, options)
    call binary%predict_proba(bx, probabilities, status)
    call check(status_ok(status), "binary fit/probability status", failures)
    call check(maxval(abs(probabilities(:, 2) - [1.0_real64/(1.0_real64+exp(2.0_real64)), &
        1.0_real64/(1.0_real64+exp(2.0_real64)), 1.0_real64/(1.0_real64+exp(-2.0_real64)), &
        1.0_real64/(1.0_real64+exp(-2.0_real64))])) < 1.0e-11_real64, &
        "binary logistic Newton oracle", failures)

    xdot = 0.0_real64
    ybar = 0.0_real64
    xbar = 0.0_real64
    call model%predict_jvp(x, xdot, prediction, ydot, status)
    call check(status_ok(status) .and. maxval(abs(ydot)) < 1.0e-14_real64, &
        "piecewise input JVP away from split", failures)
    call model%predict_vjp(x, prediction, xbar, status)
    call check(status_ok(status) .and. maxval(abs(xbar)) < 1.0e-14_real64, &
        "piecewise input VJP away from split", failures)
    boundary(1, 1) = 4.5_real64
    call model%predict_jvp(boundary, boundary, ybar(1:1), ydot(1:1), status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "split-boundary JVP refusal", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call check(model%device_supported(FORTML_DEVICE_CPU) .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), "device capability", failures)
    call model%predict_device(cuda, x, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA typed refusal", failures)
    call model%predict_device(cpu, x, prediction, status)
    call check(status_ok(status), "CPU device dispatch", failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " LightGBM test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM leaf-wise independent behavioral oracle"

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

end program test_lightgbm
