program fortml_bench_lightgbm_early_stopping
    !! Release-app protocol for the LightGBM validation/early-stop contract.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    implicit none

    integer, parameter :: n = 8
    real(real64) :: x(n, 1), target(n), validation_target(n)
    type(lightgbm_t) :: restore_model, retain_model, binary_model
    type(lightgbm_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    integer :: i

    x(:, 1) = real([(i, i = 0, n-1)], real64)
    target = [0.0_real64, 0.0_real64, 0.0_real64, 0.0_real64, &
        10.0_real64, 10.0_real64, 10.0_real64, 10.0_real64]
    validation_target = 10.0_real64-target
    options = lightgbm_options_t()
    options%n_estimators = n
    options%num_leaves = 2
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 1.0_real64
    options%l2 = 1.0_real64
    options%early_stopping_rounds = 2
    options%restore_best = .true.
    call restore_model%fit_regression(x, target, status, options, &
        validation_x=x, validation_y=validation_target)
    if (status%code /= FORTNUM_OK) error stop "lightgbm squared restore fit failed"
    call emit("squared", "restore_best", restore_model)

    options%restore_best = .false.
    call retain_model%fit_regression(x, target, status, options, &
        validation_x=x, validation_y=validation_target)
    if (status%code /= FORTNUM_OK) error stop "lightgbm squared retain fit failed"
    call emit("squared", "retain_all", retain_model)

    options%restore_best = .true.
    call binary_model%fit_binary(x, 1.0_real64-target/10.0_real64, status, options, &
        validation_x=x, validation_y=target/10.0_real64)
    if (status%code /= FORTNUM_OK) error stop "lightgbm binary fit failed"
    call emit("binary", "restore_best", binary_model)

    options%restore_best = .false.
    call binary_model%fit_binary(x, target/10.0_real64, status, options, &
        validation_x=x, validation_y=1.0_real64-target/10.0_real64)
    if (status%code /= FORTNUM_OK) error stop "lightgbm binary inverse fit failed"
    call emit("binary", "retain_all", binary_model)

    call retain_model%fit_binary(x, target/10.0_real64, status, options, &
        validation_x=x)
    write (*, '(a,i0)') "lgbm_early_invalid_validation,", status%code

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call restore_model%predict_device(cuda, x, target, status)
    write (*, '(a,i0)') "lgbm_early_cuda,", status%code

contains

    subroutine emit(objective, policy, model)
        character(len=*), intent(in) :: objective, policy
        type(lightgbm_t), intent(in) :: model

        write (*, '(a,a,a,a,a,i0,a,i0,a,l1,a,es24.16)') "lgbm_early_", &
            trim(objective), ",", trim(policy), ",", model%best_iteration(), ",", &
            model%estimator_count(), ",", model%early_stopped(), ",", &
            model%best_validation_loss()
    end subroutine emit

end program fortml_bench_lightgbm_early_stopping
