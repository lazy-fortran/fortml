program fortml_bench_xgboost_multiclass_validation
    !! Release workload for weighted multiclass XGBoost validation monitoring.
    !! The companion Python lane independently replays every depth-one
    !! logistic OVR update before accepting these records.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_options_t
    use fortml_xgboost_multiclass, only: xgboost_multiclass_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    real(dp) :: x(9, 1), validation_x(6, 1), staged(6, 3, 1)
    real(dp) :: probabilities(6, 3), before(6, 3), after(6, 3)
    real(dp) :: validation_weight(6), expected_loss, start_time, finish_time
    integer :: labels(9), validation_labels(6), invalid_labels(6)
    type(xgboost_multiclass_t) :: model
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    logical :: refused

    x(:, 1) = [-4.0_dp, -3.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, &
        2.0_dp, 3.0_dp, 4.0_dp]
    labels = [-8, -8, -8, 2, 2, 2, 11, 11, 11]
    validation_x(:, 1) = [-3.5_dp, -1.5_dp, -0.2_dp, 0.8_dp, 2.2_dp, 3.7_dp]
    validation_labels = [-8, -8, 2, 2, 11, 11]
    validation_weight = [1.0_dp, 2.0_dp, 1.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    options%n_estimators = 5
    options%max_depth = 1
    options%learning_rate = 0.4_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.0_dp
    options%early_stopping_rounds = 1
    options%early_stopping_min_delta = 1.0e6_dp
    options%restore_best = .true.

    call cpu_time(start_time)
    call model%fit(x, labels, status, options, validation_x=validation_x, &
        validation_labels=validation_labels, validation_weight=validation_weight)
    call cpu_time(finish_time)
    if (.not. status_ok(status)) error stop "multiclass validation fit failed"
    write (*, '(a,1x,es24.16)') "xgb_mc_validation_fit_seconds", &
        finish_time - start_time

    call model%predict_proba_staged(validation_x, staged, status)
    if (.not. status_ok(status)) error stop "multiclass staged prediction failed"
    call model%predict_proba(validation_x, probabilities, status)
    if (.not. status_ok(status)) error stop "multiclass probability prediction failed"
    expected_loss = weighted_log_loss(staged(:, :, 1), validation_labels, &
        validation_weight)
    write (*, '(a,1x,i0)') "xgb_mc_validation_best_iteration", model%best_iteration()
    write (*, '(a,1x,i0)') "xgb_mc_validation_requested", &
        model%requested_estimator_count()
    write (*, '(a,1x,i0)') "xgb_mc_validation_retained", model%estimator_count()
    write (*, '(a,1x,i0)') "xgb_mc_validation_early_stopped", &
        merge(1, 0, model%early_stopped())
    write (*, '(a,1x,es24.16)') "xgb_mc_validation_best_loss", &
        model%best_validation_loss()
    write (*, '(a,1x,es24.16)') "xgb_mc_validation_oracle_loss", expected_loss
    write (*, '(a,1x,es24.16)') "xgb_mc_validation_staged_error", &
        maxval(abs(probabilities - staged(:, :, 1)))

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, validation_x, probabilities, status)
    write (*, '(a,1x,i0)') "xgb_mc_validation_cuda_status", status%code

    call model%predict_proba(validation_x, before, status)
    invalid_labels = validation_labels
    invalid_labels(1) = 999
    call model%fit(x, labels, status, options, validation_x=validation_x, &
        validation_labels=invalid_labels, validation_weight=validation_weight)
    refused = .not. status_ok(status)
    call model%predict_proba(validation_x, after, status)
    write (*, '(a,1x,i0)') "xgb_mc_validation_invalid_status", merge(1, 0, refused)
    write (*, '(a,1x,es24.16)') "xgb_mc_validation_transaction_error", &
        maxval(abs(after - before))

contains

    real(dp) function weighted_log_loss(probabilities, labels, weights) result(loss)
        real(dp), intent(in) :: probabilities(:, :), weights(:)
        integer, intent(in) :: labels(:)
        integer :: i, class_index

        loss = 0.0_dp
        do i = 1, size(labels)
            class_index = 1
            if (labels(i) == 2) class_index = 2
            if (labels(i) == 11) class_index = 3
            loss = loss - weights(i)*log(max(probabilities(i, class_index), 1.0e-15_dp))
        end do
        loss = loss/sum(weights)
    end function weighted_log_loss

end program fortml_bench_xgboost_multiclass_validation
