program fortml_bench_lightgbm_multiclass
    !! Release workload for normalized LightGBM-style OVR classification.
    !! The companion Python lane independently replays the probability and
    !! validation contracts before accepting these timings.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_lightgbm, only: lightgbm_options_t
    use fortml_lightgbm_multiclass, only: lightgbm_multiclass_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    real(dp) :: x(9, 1), validation_x(6, 1), query(3, 1)
    real(dp) :: probabilities(6, 3), before(6, 3), after(6, 3)
    real(dp) :: staged(6, 3, 1), margins(6, 3, 1)
    real(dp) :: query_dot(3, 1), probabilities_query(3, 3), probabilities_dot(3, 3)
    real(dp) :: probabilities_bar(3, 3), query_bar(3, 1)
    real(dp) :: validation_weight(6), expected_loss, start_time, finish_time
    integer :: labels(9), validation_labels(6), invalid_labels(6)
    type(lightgbm_multiclass_t) :: model
    type(lightgbm_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    logical :: refused

    call make_fixture(x, labels)
    validation_x = x(1:6, :)
    validation_labels = labels(1:6)
    validation_weight = [1.0_dp, 2.0_dp, 1.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    query(:, 1) = [-3.31_dp, 0.37_dp, 2.29_dp]
    query_dot(:, 1) = [0.07_dp, -0.03_dp, 0.02_dp]
    probabilities_bar = reshape([0.4_dp, -0.2_dp, 0.3_dp, -0.1_dp, &
        0.6_dp, -0.5_dp, 0.2_dp, 0.1_dp, -0.3_dp], shape(probabilities_bar))
    options%n_estimators = 4
    options%num_leaves = 2
    options%max_depth = 1
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 0.4_dp
    options%l2 = 1.0_dp
    options%seed = 19
    options%early_stopping_rounds = 1
    options%early_stopping_min_delta = 1.0e6_dp
    options%restore_best = .true.

    call cpu_time(start_time)
    call model%fit(x, labels, status, options, validation_x=validation_x, &
        validation_labels=validation_labels, validation_weight=validation_weight)
    call cpu_time(finish_time)
    if (.not. status_ok(status)) error stop "LightGBM multiclass fit failed"
    write (*, '(a,1x,es24.16)') "lightgbm_mc_fit_seconds", finish_time-start_time

    call model%predict_proba(validation_x, probabilities, status)
    if (.not. status_ok(status)) error stop "LightGBM multiclass prediction failed"
    call model%predict_proba_staged(validation_x, staged, status)
    if (.not. status_ok(status)) error stop "LightGBM multiclass staged prediction failed"
    call model%decision_function_staged(validation_x, margins, status)
    if (.not. status_ok(status)) error stop "LightGBM multiclass staged margin failed"
    expected_loss = weighted_log_loss(staged(:, :, 1), validation_labels, validation_weight)
    write (*, '(a,1x,i0)') "lightgbm_mc_requested", model%requested_estimator_count()
    write (*, '(a,1x,i0)') "lightgbm_mc_best_iteration", model%best_iteration()
    write (*, '(a,1x,i0)') "lightgbm_mc_retained", model%estimator_count()
    write (*, '(a,1x,i0)') "lightgbm_mc_early_stopped", merge(1, 0, model%early_stopped())
    write (*, '(a,1x,es24.16)') "lightgbm_mc_best_loss", model%best_validation_loss()
    write (*, '(a,1x,es24.16)') "lightgbm_mc_oracle_loss", expected_loss
    write (*, '(a,1x,es24.16)') "lightgbm_mc_stage_error", &
        maxval(abs(probabilities - staged(:, :, 1)))
    write (*, '(a,1x,es24.16)') "lightgbm_mc_probability_sum_error", &
        maxval(abs(sum(probabilities, dim=2) - 1.0_dp))

    call model%predict_proba_jvp(query, query_dot, probabilities_query, &
        probabilities_dot, status)
    if (.not. status_ok(status)) error stop "LightGBM multiclass JVP failed"
    write (*, '(a,1x,es24.16)') "lightgbm_mc_jvp_error", maxval(abs(probabilities_dot))
    call model%predict_proba_vjp(query, probabilities_bar, query_bar, status)
    if (.not. status_ok(status)) error stop "LightGBM multiclass VJP failed"
    write (*, '(a,1x,es24.16)') "lightgbm_mc_vjp_error", maxval(abs(query_bar))

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, validation_x, probabilities, status)
    write (*, '(a,1x,i0)') "lightgbm_mc_cuda_status", status%code

    call model%predict_proba(validation_x, before, status)
    invalid_labels = validation_labels
    invalid_labels(1) = 999
    call model%fit(x, labels, status, options, validation_x=validation_x, &
        validation_labels=invalid_labels, validation_weight=validation_weight)
    refused = .not. status_ok(status)
    call model%predict_proba(validation_x, after, status)
    write (*, '(a,1x,i0)') "lightgbm_mc_invalid_status", merge(1, 0, refused)
    write (*, '(a,1x,es24.16)') "lightgbm_mc_transaction_error", maxval(abs(after-before))

contains

    subroutine make_fixture(x, labels)
        real(dp), intent(out) :: x(:, :)
        integer, intent(out) :: labels(:)

        x(:, 1) = [-4.0_dp, -3.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, &
            2.0_dp, 3.0_dp, 4.0_dp]
        labels = [-8, -8, -8, 2, 2, 2, 11, 11, 11]
    end subroutine make_fixture

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

end program fortml_bench_lightgbm_multiclass
