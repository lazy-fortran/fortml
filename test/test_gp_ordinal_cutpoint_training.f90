program test_gp_ordinal_cutpoint_training
    !! Independent fixed-latent cut-point products, training, and rollback oracle.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_ordinal_classification, only: gp_ordinal_classification_t, &
        gp_ordinal_classification_options_t, GP_ORDINAL_LIKELIHOOD_PROBIT
    use fortml_gp_ordinal_cutpoint_training, only: gp_ordinal_cutpoint_options_t, &
        gp_ordinal_cutpoint_result_t, gp_ordinal_cutpoint_value_gradient, &
        gp_ordinal_cutpoint_hvp, gp_ordinal_cutpoint_value_gradient_device, &
        gp_ordinal_cutpoint_device_supported, gp_ordinal_optimize_cutpoints
    implicit none

    integer, parameter :: n = 15
    type(gp_ordinal_classification_t) :: model, model_plus, model_minus, rollback_model
    type(gp_ordinal_classification_options_t) :: fit_options
    type(gp_ordinal_cutpoint_options_t) :: train_options, invalid_options, short_options
    type(gp_ordinal_cutpoint_result_t) :: result
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(n, 1), query(5, 1), weights(n), thresholds(2), thresholds_before(2)
    real(dp) :: direction(2), gradient(2), gradient_plus(2), gradient_minus(2), product(2)
    real(dp) :: probabilities(5, 3), probabilities_dot(5, 3)
    real(dp) :: probabilities_plus(5, 3), probabilities_minus(5, 3)
    real(dp) :: probabilities_bar(5, 3), threshold_bar(2)
    real(dp) :: log_probabilities(5, 3), log_probabilities_dot(5, 3)
    real(dp) :: log_plus(5, 3), log_minus(5, 3), log_bar(5, 3)
    real(dp) :: value, value_plus, value_minus, oracle, h, lhs, rhs
    integer :: labels(n), invalid_labels(n), failures, i

    do i = 1, n
        x(i, 1) = -1.75_dp + 3.5_dp*real(i - 1, dp)/real(n - 1, dp)
        if (x(i, 1) < -0.65_dp) then
            labels(i) = -4
        else if (x(i, 1) < 0.45_dp) then
            labels(i) = 7
        else
            labels(i) = 19
        end if
        weights(i) = 0.7_dp + 0.11_dp*real(mod(3*i, 7), dp)
    end do
    query(:, 1) = [-1.3_dp, -0.45_dp, 0.1_dp, 0.7_dp, 1.35_dp]
    thresholds = [1.2_dp, 2.75_dp]
    direction = [0.17_dp, -0.09_dp]
    h = 2.0e-6_dp
    failures = 0

    kernel = make_rbf_kernel(1, 1.25_dp, 0.78_dp, status)
    fit_options%noise_variance = 0.08_dp
    fit_options%jitter = 1.0e-8_dp
    call model%fit(x, labels, kernel, status, fit_options)
    call check(status_ok(status), "model fit", failures)
    call model%set_thresholds(thresholds, status)
    call check(status_ok(status), "threshold setter", failures)

    call gp_ordinal_cutpoint_value_gradient(model, x, labels, thresholds, &
        GP_ORDINAL_LIKELIHOOD_PROBIT, value, gradient, status, weights)
    call check(status_ok(status), "weighted objective", failures)
    oracle = independent_objective(model, x, labels, thresholds, weights)
    call check(abs(value - oracle) < 3.0e-13_dp, "independent objective value", failures)
    do i = 1, 2
        call objective_at_offset(model, x, labels, thresholds, weights, i, h, &
            value_plus, value_minus, status)
        call check(status_ok(status), "objective finite-difference status", failures)
        call check(abs(gradient(i) - (value_plus - value_minus)/(2.0_dp*h)) < 4.0e-8_dp, &
            "objective gradient finite difference", failures)
    end do
    call gp_ordinal_cutpoint_hvp(model, x, labels, thresholds, &
        GP_ORDINAL_LIKELIHOOD_PROBIT, direction, value, gradient, product, status, weights)
    call check(status_ok(status), "objective HVP", failures)
    call gp_ordinal_cutpoint_value_gradient(model, x, labels, thresholds+h*direction, &
        GP_ORDINAL_LIKELIHOOD_PROBIT, value_plus, gradient_plus, status, weights)
    call gp_ordinal_cutpoint_value_gradient(model, x, labels, thresholds-h*direction, &
        GP_ORDINAL_LIKELIHOOD_PROBIT, value_minus, gradient_minus, status, weights)
    call check(maxval(abs(product - (gradient_plus - gradient_minus)/(2.0_dp*h))) < &
        2.0e-7_dp, "objective HVP finite difference", failures)

    call model%predict_proba_threshold_jvp(query, direction, probabilities, &
        probabilities_dot, status)
    call check(status_ok(status), "probability threshold JVP", failures)
    model_plus = model
    model_minus = model
    call model_plus%set_thresholds(thresholds+h*direction, status)
    call model_minus%set_thresholds(thresholds-h*direction, status)
    call model_plus%predict_proba(query, probabilities_plus, status)
    call model_minus%predict_proba(query, probabilities_minus, status)
    call check(maxval(abs(probabilities_dot - (probabilities_plus - probabilities_minus)/ &
        (2.0_dp*h))) < 2.0e-9_dp, "probability threshold JVP finite difference", failures)
    probabilities_bar = reshape([ &
        0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp, 0.1_dp, &
        -0.3_dp, 0.4_dp, -0.2_dp, 0.1_dp, 0.5_dp, &
        0.1_dp, -0.2_dp, 0.4_dp, 0.2_dp, -0.4_dp], shape(probabilities_bar))
    call model%predict_proba_threshold_vjp(query, probabilities_bar, threshold_bar, status)
    lhs = dot_product(threshold_bar, direction)
    rhs = sum(probabilities_bar*probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-12_dp, &
        "probability threshold adjoint", failures)

    call model%predict_log_proba_threshold_jvp(query, direction, log_probabilities, &
        log_probabilities_dot, status)
    call model_plus%predict_log_proba(query, log_plus, status)
    call model_minus%predict_log_proba(query, log_minus, status)
    call check(maxval(abs(log_probabilities_dot - (log_plus - log_minus)/(2.0_dp*h))) < &
        2.0e-8_dp, "log-probability threshold JVP finite difference", failures)
    log_bar = 0.3_dp*probabilities_bar
    call model%predict_log_proba_threshold_vjp(query, log_bar, threshold_bar, status)
    lhs = dot_product(threshold_bar, direction)
    rhs = sum(log_bar*log_probabilities_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-11_dp, &
        "log-probability threshold adjoint", failures)

    thresholds_before = model%thresholds()
    call model%set_thresholds([2.4_dp, 1.1_dp], status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid threshold refusal", failures)
    call check(maxval(abs(model%thresholds() - thresholds_before)) == 0.0_dp, &
        "invalid threshold rollback", failures)

    train_options%likelihood = GP_ORDINAL_LIKELIHOOD_PROBIT
    train_options%max_iterations = 160
    train_options%max_line_search = 50
    train_options%gradient_tolerance = 2.0e-7_dp
    train_options%location_lower = -1.0_dp
    train_options%location_upper = 4.0_dp
    train_options%log_gap_lower = -4.0_dp
    train_options%log_gap_upper = 2.0_dp
    call gp_ordinal_optimize_cutpoints(model, x, labels, train_options, result, status, &
        sample_weight=weights)
    call check(status_ok(status) .and. result%converged, &
        "bounded cut-point convergence", failures)
    call check(result%negative_log_likelihood < result%initial_negative_log_likelihood .and. &
        result%gradient_norm < 2.0e-5_dp, "objective improvement and stationarity", failures)
    thresholds = model%thresholds()
    call check(thresholds(2) > thresholds(1) + train_options%minimum_gap, &
        "strict transformed gap", failures)

    rollback_model = model
    thresholds_before = rollback_model%thresholds()
    invalid_options = train_options
    invalid_options%location_lower = 2.0_dp
    invalid_options%location_upper = 1.0_dp
    call gp_ordinal_optimize_cutpoints(rollback_model, x, labels, invalid_options, &
        result, status, sample_weight=weights)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "invalid options refusal", failures)
    call check(maxval(abs(rollback_model%thresholds() - thresholds_before)) == 0.0_dp, &
        "invalid options rollback", failures)
    short_options = train_options
    short_options%max_iterations = 1
    short_options%gradient_tolerance = 0.0_dp
    call rollback_model%set_thresholds([0.4_dp, 3.6_dp], status)
    thresholds_before = rollback_model%thresholds()
    call gp_ordinal_optimize_cutpoints(rollback_model, x, labels, short_options, &
        result, status, sample_weight=weights)
    call check(status%code == FORTNUM_CONVERGENCE_ERROR, &
        "nonconvergence refusal", failures)
    call check(maxval(abs(rollback_model%thresholds() - thresholds_before)) == 0.0_dp, &
        "nonconvergence rollback", failures)
    invalid_labels = labels
    invalid_labels(1) = 12345
    call gp_ordinal_optimize_cutpoints(rollback_model, x, invalid_labels, train_options, &
        result, status, sample_weight=weights)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "unknown-label refusal", failures)
    call check(maxval(abs(rollback_model%thresholds() - thresholds_before)) == 0.0_dp, &
        "unknown-label rollback", failures)

    call check(gp_ordinal_cutpoint_device_supported(FORTML_DEVICE_CPU), &
        "CPU capability", failures)
    call check(.not. gp_ordinal_cutpoint_device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability refusal", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    gradient = 9.0_dp
    call gp_ordinal_cutpoint_value_gradient_device(cuda, model, x, labels, &
        model%thresholds(), GP_ORDINAL_LIKELIHOOD_PROBIT, value, gradient, status, weights)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. value == 0.0_dp .and. &
        all(gradient == 0.0_dp), "CUDA objective refusal", failures)
    probabilities = 9.0_dp
    probabilities_dot = 8.0_dp
    call model%predict_proba_threshold_jvp_device(cuda, query, direction, &
        probabilities, probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(probabilities == 0.0_dp) .and. all(probabilities_dot == 0.0_dp), &
        "CUDA prediction-product refusal", failures)
    thresholds_before = model%thresholds()
    call gp_ordinal_optimize_cutpoints(model, x, labels, train_options, result, status, &
        cuda, weights)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA training refusal", failures)
    call check(maxval(abs(model%thresholds() - thresholds_before)) == 0.0_dp, &
        "CUDA training rollback", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL ordinal GP cut-point cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS ordinal GP cut-point independent products/training oracle"

contains

    subroutine objective_at_offset(fitted, inputs, targets, cuts, sample_weights, &
            coordinate, step, plus, minus, objective_status)
        type(gp_ordinal_classification_t), intent(in) :: fitted
        real(dp), intent(in) :: inputs(:, :), cuts(:), sample_weights(:), step
        integer, intent(in) :: targets(:), coordinate
        real(dp), intent(out) :: plus, minus
        type(fortnum_status_t), intent(out) :: objective_status
        real(dp) :: cuts_plus(size(cuts)), cuts_minus(size(cuts)), scratch(size(cuts))

        cuts_plus = cuts
        cuts_minus = cuts
        cuts_plus(coordinate) = cuts_plus(coordinate) + step
        cuts_minus(coordinate) = cuts_minus(coordinate) - step
        call gp_ordinal_cutpoint_value_gradient(fitted, inputs, targets, cuts_plus, &
            GP_ORDINAL_LIKELIHOOD_PROBIT, plus, scratch, objective_status, sample_weights)
        if (.not. status_ok(objective_status)) return
        call gp_ordinal_cutpoint_value_gradient(fitted, inputs, targets, cuts_minus, &
            GP_ORDINAL_LIKELIHOOD_PROBIT, minus, scratch, objective_status, sample_weights)
    end subroutine objective_at_offset

    real(dp) function independent_objective(fitted, inputs, targets, cuts, &
            sample_weights) result(total)
        type(gp_ordinal_classification_t), intent(in) :: fitted
        real(dp), intent(in) :: inputs(:, :), cuts(:), sample_weights(:)
        integer, intent(in) :: targets(:)
        real(dp) :: eta(size(targets)), variance(size(targets)), upper, lower, mass
        integer, allocatable :: classes(:)
        integer :: i, rank
        type(fortnum_status_t) :: oracle_status

        call fitted%predict_latent(inputs, eta, variance, oracle_status)
        if (.not. status_ok(oracle_status)) error stop "independent latent oracle failed"
        classes = fitted%classes()
        mass = sum(sample_weights)
        total = 0.0_dp
        do i = 1, size(targets)
            rank = independent_rank(classes, targets(i))
            upper = 1.0_dp
            if (rank <= size(cuts)) upper = 0.5_dp*erfc(-(cuts(rank) - eta(i))/sqrt(2.0_dp))
            lower = 0.0_dp
            if (rank > 1) then
                lower = 0.5_dp*erfc(-(cuts(rank - 1) - eta(i))/sqrt(2.0_dp))
            end if
            total = total - sample_weights(i)*log(upper - lower)/mass
        end do
    end function independent_objective

    integer function independent_rank(classes, label) result(rank)
        integer, intent(in) :: classes(:), label
        integer :: j

        rank = 0
        do j = 1, size(classes)
            if (classes(j) == label) then
                rank = j
                return
            end if
        end do
    end function independent_rank

    subroutine check(condition, description, count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: count

        if (.not. condition) then
            count = count + 1
            write (error_unit, '(a)') "  FAIL [ordinal-cutpoint] "//description
        end if
    end subroutine check

end program test_gp_ordinal_cutpoint_training
