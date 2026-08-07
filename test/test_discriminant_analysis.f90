program test_discriminant_analysis
    !! Independent analytic and finite-difference checks for weighted LDA/QDA.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_discriminant_analysis, only: lda_classifier_t, qda_classifier_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(lda_classifier_t) :: lda, lda_unfitted
    type(qda_classifier_t) :: qda
    type(fortml_device_t) :: device
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 2), query(3, 2), query_dot(3, 2)
    real(dp) :: weights(8), lda_covariance(2, 2), qda_covariances(2, 2, 2)
    real(dp) :: means(2, 2), expected_means(2, 2), prior(2), counts(2)
    real(dp) :: log_probability(3, 2), log_plus(3, 2), log_minus(3, 2)
    real(dp) :: log_dot(3, 2), log_bar(3, 2), x_bar(3, 2)
    real(dp) :: probability(3, 2), probability_dot(3, 2)
    real(dp), allocatable :: theta(:), theta_dot(:), parameter_dot(:, :), parameter_plus(:, :)
    real(dp), allocatable :: parameter_bar(:), tangent_bar(:)
    real(dp) :: lhs, rhs, step
    integer :: labels(8), prediction(3), classes(2), failures
    integer :: i, c

    x(:, 1) = [ -2.0_dp, -1.2_dp, -1.7_dp, -0.8_dp, 1.2_dp, 1.8_dp, &
        1.0_dp, 2.1_dp ]
    x(:, 2) = [ -1.0_dp, -1.8_dp, -0.4_dp, -1.1_dp, 1.0_dp, 1.7_dp, &
        2.0_dp, 0.8_dp ]
    labels = [9, 9, 9, 9, -3, -3, -3, -3]
    weights = [1.0_dp, 2.0_dp, 1.0_dp, 1.5_dp, 1.0_dp, 2.0_dp, 1.5_dp, 1.0_dp]
    query(1, :) = [-1.0_dp, -1.0_dp]
    query(2, :) = [0.0_dp, 0.0_dp]
    query(3, :) = [1.2_dp, 1.2_dp]
    query_dot(1, :) = [0.1_dp, -0.2_dp]
    query_dot(2, :) = [-0.3_dp, 0.2_dp]
    query_dot(3, :) = [0.2_dp, 0.1_dp]
    failures = 0

    call lda%fit(x, labels, status, reg_param=0.02_dp, sample_weight=weights, &
        priors=[2.0_dp, 1.0_dp])
    call check(status_ok(status) .and. lda%fitted(), "weighted LDA fit", failures)
    classes = lda%classes()
    call check(all(classes == [-3, 9]) .and. lda%feature_count() == 2 .and. &
        lda%class_count() == 2, "sorted arbitrary LDA classes", failures)
    counts = lda%weighted_class_counts()
    call check(maxval(abs(counts - [5.5_dp, 5.5_dp])) < 2.0e-14_dp, &
        "weighted class counts", failures)
    prior = lda%class_prior()
    call check(maxval(abs(prior - [2.0_dp/3.0_dp, 1.0_dp/3.0_dp])) < 2.0e-14_dp, &
        "normalized explicit priors", failures)
    means = lda%means()
    expected_means(:, 1) = [ (1.0_dp*(-1.2_dp) + 2.0_dp*(-1.8_dp) + &
        1.5_dp*(-1.1_dp) + 1.0_dp*(-0.4_dp))/5.5_dp, &
        (1.0_dp*(-1.8_dp) + 2.0_dp*(-1.7_dp) + 1.5_dp*(-0.8_dp) + &
        1.0_dp*(-2.1_dp))/5.5_dp ]
    ! Rebuild the means independently from the source data, including labels.
    expected_means = 0.0_dp
    counts = 0.0_dp
    do i = 1, size(x, 1)
        c = merge(2, 1, labels(i) == 9)
        counts(c) = counts(c) + weights(i)
        expected_means(:, c) = expected_means(:, c) + weights(i)*x(i, :)
    end do
    do c = 1, 2
        expected_means(:, c) = expected_means(:, c)/counts(c)
    end do
    call check(maxval(abs(means - expected_means)) < 2.0e-14_dp, &
        "weighted means", failures)
    lda_covariance = lda%covariance()
    call check(abs(lda_covariance(1, 2) - lda_covariance(2, 1)) < 2.0e-14_dp .and. &
        minval([(lda_covariance(i, i), i=1,2)]) > 0.0_dp, &
        "pooled SPD covariance", failures)
    call lda%predict_log_proba(query, log_probability, status)
    call lda%predict_proba(query, probability, status)
    call lda%predict(query, prediction, status)
    call check(status_ok(status) .and. all(prediction == [9, 9, -3]) .and. &
        maxval(abs(sum(probability, dim=2) - 1.0_dp)) < 2.0e-14_dp .and. &
        maxval(abs(exp(log_probability) - probability)) < 2.0e-14_dp, &
        "stable LDA probabilities and prediction", failures)

    call lda%predict_log_proba_jvp(query, query_dot, log_probability, log_dot, status)
    step = 1.0e-6_dp
    call lda%predict_log_proba(query + step*query_dot, log_plus, status)
    call lda%predict_log_proba(query - step*query_dot, log_minus, status)
    call check(status_ok(status) .and. maxval(abs(log_dot - &
        (log_plus - log_minus)/(2.0_dp*step))) < 3.0e-7_dp, &
        "LDA input JVP finite difference", failures)
    log_bar = reshape([0.2_dp, -0.1_dp, 0.3_dp, -0.4_dp, 0.5_dp, -0.2_dp], &
        shape(log_bar))
    call lda%predict_log_proba_vjp(query, log_bar, x_bar, status)
    lhs = sum(log_bar*log_dot)
    rhs = sum(x_bar*query_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 3.0e-7_dp, &
        "LDA input VJP adjoint", failures)
    theta = lda%parameters()
    call check(size(theta) == lda%parameter_count() .and. size(theta) == 9, &
        "LDA packed parameter metadata", failures)
    allocate(theta_dot(size(theta)), parameter_dot(3, 2), parameter_plus(3, 2), &
        parameter_bar(size(theta)), tangent_bar(size(theta)))
    theta_dot = 0.0_dp
    theta_dot(1:4) = [0.01_dp, -0.02_dp, 0.03_dp, -0.01_dp]
    theta_dot(5:7) = [0.02_dp, 0.003_dp, -0.01_dp]
    theta_dot(8:9) = [0.01_dp, -0.01_dp]
    call lda%predict_log_proba_parameter_jvp(query, theta_dot, log_probability, &
        parameter_dot, status)
    call lda%set_parameters(theta + step*theta_dot, status)
    call lda%predict_log_proba(query, log_plus, status)
    call lda%set_parameters(theta - step*theta_dot, status)
    call lda%predict_log_proba(query, log_minus, status)
    call lda%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(parameter_dot - &
        (log_plus-log_minus)/(2.0_dp*step))) < 5.0e-7_dp, &
        "LDA parameter JVP finite difference", failures)
    parameter_bar = [(0.01_dp*real(i, dp), i=1,size(theta))]
    call lda%predict_log_proba_parameter_vjp(query, log_bar, tangent_bar, status)
    lhs = sum(log_bar*parameter_dot)
    rhs = sum(tangent_bar*theta_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 5.0e-7_dp, &
        "LDA parameter VJP adjoint", failures)

    call qda%fit(x, labels, status, reg_param=0.02_dp, sample_weight=weights)
    call check(status_ok(status) .and. qda%fitted(), "weighted QDA fit", failures)
    qda_covariances = qda%covariances()
    do c = 1, 2
        call check(abs(qda_covariances(1, 2, c) - qda_covariances(2, 1, c)) < &
            2.0e-14_dp .and. minval([(qda_covariances(i, i, c), i=1,2)]) > 0.0_dp, &
            "QDA SPD covariance", failures)
    end do
    call qda%predict_proba(query, probability, status)
    call qda%predict(query, prediction, status)
    call check(status_ok(status) .and. maxval(abs(sum(probability, dim=2) - 1.0_dp)) < &
        2.0e-14_dp .and. all(prediction == [9, 9, -3]), &
        "stable QDA probabilities and prediction", failures)
    call qda%predict_proba_jvp(query, query_dot, probability, probability_dot, status)
    call qda%predict_proba(query + step*query_dot, log_plus, status)
    call qda%predict_proba(query - step*query_dot, log_minus, status)
    call check(status_ok(status) .and. maxval(abs(probability_dot - &
        (log_plus-log_minus)/(2.0_dp*step))) < 4.0e-7_dp, &
        "QDA input probability JVP finite difference", failures)
    theta = qda%parameters()
    call check(size(theta) == qda%parameter_count() .and. size(theta) == 12, &
        "QDA packed parameter metadata", failures)
    call qda%set_parameters(theta, status)

    device%kind = FORTML_DEVICE_CUDA
    device%selected = .true.
    device%available = .true.
    call lda%predict_proba_device(device, query, probability, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "LDA CUDA refusal", failures)
    call qda%predict_device(device, query, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "QDA CUDA refusal", failures)
    call check(.not. lda%device_supported(FORTML_DEVICE_CUDA) .and. &
        .not. qda%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability contract", failures)
    call lda_unfitted%predict_proba(query, probability, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "unfitted refusal", failures)
    call lda%fit(x, labels, status, reg_param=-0.1_dp)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "negative regularization refusal", failures)
    call lda%fit(x, labels, status, sample_weight=[1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
        0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp])
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero class mass refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL discriminant-analysis cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS weighted LDA/QDA independent analytic oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [discriminant] "//name
        end if
    end subroutine check

end program test_discriminant_analysis
