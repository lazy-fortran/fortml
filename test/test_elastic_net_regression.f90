program test_elastic_net_regression
    !! Independent one-feature oracle and derivative checks for elastic net.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_elastic_net_regression, only: elastic_net_regression_t
    implicit none

    type(elastic_net_regression_t) :: model
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 1), target(6), weights(6), prediction(6)
    real(dp) :: prediction_matrix(6, 1), y_dot_matrix(6, 1), target_matrix(6, 2)
    real(dp) :: theta_dot(2), x_dot(6, 1), y_dot(6), theta_bar(2), x_bar(6, 1)
    real(dp), allocatable :: coefficients(:, :), parameters(:), matrix_coefficients(:, :)
    real(dp) :: expected_beta, expected_beta2, expected_intercept, expected_intercept2
    real(dp) :: x_mean, y_mean, centered_covariance, centered_variance
    integer :: failures

    failures = 0
    x(:, 1) = [-2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
    target = [4.0_dp, 2.5_dp, 1.0_dp, -0.5_dp, -2.0_dp, -3.5_dp]
    weights = [1.0_dp, 2.0_dp, 1.0_dp, 3.0_dp, 2.0_dp, 1.0_dp]

    call model%fit(x, target, status, alpha=0.2_dp, l1_ratio=1.0_dp, &
        sample_weight=weights, max_iterations=5000, tolerance=1.0e-12_dp)
    call check(status_ok(status), "elastic-net weighted fit", failures)
    call check(model%fitted() .and. model%feature_count() == 1 .and. &
        model%output_count() == 1, "elastic-net fitted metadata", failures)

    x_mean = sum(weights*x(:, 1))/sum(weights)
    y_mean = sum(weights*target)/sum(weights)
    centered_covariance = sum(weights*(x(:, 1)-x_mean)*(target-y_mean))/sum(weights)
    centered_variance = sum(weights*(x(:, 1)-x_mean)**2)/sum(weights)
    expected_beta = soft_threshold_reference(centered_covariance, 0.2_dp)/centered_variance
    expected_intercept = y_mean-expected_beta*x_mean
    coefficients = model%coefficients()
    call check(abs(coefficients(1, 1)-expected_intercept) < 2.0e-10_dp, &
        "elastic-net intercept oracle", failures)
    call check(abs(coefficients(2, 1)-expected_beta) < 2.0e-10_dp, &
        "elastic-net lasso coefficient oracle", failures)

    call model%predict(x, prediction, status)
    call check(status_ok(status), "elastic-net prediction", failures)
    call check(maxval(abs(prediction-(expected_intercept+expected_beta*x(:, 1)))) < 2.0e-10_dp, &
        "elastic-net prediction oracle", failures)

    parameters = model%parameters()
    theta_dot = [0.13_dp, -0.21_dp]
    x_dot(:, 1) = [0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp, 0.1_dp, -0.4_dp]
    call model%jvp(x, theta_dot, x_dot, prediction_matrix, y_dot_matrix, status)
    y_dot = y_dot_matrix(:, 1)
    call check(status_ok(status), "elastic-net prediction JVP", failures)
    call check(maxval(abs(y_dot-(theta_dot(1)+theta_dot(2)*x(:, 1)+ &
        expected_beta*x_dot(:, 1)))) < 2.0e-12_dp, &
        "elastic-net JVP oracle", failures)
    call model%vjp(x, reshape([1.0_dp, -0.5_dp, 0.25_dp, 0.75_dp, -0.2_dp, 0.4_dp], [6, 1]), &
        theta_bar, x_bar, status)
    call check(status_ok(status), "elastic-net prediction VJP", failures)
    call check(maxval(abs(theta_bar-[sum([1.0_dp, -0.5_dp, 0.25_dp, 0.75_dp, -0.2_dp, 0.4_dp]), &
        sum([1.0_dp, -0.5_dp, 0.25_dp, 0.75_dp, -0.2_dp, 0.4_dp]*x(:, 1))])) < 2.0e-12_dp, &
        "elastic-net VJP parameter oracle", failures)
    call check(maxval(abs(x_bar(:, 1)-[1.0_dp, -0.5_dp, 0.25_dp, 0.75_dp, -0.2_dp, 0.4_dp]*expected_beta)) < &
        2.0e-12_dp, "elastic-net VJP input oracle", failures)

    target_matrix(:, 1) = target
    target_matrix(:, 2) = 2.0_dp*target+1.0_dp
    expected_beta2 = soft_threshold_reference(2.0_dp*centered_covariance, 0.2_dp)/ &
        centered_variance
    expected_intercept2 = 2.0_dp*y_mean+1.0_dp-expected_beta2*x_mean
    call model%fit(x, target_matrix, status, alpha=0.2_dp, l1_ratio=1.0_dp, &
        sample_weight=weights, max_iterations=5000, tolerance=1.0e-12_dp)
    call check(status_ok(status), "elastic-net multi-output fit", failures)
    matrix_coefficients = model%coefficients()
    call check(all(shape(matrix_coefficients) == [2, 2]), &
        "elastic-net multi-output coefficient shape", failures)
    call check(abs(matrix_coefficients(1, 1)-expected_intercept) < 2.0e-10_dp .and. &
        abs(matrix_coefficients(2, 1)-expected_beta) < 2.0e-10_dp .and. &
        abs(matrix_coefficients(1, 2)-expected_intercept2) < 2.0e-10_dp .and. &
        abs(matrix_coefficients(2, 2)-expected_beta2) < 2.0e-10_dp, &
        "elastic-net multi-output oracle", failures)

    call model%fit(x, target, status, alpha=0.2_dp, l1_ratio=1.2_dp)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "elastic-net invalid l1 ratio refusal", failures)
    weights(1) = -1.0_dp
    call model%fit(x, target, status, sample_weight=weights)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "elastic-net invalid weight refusal", failures)

    if (failures > 0) error stop 1
    write (*, '(a)') "PASS elastic-net regression independent behavioral oracles"

contains

    pure real(dp) function soft_threshold_reference(value, threshold) result(output)
        real(dp), intent(in) :: value, threshold

        if (value > threshold) then
            output = value-threshold
        else if (value < -threshold) then
            output = value+threshold
        else
            output = 0.0_dp
        end if
    end function soft_threshold_reference

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures+1
            write (*, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

end program test_elastic_net_regression
