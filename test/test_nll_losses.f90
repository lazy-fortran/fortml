program test_nll_losses
    !! Independent finite-difference and adjoint oracles for neural NLLs.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_losses, only: gaussian_nll_value, gaussian_nll_jvp, &
        gaussian_nll_vjp, gaussian_nll_hvp, poisson_nll_value, poisson_nll_jvp, &
        poisson_nll_vjp, poisson_nll_hvp, poisson_count_nll_value, &
        LOSS_REDUCTION_SUM
    implicit none

    integer :: failures

    failures = 0
    call test_gaussian_nll(failures)
    call test_poisson_nll(failures)
    call test_nll_domain_refusals(failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL NLL loss cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS Gaussian/Poisson NLL independent behavioral oracles"

contains

    subroutine test_gaussian_nll(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: prediction(3, 2), targets(3, 2), log_variance(3, 2)
        real(dp) :: prediction_dot(3, 2), log_variance_dot(3, 2), weights(3)
        real(dp) :: prediction_bar(3, 2), log_variance_bar(3, 2)
        real(dp) :: prediction_hvp(3, 2), log_variance_hvp(3, 2)
        real(dp) :: value, value_dot, value_plus, value_minus, finite_dot
        real(dp) :: value_bar, lhs, rhs, expected, sum_value, h
        real(dp) :: grad_prediction_plus(3, 2), grad_prediction_minus(3, 2)
        real(dp) :: grad_variance_plus(3, 2), grad_variance_minus(3, 2)
        real(dp) :: expected_hvp, expected_variance_hvp
        integer :: i, j

        prediction = reshape([0.2_dp, -0.5_dp, 1.3_dp, 0.8_dp, 0.1_dp, -0.7_dp], &
            shape(prediction))
        targets = reshape([0.0_dp, -0.2_dp, 1.0_dp, 0.3_dp, 0.4_dp, -0.1_dp], &
            shape(targets))
        log_variance = reshape([-0.3_dp, 0.2_dp, -0.1_dp, 0.4_dp, 0.5_dp, -0.2_dp], &
            shape(log_variance))
        prediction_dot = reshape([0.3_dp, -0.4_dp, 0.2_dp, -0.1_dp, 0.5_dp, 0.6_dp], &
            shape(prediction_dot))
        log_variance_dot = reshape([-0.2_dp, 0.1_dp, 0.4_dp, 0.3_dp, -0.5_dp, 0.2_dp], &
            shape(log_variance_dot))
        weights = [1.0_dp, 2.0_dp, 4.0_dp]
        value_bar = -1.7_dp
        h = 2.0e-6_dp

        call gaussian_nll_value(prediction, targets, log_variance, value, status, weights)
        expected = 0.0_dp
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                expected = expected + weights(i)*reference_gaussian( &
                    prediction(i, j), targets(i, j), log_variance(i, j))
            end do
        end do
        expected = expected/sum(weights)
        call check(status_ok(status) .and. abs(value - expected) < 2.0e-14_dp, &
            "Gaussian NLL independent weighted value oracle", failures)

        call gaussian_nll_jvp(prediction, targets, log_variance, prediction_dot, &
            log_variance_dot, value, value_dot, status, weights)
        call gaussian_nll_value(prediction + h*prediction_dot, targets, &
            log_variance + h*log_variance_dot, value_plus, status, weights)
        call gaussian_nll_value(prediction - h*prediction_dot, targets, &
            log_variance - h*log_variance_dot, value_minus, status, weights)
        finite_dot = (value_plus - value_minus)/(2.0_dp*h)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 3.0e-9_dp, &
            "Gaussian NLL JVP finite difference", failures)

        call gaussian_nll_vjp(prediction, targets, log_variance, value_bar, &
            prediction_bar, log_variance_bar, status, weights)
        lhs = value_bar*value_dot
        rhs = sum(prediction_bar*prediction_dot) + &
            sum(log_variance_bar*log_variance_dot)
        call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-14_dp, &
            "Gaussian NLL VJP adjoint identity", failures)

        call gaussian_nll_hvp(prediction, targets, log_variance, prediction_dot, &
            log_variance_dot, prediction_hvp, log_variance_hvp, status, weights)
        call gaussian_nll_vjp(prediction + h*prediction_dot, targets, &
            log_variance + h*log_variance_dot, 1.0_dp, grad_prediction_plus, &
            grad_variance_plus, status, weights)
        call gaussian_nll_vjp(prediction - h*prediction_dot, targets, &
            log_variance - h*log_variance_dot, 1.0_dp, grad_prediction_minus, &
            grad_variance_minus, status, weights)
        expected_hvp = sum(((grad_prediction_plus - grad_prediction_minus)/(2.0_dp*h) - &
            prediction_hvp)**2)
        expected_variance_hvp = sum(((grad_variance_plus - grad_variance_minus)/ &
            (2.0_dp*h) - log_variance_hvp)**2)
        call check(status_ok(status) .and. expected_hvp < 2.0e-15_dp .and. &
            expected_variance_hvp < 2.0e-15_dp, &
            "Gaussian NLL HVP finite-difference oracle", failures)

        call gaussian_nll_value(prediction, targets, log_variance, sum_value, status, &
            weights, LOSS_REDUCTION_SUM)
        call check(status_ok(status) .and. abs(sum_value - value*sum(weights)) < &
            2.0e-14_dp, "Gaussian NLL sum/mean reduction relation", failures)
    end subroutine test_gaussian_nll

    subroutine test_poisson_nll(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: log_rate(3, 2), targets(3, 2), direction(3, 2), weights(3)
        real(dp) :: log_rate_bar(3, 2), log_rate_bar_plus(3, 2)
        real(dp) :: log_rate_bar_minus(3, 2), log_rate_hvp(3, 2)
        real(dp) :: value, value_dot, value_plus, value_minus, finite_dot
        real(dp) :: lhs, rhs, expected, sum_value, h
        integer :: i, j

        log_rate = reshape([-1.1_dp, 0.8_dp, 0.4_dp, -0.6_dp, 1.2_dp, 0.1_dp], &
            shape(log_rate))
        targets = reshape([0.0_dp, 2.0_dp, 1.0_dp, 3.0_dp, 4.0_dp, 0.5_dp], &
            shape(targets))
        direction = reshape([0.3_dp, -0.4_dp, 0.2_dp, 0.1_dp, -0.5_dp, 0.6_dp], &
            shape(direction))
        weights = [1.0_dp, 2.0_dp, 4.0_dp]
        h = 2.0e-6_dp

        call poisson_count_nll_value(log_rate, targets, value, status, weights)
        expected = 0.0_dp
        do j = 1, size(log_rate, 2)
            do i = 1, size(log_rate, 1)
                expected = expected + weights(i)*reference_poisson( &
                    log_rate(i, j), targets(i, j))
            end do
        end do
        expected = expected/sum(weights)
        call check(status_ok(status) .and. abs(value - expected) < 2.0e-14_dp, &
            "Poisson NLL independent weighted value oracle", failures)

        call poisson_nll_jvp(log_rate, targets, direction, value, value_dot, status, &
            weights)
        call poisson_nll_value(log_rate + h*direction, targets, value_plus, status, &
            weights)
        call poisson_nll_value(log_rate - h*direction, targets, value_minus, status, &
            weights)
        finite_dot = (value_plus - value_minus)/(2.0_dp*h)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 3.0e-9_dp, &
            "Poisson NLL JVP finite difference", failures)

        call poisson_nll_vjp(log_rate, targets, -0.8_dp, log_rate_bar, status, weights)
        lhs = -0.8_dp*value_dot
        rhs = sum(log_rate_bar*direction)
        call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-14_dp, &
            "Poisson NLL VJP adjoint identity", failures)

        call poisson_nll_hvp(log_rate, targets, direction, log_rate_hvp, status, weights)
        call poisson_nll_vjp(log_rate + h*direction, targets, 1.0_dp, log_rate_bar_plus, &
            status, weights)
        call poisson_nll_vjp(log_rate - h*direction, targets, 1.0_dp, log_rate_bar_minus, &
            status, weights)
        expected = sum(((log_rate_bar_plus - log_rate_bar_minus)/(2.0_dp*h) - &
            log_rate_hvp)**2)
        call check(status_ok(status) .and. expected < 2.0e-15_dp, &
            "Poisson NLL HVP finite-difference oracle", failures)

        call poisson_nll_value(log_rate, targets, sum_value, status, weights, &
            LOSS_REDUCTION_SUM)
        call check(status_ok(status) .and. abs(sum_value - value*sum(weights)) < &
            2.0e-14_dp, "Poisson NLL sum/mean reduction relation", failures)
    end subroutine test_poisson_nll

    subroutine test_nll_domain_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: prediction(1, 1), targets(1, 1), log_variance(1, 1)
        real(dp) :: direction(1, 1), value, value_dot
        real(dp) :: log_rate(1, 1), count(1, 1)

        prediction = 0.0_dp
        targets = 0.0_dp
        log_variance = -1000.0_dp
        direction = 1.0_dp
        call gaussian_nll_value(prediction, targets, log_variance, value, status)
        call check(.not. status_ok(status), "Gaussian NLL unsafe precision refusal", failures)
        call gaussian_nll_jvp(prediction, targets, log_variance, direction, direction, &
            value, value_dot, status)
        call check(.not. status_ok(status), "Gaussian NLL JVP unsafe precision refusal", failures)

        log_rate = 1000.0_dp
        count = 1.0_dp
        call poisson_nll_value(log_rate, count, value, status)
        call check(.not. status_ok(status), "Poisson NLL overflow refusal", failures)
        count = -1.0_dp
        log_rate = 0.0_dp
        call poisson_nll_value(log_rate, count, value, status)
        call check(.not. status_ok(status), "Poisson NLL negative-count refusal", failures)
    end subroutine test_nll_domain_refusals

    real(dp) function reference_gaussian(prediction, target, log_variance) result(value)
        real(dp), intent(in) :: prediction, target, log_variance
        real(dp) :: residual

        residual = prediction - target
        value = 0.5_dp*(log_variance + residual*residual*exp(-log_variance) + &
            log(2.0_dp*acos(-1.0_dp)))
    end function reference_gaussian

    real(dp) function reference_poisson(log_rate, target) result(value)
        real(dp), intent(in) :: log_rate, target

        value = exp(log_rate) - target*log_rate + log_gamma(target + 1.0_dp)
    end function reference_poisson

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [nll] "//trim(name)
        end if
    end subroutine check

end program test_nll_losses
