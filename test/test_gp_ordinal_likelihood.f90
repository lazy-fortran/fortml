program test_gp_ordinal_likelihood
    !! Independent ordered-logit/probit likelihood and derivative oracle.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_gp_ordinal_classification, only: &
        GP_ORDINAL_LIKELIHOOD_LOGISTIC, GP_ORDINAL_LIKELIHOOD_PROBIT, &
        gp_ordinal_log_likelihood_value, gp_ordinal_log_likelihood_jvp, &
        gp_ordinal_log_likelihood_vjp, gp_ordinal_log_likelihood_hvp, &
        gp_ordinal_likelihood_device_supported
    implicit none

    real(dp) :: eta(5), eta_dot(5), thresholds(2), thresholds_dot(2)
    real(dp) :: eta_bar(5), thresholds_bar(2), eta_hvp(5), thresholds_hvp(2)
    real(dp) :: eta_plus(5), eta_minus(5), thresholds_plus(2), thresholds_minus(2)
    real(dp) :: eta_bar_plus(5), eta_bar_minus(5), threshold_bar_plus(2), threshold_bar_minus(2)
    real(dp) :: value, value_dot, value_plus, value_minus, oracle, h
    real(dp) :: lhs, rhs, value_bar
    integer :: labels(5), likelihood, failures
    type(fortnum_status_t) :: status

    eta = [-1.2_dp, -0.15_dp, 0.35_dp, 1.1_dp, 0.2_dp]
    eta_dot = [0.17_dp, -0.11_dp, 0.08_dp, 0.13_dp, -0.09_dp]
    thresholds = [-0.45_dp, 0.8_dp]
    thresholds_dot = [0.06_dp, -0.04_dp]
    labels = [1, 2, 3, 2, 1]
    value_bar = 1.7_dp
    failures = 0
    h = 2.0e-6_dp

    do likelihood = GP_ORDINAL_LIKELIHOOD_LOGISTIC, GP_ORDINAL_LIKELIHOOD_PROBIT
        call gp_ordinal_log_likelihood_value(eta, labels, thresholds, likelihood, value, status)
        call check(status_ok(status), "likelihood value status", failures)
        oracle = independent_value(eta, labels, thresholds, likelihood)
        call check(abs(value - oracle) < 2.0e-13_dp, "independent value oracle", failures)

        call gp_ordinal_log_likelihood_jvp(eta, labels, thresholds, likelihood, eta_dot, &
            thresholds_dot, value, value_dot, status)
        call check(status_ok(status), "likelihood JVP status", failures)
        eta_plus = eta + h*eta_dot
        eta_minus = eta - h*eta_dot
        thresholds_plus = thresholds + h*thresholds_dot
        thresholds_minus = thresholds - h*thresholds_dot
        call gp_ordinal_log_likelihood_value(eta_plus, labels, thresholds_plus, likelihood, &
            value_plus, status)
        call gp_ordinal_log_likelihood_value(eta_minus, labels, thresholds_minus, likelihood, &
            value_minus, status)
        call check(abs(value_dot - (value_plus - value_minus)/(2.0_dp*h)) < 3.0e-8_dp, &
            "likelihood JVP finite difference", failures)

        call gp_ordinal_log_likelihood_vjp(eta, labels, thresholds, likelihood, value_bar, &
            eta_bar, thresholds_bar, status)
        call check(status_ok(status), "likelihood VJP status", failures)
        lhs = sum(eta_bar*eta_dot) + dot_product(thresholds_bar, thresholds_dot)
        rhs = value_bar*value_dot
        call check(abs(lhs - rhs) < 3.0e-12_dp, "likelihood VJP/JVP adjoint", failures)

        call gp_ordinal_log_likelihood_hvp(eta, labels, thresholds, likelihood, value_bar, &
            eta_dot, thresholds_dot, eta_hvp, thresholds_hvp, status)
        call check(status_ok(status), "likelihood HVP status", failures)
        call gp_ordinal_log_likelihood_vjp(eta_plus, labels, thresholds_plus, likelihood, &
            value_bar, eta_bar_plus, threshold_bar_plus, status)
        call gp_ordinal_log_likelihood_vjp(eta_minus, labels, thresholds_minus, likelihood, &
            value_bar, eta_bar_minus, threshold_bar_minus, status)
        call check(maxval(abs(eta_hvp - (eta_bar_plus - eta_bar_minus)/(2.0_dp*h))) < 2.0e-7_dp, &
            "likelihood eta HVP finite difference", failures)
        call check(maxval(abs(thresholds_hvp - (threshold_bar_plus - threshold_bar_minus)/ &
            (2.0_dp*h))) < 2.0e-7_dp, "likelihood threshold HVP finite difference", failures)
    end do

    eta_bar = 9.0_dp
    thresholds_bar = 8.0_dp
    call gp_ordinal_log_likelihood_vjp(eta, labels, [0.8_dp, -0.45_dp], &
        GP_ORDINAL_LIKELIHOOD_LOGISTIC, value_bar, eta_bar, thresholds_bar, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. all(eta_bar == 0.0_dp) .and. &
        all(thresholds_bar == 0.0_dp), "invalid threshold transaction", failures)
    call check(gp_ordinal_likelihood_device_supported(FORTML_DEVICE_CPU), &
        "CPU likelihood capability", failures)
    call check(.not. gp_ordinal_likelihood_device_supported(FORTML_DEVICE_CUDA), &
        "typed CUDA likelihood capability refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL ordinal likelihood cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS ordinal likelihood independent value/JVP/VJP/HVP oracle"

contains

    real(dp) function independent_value(scores, ranks, cuts, kind) result(total)
        real(dp), intent(in) :: scores(:), cuts(:)
        integer, intent(in) :: ranks(:), kind
        real(dp) :: upper, lower
        integer :: i

        total = 0.0_dp
        do i = 1, size(scores)
            if (ranks(i) <= size(cuts)) then
                upper = independent_cdf(cuts(ranks(i)) - scores(i), kind)
            else
                upper = 1.0_dp
            end if
            if (ranks(i) > 1) then
                lower = independent_cdf(cuts(ranks(i) - 1) - scores(i), kind)
            else
                lower = 0.0_dp
            end if
            total = total + log(upper - lower)
        end do
    end function independent_value

    real(dp) function independent_cdf(z, kind) result(value)
        real(dp), intent(in) :: z
        integer, intent(in) :: kind

        if (kind == GP_ORDINAL_LIKELIHOOD_LOGISTIC) then
            if (z >= 0.0_dp) then
                value = 1.0_dp/(1.0_dp + exp(-z))
            else
                value = exp(z)/(1.0_dp + exp(z))
            end if
        else
            value = 0.5_dp*erfc(-z/sqrt(2.0_dp))
        end if
    end function independent_cdf

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [ordinal-likelihood] "//description
        end if
    end subroutine check

end program test_gp_ordinal_likelihood
