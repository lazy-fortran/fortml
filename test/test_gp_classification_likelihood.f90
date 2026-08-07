program test_gp_classification_likelihood
    !! Independent scalar and finite-difference checks for GP likelihood products.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_gp_classification, only: &
        gp_classification_log_likelihood_value, &
        gp_classification_log_likelihood_jvp, &
        gp_classification_log_likelihood_vjp, &
        GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
    implicit none

    real(dp) :: eta(5), eta_dot(5), eta_plus(5), eta_minus(5), eta_bar(5)
    real(dp) :: value, value_dot, value_plus, value_minus, value_bar
    real(dp) :: expected, expected_dot, h
    type(fortnum_status_t) :: status
    integer :: failures

    eta = [-2.0_dp, -0.4_dp, 0.0_dp, 0.7_dp, 2.5_dp]
    eta_dot = [0.3_dp, -0.8_dp, 0.2_dp, 0.6_dp, -0.1_dp]
    eta_bar = 0.0_dp
    value_bar = 1.7_dp
    h = 1.0e-6_dp
    failures = 0

    call gp_classification_log_likelihood_value(eta, GP_LIKELIHOOD_LOGISTIC, &
        value, status)
    expected = sum(-log(1.0_dp + exp(-eta)))
    call check(status_ok(status) .and. abs(value - expected) < 2.0e-14_dp, &
        "logistic value", failures)

    call gp_classification_log_likelihood_jvp(eta, GP_LIKELIHOOD_LOGISTIC, &
        eta_dot, value, value_dot, status)
    expected_dot = sum((1.0_dp - 1.0_dp/(1.0_dp + exp(-eta)))*eta_dot)
    eta_plus = eta + h*eta_dot
    eta_minus = eta - h*eta_dot
    call gp_classification_log_likelihood_value(eta_plus, GP_LIKELIHOOD_LOGISTIC, &
        value_plus, status)
    call gp_classification_log_likelihood_value(eta_minus, GP_LIKELIHOOD_LOGISTIC, &
        value_minus, status)
    call check(status_ok(status) .and. abs(value_dot - expected_dot) < 2.0e-14_dp .and. &
        abs(value_dot - (value_plus - value_minus)/(2.0_dp*h)) < 2.0e-9_dp, &
        "logistic JVP oracle", failures)

    call gp_classification_log_likelihood_vjp(eta, GP_LIKELIHOOD_LOGISTIC, &
        value_bar, eta_bar, status)
    call check(status_ok(status) .and. abs(dot_product(eta_bar, eta_dot) - &
        value_bar*value_dot) < 2.0e-13_dp, "logistic VJP dot product", failures)

    call gp_classification_log_likelihood_value(eta, GP_LIKELIHOOD_PROBIT, &
        value, status)
    call check(status_ok(status) .and. value < 0.0_dp .and. value > -20.0_dp, &
        "probit value", failures)
    call gp_classification_log_likelihood_jvp(eta, GP_LIKELIHOOD_PROBIT, &
        eta_dot, value, value_dot, status)
    eta_plus = eta + h*eta_dot
    eta_minus = eta - h*eta_dot
    call gp_classification_log_likelihood_value(eta_plus, GP_LIKELIHOOD_PROBIT, &
        value_plus, status)
    call gp_classification_log_likelihood_value(eta_minus, GP_LIKELIHOOD_PROBIT, &
        value_minus, status)
    call check(status_ok(status) .and. abs(value_dot - &
        (value_plus - value_minus)/(2.0_dp*h)) < 2.0e-8_dp, &
        "probit JVP finite difference", failures)
    call gp_classification_log_likelihood_vjp(eta, GP_LIKELIHOOD_PROBIT, &
        value_bar, eta_bar, status)
    call check(status_ok(status) .and. abs(dot_product(eta_bar, eta_dot) - &
        value_bar*value_dot) < 2.0e-8_dp, "probit VJP dot product", failures)

    ! The negative probit tail must remain finite even when erfc underflows.
    eta = [-100.0_dp, -40.0_dp, 40.0_dp, 100.0_dp, 0.0_dp]
    call gp_classification_log_likelihood_value(eta, GP_LIKELIHOOD_PROBIT, &
        value, status)
    call check(status_ok(status) .and. ieee_is_finite(value), &
        "probit tail stability", failures)

    call gp_classification_log_likelihood_value(eta, 99, value, status)
    call check(.not. status_ok(status), "unsupported likelihood refusal", failures)
    call gp_classification_log_likelihood_jvp(eta, GP_LIKELIHOOD_LOGISTIC, &
        eta_dot(1:4), value, value_dot, status)
    call check(.not. status_ok(status), "tangent shape refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL GP classification likelihood cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS GP classification likelihood independent behavioral oracles"

contains

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [gp-likelihood] "//name
        end if
    end subroutine check

end program test_gp_classification_likelihood
