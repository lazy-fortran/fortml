program test_linear_classifier_products
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_logistic_regression, only: logistic_regression_t
    use fortml_softmax_regression, only: softmax_regression_t
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_logistic_products(failures)
    call test_softmax_products(failures)
    if (failures /= 0) then
        write (*, '(a,i0)') "FAIL linear classifier product cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS linear classifier independent JVP/VJP oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

    subroutine test_logistic_products(failures)
        integer, intent(inout) :: failures
        type(logistic_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), x_dot(4, 2), x_plus(4, 2), x_minus(4, 2)
        real(dp) :: scores(4), scores_dot(4), scores_plus(4), scores_minus(4)
        real(dp) :: probabilities(4, 2), probabilities_dot(4, 2)
        real(dp) :: probabilities_plus(4, 2), probabilities_minus(4, 2)
        real(dp) :: probabilities_bar(4, 2), scores_bar(4)
        real(dp) :: theta(3), direction(3), theta_bar(3), x_bar(4, 2)
        real(dp) :: theta_probability_bar(3), x_probability_bar(4, 2)
        real(dp) :: h, lhs, rhs
        integer :: labels(4)

        x = reshape([0.2_dp, -0.5_dp, 1.1_dp, -1.3_dp, &
            -0.8_dp, 0.4_dp, 0.7_dp, 1.6_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.5_dp, -0.4_dp, &
            0.7_dp, -0.1_dp, 0.2_dp, 0.6_dp], shape(x_dot))
        labels = [-4, -4, 9, 9]
        h = 1.0e-6_dp

        call model%fit(x, labels, status, l2=0.2_dp, max_iterations=1000, &
            tolerance=1.0e-8_dp)
        call check(status_ok(status), "logistic product setup", failures)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        direction = [0.13_dp, -0.21_dp, 0.17_dp]
        call check(model%parameter_count() == 3, &
            "logistic packed parameter count", failures)
        call model%decision_function_jvp(x, direction, x_dot, scores, scores_dot, &
            status)
        call check(status_ok(status), "logistic decision JVP", failures)
        call model%predict_proba_jvp(x, direction, x_dot, probabilities, &
            probabilities_dot, status)
        call check(status_ok(status), "logistic probability JVP", failures)

        x_plus = x + h*x_dot
        x_minus = x - h*x_dot
        call model%set_parameters(theta + h*direction, status)
        call model%decision_function(x_plus, scores_plus, status)
        call model%predict_proba(x_plus, probabilities_plus, status)
        call model%set_parameters(theta - h*direction, status)
        call model%decision_function(x_minus, scores_minus, status)
        call model%predict_proba(x_minus, probabilities_minus, status)
        call model%set_parameters(theta, status)
        call check(maxval(abs(scores_dot - (scores_plus - scores_minus)/(2.0_dp*h))) &
            < 2.0e-9_dp, "logistic decision JVP finite difference", failures)
        call check(maxval(abs(probabilities_dot - &
            (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 2.0e-9_dp, &
            "logistic probability JVP finite difference", failures)

        scores_bar = [0.2_dp, -0.4_dp, 0.7_dp, -0.1_dp]
        call model%decision_function_vjp(x, scores_bar, theta_bar, x_bar, status)
        lhs = sum(scores_bar*scores_dot)
        rhs = sum(direction*theta_bar) + sum(x_dot*x_bar)
        call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-12_dp, &
            "logistic decision VJP adjoint identity", failures)

        probabilities_bar = reshape([0.2_dp, -0.1_dp, -0.3_dp, 0.6_dp, &
            0.4_dp, 0.5_dp, -0.2_dp, 0.8_dp], shape(probabilities_bar))
        call model%predict_proba_vjp(x, probabilities_bar, theta_probability_bar, &
            x_probability_bar, status)
        lhs = sum(probabilities_bar*probabilities_dot)
        rhs = sum(direction*theta_probability_bar) + sum(x_dot*x_probability_bar)
        call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-12_dp, &
            "logistic probability VJP adjoint identity", failures)

        x_dot(1, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
        call model%decision_function_jvp(x, direction, x_dot, scores, scores_dot, &
            status)
        call check(.not. status_ok(status), "logistic nonfinite tangent refusal", failures)
    end subroutine test_logistic_products

    subroutine test_softmax_products(failures)
        integer, intent(inout) :: failures
        type(softmax_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 2), x_dot(6, 2), x_plus(6, 2), x_minus(6, 2)
        real(dp) :: scores(6, 3), scores_dot(6, 3), scores_plus(6, 3), scores_minus(6, 3)
        real(dp) :: probabilities(6, 3), probabilities_dot(6, 3)
        real(dp) :: probabilities_plus(6, 3), probabilities_minus(6, 3)
        real(dp) :: probabilities_bar(6, 3), scores_bar(6, 3)
        real(dp) :: theta(9), direction(9), theta_bar(9), x_bar(6, 2)
        real(dp) :: theta_probability_bar(9), x_probability_bar(6, 2)
        real(dp) :: h, lhs, rhs
        integer :: labels(6)

        x = reshape([2.0_dp, 1.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 0.0_dp, &
            0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 2.0_dp, 1.0_dp], shape(x))
        x_dot = reshape([-0.2_dp, 0.3_dp, 0.1_dp, -0.4_dp, 0.5_dp, -0.6_dp, &
            0.7_dp, -0.8_dp, 0.2_dp, 0.1_dp, -0.3_dp, 0.4_dp], shape(x_dot))
        labels = [-2, -2, 7, 7, 42, 42]
        h = 1.0e-6_dp

        call model%fit(x, labels, status, l2=0.1_dp, max_iterations=1000, &
            tolerance=1.0e-8_dp)
        call check(status_ok(status), "softmax product setup", failures)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        direction = [0.13_dp, -0.21_dp, 0.17_dp, -0.09_dp, 0.05_dp, -0.11_dp, &
            0.07_dp, -0.03_dp, 0.19_dp]
        call check(model%parameter_count() == 9, &
            "softmax packed parameter count", failures)
        call model%decision_function_jvp(x, direction, x_dot, scores, scores_dot, &
            status)
        call check(status_ok(status), "softmax decision JVP", failures)
        call model%predict_proba_jvp(x, direction, x_dot, probabilities, &
            probabilities_dot, status)
        call check(status_ok(status), "softmax probability JVP", failures)

        x_plus = x + h*x_dot
        x_minus = x - h*x_dot
        call model%set_parameters(theta + h*direction, status)
        call model%decision_function(x_plus, scores_plus, status)
        call model%predict_proba(x_plus, probabilities_plus, status)
        call model%set_parameters(theta - h*direction, status)
        call model%decision_function(x_minus, scores_minus, status)
        call model%predict_proba(x_minus, probabilities_minus, status)
        call model%set_parameters(theta, status)
        call check(maxval(abs(scores_dot - (scores_plus - scores_minus)/(2.0_dp*h))) &
            < 2.0e-8_dp, "softmax decision JVP finite difference", failures)
        call check(maxval(abs(probabilities_dot - &
            (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 2.0e-8_dp, &
            "softmax probability JVP finite difference", failures)

        scores_bar = reshape([0.2_dp, -0.1_dp, 0.3_dp, -0.4_dp, 0.5_dp, -0.6_dp, &
            0.7_dp, -0.8_dp, 0.9_dp, -0.2_dp, 0.4_dp, -0.3_dp, &
            0.6_dp, -0.5_dp, 0.1_dp, 0.3_dp, -0.7_dp, 0.2_dp], shape(scores_bar))
        call model%decision_function_vjp(x, scores_bar, theta_bar, x_bar, status)
        lhs = sum(scores_bar*scores_dot)
        rhs = sum(direction*theta_bar) + sum(x_dot*x_bar)
        call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-12_dp, &
            "softmax decision VJP adjoint identity", failures)

        probabilities_bar = reshape([0.2_dp, -0.1_dp, 0.3_dp, -0.4_dp, 0.5_dp, -0.6_dp, &
            0.7_dp, -0.8_dp, 0.9_dp, -0.2_dp, 0.4_dp, -0.3_dp, &
            0.6_dp, -0.5_dp, 0.1_dp, 0.3_dp, -0.7_dp, 0.2_dp], &
            shape(probabilities_bar))
        call model%predict_proba_vjp(x, probabilities_bar, theta_probability_bar, &
            x_probability_bar, status)
        lhs = sum(probabilities_bar*probabilities_dot)
        rhs = sum(direction*theta_probability_bar) + sum(x_dot*x_probability_bar)
        call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-12_dp, &
            "softmax probability VJP adjoint identity", failures)

        call model%decision_function_jvp(x, direction(:8), x_dot, scores, scores_dot, &
            status)
        call check(.not. status_ok(status), "softmax parameter tangent refusal", failures)
    end subroutine test_softmax_products

end program test_linear_classifier_products
