program test_losses
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_losses, only: sigmoid_value, sigmoid_jvp, sigmoid_vjp, &
        binary_cross_entropy_with_logits_value, &
        binary_cross_entropy_with_logits_jvp, &
        binary_cross_entropy_with_logits_vjp, softmax_value, softmax_jvp, &
        softmax_vjp, log_softmax_value, log_softmax_jvp, log_softmax_vjp, &
        softmax_cross_entropy_value, softmax_cross_entropy_jvp, &
        softmax_cross_entropy_vjp, huber_loss_value, huber_loss_jvp, &
        huber_loss_vjp, quantile_loss_value, quantile_loss_jvp, quantile_loss_vjp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_sigmoid(failures)
    call test_binary_cross_entropy(failures)
    call test_softmax(failures)
    call test_log_softmax(failures)
    call test_softmax_cross_entropy(failures)
    call test_huber_and_quantile(failures)
    call test_invalid_inputs(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " classification loss test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_sigmoid(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: extreme(5, 1), extreme_value(5, 1), expected(5, 1)
        real(dp) :: logits(2, 3), direction(2, 3), cotangent(2, 3)
        real(dp) :: value(2, 3), value_dot(2, 3), value_plus(2, 3)
        real(dp) :: value_minus(2, 3), logits_bar(2, 3), finite_dot(2, 3)
        real(dp) :: h, lhs, rhs

        extreme(:, 1) = [-1000.0_dp, -2.0_dp, 0.0_dp, 2.0_dp, 1000.0_dp]
        expected(:, 1) = [0.0_dp, 0.11920292202211756_dp, 0.5_dp, &
            0.88079707797788231_dp, 1.0_dp]
        call sigmoid_value(extreme, extreme_value, status)
        call check(status_ok(status) .and. &
            maxval(abs(extreme_value - expected)) < 2.0e-15_dp, &
            "sigmoid stable extreme values", failures)

        logits(1, :) = [-1.2_dp, 0.3_dp, 1.1_dp]
        logits(2, :) = [0.7_dp, -0.8_dp, 0.2_dp]
        direction(1, :) = [0.2_dp, -0.4_dp, 0.1_dp]
        direction(2, :) = [-0.3_dp, 0.5_dp, 0.6_dp]
        cotangent(1, :) = [0.7_dp, -0.2_dp, 0.4_dp]
        cotangent(2, :) = [-0.1_dp, 0.3_dp, -0.6_dp]
        h = 1.0e-6_dp
        call sigmoid_jvp(logits, direction, value, value_dot, status)
        call reference_sigmoid(logits + h*direction, value_plus)
        call reference_sigmoid(logits - h*direction, value_minus)
        finite_dot = (value_plus - value_minus)/(2.0_dp*h)
        call check(status_ok(status) .and. &
            maxval(abs(value_dot - finite_dot)) < 2.0e-10_dp, &
            "sigmoid JVP independent finite difference", failures)
        call sigmoid_vjp(logits, cotangent, logits_bar, status)
        lhs = sum(cotangent*value_dot)
        rhs = sum(logits_bar*direction)
        call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-14_dp, &
            "sigmoid VJP adjoint identity", failures)
    end subroutine test_sigmoid

    subroutine test_binary_cross_entropy(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: extreme(2, 2), extreme_targets(2, 2), extreme_loss
        real(dp) :: logits(2, 3), targets(2, 3), direction(2, 3)
        real(dp) :: logits_bar(2, 3), finite_gradient(2, 3)
        real(dp) :: value, value_dot, reference, finite_dot, h, value_bar

        extreme(1, :) = [-1000.0_dp, 1000.0_dp]
        extreme(2, :) = [-1000.0_dp, 1000.0_dp]
        extreme_targets(1, :) = [0.0_dp, 1.0_dp]
        extreme_targets(2, :) = [1.0_dp, 0.0_dp]
        call binary_cross_entropy_with_logits_value( &
            extreme, extreme_targets, extreme_loss, status)
        call check(status_ok(status) .and. ieee_is_finite(extreme_loss) .and. &
            abs(extreme_loss - 500.0_dp) < 1.0e-13_dp, &
            "binary cross entropy stable extreme value", failures)

        logits(1, :) = [-1.1_dp, 0.4_dp, 1.2_dp]
        logits(2, :) = [0.8_dp, -0.6_dp, 0.1_dp]
        targets(1, :) = [0.0_dp, 1.0_dp, 0.25_dp]
        targets(2, :) = [1.0_dp, 0.0_dp, 0.75_dp]
        direction(1, :) = [0.3_dp, -0.2_dp, 0.5_dp]
        direction(2, :) = [-0.4_dp, 0.1_dp, 0.6_dp]
        call binary_cross_entropy_with_logits_value(logits, targets, value, status)
        reference = reference_binary_cross_entropy(logits, targets)
        call check(status_ok(status) .and. abs(value - reference) < 3.0e-15_dp, &
            "binary cross entropy independent probability formula", failures)

        h = 1.0e-6_dp
        call binary_cross_entropy_with_logits_jvp( &
            logits, targets, direction, value, value_dot, status)
        finite_dot = (reference_binary_cross_entropy(logits + h*direction, targets) &
            - reference_binary_cross_entropy(logits - h*direction, targets))/ &
            (2.0_dp*h)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 2.0e-10_dp, &
            "binary cross entropy JVP independent finite difference", failures)

        value_bar = 0.7_dp
        call binary_cross_entropy_with_logits_vjp( &
            logits, targets, value_bar, logits_bar, status)
        call finite_binary_gradient(logits, targets, value_bar, h, finite_gradient)
        call check(status_ok(status) .and. &
            maxval(abs(logits_bar - finite_gradient)) < 2.0e-10_dp, &
            "binary cross entropy VJP independent full gradient", failures)
    end subroutine test_binary_cross_entropy

    subroutine test_softmax(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(2, 3), expected(2, 3), probabilities(2, 3)
        real(dp) :: direction(2, 3), cotangent(2, 3), probability_dot(2, 3)
        real(dp) :: plus(2, 3), minus(2, 3), finite_dot(2, 3)
        real(dp) :: logits_bar(2, 3), h, lhs, rhs

        logits(1, :) = [1.0_dp, 2.0_dp, 3.0_dp]
        logits(2, :) = [1000.0_dp, 1001.0_dp, 1002.0_dp]
        expected(1, :) = [0.09003057317038046_dp, 0.24472847105479764_dp, &
            0.66524095577482190_dp]
        expected(2, :) = expected(1, :)
        call softmax_value(logits, probabilities, status)
        call check(status_ok(status) .and. &
            maxval(abs(probabilities - expected)) < 2.0e-15_dp .and. &
            maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-15_dp, &
            "softmax stable shifted values", failures)

        logits(2, :) = [-0.5_dp, 0.2_dp, 1.1_dp]
        direction(1, :) = [0.2_dp, -0.4_dp, 0.1_dp]
        direction(2, :) = [-0.3_dp, 0.5_dp, 0.6_dp]
        cotangent(1, :) = [0.7_dp, -0.2_dp, 0.4_dp]
        cotangent(2, :) = [-0.1_dp, 0.3_dp, -0.6_dp]
        h = 1.0e-6_dp
        call softmax_jvp(logits, direction, probabilities, probability_dot, status)
        call reference_softmax(logits + h*direction, plus)
        call reference_softmax(logits - h*direction, minus)
        finite_dot = (plus - minus)/(2.0_dp*h)
        call check(status_ok(status) .and. &
            maxval(abs(probability_dot - finite_dot)) < 2.0e-10_dp, &
            "softmax JVP independent finite difference", failures)
        call softmax_vjp(logits, cotangent, logits_bar, status)
        lhs = sum(cotangent*probability_dot)
        rhs = sum(logits_bar*direction)
        call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-14_dp, &
            "softmax VJP adjoint identity", failures)
    end subroutine test_softmax

    subroutine test_log_softmax(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(2, 3), expected(2, 3), log_probabilities(2, 3)
        real(dp) :: direction(2, 3), cotangent(2, 3), log_probability_dot(2, 3)
        real(dp) :: plus(2, 3), minus(2, 3), finite_dot(2, 3)
        real(dp) :: logits_bar(2, 3), h, lhs, rhs

        logits(1, :) = [1.0_dp, 2.0_dp, 3.0_dp]
        logits(2, :) = [1000.0_dp, 1001.0_dp, 1002.0_dp]
        expected(1, :) = [-2.4076059644443801_dp, -1.4076059644443804_dp, &
            -0.40760596444438041_dp]
        expected(2, :) = expected(1, :)
        call log_softmax_value(logits, log_probabilities, status)
        call check(status_ok(status) .and. &
            maxval(abs(log_probabilities - expected)) < 2.0e-15_dp, &
            "log softmax stable shifted values", failures)

        logits(2, :) = [-0.5_dp, 0.2_dp, 1.1_dp]
        direction(1, :) = [0.2_dp, -0.4_dp, 0.1_dp]
        direction(2, :) = [-0.3_dp, 0.5_dp, 0.6_dp]
        cotangent(1, :) = [0.7_dp, -0.2_dp, 0.4_dp]
        cotangent(2, :) = [-0.1_dp, 0.3_dp, -0.6_dp]
        h = 1.0e-6_dp
        call log_softmax_jvp( &
            logits, direction, log_probabilities, log_probability_dot, status)
        call reference_log_softmax(logits + h*direction, plus)
        call reference_log_softmax(logits - h*direction, minus)
        finite_dot = (plus - minus)/(2.0_dp*h)
        call check(status_ok(status) .and. &
            maxval(abs(log_probability_dot - finite_dot)) < 3.0e-10_dp, &
            "log softmax JVP independent finite difference", failures)
        call log_softmax_vjp(logits, cotangent, logits_bar, status)
        lhs = sum(cotangent*log_probability_dot)
        rhs = sum(logits_bar*direction)
        call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-14_dp, &
            "log softmax VJP adjoint identity", failures)
    end subroutine test_log_softmax

    subroutine test_softmax_cross_entropy(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(2, 3), direction(2, 3), logits_bar(2, 3)
        real(dp) :: finite_gradient(2, 3), value, value_dot, finite_dot
        real(dp) :: h, value_bar
        integer :: labels(2)

        logits(1, :) = [1.0_dp, 2.0_dp, 3.0_dp]
        logits(2, :) = [1000.0_dp, 1001.0_dp, 1002.0_dp]
        labels = [3, 1]
        call softmax_cross_entropy_value(logits, labels, value, status)
        call check(status_ok(status) .and. &
            abs(value - 1.4076059644443804_dp) < 2.0e-13_dp, &
            "softmax cross entropy stable shifted value", failures)

        logits(2, :) = [-0.5_dp, 0.2_dp, 1.1_dp]
        direction(1, :) = [0.2_dp, -0.4_dp, 0.1_dp]
        direction(2, :) = [-0.3_dp, 0.5_dp, 0.6_dp]
        h = 1.0e-6_dp
        call softmax_cross_entropy_jvp( &
            logits, labels, direction, value, value_dot, status)
        finite_dot = (reference_cross_entropy(logits + h*direction, labels) - &
            reference_cross_entropy(logits - h*direction, labels))/(2.0_dp*h)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 2.0e-10_dp, &
            "softmax cross entropy JVP independent finite difference", failures)

        value_bar = -0.8_dp
        call softmax_cross_entropy_vjp( &
            logits, labels, value_bar, logits_bar, status)
        call finite_cross_entropy_gradient( &
            logits, labels, value_bar, h, finite_gradient)
        call check(status_ok(status) .and. &
            maxval(abs(logits_bar - finite_gradient)) < 2.0e-10_dp, &
            "softmax cross entropy VJP independent full gradient", failures)
    end subroutine test_softmax_cross_entropy

    subroutine test_invalid_inputs(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(2, 3), targets(2, 3), value
        integer :: labels(2)

        logits = 0.0_dp
        targets = 0.5_dp
        targets(1, 1) = -0.1_dp
        call binary_cross_entropy_with_logits_value(logits, targets, value, status)
        call check(.not. status_ok(status), &
            "binary cross entropy rejects invalid targets", failures)
        labels = [1, 0]
        call softmax_cross_entropy_value(logits, labels, value, status)
        call check(.not. status_ok(status), &
            "softmax cross entropy rejects invalid labels", failures)
    end subroutine test_invalid_inputs

    subroutine test_huber_and_quantile(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: prediction(2, 3), targets(2, 3), direction(2, 3)
        real(dp) :: cotangent(2, 3), prediction_bar(2, 3)
        real(dp) :: value, value_dot, value_plus, value_minus, finite_dot
        real(dp) :: h, lhs, rhs, reference, quantile

        prediction = reshape([0.2_dp, 1.4_dp, -0.5_dp, 2.0_dp, 0.8_dp, -1.3_dp], &
            shape(prediction))
        targets = reshape([0.0_dp, 1.0_dp, -1.0_dp, 0.5_dp, 1.4_dp, -0.2_dp], &
            shape(targets))
        direction = reshape([0.3_dp, -0.4_dp, 0.2_dp, -0.1_dp, 0.5_dp, 0.6_dp], &
            shape(direction))
        cotangent = reshape([0.7_dp, -0.2_dp, 0.4_dp, -0.3_dp, 0.5_dp, -0.6_dp], &
            shape(cotangent))
        h = 1.0e-6_dp
        call huber_loss_value(prediction, targets, 0.75_dp, value, status)
        reference = reference_huber(prediction, targets, 0.75_dp)
        call check(status_ok(status) .and. abs(value - reference) < 2.0e-15_dp, &
            "Huber value independent formula", failures)
        call huber_loss_jvp(prediction, targets, 0.75_dp, direction, value, &
            value_dot, status)
        call huber_loss_value(prediction + h*direction, targets, 0.75_dp, &
            value_plus, status)
        call huber_loss_value(prediction - h*direction, targets, 0.75_dp, &
            value_minus, status)
        finite_dot = (value_plus - value_minus)/(2.0_dp*h)
        call huber_loss_vjp(prediction, targets, 0.75_dp, 1.3_dp, prediction_bar, &
            status)
        lhs = 1.3_dp*value_dot
        rhs = sum(prediction_bar*direction)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 2.0e-10_dp, &
            "Huber JVP finite difference", failures)
        call check(abs(lhs - rhs) < 2.0e-14_dp, "Huber VJP adjoint identity", failures)

        quantile = 0.7_dp
        call quantile_loss_value(prediction, targets, quantile, value, status)
        reference = reference_quantile(prediction, targets, quantile)
        call check(status_ok(status) .and. abs(value - reference) < 2.0e-15_dp, &
            "quantile value independent formula", failures)
        call quantile_loss_jvp(prediction, targets, quantile, direction, value, &
            value_dot, status)
        call quantile_loss_value(prediction + h*direction, targets, quantile, &
            value_plus, status)
        call quantile_loss_value(prediction - h*direction, targets, quantile, &
            value_minus, status)
        finite_dot = (value_plus - value_minus)/(2.0_dp*h)
        call quantile_loss_vjp(prediction, targets, quantile, -0.8_dp, prediction_bar, &
            status)
        lhs = -0.8_dp*value_dot
        rhs = sum(prediction_bar*direction)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 2.0e-10_dp, &
            "quantile JVP finite difference", failures)
        call check(abs(lhs - rhs) < 2.0e-14_dp, "quantile VJP adjoint identity", failures)
        targets(1, 1) = prediction(1, 1)
        call quantile_loss_jvp(prediction, targets, quantile, direction, value, &
            value_dot, status)
        call check(.not. status_ok(status), "quantile kink derivative refusal", failures)
    end subroutine test_huber_and_quantile

    real(dp) function reference_huber(prediction, targets, delta) result(value)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), delta
        real(dp) :: residual
        integer :: i, j

        value = 0.0_dp
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                if (abs(residual) <= delta) then
                    value = value + 0.5_dp*residual*residual
                else
                    value = value + delta*(abs(residual) - 0.5_dp*delta)
                end if
            end do
        end do
        value = value/real(size(prediction), dp)
    end function reference_huber

    real(dp) function reference_quantile(prediction, targets, quantile) result(value)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), quantile
        real(dp) :: residual
        integer :: i, j

        value = 0.0_dp
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = targets(i, j) - prediction(i, j)
                if (residual >= 0.0_dp) then
                    value = value + quantile*residual
                else
                    value = value + (quantile - 1.0_dp)*residual
                end if
            end do
        end do
        value = value/real(size(prediction), dp)
    end function reference_quantile

    subroutine reference_sigmoid(logits, probabilities)
        real(dp), intent(in) :: logits(:, :)
        real(dp), intent(out) :: probabilities(:, :)

        probabilities = 1.0_dp/(1.0_dp + exp(-logits))
    end subroutine reference_sigmoid

    real(dp) function reference_binary_cross_entropy(logits, targets) result(value)
        real(dp), intent(in) :: logits(:, :), targets(:, :)
        real(dp) :: probabilities(size(logits, 1), size(logits, 2))

        call reference_sigmoid(logits, probabilities)
        value = -sum(targets*log(probabilities) + &
            (1.0_dp - targets)*log(1.0_dp - probabilities))/real(size(logits), dp)
    end function reference_binary_cross_entropy

    subroutine finite_binary_gradient(logits, targets, cotangent, h, gradient)
        real(dp), intent(in) :: logits(:, :), targets(:, :), cotangent, h
        real(dp), intent(out) :: gradient(:, :)
        real(dp) :: plus(size(logits, 1), size(logits, 2))
        real(dp) :: minus(size(logits, 1), size(logits, 2))
        integer :: i, j

        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                plus = logits
                minus = logits
                plus(i, j) = plus(i, j) + h
                minus(i, j) = minus(i, j) - h
                gradient(i, j) = cotangent*( &
                    reference_binary_cross_entropy(plus, targets) - &
                    reference_binary_cross_entropy(minus, targets))/(2.0_dp*h)
            end do
        end do
    end subroutine finite_binary_gradient

    subroutine reference_softmax(logits, probabilities)
        real(dp), intent(in) :: logits(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        real(dp) :: normalizer
        integer :: i

        do i = 1, size(logits, 1)
            probabilities(i, :) = exp(logits(i, :))
            normalizer = sum(probabilities(i, :))
            probabilities(i, :) = probabilities(i, :)/normalizer
        end do
    end subroutine reference_softmax

    subroutine reference_log_softmax(logits, log_probabilities)
        real(dp), intent(in) :: logits(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        real(dp) :: probabilities(size(logits, 1), size(logits, 2))

        call reference_softmax(logits, probabilities)
        log_probabilities = log(probabilities)
    end subroutine reference_log_softmax

    real(dp) function reference_cross_entropy(logits, labels) result(value)
        real(dp), intent(in) :: logits(:, :)
        integer, intent(in) :: labels(:)
        real(dp) :: probabilities(size(logits, 1), size(logits, 2))
        integer :: i

        call reference_softmax(logits, probabilities)
        value = 0.0_dp
        do i = 1, size(logits, 1)
            value = value - log(probabilities(i, labels(i)))
        end do
        value = value/real(size(logits, 1), dp)
    end function reference_cross_entropy

    subroutine finite_cross_entropy_gradient(logits, labels, cotangent, h, gradient)
        real(dp), intent(in) :: logits(:, :), cotangent, h
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: gradient(:, :)
        real(dp) :: plus(size(logits, 1), size(logits, 2))
        real(dp) :: minus(size(logits, 1), size(logits, 2))
        integer :: i, j

        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                plus = logits
                minus = logits
                plus(i, j) = plus(i, j) + h
                minus(i, j) = minus(i, j) - h
                gradient(i, j) = cotangent*( &
                    reference_cross_entropy(plus, labels) - &
                    reference_cross_entropy(minus, labels))/(2.0_dp*h)
            end do
        end do
    end subroutine finite_cross_entropy_gradient

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(name)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_losses
