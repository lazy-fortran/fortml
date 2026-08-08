program test_neural_loss_products
    !! Independent value and product oracles for the neural loss catalog.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_losses, only: LOSS_REDUCTION_SUM, softmax_value, softmax_jvp, &
        softmax_vjp, softmax_hvp, log_softmax_value, log_softmax_jvp, &
        log_softmax_vjp, log_softmax_hvp, softmax_cross_entropy_value, &
        softmax_cross_entropy_jvp, softmax_cross_entropy_vjp, &
        softmax_cross_entropy_hvp, focal_binary_cross_entropy_with_logits_value, &
        focal_binary_cross_entropy_with_logits_jvp, &
        focal_binary_cross_entropy_with_logits_vjp, &
        focal_binary_cross_entropy_with_logits_hvp, softmax_value_device, &
        log_softmax_value_device, softmax_cross_entropy_value_device, &
        focal_binary_cross_entropy_with_logits_value_device
    implicit none

    integer :: failures

    failures = 0
    call test_softmax_products(failures)
    call test_log_softmax_products(failures)
    call test_weighted_cross_entropy_products(failures)
    call test_focal_hvp(failures)
    call test_device_refusals(failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL neural loss cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS neural loss independent behavioral oracles"

contains

    subroutine test_softmax_products(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(3, 4), direction(3, 4), cotangent(3, 4)
        real(dp) :: probabilities(3, 4), probabilities_dot(3, 4)
        real(dp) :: logits_bar(3, 4), hvp(3, 4), bar_plus(3, 4), bar_minus(3, 4)
        real(dp) :: value_plus, value_minus, h, lhs, rhs

        logits = reshape([ -2.0_dp, 0.25_dp, 1.5_dp, 4.0_dp, &
            1.0_dp, -0.75_dp, 0.5_dp, 2.0_dp, &
            1000.0_dp, 999.0_dp, 998.0_dp, 997.0_dp ], shape(logits))
        direction = reshape([ 0.3_dp, -0.2_dp, 0.4_dp, -0.1_dp, &
            -0.1_dp, 0.5_dp, -0.3_dp, 0.2_dp, &
            0.2_dp, -0.4_dp, 0.1_dp, 0.6_dp ], shape(direction))
        cotangent = reshape([ 0.7_dp, -0.5_dp, 0.2_dp, 0.3_dp, &
            -0.2_dp, 0.4_dp, 0.9_dp, -0.1_dp, &
            0.3_dp, -0.8_dp, 0.6_dp, 0.5_dp ], shape(cotangent))
        h = 2.0e-6_dp

        call softmax_value(logits, probabilities, status)
        call check(status_ok(status) .and. all(ieee_is_finite(probabilities)) .and. &
            maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 2.0e-14_dp, &
            "softmax stable simplex oracle", failures)
        call softmax_jvp(logits, direction, probabilities, probabilities_dot, status)
        call softmax_value(logits + h*direction, bar_plus, status)
        call softmax_value(logits - h*direction, bar_minus, status)
        call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
            (bar_plus - bar_minus)/(2.0_dp*h))) < 3.0e-9_dp, &
            "softmax JVP finite difference", failures)

        call softmax_vjp(logits, cotangent, logits_bar, status)
        lhs = sum(logits_bar*direction)
        rhs = sum(cotangent*probabilities_dot)
        call check(status_ok(status) .and. abs(lhs-rhs) < 2.0e-14_dp, &
            "softmax VJP adjoint", failures)
        call softmax_hvp(logits, direction, cotangent, hvp, status)
        call softmax_vjp(logits + h*direction, cotangent, bar_plus, status)
        call softmax_vjp(logits - h*direction, cotangent, bar_minus, status)
        call check(status_ok(status) .and. maxval(abs(hvp - &
            (bar_plus - bar_minus)/(2.0_dp*h))) < 3.0e-9_dp, &
            "softmax HVP finite difference", failures)
    end subroutine test_softmax_products

    subroutine test_log_softmax_products(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(2, 3), direction(2, 3), cotangent(2, 3)
        real(dp) :: log_probabilities(2, 3), log_probabilities_dot(2, 3)
        real(dp) :: logits_bar(2, 3), hvp(2, 3), bar_plus(2, 3), bar_minus(2, 3)
        real(dp) :: h

        logits = reshape([ -1.0_dp, 0.5_dp, 2.0_dp, &
            7.0_dp, 6.0_dp, 5.0_dp ], shape(logits))
        direction = reshape([ 0.2_dp, -0.3_dp, 0.4_dp, &
            -0.4_dp, 0.1_dp, 0.25_dp ], shape(direction))
        cotangent = reshape([ 0.7_dp, -0.2_dp, 0.4_dp, &
            -0.3_dp, 0.5_dp, 0.9_dp ], shape(cotangent))
        h = 2.0e-6_dp

        call log_softmax_value(logits, log_probabilities, status)
        call check(status_ok(status) .and. all(ieee_is_finite(log_probabilities)) .and. &
            maxval(abs(exp(log_probabilities) - reference_softmax(logits))) < 2.0e-14_dp, &
            "log-softmax stable value oracle", failures)
        call log_softmax_jvp(logits, direction, log_probabilities, &
            log_probabilities_dot, status)
        call log_softmax_value(logits + h*direction, bar_plus, status)
        call log_softmax_value(logits - h*direction, bar_minus, status)
        call check(status_ok(status) .and. maxval(abs(log_probabilities_dot - &
            (bar_plus - bar_minus)/(2.0_dp*h))) < 3.0e-9_dp, &
            "log-softmax JVP finite difference", failures)
        call log_softmax_vjp(logits, cotangent, logits_bar, status)
        call check(status_ok(status) .and. abs(sum(logits_bar*direction) - &
            sum(cotangent*log_probabilities_dot)) < 2.0e-14_dp, &
            "log-softmax VJP adjoint", failures)
        call log_softmax_hvp(logits, direction, cotangent, hvp, status)
        call log_softmax_vjp(logits + h*direction, cotangent, bar_plus, status)
        call log_softmax_vjp(logits - h*direction, cotangent, bar_minus, status)
        call check(status_ok(status) .and. maxval(abs(hvp - &
            (bar_plus - bar_minus)/(2.0_dp*h))) < 3.0e-9_dp, &
            "log-softmax HVP finite difference", failures)
    end subroutine test_log_softmax_products

    subroutine test_weighted_cross_entropy_products(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(3, 3), direction(3, 3), weights(3)
        real(dp) :: probabilities(3, 3), cotangent(3, 3), logits_bar(3, 3)
        real(dp) :: hvp(3, 3), bar_plus(3, 3), bar_minus(3, 3)
        real(dp) :: bad_direction(2, 3), zero_weights(3)
        real(dp) :: value, value_dot, value_plus, value_minus, sum_value
        real(dp) :: h, expected, finite_dot
        integer :: labels(3)

        logits = reshape([ 0.2_dp, -0.5_dp, 1.3_dp, &
            -0.8_dp, 0.7_dp, 0.1_dp, &
            1.2_dp, -0.1_dp, 0.4_dp ], shape(logits))
        direction = reshape([ 0.3_dp, -0.4_dp, 0.2_dp, &
            -0.2_dp, 0.5_dp, 0.1_dp, &
            0.4_dp, -0.3_dp, 0.6_dp ], shape(direction))
        cotangent = reshape([ 0.3_dp, -0.1_dp, 0.5_dp, &
            -0.2_dp, 0.6_dp, 0.4_dp, &
            0.7_dp, -0.8_dp, 0.2_dp ], shape(cotangent))
        labels = [3, 1, 2]
        weights = [1.0_dp, 2.0_dp, 4.0_dp]
        h = 2.0e-6_dp

        call softmax_cross_entropy_value(logits, labels, value, status, weights)
        call softmax_cross_entropy_jvp(logits, labels, direction, value, value_dot, &
            status, weights)
        call softmax_cross_entropy_value(logits + h*direction, labels, value_plus, &
            status, weights)
        call softmax_cross_entropy_value(logits - h*direction, labels, value_minus, &
            status, weights)
        finite_dot = (value_plus-value_minus)/(2.0_dp*h)
        call check(status_ok(status) .and. abs(value_dot-finite_dot) < 3.0e-9_dp, &
            "weighted softmax cross-entropy JVP", failures)
        call softmax_cross_entropy_vjp(logits, labels, -0.7_dp, logits_bar, status, &
            weights)
        call check(status_ok(status) .and. abs(sum(logits_bar*direction) + &
            0.7_dp*value_dot) < 2.0e-14_dp, &
            "weighted softmax cross-entropy VJP adjoint", failures)
        call softmax_cross_entropy_hvp(logits, labels, direction, hvp, status, weights)
        call softmax_cross_entropy_vjp(logits + h*direction, labels, 1.0_dp, &
            bar_plus, status, weights)
        call softmax_cross_entropy_vjp(logits - h*direction, labels, 1.0_dp, &
            bar_minus, status, weights)
        call check(status_ok(status) .and. maxval(abs(hvp - &
            (bar_plus-bar_minus)/(2.0_dp*h))) < 3.0e-9_dp, &
            "weighted softmax cross-entropy HVP", failures)
        call softmax_cross_entropy_value(logits, labels, sum_value, status, weights, &
            LOSS_REDUCTION_SUM)
        call check(status_ok(status) .and. abs(sum_value-value*sum(weights)) < &
            2.0e-14_dp, "softmax cross-entropy mean/sum reduction", failures)
        call softmax_value(logits, probabilities, status)
        expected = 0.0_dp
        expected = sum(weights*[-log(probabilities(1, labels(1))), &
            -log(probabilities(2, labels(2))), -log(probabilities(3, labels(3)))]) &
            /sum(weights)
        call check(status_ok(status) .and. abs(value-expected) < 2.0e-14_dp, &
            "softmax cross-entropy independent NumPy-style value", failures)
        zero_weights = 0.0_dp
        call softmax_cross_entropy_value(logits, labels, value, status, zero_weights)
        call check(.not. status_ok(status), &
            "softmax cross-entropy zero-support refusal", failures)
        bad_direction = 0.0_dp
        call softmax_cross_entropy_hvp(logits, labels, bad_direction, hvp, status, &
            weights)
        call check(.not. status_ok(status), &
            "softmax cross-entropy tangent-shape refusal", failures)
        labels(3) = 4
        call softmax_cross_entropy_value(logits, labels, value, status, weights)
        call check(.not. status_ok(status), "softmax cross-entropy label refusal", &
            failures)
    end subroutine test_weighted_cross_entropy_products

    subroutine test_focal_hvp(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(3, 2), targets(3, 2), direction(3, 2), weights(3)
        real(dp) :: hvp(3, 2), bar_plus(3, 2), bar_minus(3, 2)
        real(dp) :: value, value_dot, value_plus, value_minus, finite_dot, h

        logits = reshape([ -1.1_dp, 0.8_dp, 0.4_dp, -0.6_dp, 1.2_dp, 0.1_dp ], &
            shape(logits))
        targets = reshape([ 0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 0.25_dp, 0.75_dp ], &
            shape(targets))
        direction = reshape([ 0.3_dp, -0.4_dp, 0.2_dp, 0.1_dp, -0.5_dp, 0.6_dp ], &
            shape(direction))
        weights = [1.0_dp, 2.0_dp, 4.0_dp]
        h = 2.0e-6_dp
        call focal_binary_cross_entropy_with_logits_jvp(logits, targets, 0.25_dp, &
            2.0_dp, direction, value, value_dot, status, weights)
        call focal_binary_cross_entropy_with_logits_value(logits + h*direction, &
            targets, 0.25_dp, 2.0_dp, value_plus, status, weights)
        call focal_binary_cross_entropy_with_logits_value(logits - h*direction, &
            targets, 0.25_dp, 2.0_dp, value_minus, status, weights)
        finite_dot = (value_plus-value_minus)/(2.0_dp*h)
        call check(status_ok(status) .and. abs(value_dot-finite_dot) < 3.0e-9_dp, &
            "focal BCE weighted JVP", failures)
        call focal_binary_cross_entropy_with_logits_hvp(logits, targets, 0.25_dp, &
            2.0_dp, direction, hvp, status, weights)
        call focal_binary_cross_entropy_with_logits_vjp(logits + h*direction, targets, &
            0.25_dp, 2.0_dp, 1.0_dp, bar_plus, status, weights)
        call focal_binary_cross_entropy_with_logits_vjp(logits - h*direction, targets, &
            0.25_dp, 2.0_dp, 1.0_dp, bar_minus, status, weights)
        call check(status_ok(status) .and. maxval(abs(hvp - &
            (bar_plus-bar_minus)/(2.0_dp*h))) < 4.0e-8_dp, &
            "focal BCE weighted HVP", failures)
    end subroutine test_focal_hvp

    subroutine test_device_refusals(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(2, 3), output(2, 3), targets(2, 3), value
        integer :: labels(2)

        logits = 0.0_dp
        targets = 0.5_dp
        labels = [1, 2]
        output = -31.0_dp
        value = -37.0_dp
        call softmax_value_device(logits, output, status, FORTML_DEVICE_CUDA)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            all(output == -31.0_dp), "softmax CUDA refusal is typed", failures)
        call log_softmax_value_device(logits, output, status, FORTML_DEVICE_CUDA)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            all(output == -31.0_dp), "log-softmax CUDA refusal is typed", failures)
        call softmax_cross_entropy_value_device(logits, labels, value, status, &
            FORTML_DEVICE_CUDA)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. value == -37.0_dp, &
            "softmax cross-entropy CUDA refusal is typed", failures)
        call focal_binary_cross_entropy_with_logits_value_device(logits, targets, &
            0.25_dp, 2.0_dp, value, status, FORTML_DEVICE_CUDA)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. value == -37.0_dp, &
            "focal BCE CUDA refusal is typed", failures)
    end subroutine test_device_refusals

    function reference_softmax(logits) result(probabilities)
        real(dp), intent(in) :: logits(:, :)
        real(dp) :: probabilities(size(logits, 1), size(logits, 2))
        real(dp) :: maximum, normalizer
        integer :: i, j

        do i = 1, size(logits, 1)
            maximum = maxval(logits(i, :))
            normalizer = sum(exp(logits(i, :)-maximum))
            do j = 1, size(logits, 2)
                probabilities(i, j) = exp(logits(i, j)-maximum)/normalizer
            end do
        end do
    end function reference_softmax

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL ["//trim(description)//"]"
        end if
    end subroutine check

end program test_neural_loss_products
