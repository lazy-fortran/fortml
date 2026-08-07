program test_mae_focal_losses
    !! Independent behavioral oracles for MAE and focal BCE losses.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_losses, only: mae_loss_value, mae_loss_jvp, mae_loss_vjp, &
        focal_binary_cross_entropy_with_logits_value, &
        focal_binary_cross_entropy_with_logits_jvp, &
        focal_binary_cross_entropy_with_logits_vjp, LOSS_REDUCTION_SUM
    implicit none

    integer :: failures

    failures = 0
    call test_mae_products(failures)
    call test_focal_products(failures)
    call test_refusals_and_stability(failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL MAE/focal loss cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MAE/focal loss independent behavioral oracles"

contains

    subroutine test_mae_products(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: prediction(3, 2), targets(3, 2), direction(3, 2)
        real(dp) :: weights(3), prediction_bar(3, 2)
        real(dp) :: value, value_dot, value_plus, value_minus, finite_dot
        real(dp) :: lhs, rhs, h, sum_value

        prediction = reshape([0.2_dp, -0.5_dp, 1.3_dp, 0.8_dp, 0.1_dp, -0.7_dp], &
            shape(prediction))
        targets = reshape([0.0_dp, -0.2_dp, 1.0_dp, 0.3_dp, 0.4_dp, -0.1_dp], &
            shape(targets))
        direction = reshape([0.3_dp, -0.4_dp, 0.2_dp, -0.1_dp, 0.5_dp, 0.6_dp], &
            shape(direction))
        weights = [1.0_dp, 2.0_dp, 4.0_dp]
        h = 2.0e-6_dp
        call mae_loss_value(prediction, targets, value, status, weights)
        call mae_loss_jvp(prediction, targets, direction, value, value_dot, status, &
            weights)
        call mae_loss_value(prediction + h*direction, targets, value_plus, status, &
            weights)
        call mae_loss_value(prediction - h*direction, targets, value_minus, status, &
            weights)
        finite_dot = (value_plus - value_minus)/(2.0_dp*h)
        call mae_loss_vjp(prediction, targets, -1.7_dp, prediction_bar, status, &
            weights)
        lhs = -1.7_dp*value_dot
        rhs = sum(prediction_bar*direction)
        call mae_loss_value(prediction, targets, sum_value, status, weights, &
            LOSS_REDUCTION_SUM)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 2.0e-9_dp, &
            "MAE JVP independent finite difference", failures)
        call check(abs(lhs - rhs) < 2.0e-14_dp, "MAE VJP adjoint identity", failures)
        call check(abs(sum_value - value*sum(weights)) < 2.0e-14_dp, &
            "MAE sum/mean reduction relation", failures)
    end subroutine test_mae_products

    subroutine test_focal_products(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(3, 2), targets(3, 2), direction(3, 2), weights(3)
        real(dp) :: logits_bar(3, 2), value, value_dot, value_plus, value_minus
        real(dp) :: finite_dot, lhs, rhs, h, expected, sum_value
        integer :: i, j

        logits = reshape([-1.1_dp, 0.8_dp, 0.4_dp, -0.6_dp, 1.2_dp, 0.1_dp], &
            shape(logits))
        targets = reshape([0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 0.25_dp, 0.75_dp], &
            shape(targets))
        direction = reshape([0.3_dp, -0.4_dp, 0.2_dp, 0.1_dp, -0.5_dp, 0.6_dp], &
            shape(direction))
        weights = [1.0_dp, 2.0_dp, 4.0_dp]
        h = 2.0e-6_dp
        call focal_binary_cross_entropy_with_logits_value(logits, targets, 0.25_dp, &
            2.0_dp, value, status, weights)
        call focal_binary_cross_entropy_with_logits_jvp(logits, targets, 0.25_dp, &
            2.0_dp, direction, value, value_dot, status, weights)
        call focal_binary_cross_entropy_with_logits_value(logits + h*direction, &
            targets, 0.25_dp, 2.0_dp, value_plus, status, weights)
        call focal_binary_cross_entropy_with_logits_value(logits - h*direction, &
            targets, 0.25_dp, 2.0_dp, value_minus, status, weights)
        finite_dot = (value_plus - value_minus)/(2.0_dp*h)
        call focal_binary_cross_entropy_with_logits_vjp(logits, targets, 0.25_dp, &
            2.0_dp, -0.8_dp, logits_bar, status, weights)
        lhs = -0.8_dp*value_dot
        rhs = sum(logits_bar*direction)
        call focal_binary_cross_entropy_with_logits_value(logits, targets, 0.25_dp, &
            2.0_dp, sum_value, status, weights, LOSS_REDUCTION_SUM)
        expected = 0.0_dp
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                expected = expected + weights(i)*reference_focal(logits(i, j), &
                    targets(i, j), 0.25_dp, 2.0_dp)
            end do
        end do
        call check(status_ok(status) .and. abs(value - expected/sum(weights)) < &
            2.0e-14_dp, "focal BCE stable weighted value oracle", failures)
        call check(abs(value_dot - finite_dot) < 2.0e-9_dp, &
            "focal BCE JVP independent finite difference", failures)
        call check(abs(lhs - rhs) < 2.0e-14_dp, &
            "focal BCE VJP adjoint identity", failures)
        call check(abs(sum_value - value*sum(weights)) < 2.0e-14_dp, &
            "focal BCE sum/mean reduction relation", failures)
    end subroutine test_focal_products

    subroutine test_refusals_and_stability(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: prediction(2, 1), targets(2, 1), direction(2, 1), value, value_dot
        real(dp) :: bar(2, 1), logits(2, 1), focal_value
        real(dp) :: weights(2)

        prediction(:, 1) = [0.2_dp, -0.4_dp]
        targets(:, 1) = [0.2_dp, 0.1_dp]
        direction = 1.0_dp
        weights = 1.0_dp
        call mae_loss_jvp(prediction, targets, direction, value, value_dot, status, weights)
        call check(.not. status_ok(status), "MAE exact kink refusal", failures)
        call mae_loss_vjp(prediction, targets, 1.0_dp, bar, status, weights)
        call check(.not. status_ok(status), "MAE VJP exact kink refusal", failures)
        logits(:, 1) = [-1000.0_dp, 1000.0_dp]
        targets(:, 1) = [0.0_dp, 1.0_dp]
        call focal_binary_cross_entropy_with_logits_value(logits, targets, 0.25_dp, &
            2.0_dp, focal_value, status)
        call check(status_ok(status) .and. ieee_is_finite(focal_value) .and. &
            focal_value < 1.0e-12_dp, "focal BCE stable extreme logits", failures)
        call focal_binary_cross_entropy_with_logits_value(logits, targets, -0.1_dp, &
            2.0_dp, focal_value, status)
        call check(.not. status_ok(status), "focal BCE alpha domain refusal", failures)
        call focal_binary_cross_entropy_with_logits_value(logits, targets, 0.25_dp, &
            -1.0_dp, focal_value, status)
        call check(.not. status_ok(status), "focal BCE gamma domain refusal", failures)
    end subroutine test_refusals_and_stability

    real(dp) function reference_focal(logit, target, alpha, gamma) result(value)
        real(dp), intent(in) :: logit, target, alpha, gamma
        real(dp) :: probability, pt, bce, alpha_t

        probability = 1.0_dp/(1.0_dp + exp(-logit))
        pt = target*probability + (1.0_dp - target)*(1.0_dp - probability)
        bce = -target*log(probability) - (1.0_dp - target)*log(1.0_dp - probability)
        alpha_t = alpha*target + (1.0_dp - alpha)*(1.0_dp - target)
        value = alpha_t*(1.0_dp - pt)**gamma*bce
    end function reference_focal

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
        end if
    end subroutine check

end program test_mae_focal_losses
