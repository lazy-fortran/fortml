program test_neural_losses
    !! Independent behavioral oracles for the differentiable neural loss layer.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_losses, only: &
        binary_cross_entropy_with_logits_value, &
        binary_cross_entropy_with_logits_vjp, &
        binary_cross_entropy_with_logits_hvp, &
        softmax_cross_entropy_value, softmax_cross_entropy_vjp, &
        softmax_cross_entropy_hvp, weighted_mse_loss_value, &
        weighted_mse_loss_jvp, weighted_mse_loss_vjp, weighted_mse_loss_hvp, &
        huber_loss_hvp, multilabel_binary_cross_entropy_with_logits_value, &
        multilabel_binary_cross_entropy_with_logits_jvp, &
        multilabel_binary_cross_entropy_with_logits_vjp, &
        multilabel_binary_cross_entropy_with_logits_hvp, &
        ordinal_cumulative_logit_loss_value, &
        ordinal_cumulative_logit_loss_jvp, ordinal_cumulative_logit_loss_vjp, &
        ordinal_cumulative_logit_loss_hvp, LOSS_REDUCTION_SUM
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, &
        mlp_loss_diagnostics_t
    implicit none

    integer :: failures

    failures = 0
    call test_binary_hvp(failures)
    call test_softmax_hvp(failures)
    call test_weighted_mse_products(failures)
    call test_huber_kink_refusal(failures)
    call test_multilabel_bce_products(failures)
    call test_ordinal_loss_products(failures)
    call test_mlp_weighted_integration(failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL neural loss cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS neural loss independent behavioral oracles"

contains

    subroutine test_binary_hvp(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(2, 3), targets(2, 3), direction(2, 3)
        real(dp) :: hvp(2, 3), plus(2, 3), minus(2, 3), finite_hvp(2, 3)
        real(dp) :: value, h

        logits = reshape([-1.1_dp, 0.8_dp, 0.4_dp, -0.6_dp, 1.2_dp, 0.1_dp], &
            shape(logits))
        targets = reshape([0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 0.25_dp, 0.75_dp], &
            shape(targets))
        direction = reshape([0.3_dp, -0.4_dp, 0.2_dp, 0.1_dp, -0.5_dp, 0.6_dp], &
            shape(direction))
        h = 2.0e-5_dp
        call binary_cross_entropy_with_logits_hvp(logits, targets, direction, &
            hvp, status)
        call binary_cross_entropy_with_logits_vjp(logits + h*direction, targets, &
            1.0_dp, plus, status)
        call binary_cross_entropy_with_logits_vjp(logits - h*direction, targets, &
            1.0_dp, minus, status)
        finite_hvp = (plus - minus)/(2.0_dp*h)
        call binary_cross_entropy_with_logits_value(logits, targets, value, status)
        call check(status_ok(status) .and. value > 0.0_dp .and. &
            maxval(abs(hvp - finite_hvp)) < 2.0e-9_dp, &
            "BCE logits HVP independent gradient finite difference", failures)
    end subroutine test_binary_hvp

    subroutine test_softmax_hvp(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(2, 3), direction(2, 3), hvp(2, 3)
        real(dp) :: plus(2, 3), minus(2, 3), finite_hvp(2, 3)
        real(dp) :: value
        integer :: labels(2)
        real(dp) :: h

        logits = reshape([1.0_dp, -0.5_dp, 2.0_dp, 0.2_dp, 3.0_dp, 1.1_dp], &
            shape(logits))
        labels = [3, 1]
        direction = reshape([0.3_dp, -0.4_dp, 0.2_dp, 0.1_dp, -0.5_dp, 0.6_dp], &
            shape(direction))
        h = 2.0e-5_dp
        call softmax_cross_entropy_hvp(logits, labels, direction, hvp, status)
        call softmax_cross_entropy_vjp(logits + h*direction, labels, 1.0_dp, &
            plus, status)
        call softmax_cross_entropy_vjp(logits - h*direction, labels, 1.0_dp, &
            minus, status)
        finite_hvp = (plus - minus)/(2.0_dp*h)
        call softmax_cross_entropy_value(logits, labels, value, status)
        call check(status_ok(status) .and. value > 0.0_dp .and. &
            maxval(abs(hvp - finite_hvp)) < 2.0e-9_dp, &
            "softmax cross entropy HVP independent gradient finite difference", failures)
    end subroutine test_softmax_hvp

    subroutine test_weighted_mse_products(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: prediction(3, 2), targets(3, 2), direction(3, 2)
        real(dp) :: weights(3), prediction_bar(3, 2), prediction_hvp(3, 2)
        real(dp) :: value, value_dot, value_plus, value_minus, finite_dot
        real(dp) :: lhs, rhs, normalization, h

        prediction = reshape([0.2_dp, -0.5_dp, 1.3_dp, 0.8_dp, 0.1_dp, -0.7_dp], &
            shape(prediction))
        targets = reshape([0.0_dp, -0.2_dp, 1.0_dp, 0.3_dp, 0.4_dp, -0.1_dp], &
            shape(targets))
        direction = reshape([0.3_dp, -0.4_dp, 0.2_dp, -0.1_dp, 0.5_dp, 0.6_dp], &
            shape(direction))
        weights = [1.0_dp, 2.0_dp, 4.0_dp]
        normalization = sum(weights)
        h = 2.0e-6_dp
        call weighted_mse_loss_value(prediction, targets, weights, value, status)
        call weighted_mse_loss_jvp(prediction, targets, weights, direction, &
            value, value_dot, status)
        call weighted_mse_loss_value(prediction + h*direction, targets, weights, &
            value_plus, status)
        call weighted_mse_loss_value(prediction - h*direction, targets, weights, &
            value_minus, status)
        finite_dot = (value_plus - value_minus)/(2.0_dp*h)
        call weighted_mse_loss_vjp(prediction, targets, weights, -1.7_dp, &
            prediction_bar, status)
        lhs = -1.7_dp*value_dot
        rhs = sum(prediction_bar*direction)
        call weighted_mse_loss_hvp(prediction, targets, weights, direction, &
            prediction_hvp, status)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 2.0e-9_dp, &
            "weighted MSE JVP independent finite difference", failures)
        call check(abs(lhs - rhs) < 2.0e-14_dp, &
            "weighted MSE VJP adjoint identity", failures)
        call check(maxval(abs(prediction_hvp - spread(weights/normalization, 2, 2)* &
            direction)) < 2.0e-14_dp, "weighted MSE HVP analytic curvature", failures)
    end subroutine test_weighted_mse_products

    subroutine test_huber_kink_refusal(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: prediction(2, 2), targets(2, 2), direction(2, 2), hvp(2, 2)

        prediction = reshape([0.1_dp, 2.0_dp, -0.5_dp, 1.3_dp], shape(prediction))
        targets = reshape([0.0_dp, 1.0_dp, -0.2_dp, 0.3_dp], shape(targets))
        direction = 1.0_dp
        call huber_loss_hvp(prediction, targets, 0.75_dp, direction, hvp, status)
        call check(status_ok(status) .and. maxval(abs(hvp - &
            reshape([0.25_dp, 0.0_dp, 0.25_dp, 0.0_dp], shape(hvp)))) < &
            2.0e-14_dp, "Huber HVP piecewise curvature", failures)
        targets(1, 1) = prediction(1, 1) - 0.75_dp
        call huber_loss_hvp(prediction, targets, 0.75_dp, direction, hvp, status)
        call check(.not. status_ok(status), "Huber HVP kink refusal", failures)
    end subroutine test_huber_kink_refusal

    subroutine test_multilabel_bce_products(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(3, 2), targets(3, 2), direction(3, 2), weights(3)
        real(dp) :: value, value_dot, value_plus, value_minus, finite_dot
        real(dp) :: logits_bar(3, 2), hvp(3, 2), plus_bar(3, 2), minus_bar(3, 2)
        real(dp) :: lhs, rhs, h, reference, sum_reference, sum_value

        logits = reshape([-1.2_dp, 0.4_dp, 1.1_dp, 0.7_dp, -0.8_dp, 0.2_dp], &
            shape(logits))
        targets = reshape([0.0_dp, 1.0_dp, 0.25_dp, 1.0_dp, 0.0_dp, 0.75_dp], &
            shape(targets))
        direction = reshape([0.3_dp, -0.4_dp, 0.2_dp, 0.1_dp, -0.5_dp, 0.6_dp], &
            shape(direction))
        weights = [1.0_dp, 2.0_dp, 4.0_dp]
        h = 2.0e-6_dp
        reference = reference_multilabel_bce(logits, targets, weights, .false.)
        call multilabel_binary_cross_entropy_with_logits_value(logits, targets, &
            value, status, weights)
        call check(status_ok(status) .and. abs(value - reference) < 3.0e-14_dp, &
            "multilabel BCE weighted formula", failures)
        call multilabel_binary_cross_entropy_with_logits_jvp(logits, targets, &
            direction, value, value_dot, status, weights)
        finite_dot = (reference_multilabel_bce(logits + h*direction, targets, &
            weights, .false.) - reference_multilabel_bce(logits - h*direction, &
            targets, weights, .false.))/(2.0_dp*h)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 2.0e-9_dp, &
            "multilabel BCE weighted JVP finite difference", failures)
        call multilabel_binary_cross_entropy_with_logits_vjp(logits, targets, &
            -0.7_dp, logits_bar, status, weights)
        lhs = -0.7_dp*value_dot
        rhs = sum(logits_bar*direction)
        call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-14_dp, &
            "multilabel BCE weighted VJP adjoint", failures)
        call multilabel_binary_cross_entropy_with_logits_hvp(logits, targets, &
            direction, hvp, status, weights)
        call multilabel_binary_cross_entropy_with_logits_vjp(logits + h*direction, &
            targets, 1.0_dp, plus_bar, status, weights)
        call multilabel_binary_cross_entropy_with_logits_vjp(logits - h*direction, &
            targets, 1.0_dp, minus_bar, status, weights)
        call check(status_ok(status) .and. maxval(abs(hvp - (plus_bar - minus_bar)/ &
            (2.0_dp*h))) < 2.0e-9_dp, &
            "multilabel BCE weighted HVP finite difference", failures)
        sum_reference = reference_multilabel_bce(logits, targets, weights, .true.)
        call multilabel_binary_cross_entropy_with_logits_value(logits, targets, &
            sum_value, status, weights, LOSS_REDUCTION_SUM)
        call check(status_ok(status) .and. abs(sum_value - sum_reference) < 3.0e-14_dp, &
            "multilabel BCE sum reduction", failures)
    end subroutine test_multilabel_bce_products

    subroutine test_ordinal_loss_products(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: logits(3, 2), direction(3, 2), weights(3)
        real(dp) :: value, value_dot, value_plus, value_minus, finite_dot
        real(dp) :: logits_bar(3, 2), hvp(3, 2), plus_bar(3, 2), minus_bar(3, 2)
        real(dp) :: lhs, rhs, h, reference, sum_reference, sum_value
        integer :: labels(3)

        logits = reshape([-1.1_dp, 0.2_dp, 0.8_dp, 0.9_dp, 1.4_dp, 2.0_dp], &
            shape(logits))
        direction = reshape([0.3_dp, -0.4_dp, 0.2_dp, 0.1_dp, -0.5_dp, 0.6_dp], &
            shape(direction))
        labels = [1, 2, 3]
        weights = [1.0_dp, 2.0_dp, 4.0_dp]
        h = 2.0e-6_dp
        reference = reference_ordinal_loss(logits, labels, weights, .false.)
        call ordinal_cumulative_logit_loss_value(logits, labels, value, status, weights)
        call check(status_ok(status) .and. abs(value - reference) < 3.0e-14_dp, &
            "ordinal cumulative-logit weighted formula", failures)
        call ordinal_cumulative_logit_loss_jvp(logits, labels, direction, value, &
            value_dot, status, weights)
        finite_dot = (reference_ordinal_loss(logits + h*direction, labels, weights, &
            .false.) - reference_ordinal_loss(logits - h*direction, labels, weights, &
            .false.))/(2.0_dp*h)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 2.0e-9_dp, &
            "ordinal cumulative-logit JVP finite difference", failures)
        call ordinal_cumulative_logit_loss_vjp(logits, labels, -0.7_dp, logits_bar, &
            status, weights)
        lhs = -0.7_dp*value_dot
        rhs = sum(logits_bar*direction)
        call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-14_dp, &
            "ordinal cumulative-logit VJP adjoint", failures)
        call ordinal_cumulative_logit_loss_hvp(logits, labels, direction, hvp, &
            status, weights)
        call ordinal_cumulative_logit_loss_vjp(logits + h*direction, labels, 1.0_dp, &
            plus_bar, status, weights)
        call ordinal_cumulative_logit_loss_vjp(logits - h*direction, labels, 1.0_dp, &
            minus_bar, status, weights)
        call check(status_ok(status) .and. maxval(abs(hvp - (plus_bar - minus_bar)/ &
            (2.0_dp*h))) < 2.0e-9_dp, &
            "ordinal cumulative-logit HVP finite difference", failures)
        sum_reference = reference_ordinal_loss(logits, labels, weights, .true.)
        call ordinal_cumulative_logit_loss_value(logits, labels, sum_value, status, &
            weights, LOSS_REDUCTION_SUM)
        call check(status_ok(status) .and. abs(sum_value - sum_reference) < 3.0e-14_dp, &
            "ordinal cumulative-logit sum reduction", failures)
        logits(2, 2) = logits(2, 1) - 0.1_dp
        call ordinal_cumulative_logit_loss_value(logits, labels, value, status, weights)
        call check(.not. status_ok(status), "ordinal cumulative-logit ordering refusal", &
            failures)
    end subroutine test_ordinal_loss_products

    real(dp) function reference_multilabel_bce(logits, targets, weights, sum_reduction) &
            result(value)
        real(dp), intent(in) :: logits(:, :), targets(:, :), weights(:)
        logical, intent(in) :: sum_reduction
        integer :: i, j

        value = 0.0_dp
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                value = value + weights(i)*(log(1.0_dp + exp(-abs(logits(i, j)))) &
                    + merge((1.0_dp - targets(i, j))*logits(i, j), &
                    -targets(i, j)*logits(i, j), logits(i, j) >= 0.0_dp))
            end do
        end do
        if (.not. sum_reduction) value = value/sum(weights)
    end function reference_multilabel_bce

    real(dp) function reference_ordinal_loss(logits, labels, weights, sum_reduction) &
            result(value)
        real(dp), intent(in) :: logits(:, :), weights(:)
        integer, intent(in) :: labels(:)
        logical, intent(in) :: sum_reduction
        real(dp) :: cumulative(size(logits, 2)), probability
        integer :: i, j

        value = 0.0_dp
        do i = 1, size(logits, 1)
            do j = 1, size(logits, 2)
                cumulative(j) = 1.0_dp/(1.0_dp + exp(-logits(i, j)))
            end do
            if (labels(i) == 1) then
                probability = cumulative(1)
            else if (labels(i) == size(logits, 2) + 1) then
                probability = 1.0_dp - cumulative(size(logits, 2))
            else
                probability = cumulative(labels(i)) - cumulative(labels(i) - 1)
            end if
            value = value - weights(i)*log(probability)
        end do
        if (.not. sum_reduction) value = value/sum(weights)
    end function reference_ordinal_loss

    subroutine test_mlp_weighted_integration(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_loss_diagnostics_t) :: diagnostics
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), target(4, 1), weights(4), value, l2_gradient
        real(dp), allocatable :: gradient(:), theta(:), plus_gradient(:)
        real(dp) :: plus, minus, h, finite_gradient

        x(:, 1) = [-1.5_dp, -0.25_dp, 0.75_dp, 1.8_dp]
        target(:, 1) = [0.2_dp, -0.1_dp, 0.7_dp, 1.5_dp]
        weights = [1.0_dp, 0.0_dp, 2.0_dp, 3.0_dp]
        call model%initialize([1, 2, 1], status, initialization_seed=73)
        allocate(gradient(model%parameter_count()), plus_gradient(model%parameter_count()))
        call mlp_loss_value_gradient(model, x, target, 0.0_dp, value, gradient, &
            l2_gradient, status, sample_weight=weights, diagnostics=diagnostics)
        theta = model%parameters()
        h = 2.0e-6_dp
        theta(1) = theta(1) + h
        call model%set_parameters(theta, status)
        call mlp_loss_value_gradient(model, x, target, 0.0_dp, plus, plus_gradient, &
            l2_gradient, status, sample_weight=weights)
        theta(1) = theta(1) - 2.0_dp*h
        call model%set_parameters(theta, status)
        call mlp_loss_value_gradient(model, x, target, 0.0_dp, minus, plus_gradient, &
            l2_gradient, status, sample_weight=weights)
        finite_gradient = (plus - minus)/(2.0_dp*h)
        call check(status_ok(status) .and. diagnostics%weight_mass == sum(weights) .and. &
            abs(gradient(1) - finite_gradient) < 2.0e-7_dp, &
            "MLP objective uses weighted-MSE derivative products", failures)
    end subroutine test_mlp_weighted_integration

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
        end if
    end subroutine check

end program test_neural_losses
