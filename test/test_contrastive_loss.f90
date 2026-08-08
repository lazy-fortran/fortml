program test_contrastive_loss
    !! Independent value/product oracles for pairwise contrastive loss.
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_losses, only: contrastive_loss_value, contrastive_loss_jvp, &
        contrastive_loss_vjp, contrastive_loss_hvp, contrastive_loss_value_device, &
        LOSS_REDUCTION_SUM
    implicit none

    integer :: failures

    failures = 0
    call test_products(failures)
    call test_reductions_and_device(failures)
    call test_nondifferentiable_boundaries(failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL contrastive loss cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS contrastive loss independent behavioral oracles"

contains

    subroutine test_products(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: a(3, 2), b(3, 2), a_dot(3, 2), b_dot(3, 2)
        real(dp) :: a_bar(3, 2), b_bar(3, 2), a_hvp(3, 2), b_hvp(3, 2)
        real(dp) :: value, value_dot, value_plus, value_minus, finite_dot
        real(dp) :: a_bar_plus(3, 2), b_bar_plus(3, 2)
        real(dp) :: a_bar_minus(3, 2), b_bar_minus(3, 2)
        real(dp) :: labels_weight(3), margin, h, lhs, rhs
        integer :: labels(3)

        a = reshape([0.2_dp, -0.1_dp, 1.1_dp, 0.4_dp, -0.7_dp, 1.2_dp], shape(a))
        b = reshape([-0.1_dp, 0.3_dp, 0.2_dp, 0.8_dp, -0.4_dp, 0.9_dp], shape(b))
        a_dot = reshape([0.13_dp, -0.21_dp, 0.07_dp, 0.11_dp, -0.17_dp, &
            0.23_dp], shape(a_dot))
        b_dot = reshape([-0.09_dp, 0.14_dp, -0.16_dp, 0.05_dp, 0.12_dp, &
            -0.08_dp], shape(b_dot))
        labels = [1, 0, 0]
        labels_weight = [1.0_dp, 2.0_dp, 3.0_dp]
        margin = 1.3_dp
        h = 2.0e-6_dp

        call contrastive_loss_value(a, b, labels, margin, value, status, &
            labels_weight)
        call check(status_ok(status) .and. abs(value - reference_value(a, b, labels, &
            margin, labels_weight, .false.)) < 3.0e-14_dp, &
            "contrastive weighted value independent oracle", failures)
        call contrastive_loss_jvp(a, b, labels, margin, a_dot, b_dot, value, &
            value_dot, status, labels_weight)
        call contrastive_loss_value(a + h*a_dot, b + h*b_dot, labels, margin, &
            value_plus, status, labels_weight)
        call contrastive_loss_value(a - h*a_dot, b - h*b_dot, labels, margin, &
            value_minus, status, labels_weight)
        finite_dot = (value_plus - value_minus)/(2.0_dp*h)
        call check(status_ok(status) .and. abs(value_dot - finite_dot) < 3.0e-9_dp, &
            "contrastive weighted JVP finite difference", failures)

        call contrastive_loss_vjp(a, b, labels, margin, -0.8_dp, a_bar, b_bar, &
            status, labels_weight)
        lhs = sum(a_bar*a_dot) + sum(b_bar*b_dot)
        rhs = -0.8_dp*value_dot
        call check(status_ok(status) .and. abs(lhs - rhs) < 3.0e-14_dp, &
            "contrastive VJP adjoint identity", failures)

        call contrastive_loss_hvp(a, b, labels, margin, a_dot, b_dot, a_hvp, &
            b_hvp, status, labels_weight)
        call contrastive_loss_vjp(a + h*a_dot, b + h*b_dot, labels, margin, 1.0_dp, &
            a_bar_plus, b_bar_plus, status, labels_weight)
        call contrastive_loss_vjp(a - h*a_dot, b - h*b_dot, labels, margin, 1.0_dp, &
            a_bar_minus, b_bar_minus, status, labels_weight)
        call check(status_ok(status) .and. maxval(abs(a_hvp - &
            (a_bar_plus - a_bar_minus)/(2.0_dp*h))) < 4.0e-8_dp .and. &
            maxval(abs(b_hvp - (b_bar_plus - b_bar_minus)/(2.0_dp*h))) < 4.0e-8_dp, &
            "contrastive HVP finite difference", failures)
    end subroutine test_products

    subroutine test_reductions_and_device(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: a(2, 2), b(2, 2), weights(2), value, sum_value
        integer :: labels(2)

        a = reshape([0.0_dp, 0.0_dp, 0.4_dp, 0.0_dp], shape(a))
        b = reshape([0.0_dp, 0.5_dp, 0.0_dp, 0.0_dp], shape(b))
        labels = [1, 0]
        weights = [2.0_dp, 5.0_dp]
        call contrastive_loss_value(a, b, labels, 1.0_dp, value, status, weights)
        call contrastive_loss_value(a, b, labels, 1.0_dp, sum_value, status, weights, &
            LOSS_REDUCTION_SUM)
        call check(status_ok(status) .and. abs(sum_value - value*sum(weights)) < &
            3.0e-14_dp, "contrastive mean/sum reduction", failures)

        value = 9.0_dp
        call contrastive_loss_value_device(a, b, labels, 1.0_dp, value, status, &
            FORTML_DEVICE_CUDA, weights)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. value == 0.0_dp, &
            "contrastive typed CUDA refusal", failures)
    end subroutine test_reductions_and_device

    subroutine test_nondifferentiable_boundaries(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: a(1, 2), b(1, 2), direction(1, 2), output(1, 2)
        real(dp) :: output_b(1, 2)
        real(dp) :: value, value_dot
        integer :: labels(1)

        a = 0.0_dp
        b = 0.0_dp
        direction = 1.0_dp
        labels = 0
        call contrastive_loss_jvp(a, b, labels, 1.0_dp, direction, direction, &
            value, value_dot, status)
        call check(.not. status_ok(status), &
            "contrastive non-matching zero-distance refusal", failures)

        a = reshape([1.0_dp, 0.0_dp], shape(a))
        call contrastive_loss_hvp(a, b, labels, 1.0_dp, direction, direction, &
            output, output_b, status)
        call check(.not. status_ok(status), "contrastive margin-kink refusal", failures)
    end subroutine test_nondifferentiable_boundaries

    real(dp) function reference_value(a, b, labels, margin, weights, sum_reduction) &
            result(value)
        real(dp), intent(in) :: a(:, :), b(:, :), margin, weights(:)
        integer, intent(in) :: labels(:)
        logical, intent(in) :: sum_reduction
        real(dp) :: distance, gap
        integer :: i

        value = 0.0_dp
        do i = 1, size(labels)
            distance = sqrt(sum((a(i, :) - b(i, :))**2))
            if (labels(i) == 1) then
                value = value + weights(i)*0.5_dp*distance*distance
            else
                gap = max(0.0_dp, margin - distance)
                value = value + weights(i)*0.5_dp*gap*gap
            end if
        end do
        if (.not. sum_reduction) value = value/sum(weights)
    end function reference_value

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL: "//trim(description)
        end if
    end subroutine check

end program test_contrastive_loss
