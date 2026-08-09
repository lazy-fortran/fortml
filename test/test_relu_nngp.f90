program test_relu_nngp
    !! Independent arc-cosine recurrence oracle for the ReLU NNGP kernel.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_relu_nngp, only: relu_nngp_t, relu_nngp_metadata_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_covariance_oracle(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " ReLU NNGP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_covariance_oracle(failures)
        integer, intent(inout) :: failures
        type(relu_nngp_t) :: kernel
        type(relu_nngp_metadata_t) :: metadata
        type(fortnum_status_t) :: status
        real(dp) :: x_left(3, 2), x_right(2, 2)
        real(dp), allocatable :: actual(:, :), expected(:, :)

        x_left = reshape([-1.0_dp, 0.2_dp, 0.7_dp, 0.4_dp, -0.8_dp, 1.1_dp], &
            shape(x_left))
        x_right = reshape([0.3_dp, -0.6_dp, -0.2_dp, 0.9_dp], shape(x_right))
        call kernel%configure(2, 2, status, weight_variance=1.7_dp, &
            bias_variance=0.15_dp)
        call check(status_ok(status) .and. kernel%configured(), "configuration", failures)
        call kernel%covariance(x_left, x_right, actual, status)
        call covariance_oracle(x_left, x_right, 2, 1.7_dp, 0.15_dp, expected)
        call check(status_ok(status) .and. maxval(abs(actual - expected)) < 3.0e-14_dp, &
            "two-layer analytic covariance", failures)
        metadata = kernel%metadata()
        call check(metadata%exact_infinite_width .and. .not. metadata%cuda_supported .and. &
            .not. metadata%finite_mlp_weight_map_supported .and. &
            metadata%hidden_layer_count == 2 .and. &
            abs(metadata%weight_variance - 1.7_dp) < 1.0e-15_dp, &
            "exact-limit metadata", failures)
    end subroutine test_covariance_oracle

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(relu_nngp_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 2)
        real(dp), allocatable :: result(:, :), sentinel(:, :)

        x = 0.2_dp
        call kernel%covariance(x, x, result, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "unconfigured refusal", failures)
        call kernel%configure(2, 1, status)
        call kernel%covariance(x(:, 1:1), x, result, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "input shape refusal", failures)
        allocate(sentinel(2, 2))
        sentinel = 77.0_dp
        call kernel%covariance_cuda(x, x, sentinel, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            maxval(abs(sentinel - 77.0_dp)) == 0.0_dp, "CUDA refusal", failures)
    end subroutine test_refusals

    subroutine covariance_oracle(x_left, x_right, depth, sigma_w_squared, &
            sigma_b_squared, value)
        real(dp), intent(in) :: x_left(:, :), x_right(:, :)
        integer, intent(in) :: depth
        real(dp), intent(in) :: sigma_w_squared, sigma_b_squared
        real(dp), allocatable, intent(out) :: value(:, :)
        real(dp), allocatable :: left_variance(:), right_variance(:)
        real(dp) :: denominator, rho, theta
        integer :: i, j, layer_index

        allocate(value(size(x_left, 1), size(x_right, 1)))
        value = matmul(x_left, transpose(x_right))/real(size(x_left, 2), dp)
        left_variance = sum(x_left**2, dim=2)/real(size(x_left, 2), dp)
        right_variance = sum(x_right**2, dim=2)/real(size(x_right, 2), dp)
        do layer_index = 1, depth
            do j = 1, size(value, 2)
                do i = 1, size(value, 1)
                    denominator = sqrt(left_variance(i)*right_variance(j))
                    if (denominator == 0.0_dp) then
                        value(i, j) = sigma_b_squared
                    else
                        rho = max(-1.0_dp, min(1.0_dp, value(i, j)/denominator))
                        theta = acos(rho)
                        value(i, j) = sigma_w_squared*denominator*(sin(theta) + &
                            (acos(-1.0_dp) - theta)*rho)/(2.0_dp*acos(-1.0_dp)) + &
                            sigma_b_squared
                    end if
                end do
            end do
            left_variance = 0.5_dp*sigma_w_squared*left_variance + sigma_b_squared
            right_variance = 0.5_dp*sigma_w_squared*right_variance + sigma_b_squared
        end do
    end subroutine covariance_oracle

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [" // trim(label) // "]"
        end if
    end subroutine check

end program test_relu_nngp
