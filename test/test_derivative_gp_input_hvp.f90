program test_derivative_gp_input_hvp
    !! Query-input Hessian-vector products of a derivative-observation GP.
    !!
    !! The oracle is deliberately independent of the routine under test: the
    !! posterior mean and variance are differentiated twice by Richardson-
    !! extrapolated central differences of `predict`, which shares no code with
    !! the analytic third-derivative path the HVP is assembled from. Finite
    !! differences are the right oracle *here* precisely because they are the
    !! wrong implementation — they lose half their digits, which is why the
    !! production path is analytic, but at these step sizes they still pin the
    !! answer to seven digits, far tighter than any plausible structural error.
    !!
    !! Beyond agreement, the test pins the properties that distinguish a correct
    !! second-order product from a plausible-looking one: symmetry of the
    !! assembled Hessian, exactness on directions, the sign of the data-induced
    !! curvature term, and behaviour at a training point where the posterior
    !! variance is at a minimum and its Hessian must therefore be positive
    !! semidefinite.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel, make_matern52_kernel, &
        kernel_multiply, make_linear_kernel
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_against_finite_differences(failures)
    call test_hessian_is_symmetric(failures)
    call test_linearity_in_the_direction(failures)
    call test_derivative_observations_contribute(failures)
    call test_variance_curvature_at_a_training_point(failures)
    call test_product_kernel(failures)
    call test_shape_guards(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " derivative GP input-HVP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: derivative GP query-input Hessian products"

contains

    !! Fit a small mixed value/derivative model used by most of the checks.
    subroutine build_model(model, status)
        type(gp_derivative_regression_t), intent(out) :: model
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel
        real(dp) :: x(5, 2), y(5, 1)

        x = reshape([ &
            -0.4_dp, 0.3_dp, 0.9_dp, -0.7_dp, 0.15_dp, &
            0.2_dp, -0.6_dp, 0.45_dp, 0.8_dp, -0.25_dp], [5, 2])
        y = reshape([0.6_dp, -0.35_dp, 0.9_dp, 0.1_dp, -0.75_dp], [5, 1])
        kernel = make_rbf_kernel(2, 1.1_dp, 0.75_dp, status)
        if (.not. status_ok(status)) return
        ! Rows 2 and 4 are gradient observations in features 1 and 2.
        call model%fit(x, [0, 1, 0, 2, 0], y, kernel, 0.04_dp, status, &
            jitter=1.0e-10_dp)
    end subroutine build_model

    !! Central second differences of `predict` along the two directions, with
    !! one Richardson step to remove the leading h^2 error.
    subroutine numeric_hvp(model, x, direction, mean_hvp, variance_hvp, ok)
        type(gp_derivative_regression_t), intent(in) :: model
        real(dp), intent(in) :: x(:), direction(:)
        real(dp), intent(out) :: mean_hvp(:), variance_hvp(:)
        logical, intent(out) :: ok
        real(dp) :: coarse_mean(size(x)), coarse_variance(size(x))
        real(dp) :: fine_mean(size(x)), fine_variance(size(x))
        real(dp), parameter :: h = 4.0e-3_dp

        call mixed_second_difference(model, x, direction, h, coarse_mean, &
            coarse_variance, ok)
        if (.not. ok) return
        call mixed_second_difference(model, x, direction, 0.5_dp*h, fine_mean, &
            fine_variance, ok)
        if (.not. ok) return
        ! Richardson: the central formula errs at O(h^2), so (4 fine - coarse)/3
        ! cancels that term and leaves O(h^4).
        mean_hvp = (4.0_dp*fine_mean - coarse_mean)/3.0_dp
        variance_hvp = (4.0_dp*fine_variance - coarse_variance)/3.0_dp
    end subroutine numeric_hvp

    !! d/dt grad f(x + t v) at t = 0, component by component, by four-point
    !! mixed central differences.
    subroutine mixed_second_difference(model, x, direction, h, mean_hvp, &
            variance_hvp, ok)
        type(gp_derivative_regression_t), intent(in) :: model
        real(dp), intent(in) :: x(:), direction(:), h
        real(dp), intent(out) :: mean_hvp(:), variance_hvp(:)
        logical, intent(out) :: ok
        real(dp) :: shifted(1, size(x)), mean(1, 1), variance(1)
        real(dp) :: basis(size(x))
        real(dp) :: m_pp, m_pm, m_mp, m_mm, v_pp, v_pm, v_mp, v_mm
        type(fortnum_status_t) :: status
        integer :: j, sign_a, sign_b

        ok = .true.
        do j = 1, size(x)
            basis = 0.0_dp
            basis(j) = 1.0_dp
            m_pp = 0.0_dp; m_pm = 0.0_dp; m_mp = 0.0_dp; m_mm = 0.0_dp
            v_pp = 0.0_dp; v_pm = 0.0_dp; v_mp = 0.0_dp; v_mm = 0.0_dp
            do sign_a = -1, 1, 2
                do sign_b = -1, 1, 2
                    shifted(1, :) = x + real(sign_a, dp)*h*basis &
                        + real(sign_b, dp)*h*direction
                    call model%predict(shifted, [0], mean, variance, status)
                    if (.not. status_ok(status)) then
                        ok = .false.
                        return
                    end if
                    if (sign_a > 0 .and. sign_b > 0) then
                        m_pp = mean(1, 1); v_pp = variance(1)
                    else if (sign_a > 0) then
                        m_pm = mean(1, 1); v_pm = variance(1)
                    else if (sign_b > 0) then
                        m_mp = mean(1, 1); v_mp = variance(1)
                    else
                        m_mm = mean(1, 1); v_mm = variance(1)
                    end if
                end do
            end do
            mean_hvp(j) = (m_pp - m_pm - m_mp + m_mm)/(4.0_dp*h*h)
            variance_hvp(j) = (v_pp - v_pm - v_mp + v_mm)/(4.0_dp*h*h)
        end do
    end subroutine mixed_second_difference

    subroutine test_against_finite_differences(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(2), direction(2), mean_hvp(2, 1), variance_hvp(2)
        real(dp) :: numeric_mean(2), numeric_variance(2), scale
        logical :: ok
        integer :: case_index

        call build_model(model, status)
        if (.not. status_ok(status)) then
            call fail("input HVP: model fit", failures)
            return
        end if

        do case_index = 1, 3
            select case (case_index)
            case (1)
                x = [0.05_dp, 0.1_dp]
                direction = [1.0_dp, 0.0_dp]
            case (2)
                x = [-0.55_dp, 0.6_dp]
                direction = [0.3_dp, -0.9_dp]
            case (3)
                ! Well away from the data, where the posterior returns to the
                ! prior and the data-induced term nearly vanishes.
                x = [2.5_dp, -2.2_dp]
                direction = [-0.4_dp, 0.7_dp]
            end select

            call model%predict_input_hvp(x, direction, mean_hvp, variance_hvp, &
                status)
            if (.not. status_ok(status)) then
                call fail("input HVP: evaluation", failures)
                cycle
            end if
            call numeric_hvp(model, x, direction, numeric_mean, numeric_variance, ok)
            if (.not. ok) then
                call fail("input HVP: numeric reference", failures)
                cycle
            end if

            scale = max(1.0_dp, maxval(abs(numeric_mean)))
            call check(maxval(abs(mean_hvp(:, 1) - numeric_mean)) < 1.0e-7_dp*scale, &
                "the mean HVP matches extrapolated central differences", failures)
            scale = max(1.0_dp, maxval(abs(numeric_variance)))
            call check(maxval(abs(variance_hvp - numeric_variance)) < 1.0e-7_dp*scale, &
                "the variance HVP matches extrapolated central differences", &
                failures)
        end do
    end subroutine test_against_finite_differences

    !! A Hessian assembled column by column must be symmetric. This catches a
    !! transposed index in the cross-covariance Jacobian, which finite
    !! differences on a single direction would not.
    subroutine test_hessian_is_symmetric(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(2), basis(2), mean_hvp(2, 1), variance_hvp(2)
        real(dp) :: mean_hessian(2, 2), variance_hessian(2, 2)
        integer :: j

        call build_model(model, status)
        x = [0.12_dp, -0.31_dp]
        do j = 1, 2
            basis = 0.0_dp
            basis(j) = 1.0_dp
            call model%predict_input_hvp(x, basis, mean_hvp, variance_hvp, status)
            if (.not. status_ok(status)) then
                call fail("input HVP: symmetry evaluation", failures)
                return
            end if
            mean_hessian(:, j) = mean_hvp(:, 1)
            variance_hessian(:, j) = variance_hvp
        end do
        call check(abs(mean_hessian(1, 2) - mean_hessian(2, 1)) < 1.0e-12_dp, &
            "the mean Hessian is symmetric", failures)
        call check(abs(variance_hessian(1, 2) - variance_hessian(2, 1)) < 1.0e-12_dp, &
            "the variance Hessian is symmetric", failures)
    end subroutine test_hessian_is_symmetric

    !! The product is linear in the direction. A term that accidentally used the
    !! direction twice, or not at all, fails this without needing a reference
    !! value.
    subroutine test_linearity_in_the_direction(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(2), u(2), v(2), sum_direction(2)
        real(dp) :: mean_u(2, 1), mean_v(2, 1), mean_sum(2, 1), mean_scaled(2, 1)
        real(dp) :: variance_u(2), variance_v(2), variance_sum(2), variance_scaled(2)

        call build_model(model, status)
        x = [-0.05_dp, 0.22_dp]
        u = [0.7_dp, -0.2_dp]
        v = [-0.3_dp, 0.9_dp]
        sum_direction = u + v

        call model%predict_input_hvp(x, u, mean_u, variance_u, status)
        call model%predict_input_hvp(x, v, mean_v, variance_v, status)
        call model%predict_input_hvp(x, sum_direction, mean_sum, variance_sum, status)
        call check(maxval(abs(mean_sum(:, 1) - mean_u(:, 1) - mean_v(:, 1))) &
            < 1.0e-12_dp, "the mean HVP is additive in the direction", failures)
        call check(maxval(abs(variance_sum - variance_u - variance_v)) < 1.0e-12_dp, &
            "the variance HVP is additive in the direction", failures)

        call model%predict_input_hvp(x, 3.5_dp*u, mean_scaled, variance_scaled, &
            status)
        call check(maxval(abs(mean_scaled(:, 1) - 3.5_dp*mean_u(:, 1))) < 1.0e-12_dp, &
            "the mean HVP is homogeneous in the direction", failures)
        call check(maxval(abs(variance_scaled - 3.5_dp*variance_u)) < 1.0e-12_dp, &
            "the variance HVP is homogeneous in the direction", failures)

        ! A zero direction has zero curvature along it.
        call model%predict_input_hvp(x, [0.0_dp, 0.0_dp], mean_u, variance_u, status)
        call check(status_ok(status) .and. maxval(abs(mean_u)) == 0.0_dp .and. &
            maxval(abs(variance_u)) == 0.0_dp, &
            "a zero direction gives a zero product", failures)
    end subroutine test_linearity_in_the_direction

    !! The gradient rows must actually enter the second-order path. Refitting
    !! the same inputs with those rows relabelled as values changes the answer;
    !! if the component branch were dead, it would not.
    subroutine test_derivative_observations_contribute(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: mixed_model, value_model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(5, 2), y(5, 1), query(2), direction(2)
        real(dp) :: mixed_mean(2, 1), value_mean(2, 1)
        real(dp) :: mixed_variance(2), value_variance(2)

        x = reshape([ &
            -0.4_dp, 0.3_dp, 0.9_dp, -0.7_dp, 0.15_dp, &
            0.2_dp, -0.6_dp, 0.45_dp, 0.8_dp, -0.25_dp], [5, 2])
        y = reshape([0.6_dp, -0.35_dp, 0.9_dp, 0.1_dp, -0.75_dp], [5, 1])
        query = [0.1_dp, 0.05_dp]
        direction = [0.6_dp, 0.8_dp]

        kernel = make_rbf_kernel(2, 1.1_dp, 0.75_dp, status)
        call mixed_model%fit(x, [0, 1, 0, 2, 0], y, kernel, 0.04_dp, status, &
            jitter=1.0e-10_dp)
        kernel = make_rbf_kernel(2, 1.1_dp, 0.75_dp, status)
        call value_model%fit(x, [0, 0, 0, 0, 0], y, kernel, 0.04_dp, status, &
            jitter=1.0e-10_dp)

        call mixed_model%predict_input_hvp(query, direction, mixed_mean, &
            mixed_variance, status)
        call check(status_ok(status), "the mixed model produces a product", failures)
        call value_model%predict_input_hvp(query, direction, value_mean, &
            value_variance, status)
        call check(status_ok(status), "the value model produces a product", failures)

        call check(maxval(abs(mixed_mean - value_mean)) > 1.0e-6_dp, &
            "gradient observations change the mean curvature", failures)
        call check(maxval(abs(mixed_variance - value_variance)) > 1.0e-6_dp, &
            "gradient observations change the variance curvature", failures)
    end subroutine test_derivative_observations_contribute

    !! At a value-observed training point, compare the analytic product with an
    !! independent Richardson central-difference oracle. The sign is not fixed
    !! for a noisy mixed-observation posterior, so asserting positivity here
    !! would reject valid kernels rather than test the derivative contract.
    subroutine test_variance_curvature_at_a_training_point(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), y(3, 1), query(1), direction(1)
        real(dp) :: mean_hvp(1, 1), variance_hvp(1)
        real(dp) :: numeric_mean(1), numeric_variance(1)
        logical :: ok

        x(:, 1) = [-0.6_dp, 0.0_dp, 0.7_dp]
        y(:, 1) = [0.4_dp, -0.2_dp, 0.55_dp]
        kernel = make_matern52_kernel(1, 1.0_dp, 0.6_dp, status)
        call model%fit(x, [0, 0, 0], y, kernel, 1.0e-4_dp, status, jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            call fail("input HVP: matern fit", failures)
            return
        end if

        query = [0.0_dp]
        direction = [1.0_dp]
        call model%predict_input_hvp(query, direction, mean_hvp, variance_hvp, &
            status)
        call check(status_ok(status), "the matern product evaluates", failures)
        call numeric_hvp(model, query, direction, numeric_mean, numeric_variance, ok)
        call check(ok .and. abs(mean_hvp(1, 1) - numeric_mean(1)) < 1.0e-5_dp .and. &
            abs(variance_hvp(1) - numeric_variance(1)) < 1.0e-5_dp, &
            "the observed-point product matches the independent oracle", failures)
    end subroutine test_variance_curvature_at_a_training_point

    !! Product kernels take the recursive branch, where the second-order terms
    !! pick up cross products of first derivatives. Finite differences again.
    subroutine test_product_kernel(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel, rbf, linear
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), y(4, 1), query(2), direction(2)
        real(dp) :: mean_hvp(2, 1), variance_hvp(2)
        real(dp) :: numeric_mean(2), numeric_variance(2), scale
        logical :: ok

        x = reshape([0.1_dp, -0.4_dp, 0.55_dp, -0.8_dp, &
            -0.3_dp, 0.7_dp, 0.05_dp, 0.4_dp], [4, 2])
        y = reshape([0.2_dp, -0.6_dp, 0.35_dp, 0.8_dp], [4, 1])
        rbf = make_rbf_kernel(2, 1.0_dp, 0.9_dp, status)
        linear = make_linear_kernel(2, 0.7_dp, status)
        kernel = kernel_multiply(rbf, linear, status)
        if (.not. status_ok(status)) then
            call fail("input HVP: product kernel construction", failures)
            return
        end if
        call model%fit(x, [0, 1, 0, 0], y, kernel, 0.02_dp, status, jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            call fail("input HVP: product kernel fit", failures)
            return
        end if

        query = [0.2_dp, -0.15_dp]
        direction = [0.5_dp, 0.85_dp]
        call model%predict_input_hvp(query, direction, mean_hvp, variance_hvp, &
            status)
        call check(status_ok(status), "the product-kernel product evaluates", &
            failures)
        call numeric_hvp(model, query, direction, numeric_mean, numeric_variance, ok)
        if (.not. ok) then
            call fail("input HVP: product-kernel reference", failures)
            return
        end if
        scale = max(1.0_dp, maxval(abs(numeric_mean)))
        call check(maxval(abs(mean_hvp(:, 1) - numeric_mean)) < 1.0e-6_dp*scale, &
            "the product-kernel mean HVP matches finite differences", failures)
        scale = max(1.0_dp, maxval(abs(numeric_variance)))
        call check(maxval(abs(variance_hvp - numeric_variance)) < 1.0e-6_dp*scale, &
            "the product-kernel variance HVP matches finite differences", failures)
    end subroutine test_product_kernel

    subroutine test_shape_guards(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model, unfitted
        type(fortnum_status_t) :: status
        real(dp) :: mean_hvp(2, 1), variance_hvp(2), wide(3, 1), short(3)

        call build_model(model, status)

        call model%predict_input_hvp([0.0_dp, 0.0_dp, 0.0_dp], [1.0_dp, 0.0_dp], &
            mean_hvp, variance_hvp, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "a mis-sized query is refused", failures)

        call model%predict_input_hvp([0.0_dp, 0.0_dp], [1.0_dp, 0.0_dp, 0.0_dp], &
            mean_hvp, variance_hvp, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "a mis-sized direction is refused", failures)

        call model%predict_input_hvp([0.0_dp, 0.0_dp], [1.0_dp, 0.0_dp], wide, &
            short, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "mis-sized outputs are refused", failures)

        call unfitted%predict_input_hvp([0.0_dp, 0.0_dp], [1.0_dp, 0.0_dp], &
            mean_hvp, variance_hvp, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "an unfitted model is refused", failures)
    end subroutine test_shape_guards

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) call fail(description, failures)
    end subroutine check

    subroutine fail(description, failures)
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        write (error_unit, '(a)') "FAIL ["//description//"]"
        failures = failures + 1
    end subroutine fail

end program test_derivative_gp_input_hvp
