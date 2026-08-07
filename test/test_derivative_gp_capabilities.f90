program test_derivative_gp_capabilities
    !! Independent contract checks for derivative-GP capability boundaries.
    !!
    !! These are refusal tests, not implementation-shape checks: a supported
    !! value-only model must remain usable after a query-product refusal, while
    !! nonsmooth or unavailable third-input products return their documented
    !! status code instead of silently using finite differences.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_kernels, only: kernel_t, make_matern12_kernel, &
        make_matern32_kernel, make_user_kernel, make_white_noise_kernel
    use fortml_kernel_formula, only: kernel_formula_t
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    implicit none

    integer :: failures

    failures = 0
    call check_white_noise_observation_refusal(failures)
    call check_matern12_coincident_query_refusal(failures)
    call check_matern32_coincident_third_input_refusal(failures)
    call check_user_kernel_query_refusal(failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL derivative-GP capability cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS derivative-GP capability refusal oracle"

contains

    subroutine check_white_noise_observation_refusal(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), y(2, 1)

        x(:, 1) = [0.0_dp, 1.0_dp]
        y(:, 1) = [0.2_dp, -0.4_dp]
        kernel = make_white_noise_kernel(1, 0.3_dp, status)
        call model%fit(x, [1, 0], y, kernel, 0.05_dp, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "white-noise derivative observations refuse", failures)
    end subroutine check_white_noise_observation_refusal

    subroutine check_matern12_coincident_query_refusal(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), y(2, 1), query(1, 1), direction(1, 1)
        real(dp) :: mean(1, 1), mean_dot(1, 1), variance(1), variance_dot(1)
        real(dp) :: reference(1, 1), reference_variance(1)

        x(:, 1) = [0.0_dp, 1.0_dp]
        y(:, 1) = [0.2_dp, -0.4_dp]
        query(1, 1) = 0.25_dp
        direction(1, 1) = 0.1_dp
        kernel = make_matern12_kernel(1, 1.2_dp, 0.8_dp, status)
        call model%fit(x, [0, 0], y, kernel, 0.05_dp, status)
        call check(status_ok(status), "Matern 1/2 value-only fit", failures)
        call model%predict_input_jvp(query, [0], direction, mean, mean_dot, &
            variance, variance_dot, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "Matern 1/2 coincident third-input product refuses", failures)

        ! A refused product must not invalidate the fitted value-only model.
        call model%predict(query, [0], reference, reference_variance, status)
        call check(status_ok(status) .and. all(reference == reference) .and. &
            all(reference_variance == reference_variance), &
            "Matern 1/2 model remains usable after refusal", failures)
    end subroutine check_matern12_coincident_query_refusal

    subroutine check_matern32_coincident_third_input_refusal(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), y(2, 1), query(1, 1), direction(1, 1)
        real(dp) :: mean(1, 1), mean_dot(1, 1), variance(1), variance_dot(1)

        x(:, 1) = [0.0_dp, 1.0_dp]
        y(:, 1) = [0.2_dp, -0.4_dp]
        query(1, 1) = 0.0_dp
        direction(1, 1) = 0.1_dp
        kernel = make_matern32_kernel(1, 1.2_dp, 0.8_dp, status)
        call model%fit(x, [0, 0], y, kernel, 0.05_dp, status)
        call check(status_ok(status), "Matern 3/2 value-only fit", failures)
        call model%predict_input_jvp(query, [0], direction, mean, mean_dot, &
            variance, variance_dot, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "Matern 3/2 coincident third-input product refuses", failures)
    end subroutine check_matern32_coincident_third_input_refusal

    subroutine check_user_kernel_query_refusal(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_formula_t) :: formula
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), y(2, 1), query(1, 1), direction(1, 1)
        real(dp) :: mean(1, 1), mean_dot(1, 1), variance(1), variance_dot(1)

        call formula%reset()
        call formula%push_squared_distance()
        call formula%divide_by_constant(-1.28_dp)
        call formula%exponential()
        call formula%validate(status)
        kernel = make_user_kernel(1, 1.2_dp, formula, status)
        x(:, 1) = [0.0_dp, 1.0_dp]
        y(:, 1) = [0.2_dp, -0.4_dp]
        query(1, 1) = 0.25_dp
        direction(1, 1) = 0.1_dp
        call model%fit(x, [0, 0], y, kernel, 0.05_dp, status)
        call check(status_ok(status), "validated user-kernel value-only fit", failures)
        call model%predict_input_jvp(query, [0], direction, mean, mean_dot, &
            variance, variance_dot, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "user-kernel third-input product refuses", failures)
    end subroutine check_user_kernel_query_refusal

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [derivative-GP-capability] "//description
        end if
    end subroutine check

end program test_derivative_gp_capabilities
