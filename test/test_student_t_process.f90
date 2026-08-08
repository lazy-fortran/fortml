program test_student_t_process
    !! Student-t process regression.
    !!
    !! Oracles:
    !!
    !!   * **the large-nu limit is an exact GP.** As the degrees of freedom
    !!     grow, a TP converges to the GP with the same kernel, so an
    !!     independently fitted `gp_regression_t` is the reference for both the
    !!     mean and the variance. This is the check that pins the paper's
    !!     unusual parameterization: it uses `cov = K` directly, and a
    !!     scale-matrix convention would inflate every variance by
    !!     `nu/(nu - 2)`, which this limit would expose;
    !!   * **the mean is the GP's exactly, at every nu.** The inverse Wishart
    !!     marginalization touches only the covariance, so any nu-dependence in
    !!     the mean would be a mistake;
    !!   * **the covariance responds to the data**, which is the whole point.
    !!     Two fits on the same inputs with differently surprising outputs must
    !!     give different predictive variances — something a GP cannot do.

    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_student_t_process, only: student_t_process_t
    implicit none

    integer :: failures

    failures = 0
    call test_large_dof_matches_a_gaussian_process(failures)
    call test_mean_is_the_gaussian_process_mean(failures)
    call test_covariance_responds_to_the_data(failures)
    call test_posterior_dof_and_scale(failures)
    call test_refusals(failures)

    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " student-t process test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS: student-t process"

contains

    subroutine training_set(x, y)
        real(dp), intent(out) :: x(7, 1)
        real(dp), intent(out) :: y(7)
        integer :: k

        do k = 1, 7
            x(k, 1) = -1.5_dp + 0.5_dp*real(k - 1, dp)
            y(k) = sin(1.7_dp*x(k, 1)) + 0.25_dp*x(k, 1)
        end do
    end subroutine training_set

    !! The limit that pins the parameterization. A scale-matrix convention would
    !! make every TP variance larger than the GP's by nu/(nu - 2), which at
    !! nu = 1e6 is one part in 1e6 — small, but far above the tolerance here,
    !! and it would not vanish.
    subroutine test_large_dof_matches_a_gaussian_process(failures)
        integer, intent(inout) :: failures
        type(student_t_process_t) :: process
        type(gp_regression_t) :: reference
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(7, 1), y(7), query(5, 1)
        real(dp) :: t_mean(5), t_variance(5)
        real(dp) :: g_mean(5, 1), g_variance(5)
        integer :: k

        call training_set(x, y)
        do k = 1, 5
            query(k, 1) = -1.2_dp + 0.6_dp*real(k - 1, dp)
        end do

        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call process%fit(x, y, kernel, 1.0e6_dp, 0.01_dp, status)
        call check(status_ok(status), "the process fits at large nu", failures)
        call process%predict(query, t_mean, t_variance, status)
        call check(status_ok(status), "the process predicts", failures)

        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call reference%fit(x, reshape(y, [7, 1]), kernel, 0.01_dp, status)
        call check(status_ok(status), "the reference GP fits", failures)
        call reference%predict(query, g_mean, g_variance, status)

        call check(maxval(abs(t_mean - g_mean(:, 1))) < 1.0e-10_dp, &
            "the large-nu mean matches the GP's", failures)
        call check(maxval(abs(t_variance - g_variance)) &
            < 1.0e-4_dp*maxval(abs(g_variance)), &
            "the large-nu variance matches the GP's", failures)
    end subroutine test_large_dof_matches_a_gaussian_process

    !! The marginalization touches only the covariance, so the mean must not
    !! move with nu at all.
    subroutine test_mean_is_the_gaussian_process_mean(failures)
        integer, intent(inout) :: failures
        type(student_t_process_t) :: low, high
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(7, 1), y(7), query(4, 1)
        real(dp) :: low_mean(4), low_variance(4)
        real(dp) :: high_mean(4), high_variance(4)
        integer :: k

        call training_set(x, y)
        do k = 1, 4
            query(k, 1) = -1.0_dp + 0.7_dp*real(k - 1, dp)
        end do

        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call low%fit(x, y, kernel, 3.0_dp, 0.01_dp, status)
        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call high%fit(x, y, kernel, 200.0_dp, 0.01_dp, status)

        call low%predict(query, low_mean, low_variance, status)
        call high%predict(query, high_mean, high_variance, status)

        call check(maxval(abs(low_mean - high_mean)) < 1.0e-12_dp, &
            "the predictive mean does not depend on nu", failures)
        call check(maxval(abs(low_variance - high_variance)) > 1.0e-8_dp, &
            "the predictive variance does depend on nu", failures)
    end subroutine test_mean_is_the_gaussian_process_mean

    !! The property a GP does not have. Same inputs, same kernel, same noise;
    !! only the observed values differ, and the predictive variance moves.
    subroutine test_covariance_responds_to_the_data(failures)
        integer, intent(inout) :: failures
        type(student_t_process_t) :: quiet, surprising
        type(gp_regression_t) :: gp_quiet, gp_surprising
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(7, 1), y(7), calm(7), wild(7), query(3, 1)
        real(dp) :: quiet_mean(3), quiet_variance(3)
        real(dp) :: wild_mean(3), wild_variance(3)
        real(dp) :: gp_mean(3, 1), gp_a(3), gp_b(3)
        integer :: k

        call training_set(x, y)
        do k = 1, 3
            query(k, 1) = -0.8_dp + 0.8_dp*real(k - 1, dp)
            calm(k) = 0.0_dp
        end do
        ! Values close to the prior mean, and values far from it.
        calm = 0.02_dp*y
        wild = 6.0_dp*y

        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call quiet%fit(x, calm, kernel, 4.0_dp, 0.01_dp, status)
        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call surprising%fit(x, wild, kernel, 4.0_dp, 0.01_dp, status)

        call quiet%predict(query, quiet_mean, quiet_variance, status)
        call surprising%predict(query, wild_mean, wild_variance, status)

        call check(all(wild_variance > quiet_variance), &
            "surprising data widen the predictive variance", failures)
        call check(surprising%covariance_scale() > quiet%covariance_scale(), &
            "the covariance scale tracks the Mahalanobis distance", failures)
        call check(quiet%covariance_scale() < 1.0_dp, &
            "data tamer than the prior expected shrink the variance", failures)

        ! A GP cannot do this: its posterior variance is fixed by the inputs.
        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call gp_quiet%fit(x, reshape(calm, [7, 1]), kernel, 0.01_dp, status)
        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call gp_surprising%fit(x, reshape(wild, [7, 1]), kernel, 0.01_dp, status)
        call gp_quiet%predict(query, gp_mean, gp_a, status)
        call gp_surprising%predict(query, gp_mean, gp_b, status)
        call check(maxval(abs(gp_a - gp_b)) < 1.0e-12_dp, &
            "a GP's variance ignores the observed values, as the contrast needs", &
            failures)
    end subroutine test_covariance_responds_to_the_data

    subroutine test_posterior_dof_and_scale(failures)
        integer, intent(inout) :: failures
        type(student_t_process_t) :: process
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(7, 1), y(7)

        call training_set(x, y)
        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call process%fit(x, y, kernel, 5.0_dp, 0.01_dp, status)

        ! Every observation adds one degree of freedom.
        call check(abs(process%posterior_dof() - 12.0_dp) < 1.0e-12_dp, &
            "the posterior degrees of freedom are nu plus the sample count", &
            failures)
        call check(process%covariance_scale() > 0.0_dp, &
            "the covariance scale is positive", failures)
    end subroutine test_posterior_dof_and_scale

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(student_t_process_t) :: process, unfitted
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(7, 1), y(7), query(2, 1), mean(2), variance(2)

        call training_set(x, y)
        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)

        ! At or below two degrees of freedom the covariance does not exist.
        call process%fit(x, y, kernel, 2.0_dp, 0.01_dp, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "nu at two is refused, since the covariance is undefined there", &
            failures)

        kernel = make_rbf_kernel(1, 1.0_dp, 0.8_dp, status)
        call process%fit(x, y(1:5), kernel, 5.0_dp, 0.01_dp, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "mismatched inputs and targets are refused", failures)

        query = 0.0_dp
        call unfitted%predict(query, mean, variance, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "predicting before fitting is refused", failures)
    end subroutine test_refusals

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//description//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_student_t_process
