program test_fortsym_matern32
    !! Independent numerical oracle for the FortSym-generated Matérn-3/2 leaf.
    !! The oracle evaluates the closed form and central differences its value
    !! and gradient; it never calls the generated routine for either reference.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_generated_matern32_products, only: fortml_matern32_hvp
    implicit none

    integer, parameter :: dp = real64
    real(dp), parameter :: kappa = 1.7320508075688772935_dp
    real(dp), parameter :: distance = 0.73_dp, distance_d = 0.21_dp
    real(dp), parameter :: lv = log(1.4_dp), lv_d = -0.17_dp
    real(dp), parameter :: ll = log(0.82_dp), ll_d = 0.11_dp
    real(dp), parameter :: y_bar = 0.83_dp, h = 2.0e-6_dp
    real(dp) :: y, y_d, distance_bar, distance_bar_d, lv_bar, lv_bar_d
    real(dp) :: ll_bar, ll_bar_d, expected, expected_d, expected_distance_bar
    real(dp) :: expected_lv_bar, expected_ll_bar, plus(3), minus(3)
    real(dp) :: gradient_plus(3), gradient_minus(3), gradient_fd(3)
    integer :: failures

    failures = 0
    call fortml_matern32_hvp(distance, distance_d, lv, lv_d, ll, ll_d, y, y_d, &
        y_bar, distance_bar, distance_bar_d, lv_bar, lv_bar_d, ll_bar, ll_bar_d)

    expected = matern32_value(distance, lv, ll)
    expected_d = (matern32_value(distance + h*distance_d, lv + h*lv_d, &
        ll + h*ll_d) - matern32_value(distance - h*distance_d, lv - h*lv_d, &
        ll - h*ll_d))/(2.0_dp*h)
    call check(abs(y - expected) < 2.0e-14_dp, &
        'value closed-form oracle', failures)
    call check(abs(y_d - expected_d) < 2.0e-9_dp, &
        'value JVP independent central difference', failures)

    call matern32_gradient(distance, lv, ll, y_bar, expected_distance_bar, &
        expected_lv_bar, expected_ll_bar)
    call check(maxval(abs([distance_bar, lv_bar, ll_bar] - &
        [expected_distance_bar, expected_lv_bar, expected_ll_bar])) < 3.0e-14_dp, &
        'reverse gradient closed-form oracle', failures)

    plus = [distance + h*distance_d, lv + h*lv_d, ll + h*ll_d]
    minus = [distance - h*distance_d, lv - h*lv_d, ll - h*ll_d]
    call matern32_gradient(plus(1), plus(2), plus(3), y_bar, gradient_plus(1), &
        gradient_plus(2), gradient_plus(3))
    call matern32_gradient(minus(1), minus(2), minus(3), y_bar, gradient_minus(1), &
        gradient_minus(2), gradient_minus(3))
    gradient_fd = (gradient_plus - gradient_minus)/(2.0_dp*h)
    call check(maxval(abs([distance_bar_d, lv_bar_d, ll_bar_d] - gradient_fd)) < &
        3.0e-8_dp, 'reverse tangent independent central difference', failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') 'FAIL FortSym Matern32 cases: ', failures
        error stop 1
    end if
    write (*, '(a)') 'PASS FortSym-generated Matern32 independent oracles'

contains

    real(dp) function matern32_value(d, log_variance, log_lengthscale) result(value)
        real(dp), intent(in) :: d, log_variance, log_lengthscale
        real(dp) :: r

        r = kappa*d*exp(-log_lengthscale)
        value = exp(log_variance)*(1.0_dp + r)*exp(-r)
    end function matern32_value

    subroutine matern32_gradient(d, log_variance, log_lengthscale, cotangent, &
            d_bar, lv_bar_out, ll_bar_out)
        real(dp), intent(in) :: d, log_variance, log_lengthscale, cotangent
        real(dp), intent(out) :: d_bar, lv_bar_out, ll_bar_out
        real(dp) :: r, exp_lv, exp_minus_r, dvalue_dr

        r = kappa*d*exp(-log_lengthscale)
        exp_lv = exp(log_variance)
        exp_minus_r = exp(-r)
        dvalue_dr = -exp_lv*r*exp_minus_r
        d_bar = cotangent*dvalue_dr*kappa*exp(-log_lengthscale)
        lv_bar_out = cotangent*exp_lv*(1.0_dp + r)*exp_minus_r
        ll_bar_out = cotangent*dvalue_dr*(-r)
    end subroutine matern32_gradient

    subroutine check(condition, name, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: name
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') '  FAIL [fortsym-matern32] '//name
        end if
    end subroutine check

end program test_fortsym_matern32
