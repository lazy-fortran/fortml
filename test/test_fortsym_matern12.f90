program test_fortsym_matern12
    ! Independent analytic and directional finite-difference oracle for the
    ! FortSym-generated Matérn-1/2 HVP leaf.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_generated_matern12_products, only: fortml_matern12_hvp
    implicit none
    integer, parameter :: dp = real64
    integer :: failures

    failures = 0
    call check_point(0.83_dp, log(1.7_dp), log(1.2_dp), 0.07_dp, -0.11_dp, &
        0.05_dp, 0.9_dp, failures)
    call check_point(2.1_dp, log(0.4_dp), log(0.65_dp), -0.13_dp, 0.08_dp, &
        -0.04_dp, 1.3_dp, failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            ' FortSym Matérn-1/2 leaf check(s) failed'
        error stop 1
    end if
    write (*, '(a)') 'PASS'

contains

    subroutine check_point(distance, lv, ll, distance_d, lv_d, ll_d, y_b, failures)
        real(dp), intent(in) :: distance, lv, ll, distance_d, lv_d, ll_d, y_b
        integer, intent(inout) :: failures
        real(dp) :: y, y_d, distance_b, distance_b_d, lv_b, lv_b_d
        real(dp) :: ll_b, ll_b_d, scale, expected, expected_d
        real(dp) :: expected_distance_b, expected_lv_b, expected_ll_b
        real(dp) :: h, plus_distance_b, minus_distance_b
        real(dp) :: plus_lv_b, minus_lv_b, plus_ll_b, minus_ll_b
        real(dp) :: ignored_y, ignored_y_d, ignored_distance_b_d
        real(dp) :: ignored_lv_b_d, ignored_ll_b_d

        call fortml_matern12_hvp(distance, distance_d, lv, lv_d, ll, ll_d, &
            y, y_d, y_b, distance_b, distance_b_d, lv_b, lv_b_d, ll_b, ll_b_d)
        scale = exp(-ll)
        expected = exp(lv - distance*scale)
        expected_d = expected*(lv_d - scale*(distance_d - distance*ll_d))
        expected_distance_b = -y_b*scale*expected
        expected_lv_b = y_b*expected
        expected_ll_b = y_b*distance*scale*expected
        if (maxval(abs([y - expected, y_d - expected_d, &
            distance_b - expected_distance_b, lv_b - expected_lv_b, &
            ll_b - expected_ll_b])) > 2.0e-14_dp) then
            failures = failures + 1
            write (error_unit, '(a)') 'FAIL [analytic] generated Matern12 leaf'
        end if

        h = 1.0e-6_dp
        call fortml_matern12_hvp(distance + h*distance_d, 0.0_dp, &
            lv + h*lv_d, 0.0_dp, ll + h*ll_d, 0.0_dp, ignored_y, &
            ignored_y_d, y_b, plus_distance_b, ignored_distance_b_d, plus_lv_b, &
            ignored_lv_b_d, plus_ll_b, ignored_ll_b_d)
        call fortml_matern12_hvp(distance - h*distance_d, 0.0_dp, &
            lv - h*lv_d, 0.0_dp, ll - h*ll_d, 0.0_dp, ignored_y, &
            ignored_y_d, y_b, minus_distance_b, ignored_distance_b_d, minus_lv_b, &
            ignored_lv_b_d, minus_ll_b, ignored_ll_b_d)
        if (maxval(abs([distance_b_d - (plus_distance_b - minus_distance_b)/(2.0_dp*h), &
            lv_b_d - (plus_lv_b - minus_lv_b)/(2.0_dp*h), &
            ll_b_d - (plus_ll_b - minus_ll_b)/(2.0_dp*h)])) > 3.0e-9_dp) then
            failures = failures + 1
            write (error_unit, '(a)') 'FAIL [fd] generated Matern12 reverse tangent'
        end if
    end subroutine check_point

end program test_fortsym_matern12
