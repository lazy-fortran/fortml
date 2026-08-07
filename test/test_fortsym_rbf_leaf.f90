program test_fortsym_rbf_leaf
    ! Independent dense oracle for the FortSym-generated natural RBF leaf.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    implicit none
    integer, parameter :: dp = real64
    integer :: failures

    interface
        subroutine fortml_generated_rbf_leaf_derivatives( &
                variance, distance, lengthscale, value, dvariance, ddistance, &
                dlengthscale)
            use, intrinsic :: iso_fortran_env, only: real64
            real(real64), intent(in) :: variance, distance, lengthscale
            real(real64), intent(out) :: value, dvariance, ddistance, &
                dlengthscale
        end subroutine fortml_generated_rbf_leaf_derivatives
    end interface

    failures = 0
    call check_point(1.7_dp, 0.83_dp, 1.2_dp, failures)
    call check_point(0.4_dp, 2.1_dp, 0.65_dp, failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            ' FortSym RBF leaf check(s) failed'
        error stop 1
    end if
    write (*, '(a)') 'PASS'

contains

    subroutine check_point(variance, distance, lengthscale, failures)
        real(dp), intent(in) :: variance, distance, lengthscale
        integer, intent(inout) :: failures
        real(dp) :: value, dvariance, ddistance, dlengthscale
        real(dp) :: expected, expected_dvariance, expected_ddistance
        real(dp) :: expected_dlengthscale, h, plus, minus, ignored(3)

        call fortml_generated_rbf_leaf_derivatives(variance, distance, &
            lengthscale, value, dvariance, ddistance, dlengthscale)
        expected = variance*exp(-0.5_dp*distance/lengthscale**2)
        expected_dvariance = exp(-0.5_dp*distance/lengthscale**2)
        expected_ddistance = -0.5_dp*expected/lengthscale**2
        expected_dlengthscale = expected*distance/lengthscale**3
        if (maxval(abs([value - expected, dvariance - expected_dvariance, &
            ddistance - expected_ddistance, dlengthscale - &
            expected_dlengthscale])) > 2.0e-14_dp) then
            failures = failures + 1
            write (error_unit, '(a)') 'FAIL [oracle] generated RBF leaf'
        end if

        h = 1.0e-6_dp
        call fortml_generated_rbf_leaf_derivatives(variance + h, distance, &
            lengthscale, plus, ignored(1), ignored(2), ignored(3))
        call fortml_generated_rbf_leaf_derivatives(variance - h, distance, &
            lengthscale, minus, ignored(1), ignored(2), ignored(3))
        if (abs(dvariance - (plus - minus)/(2.0_dp*h)) > 2.0e-9_dp) then
            failures = failures + 1
            write (error_unit, '(a)') 'FAIL [fd] variance derivative'
        end if
        call fortml_generated_rbf_leaf_derivatives(variance, distance + h, &
            lengthscale, plus, ignored(1), ignored(2), ignored(3))
        call fortml_generated_rbf_leaf_derivatives(variance, distance - h, &
            lengthscale, minus, ignored(1), ignored(2), ignored(3))
        if (abs(ddistance - (plus - minus)/(2.0_dp*h)) > 2.0e-9_dp) then
            failures = failures + 1
            write (error_unit, '(a)') 'FAIL [fd] distance derivative'
        end if
        call fortml_generated_rbf_leaf_derivatives(variance, distance, &
            lengthscale + h, plus, ignored(1), ignored(2), ignored(3))
        call fortml_generated_rbf_leaf_derivatives(variance, distance, &
            lengthscale - h, minus, ignored(1), ignored(2), ignored(3))
        if (abs(dlengthscale - (plus - minus)/(2.0_dp*h)) > 2.0e-9_dp) then
            failures = failures + 1
            write (error_unit, '(a)') 'FAIL [fd] lengthscale derivative'
        end if
    end subroutine check_point

end program test_fortsym_rbf_leaf
