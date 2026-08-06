program test_banded_precision
    !! Oracles for the banded Markov-precision path.
    !!
    !! The strongest available oracle is the exact Markov identity: the
    !! tridiagonal Ornstein-Uhlenbeck precision is the exact inverse of the
    !! dense exponential covariance on a uniform grid. So the band times that
    !! dense covariance must be the identity, a solve must reproduce the dense
    !! covariance product, and the log determinant must match a dense Cholesky.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_banded_precision, only: banded_precision_operator_t, &
        make_ornstein_uhlenbeck_precision
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 12
    real(dp), parameter :: spacing = 0.35_dp
    real(dp), parameter :: variance = 1.7_dp
    real(dp), parameter :: lengthscale = 0.8_dp
    real(dp) :: covariance(n, n)
    integer :: failures

    call build_covariance(covariance)
    failures = 0
    call test_precision_inverts_the_covariance(failures)
    call test_products_and_solves(failures)
    call test_log_determinant(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " banded precision test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_covariance(matrix)
        real(dp), intent(out) :: matrix(:, :)
        integer :: i, j

        do j = 1, n
            do i = 1, n
                matrix(i, j) = variance*exp(-abs(real(i - j, dp)*spacing)/ &
                    lengthscale)
            end do
        end do
    end subroutine build_covariance

    subroutine build_operator(operator, status)
        type(banded_precision_operator_t), intent(out) :: operator
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: band(:, :)

        call make_ornstein_uhlenbeck_precision(n, spacing, variance, &
            lengthscale, band, status)
        if (.not. status_ok(status)) return
        call operator%initialize(band, status)
    end subroutine build_operator

    subroutine test_precision_inverts_the_covariance(failures)
        integer, intent(inout) :: failures
        type(banded_precision_operator_t) :: operator
        type(fortnum_status_t) :: status
        real(dp) :: product(n, n), identity(n, n)
        integer :: column, i

        call build_operator(operator, status)
        do column = 1, n
            call operator%matvec(covariance(:, column), product(:, column))
        end do
        identity = 0.0_dp
        do i = 1, n
            identity(i, i) = 1.0_dp
        end do
        if (.not. status_ok(status) .or. &
            maxval(abs(product - identity)) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [markov] the band is not the covariance inverse ", &
                maxval(abs(product - identity))
            failures = failures + 1
        end if
    end subroutine test_precision_inverts_the_covariance

    subroutine test_products_and_solves(failures)
        integer, intent(inout) :: failures
        type(banded_precision_operator_t) :: operator
        type(fortnum_status_t) :: status
        real(dp) :: vector(n), solved(n), expected(n)
        real(dp) :: block(n, 2), block_out(n, 2), dense(n)
        integer :: i

        call build_operator(operator, status)
        do i = 1, n
            vector(i) = 0.4_dp*sin(real(i, dp)) - 0.15_dp
        end do
        ! Solving with the precision applies the covariance.
        call operator%solve(vector, solved, status)
        expected = matmul(covariance, vector)
        if (.not. status_ok(status) .or. &
            maxval(abs(solved - expected)) > 1.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [solve] banded solve is not the covariance product ", &
                maxval(abs(solved - expected))
            failures = failures + 1
        end if

        ! The matmat path must agree with the matvec path column by column.
        block(:, 1) = vector
        block(:, 2) = [(cos(0.3_dp*real(i, dp)), i=1, n)]
        call operator%matmat(block, block_out)
        call operator%matvec(block(:, 2), dense)
        if (maxval(abs(block_out(:, 2) - dense)) > 0.0_dp) then
            write (error_unit, '(a)') &
                "FAIL [product] banded matmat disagrees with matvec"
            failures = failures + 1
        end if
        if (maxval(abs(operator%diagonal() - [(operator_band(i), i=1, n)])) > &
            0.0_dp) then
            write (error_unit, '(a)') "FAIL [product] banded diagonal is wrong"
            failures = failures + 1
        end if
    end subroutine test_products_and_solves

    real(dp) function operator_band(i) result(value)
        !! The diagonal the constructor must have produced, written out here.
        integer, intent(in) :: i
        real(dp) :: rho, scale

        rho = exp(-spacing/lengthscale)
        scale = 1.0_dp/(variance*(1.0_dp - rho*rho))
        if (i == 1 .or. i == n) then
            value = scale
        else
            value = scale*(1.0_dp + rho*rho)
        end if
    end function operator_band

    subroutine test_log_determinant(failures)
        integer, intent(inout) :: failures
        type(banded_precision_operator_t) :: operator
        type(fortnum_status_t) :: status
        real(dp) :: value, covariance_value, exact
        real(dp) :: factor(n, n), total
        integer :: i, j, k

        call build_operator(operator, status)
        call operator%log_determinant(value, status)

        ! Dense Cholesky of the covariance gives log det of the covariance;
        ! the precision determinant is its negation.
        factor = 0.0_dp
        do i = 1, n
            do j = 1, i
                total = covariance(i, j)
                do k = 1, j - 1
                    total = total - factor(i, k)*factor(j, k)
                end do
                if (i == j) then
                    factor(i, i) = sqrt(total)
                else
                    factor(i, j) = total/factor(j, j)
                end if
            end do
        end do
        exact = 0.0_dp
        do i = 1, n
            exact = exact + 2.0_dp*log(factor(i, i))
        end do

        call operator%covariance_log_determinant(covariance_value, status)
        if (.not. status_ok(status) .or. abs(value + exact) > 1.0e-10_dp .or. &
            abs(covariance_value - exact) > 1.0e-10_dp) then
            write (error_unit, '(a,3es12.4)') &
                "FAIL [logdet] banded log determinants ", value, &
                covariance_value, exact
            failures = failures + 1
        end if
    end subroutine test_log_determinant

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(banded_precision_operator_t) :: operator
        type(fortnum_status_t) :: status
        real(dp) :: band(0:1, 3), short(2), solution(3)
        real(dp), allocatable :: bad(:, :)

        call make_ornstein_uhlenbeck_precision(1, spacing, variance, &
            lengthscale, bad, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a one-point grid accepted"
            failures = failures + 1
        end if
        call make_ornstein_uhlenbeck_precision(4, -1.0_dp, variance, &
            lengthscale, bad, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] negative spacing accepted"
            failures = failures + 1
        end if

        ! An indefinite band must be refused rather than factorized.
        band = 0.0_dp
        band(0, :) = [1.0_dp, -1.0_dp, 1.0_dp]
        call operator%initialize(band, status)
        call operator%factorize(status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] an indefinite precision was factorized"
            failures = failures + 1
        end if

        band(0, :) = [2.0_dp, 2.0_dp, 2.0_dp]
        call operator%initialize(band, status)
        call operator%solve(short, solution, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a wrong solve shape accepted"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_banded_precision
