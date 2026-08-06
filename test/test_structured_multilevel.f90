program test_structured_multilevel
    !! Oracles for the multilevel tensor-grid embedding and for the separable
    !! derivative products of the structured GP operator.
    !!
    !! The embedding is checked by the two properties that make it usable: a
    !! function that is linear on the coarse grid must prolong to its exact
    !! values on the fine grid, and restriction must be the exact transpose of
    !! prolongation. The derivative products are checked against the dense
    !! Kronecker product assembled here, which shares no code with the
    !! matrix-free contraction.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_multilevel_grid, only: multilevel_grid_t
    use fortml_structured_operator, only: structured_gp_operator_t, tensor_factor_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_level_shapes(failures)
    call test_prolongation_reproduces_linear_functions(failures)
    call test_restriction_is_the_transpose(failures)
    call test_derivative_products(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " multilevel test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_level_shapes(failures)
        integer, intent(inout) :: failures
        type(multilevel_grid_t) :: grid
        type(fortnum_status_t) :: status
        integer, allocatable :: dimensions(:)

        call grid%initialize([9, 5], 4, status)
        if (.not. status_ok(status) .or. grid%level_count() /= 3) then
            write (error_unit, '(a,i0)') &
                "FAIL [levels] hierarchy depth ", grid%level_count()
            failures = failures + 1
            return
        end if
        dimensions = grid%level_dimensions(1)
        if (any(dimensions /= [9, 5])) then
            write (error_unit, '(a)') "FAIL [levels] finest level shape"
            failures = failures + 1
        end if
        dimensions = grid%level_dimensions(2)
        if (any(dimensions /= [5, 3])) then
            write (error_unit, '(a)') "FAIL [levels] second level shape"
            failures = failures + 1
        end if
        dimensions = grid%level_dimensions(3)
        if (any(dimensions /= [3, 2])) then
            write (error_unit, '(a)') "FAIL [levels] third level shape"
            failures = failures + 1
        end if
        if (grid%level_size(1) /= 45 .or. grid%level_size(3) /= 6) then
            write (error_unit, '(a)') "FAIL [levels] level sizes"
            failures = failures + 1
        end if
    end subroutine test_level_shapes

    subroutine test_prolongation_reproduces_linear_functions(failures)
        !! Linear interpolation is exact on linear data. Sampling
        !! `f(x, y) = 2x - 3y + 1` on the coarse grid and prolonging must give
        !! the same function sampled on the fine grid.
        integer, intent(inout) :: failures
        type(multilevel_grid_t) :: grid
        type(fortnum_status_t) :: status
        real(dp), allocatable :: coarse(:), fine(:), expected(:)
        integer :: i, j, index
        real(dp) :: x, y

        call grid%initialize([9, 5], 2, status)
        allocate(coarse(grid%level_size(2)), fine(grid%level_size(1)))
        allocate(expected(grid%level_size(1)))
        index = 0
        do j = 1, 3
            do i = 1, 5
                index = index + 1
                x = real(2*(i - 1), dp)
                y = real(2*(j - 1), dp)
                coarse(index) = 2.0_dp*x - 3.0_dp*y + 1.0_dp
            end do
        end do
        index = 0
        do j = 1, 5
            do i = 1, 9
                index = index + 1
                x = real(i - 1, dp)
                y = real(j - 1, dp)
                expected(index) = 2.0_dp*x - 3.0_dp*y + 1.0_dp
            end do
        end do

        call grid%prolong(1, coarse, fine, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(fine - expected)) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [prolong] linear function is not reproduced ", &
                maxval(abs(fine - expected))
            failures = failures + 1
        end if
    end subroutine test_prolongation_reproduces_linear_functions

    subroutine test_restriction_is_the_transpose(failures)
        !! `<P c, f> = <c, R f>` for every pair, which is what keeps a
        !! two-level correction symmetric.
        integer, intent(inout) :: failures
        type(multilevel_grid_t) :: grid
        type(fortnum_status_t) :: status
        real(dp), allocatable :: coarse(:), fine(:), prolonged(:), restricted(:)
        real(dp) :: left, right
        integer :: i

        call grid%initialize([8, 5, 3], 2, status)
        allocate(coarse(grid%level_size(2)), restricted(grid%level_size(2)))
        allocate(fine(grid%level_size(1)), prolonged(grid%level_size(1)))
        do i = 1, size(coarse)
            coarse(i) = 0.37_dp*sin(real(i, dp)) - 0.11_dp
        end do
        do i = 1, size(fine)
            fine(i) = 0.21_dp*cos(real(2*i, dp)) + 0.05_dp
        end do

        call grid%prolong(1, coarse, prolonged, status)
        call grid%restrict(1, fine, restricted, status)
        left = sum(prolonged*fine)
        right = sum(coarse*restricted)
        if (.not. status_ok(status) .or. abs(left - right) > 1.0e-12_dp) then
            write (error_unit, '(a,2es14.6)') &
                "FAIL [transpose] adjoint identity ", left, right
            failures = failures + 1
        end if
    end subroutine test_restriction_is_the_transpose

    subroutine test_derivative_products(failures)
        integer, intent(inout) :: failures
        type(structured_gp_operator_t) :: operator
        type(fortnum_status_t) :: status
        type(tensor_factor_t) :: factors(2), derivatives(2)
        real(dp) :: dense(12, 12), input(12), output(12), expected(12)
        real(dp) :: block_input(12, 2), block_output(12, 2)
        integer :: i, j

        allocate(factors(1)%values(3, 3), factors(2)%values(4, 4))
        allocate(derivatives(1)%values(3, 3), derivatives(2)%values(4, 4))
        do j = 1, 3
            do i = 1, 3
                factors(1)%values(i, j) = 1.0_dp/(1.0_dp + abs(i - j))
                derivatives(1)%values(i, j) = 0.3_dp*real(i - j, dp)
            end do
        end do
        do j = 1, 4
            do i = 1, 4
                factors(2)%values(i, j) = 0.5_dp**abs(i - j)
                derivatives(2)%values(i, j) = -0.2_dp*real(i - j, dp)
            end do
        end do

        call operator%initialize(factors, status)
        call operator%set_derivative_factors(derivatives, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [derivative] factors were refused"
            failures = failures + 1
            return
        end if
        do i = 1, 12
            input(i) = 0.4_dp*sin(real(i, dp)) - 0.2_dp
        end do

        ! Derivative along the first (fastest varying) dimension.
        call kronecker(derivatives(1)%values, factors(2)%values, dense)
        expected = matmul(dense, input)
        call operator%derivative_matvec(1, input, output, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(output - expected)) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [derivative] dimension 1 product ", &
                maxval(abs(output - expected))
            failures = failures + 1
        end if

        call kronecker(factors(1)%values, derivatives(2)%values, dense)
        expected = matmul(dense, input)
        call operator%derivative_matvec(2, input, output, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(output - expected)) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [derivative] dimension 2 product ", &
                maxval(abs(output - expected))
            failures = failures + 1
        end if

        block_input(:, 1) = input
        block_input(:, 2) = 1.0_dp
        call operator%derivative_matmat(2, block_input, block_output, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(block_output(:, 1) - expected)) > 1.0e-12_dp .or. &
            maxval(abs(block_output(:, 2) - matmul(dense, block_input(:, 2)))) &
            > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [derivative] batched product"
            failures = failures + 1
        end if
    end subroutine test_derivative_products

    subroutine kronecker(second, first, dense)
        !! `first (x) second` with factor 1 innermost, matching the operator's
        !! index order: entry ((j-1)*n1 + i) uses first(j, ...) and
        !! second(i, ...).
        real(dp), intent(in) :: second(:, :), first(:, :)
        real(dp), intent(out) :: dense(:, :)
        integer :: i, j, k, l, n1, row, column

        n1 = size(second, 1)
        do l = 1, size(first, 2)
            do k = 1, size(second, 2)
                do j = 1, size(first, 1)
                    do i = 1, n1
                        row = (j - 1)*n1 + i
                        column = (l - 1)*n1 + k
                        dense(row, column) = first(j, l)*second(i, k)
                    end do
                end do
            end do
        end do
    end subroutine kronecker

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(multilevel_grid_t) :: grid
        type(structured_gp_operator_t) :: operator
        type(fortnum_status_t) :: status
        type(tensor_factor_t) :: factors(2), wrong(1)
        real(dp) :: coarse(6), fine(45), value(12)

        call grid%initialize([1, 4], 2, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a one-point dimension accepted"
            failures = failures + 1
        end if
        call grid%initialize([9, 5], 2, status)
        call grid%prolong(2, coarse, fine, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a transfer below the coarsest"
            failures = failures + 1
        end if
        call grid%prolong(1, fine, fine, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a mismatched transfer shape"
            failures = failures + 1
        end if

        allocate(factors(1)%values(3, 3), factors(2)%values(4, 4))
        factors(1)%values = 0.0_dp
        factors(2)%values = 0.0_dp
        call operator%initialize(factors, status)
        allocate(wrong(1)%values(3, 3))
        wrong(1)%values = 0.0_dp
        call operator%set_derivative_factors(wrong, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] a short derivative factor list accepted"
            failures = failures + 1
        end if
        call operator%derivative_matvec(1, value, value, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] a derivative product without factors accepted"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_structured_multilevel
