program test_ski_gp
    !! Oracles for structured kernel interpolation and subset of data.
    !!
    !! The SKI product is checked against the dense `W K_uu W^T + noise I`
    !! assembled here from the interpolation weights and the grid kernel, which
    !! shares no code with the Toeplitz path the operator uses. The
    !! approximation is then checked to converge to the exact kernel matrix as
    !! the grid is refined, which is the property that makes SKI usable at all,
    !! and the interpolation weights are checked to be the local linear ones
    !! (non-negative, summing to one, on adjacent nodes) that the review's
    !! equation (19) describes.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_ski_gp, only: ski_operator_t, subset_of_data_indices
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 15
    real(dp), parameter :: variance = 1.4_dp
    real(dp), parameter :: lengthscale = 0.9_dp
    real(dp), parameter :: noise = 0.1_dp
    real(dp) :: x(n, 1)
    integer :: failures

    call build_inputs(x)
    failures = 0
    call test_product_against_dense_interpolation(failures)
    call test_weights_are_local_linear(failures)
    call test_refinement_converges_to_the_exact_kernel(failures)
    call test_subset_of_data(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " SKI test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_inputs(inputs)
        real(dp), intent(out) :: inputs(:, :)
        integer :: i

        do i = 1, n
            inputs(i, 1) = -2.0_dp + 0.29_dp*real(i, dp) &
                + 0.03_dp*sin(real(3*i, dp))
        end do
    end subroutine build_inputs

    real(dp) function rbf(a, b) result(value)
        real(dp), intent(in) :: a, b

        value = variance*exp(-0.5_dp*(a - b)**2/(lengthscale*lengthscale))
    end function rbf

    subroutine dense_ski_matrix(operator, n_grid, matrix)
        !! `W K_uu W^T + noise I`, assembled from the reported weights.
        type(ski_operator_t), intent(in) :: operator
        integer, intent(in) :: n_grid
        real(dp), intent(out) :: matrix(:, :)
        real(dp) :: weights(n, n_grid), grid(n_grid), values(2)
        real(dp) :: lower, upper, spacing, grid_kernel(n_grid, n_grid)
        integer :: i, j, indices(2)

        lower = minval(x(:, 1))
        upper = maxval(x(:, 1))
        spacing = (upper - lower)/real(n_grid - 1, dp)
        do i = 1, n_grid
            grid(i) = lower + real(i - 1, dp)*spacing
        end do
        do j = 1, n_grid
            do i = 1, n_grid
                grid_kernel(i, j) = rbf(grid(i), grid(j))
            end do
        end do
        weights = 0.0_dp
        do i = 1, n
            call operator%interpolation_weights(i, indices, values)
            weights(i, indices(1)) = weights(i, indices(1)) + values(1)
            weights(i, indices(2)) = weights(i, indices(2)) + values(2)
        end do
        matrix = matmul(weights, matmul(grid_kernel, transpose(weights)))
        do i = 1, n
            matrix(i, i) = matrix(i, i) + noise
        end do
    end subroutine dense_ski_matrix

    subroutine test_product_against_dense_interpolation(failures)
        integer, intent(inout) :: failures
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        integer, parameter :: n_grid = 12
        real(dp) :: matrix(n, n), input(n), output(n), expected(n)
        real(dp) :: block_input(n, 2), block_output(n, 2)
        integer :: i

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call operator%initialize(x, kernel, n_grid, noise, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [ski] initialization refused"
            failures = failures + 1
            return
        end if
        call dense_ski_matrix(operator, n_grid, matrix)
        do i = 1, n
            input(i) = 0.4_dp*sin(real(i, dp)) - 0.1_dp
        end do
        expected = matmul(matrix, input)
        call operator%matvec(input, output)
        if (maxval(abs(output - expected)) > 1.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [ski] product differs from the dense interpolation ", &
                maxval(abs(output - expected))
            failures = failures + 1
        end if

        block_input(:, 1) = input
        block_input(:, 2) = 1.0_dp
        call operator%matmat(block_input, block_output)
        if (maxval(abs(block_output(:, 1) - expected)) > 1.0e-11_dp .or. &
            maxval(abs(block_output(:, 2) - matmul(matrix, block_input(:, 2)))) &
            > 1.0e-11_dp) then
            write (error_unit, '(a)') "FAIL [ski] batched product"
            failures = failures + 1
        end if
        if (operator%sample_count() /= n) then
            write (error_unit, '(a)') "FAIL [ski] sample count"
            failures = failures + 1
        end if
    end subroutine test_product_against_dense_interpolation

    subroutine test_weights_are_local_linear(failures)
        integer, intent(inout) :: failures
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        integer :: i, indices(2)
        real(dp) :: values(2)

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call operator%initialize(x, kernel, 10, noise, status)
        do i = 1, n
            call operator%interpolation_weights(i, indices, values)
            if (indices(2) /= indices(1) + 1 .or. indices(1) < 1 .or. &
                indices(2) > 10) then
                write (error_unit, '(a,i0)') &
                    "FAIL [ski] weights are not on adjacent nodes at row ", i
                failures = failures + 1
                return
            end if
            if (any(values < 0.0_dp) .or. abs(sum(values) - 1.0_dp) > 1.0e-14_dp) then
                write (error_unit, '(a,i0)') &
                    "FAIL [ski] weights are not a partition of unity at row ", i
                failures = failures + 1
                return
            end if
        end do
    end subroutine test_weights_are_local_linear

    subroutine test_refinement_converges_to_the_exact_kernel(failures)
        !! Refining the grid must reduce the gap to the exact kernel matrix.
        integer, intent(inout) :: failures
        real(dp) :: coarse_error, fine_error, finer_error

        coarse_error = approximation_error(8)
        fine_error = approximation_error(24)
        finer_error = approximation_error(72)
        if (.not. (fine_error < coarse_error .and. finer_error < fine_error)) then
            write (error_unit, '(a,3es12.4)') &
                "FAIL [ski] refinement does not converge ", coarse_error, &
                fine_error, finer_error
            failures = failures + 1
        end if
    end subroutine test_refinement_converges_to_the_exact_kernel

    real(dp) function approximation_error(n_grid) result(error)
        integer, intent(in) :: n_grid
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: matrix(n, n), exact(n, n)
        integer :: i, j

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call operator%initialize(x, kernel, n_grid, noise, status)
        call dense_ski_matrix(operator, n_grid, matrix)
        do j = 1, n
            do i = 1, n
                exact(i, j) = rbf(x(i, 1), x(j, 1))
            end do
            exact(j, j) = exact(j, j) + noise
        end do
        error = maxval(abs(matrix - exact))
    end function approximation_error

    subroutine test_subset_of_data(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        integer, allocatable :: indices(:)
        integer :: i

        call subset_of_data_indices(n, 5, indices, status)
        if (.not. status_ok(status) .or. size(indices) /= 5) then
            write (error_unit, '(a)') "FAIL [sod] subset size"
            failures = failures + 1
            return
        end if
        if (indices(1) /= 1 .or. indices(5) /= n) then
            write (error_unit, '(a)') "FAIL [sod] the subset does not span"
            failures = failures + 1
        end if
        do i = 2, 5
            if (indices(i) <= indices(i - 1)) then
                write (error_unit, '(a)') "FAIL [sod] indices are not increasing"
                failures = failures + 1
                return
            end if
        end do
        call subset_of_data_indices(n, n + 1, indices, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [sod] an oversized subset accepted"
            failures = failures + 1
        end if
    end subroutine test_subset_of_data

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: flat(4, 1), wide(4, 2)

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call operator%initialize(x, kernel, 1, noise, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a one-node grid accepted"
            failures = failures + 1
        end if
        flat = 0.5_dp
        call operator%initialize(flat, kernel, 8, noise, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a degenerate input range"
            failures = failures + 1
        end if
        wide = 0.0_dp
        call operator%initialize(wide, kernel, 8, noise, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a two-dimensional input"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_ski_gp
