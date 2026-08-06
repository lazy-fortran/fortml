program test_ski_gp
    !! Oracles for structured kernel interpolation and subset of data.
    !!
    !! The SKI product is checked against the dense `W K_uu W^T + noise I`
    !! assembled here from the interpolation weights and the grid kernel, which
    !! shares no code with the Toeplitz path the operator uses. The
    !! approximation is then checked to converge to the exact kernel matrix as
    !! the grid is refined, which is the property that makes SKI usable at all.
    !! The two-dimensional oracle independently constructs its bilinear `W`,
    !! full RBF `K_uu`, dense `W K_uu W^T`, and a separate query interpolation
    !! table for `W_query K_uu W_train^T`; it therefore checks the Kronecker
    !! implementation rather than repeating it.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use fortml_kernels, only: kernel_t, make_matern32_kernel, make_rbf_kernel
    use fortml_ski_gp, only: ski_operator_t, subset_of_data_indices
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 15
    integer, parameter :: n2 = 12
    real(dp), parameter :: variance = 1.4_dp
    real(dp), parameter :: lengthscale = 0.9_dp
    real(dp), parameter :: noise = 0.1_dp
    real(dp) :: x(n, 1)
    real(dp) :: x2(n2, 2)
    integer :: failures

    call build_inputs(x)
    call build_inputs_2d(x2)
    failures = 0
    call test_product_against_dense_interpolation(failures)
    call test_two_dimensional_products(failures)
    call test_two_dimensional_cross_product(failures)
    call test_weights_are_local_linear(failures)
    call test_refinement_converges_to_the_exact_kernel(failures)
    call test_two_dimensional_refinement(failures)
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

    subroutine build_inputs_2d(inputs)
        real(dp), intent(out) :: inputs(:, :)
        integer :: i

        do i = 1, size(inputs, 1)
            inputs(i, 1) = -1.8_dp + 3.6_dp*real(i - 1, dp)/ &
                real(size(inputs, 1) - 1, dp)
            inputs(i, 2) = -1.1_dp + 0.17_dp*real(i - 1, dp) + &
                0.35_dp*sin(0.8_dp*real(i - 1, dp))
        end do
    end subroutine build_inputs_2d

    real(dp) function rbf(a, b) result(value)
        real(dp), intent(in) :: a, b

        value = variance*exp(-0.5_dp*(a - b)**2/(lengthscale*lengthscale))
    end function rbf

    real(dp) function rbf_vector(a, b) result(value)
        real(dp), intent(in) :: a(:), b(:)

        value = variance*exp(-0.5_dp*sum((a - b)**2)/ &
            (lengthscale*lengthscale))
    end function rbf_vector

    subroutine dense_linear_weights_1d(inputs, n_grid, weights)
        !! Independent reference for the production interpolation table.
        real(dp), intent(in) :: inputs(:, :)
        integer, intent(in) :: n_grid
        real(dp), intent(out) :: weights(:, :)
        real(dp) :: lower, upper, spacing, position, fraction
        integer :: cell, i

        lower = minval(inputs(:, 1))
        upper = maxval(inputs(:, 1))
        spacing = (upper - lower)/real(n_grid - 1, dp)
        weights = 0.0_dp
        do i = 1, size(inputs, 1)
            position = (inputs(i, 1) - lower)/spacing
            cell = min(max(int(position) + 1, 1), n_grid - 1)
            fraction = position - real(cell - 1, dp)
            fraction = min(max(fraction, 0.0_dp), 1.0_dp)
            weights(i, cell) = 1.0_dp - fraction
            weights(i, cell + 1) = fraction
        end do
    end subroutine dense_linear_weights_1d

    subroutine dense_ski_matrix(operator, n_grid, matrix)
        !! `W K_uu W^T + noise I`, assembled from the reported weights.
        type(ski_operator_t), intent(in) :: operator
        integer, intent(in) :: n_grid
        real(dp), intent(out) :: matrix(:, :)
        real(dp) :: weights(n, n_grid), grid(n_grid)
        real(dp) :: lower, upper, spacing, grid_kernel(n_grid, n_grid)
        integer :: i, j

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
        call dense_linear_weights_1d(x, n_grid, weights)
        matrix = matmul(weights, matmul(grid_kernel, transpose(weights)))
        do i = 1, n
            matrix(i, i) = matrix(i, i) + noise
        end do
    end subroutine dense_ski_matrix

    subroutine dense_ski_matrix_2d(inputs, axis_extent, matrix)
        !! Bilinear `W K_uu W^T + noise I`, assembled independently of the
        !! structured operator and its flattened interpolation table.
        real(dp), intent(in) :: inputs(:, :)
        integer, intent(in) :: axis_extent
        real(dp), intent(out) :: matrix(:, :)

        real(dp), allocatable :: grid(:, :), grid_kernel(:, :), weights(:, :)
        real(dp) :: lower(2), spacing(2), upper(2)
        integer :: grid_points, i, i1, i2, j

        grid_points = axis_extent*axis_extent
        allocate(grid(grid_points, 2), grid_kernel(grid_points, grid_points))
        allocate(weights(size(inputs, 1), grid_points))
        lower = minval(inputs, dim=1)
        upper = maxval(inputs, dim=1)
        spacing = (upper - lower)/real(axis_extent - 1, dp)

        do i2 = 1, axis_extent
            do i1 = 1, axis_extent
                i = flatten_2d(i1, i2, axis_extent)
                grid(i, 1) = lower(1) + real(i1 - 1, dp)*spacing(1)
                grid(i, 2) = lower(2) + real(i2 - 1, dp)*spacing(2)
            end do
        end do
        do j = 1, grid_points
            do i = 1, grid_points
                grid_kernel(i, j) = rbf_vector(grid(i, :), grid(j, :))
            end do
        end do

        call dense_bilinear_weights_2d(inputs, lower, upper, spacing, &
            axis_extent, weights)

        matrix = matmul(weights, matmul(grid_kernel, transpose(weights)))
        do i = 1, size(inputs, 1)
            matrix(i, i) = matrix(i, i) + noise
        end do
    end subroutine dense_ski_matrix_2d

    subroutine dense_ski_cross_2d(train_inputs, query, axis_extent, cross)
        !! Independently assemble `W_query K_uu W_train^T`. The grid is fixed
        !! by the training bounds; query coordinates beyond it clamp to a face.
        real(dp), intent(in) :: train_inputs(:, :), query(:, :)
        integer, intent(in) :: axis_extent
        real(dp), intent(out) :: cross(:, :)
        real(dp), allocatable :: grid(:, :), grid_kernel(:, :)
        real(dp), allocatable :: query_weights(:, :), train_weights(:, :)
        real(dp) :: lower(2), spacing(2), upper(2)
        integer :: grid_points, i, i1, i2, j

        grid_points = axis_extent*axis_extent
        allocate(grid(grid_points, 2), grid_kernel(grid_points, grid_points))
        allocate(query_weights(size(query, 1), grid_points))
        allocate(train_weights(size(train_inputs, 1), grid_points))
        lower = minval(train_inputs, dim=1)
        upper = maxval(train_inputs, dim=1)
        spacing = (upper - lower)/real(axis_extent - 1, dp)
        do i2 = 1, axis_extent
            do i1 = 1, axis_extent
                i = flatten_2d(i1, i2, axis_extent)
                grid(i, 1) = lower(1) + real(i1 - 1, dp)*spacing(1)
                grid(i, 2) = lower(2) + real(i2 - 1, dp)*spacing(2)
            end do
        end do
        do j = 1, grid_points
            do i = 1, grid_points
                grid_kernel(i, j) = rbf_vector(grid(i, :), grid(j, :))
            end do
        end do
        call dense_bilinear_weights_2d(train_inputs, lower, upper, spacing, &
            axis_extent, train_weights)
        call dense_bilinear_weights_2d(query, lower, upper, spacing, &
            axis_extent, query_weights)
        cross = matmul(query_weights, &
            matmul(grid_kernel, transpose(train_weights)))
    end subroutine dense_ski_cross_2d

    subroutine dense_bilinear_weights_2d(points, lower, upper, spacing, &
            axis_extent, weights)
        real(dp), intent(in) :: points(:, :), lower(2), upper(2), spacing(2)
        integer, intent(in) :: axis_extent
        real(dp), intent(out) :: weights(:, :)
        real(dp) :: fraction(2), position
        integer :: cell(2), i, j, nodes(4)

        weights = 0.0_dp
        do i = 1, size(points, 1)
            do j = 1, 2
                if (points(i, j) <= lower(j)) then
                    cell(j) = 1
                    fraction(j) = 0.0_dp
                else if (points(i, j) >= upper(j)) then
                    cell(j) = axis_extent - 1
                    fraction(j) = 1.0_dp
                else
                    position = (points(i, j) - lower(j))/spacing(j)
                    cell(j) = min(max(int(position) + 1, 1), axis_extent - 1)
                    fraction(j) = position - real(cell(j) - 1, dp)
                end if
            end do
            nodes = [ &
                flatten_2d(cell(1), cell(2), axis_extent), &
                flatten_2d(cell(1) + 1, cell(2), axis_extent), &
                flatten_2d(cell(1), cell(2) + 1, axis_extent), &
                flatten_2d(cell(1) + 1, cell(2) + 1, axis_extent)]
            weights(i, nodes(1)) = (1.0_dp - fraction(1))* &
                (1.0_dp - fraction(2))
            weights(i, nodes(2)) = fraction(1)*(1.0_dp - fraction(2))
            weights(i, nodes(3)) = (1.0_dp - fraction(1))*fraction(2)
            weights(i, nodes(4)) = fraction(1)*fraction(2)
        end do
    end subroutine dense_bilinear_weights_2d

    integer function flatten_2d(i1, i2, axis_extent) result(index)
        integer, intent(in) :: i1, i2, axis_extent

        index = i1 + axis_extent*(i2 - 1)
    end function flatten_2d

    subroutine test_product_against_dense_interpolation(failures)
        integer, intent(inout) :: failures
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        integer, parameter :: n_grid = 12
        real(dp) :: cross_output(n), matrix(n, n), input(n), output(n)
        real(dp) :: expected(n)
        real(dp) :: expected_diagonal(n)
        real(dp) :: block_input(n, 2), block_output(n, 2)
        real(dp), allocatable :: operator_diagonal(:)
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
            expected_diagonal(i) = matrix(i, i)
        end do
        operator_diagonal = operator%diagonal()
        if (maxval(abs(operator_diagonal - expected_diagonal)) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [ski] diagonal differs from the dense interpolation ", &
                maxval(abs(operator_diagonal - expected_diagonal))
            failures = failures + 1
        end if
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
        call operator%cross_matvec(x, input, cross_output, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(cross_output - (expected - noise*input))) > &
            1.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [ski] cross product differs from dense W K W^T ", &
                maxval(abs(cross_output - (expected - noise*input)))
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

    subroutine test_two_dimensional_products(failures)
        !! A budget of 18 means a 4-by-4 grid: 4**2 <= 18 < 5**2.
        integer, intent(inout) :: failures
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        integer, parameter :: axis_extent = 4, grid_budget = 18
        real(dp) :: block_input(n2, 3), block_output(n2, 3)
        real(dp) :: diagonal_expected(n2), expected(n2), input(n2)
        real(dp) :: matrix(n2, n2), output(n2)
        real(dp), allocatable :: diagonal_actual(:)
        integer :: column, i

        kernel = make_rbf_kernel(2, variance, lengthscale, status)
        call operator%initialize(x2, kernel, grid_budget, noise, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [ski-2d] initialization refused"
            failures = failures + 1
            return
        end if
        call dense_ski_matrix_2d(x2, axis_extent, matrix)
        do i = 1, n2
            input(i) = 0.2_dp*cos(0.3_dp*real(i, dp)) + 0.04_dp*real(i, dp)
            diagonal_expected(i) = matrix(i, i)
        end do

        expected = matmul(matrix, input)
        call operator%matvec(input, output)
        if (maxval(abs(output - expected)) > 2.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [ski-2d] matvec differs from dense W K W^T ", &
                maxval(abs(output - expected))
            failures = failures + 1
        end if

        do column = 1, 3
            block_input(:, column) = input + 0.13_dp*real(column - 1, dp)
        end do
        call operator%matmat(block_input, block_output)
        if (maxval(abs(block_output - matmul(matrix, block_input))) > 2.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [ski-2d] matmat differs from dense W K W^T ", &
                maxval(abs(block_output - matmul(matrix, block_input)))
            failures = failures + 1
        end if

        diagonal_actual = operator%diagonal()
        if (maxval(abs(diagonal_actual - diagonal_expected)) > 2.0e-12_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [ski-2d] diagonal differs from dense W K W^T ", &
                maxval(abs(diagonal_actual - diagonal_expected))
            failures = failures + 1
        end if
    end subroutine test_two_dimensional_products

    subroutine test_two_dimensional_cross_product(failures)
        integer, intent(inout) :: failures
        integer, parameter :: axis_extent = 4, grid_budget = 18, n_query = 5
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: cross(n_query, n2), expected(n_query), input(n2)
        real(dp) :: output(n_query), query(n_query, 2)
        integer :: i

        query(1, :) = [-2.3_dp, -1.5_dp]
        query(2, :) = [-0.8_dp, -0.4_dp]
        query(3, :) = [0.15_dp, 0.2_dp]
        query(4, :) = [1.35_dp, 0.65_dp]
        query(5, :) = [2.4_dp, 1.4_dp]
        do i = 1, n2
            input(i) = 0.17_dp*sin(0.6_dp*real(i, dp)) - 0.03_dp*real(i, dp)
        end do
        kernel = make_rbf_kernel(2, variance, lengthscale, status)
        call operator%initialize(x2, kernel, grid_budget, noise, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [ski-2d-cross] initialization refused"
            failures = failures + 1
            return
        end if
        call dense_ski_cross_2d(x2, query, axis_extent, cross)
        expected = matmul(cross, input)
        call operator%cross_matvec(query, input, output, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(output - expected)) > 2.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [ski-2d-cross] differs from dense Wq K Wt^T ", &
                maxval(abs(output - expected))
            failures = failures + 1
        end if
    end subroutine test_two_dimensional_cross_product

    subroutine test_weights_are_local_linear(failures)
        integer, intent(inout) :: failures
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        integer :: i, indices(2), expected_indices(2), cell
        real(dp) :: values(2), expected_values(2)
        real(dp) :: lower, upper, spacing, position, fraction

        kernel = make_rbf_kernel(1, variance, lengthscale, status)
        call operator%initialize(x, kernel, 10, noise, status)
        lower = minval(x(:, 1))
        upper = maxval(x(:, 1))
        spacing = (upper - lower)/9.0_dp
        do i = 1, n
            call operator%interpolation_weights(i, indices, values)
            position = (x(i, 1) - lower)/spacing
            cell = min(max(int(position) + 1, 1), 9)
            fraction = min(max(position - real(cell - 1, dp), 0.0_dp), 1.0_dp)
            expected_indices = [cell, cell + 1]
            expected_values = [1.0_dp - fraction, fraction]
            if (any(indices /= expected_indices) .or. &
                maxval(abs(values - expected_values)) > 1.0e-14_dp) then
                write (error_unit, '(a,i0)') &
                    "FAIL [ski] interpolation differs from independent reference at row ", i
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

    subroutine test_two_dimensional_refinement(failures)
        integer, intent(inout) :: failures
        real(dp) :: coarse_error, fine_error, finer_error

        coarse_error = approximation_error_2d(3)
        fine_error = approximation_error_2d(5)
        finer_error = approximation_error_2d(9)
        if (.not. (fine_error < coarse_error .and. finer_error < fine_error)) then
            write (error_unit, '(a,3es12.4)') &
                "FAIL [ski-2d] refinement does not converge ", coarse_error, &
                fine_error, finer_error
            failures = failures + 1
        end if
    end subroutine test_two_dimensional_refinement

    real(dp) function approximation_error_2d(axis_extent) result(error)
        integer, intent(in) :: axis_extent
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: approximate(n2, n2), exact(n2, n2), identity(n2, n2)
        integer :: i, j

        kernel = make_rbf_kernel(2, variance, lengthscale, status)
        call operator%initialize(x2, kernel, axis_extent*axis_extent, noise, &
            status)
        if (.not. status_ok(status)) then
            error = huge(1.0_dp)
            return
        end if
        identity = 0.0_dp
        do i = 1, n2
            identity(i, i) = 1.0_dp
        end do
        call operator%matmat(identity, approximate)
        do j = 1, n2
            do i = 1, n2
                exact(i, j) = rbf_vector(x2(i, :), x2(j, :))
            end do
            exact(j, j) = exact(j, j) + noise
        end do
        error = maxval(abs(approximate - exact))
    end function approximation_error_2d

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
        type(kernel_t) :: high_kernel, kernel, kernel2, nonseparable
        type(fortnum_status_t) :: status
        real(dp) :: degenerate(4, 2), extreme(2, 1), flat(4, 1)
        real(dp) :: high_dimensional(2, 31)
        real(dp) :: nonfinite(4, 2)
        real(dp) :: wide(4, 2)
        integer :: i

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
        extreme(:, 1) = [-huge(1.0_dp), huge(1.0_dp)]
        call operator%initialize(extreme, kernel, 2, noise, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] an overflowing grid range accepted"
            failures = failures + 1
        end if
        wide = 0.0_dp
        call operator%initialize(wide, kernel, 8, noise, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a two-dimensional input"
            failures = failures + 1
        end if

        do i = 1, 4
            wide(i, 1) = 0.2_dp*real(i - 1, dp)
            wide(i, 2) = -0.1_dp + 0.3_dp*real(i - 1, dp)
        end do
        kernel2 = make_rbf_kernel(2, variance, lengthscale, status)
        call operator%initialize(wide, kernel2, 3, noise, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] a 2-D budget below four points accepted"
            failures = failures + 1
        end if

        nonseparable = make_matern32_kernel(2, variance, lengthscale, status)
        call operator%initialize(wide, nonseparable, 16, noise, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] a nonseparable multidimensional kernel accepted"
            failures = failures + 1
        end if

        degenerate(:, 1) = wide(:, 1)
        degenerate(:, 2) = 0.0_dp
        call operator%initialize(degenerate, kernel2, 16, noise, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] a degenerate 2-D grid range accepted"
            failures = failures + 1
        end if

        nonfinite = wide
        nonfinite(2, 2) = ieee_value(0.0_dp, ieee_quiet_nan)
        call operator%initialize(nonfinite, kernel2, 16, noise, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] a non-finite grid coordinate accepted"
            failures = failures + 1
        end if

        high_dimensional(1, :) = 0.0_dp
        high_dimensional(2, :) = 1.0_dp
        high_kernel = make_rbf_kernel(31, variance, lengthscale, status)
        call operator%initialize(high_dimensional, high_kernel, huge(0), noise, &
            status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] an infeasible interpolation corner count accepted"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_ski_gp
