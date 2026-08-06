module fortml_ski_gp
    !! Structured kernel interpolation (SKI/KISS-GP) and subset of data (SoD).
    !!
    !! Following Liu, Ong, Shen and Cai, "When Gaussian Process Meets Big Data"
    !! (IEEE TNNLS 31(11):4405-4423, 2020), Sections III-A and III-C3.
    !!
    !! SoD is the baseline of Section III-A: fit an exact GP on `m` of the `n`
    !! points, costing `O(m^3)`, and accept that the discarded data are simply
    !! gone. It is implemented here as an index selection so a benchmark can
    !! compare it on the same footing as the approximations.
    !!
    !! SKI places the inducing points on a regular grid and interpolates the
    !! cross-covariance instead of evaluating it, `K_nm ~ W K_uu`, giving
    !!
    !!     K_nn ~ W K_uu W^T
    !!
    !! with `W` holding `2**d` nonzeros per row for local multilinear
    !! interpolation (paper, (19)-(20)). One matrix-vector product is then two
    !! sparse applications of `W` around one structured product with `K_uu`.
    !! A one-dimensional grid keeps the cached Toeplitz FFT implementation. In
    !! more dimensions, `n_grid` is a maximum total grid-point budget: every
    !! axis gets the largest common extent `q` for which `q**d <= n_grid`, and
    !! the separable grid has exactly `q**d` points. The multidimensional path
    !! accepts only a single isotropic RBF leaf, whose covariance is a
    !! Kronecker product of one-dimensional factors.
    !!
    !! Multilinear interpolation is continuous across grid-cell boundaries,
    !! but its input gradient is only piecewise continuous and can kink there.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortml_kernels, only: kernel_t, KERNEL_RBF
    use fortml_linear_operator, only: linear_operator_t
    use fortml_structured_operator, only: structured_gp_operator_t, &
        tensor_factor_t
    use fortml_toeplitz_operator, only: toeplitz_gp_operator_t
    implicit none
    private

    type, extends(linear_operator_t), public :: ski_operator_t
        !! `W K_uu W^T + noise I` on a regular tensor grid.
        type(toeplitz_gp_operator_t) :: grid_operator
        type(structured_gp_operator_t) :: structured_grid_operator
        real(dp), allocatable :: weight_value(:, :)
        integer, allocatable :: weight_index(:, :)
        real(dp), allocatable :: axis_weight_value(:, :, :)
        integer, allocatable :: axis_weight_index(:, :, :)
        real(dp), allocatable :: grid_lower(:)
        real(dp), allocatable :: grid_upper(:)
        real(dp), allocatable :: grid_spacing(:)
        real(dp) :: noise_variance = 0.0_dp
        real(dp) :: adjacent_grid_covariance = 0.0_dp
        integer :: n_points = 0
        integer :: n_grid = 0
        integer :: n_dimensions = 0
        integer :: n_corners = 0
        integer :: axis_extent = 0
    contains
        procedure, public :: initialize => ski_initialize
        procedure, public :: matvec => ski_matvec
        procedure, public :: matmat => ski_matmat
        procedure, public :: diagonal => ski_diagonal
        procedure, public :: cross_matvec => ski_cross_matvec
        procedure, public :: sample_count => ski_sample_count
        procedure, public :: interpolation_weights => ski_interpolation_weights
    end type ski_operator_t

    public :: subset_of_data_indices

contains

    subroutine subset_of_data_indices(n_samples, subset_size, indices, status)
        !! Evenly spaced subset selection: deterministic, spans the ordering,
        !! and needs no random stream, so a benchmark repeats exactly.
        integer, intent(in) :: n_samples, subset_size
        integer, allocatable, intent(out) :: indices(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (n_samples < 1 .or. subset_size < 1 .or. subset_size > n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "subset of data: the subset size must lie in [1, n]")
            return
        end if
        allocate(indices(subset_size))
        do i = 1, subset_size
            indices(i) = 1 + ((i - 1)*(n_samples - 1))/max(subset_size - 1, 1)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine subset_of_data_indices

    subroutine ski_initialize(self, inputs, kernel, n_grid, noise_variance, &
            status)
        class(ski_operator_t), intent(out) :: self
        real(dp), intent(in) :: inputs(:, :)
        type(kernel_t), intent(in) :: kernel
        integer, intent(in) :: n_grid
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status

        if (size(inputs, 1) < 1 .or. size(inputs, 2) < 1 .or. n_grid < 2 .or. &
            noise_variance < 0.0_dp .or. .not. ieee_is_finite(noise_variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: grid size, sample count or noise is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(inputs))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: input coordinates must be finite")
            return
        end if
        if (kernel%input_dim /= size(inputs, 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the kernel and sample dimensions differ")
            return
        end if

        self%n_points = size(inputs, 1)
        self%n_dimensions = size(inputs, 2)
        self%noise_variance = noise_variance
        if (self%n_dimensions == 1) then
            self%n_grid = n_grid
            self%n_corners = 2
            call ski_initialize_one_dimension(self, inputs, kernel, status)
        else
            call ski_initialize_structured(self, inputs, kernel, n_grid, status)
        end if
    end subroutine ski_initialize

    subroutine ski_initialize_one_dimension(self, inputs, kernel, status)
        class(ski_operator_t), intent(inout) :: self
        real(dp), intent(in) :: inputs(:, :)
        type(kernel_t), intent(in) :: kernel
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: column(:), grid_point(:), origin(:)
        real(dp) :: lower, upper, spacing, position, fraction
        integer :: allocation_status, i, cell

        lower = minval(inputs(:, 1))
        upper = maxval(inputs(:, 1))
        if (upper <= lower) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the inputs span no range to grid")
            return
        end if
        if (lower < 0.0_dp) then
            if (upper > huge(1.0_dp) + lower) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "SKI: the grid range is not finite")
                return
            end if
        end if
        spacing = (upper - lower)/real(self%n_grid - 1, dp)
        if (.not. ieee_is_finite(spacing) .or. spacing <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the grid range is not finite")
            return
        end if

        ! Local linear interpolation: two nonzeros per row.
        allocate(self%weight_value(2, self%n_points), &
            self%weight_index(2, self%n_points), self%grid_lower(1), &
            self%grid_upper(1), self%grid_spacing(1), stat=allocation_status)
        if (allocation_status /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the interpolation table is too large")
            return
        end if
        self%grid_lower(1) = lower
        self%grid_upper(1) = upper
        self%grid_spacing(1) = spacing
        self%axis_extent = self%n_grid
        do i = 1, self%n_points
            position = (inputs(i, 1) - lower)/spacing
            cell = min(max(int(position) + 1, 1), self%n_grid - 1)
            fraction = position - real(cell - 1, dp)
            fraction = min(max(fraction, 0.0_dp), 1.0_dp)
            self%weight_index(1, i) = cell
            self%weight_index(2, i) = cell + 1
            self%weight_value(1, i) = 1.0_dp - fraction
            self%weight_value(2, i) = fraction
        end do

        ! The grid covariance is Toeplitz for a stationary kernel on a regular
        ! grid, so only its first column is needed.
        allocate(column(self%n_grid), grid_point(1), origin(1), &
            stat=allocation_status)
        if (allocation_status /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the one-dimensional grid is too large")
            return
        end if
        origin(1) = lower
        do i = 1, self%n_grid
            grid_point(1) = lower + real(i - 1, dp)*spacing
            column(i) = kernel%value(origin, grid_point)
        end do
        self%adjacent_grid_covariance = column(2)
        call self%grid_operator%initialize(column, status)
    end subroutine ski_initialize_one_dimension

    subroutine ski_initialize_structured(self, inputs, kernel, grid_budget, &
            status)
        class(ski_operator_t), intent(inout) :: self
        real(dp), intent(in) :: inputs(:, :)
        type(kernel_t), intent(in) :: kernel
        integer, intent(in) :: grid_budget
        type(fortnum_status_t), intent(out) :: status

        type(tensor_factor_t), allocatable :: factors(:)
        real(dp), allocatable :: lower(:), spacing(:), upper(:)
        real(dp) :: axis_covariance, fraction, grid_i, grid_j, lengthscale
        real(dp) :: position, variance, weight
        integer :: allocation_status, axis_extent, cell, choice, corner
        integer :: dimension, flat_index, grid_points, i, j, stride

        if (kernel%kind /= KERNEL_RBF) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: multidimensional grids require one separable RBF leaf")
            return
        end if
        if (.not. allocated(kernel%log_parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the RBF kernel parameters are unavailable")
            return
        end if
        if (size(kernel%log_parameters) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the RBF kernel parameter layout is invalid")
            return
        end if

        if (any(.not. ieee_is_finite(kernel%log_parameters)) .or. &
            any(kernel%log_parameters > log(huge(1.0_dp))) .or. &
            any(kernel%log_parameters < log(tiny(1.0_dp)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the RBF kernel scales must be finite and positive")
            return
        end if
        variance = exp(kernel%log_parameters(1))
        lengthscale = exp(kernel%log_parameters(2))
        if (.not. ieee_is_finite(variance) .or. variance <= 0.0_dp .or. &
            .not. ieee_is_finite(lengthscale) .or. lengthscale <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the RBF kernel scales must be finite and positive")
            return
        end if

        axis_extent = largest_equal_grid_extent(grid_budget, self%n_dimensions)
        if (axis_extent < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the grid budget cannot provide two points per dimension")
            return
        end if
        if (axis_extent > huge(0)/axis_extent) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: a structured grid factor is too large")
            return
        end if

        grid_points = 1
        self%n_corners = 1
        do dimension = 1, self%n_dimensions
            if (grid_points > grid_budget/axis_extent) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "SKI: the structured grid size overflows its budget")
                return
            end if
            grid_points = grid_points*axis_extent
            if (self%n_corners > huge(0)/2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "SKI: the interpolation corner count overflows")
                return
            end if
            self%n_corners = 2*self%n_corners
        end do
        if (self%n_points > huge(0)/self%n_corners) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the interpolation table is too large")
            return
        end if
        self%n_grid = grid_points

        allocate(lower(self%n_dimensions), upper(self%n_dimensions), &
            spacing(self%n_dimensions), factors(self%n_dimensions), &
            stat=allocation_status)
        if (allocation_status /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the structured-grid metadata is too large")
            return
        end if
        do dimension = 1, self%n_dimensions
            lower(dimension) = minval(inputs(:, dimension))
            upper(dimension) = maxval(inputs(:, dimension))
            if (upper(dimension) <= lower(dimension)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "SKI: every input dimension must span a grid range")
                return
            end if
            if (lower(dimension) < 0.0_dp) then
                if (upper(dimension) > huge(1.0_dp) + lower(dimension)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "SKI: a grid range is not finite")
                    return
                end if
            end if
            spacing(dimension) = (upper(dimension) - lower(dimension))/ &
                real(axis_extent - 1, dp)
            if (.not. ieee_is_finite(spacing(dimension)) .or. &
                spacing(dimension) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "SKI: a grid range is not finite")
                return
            end if
        end do

        allocate(self%axis_weight_value(2, self%n_dimensions, self%n_points), &
            self%axis_weight_index(2, self%n_dimensions, self%n_points), &
            self%weight_value(self%n_corners, self%n_points), &
            self%weight_index(self%n_corners, self%n_points), &
            self%grid_lower(self%n_dimensions), &
            self%grid_upper(self%n_dimensions), &
            self%grid_spacing(self%n_dimensions), &
            stat=allocation_status)
        if (allocation_status /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the interpolation table is too large")
            return
        end if
        self%grid_lower = lower
        self%grid_upper = upper
        self%grid_spacing = spacing
        self%axis_extent = axis_extent

        do i = 1, self%n_points
            do dimension = 1, self%n_dimensions
                position = (inputs(i, dimension) - lower(dimension))/ &
                    spacing(dimension)
                cell = min(max(int(position) + 1, 1), axis_extent - 1)
                fraction = position - real(cell - 1, dp)
                fraction = min(max(fraction, 0.0_dp), 1.0_dp)
                self%axis_weight_index(:, dimension, i) = [cell, cell + 1]
                self%axis_weight_value(:, dimension, i) = &
                    [1.0_dp - fraction, fraction]
            end do

            do corner = 0, self%n_corners - 1
                flat_index = 1
                stride = 1
                weight = 1.0_dp
                do dimension = 1, self%n_dimensions
                    choice = 1
                    if (btest(corner, dimension - 1)) choice = 2
                    flat_index = flat_index + &
                        (self%axis_weight_index(choice, dimension, i) - 1)*stride
                    weight = weight*self%axis_weight_value(choice, dimension, i)
                    stride = stride*axis_extent
                end do
                self%weight_index(corner + 1, i) = flat_index
                self%weight_value(corner + 1, i) = weight
            end do
        end do

        do dimension = 1, self%n_dimensions
            allocate(factors(dimension)%values(axis_extent, axis_extent), &
                stat=allocation_status)
            if (allocation_status /= 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "SKI: a structured grid factor is too large")
                return
            end if
            do j = 1, axis_extent
                grid_j = lower(dimension) + real(j - 1, dp)*spacing(dimension)
                do i = 1, axis_extent
                    grid_i = lower(dimension) + &
                        real(i - 1, dp)*spacing(dimension)
                    axis_covariance = rbf_axis_correlation( &
                        abs(grid_i - grid_j), lengthscale)
                    if (dimension == 1) axis_covariance = variance*axis_covariance
                    factors(dimension)%values(i, j) = axis_covariance
                end do
            end do
        end do
        call self%structured_grid_operator%initialize(factors, status)
    end subroutine ski_initialize_structured

    integer function largest_equal_grid_extent(grid_budget, n_dimensions) &
            result(extent)
        integer, intent(in) :: grid_budget, n_dimensions
        integer :: comparison, lower, middle, upper

        extent = 0
        if (grid_budget < 1 .or. n_dimensions < 1) return
        lower = 1
        upper = grid_budget
        do while (lower <= upper)
            middle = lower + (upper - lower)/2
            comparison = compare_integer_power(middle, n_dimensions, grid_budget)
            if (comparison <= 0) then
                extent = middle
                if (middle == huge(0)) exit
                lower = middle + 1
            else
                upper = middle - 1
            end if
        end do
    end function largest_equal_grid_extent

    integer function compare_integer_power(base, exponent, limit) result(comparison)
        integer, intent(in) :: base, exponent, limit
        integer :: factor, product

        product = 1
        do factor = 1, exponent
            if (product > limit/base) then
                comparison = 1
                return
            end if
            product = product*base
        end do
        if (product < limit) then
            comparison = -1
        else
            comparison = 0
        end if
    end function compare_integer_power

    real(dp) function rbf_axis_correlation(distance, lengthscale) result(value)
        !! Evaluate exp(-0.5*(distance/lengthscale)**2) without overflowing
        !! either the division or the square; the limiting value is zero.
        real(dp), intent(in) :: distance, lengthscale
        real(dp) :: scaled_distance

        if (lengthscale < 1.0_dp) then
            if (distance > huge(1.0_dp)*lengthscale) then
                value = 0.0_dp
                return
            end if
        end if
        scaled_distance = distance/lengthscale
        if (scaled_distance > sqrt(huge(1.0_dp))) then
            value = 0.0_dp
        else
            value = exp(-0.5_dp*scaled_distance**2)
        end if
    end function rbf_axis_correlation

    subroutine ski_interpolation_weights(self, row, indices, values)
        !! The two grid nodes and weights of a one-dimensional data row.
        class(ski_operator_t), intent(in) :: self
        integer, intent(in) :: row
        integer, intent(out) :: indices(2)
        real(dp), intent(out) :: values(2)

        indices = 0
        values = 0.0_dp
        if (row < 1 .or. row > self%n_points) return
        if (self%n_dimensions /= 1) return
        indices = self%weight_index(:, row)
        values = self%weight_value(:, row)
    end subroutine ski_interpolation_weights

    subroutine ski_matvec(self, input, output)
        class(ski_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)
        real(dp), allocatable :: grid_input(:), grid_output(:)
        integer :: corner, i

        output = 0.0_dp
        if (self%n_points < 1) return
        if (size(input) /= self%n_points .or. size(output) /= self%n_points) return
        allocate(grid_input(self%n_grid), grid_output(self%n_grid))
        grid_input = 0.0_dp
        ! W^T v
        do i = 1, self%n_points
            do corner = 1, self%n_corners
                grid_input(self%weight_index(corner, i)) = &
                    grid_input(self%weight_index(corner, i)) + &
                    self%weight_value(corner, i)*input(i)
            end do
        end do
        if (self%n_dimensions == 1) then
            call self%grid_operator%matvec(grid_input, grid_output)
        else
            call self%structured_grid_operator%matvec(grid_input, grid_output)
        end if
        ! W (K_uu W^T v) + noise v
        do i = 1, self%n_points
            do corner = 1, self%n_corners
                output(i) = output(i) + self%weight_value(corner, i)* &
                    grid_output(self%weight_index(corner, i))
            end do
            output(i) = output(i) + self%noise_variance*input(i)
        end do
    end subroutine ski_matvec

    subroutine ski_matmat(self, input, output)
        class(ski_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        real(dp), allocatable :: grid_input(:, :), grid_output(:, :)
        integer :: column, corner, i

        output = 0.0_dp
        if (self%n_points < 1) return
        if (size(input, 1) /= self%n_points) return
        if (size(output, 1) /= self%n_points) return
        if (size(input, 2) /= size(output, 2)) return
        if (self%n_dimensions == 1) then
            do column = 1, size(input, 2)
                call ski_matvec(self, input(:, column), output(:, column))
            end do
            return
        end if
        if (size(input, 2) < 1) return

        allocate(grid_input(self%n_grid, size(input, 2)))
        allocate(grid_output(self%n_grid, size(input, 2)))
        grid_input = 0.0_dp
        do column = 1, size(input, 2)
            do i = 1, self%n_points
                do corner = 1, self%n_corners
                    grid_input(self%weight_index(corner, i), column) = &
                        grid_input(self%weight_index(corner, i), column) + &
                        self%weight_value(corner, i)*input(i, column)
                end do
            end do
        end do
        call self%structured_grid_operator%matmat(grid_input, grid_output)
        do column = 1, size(input, 2)
            do i = 1, self%n_points
                do corner = 1, self%n_corners
                    output(i, column) = output(i, column) + &
                        self%weight_value(corner, i)* &
                        grid_output(self%weight_index(corner, i), column)
                end do
                output(i, column) = output(i, column) + &
                    self%noise_variance*input(i, column)
            end do
        end do
    end subroutine ski_matmat

    subroutine ski_cross_matvec(self, query, input, output, status)
        !! Apply the interpolated cross-covariance
        !! `W_query K_uu W_train^T` to training-space coefficients. Queries
        !! beyond a training-grid face use the nearest boundary interpolation.
        class(ski_operator_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :), input(:)
        real(dp), intent(out) :: output(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: fraction(:), grid_input(:), grid_output(:)
        real(dp) :: weight
        integer, allocatable :: cell(:)
        integer :: allocation_status, choice, corner, dimension, flat_index
        integer :: i, stride

        output = 0.0_dp
        if (self%n_points < 1 .or. self%n_grid < 1 .or. &
            self%n_dimensions < 1 .or. self%axis_extent < 2 .or. &
            size(input) /= self%n_points .or. &
            size(query, 2) /= self%n_dimensions .or. &
            size(output) /= size(query, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: cross-covariance product shape is invalid")
            return
        end if
        if (.not. allocated(self%grid_lower) .or. &
            .not. allocated(self%grid_upper) .or. &
            .not. allocated(self%grid_spacing)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: cross-covariance grid metadata is unavailable")
            return
        end if
        if (any(.not. ieee_is_finite(query)) .or. &
            any(.not. ieee_is_finite(input))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: cross-covariance inputs must be finite")
            return
        end if

        allocate(grid_input(self%n_grid), grid_output(self%n_grid), &
            cell(self%n_dimensions), fraction(self%n_dimensions), &
            stat=allocation_status)
        if (allocation_status /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: cross-covariance workspace is too large")
            return
        end if
        grid_input = 0.0_dp
        do i = 1, self%n_points
            do corner = 1, self%n_corners
                grid_input(self%weight_index(corner, i)) = &
                    grid_input(self%weight_index(corner, i)) + &
                    self%weight_value(corner, i)*input(i)
            end do
        end do
        if (self%n_dimensions == 1) then
            call self%grid_operator%matvec(grid_input, grid_output)
        else
            call self%structured_grid_operator%matvec(grid_input, grid_output)
        end if

        do i = 1, size(query, 1)
            do dimension = 1, self%n_dimensions
                call query_interpolation_coordinate(self, &
                    query(i, dimension), dimension, cell(dimension), &
                    fraction(dimension))
            end do
            do corner = 0, self%n_corners - 1
                flat_index = 1
                stride = 1
                weight = 1.0_dp
                do dimension = 1, self%n_dimensions
                    choice = 0
                    if (btest(corner, dimension - 1)) choice = 1
                    flat_index = flat_index + &
                        (cell(dimension) + choice - 1)*stride
                    if (choice == 0) then
                        weight = weight*(1.0_dp - fraction(dimension))
                    else
                        weight = weight*fraction(dimension)
                    end if
                    stride = stride*self%axis_extent
                end do
                output(i) = output(i) + weight*grid_output(flat_index)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ski_cross_matvec

    subroutine query_interpolation_coordinate(self, value, dimension, cell, &
            fraction)
        class(ski_operator_t), intent(in) :: self
        real(dp), intent(in) :: value
        integer, intent(in) :: dimension
        integer, intent(out) :: cell
        real(dp), intent(out) :: fraction
        real(dp) :: position

        if (value <= self%grid_lower(dimension)) then
            cell = 1
            fraction = 0.0_dp
        else if (value >= self%grid_upper(dimension)) then
            cell = self%axis_extent - 1
            fraction = 1.0_dp
        else
            position = (value - self%grid_lower(dimension))/ &
                self%grid_spacing(dimension)
            cell = min(max(int(position) + 1, 1), self%axis_extent - 1)
            fraction = position - real(cell - 1, dp)
            fraction = min(max(fraction, 0.0_dp), 1.0_dp)
        end if
    end subroutine query_interpolation_coordinate

    function ski_diagonal(self) result(values)
        class(ski_operator_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        real(dp), allocatable :: grid_diagonal(:)
        real(dp) :: axis_value, left_weight, point_value, right_weight
        integer :: dimension, i, left_index, right_index

        allocate(values(max(self%n_points, 0)))
        if (self%n_points < 1) return
        if (self%n_dimensions == 1) then
            grid_diagonal = self%grid_operator%diagonal()
            do i = 1, self%n_points
                values(i) = (self%weight_value(1, i)**2 + &
                    self%weight_value(2, i)**2)*grid_diagonal(1) + &
                    2.0_dp*self%weight_value(1, i)*self%weight_value(2, i)* &
                    self%adjacent_grid_covariance + self%noise_variance
            end do
            return
        end if

        do i = 1, self%n_points
            point_value = 1.0_dp
            do dimension = 1, self%n_dimensions
                left_index = self%axis_weight_index(1, dimension, i)
                right_index = self%axis_weight_index(2, dimension, i)
                left_weight = self%axis_weight_value(1, dimension, i)
                right_weight = self%axis_weight_value(2, dimension, i)
                axis_value = left_weight*left_weight* &
                    self%structured_grid_operator%product_operator% &
                    factors(dimension)%values(left_index, left_index)
                axis_value = axis_value + left_weight*right_weight*( &
                    self%structured_grid_operator%product_operator% &
                    factors(dimension)%values(left_index, right_index) + &
                    self%structured_grid_operator%product_operator% &
                    factors(dimension)%values(right_index, left_index))
                axis_value = axis_value + right_weight*right_weight* &
                    self%structured_grid_operator%product_operator% &
                    factors(dimension)%values(right_index, right_index)
                point_value = point_value*axis_value
            end do
            values(i) = point_value + self%noise_variance
        end do
    end function ski_diagonal

    integer function ski_sample_count(self) result(count)
        class(ski_operator_t), intent(in) :: self

        count = self%n_points
    end function ski_sample_count

end module fortml_ski_gp
