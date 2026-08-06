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
    !! with `W` holding two nonzeros per row per dimension for local linear
    !! interpolation (paper, (19)-(20)). One matrix-vector product is then two
    !! sparse applications of `W` around one structured product with `K_uu`,
    !! which on a one-dimensional grid is the cached Toeplitz FFT product this
    !! repository already owns. The cost is `O(n + m log m)` per product
    !! against `O(n^2)` for the dense kernel, and the approximation converges
    !! to the exact kernel as the grid is refined.
    !!
    !! The interpolation is what produces the discontinuous predictions the
    !! review reports for SKI in its Fig. 4: the weights are piecewise linear
    !! in the input, so the predictive mean has a kink at every grid cell edge.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_kernels, only: kernel_t
    use fortml_linear_operator, only: linear_operator_t
    use fortml_toeplitz_operator, only: toeplitz_gp_operator_t
    implicit none
    private

    type, extends(linear_operator_t), public :: ski_operator_t
        !! `W K_uu W^T + noise I` for a one-dimensional regular grid.
        type(toeplitz_gp_operator_t) :: grid_operator
        real(dp), allocatable :: weight_value(:, :)
        integer, allocatable :: weight_index(:, :)
        real(dp) :: noise_variance = 0.0_dp
        integer :: n_points = 0
        integer :: n_grid = 0
    contains
        procedure, public :: initialize => ski_initialize
        procedure, public :: matvec => ski_matvec
        procedure, public :: matmat => ski_matmat
        procedure, public :: diagonal => ski_diagonal
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

        real(dp), allocatable :: column(:), grid_point(:), origin(:)
        real(dp) :: lower, upper, spacing, position, fraction
        integer :: i, cell

        if (size(inputs, 2) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: this grid path covers one input dimension")
            return
        end if
        if (size(inputs, 1) < 1 .or. n_grid < 2 .or. noise_variance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: grid size, sample count or noise is invalid")
            return
        end if
        if (kernel%input_dim /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the kernel must be one-dimensional here")
            return
        end if

        self%n_points = size(inputs, 1)
        self%n_grid = n_grid
        self%noise_variance = noise_variance
        lower = minval(inputs(:, 1))
        upper = maxval(inputs(:, 1))
        if (upper <= lower) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "SKI: the inputs span no range to grid")
            return
        end if
        spacing = (upper - lower)/real(n_grid - 1, dp)

        ! Local linear interpolation: two nonzeros per row.
        allocate(self%weight_value(2, self%n_points))
        allocate(self%weight_index(2, self%n_points))
        do i = 1, self%n_points
            position = (inputs(i, 1) - lower)/spacing
            cell = min(max(int(position) + 1, 1), n_grid - 1)
            fraction = position - real(cell - 1, dp)
            fraction = min(max(fraction, 0.0_dp), 1.0_dp)
            self%weight_index(1, i) = cell
            self%weight_index(2, i) = cell + 1
            self%weight_value(1, i) = 1.0_dp - fraction
            self%weight_value(2, i) = fraction
        end do

        ! The grid covariance is Toeplitz for a stationary kernel on a regular
        ! grid, so only its first column is needed.
        allocate(column(n_grid), grid_point(1), origin(1))
        origin(1) = lower
        do i = 1, n_grid
            grid_point(1) = lower + real(i - 1, dp)*spacing
            column(i) = kernel%value(origin, grid_point)
        end do
        call self%grid_operator%initialize(column, status)
    end subroutine ski_initialize

    subroutine ski_interpolation_weights(self, row, indices, values)
        !! The two grid nodes and weights of one data row, so a test can
        !! rebuild `W` densely without reaching into the type's storage layout.
        class(ski_operator_t), intent(in) :: self
        integer, intent(in) :: row
        integer, intent(out) :: indices(2)
        real(dp), intent(out) :: values(2)

        indices = 0
        values = 0.0_dp
        if (row < 1 .or. row > self%n_points) return
        indices = self%weight_index(:, row)
        values = self%weight_value(:, row)
    end subroutine ski_interpolation_weights

    subroutine ski_matvec(self, input, output)
        class(ski_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)
        real(dp), allocatable :: grid_input(:), grid_output(:)
        integer :: i

        output = 0.0_dp
        if (size(input) /= self%n_points .or. size(output) /= self%n_points) return
        allocate(grid_input(self%n_grid), grid_output(self%n_grid))
        grid_input = 0.0_dp
        ! W^T v
        do i = 1, self%n_points
            grid_input(self%weight_index(1, i)) = &
                grid_input(self%weight_index(1, i)) + self%weight_value(1, i)*input(i)
            grid_input(self%weight_index(2, i)) = &
                grid_input(self%weight_index(2, i)) + self%weight_value(2, i)*input(i)
        end do
        call self%grid_operator%matvec(grid_input, grid_output)
        ! W (K_uu W^T v) + noise v
        do i = 1, self%n_points
            output(i) = self%weight_value(1, i)*grid_output(self%weight_index(1, i)) &
                + self%weight_value(2, i)*grid_output(self%weight_index(2, i)) &
                + self%noise_variance*input(i)
        end do
    end subroutine ski_matvec

    subroutine ski_matmat(self, input, output)
        class(ski_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        integer :: column

        output = 0.0_dp
        if (size(input, 2) /= size(output, 2)) return
        do column = 1, size(input, 2)
            call ski_matvec(self, input(:, column), output(:, column))
        end do
    end subroutine ski_matmat

    function ski_diagonal(self) result(values)
        class(ski_operator_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        real(dp), allocatable :: grid_diagonal(:)
        integer :: i

        allocate(values(max(self%n_points, 0)))
        if (self%n_points < 1) return
        grid_diagonal = self%grid_operator%diagonal()
        do i = 1, self%n_points
            ! Only the diagonal grid entries contribute at leading order; the
            ! cross term uses the first off-diagonal of the Toeplitz column.
            values(i) = (self%weight_value(1, i)**2 + self%weight_value(2, i)**2)* &
                grid_diagonal(1) + self%noise_variance
        end do
    end function ski_diagonal

    integer function ski_sample_count(self) result(count)
        class(ski_operator_t), intent(in) :: self

        count = self%n_points
    end function ski_sample_count

end module fortml_ski_gp
