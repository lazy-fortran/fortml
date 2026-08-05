module fortml_kernel_operator
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_linear_operator, only: linear_operator_t
    use fortml_kernels, only: kernel_t
    implicit none
    private

    integer, parameter :: DEFAULT_TILE_SIZE = 128

    type, extends(linear_operator_t), public :: rbf_operator_t
        ! Neighbor index is contiguous for the matrix-free inner reduction.
        real(dp), allocatable :: points(:, :)
        real(dp) :: variance = 1.0_dp
        real(dp) :: lengthscale = 1.0_dp
        real(dp) :: diagonal_shift = 0.0_dp
        integer :: tile_size = DEFAULT_TILE_SIZE
    contains
        procedure, public :: initialize => rbf_operator_initialize
        procedure, public :: matvec => rbf_operator_matvec
        procedure, public :: matmat => rbf_operator_matmat
        procedure, public :: diagonal => rbf_operator_diagonal
        procedure, public :: sample_count => rbf_operator_sample_count
    end type rbf_operator_t

    type, extends(linear_operator_t), public :: kernel_operator_t
        type(kernel_t) :: kernel
        real(dp), allocatable :: points(:, :)
        real(dp) :: diagonal_shift = 0.0_dp
        integer :: tile_size = DEFAULT_TILE_SIZE
    contains
        procedure, public :: initialize => kernel_operator_initialize
        procedure, public :: matvec => kernel_operator_matvec
        procedure, public :: matmat => kernel_operator_matmat
        procedure, public :: diagonal => kernel_operator_diagonal
        procedure, public :: sample_count => kernel_operator_sample_count
    end type kernel_operator_t

    public :: rbf_matvec_tiled
    public :: rbf_matmat_tiled

contains

    subroutine rbf_operator_initialize( &
            self, sample_points, variance, lengthscale, diagonal_shift, &
            status, tile_size)
        class(rbf_operator_t), intent(out) :: self
        real(dp), intent(in) :: sample_points(:, :)
        real(dp), intent(in) :: variance, lengthscale, diagonal_shift
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: tile_size

        if (size(sample_points, 1) < 1 .or. size(sample_points, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF operator: sample points must be nonempty")
            return
        end if
        if (variance <= 0.0_dp .or. lengthscale <= 0.0_dp .or. &
            diagonal_shift < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF operator: scale and diagonal shift are invalid")
            return
        end if
        self%variance = variance
        self%lengthscale = lengthscale
        self%diagonal_shift = diagonal_shift
        self%tile_size = DEFAULT_TILE_SIZE
        if (present(tile_size)) self%tile_size = tile_size
        if (self%tile_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF operator: tile size must be positive")
            return
        end if
        allocate(self%points(size(sample_points, 1), size(sample_points, 2)))
        self%points = sample_points
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_operator_initialize

    subroutine rbf_operator_matvec(self, input, output)
        class(rbf_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)

        call rbf_matvec_tiled( &
            self%points, input, output, self%variance, self%lengthscale, &
            self%diagonal_shift, self%tile_size)
    end subroutine rbf_operator_matvec

    subroutine rbf_operator_matmat(self, input, output)
        class(rbf_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)

        call rbf_matmat_tiled( &
            self%points, input, output, self%variance, self%lengthscale, &
            self%diagonal_shift, self%tile_size)
    end subroutine rbf_operator_matmat

    function rbf_operator_diagonal(self) result(values)
        class(rbf_operator_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        allocate(values(self%sample_count()))
        values = self%variance + self%diagonal_shift
    end function rbf_operator_diagonal

    integer function rbf_operator_sample_count(self) result(count)
        class(rbf_operator_t), intent(in) :: self

        if (allocated(self%points)) then
            count = size(self%points, 1)
        else
            count = 0
        end if
    end function rbf_operator_sample_count

    subroutine kernel_operator_initialize( &
            self, sample_points, kernel, diagonal_shift, status, tile_size)
        class(kernel_operator_t), intent(out) :: self
        real(dp), intent(in) :: sample_points(:, :)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: diagonal_shift
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: tile_size

        if (size(sample_points, 1) < 1 .or. size(sample_points, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: sample points must be nonempty")
            return
        end if
        if (kernel%input_dim /= size(sample_points, 2) .or. &
            kernel%parameter_count() < 1 .or. diagonal_shift < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: kernel or diagonal shift is invalid")
            return
        end if
        self%kernel = kernel
        self%diagonal_shift = diagonal_shift
        self%tile_size = DEFAULT_TILE_SIZE
        if (present(tile_size)) self%tile_size = tile_size
        if (self%tile_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: tile size must be positive")
            return
        end if
        allocate(self%points(size(sample_points, 1), size(sample_points, 2)))
        self%points = sample_points
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_operator_initialize

    subroutine kernel_operator_matvec(self, input, output)
        class(kernel_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)

        real(dp), allocatable :: matrix_block(:, :)
        type(fortnum_status_t) :: status
        integer :: block_size, first_row, last_row, n_samples

        output = 0.0_dp
        n_samples = self%sample_count()
        if (n_samples < 1) return
        if (size(input) /= n_samples .or. size(output) /= n_samples) return
        if (self%tile_size < 1) return
        block_size = min(self%tile_size, n_samples)
        allocate(matrix_block(block_size, n_samples))
        do first_row = 1, n_samples, block_size
            last_row = min(n_samples, first_row + block_size - 1)
            call self%kernel%matrix( &
                self%points(first_row:last_row, :), self%points, &
                matrix_block(:last_row - first_row + 1, :), status)
            if (status%code /= FORTNUM_OK) return
            output(first_row:last_row) = self%diagonal_shift* &
                input(first_row:last_row) + matmul( &
                matrix_block(:last_row - first_row + 1, :), input)
        end do
    end subroutine kernel_operator_matvec

    subroutine kernel_operator_matmat(self, input, output)
        class(kernel_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        integer :: right_hand_side

        output = 0.0_dp
        if (size(output, 1) /= self%sample_count()) return
        if (size(output, 2) /= size(input, 2)) return
        do right_hand_side = 1, size(input, 2)
            call self%matvec(input(:, right_hand_side), output(:, right_hand_side))
        end do
    end subroutine kernel_operator_matmat

    function kernel_operator_diagonal(self) result(values)
        class(kernel_operator_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: i

        allocate(values(self%sample_count()))
        do i = 1, self%sample_count()
            values(i) = self%diagonal_shift + self%kernel%value( &
                self%points(i, :), self%points(i, :))
        end do
    end function kernel_operator_diagonal

    integer function kernel_operator_sample_count(self) result(count)
        class(kernel_operator_t), intent(in) :: self

        if (allocated(self%points)) then
            count = size(self%points, 1)
        else
            count = 0
        end if
    end function kernel_operator_sample_count

    subroutine rbf_matvec_tiled( &
            points, input, output, variance, lengthscale, diagonal_shift, &
            tile_size)
        real(dp), intent(in) :: points(:, :), input(:)
        real(dp), intent(out) :: output(:)
        real(dp), intent(in) :: variance, lengthscale, diagonal_shift
        integer, intent(in) :: tile_size
        real(dp) :: inverse_two_lengthscale_squared
        real(dp) :: accumulated, distance, difference, kernel_value
        integer :: block_size, first_neighbor, last_neighbor
        integer :: feature, i, j, n_features, n_samples

        output = 0.0_dp
        if (size(points, 1) < 1 .or. size(points, 2) < 1) return
        if (size(input) /= size(points, 1)) return
        if (size(output) /= size(input)) return
        if (variance <= 0.0_dp .or. lengthscale <= 0.0_dp) return
        if (tile_size < 1) return

        n_samples = size(points, 1)
        n_features = size(points, 2)
        block_size = tile_size
        inverse_two_lengthscale_squared = &
            0.5_dp/(lengthscale*lengthscale)

        if (n_features == 8) then
            call rbf_matvec_tiled_8( &
                points, input, output, variance, lengthscale, diagonal_shift, &
                tile_size)
            return
        end if

        ! Without an enclosing data region the compiler maps the arrays for
        ! this call. An enclosing resident region suppresses those transfers.
        !$acc parallel loop
        !$omp parallel do schedule(static) private( &
        !$omp& accumulated, distance, difference, kernel_value, &
        !$omp& first_neighbor, last_neighbor, feature, j)
        do i = 1, n_samples
            accumulated = diagonal_shift*input(i)
            do first_neighbor = 1, n_samples, block_size
                last_neighbor = min(n_samples, first_neighbor + block_size - 1)
                !$acc loop vector reduction(+:accumulated)
                !$omp simd reduction(+:accumulated)
                do j = first_neighbor, last_neighbor
                    distance = 0.0_dp
                    do feature = 1, n_features
                        difference = points(i, feature) - points(j, feature)
                        distance = distance + difference*difference
                    end do
                    kernel_value = variance*exp( &
                        -inverse_two_lengthscale_squared*distance)
                    accumulated = accumulated + kernel_value*input(j)
                end do
            end do
            output(i) = accumulated
        end do
    end subroutine rbf_matvec_tiled

    subroutine rbf_matvec_tiled_8( &
            points, input, output, variance, lengthscale, diagonal_shift, &
            tile_size)
        real(dp), intent(in) :: points(:, :), input(:)
        real(dp), intent(out) :: output(:)
        real(dp), intent(in) :: variance, lengthscale, diagonal_shift
        integer, intent(in) :: tile_size
        real(dp) :: inverse_two_lengthscale_squared
        real(dp) :: accumulated, distance, kernel_value
        real(dp) :: point_1, point_2, point_3, point_4
        real(dp) :: point_5, point_6, point_7, point_8
        integer :: block_size, first_neighbor, last_neighbor
        integer :: i, j, n_samples

        n_samples = size(points, 1)
        block_size = tile_size
        inverse_two_lengthscale_squared = &
            0.5_dp/(lengthscale*lengthscale)
        output = 0.0_dp

        !$acc parallel loop
        !$omp parallel do schedule(static) private( &
        !$omp& accumulated, distance, kernel_value, point_1, point_2, &
        !$omp& point_3, point_4, point_5, point_6, point_7, point_8, &
        !$omp& first_neighbor, last_neighbor, j)
        do i = 1, n_samples
            point_1 = points(i, 1)
            point_2 = points(i, 2)
            point_3 = points(i, 3)
            point_4 = points(i, 4)
            point_5 = points(i, 5)
            point_6 = points(i, 6)
            point_7 = points(i, 7)
            point_8 = points(i, 8)
            accumulated = diagonal_shift*input(i)
            do first_neighbor = 1, n_samples, block_size
                last_neighbor = min(n_samples, first_neighbor + block_size - 1)
                !$acc loop vector reduction(+:accumulated)
                !$omp simd reduction(+:accumulated)
                do j = first_neighbor, last_neighbor
                    distance = (point_1 - points(j, 1))*(point_1 - points(j, 1)) + &
                        (point_2 - points(j, 2))*(point_2 - points(j, 2)) + &
                        (point_3 - points(j, 3))*(point_3 - points(j, 3)) + &
                        (point_4 - points(j, 4))*(point_4 - points(j, 4)) + &
                        (point_5 - points(j, 5))*(point_5 - points(j, 5)) + &
                        (point_6 - points(j, 6))*(point_6 - points(j, 6)) + &
                        (point_7 - points(j, 7))*(point_7 - points(j, 7)) + &
                        (point_8 - points(j, 8))*(point_8 - points(j, 8))
                    kernel_value = variance*exp( &
                        -inverse_two_lengthscale_squared*distance)
                    accumulated = accumulated + kernel_value*input(j)
                end do
            end do
            output(i) = accumulated
        end do
    end subroutine rbf_matvec_tiled_8

    subroutine rbf_matmat_tiled( &
            points, input, output, variance, lengthscale, diagonal_shift, &
            tile_size)
        real(dp), intent(in) :: points(:, :), input(:, :)
        real(dp), intent(out) :: output(:, :)
        real(dp), intent(in) :: variance, lengthscale, diagonal_shift
        integer, intent(in) :: tile_size
        integer :: right_hand_side

        output = 0.0_dp
        if (size(output, 1) /= size(points, 1)) return
        if (size(output, 2) /= size(input, 2)) return
        do right_hand_side = 1, size(input, 2)
            call rbf_matvec_tiled( &
                points, input(:, right_hand_side), output(:, right_hand_side), &
                variance, lengthscale, diagonal_shift, tile_size)
        end do
    end subroutine rbf_matmat_tiled

end module fortml_kernel_operator
