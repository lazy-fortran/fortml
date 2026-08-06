module fortml_kernel_operator
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_loc, c_ptr
    use fortnum_kinds, only: dp
    use fortnum_krylov, only: KRYLOV_BREAKDOWN, KRYLOV_INVALID_ARGUMENT, &
        KRYLOV_MAX_ITERATIONS, KRYLOV_OK
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_linear_operator, only: linear_operator_t
    use fortml_kernels, only: KERNEL_CONSTANT, KERNEL_LINEAR, KERNEL_MATERN12, &
        KERNEL_MATERN32, KERNEL_MATERN52, KERNEL_PRODUCT, KERNEL_RBF, &
        KERNEL_SUM, KERNEL_WHITE_NOISE, kernel_t
    implicit none
    private

    integer, parameter :: DEFAULT_TILE_SIZE = 128
    integer, parameter :: MAX_FUSED_RHS = 8
    integer, parameter :: MAX_KERNEL_PROGRAM = 64

    type :: rbf_krylov_workspace_t
        real(dp), allocatable :: residual(:, :), direction(:, :)
        real(dp), allocatable :: preconditioned(:, :)
        real(dp), allocatable :: operator_direction(:, :), work(:, :)
        real(dp), allocatable :: right_hand_side_norm(:), target(:)
        real(dp), allocatable :: rho(:), next_rho(:), denominator(:)
        logical, allocatable :: active(:), candidate(:)
        integer :: n_samples = 0
        integer :: n_rhs = 0
        logical :: device_resident = .false.
    end type rbf_krylov_workspace_t

    type :: rbf_block_preconditioner_t
        real(dp), allocatable :: factors(:, :, :)
        integer :: block_size = 0
        integer :: n_blocks = 0
    end type rbf_block_preconditioner_t

    type :: rbf_nystrom_preconditioner_t
        real(dp), allocatable :: features(:, :)
        real(dp), allocatable :: factor(:, :)
        integer :: rank = 0
        real(dp) :: regularization = 0.0_dp
    end type rbf_nystrom_preconditioner_t

    type, extends(linear_operator_t), public :: rbf_operator_t
        ! Neighbor index is contiguous for the matrix-free inner reduction.
        real(dp), allocatable :: points(:, :)
        real(dp) :: variance = 1.0_dp
        real(dp) :: lengthscale = 1.0_dp
        real(dp) :: diagonal_shift = 0.0_dp
        integer :: tile_size = DEFAULT_TILE_SIZE
        logical :: points_device_resident = .false.
        type(rbf_krylov_workspace_t) :: krylov_workspace
        type(rbf_block_preconditioner_t) :: block_preconditioner
        type(rbf_nystrom_preconditioner_t) :: nystrom_preconditioner
    contains
        procedure, public :: initialize => rbf_operator_initialize
        procedure, public :: enter_data => rbf_operator_enter_data
        procedure, public :: exit_data => rbf_operator_exit_data
        procedure, public :: matvec => rbf_operator_matvec
        procedure, public :: matmat => rbf_operator_matmat
        procedure, public :: diagonal => rbf_operator_diagonal
        procedure, public :: sample_count => rbf_operator_sample_count
        procedure, public :: solve_cg => rbf_operator_solve_cg
        procedure, public :: solve_cg_multi => rbf_operator_solve_cg_multi
        procedure, public :: solve_cg_multi_block => &
            rbf_operator_solve_cg_multi_block
        procedure, public :: solve_cg_multi_nystrom => &
            rbf_operator_solve_cg_multi_nystrom
    end type rbf_operator_t

    type, extends(linear_operator_t), public :: kernel_operator_t
        type(kernel_t) :: kernel
        real(dp), allocatable :: points(:, :)
        real(dp) :: diagonal_shift = 0.0_dp
        integer :: tile_size = DEFAULT_TILE_SIZE
        logical :: points_device_resident = .false.
        integer, allocatable :: program_kind(:)
        real(dp), allocatable :: program_variance(:), program_lengthscale(:)
        logical :: program_device_resident = .false.
    contains
        procedure, public :: initialize => kernel_operator_initialize
        procedure, public :: enter_data => kernel_operator_enter_data
        procedure, public :: exit_data => kernel_operator_exit_data
        procedure, public :: matvec => kernel_operator_matvec
        procedure, public :: matmat => kernel_operator_matmat
        procedure, public :: matvec_device => kernel_operator_matvec_device
        procedure, public :: matmat_device => kernel_operator_matmat_device
        procedure, public :: solve_cg_device => kernel_operator_solve_cg_device
        procedure, public :: solve_cg_multi_device => &
            kernel_operator_solve_cg_multi_device
        procedure, public :: device_supported => kernel_operator_device_supported
        procedure, public :: diagonal => kernel_operator_diagonal
        procedure, public :: sample_count => kernel_operator_sample_count
    end type kernel_operator_t

    public :: rbf_matvec_tiled
    public :: rbf_matmat_tiled

    interface
        function fortml_cuda_rbf_available() bind(C, &
                name="fortml_cuda_rbf_available") result(available)
            import :: c_int
            integer(c_int) :: available
        end function fortml_cuda_rbf_available

        function fortml_cuda_rbf_matvec( &
                points, input, output, n_samples, variance, inverse_scale, &
                diagonal_shift) bind(C, name="fortml_cuda_rbf_matvec") &
                result(status)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: points, input, output
            integer(c_int), value :: n_samples
            real(c_double), value :: variance, inverse_scale, diagonal_shift
            integer(c_int) :: status
        end function fortml_cuda_rbf_matvec

        function fortml_cuda_rbf_matmat( &
                points, input, output, n_samples, n_rhs, variance, &
                inverse_scale, diagonal_shift) bind(C, &
                name="fortml_cuda_rbf_matmat") result(status)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: points, input, output
            integer(c_int), value :: n_samples, n_rhs
            real(c_double), value :: variance, inverse_scale, diagonal_shift
            integer(c_int) :: status
        end function fortml_cuda_rbf_matmat
    end interface

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
        self%points_device_resident = .false.
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

    subroutine rbf_operator_enter_data(self, status, n_rhs)
        class(rbf_operator_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_rhs

        if (.not. allocated(self%points)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF operator: cannot enter data before initialize")
            return
        end if
        if (.not. self%points_device_resident) then
            !$acc enter data copyin(self%points)
            self%points_device_resident = .true.
        end if
        if (present(n_rhs)) then
            call ensure_krylov_workspace(self, n_rhs, status)
            if (status%code /= FORTNUM_OK) return
        end if
        if (allocated(self%krylov_workspace%residual) .and. &
            .not. self%krylov_workspace%device_resident) then
            !$acc enter data create( &
            !$acc& self%krylov_workspace%residual, &
            !$acc& self%krylov_workspace%direction, &
            !$acc& self%krylov_workspace%preconditioned, &
            !$acc& self%krylov_workspace%operator_direction, &
            !$acc& self%krylov_workspace%work)
            self%krylov_workspace%device_resident = .true.
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_operator_enter_data

    subroutine rbf_operator_exit_data(self, status)
        class(rbf_operator_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status

        if (self%points_device_resident) then
            !$acc exit data delete(self%points)
            self%points_device_resident = .false.
        end if
        if (self%krylov_workspace%device_resident) then
            !$acc exit data delete( &
            !$acc& self%krylov_workspace%residual, &
            !$acc& self%krylov_workspace%direction, &
            !$acc& self%krylov_workspace%preconditioned, &
            !$acc& self%krylov_workspace%operator_direction, &
            !$acc& self%krylov_workspace%work)
            self%krylov_workspace%device_resident = .false.
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_operator_exit_data

    subroutine ensure_krylov_workspace(self, n_rhs, status)
        class(rbf_operator_t), intent(inout) :: self
        integer, intent(in) :: n_rhs
        type(fortnum_status_t), intent(out) :: status
        integer :: n_samples

        n_samples = self%sample_count()
        if (n_samples < 1 .or. n_rhs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF operator: Krylov workspace shape is invalid")
            return
        end if
        if (allocated(self%krylov_workspace%residual)) then
            if (self%krylov_workspace%n_samples == n_samples .and. &
                self%krylov_workspace%n_rhs == n_rhs) then
                call status_set(status, FORTNUM_OK, "")
                return
            end if
            if (self%krylov_workspace%device_resident) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF operator: exit device data before resizing workspace")
                return
            end if
            deallocate( &
                self%krylov_workspace%residual, &
                self%krylov_workspace%direction, &
                self%krylov_workspace%preconditioned, &
                self%krylov_workspace%operator_direction, &
                self%krylov_workspace%work, &
                self%krylov_workspace%right_hand_side_norm, &
                self%krylov_workspace%target, self%krylov_workspace%rho, &
                self%krylov_workspace%next_rho, &
                self%krylov_workspace%denominator, &
                self%krylov_workspace%active, &
                self%krylov_workspace%candidate)
        end if
        allocate( &
            self%krylov_workspace%residual(n_samples, n_rhs), &
            self%krylov_workspace%direction(n_samples, n_rhs), &
            self%krylov_workspace%preconditioned(n_samples, n_rhs), &
            self%krylov_workspace%operator_direction(n_samples, n_rhs), &
            self%krylov_workspace%work(n_samples, n_rhs), &
            self%krylov_workspace%right_hand_side_norm(n_rhs), &
            self%krylov_workspace%target(n_rhs), &
            self%krylov_workspace%rho(n_rhs), &
            self%krylov_workspace%next_rho(n_rhs), &
            self%krylov_workspace%denominator(n_rhs), &
            self%krylov_workspace%active(n_rhs), &
            self%krylov_workspace%candidate(n_rhs))
        self%krylov_workspace%n_samples = n_samples
        self%krylov_workspace%n_rhs = n_rhs
        call status_set(status, FORTNUM_OK, "")
    end subroutine ensure_krylov_workspace

    subroutine ensure_block_preconditioner(self, block_size, status)
        class(rbf_operator_t), intent(inout) :: self
        integer, intent(in) :: block_size
        type(fortnum_status_t), intent(out) :: status

        integer :: block, column, first, last, local_size, row, k
        integer :: n_blocks, n_samples
        real(dp) :: distance, difference, value

        n_samples = self%sample_count()
        if (n_samples < 1 .or. block_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF operator: block preconditioner shape is invalid")
            return
        end if
        n_blocks = (n_samples + block_size - 1)/block_size
        if (allocated(self%block_preconditioner%factors) .and. &
            self%block_preconditioner%block_size == block_size .and. &
            self%block_preconditioner%n_blocks == n_blocks) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (allocated(self%block_preconditioner%factors)) then
            deallocate(self%block_preconditioner%factors)
        end if
        allocate(self%block_preconditioner%factors( &
            block_size, block_size, n_blocks))
        self%block_preconditioner%factors = 0.0_dp
        self%block_preconditioner%block_size = block_size
        self%block_preconditioner%n_blocks = n_blocks

        do block = 1, n_blocks
            first = (block - 1)*block_size + 1
            last = min(n_samples, first + block_size - 1)
            local_size = last - first + 1
            do row = 1, local_size
                do column = 1, row
                    distance = 0.0_dp
                    do k = 1, size(self%points, 2)
                        difference = self%points(first + row - 1, k) - &
                            self%points(first + column - 1, k)
                        distance = distance + difference*difference
                    end do
                    value = self%variance*exp( &
                        -0.5_dp*distance/(self%lengthscale*self%lengthscale))
                    if (row == column) value = value + self%diagonal_shift
                    do k = 1, column - 1
                        value = value - self%block_preconditioner%factors( &
                            row, k, block)*self%block_preconditioner%factors( &
                            column, k, block)
                    end do
                    if (row == column) then
                        if (value <= 0.0_dp .or. .not. ieee_is_finite(value)) then
                            deallocate(self%block_preconditioner%factors)
                            self%block_preconditioner%block_size = 0
                            self%block_preconditioner%n_blocks = 0
                            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                                "RBF operator: block preconditioner is not SPD")
                            return
                        end if
                        self%block_preconditioner%factors(row, column, block) = &
                            sqrt(value)
                    else
                        self%block_preconditioner%factors(row, column, block) = &
                            value/self%block_preconditioner%factors(column, column, block)
                    end if
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ensure_block_preconditioner

    subroutine ensure_nystrom_preconditioner(self, requested_rank, status)
        class(rbf_operator_t), intent(inout) :: self
        integer, intent(in) :: requested_rank
        type(fortnum_status_t), intent(out) :: status

        integer, allocatable :: landmarks(:)
        real(dp), allocatable :: landmark_factor(:, :)
        integer :: i, j, k, rank, n_samples
        real(dp) :: difference, distance, jitter, regularization, value

        n_samples = self%sample_count()
        if (n_samples < 1 .or. requested_rank < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF operator: Nystrom rank is invalid")
            return
        end if
        rank = min(requested_rank, n_samples)
        if (allocated(self%nystrom_preconditioner%features) .and. &
            self%nystrom_preconditioner%rank == rank) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (allocated(self%nystrom_preconditioner%features)) then
            deallocate(self%nystrom_preconditioner%features)
            deallocate(self%nystrom_preconditioner%factor)
        end if
        allocate( &
            self%nystrom_preconditioner%features(n_samples, rank), &
            self%nystrom_preconditioner%factor(rank, rank), &
            landmarks(rank), landmark_factor(rank, rank))
        self%nystrom_preconditioner%rank = rank

        do j = 1, rank
            landmarks(j) = 1 + (j - 1)*n_samples/rank
        end do
        jitter = max(1.0e-10_dp*self%variance, 1.0e-12_dp)
        do i = 1, rank
            do j = 1, i
                distance = 0.0_dp
                do k = 1, size(self%points, 2)
                    difference = self%points(landmarks(i), k) - &
                        self%points(landmarks(j), k)
                    distance = distance + difference*difference
                end do
                value = self%variance*exp( &
                    -0.5_dp*distance/(self%lengthscale*self%lengthscale))
                if (i == j) value = value + jitter
                landmark_factor(i, j) = value
                landmark_factor(j, i) = value
            end do
        end do
        do i = 1, rank
            do j = 1, i
                value = landmark_factor(i, j)
                do k = 1, j - 1
                    value = value - landmark_factor(i, k)* &
                        landmark_factor(j, k)
                end do
                if (i == j) then
                    if (value <= 0.0_dp .or. .not. ieee_is_finite(value)) then
                        call clear_nystrom_preconditioner(self)
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "RBF operator: Nystrom landmark matrix is not SPD")
                        return
                    end if
                    landmark_factor(i, j) = sqrt(value)
                else
                    landmark_factor(i, j) = value/landmark_factor(j, j)
                end if
            end do
            do j = i + 1, rank
                landmark_factor(i, j) = 0.0_dp
            end do
        end do

        do i = 1, n_samples
            do j = 1, rank
                distance = 0.0_dp
                do k = 1, size(self%points, 2)
                    difference = self%points(i, k) - &
                        self%points(landmarks(j), k)
                    distance = distance + difference*difference
                end do
                self%nystrom_preconditioner%features(i, j) = self%variance*exp( &
                    -0.5_dp*distance/(self%lengthscale*self%lengthscale))
            end do
        end do
        do i = 1, n_samples
            do j = 1, rank
                value = self%nystrom_preconditioner%features(i, j)
                do k = 1, j - 1
                    value = value - landmark_factor(j, k)* &
                        self%nystrom_preconditioner%features(i, k)
                end do
                self%nystrom_preconditioner%features(i, j) = value/ &
                    landmark_factor(j, j)
            end do
        end do

        regularization = max(self%diagonal_shift, jitter)
        self%nystrom_preconditioner%regularization = regularization
        self%nystrom_preconditioner%factor = 0.0_dp
        do i = 1, rank
            do j = i, rank
                value = dot_product( &
                    self%nystrom_preconditioner%features(:, i), &
                    self%nystrom_preconditioner%features(:, j))
                self%nystrom_preconditioner%factor(i, j) = value
                self%nystrom_preconditioner%factor(j, i) = value
            end do
            self%nystrom_preconditioner%factor(i, i) = &
                self%nystrom_preconditioner%factor(i, i) + regularization
        end do
        do i = 1, rank
            do j = 1, i
                value = self%nystrom_preconditioner%factor(i, j)
                do k = 1, j - 1
                    value = value - self%nystrom_preconditioner%factor(i, k)* &
                        self%nystrom_preconditioner%factor(j, k)
                end do
                if (i == j) then
                    if (value <= 0.0_dp .or. .not. ieee_is_finite(value)) then
                        call clear_nystrom_preconditioner(self)
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "RBF operator: Nystrom small matrix is not SPD")
                        return
                    end if
                    self%nystrom_preconditioner%factor(i, j) = sqrt(value)
                else
                    self%nystrom_preconditioner%factor(i, j) = value/ &
                        self%nystrom_preconditioner%factor(j, j)
                end if
            end do
            do j = i + 1, rank
                self%nystrom_preconditioner%factor(i, j) = 0.0_dp
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ensure_nystrom_preconditioner

    subroutine clear_nystrom_preconditioner(self)
        class(rbf_operator_t), intent(inout) :: self

        if (allocated(self%nystrom_preconditioner%features)) then
            deallocate(self%nystrom_preconditioner%features)
        end if
        if (allocated(self%nystrom_preconditioner%factor)) then
            deallocate(self%nystrom_preconditioner%factor)
        end if
        self%nystrom_preconditioner%rank = 0
        self%nystrom_preconditioner%regularization = 0.0_dp
    end subroutine clear_nystrom_preconditioner

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

    subroutine rbf_operator_solve_cg( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            info, iterations, residual_norm, use_diagonal_preconditioner)
        class(rbf_operator_t), intent(in) :: self
        real(dp), intent(in) :: right_hand_side(:)
        real(dp), intent(inout) :: solution(:)
        real(dp), intent(in) :: tolerance
        integer, intent(in) :: max_iterations
        integer, intent(out) :: info, iterations
        real(dp), intent(out) :: residual_norm
        logical, intent(in), optional :: use_diagonal_preconditioner

        logical :: use_preconditioner, done
        real(dp), allocatable :: residual(:), direction(:), preconditioned(:)
        real(dp), allocatable :: operator_direction(:), work(:), diagonal_values(:)
        real(dp) :: right_hand_side_norm, target, rho, next_rho
        real(dp) :: denominator, step, beta
        integer :: n_samples

        info = KRYLOV_INVALID_ARGUMENT
        iterations = 0
        residual_norm = huge(1.0_dp)
        n_samples = self%sample_count()
        if (n_samples < 1) return
        if (size(right_hand_side) /= n_samples) return
        if (size(solution) /= n_samples) return
        if (tolerance <= 0.0_dp .or. max_iterations < 1) return

        use_preconditioner = .true.
        if (present(use_diagonal_preconditioner)) then
            use_preconditioner = use_diagonal_preconditioner
        end if
        allocate( &
            residual(n_samples), direction(n_samples), &
            preconditioned(n_samples), operator_direction(n_samples), &
            work(n_samples), diagonal_values(n_samples))
        diagonal_values = 1.0_dp
        if (use_preconditioner) then
            diagonal_values = self%diagonal()
            if (size(diagonal_values) /= n_samples) return
            if (any(diagonal_values <= 0.0_dp)) return
        end if

        right_hand_side_norm = sqrt(sum(right_hand_side*right_hand_side))
        target = tolerance*max(right_hand_side_norm, 1.0_dp)
        if (right_hand_side_norm == 0.0_dp) then
            solution = 0.0_dp
            residual_norm = 0.0_dp
            info = KRYLOV_OK
            return
        end if

        done = .false.
        info = KRYLOV_MAX_ITERATIONS
        !$acc data copyin(self%points, right_hand_side, diagonal_values) &
        !$acc& copy(solution) create(residual, direction, preconditioned, &
        !$acc& operator_direction, work)
        call self%matvec(solution, work)
        call subtract_vectors(right_hand_side, work, residual)
        residual_norm = acc_norm2(residual)
        if (residual_norm <= target) then
            info = KRYLOV_OK
            done = .true.
        else
            call apply_diagonal_preconditioner(residual, preconditioned)
            rho = acc_dot(residual, preconditioned)
            if (rho <= 0.0_dp .or. .not. ieee_is_finite(rho)) then
                info = KRYLOV_BREAKDOWN
                done = .true.
            else
                call copy_vector(direction, preconditioned)
                do while (iterations < max_iterations .and. .not. done)
                    call self%matvec(direction, operator_direction)
                    denominator = acc_dot(direction, operator_direction)
                    if (denominator <= 0.0_dp .or. &
                        .not. ieee_is_finite(denominator)) then
                        info = KRYLOV_BREAKDOWN
                        done = .true.
                    else
                        step = rho/denominator
                        call update_solution(solution, step, direction)
                        call subtract_scaled( &
                            residual, step, operator_direction)
                        iterations = iterations + 1
                        residual_norm = acc_norm2(residual)
                        if (residual_norm <= target) then
                            call self%matvec(solution, work)
                            call subtract_vectors(right_hand_side, work, residual)
                            residual_norm = acc_norm2(residual)
                            if (residual_norm <= target) then
                                info = KRYLOV_OK
                                done = .true.
                            end if
                        end if
                        if (.not. done) then
                            call apply_diagonal_preconditioner( &
                                residual, preconditioned)
                            next_rho = acc_dot(residual, preconditioned)
                            if (next_rho <= 0.0_dp .or. &
                                .not. ieee_is_finite(next_rho)) then
                                info = KRYLOV_BREAKDOWN
                                done = .true.
                            else
                                beta = next_rho/rho
                                call combine_direction( &
                                    direction, preconditioned, beta)
                                rho = next_rho
                            end if
                        end if
                    end if
                end do
            end if
        end if
        if (.not. done) then
            call self%matvec(solution, work)
            call subtract_vectors(right_hand_side, work, residual)
            residual_norm = acc_norm2(residual)
            if (residual_norm <= target) info = KRYLOV_OK
        end if
        !$acc end data

    contains

        function acc_dot(left, right) result(value)
            real(dp), intent(in) :: left(:), right(:)
            real(dp) :: value
            integer :: i

            value = 0.0_dp
            !$acc parallel loop reduction(+:value)
            !$omp parallel do reduction(+:value)
            do i = 1, size(left)
                value = value + left(i)*right(i)
            end do
        end function acc_dot

        function acc_norm2(vector) result(value)
            real(dp), intent(in) :: vector(:)
            real(dp) :: value

            value = sqrt(acc_dot(vector, vector))
        end function acc_norm2

        subroutine subtract_vectors(left, right, result)
            real(dp), intent(in) :: left(:), right(:)
            real(dp), intent(out) :: result(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(result)
                result(i) = left(i) - right(i)
            end do
        end subroutine subtract_vectors

        subroutine copy_vector(target, source)
            real(dp), intent(out) :: target(:)
            real(dp), intent(in) :: source(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(target)
                target(i) = source(i)
            end do
        end subroutine copy_vector

        subroutine apply_diagonal_preconditioner(input, output)
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(output)
                output(i) = input(i)/diagonal_values(i)
            end do
        end subroutine apply_diagonal_preconditioner

        subroutine update_solution(current, scale, vector)
            real(dp), intent(inout) :: current(:)
            real(dp), intent(in) :: scale, vector(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(current)
                current(i) = current(i) + scale*vector(i)
            end do
        end subroutine update_solution

        subroutine subtract_scaled(current, scale, vector)
            real(dp), intent(inout) :: current(:)
            real(dp), intent(in) :: scale, vector(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(current)
                current(i) = current(i) - scale*vector(i)
            end do
        end subroutine subtract_scaled

        subroutine combine_direction(current, preconditioned, scale)
            real(dp), intent(inout) :: current(:)
            real(dp), intent(in) :: preconditioned(:), scale
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(current)
                current(i) = preconditioned(i) + scale*current(i)
            end do
        end subroutine combine_direction

    end subroutine rbf_operator_solve_cg

    subroutine rbf_operator_solve_cg_multi( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            info, iterations, residual_norm, use_diagonal_preconditioner, &
            preconditioner_block_size, preconditioner_nystrom_rank)
        class(rbf_operator_t), intent(inout) :: self
        real(dp), intent(in) :: right_hand_side(:, :)
        real(dp), intent(inout) :: solution(:, :)
        real(dp), intent(in) :: tolerance
        integer, intent(in) :: max_iterations
        integer, intent(out) :: info(:), iterations(:)
        real(dp), intent(out) :: residual_norm(:)
        logical, intent(in), optional :: use_diagonal_preconditioner
        integer, intent(in), optional :: preconditioner_block_size
        integer, intent(in), optional :: preconditioner_nystrom_rank

        logical :: use_preconditioner
        real(dp), allocatable :: diagonal_values(:)
        type(fortnum_status_t) :: status
        real(dp) :: step, beta
        integer :: n_samples, n_rhs, column, block_size, nystrom_rank
        logical :: use_block_preconditioner, use_nystrom_preconditioner

        info = KRYLOV_INVALID_ARGUMENT
        iterations = 0
        residual_norm = huge(1.0_dp)
        n_samples = self%sample_count()
        n_rhs = size(right_hand_side, 2)
        if (n_samples < 1 .or. n_rhs < 1 .or. &
            size(right_hand_side, 1) /= n_samples .or. &
            any(shape(solution) /= [n_samples, n_rhs]) .or. &
            size(info) /= n_rhs .or. size(iterations) /= n_rhs .or. &
            size(residual_norm) /= n_rhs .or. tolerance <= 0.0_dp .or. &
            max_iterations < 1) return

        call ensure_krylov_workspace(self, n_rhs, status)
        if (status%code /= FORTNUM_OK) return

        use_preconditioner = .true.
        if (present(use_diagonal_preconditioner)) then
            use_preconditioner = use_diagonal_preconditioner
        end if
        use_block_preconditioner = .false.
        block_size = 0
        if (present(preconditioner_block_size)) then
            block_size = preconditioner_block_size
            use_block_preconditioner = block_size > 0
            if (block_size < 0 .or. (use_block_preconditioner .and. &
                .not. use_preconditioner)) return
        end if
        use_nystrom_preconditioner = .false.
        nystrom_rank = 0
        if (present(preconditioner_nystrom_rank)) then
            nystrom_rank = preconditioner_nystrom_rank
            use_nystrom_preconditioner = nystrom_rank > 0
            if (nystrom_rank < 0 .or. (use_nystrom_preconditioner .and. &
                .not. use_preconditioner) .or. (use_nystrom_preconditioner .and. &
                use_block_preconditioner)) return
        end if
        allocate(diagonal_values(n_samples))
        diagonal_values = 1.0_dp
        if (use_nystrom_preconditioner) then
            call ensure_nystrom_preconditioner(self, nystrom_rank, status)
            if (status%code /= FORTNUM_OK) return
        else if (use_block_preconditioner) then
            call ensure_block_preconditioner(self, block_size, status)
            if (status%code /= FORTNUM_OK) return
        else if (use_preconditioner) then
            diagonal_values = self%diagonal()
            if (size(diagonal_values) /= n_samples) return
            if (any(diagonal_values <= 0.0_dp)) return
        end if
        do column = 1, n_rhs
            self%krylov_workspace%right_hand_side_norm(column) = sqrt(sum( &
                right_hand_side(:, column)*right_hand_side(:, column)))
            self%krylov_workspace%target(column) = tolerance*max( &
                self%krylov_workspace%right_hand_side_norm(column), 1.0_dp)
        end do
        info = KRYLOV_MAX_ITERATIONS
        iterations = 0
        residual_norm = huge(1.0_dp)
        self%krylov_workspace%active = .false.
        self%krylov_workspace%candidate = .false.

        if (use_nystrom_preconditioner) then
            !$acc enter data copyin( &
            !$acc& self%nystrom_preconditioner%features, &
            !$acc& self%nystrom_preconditioner%factor)
        else if (use_block_preconditioner) then
            !$acc enter data copyin(self%block_preconditioner%factors)
        end if
        !$acc data copyin(self%points, right_hand_side, diagonal_values) &
        !$acc& copy(solution) create( &
        !$acc& self%krylov_workspace%residual, &
        !$acc& self%krylov_workspace%direction, &
        !$acc& self%krylov_workspace%preconditioned, &
        !$acc& self%krylov_workspace%operator_direction, &
        !$acc& self%krylov_workspace%work)
        call self%matmat( &
            solution, self%krylov_workspace%work)
        call subtract_matrix( &
            right_hand_side, self%krylov_workspace%work, &
            self%krylov_workspace%residual)
        do column = 1, n_rhs
            residual_norm(column) = acc_norm2_column( &
                self%krylov_workspace%residual, column)
            if (residual_norm(column) <= self%krylov_workspace%target(column)) then
                info(column) = KRYLOV_OK
            end if
        end do
        if (use_nystrom_preconditioner) then
            call apply_nystrom_preconditioner( &
                self%krylov_workspace%residual, &
                self%krylov_workspace%preconditioned, &
                self%krylov_workspace%work, &
                self%nystrom_preconditioner%features, &
                self%nystrom_preconditioner%factor, &
                self%nystrom_preconditioner%rank, &
                self%nystrom_preconditioner%regularization)
        end if
        do column = 1, n_rhs
            if (info(column) == KRYLOV_MAX_ITERATIONS) then
                if (.not. use_nystrom_preconditioner) then
                    call apply_preconditioner_column( &
                        self%krylov_workspace%residual, &
                        self%krylov_workspace%preconditioned, column)
                end if
                self%krylov_workspace%rho(column) = acc_dot_column( &
                    self%krylov_workspace%residual, &
                    self%krylov_workspace%preconditioned, column)
                if (self%krylov_workspace%rho(column) <= 0.0_dp .or. &
                    .not. ieee_is_finite(self%krylov_workspace%rho(column))) then
                    info(column) = KRYLOV_BREAKDOWN
                else
                    call copy_column( &
                        self%krylov_workspace%direction, &
                        self%krylov_workspace%preconditioned, column)
                    self%krylov_workspace%active(column) = .true.
                end if
            end if
        end do

        do while (any(self%krylov_workspace%active))
            call self%matmat( &
                self%krylov_workspace%direction, &
                self%krylov_workspace%operator_direction)
            self%krylov_workspace%candidate = .false.
            do column = 1, n_rhs
                if (self%krylov_workspace%active(column)) then
                    self%krylov_workspace%denominator(column) = &
                        acc_dot_column( &
                        self%krylov_workspace%direction, &
                        self%krylov_workspace%operator_direction, column)
                    if (self%krylov_workspace%denominator(column) <= 0.0_dp .or. &
                        .not. ieee_is_finite( &
                        self%krylov_workspace%denominator(column))) then
                        info(column) = KRYLOV_BREAKDOWN
                        self%krylov_workspace%active(column) = .false.
                    else
                        step = self%krylov_workspace%rho(column)/ &
                            self%krylov_workspace%denominator(column)
                        call update_column( &
                            solution, step, self%krylov_workspace%direction, &
                            column)
                        call subtract_scaled_column( &
                            self%krylov_workspace%residual, step, &
                            self%krylov_workspace%operator_direction, column)
                        iterations(column) = iterations(column) + 1
                        residual_norm(column) = &
                            acc_norm2_column( &
                            self%krylov_workspace%residual, column)
                        if (residual_norm(column) <= &
                            self%krylov_workspace%target(column) .or. &
                            iterations(column) >= max_iterations) then
                            self%krylov_workspace%candidate(column) = .true.
                        end if
                    end if
                end if
            end do

            if (any(self%krylov_workspace%candidate)) then
                call self%matmat( &
                    solution, self%krylov_workspace%work)
                do column = 1, n_rhs
                    if (self%krylov_workspace%candidate(column)) then
                        call subtract_column( &
                            right_hand_side, self%krylov_workspace%work, &
                            self%krylov_workspace%residual, column)
                        residual_norm(column) = &
                            acc_norm2_column( &
                            self%krylov_workspace%residual, column)
                        if (residual_norm(column) <= &
                            self%krylov_workspace%target(column)) then
                            info(column) = KRYLOV_OK
                            self%krylov_workspace%active(column) = .false.
                            call zero_column( &
                                self%krylov_workspace%direction, column)
                        else if (iterations(column) >= max_iterations) then
                            info(column) = KRYLOV_MAX_ITERATIONS
                            self%krylov_workspace%active(column) = .false.
                            call zero_column( &
                                self%krylov_workspace%direction, column)
                        else
                            self%krylov_workspace%candidate(column) = .false.
                        end if
                    end if
                end do
            end if

            if (use_nystrom_preconditioner .and. &
                any(self%krylov_workspace%active)) then
                call apply_nystrom_preconditioner( &
                    self%krylov_workspace%residual, &
                    self%krylov_workspace%preconditioned, &
                    self%krylov_workspace%work, &
                    self%nystrom_preconditioner%features, &
                    self%nystrom_preconditioner%factor, &
                    self%nystrom_preconditioner%rank, &
                    self%nystrom_preconditioner%regularization)
            end if
            do column = 1, n_rhs
                if (self%krylov_workspace%active(column)) then
                    if (.not. use_nystrom_preconditioner) then
                        call apply_preconditioner_column( &
                            self%krylov_workspace%residual, &
                            self%krylov_workspace%preconditioned, column)
                    end if
                    self%krylov_workspace%next_rho(column) = &
                        acc_dot_column( &
                        self%krylov_workspace%residual, &
                        self%krylov_workspace%preconditioned, column)
                    if (self%krylov_workspace%next_rho(column) <= 0.0_dp .or. &
                        .not. ieee_is_finite( &
                        self%krylov_workspace%next_rho(column))) then
                        info(column) = KRYLOV_BREAKDOWN
                        self%krylov_workspace%active(column) = .false.
                    else
                        beta = self%krylov_workspace%next_rho(column)/ &
                            self%krylov_workspace%rho(column)
                        call combine_column( &
                            self%krylov_workspace%direction, &
                            self%krylov_workspace%preconditioned, beta, column)
                        self%krylov_workspace%rho(column) = &
                            self%krylov_workspace%next_rho(column)
                    end if
                end if
            end do
        end do

        call self%matmat(solution, self%krylov_workspace%work)
        do column = 1, n_rhs
            call subtract_column( &
                right_hand_side, self%krylov_workspace%work, &
                self%krylov_workspace%residual, column)
            residual_norm(column) = acc_norm2_column( &
                self%krylov_workspace%residual, column)
            if (residual_norm(column) <= self%krylov_workspace%target(column)) then
                info(column) = KRYLOV_OK
            end if
        end do
        !$acc end data
        if (use_nystrom_preconditioner) then
            !$acc exit data delete( &
            !$acc& self%nystrom_preconditioner%features, &
            !$acc& self%nystrom_preconditioner%factor)
        else if (use_block_preconditioner) then
            !$acc exit data delete(self%block_preconditioner%factors)
        end if

    contains

        function acc_dot_column(left, right, column) result(value)
            real(dp), intent(in) :: left(:, :), right(:, :)
            integer, intent(in) :: column
            real(dp) :: value
            integer :: i

            value = 0.0_dp
            !$acc parallel loop reduction(+:value)
            !$omp parallel do reduction(+:value)
            do i = 1, size(left, 1)
                value = value + left(i, column)*right(i, column)
            end do
        end function acc_dot_column

        function acc_norm2_column(vector, column) result(value)
            real(dp), intent(in) :: vector(:, :)
            integer, intent(in) :: column
            real(dp) :: value

            value = sqrt(acc_dot_column(vector, vector, column))
        end function acc_norm2_column

        subroutine subtract_matrix(left, right, result)
            real(dp), intent(in) :: left(:, :), right(:, :)
            real(dp), intent(out) :: result(:, :)
            integer :: i, j

            do j = 1, size(result, 2)
                !$acc parallel loop
                !$omp parallel do
                do i = 1, size(result, 1)
                    result(i, j) = left(i, j) - right(i, j)
                end do
            end do
        end subroutine subtract_matrix

        subroutine subtract_column(left, right, result, column)
            real(dp), intent(in) :: left(:, :), right(:, :)
            real(dp), intent(inout) :: result(:, :)
            integer, intent(in) :: column
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(result, 1)
                result(i, column) = left(i, column) - right(i, column)
            end do
        end subroutine subtract_column

        subroutine apply_preconditioner_column(input, output, column)
            real(dp), intent(in) :: input(:, :)
            real(dp), intent(inout) :: output(:, :)
            integer, intent(in) :: column
            integer :: i

            if (use_block_preconditioner) then
                call apply_block_preconditioner( &
                    input, output, column, self%block_preconditioner%factors, &
                    self%block_preconditioner%n_blocks)
            else
                !$acc parallel loop
                !$omp parallel do
                do i = 1, size(output, 1)
                    output(i, column) = input(i, column)/diagonal_values(i)
                end do
            end if
        end subroutine apply_preconditioner_column

        subroutine apply_block_preconditioner( &
                input, output, column, factors, n_blocks)
            real(dp), intent(in) :: input(:, :)
            real(dp), intent(inout) :: output(:, :)
            integer, intent(in) :: column
            real(dp), intent(in) :: factors(:, :, :)
            integer, intent(in) :: n_blocks
            integer :: block, first, last, local_size, row, k
            real(dp) :: value

            !$acc parallel loop gang private(first, last, local_size, row, k, value)
            !$omp parallel do schedule(static) private(first, last, local_size, row, k, value)
            do block = 1, n_blocks
                first = (block - 1)*block_size + 1
                last = min(size(input, 1), first + block_size - 1)
                local_size = last - first + 1
                !$acc loop seq
                do row = 1, local_size
                    value = input(first + row - 1, column)
                    do k = 1, row - 1
                        value = value - factors( &
                            row, k, block)*output(first + k - 1, column)
                    end do
                    output(first + row - 1, column) = value/ &
                        factors(row, row, block)
                end do
                !$acc loop seq
                do row = local_size, 1, -1
                    value = output(first + row - 1, column)
                    do k = row + 1, local_size
                        value = value - factors( &
                            k, row, block)*output(first + k - 1, column)
                    end do
                    output(first + row - 1, column) = value/ &
                        factors(row, row, block)
                end do
            end do
        end subroutine apply_block_preconditioner

        subroutine apply_nystrom_preconditioner( &
                input, output, scratch, features, factor, rank, regularization)
            real(dp), intent(in) :: input(:, :)
            real(dp), intent(out) :: output(:, :)
            real(dp), intent(inout) :: scratch(:, :)
            real(dp), intent(in) :: features(:, :), factor(:, :)
            integer, intent(in) :: rank
            real(dp), intent(in) :: regularization
            integer :: column, i, j, k
            real(dp) :: value

            !$acc parallel loop gang collapse(2) private(value, i)
            !$omp parallel do collapse(2) private(value, i) schedule(static)
            do column = 1, size(input, 2)
                do j = 1, rank
                    value = 0.0_dp
                    !$acc loop vector reduction(+:value)
                    !$omp simd reduction(+:value)
                    do i = 1, size(input, 1)
                        value = value + features(i, j)*input(i, column)
                    end do
                    scratch(j, column) = value
                end do
            end do
            !$acc parallel loop gang private(i, j, k, value)
            !$omp parallel do private(i, j, k, value) schedule(static)
            do column = 1, size(input, 2)
                !$acc loop seq
                do i = 1, rank
                    value = scratch(i, column)
                    do k = 1, i - 1
                        value = value - factor(i, k)*scratch(k, column)
                    end do
                    scratch(i, column) = value/factor(i, i)
                end do
                !$acc loop seq
                do i = rank, 1, -1
                    value = scratch(i, column)
                    do k = i + 1, rank
                        value = value - factor(k, i)*scratch(k, column)
                    end do
                    scratch(i, column) = value/factor(i, i)
                end do
            end do
            !$acc parallel loop gang collapse(2) private(value, j)
            !$omp parallel do collapse(2) private(value, j) schedule(static)
            do column = 1, size(input, 2)
                do i = 1, size(input, 1)
                    value = input(i, column)
                    do j = 1, rank
                        value = value - features(i, j)*scratch(j, column)
                    end do
                    output(i, column) = value/regularization
                end do
            end do
        end subroutine apply_nystrom_preconditioner

        subroutine copy_column(target, source, column)
            real(dp), intent(out) :: target(:, :)
            real(dp), intent(in) :: source(:, :)
            integer, intent(in) :: column
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(target, 1)
                target(i, column) = source(i, column)
            end do
        end subroutine copy_column

        subroutine zero_column(target, column)
            real(dp), intent(inout) :: target(:, :)
            integer, intent(in) :: column
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(target, 1)
                target(i, column) = 0.0_dp
            end do
        end subroutine zero_column

        subroutine update_column(target, scale, vector, column)
            real(dp), intent(inout) :: target(:, :)
            real(dp), intent(in) :: scale, vector(:, :)
            integer, intent(in) :: column
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(target, 1)
                target(i, column) = target(i, column) + &
                    scale*vector(i, column)
            end do
        end subroutine update_column

        subroutine subtract_scaled_column(target, scale, vector, column)
            real(dp), intent(inout) :: target(:, :)
            real(dp), intent(in) :: scale, vector(:, :)
            integer, intent(in) :: column
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(target, 1)
                target(i, column) = target(i, column) - &
                    scale*vector(i, column)
            end do
        end subroutine subtract_scaled_column

        subroutine combine_column(target, source, scale, column)
            real(dp), intent(inout) :: target(:, :)
            real(dp), intent(in) :: source(:, :)
            real(dp), intent(in) :: scale
            integer, intent(in) :: column
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(target, 1)
                target(i, column) = source(i, column) + &
                    scale*target(i, column)
            end do
        end subroutine combine_column

    end subroutine rbf_operator_solve_cg_multi

    subroutine rbf_operator_solve_cg_multi_block( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            block_size, info, iterations, residual_norm)
        class(rbf_operator_t), intent(inout) :: self
        real(dp), intent(in) :: right_hand_side(:, :)
        real(dp), intent(inout) :: solution(:, :)
        real(dp), intent(in) :: tolerance
        integer, intent(in) :: max_iterations, block_size
        integer, intent(out) :: info(:), iterations(:)
        real(dp), intent(out) :: residual_norm(:)

        call rbf_operator_solve_cg_multi( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            info, iterations, residual_norm, &
            preconditioner_block_size=block_size)
    end subroutine rbf_operator_solve_cg_multi_block

    subroutine rbf_operator_solve_cg_multi_nystrom( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            rank, info, iterations, residual_norm)
        class(rbf_operator_t), intent(inout) :: self
        real(dp), intent(in) :: right_hand_side(:, :)
        real(dp), intent(inout) :: solution(:, :)
        real(dp), intent(in) :: tolerance
        integer, intent(in) :: max_iterations, rank
        integer, intent(out) :: info(:), iterations(:)
        real(dp), intent(out) :: residual_norm(:)

        call rbf_operator_solve_cg_multi( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            info, iterations, residual_norm, &
            preconditioner_nystrom_rank=rank)
    end subroutine rbf_operator_solve_cg_multi_nystrom

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
        self%points_device_resident = .false.
        self%program_device_resident = .false.
        if (present(tile_size)) self%tile_size = tile_size
        if (self%tile_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: tile size must be positive")
            return
        end if
        allocate(self%points(size(sample_points, 1), size(sample_points, 2)))
        self%points = sample_points
        call initialize_kernel_program(self, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_operator_initialize

    subroutine initialize_kernel_program(self, status)
        class(kernel_operator_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer :: program_size, cursor

        program_size = kernel_program_size(self%kernel)
        if (program_size < 1 .or. program_size > MAX_KERNEL_PROGRAM) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: kernel expression is too large or invalid")
            return
        end if
        allocate( &
            self%program_kind(program_size), &
            self%program_variance(program_size), &
            self%program_lengthscale(program_size))
        cursor = 0
        call append_kernel_program( &
            self%kernel, self%program_kind, self%program_variance, &
            self%program_lengthscale, cursor, status)
        if (status%code /= FORTNUM_OK) return
        if (cursor /= program_size) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: kernel program construction failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine initialize_kernel_program

    recursive integer function kernel_program_size(kernel) result(count)
        type(kernel_t), intent(in) :: kernel

        count = 0
        select case (kernel%kind)
        case (KERNEL_SUM, KERNEL_PRODUCT)
            if (.not. associated(kernel%left)) return
            if (.not. associated(kernel%right)) return
            count = kernel_program_size(kernel%left) + &
                kernel_program_size(kernel%right) + 1
        case (KERNEL_RBF, KERNEL_MATERN12, KERNEL_MATERN32, KERNEL_MATERN52, &
                KERNEL_LINEAR, KERNEL_CONSTANT, KERNEL_WHITE_NOISE)
            count = 1
        end select
    end function kernel_program_size

    recursive subroutine append_kernel_program( &
            kernel, program_kind, program_variance, program_lengthscale, &
            cursor, status)
        type(kernel_t), intent(in) :: kernel
        integer, intent(inout) :: program_kind(:)
        real(dp), intent(inout) :: program_variance(:), program_lengthscale(:)
        integer, intent(inout) :: cursor
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_OK, "")

        select case (kernel%kind)
        case (KERNEL_SUM, KERNEL_PRODUCT)
            if (.not. associated(kernel%left)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel operator: sum/product left child is missing")
                return
            end if
            if (.not. associated(kernel%right)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel operator: sum/product right child is missing")
                return
            end if
            call append_kernel_program( &
                kernel%left, program_kind, program_variance, &
                program_lengthscale, cursor, status)
            if (status%code /= FORTNUM_OK) return
            call append_kernel_program( &
                kernel%right, program_kind, program_variance, &
                program_lengthscale, cursor, status)
            if (status%code /= FORTNUM_OK) return
            cursor = cursor + 1
            program_kind(cursor) = kernel%kind
            program_variance(cursor) = 0.0_dp
            program_lengthscale(cursor) = 0.0_dp
        case (KERNEL_RBF, KERNEL_MATERN12, KERNEL_MATERN32, KERNEL_MATERN52)
            if (.not. allocated(kernel%log_parameters)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel operator: leaf parameters are missing")
                return
            end if
            if (size(kernel%log_parameters) /= 2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel operator: leaf parameter shape is invalid")
                return
            end if
            cursor = cursor + 1
            program_kind(cursor) = kernel%kind
            program_variance(cursor) = exp(kernel%log_parameters(1))
            program_lengthscale(cursor) = exp(kernel%log_parameters(2))
        case (KERNEL_LINEAR, KERNEL_CONSTANT, KERNEL_WHITE_NOISE)
            if (.not. allocated(kernel%log_parameters)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel operator: leaf parameters are missing")
                return
            end if
            if (size(kernel%log_parameters) /= 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel operator: leaf parameter shape is invalid")
                return
            end if
            cursor = cursor + 1
            program_kind(cursor) = kernel%kind
            program_variance(cursor) = exp(kernel%log_parameters(1))
            program_lengthscale(cursor) = 1.0_dp
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: unsupported kernel program node")
        end select
    end subroutine append_kernel_program

    subroutine kernel_operator_enter_data(self, status)
        class(kernel_operator_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status

        if (.not. allocated(self%points)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: cannot enter data before initialize")
            return
        end if
        if (.not. self%device_supported()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: kernel cannot be lowered to device code")
            return
        end if
        if (.not. self%points_device_resident) then
            !$acc enter data copyin(self%points)
            self%points_device_resident = .true.
        end if
        if (.not. self%program_device_resident) then
            !$acc enter data copyin( &
            !$acc& self%program_kind, self%program_variance, &
            !$acc& self%program_lengthscale)
            self%program_device_resident = .true.
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_operator_enter_data

    subroutine kernel_operator_exit_data(self, status)
        class(kernel_operator_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status

        if (self%points_device_resident) then
            !$acc exit data delete(self%points)
            self%points_device_resident = .false.
        end if
        if (self%program_device_resident) then
            !$acc exit data delete( &
            !$acc& self%program_kind, self%program_variance, &
            !$acc& self%program_lengthscale)
            self%program_device_resident = .false.
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_operator_exit_data

    logical function kernel_operator_device_supported(self)
        class(kernel_operator_t), intent(in) :: self

        kernel_operator_device_supported = .false.
        if (.not. allocated(self%program_kind)) return
        if (.not. allocated(self%program_variance)) return
        if (.not. allocated(self%program_lengthscale)) return
        if (size(self%program_kind) < 1) return
        if (size(self%program_kind) > MAX_KERNEL_PROGRAM) return
        if (size(self%program_variance) /= size(self%program_kind)) return
        if (size(self%program_lengthscale) /= size(self%program_kind)) return
        kernel_operator_device_supported = .true.
    end function kernel_operator_device_supported

    logical function kernel_operator_simple_rbf(self)
        class(kernel_operator_t), intent(in) :: self

        kernel_operator_simple_rbf = .false.
        if (.not. allocated(self%program_kind)) return
        if (size(self%program_kind) /= 1) return
        if (self%program_kind(1) /= KERNEL_RBF) return
        kernel_operator_simple_rbf = .true.
    end function kernel_operator_simple_rbf

    subroutine kernel_operator_matvec_device(self, input, output, status)
        class(kernel_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: variance, lengthscale

        if (.not. self%device_supported()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: kernel cannot be lowered to device code")
            return
        end if
        if (.not. self%points_device_resident) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: enter_data is required before device products")
            return
        end if
        if (size(input) /= self%sample_count() .or. &
            size(output) /= self%sample_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: device vector shape is invalid")
            return
        end if
        if (kernel_operator_simple_rbf(self)) then
            variance = self%program_variance(1)
            lengthscale = self%program_lengthscale(1)
            call rbf_matvec_tiled( &
                self%points, input, output, variance, lengthscale, &
                self%diagonal_shift, self%tile_size)
        else
            call kernel_program_matvec_tiled( &
                self%points, input, output, self%program_kind, &
                self%program_variance, self%program_lengthscale, &
                self%diagonal_shift, self%tile_size)
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_operator_matvec_device

    subroutine kernel_operator_matmat_device(self, input, output, status)
        class(kernel_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: variance, lengthscale

        if (.not. self%device_supported()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: kernel cannot be lowered to device code")
            return
        end if
        if (.not. self%points_device_resident) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: enter_data is required before device products")
            return
        end if
        if (size(input, 1) /= self%sample_count() .or. &
            size(output, 1) /= self%sample_count() .or. &
            size(output, 2) /= size(input, 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel operator: device matrix shape is invalid")
            return
        end if
        if (kernel_operator_simple_rbf(self)) then
            variance = self%program_variance(1)
            lengthscale = self%program_lengthscale(1)
            call rbf_matmat_tiled( &
                self%points, input, output, variance, lengthscale, &
                self%diagonal_shift, self%tile_size)
        else
            call kernel_program_matmat_tiled( &
                self%points, input, output, self%program_kind, &
                self%program_variance, self%program_lengthscale, &
                self%diagonal_shift, self%tile_size)
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_operator_matmat_device

    subroutine kernel_operator_solve_cg_device( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            info, iterations, residual_norm, use_diagonal_preconditioner)
        class(kernel_operator_t), intent(in) :: self
        real(dp), intent(in) :: right_hand_side(:)
        real(dp), intent(inout) :: solution(:)
        real(dp), intent(in) :: tolerance
        integer, intent(in) :: max_iterations
        integer, intent(out) :: info, iterations
        real(dp), intent(out) :: residual_norm
        logical, intent(in), optional :: use_diagonal_preconditioner

        logical :: use_preconditioner, done
        real(dp), allocatable :: residual(:), direction(:), preconditioned(:)
        real(dp), allocatable :: operator_direction(:), work(:), diagonal_values(:)
        real(dp) :: right_hand_side_norm, target, rho, next_rho
        real(dp) :: denominator, step, beta
        integer :: n_samples

        info = KRYLOV_INVALID_ARGUMENT
        iterations = 0
        residual_norm = huge(1.0_dp)
        n_samples = self%sample_count()
        if (n_samples < 1) return
        if (size(right_hand_side) /= n_samples) return
        if (size(solution) /= n_samples) return
        if (tolerance <= 0.0_dp .or. max_iterations < 1) return
        if (.not. self%device_supported()) return
        if (.not. self%points_device_resident) return
        if (.not. self%program_device_resident) return

        use_preconditioner = .true.
        if (present(use_diagonal_preconditioner)) then
            use_preconditioner = use_diagonal_preconditioner
        end if
        allocate( &
            residual(n_samples), direction(n_samples), &
            preconditioned(n_samples), operator_direction(n_samples), &
            work(n_samples), diagonal_values(n_samples))
        diagonal_values = 1.0_dp
        if (use_preconditioner) then
            diagonal_values = self%diagonal()
            if (size(diagonal_values) /= n_samples) return
            if (any(diagonal_values <= 0.0_dp)) return
        end if

        right_hand_side_norm = sqrt(sum(right_hand_side*right_hand_side))
        target = tolerance*max(right_hand_side_norm, 1.0_dp)
        if (right_hand_side_norm == 0.0_dp) then
            solution = 0.0_dp
            residual_norm = 0.0_dp
            info = KRYLOV_OK
            return
        end if

        done = .false.
        info = KRYLOV_MAX_ITERATIONS
        !$acc data copyin( &
        !$acc& self%points, self%program_kind, self%program_variance, &
        !$acc& self%program_lengthscale, right_hand_side, diagonal_values) &
        !$acc& copy(solution) create( &
        !$acc& residual, direction, preconditioned, operator_direction, work)
        call self%matvec(solution, work)
        call subtract_kernel_vectors(right_hand_side, work, residual)
        residual_norm = kernel_acc_norm2(residual)
        if (residual_norm <= target) then
            info = KRYLOV_OK
            done = .true.
        else
            call apply_kernel_diagonal_preconditioner( &
                residual, preconditioned, diagonal_values)
            rho = kernel_acc_dot(residual, preconditioned)
            if (rho <= 0.0_dp .or. .not. ieee_is_finite(rho)) then
                info = KRYLOV_BREAKDOWN
                done = .true.
            else
                call copy_kernel_vector(direction, preconditioned)
                do while (iterations < max_iterations .and. .not. done)
                    call self%matvec(direction, operator_direction)
                    denominator = kernel_acc_dot(direction, operator_direction)
                    if (denominator <= 0.0_dp .or. &
                        .not. ieee_is_finite(denominator)) then
                        info = KRYLOV_BREAKDOWN
                        done = .true.
                    else
                        step = rho/denominator
                        call update_kernel_solution( &
                            solution, step, direction)
                        call subtract_kernel_scaled( &
                            residual, step, operator_direction)
                        iterations = iterations + 1
                        residual_norm = kernel_acc_norm2(residual)
                        if (residual_norm <= target) then
                            call self%matvec(solution, work)
                            call subtract_kernel_vectors( &
                                right_hand_side, work, residual)
                            residual_norm = kernel_acc_norm2(residual)
                            if (residual_norm <= target) then
                                info = KRYLOV_OK
                                done = .true.
                            end if
                        end if
                        if (.not. done) then
                            call apply_kernel_diagonal_preconditioner( &
                                residual, preconditioned, diagonal_values)
                            next_rho = kernel_acc_dot( &
                                residual, preconditioned)
                            if (next_rho <= 0.0_dp .or. &
                                .not. ieee_is_finite(next_rho)) then
                                info = KRYLOV_BREAKDOWN
                                done = .true.
                            else
                                beta = next_rho/rho
                                call combine_kernel_direction( &
                                    direction, preconditioned, beta)
                                rho = next_rho
                            end if
                        end if
                    end if
                end do
            end if
        end if
        if (.not. done) then
            call self%matvec(solution, work)
            call subtract_kernel_vectors(right_hand_side, work, residual)
            residual_norm = kernel_acc_norm2(residual)
            if (residual_norm <= target) info = KRYLOV_OK
        end if
        !$acc end data

    contains

        function kernel_acc_dot(left, right) result(value)
            real(dp), intent(in) :: left(:), right(:)
            real(dp) :: value
            integer :: i

            value = 0.0_dp
            !$acc parallel loop reduction(+:value)
            !$omp parallel do reduction(+:value)
            do i = 1, size(left)
                value = value + left(i)*right(i)
            end do
        end function kernel_acc_dot

        function kernel_acc_norm2(vector) result(value)
            real(dp), intent(in) :: vector(:)
            real(dp) :: value

            value = sqrt(kernel_acc_dot(vector, vector))
        end function kernel_acc_norm2

        subroutine subtract_kernel_vectors(left, right, result)
            real(dp), intent(in) :: left(:), right(:)
            real(dp), intent(out) :: result(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(result)
                result(i) = left(i) - right(i)
            end do
        end subroutine subtract_kernel_vectors

        subroutine copy_kernel_vector(target, source)
            real(dp), intent(out) :: target(:)
            real(dp), intent(in) :: source(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(target)
                target(i) = source(i)
            end do
        end subroutine copy_kernel_vector

        subroutine apply_kernel_diagonal_preconditioner( &
                input, output, diagonal)
            real(dp), intent(in) :: input(:), diagonal(:)
            real(dp), intent(out) :: output(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(output)
                output(i) = input(i)/diagonal(i)
            end do
        end subroutine apply_kernel_diagonal_preconditioner

        subroutine update_kernel_solution(current, scale, vector)
            real(dp), intent(inout) :: current(:)
            real(dp), intent(in) :: scale, vector(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(current)
                current(i) = current(i) + scale*vector(i)
            end do
        end subroutine update_kernel_solution

        subroutine subtract_kernel_scaled(current, scale, vector)
            real(dp), intent(inout) :: current(:)
            real(dp), intent(in) :: scale, vector(:)
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(current)
                current(i) = current(i) - scale*vector(i)
            end do
        end subroutine subtract_kernel_scaled

        subroutine combine_kernel_direction(current, preconditioned, scale)
            real(dp), intent(inout) :: current(:)
            real(dp), intent(in) :: preconditioned(:), scale
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(current)
                current(i) = preconditioned(i) + scale*current(i)
            end do
        end subroutine combine_kernel_direction

    end subroutine kernel_operator_solve_cg_device

    subroutine kernel_operator_solve_cg_multi_device( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            info, iterations, residual_norm, use_diagonal_preconditioner)
        class(kernel_operator_t), intent(in) :: self
        real(dp), intent(in) :: right_hand_side(:, :)
        real(dp), intent(inout) :: solution(:, :)
        real(dp), intent(in) :: tolerance
        integer, intent(in) :: max_iterations
        integer, intent(out) :: info(:), iterations(:)
        real(dp), intent(out) :: residual_norm(:)
        logical, intent(in), optional :: use_diagonal_preconditioner

        logical :: use_preconditioner
        real(dp), allocatable :: residual(:, :), direction(:, :)
        real(dp), allocatable :: preconditioned(:, :)
        real(dp), allocatable :: operator_direction(:, :), work(:, :)
        real(dp), allocatable :: diagonal_values(:), target(:), rho(:)
        real(dp), allocatable :: next_rho(:), denominator(:), step_values(:)
        real(dp), allocatable :: beta_values(:)
        logical, allocatable :: active(:), candidate(:)
        logical :: done
        integer :: n_samples, n_rhs, column

        info = KRYLOV_INVALID_ARGUMENT
        iterations = 0
        residual_norm = huge(1.0_dp)
        n_samples = self%sample_count()
        n_rhs = size(right_hand_side, 2)
        if (n_samples < 1 .or. n_rhs < 1) return
        if (size(right_hand_side, 1) /= n_samples) return
        if (size(solution, 1) /= n_samples .or. &
            size(solution, 2) /= n_rhs) return
        if (size(info) /= n_rhs .or. size(iterations) /= n_rhs .or. &
            size(residual_norm) /= n_rhs) return
        if (tolerance <= 0.0_dp .or. max_iterations < 1) return
        if (.not. self%device_supported()) return
        if (.not. self%points_device_resident) return
        if (.not. self%program_device_resident) return

        use_preconditioner = .true.
        if (present(use_diagonal_preconditioner)) then
            use_preconditioner = use_diagonal_preconditioner
        end if
        allocate( &
            residual(n_samples, n_rhs), direction(n_samples, n_rhs), &
            preconditioned(n_samples, n_rhs), &
            operator_direction(n_samples, n_rhs), work(n_samples, n_rhs), &
            diagonal_values(n_samples), target(n_rhs), rho(n_rhs), &
            next_rho(n_rhs), denominator(n_rhs), step_values(n_rhs), &
            beta_values(n_rhs), active(n_rhs), candidate(n_rhs))
        diagonal_values = 1.0_dp
        if (use_preconditioner) then
            diagonal_values = self%diagonal()
            if (size(diagonal_values) /= n_samples) return
            if (any(diagonal_values <= 0.0_dp)) return
        end if

        do column = 1, n_rhs
            target(column) = tolerance*max(sqrt(sum( &
                right_hand_side(:, column)*right_hand_side(:, column))), 1.0_dp)
        end do
        info = KRYLOV_MAX_ITERATIONS
        iterations = 0
        residual_norm = huge(1.0_dp)
        active = .false.
        candidate = .false.
        done = .false.

        !$acc data copyin( &
        !$acc& self%points, self%program_kind, self%program_variance, &
        !$acc& self%program_lengthscale, right_hand_side, diagonal_values) &
        !$acc& copy(solution) create( &
        !$acc& residual, direction, preconditioned, operator_direction, work, &
        !$acc& step_values, beta_values)
        call self%matmat(solution, work)
        call subtract_kernel_matrices( &
            right_hand_side, work, residual)
        call matrix_acc_norm2_columns(residual, residual_norm)
        if (use_preconditioner) then
            call apply_kernel_preconditioner_matrix( &
                residual, preconditioned, diagonal_values)
        else
            call copy_kernel_matrix(preconditioned, residual)
        end if
        call matrix_acc_dots(residual, preconditioned, rho)
        do column = 1, n_rhs
            if (residual_norm(column) <= target(column)) then
                info(column) = KRYLOV_OK
            else
                if (rho(column) <= 0.0_dp .or. &
                    .not. ieee_is_finite(rho(column))) then
                    info(column) = KRYLOV_BREAKDOWN
                else
                    active(column) = .true.
                end if
            end if
        end do
        call copy_kernel_matrix(direction, preconditioned)

        do while (any(active) .and. .not. done)
            call self%matmat(direction, operator_direction)
            call matrix_acc_dots( &
                direction, operator_direction, denominator)
            step_values = 0.0_dp
            candidate = .false.
            do column = 1, n_rhs
                if (active(column)) then
                    if (denominator(column) <= 0.0_dp .or. &
                        .not. ieee_is_finite(denominator(column))) then
                        info(column) = KRYLOV_BREAKDOWN
                        active(column) = .false.
                    else
                        step_values(column) = rho(column)/denominator(column)
                        iterations(column) = iterations(column) + 1
                    end if
                end if
            end do
            !$acc update device(step_values)
            call update_kernel_matrix(solution, step_values, direction)
            call subtract_kernel_scaled_matrix( &
                residual, step_values, operator_direction)
            call matrix_acc_norm2_columns(residual, residual_norm)
            do column = 1, n_rhs
                if (active(column) .and. &
                    (residual_norm(column) <= target(column) .or. &
                    iterations(column) >= max_iterations)) then
                    candidate(column) = .true.
                end if
            end do

            if (any(candidate)) then
                call self%matmat(solution, work)
                call subtract_kernel_matrices( &
                    right_hand_side, work, residual)
                call matrix_acc_norm2_columns(residual, residual_norm)
                do column = 1, n_rhs
                    if (candidate(column)) then
                        if (residual_norm(column) <= target(column)) then
                            info(column) = KRYLOV_OK
                            active(column) = .false.
                            call zero_kernel_column(direction, column)
                        else if (iterations(column) >= max_iterations) then
                            info(column) = KRYLOV_MAX_ITERATIONS
                            active(column) = .false.
                            call zero_kernel_column(direction, column)
                        else
                            candidate(column) = .false.
                        end if
                    end if
                end do
            end if

            if (use_preconditioner) then
                call apply_kernel_preconditioner_matrix( &
                    residual, preconditioned, diagonal_values)
            else
                call copy_kernel_matrix(preconditioned, residual)
            end if
            call matrix_acc_dots(residual, preconditioned, next_rho)
            beta_values = 0.0_dp
            do column = 1, n_rhs
                if (active(column)) then
                    if (next_rho(column) <= 0.0_dp .or. &
                        .not. ieee_is_finite(next_rho(column))) then
                        info(column) = KRYLOV_BREAKDOWN
                        active(column) = .false.
                    else
                        beta_values(column) = next_rho(column)/rho(column)
                        rho(column) = next_rho(column)
                    end if
                end if
            end do
            !$acc update device(beta_values)
            call combine_kernel_matrix(direction, preconditioned, beta_values)
            done = .not. any(active)
        end do

        call self%matmat(solution, work)
        call subtract_kernel_matrices( &
            right_hand_side, work, residual)
        call matrix_acc_norm2_columns(residual, residual_norm)
        do column = 1, n_rhs
            if (residual_norm(column) <= target(column)) then
                info(column) = KRYLOV_OK
            end if
        end do
        !$acc end data

    contains

        subroutine matrix_acc_dots(left, right, values)
            real(dp), intent(in) :: left(:, :), right(:, :)
            real(dp), intent(out) :: values(:)
            real(dp) :: value
            integer :: i, column

            !$acc parallel loop gang private(value)
            !$omp parallel do private(value)
            do column = 1, size(left, 2)
                value = 0.0_dp
                !$acc loop vector reduction(+:value)
                !$omp simd reduction(+:value)
                do i = 1, size(left, 1)
                    value = value + left(i, column)*right(i, column)
                end do
                values(column) = value
            end do
        end subroutine matrix_acc_dots

        subroutine matrix_acc_norm2_columns(vector, values)
            real(dp), intent(in) :: vector(:, :)
            real(dp), intent(out) :: values(:)

            call matrix_acc_dots(vector, vector, values)
            values = sqrt(values)
        end subroutine matrix_acc_norm2_columns

        subroutine subtract_kernel_matrices(left, right, result)
            real(dp), intent(in) :: left(:, :), right(:, :)
            real(dp), intent(out) :: result(:, :)
            integer :: i, column

            !$acc parallel loop collapse(2)
            !$omp parallel do collapse(2)
            do column = 1, size(result, 2)
                do i = 1, size(result, 1)
                    result(i, column) = left(i, column) - right(i, column)
                end do
            end do
        end subroutine subtract_kernel_matrices

        subroutine apply_kernel_preconditioner_matrix(input, output, diagonal)
            real(dp), intent(in) :: input(:, :), diagonal(:)
            real(dp), intent(out) :: output(:, :)
            integer :: i, column

            !$acc parallel loop collapse(2)
            !$omp parallel do collapse(2)
            do column = 1, size(output, 2)
                do i = 1, size(output, 1)
                    output(i, column) = input(i, column)/diagonal(i)
                end do
            end do
        end subroutine apply_kernel_preconditioner_matrix

        subroutine copy_kernel_matrix(target, source)
            real(dp), intent(out) :: target(:, :)
            real(dp), intent(in) :: source(:, :)
            integer :: i, column

            !$acc parallel loop collapse(2)
            !$omp parallel do collapse(2)
            do column = 1, size(target, 2)
                do i = 1, size(target, 1)
                    target(i, column) = source(i, column)
                end do
            end do
        end subroutine copy_kernel_matrix

        subroutine zero_kernel_column(target, column)
            real(dp), intent(inout) :: target(:, :)
            integer, intent(in) :: column
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(target, 1)
                target(i, column) = 0.0_dp
            end do
        end subroutine zero_kernel_column

        subroutine update_kernel_matrix(target, scales, vector)
            real(dp), intent(inout) :: target(:, :)
            real(dp), intent(in) :: scales(:), vector(:, :)
            integer :: i, column

            !$acc parallel loop collapse(2)
            !$omp parallel do collapse(2)
            do column = 1, size(target, 2)
                do i = 1, size(target, 1)
                    target(i, column) = target(i, column) + &
                        scales(column)*vector(i, column)
                end do
            end do
        end subroutine update_kernel_matrix

        subroutine subtract_kernel_scaled_matrix(target, scales, vector)
            real(dp), intent(inout) :: target(:, :)
            real(dp), intent(in) :: scales(:), vector(:, :)
            integer :: i, column

            !$acc parallel loop collapse(2)
            !$omp parallel do collapse(2)
            do column = 1, size(target, 2)
                do i = 1, size(target, 1)
                    target(i, column) = target(i, column) - &
                        scales(column)*vector(i, column)
                end do
            end do
        end subroutine subtract_kernel_scaled_matrix

        subroutine combine_kernel_matrix(target, source, scales)
            real(dp), intent(inout) :: target(:, :)
            real(dp), intent(in) :: source(:, :)
            real(dp), intent(in) :: scales(:)
            integer :: i, column

            !$acc parallel loop collapse(2)
            !$omp parallel do collapse(2)
            do column = 1, size(target, 2)
                do i = 1, size(target, 1)
                    target(i, column) = source(i, column) + &
                        scales(column)*target(i, column)
                end do
            end do
        end subroutine combine_kernel_matrix

    end subroutine kernel_operator_solve_cg_multi_device

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
        if (kernel_operator_simple_rbf(self)) then
            call rbf_matvec_tiled( &
                self%points, input, output, &
                self%program_variance(1), self%program_lengthscale(1), &
                self%diagonal_shift, self%tile_size)
            return
        end if
        if (self%device_supported()) then
            call kernel_program_matvec_tiled( &
                self%points, input, output, self%program_kind, &
                self%program_variance, self%program_lengthscale, &
                self%diagonal_shift, self%tile_size)
            return
        end if
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
        real(dp), allocatable :: matrix_block(:, :)
        type(fortnum_status_t) :: status
        integer :: block_size, first_row, last_row, n_samples

        output = 0.0_dp
        n_samples = self%sample_count()
        if (n_samples < 1) return
        if (size(output, 1) /= n_samples) return
        if (size(output, 2) /= size(input, 2)) return
        if (size(input, 1) /= n_samples) return
        if (self%tile_size < 1) return
        if (kernel_operator_simple_rbf(self)) then
            call rbf_matmat_tiled( &
                self%points, input, output, &
                self%program_variance(1), self%program_lengthscale(1), &
                self%diagonal_shift, self%tile_size)
            return
        end if
        if (self%device_supported()) then
            call kernel_program_matmat_tiled( &
                self%points, input, output, self%program_kind, &
                self%program_variance, self%program_lengthscale, &
                self%diagonal_shift, self%tile_size)
            return
        end if

        block_size = min(self%tile_size, n_samples)
        allocate(matrix_block(block_size, n_samples))
        do first_row = 1, n_samples, block_size
            last_row = min(n_samples, first_row + block_size - 1)
            call self%kernel%matrix( &
                self%points(first_row:last_row, :), self%points, &
                matrix_block(:last_row - first_row + 1, :), status)
            if (status%code /= FORTNUM_OK) then
                output = 0.0_dp
                return
            end if
            output(first_row:last_row, :) = self%diagonal_shift* &
                input(first_row:last_row, :) + matmul( &
                matrix_block(:last_row - first_row + 1, :), input)
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

    pure subroutine evaluate_kernel_program( &
            points, left_index, right_index, program_kind, program_variance, &
            program_lengthscale, value)
        !$acc routine seq
        real(dp), intent(in) :: points(:, :)
        integer, intent(in) :: left_index, right_index
        integer, intent(in) :: program_kind(:)
        real(dp), intent(in) :: program_variance(:), program_lengthscale(:)
        real(dp), intent(out) :: value
        real(dp) :: stack(MAX_KERNEL_PROGRAM)
        real(dp) :: distance, difference, exponential, linear_value, r, a
        integer :: feature, instruction, stack_top

        distance = 0.0_dp
        linear_value = 0.0_dp
        do feature = 1, size(points, 2)
            difference = points(left_index, feature) - &
                points(right_index, feature)
            distance = distance + difference*difference
            linear_value = linear_value + &
                points(left_index, feature)*points(right_index, feature)
        end do
        stack_top = 0
        do instruction = 1, size(program_kind)
            select case (program_kind(instruction))
            case (KERNEL_RBF)
                stack_top = stack_top + 1
                stack(stack_top) = program_variance(instruction)*exp( &
                    -0.5_dp*distance/( &
                    program_lengthscale(instruction)*program_lengthscale(instruction)))
            case (KERNEL_MATERN12)
                stack_top = stack_top + 1
                r = sqrt(distance)/program_lengthscale(instruction)
                stack(stack_top) = program_variance(instruction)*exp(-r)
            case (KERNEL_MATERN32)
                stack_top = stack_top + 1
                a = sqrt(3.0_dp)
                r = sqrt(distance)/program_lengthscale(instruction)
                exponential = exp(-a*r)
                stack(stack_top) = program_variance(instruction)* &
                    (1.0_dp + a*r)*exponential
            case (KERNEL_MATERN52)
                stack_top = stack_top + 1
                a = sqrt(5.0_dp)
                r = sqrt(distance)/program_lengthscale(instruction)
                exponential = exp(-a*r)
                stack(stack_top) = program_variance(instruction)* &
                    (1.0_dp + a*r + 5.0_dp*r*r/3.0_dp)*exponential
            case (KERNEL_LINEAR)
                stack_top = stack_top + 1
                stack(stack_top) = program_variance(instruction)*linear_value
            case (KERNEL_CONSTANT)
                stack_top = stack_top + 1
                stack(stack_top) = program_variance(instruction)
            case (KERNEL_WHITE_NOISE)
                stack_top = stack_top + 1
                stack(stack_top) = program_variance(instruction)*merge( &
                    1.0_dp, 0.0_dp, distance == 0.0_dp)
            case (KERNEL_SUM)
                stack(stack_top - 1) = stack(stack_top - 1) + stack(stack_top)
                stack_top = stack_top - 1
            case (KERNEL_PRODUCT)
                stack(stack_top - 1) = stack(stack_top - 1)*stack(stack_top)
                stack_top = stack_top - 1
            end select
        end do
        value = stack(stack_top)
    end subroutine evaluate_kernel_program

    subroutine kernel_program_matvec_tiled( &
            points, input, output, program_kind, program_variance, &
            program_lengthscale, diagonal_shift, tile_size)
        real(dp), intent(in) :: points(:, :), input(:)
        real(dp), intent(out) :: output(:)
        integer, intent(in) :: program_kind(:), tile_size
        real(dp), intent(in) :: program_variance(:), program_lengthscale(:)
        real(dp), intent(in) :: diagonal_shift
        real(dp) :: accumulated, kernel_value
        integer :: block_size, first_neighbor, last_neighbor, i, j

        output = 0.0_dp
        if (size(points, 1) < 1) return
        if (size(points, 2) < 1) return
        if (size(input) /= size(points, 1)) return
        if (size(output) /= size(input)) return
        if (size(program_kind) < 1) return
        if (size(program_kind) > MAX_KERNEL_PROGRAM) return
        if (size(program_variance) /= size(program_kind)) return
        if (size(program_lengthscale) /= size(program_kind)) return
        if (tile_size < 1) return

        block_size = tile_size
        !$acc parallel loop private(accumulated, kernel_value, &
        !$acc& first_neighbor, last_neighbor, j)
        !$omp parallel do schedule(static) private( &
        !$omp& accumulated, kernel_value, first_neighbor, last_neighbor, j)
        do i = 1, size(points, 1)
            accumulated = diagonal_shift*input(i)
            do first_neighbor = 1, size(points, 1), block_size
                last_neighbor = min( &
                    size(points, 1), first_neighbor + block_size - 1)
                !$acc loop vector reduction(+:accumulated) private(kernel_value)
                !$omp simd reduction(+:accumulated) private(kernel_value)
                do j = first_neighbor, last_neighbor
                    call evaluate_kernel_program( &
                        points, i, j, program_kind, &
                        program_variance, program_lengthscale, kernel_value)
                    accumulated = accumulated + kernel_value*input(j)
                end do
            end do
            output(i) = accumulated
        end do
    end subroutine kernel_program_matvec_tiled

    subroutine kernel_program_matmat_tiled( &
            points, input, output, program_kind, program_variance, &
            program_lengthscale, diagonal_shift, tile_size)
        real(dp), intent(in) :: points(:, :), input(:, :)
        real(dp), intent(out) :: output(:, :)
        integer, intent(in) :: program_kind(:), tile_size
        real(dp), intent(in) :: program_variance(:), program_lengthscale(:)
        real(dp), intent(in) :: diagonal_shift
        real(dp) :: accumulated_1, accumulated_2, accumulated_3
        real(dp) :: accumulated_4, accumulated_5, accumulated_6
        real(dp) :: accumulated_7, accumulated_8, kernel_value
        integer :: block_size, first_neighbor, last_neighbor, i, j, column
        integer :: n_rhs

        output = 0.0_dp
        if (size(points, 1) < 1) return
        if (size(points, 2) < 1) return
        if (size(input, 1) /= size(points, 1)) return
        if (size(output, 1) /= size(points, 1)) return
        if (size(output, 2) /= size(input, 2)) return
        if (size(program_kind) < 1) return
        if (size(program_kind) > MAX_KERNEL_PROGRAM) return
        if (size(program_variance) /= size(program_kind)) return
        if (size(program_lengthscale) /= size(program_kind)) return
        if (tile_size < 1) return
        n_rhs = size(input, 2)
        if (n_rhs < 1) return
        if (n_rhs > MAX_FUSED_RHS) then
            do column = 1, n_rhs
                call kernel_program_matvec_tiled( &
                    points, input(:, column), output(:, column), program_kind, &
                    program_variance, program_lengthscale, diagonal_shift, &
                    tile_size)
            end do
            return
        end if

        block_size = tile_size
        !$acc parallel loop private( &
        !$acc& accumulated_1, accumulated_2, accumulated_3, accumulated_4, &
        !$acc& accumulated_5, accumulated_6, accumulated_7, accumulated_8, &
        !$acc& kernel_value, first_neighbor, last_neighbor, j, column)
        !$omp parallel do schedule(static) private( &
        !$omp& accumulated_1, accumulated_2, accumulated_3, accumulated_4, &
        !$omp& accumulated_5, accumulated_6, accumulated_7, accumulated_8, &
        !$omp& kernel_value, first_neighbor, last_neighbor, j, column)
        do i = 1, size(points, 1)
            accumulated_1 = diagonal_shift*input(i, 1)
            accumulated_2 = 0.0_dp
            accumulated_3 = 0.0_dp
            accumulated_4 = 0.0_dp
            accumulated_5 = 0.0_dp
            accumulated_6 = 0.0_dp
            accumulated_7 = 0.0_dp
            accumulated_8 = 0.0_dp
            if (n_rhs >= 2) accumulated_2 = diagonal_shift*input(i, 2)
            if (n_rhs >= 3) accumulated_3 = diagonal_shift*input(i, 3)
            if (n_rhs >= 4) accumulated_4 = diagonal_shift*input(i, 4)
            if (n_rhs >= 5) accumulated_5 = diagonal_shift*input(i, 5)
            if (n_rhs >= 6) accumulated_6 = diagonal_shift*input(i, 6)
            if (n_rhs >= 7) accumulated_7 = diagonal_shift*input(i, 7)
            if (n_rhs >= 8) accumulated_8 = diagonal_shift*input(i, 8)
            do first_neighbor = 1, size(points, 1), block_size
                last_neighbor = min( &
                    size(points, 1), first_neighbor + block_size - 1)
                !$acc loop vector reduction(+:accumulated_1, accumulated_2, &
                !$acc& accumulated_3, accumulated_4, accumulated_5, &
                !$acc& accumulated_6, accumulated_7, accumulated_8) &
                !$acc& private(kernel_value)
                !$omp simd reduction(+:accumulated_1, accumulated_2, &
                !$omp& accumulated_3, accumulated_4, accumulated_5, &
                !$omp& accumulated_6, accumulated_7, accumulated_8) &
                !$omp& private(kernel_value)
                do j = first_neighbor, last_neighbor
                    call evaluate_kernel_program( &
                        points, i, j, program_kind, &
                        program_variance, program_lengthscale, kernel_value)
                    accumulated_1 = accumulated_1 + kernel_value*input(j, 1)
                    if (n_rhs >= 2) accumulated_2 = accumulated_2 + &
                        kernel_value*input(j, 2)
                    if (n_rhs >= 3) accumulated_3 = accumulated_3 + &
                        kernel_value*input(j, 3)
                    if (n_rhs >= 4) accumulated_4 = accumulated_4 + &
                        kernel_value*input(j, 4)
                    if (n_rhs >= 5) accumulated_5 = accumulated_5 + &
                        kernel_value*input(j, 5)
                    if (n_rhs >= 6) accumulated_6 = accumulated_6 + &
                        kernel_value*input(j, 6)
                    if (n_rhs >= 7) accumulated_7 = accumulated_7 + &
                        kernel_value*input(j, 7)
                    if (n_rhs >= 8) accumulated_8 = accumulated_8 + &
                        kernel_value*input(j, 8)
                end do
            end do
            output(i, 1) = accumulated_1
            if (n_rhs >= 2) output(i, 2) = accumulated_2
            if (n_rhs >= 3) output(i, 3) = accumulated_3
            if (n_rhs >= 4) output(i, 4) = accumulated_4
            if (n_rhs >= 5) output(i, 5) = accumulated_5
            if (n_rhs >= 6) output(i, 6) = accumulated_6
            if (n_rhs >= 7) output(i, 7) = accumulated_7
            if (n_rhs >= 8) output(i, 8) = accumulated_8
        end do
    end subroutine kernel_program_matmat_tiled

    subroutine rbf_matvec_tiled( &
            points, input, output, variance, lengthscale, diagonal_shift, &
            tile_size)
        real(dp), intent(in), target :: points(:, :), input(:)
        real(dp), intent(out), target :: output(:)
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
            if (fortml_cuda_rbf_available() /= 0_c_int) then
                if (rbf_matvec_cuda( &
                    points, input, output, variance, &
                    inverse_two_lengthscale_squared, diagonal_shift)) return
            end if
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

    logical function rbf_matvec_cuda( &
            points, input, output, variance, inverse_scale, diagonal_shift)
        real(dp), intent(in), target :: points(:, :), input(:)
        real(dp), intent(out), target :: output(:)
        real(dp), intent(in) :: variance, inverse_scale, diagonal_shift
        integer(c_int) :: status

        rbf_matvec_cuda = .false.
        !$acc data copyin(points, input) copyout(output)
        !$acc host_data use_device(points, input, output)
        status = fortml_cuda_rbf_matvec( &
            c_loc(points), c_loc(input), c_loc(output), &
            int(size(points, 1), c_int), real(variance, c_double), &
            real(inverse_scale, c_double), real(diagonal_shift, c_double))
        !$acc end host_data
        !$acc end data
        if (status == 0_c_int) then
            rbf_matvec_cuda = .true.
        end if
    end function rbf_matvec_cuda

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
        integer, parameter :: output_tile_size = 2
        integer :: block_size, first_neighbor, first_output, last_neighbor
        integer :: i, j, local_output, n_samples

        n_samples = size(points, 1)
        block_size = tile_size
        inverse_two_lengthscale_squared = &
            0.5_dp/(lengthscale*lengthscale)
        output = 0.0_dp

        !$acc parallel loop gang
        !$omp parallel do schedule(static) collapse(2) private( &
        !$omp& accumulated, distance, kernel_value, point_1, point_2, &
        !$omp& point_3, point_4, point_5, point_6, point_7, point_8, &
        !$omp& first_neighbor, first_output, i, j, last_neighbor, &
        !$omp& local_output)
        do first_output = 1, n_samples, output_tile_size
            !$acc loop worker
            do local_output = 1, output_tile_size
                i = first_output + local_output - 1
                if (i <= n_samples) then
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
                        last_neighbor = min( &
                            n_samples, first_neighbor + block_size - 1)
                        !$acc loop vector reduction(+:accumulated)
                        !$omp simd reduction(+:accumulated)
                        do j = first_neighbor, last_neighbor
                            distance = &
                                (point_1 - points(j, 1))*(point_1 - points(j, 1)) + &
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
                end if
            end do
        end do
    end subroutine rbf_matvec_tiled_8

    subroutine rbf_matmat_tiled( &
            points, input, output, variance, lengthscale, diagonal_shift, &
            tile_size)
        real(dp), intent(in), target :: points(:, :), input(:, :)
        real(dp), intent(out), target :: output(:, :)
        real(dp), intent(in) :: variance, lengthscale, diagonal_shift
        integer, intent(in) :: tile_size
        integer :: right_hand_side

        output = 0.0_dp
        if (size(output, 1) /= size(points, 1)) return
        if (size(output, 2) /= size(input, 2)) return
        if (size(points, 2) == 8 .and. size(input, 2) > 1 .and. &
            size(input, 2) <= 8 .and. fortml_cuda_rbf_available() /= 0_c_int) then
            if (rbf_matmat_cuda( &
                points, input, output, variance, &
                0.5_dp/(lengthscale*lengthscale), diagonal_shift)) return
        end if
        if (size(input, 2) <= MAX_FUSED_RHS) then
            call rbf_matmat_tiled_fused( &
                points, input, output, variance, lengthscale, diagonal_shift, &
                tile_size)
            return
        end if
        do right_hand_side = 1, size(input, 2)
            call rbf_matvec_tiled( &
                points, input(:, right_hand_side), output(:, right_hand_side), &
                variance, lengthscale, diagonal_shift, tile_size)
        end do
    end subroutine rbf_matmat_tiled

    subroutine rbf_matmat_tiled_fused( &
            points, input, output, variance, lengthscale, diagonal_shift, &
            tile_size)
        real(dp), intent(in) :: points(:, :), input(:, :)
        real(dp), intent(out) :: output(:, :)
        real(dp), intent(in) :: variance, lengthscale, diagonal_shift
        integer, intent(in) :: tile_size
        real(dp) :: accumulated_1, accumulated_2, accumulated_3
        real(dp) :: accumulated_4, accumulated_5, accumulated_6
        real(dp) :: accumulated_7, accumulated_8
        real(dp) :: inverse_two_lengthscale_squared
        real(dp) :: distance, difference, kernel_value
        integer :: block_size, first_neighbor, last_neighbor
        integer :: feature, i, j, n_features, n_rhs, n_samples

        output = 0.0_dp
        n_samples = size(points, 1)
        n_features = size(points, 2)
        n_rhs = size(input, 2)
        if (n_samples < 1 .or. n_features < 1 .or. n_rhs < 1 .or. &
            n_rhs > MAX_FUSED_RHS .or. size(input, 1) /= n_samples .or. &
            size(output, 1) /= n_samples .or. size(output, 2) /= n_rhs .or. &
            variance <= 0.0_dp .or. lengthscale <= 0.0_dp .or. &
            tile_size < 1) return

        block_size = tile_size
        inverse_two_lengthscale_squared = &
            0.5_dp/(lengthscale*lengthscale)
        !$acc parallel loop
        !$omp parallel do private( &
        !$omp& accumulated_1, accumulated_2, accumulated_3, accumulated_4, &
        !$omp& accumulated_5, accumulated_6, accumulated_7, accumulated_8, &
        !$omp& distance, difference, kernel_value, first_neighbor, &
        !$omp& last_neighbor, feature, j)
        do i = 1, n_samples
            accumulated_1 = diagonal_shift*input(i, 1)
            accumulated_2 = 0.0_dp
            accumulated_3 = 0.0_dp
            accumulated_4 = 0.0_dp
            accumulated_5 = 0.0_dp
            accumulated_6 = 0.0_dp
            accumulated_7 = 0.0_dp
            accumulated_8 = 0.0_dp
            if (n_rhs >= 2) accumulated_2 = diagonal_shift*input(i, 2)
            if (n_rhs >= 3) accumulated_3 = diagonal_shift*input(i, 3)
            if (n_rhs >= 4) accumulated_4 = diagonal_shift*input(i, 4)
            if (n_rhs >= 5) accumulated_5 = diagonal_shift*input(i, 5)
            if (n_rhs >= 6) accumulated_6 = diagonal_shift*input(i, 6)
            if (n_rhs >= 7) accumulated_7 = diagonal_shift*input(i, 7)
            if (n_rhs >= 8) accumulated_8 = diagonal_shift*input(i, 8)
            do first_neighbor = 1, n_samples, block_size
                last_neighbor = min(n_samples, first_neighbor + block_size - 1)
                !$acc loop vector reduction(+:accumulated_1, accumulated_2, &
                !$acc& accumulated_3, accumulated_4, accumulated_5, &
                !$acc& accumulated_6, accumulated_7, accumulated_8)
                !$omp simd reduction(+:accumulated_1, accumulated_2, &
                !$omp& accumulated_3, accumulated_4, accumulated_5, &
                !$omp& accumulated_6, accumulated_7, accumulated_8)
                do j = first_neighbor, last_neighbor
                    distance = 0.0_dp
                    do feature = 1, n_features
                        difference = points(i, feature) - points(j, feature)
                        distance = distance + difference*difference
                    end do
                    kernel_value = variance*exp( &
                        -inverse_two_lengthscale_squared*distance)
                    accumulated_1 = accumulated_1 + kernel_value*input(j, 1)
                    if (n_rhs >= 2) accumulated_2 = accumulated_2 + &
                        kernel_value*input(j, 2)
                    if (n_rhs >= 3) accumulated_3 = accumulated_3 + &
                        kernel_value*input(j, 3)
                    if (n_rhs >= 4) accumulated_4 = accumulated_4 + &
                        kernel_value*input(j, 4)
                    if (n_rhs >= 5) accumulated_5 = accumulated_5 + &
                        kernel_value*input(j, 5)
                    if (n_rhs >= 6) accumulated_6 = accumulated_6 + &
                        kernel_value*input(j, 6)
                    if (n_rhs >= 7) accumulated_7 = accumulated_7 + &
                        kernel_value*input(j, 7)
                    if (n_rhs >= 8) accumulated_8 = accumulated_8 + &
                        kernel_value*input(j, 8)
                end do
            end do
            output(i, 1) = accumulated_1
            if (n_rhs >= 2) output(i, 2) = accumulated_2
            if (n_rhs >= 3) output(i, 3) = accumulated_3
            if (n_rhs >= 4) output(i, 4) = accumulated_4
            if (n_rhs >= 5) output(i, 5) = accumulated_5
            if (n_rhs >= 6) output(i, 6) = accumulated_6
            if (n_rhs >= 7) output(i, 7) = accumulated_7
            if (n_rhs >= 8) output(i, 8) = accumulated_8
        end do
    end subroutine rbf_matmat_tiled_fused

    logical function rbf_matmat_cuda( &
            points, input, output, variance, inverse_scale, diagonal_shift)
        real(dp), intent(in), target :: points(:, :), input(:, :)
        real(dp), intent(out), target :: output(:, :)
        real(dp), intent(in) :: variance, inverse_scale, diagonal_shift
        integer(c_int) :: status

        rbf_matmat_cuda = .false.
        !$acc data copyin(points, input) copyout(output)
        !$acc host_data use_device(points, input, output)
        status = fortml_cuda_rbf_matmat( &
            c_loc(points), c_loc(input), c_loc(output), &
            int(size(points, 1), c_int), int(size(input, 2), c_int), &
            real(variance, c_double), real(inverse_scale, c_double), &
            real(diagonal_shift, c_double))
        !$acc end host_data
        !$acc end data
        if (status == 0_c_int) rbf_matmat_cuda = .true.
    end function rbf_matmat_cuda

end module fortml_kernel_operator
