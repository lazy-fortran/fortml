module fortml_kernel_operator
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_loc, c_ptr
    use fortnum_kinds, only: dp
    use fortnum_krylov, only: KRYLOV_BREAKDOWN, KRYLOV_INVALID_ARGUMENT, &
        KRYLOV_MAX_ITERATIONS, KRYLOV_OK
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
        logical :: points_device_resident = .false.
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

    subroutine rbf_operator_enter_data(self, status)
        class(rbf_operator_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status

        if (.not. allocated(self%points)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF operator: cannot enter data before initialize")
            return
        end if
        if (.not. self%points_device_resident) then
            !$acc enter data copyin(self%points)
            self%points_device_resident = .true.
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
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_operator_exit_data

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
            info, iterations, residual_norm, use_diagonal_preconditioner)
        class(rbf_operator_t), intent(in) :: self
        real(dp), intent(in) :: right_hand_side(:, :)
        real(dp), intent(inout) :: solution(:, :)
        real(dp), intent(in) :: tolerance
        integer, intent(in) :: max_iterations
        integer, intent(out) :: info(:), iterations(:)
        real(dp), intent(out) :: residual_norm(:)
        logical, intent(in), optional :: use_diagonal_preconditioner

        logical :: use_preconditioner
        logical, allocatable :: active(:), candidate(:)
        real(dp), allocatable :: residual(:, :), direction(:, :)
        real(dp), allocatable :: preconditioned(:, :), operator_direction(:, :)
        real(dp), allocatable :: work(:, :), diagonal_values(:)
        real(dp), allocatable :: right_hand_side_norm(:), target(:)
        real(dp), allocatable :: rho(:), next_rho(:), denominator(:)
        real(dp) :: step, beta
        integer :: n_samples, n_rhs, column

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

        use_preconditioner = .true.
        if (present(use_diagonal_preconditioner)) then
            use_preconditioner = use_diagonal_preconditioner
        end if
        allocate( &
            residual(n_samples, n_rhs), direction(n_samples, n_rhs), &
            preconditioned(n_samples, n_rhs), &
            operator_direction(n_samples, n_rhs), work(n_samples, n_rhs), &
            diagonal_values(n_samples), active(n_rhs), candidate(n_rhs), &
            right_hand_side_norm(n_rhs), target(n_rhs), rho(n_rhs), &
            next_rho(n_rhs), denominator(n_rhs))
        diagonal_values = 1.0_dp
        if (use_preconditioner) then
            diagonal_values = self%diagonal()
            if (size(diagonal_values) /= n_samples) return
            if (any(diagonal_values <= 0.0_dp)) return
        end if
        do column = 1, n_rhs
            right_hand_side_norm(column) = sqrt(sum( &
                right_hand_side(:, column)*right_hand_side(:, column)))
            target(column) = tolerance*max(right_hand_side_norm(column), 1.0_dp)
        end do
        info = KRYLOV_MAX_ITERATIONS
        iterations = 0
        residual_norm = huge(1.0_dp)
        active = .false.
        candidate = .false.

        !$acc data copyin(self%points, right_hand_side, diagonal_values) &
        !$acc& copy(solution) create(residual, direction, preconditioned, &
        !$acc& operator_direction, work)
        call self%matmat(solution, work)
        call subtract_matrix(right_hand_side, work, residual)
        do column = 1, n_rhs
            residual_norm(column) = acc_norm2_column(residual, column)
            if (residual_norm(column) <= target(column)) then
                info(column) = KRYLOV_OK
            else
                call apply_preconditioner_column( &
                    residual, preconditioned, column)
                rho(column) = acc_dot_column(residual, preconditioned, column)
                if (rho(column) <= 0.0_dp .or. &
                    .not. ieee_is_finite(rho(column))) then
                    info(column) = KRYLOV_BREAKDOWN
                else
                    call copy_column(direction, preconditioned, column)
                    active(column) = .true.
                end if
            end if
        end do

        do while (any(active))
            call self%matmat(direction, operator_direction)
            candidate = .false.
            do column = 1, n_rhs
                if (active(column)) then
                    denominator(column) = acc_dot_column( &
                        direction, operator_direction, column)
                    if (denominator(column) <= 0.0_dp .or. &
                        .not. ieee_is_finite(denominator(column))) then
                        info(column) = KRYLOV_BREAKDOWN
                        active(column) = .false.
                    else
                        step = rho(column)/denominator(column)
                        call update_column( &
                            solution, step, direction, column)
                        call subtract_scaled_column( &
                            residual, step, operator_direction, column)
                        iterations(column) = iterations(column) + 1
                        residual_norm(column) = &
                            acc_norm2_column(residual, column)
                        if (residual_norm(column) <= target(column) .or. &
                            iterations(column) >= max_iterations) then
                            candidate(column) = .true.
                        end if
                    end if
                end if
            end do

            if (any(candidate)) then
                call self%matmat(solution, work)
                do column = 1, n_rhs
                    if (candidate(column)) then
                        call subtract_column( &
                            right_hand_side, work, residual, column)
                        residual_norm(column) = &
                            acc_norm2_column(residual, column)
                        if (residual_norm(column) <= target(column)) then
                            info(column) = KRYLOV_OK
                            active(column) = .false.
                            call zero_column(direction, column)
                        else if (iterations(column) >= max_iterations) then
                            info(column) = KRYLOV_MAX_ITERATIONS
                            active(column) = .false.
                            call zero_column(direction, column)
                        else
                            candidate(column) = .false.
                        end if
                    end if
                end do
            end if

            do column = 1, n_rhs
                if (active(column)) then
                    call apply_preconditioner_column( &
                        residual, preconditioned, column)
                    next_rho(column) = acc_dot_column( &
                        residual, preconditioned, column)
                    if (next_rho(column) <= 0.0_dp .or. &
                        .not. ieee_is_finite(next_rho(column))) then
                        info(column) = KRYLOV_BREAKDOWN
                        active(column) = .false.
                    else
                        beta = next_rho(column)/rho(column)
                        call combine_column( &
                            direction, preconditioned, beta, column)
                        rho(column) = next_rho(column)
                    end if
                end if
            end do
        end do

        call self%matmat(solution, work)
        do column = 1, n_rhs
            call subtract_column( &
                right_hand_side, work, residual, column)
            residual_norm(column) = acc_norm2_column(residual, column)
            if (residual_norm(column) <= target(column)) then
                info(column) = KRYLOV_OK
            end if
        end do
        !$acc end data

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
            real(dp), intent(out) :: output(:, :)
            integer, intent(in) :: column
            integer :: i

            !$acc parallel loop
            !$omp parallel do
            do i = 1, size(output, 1)
                output(i, column) = input(i, column)/diagonal_values(i)
            end do
        end subroutine apply_preconditioner_column

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
        do right_hand_side = 1, size(input, 2)
            call rbf_matvec_tiled( &
                points, input(:, right_hand_side), output(:, right_hand_side), &
                variance, lengthscale, diagonal_shift, tile_size)
        end do
    end subroutine rbf_matmat_tiled

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
