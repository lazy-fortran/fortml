module fortml_linear_operator
    use fortnum_kinds, only: dp
    use fortnum_krylov, only: KRYLOV_INVALID_ARGUMENT, &
        real_conjugate_gradient_matmat_operator, &
        real_conjugate_gradient_operator
    implicit none
    private

    type, abstract, public :: linear_operator_t
    contains
        procedure(linear_operator_matvec_interface), deferred, public :: matvec
        procedure(linear_operator_matmat_interface), deferred, public :: matmat
        procedure(linear_operator_diagonal_interface), deferred, public :: diagonal
        procedure(linear_operator_sample_count_interface), deferred, public :: &
            sample_count
        procedure, public :: solve_cg => linear_operator_solve_cg
        procedure, public :: solve_cg_multi => linear_operator_solve_cg_multi
    end type linear_operator_t

    abstract interface
        subroutine linear_operator_matvec_interface(self, input, output)
            import :: dp, linear_operator_t
            class(linear_operator_t), intent(in) :: self
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)
        end subroutine linear_operator_matvec_interface

        subroutine linear_operator_matmat_interface(self, input, output)
            import :: dp, linear_operator_t
            class(linear_operator_t), intent(in) :: self
            real(dp), intent(in) :: input(:, :)
            real(dp), intent(out) :: output(:, :)
        end subroutine linear_operator_matmat_interface

        function linear_operator_diagonal_interface(self) result(values)
            import :: dp, linear_operator_t
            class(linear_operator_t), intent(in) :: self
            real(dp), allocatable :: values(:)
        end function linear_operator_diagonal_interface

        integer function linear_operator_sample_count_interface(self) result(count)
            import :: linear_operator_t
            class(linear_operator_t), intent(in) :: self
        end function linear_operator_sample_count_interface
    end interface

contains

    subroutine linear_operator_solve_cg( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            info, iterations, residual_norm, use_diagonal_preconditioner)
        class(linear_operator_t), intent(in) :: self
        real(dp), intent(in) :: right_hand_side(:)
        real(dp), intent(inout) :: solution(:)
        real(dp), intent(in) :: tolerance
        integer, intent(in) :: max_iterations
        integer, intent(out) :: info, iterations
        real(dp), intent(out) :: residual_norm
        logical, intent(in), optional :: use_diagonal_preconditioner

        logical :: use_preconditioner
        real(dp), allocatable :: diagonal_values(:)

        use_preconditioner = .true.
        if (present(use_diagonal_preconditioner)) then
            use_preconditioner = use_diagonal_preconditioner
        end if
        if (size(right_hand_side) /= self%sample_count()) then
            info = KRYLOV_INVALID_ARGUMENT
            iterations = 0
            residual_norm = huge(1.0_dp)
            return
        end if
        if (size(solution) /= self%sample_count()) then
            info = KRYLOV_INVALID_ARGUMENT
            iterations = 0
            residual_norm = huge(1.0_dp)
            return
        end if
        if (use_preconditioner) then
            diagonal_values = self%diagonal()
            if (size(diagonal_values) /= self%sample_count()) then
                info = KRYLOV_INVALID_ARGUMENT
                iterations = 0
                residual_norm = huge(1.0_dp)
                return
            end if
            if (any(diagonal_values <= 0.0_dp)) then
                info = KRYLOV_INVALID_ARGUMENT
                iterations = 0
                residual_norm = huge(1.0_dp)
                return
            end if
            call real_conjugate_gradient_operator( &
                apply_operator, right_hand_side, solution, tolerance, &
                max_iterations, info, iterations, residual_norm, &
                apply_diagonal_preconditioner)
        else
            call real_conjugate_gradient_operator( &
                apply_operator, right_hand_side, solution, tolerance, &
                max_iterations, info, iterations, residual_norm)
        end if

    contains

        subroutine apply_operator(input, output)
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)

            call self%matvec(input, output)
        end subroutine apply_operator

        subroutine apply_diagonal_preconditioner(input, output)
            real(dp), intent(in) :: input(:)
            real(dp), intent(out) :: output(:)

            output = input/diagonal_values
        end subroutine apply_diagonal_preconditioner

    end subroutine linear_operator_solve_cg

    subroutine linear_operator_solve_cg_multi( &
            self, right_hand_side, solution, tolerance, max_iterations, &
            info, iterations, residual_norm, use_diagonal_preconditioner)
        class(linear_operator_t), intent(inout) :: self
        real(dp), intent(in) :: right_hand_side(:, :)
        real(dp), intent(inout) :: solution(:, :)
        real(dp), intent(in) :: tolerance
        integer, intent(in) :: max_iterations
        integer, intent(out) :: info(:), iterations(:)
        real(dp), intent(out) :: residual_norm(:)
        logical, intent(in), optional :: use_diagonal_preconditioner

        logical :: use_preconditioner
        real(dp), allocatable :: diagonal_values(:)

        use_preconditioner = .true.
        if (present(use_diagonal_preconditioner)) then
            use_preconditioner = use_diagonal_preconditioner
        end if
        if (size(right_hand_side, 1) /= self%sample_count()) then
            info = KRYLOV_INVALID_ARGUMENT
            iterations = 0
            residual_norm = huge(1.0_dp)
            return
        end if
        if (any(shape(solution) /= shape(right_hand_side))) then
            info = KRYLOV_INVALID_ARGUMENT
            iterations = 0
            residual_norm = huge(1.0_dp)
            return
        end if
        if (size(info) /= size(right_hand_side, 2) .or. &
            size(iterations) /= size(right_hand_side, 2) .or. &
            size(residual_norm) /= size(right_hand_side, 2)) then
            info = KRYLOV_INVALID_ARGUMENT
            iterations = 0
            residual_norm = huge(1.0_dp)
            return
        end if

        if (use_preconditioner) then
            diagonal_values = self%diagonal()
            if (size(diagonal_values) /= self%sample_count()) then
                info = KRYLOV_INVALID_ARGUMENT
                iterations = 0
                residual_norm = huge(1.0_dp)
                return
            end if
            if (any(diagonal_values <= 0.0_dp)) then
                info = KRYLOV_INVALID_ARGUMENT
                iterations = 0
                residual_norm = huge(1.0_dp)
                return
            end if
            call real_conjugate_gradient_matmat_operator( &
                apply_operator, right_hand_side, solution, tolerance, &
                max_iterations, info, iterations, residual_norm, &
                apply_diagonal_preconditioner)
        else
            call real_conjugate_gradient_matmat_operator( &
                apply_operator, right_hand_side, solution, tolerance, &
                max_iterations, info, iterations, residual_norm)
        end if

    contains

        subroutine apply_operator(input, output)
            real(dp), intent(in) :: input(:, :)
            real(dp), intent(out) :: output(:, :)

            call self%matmat(input, output)
        end subroutine apply_operator

        subroutine apply_diagonal_preconditioner(input, output)
            real(dp), intent(in) :: input(:, :)
            real(dp), intent(out) :: output(:, :)

            output = input/spread( &
                diagonal_values, dim=2, ncopies=size(input, 2))
        end subroutine apply_diagonal_preconditioner

    end subroutine linear_operator_solve_cg_multi

end module fortml_linear_operator
