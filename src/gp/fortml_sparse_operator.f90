module fortml_sparse_operator
    use fortsparse_csc, only: csc_from_triplet, csc_t, csc_transpose
    use fortsparse_kinds, only: dp
    use fortsparse_status, only: FORTSPARSE_INVALID_MATRIX, FORTSPARSE_OK, &
        fortsparse_status_t, status_set
    use fortml_linear_operator, only: linear_operator_t
    implicit none
    private

    ! Sparse covariance/precision products. The public product contract is
    ! matrix-free; a CSR view is retained so each output row owns its sum on
    ! the accelerator and no atomic updates are needed.
    type, extends(linear_operator_t), public :: sparse_gp_operator_t
        integer :: sample_size = 0
        integer :: nonzero_size = 0
        integer, allocatable :: row_ptr(:)
        integer, allocatable :: column_index(:)
        real(dp), allocatable :: values(:)
        logical :: device_resident = .false.
    contains
        procedure, public :: initialize => sparse_gp_operator_initialize
        procedure, public :: matvec => sparse_gp_operator_matvec
        procedure, public :: matmat => sparse_gp_operator_matmat
        procedure, public :: enter_data => sparse_gp_operator_enter_data
        procedure, public :: exit_data => sparse_gp_operator_exit_data
        procedure, public :: matvec_device => sparse_gp_operator_matvec_device
        procedure, public :: matmat_device => sparse_gp_operator_matmat_device
        procedure, public :: diagonal => sparse_gp_operator_diagonal
        procedure, public :: sample_count => sparse_gp_operator_sample_count
        procedure, public :: nonzero_count => sparse_gp_operator_nonzero_count
    end type sparse_gp_operator_t

contains

    subroutine sparse_gp_operator_initialize(self, n_samples, rows, columns, &
            values, status)
        class(sparse_gp_operator_t), intent(out) :: self
        integer, intent(in) :: n_samples
        integer, intent(in) :: rows(:), columns(:)
        real(dp), intent(in) :: values(:)
        type(fortsparse_status_t), intent(out) :: status

        type(csc_t) :: matrix, transpose_matrix

        if (n_samples < 1) then
            call status_set(status, FORTSPARSE_INVALID_MATRIX, &
                "sparse GP operator: sample count must be positive")
            return
        end if
        call csc_from_triplet( &
            n_samples, n_samples, rows, columns, values, matrix, status)
        if (status%code /= FORTSPARSE_OK) return
        call csc_transpose(matrix, transpose_matrix, status)
        if (status%code /= FORTSPARSE_OK) return

        self%sample_size = n_samples
        self%nonzero_size = transpose_matrix%nnz
        allocate( &
            self%row_ptr(n_samples + 1), &
            self%column_index(transpose_matrix%nnz), &
            self%values(transpose_matrix%nnz))
        self%row_ptr = transpose_matrix%col_ptr
        self%column_index = transpose_matrix%row_idx
        self%values = transpose_matrix%val
        self%device_resident = .false.
        call status_set(status, FORTSPARSE_OK, "")
    end subroutine sparse_gp_operator_initialize

    subroutine sparse_gp_operator_matvec(self, input, output)
        class(sparse_gp_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)

        integer :: entry, row

        call validate_vector_shape(self, size(input), size(output))
        output = 0.0_dp
        do row = 1, self%sample_size
            do entry = self%row_ptr(row), self%row_ptr(row + 1) - 1
                output(row) = output(row) + self%values(entry)* &
                    input(self%column_index(entry))
            end do
        end do
    end subroutine sparse_gp_operator_matvec

    subroutine sparse_gp_operator_matmat(self, input, output)
        class(sparse_gp_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)

        integer :: column, entry, row

        call validate_matrix_shape(self, shape(input), shape(output))
        output = 0.0_dp
        do column = 1, size(input, 2)
            do row = 1, self%sample_size
                do entry = self%row_ptr(row), self%row_ptr(row + 1) - 1
                    output(row, column) = output(row, column) + &
                        self%values(entry)*input(self%column_index(entry), column)
                end do
            end do
        end do
    end subroutine sparse_gp_operator_matmat

    subroutine sparse_gp_operator_enter_data(self, status)
        class(sparse_gp_operator_t), intent(inout) :: self
        type(fortsparse_status_t), intent(out) :: status

        if (self%sample_size < 1 .or. .not. allocated(self%row_ptr)) then
            call status_set(status, FORTSPARSE_INVALID_MATRIX, &
                "sparse GP operator: initialize before enter_data")
            return
        end if
        if (.not. self%device_resident) then
            !$acc enter data copyin(self%row_ptr, self%column_index, self%values)
            self%device_resident = .true.
        end if
        call status_set(status, FORTSPARSE_OK, "")
    end subroutine sparse_gp_operator_enter_data

    subroutine sparse_gp_operator_exit_data(self, status)
        class(sparse_gp_operator_t), intent(inout) :: self
        type(fortsparse_status_t), intent(out) :: status

        if (self%device_resident) then
            !$acc exit data delete(self%row_ptr, self%column_index, self%values)
            self%device_resident = .false.
        end if
        call status_set(status, FORTSPARSE_OK, "")
    end subroutine sparse_gp_operator_exit_data

    subroutine sparse_gp_operator_matvec_device(self, input, output, status)
        class(sparse_gp_operator_t), intent(inout) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)
        type(fortsparse_status_t), intent(out) :: status

        integer :: entry, row
        real(dp) :: row_sum

        call validate_vector_shape_status( &
            self, size(input), size(output), status)
        if (status%code /= FORTSPARSE_OK) return
        if (.not. self%device_resident) then
            call status_set(status, FORTSPARSE_INVALID_MATRIX, &
                "sparse GP operator: enter_data is required")
            return
        end if
        !$acc data present(input, output, self%row_ptr, self%column_index, &
        !$acc& self%values)
        !$acc parallel loop gang
        do row = 1, self%sample_size
            row_sum = 0.0_dp
            do entry = self%row_ptr(row), self%row_ptr(row + 1) - 1
                row_sum = row_sum + self%values(entry)* &
                    input(self%column_index(entry))
            end do
            output(row) = row_sum
        end do
        !$acc end data
        call status_set(status, FORTSPARSE_OK, "")
    end subroutine sparse_gp_operator_matvec_device

    subroutine sparse_gp_operator_matmat_device(self, input, output, status)
        class(sparse_gp_operator_t), intent(inout) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        type(fortsparse_status_t), intent(out) :: status

        integer :: column, entry, row
        real(dp) :: row_sum

        call validate_matrix_shape_status( &
            self, shape(input), shape(output), status)
        if (status%code /= FORTSPARSE_OK) return
        if (.not. self%device_resident) then
            call status_set(status, FORTSPARSE_INVALID_MATRIX, &
                "sparse GP operator: enter_data is required")
            return
        end if
        !$acc data present(input, output, self%row_ptr, self%column_index, &
        !$acc& self%values)
        !$acc parallel loop gang collapse(2)
        do column = 1, size(input, 2)
            do row = 1, self%sample_size
                row_sum = 0.0_dp
                do entry = self%row_ptr(row), self%row_ptr(row + 1) - 1
                    row_sum = row_sum + self%values(entry)* &
                        input(self%column_index(entry), column)
                end do
                output(row, column) = row_sum
            end do
        end do
        !$acc end data
        call status_set(status, FORTSPARSE_OK, "")
    end subroutine sparse_gp_operator_matmat_device

    function sparse_gp_operator_diagonal(self) result(diagonal)
        class(sparse_gp_operator_t), intent(in) :: self
        real(dp), allocatable :: diagonal(:)

        integer :: entry, row

        allocate(diagonal(self%sample_size))
        diagonal = 0.0_dp
        do row = 1, self%sample_size
            do entry = self%row_ptr(row), self%row_ptr(row + 1) - 1
                if (self%column_index(entry) == row) then
                    diagonal(row) = diagonal(row) + self%values(entry)
                end if
            end do
        end do
    end function sparse_gp_operator_diagonal

    integer function sparse_gp_operator_sample_count(self) result(count)
        class(sparse_gp_operator_t), intent(in) :: self

        count = self%sample_size
    end function sparse_gp_operator_sample_count

    integer function sparse_gp_operator_nonzero_count(self) result(count)
        class(sparse_gp_operator_t), intent(in) :: self

        count = self%nonzero_size
    end function sparse_gp_operator_nonzero_count

    subroutine validate_vector_shape(self, input_size, output_size)
        class(sparse_gp_operator_t), intent(in) :: self
        integer, intent(in) :: input_size, output_size

        if (input_size /= self%sample_size .or. &
            output_size /= self%sample_size) then
            error stop "sparse GP operator: invalid vector shape"
        end if
    end subroutine validate_vector_shape

    subroutine validate_matrix_shape(self, input_shape, output_shape)
        class(sparse_gp_operator_t), intent(in) :: self
        integer, intent(in) :: input_shape(:), output_shape(:)

        if (size(input_shape) /= 2 .or. size(output_shape) /= 2 .or. &
            input_shape(1) /= self%sample_size .or. &
            output_shape(1) /= self%sample_size .or. &
            input_shape(2) /= output_shape(2)) then
            error stop "sparse GP operator: invalid matrix shape"
        end if
    end subroutine validate_matrix_shape

    subroutine validate_vector_shape_status( &
            self, input_size, output_size, status)
        class(sparse_gp_operator_t), intent(in) :: self
        integer, intent(in) :: input_size, output_size
        type(fortsparse_status_t), intent(out) :: status

        if (input_size /= self%sample_size .or. &
            output_size /= self%sample_size) then
            call status_set(status, FORTSPARSE_INVALID_MATRIX, &
                "sparse GP operator: invalid vector shape")
        else
            call status_set(status, FORTSPARSE_OK, "")
        end if
    end subroutine validate_vector_shape_status

    subroutine validate_matrix_shape_status( &
            self, input_shape, output_shape, status)
        class(sparse_gp_operator_t), intent(in) :: self
        integer, intent(in) :: input_shape(:), output_shape(:)
        type(fortsparse_status_t), intent(out) :: status

        if (size(input_shape) /= 2 .or. size(output_shape) /= 2 .or. &
            input_shape(1) /= self%sample_size .or. &
            output_shape(1) /= self%sample_size .or. &
            input_shape(2) /= output_shape(2)) then
            call status_set(status, FORTSPARSE_INVALID_MATRIX, &
                "sparse GP operator: invalid matrix shape")
        else
            call status_set(status, FORTSPARSE_OK, "")
        end if
    end subroutine validate_matrix_shape_status

end module fortml_sparse_operator
