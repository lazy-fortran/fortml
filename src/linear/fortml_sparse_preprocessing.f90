!> Sparse CSC preprocessing with explicit centering and derivative boundaries.
module fortml_sparse_preprocessing
    !! `sparse_standard_scaler_t` implements the sparse-safe branch of the
    !! scikit-learn StandardScaler contract.  CSC input is fitted column by
    !! column, counting implicit zeros in the mean and variance.  Centering is
    !! intentionally refused because a nonzero mean would densify a sparse
    !! matrix; callers must request `with_mean=.false.`.  Scaling, inverse
    !! scaling, and their JVP/VJP products preserve the CSC structure exactly.
    use fortnum_kinds, only: dp
    use fortsparse, only: csc_t, csc_is_valid, fortsparse_status_t, &
        status_set, FORTSPARSE_OK, FORTSPARSE_INVALID_MATRIX
    implicit none
    private

    type, public :: sparse_standard_scaler_t
        private
        real(dp), allocatable :: mean_value(:)
        real(dp), allocatable :: scale_value(:)
        integer :: n_samples = 0
        logical :: fitted_flag = .false.
        logical :: with_std_flag = .true.
    contains
        procedure, public :: fit => sparse_scaler_fit
        procedure, public :: transform => sparse_scaler_transform
        procedure, public :: inverse_transform => sparse_scaler_inverse
        procedure, public :: transform_jvp => sparse_scaler_transform_jvp
        procedure, public :: transform_vjp => sparse_scaler_transform_vjp
        procedure, public :: means => sparse_scaler_means
        procedure, public :: scales => sparse_scaler_scales
        procedure, public :: feature_count => sparse_scaler_feature_count
        procedure, public :: sample_count => sparse_scaler_sample_count
        procedure, public :: with_std => sparse_scaler_with_std
        procedure, public :: fitted => sparse_scaler_fitted
    end type sparse_standard_scaler_t

contains

    !> Fit sparse-safe scaling.  `with_mean=.true.` is rejected explicitly.
    subroutine sparse_scaler_fit(self, input, status, with_mean, with_std)
        class(sparse_standard_scaler_t), intent(out) :: self
        type(csc_t), intent(in) :: input
        type(fortsparse_status_t), intent(out) :: status
        logical, intent(in), optional :: with_mean, with_std

        logical :: center, scale
        integer :: column, entry
        real(dp) :: sum_value, sum_square, variance

        call invalidate(self)
        if (.not. csc_is_valid(input)) then
            call status_set(status, FORTSPARSE_INVALID_MATRIX, &
                "sparse standard scaler fit: input CSC matrix is invalid")
            return
        end if
        center = .false.
        if (present(with_mean)) center = with_mean
        if (center) then
            call status_set(status, FORTSPARSE_INVALID_MATRIX, &
                "sparse standard scaler fit: centering would densify CSC input")
            return
        end if
        scale = .true.
        if (present(with_std)) scale = with_std

        self%n_samples = input%nrow
        self%with_std_flag = scale
        allocate(self%mean_value(input%ncol), self%scale_value(input%ncol))
        self%mean_value = 0.0_dp
        self%scale_value = 1.0_dp
        do column = 1, input%ncol
            sum_value = 0.0_dp
            sum_square = 0.0_dp
            do entry = input%col_ptr(column), input%col_ptr(column + 1) - 1
                sum_value = sum_value + input%val(entry)
                sum_square = sum_square + input%val(entry)*input%val(entry)
            end do
            ! The absent entries are exact zeros and therefore contribute
            ! neither sum nor second moment, but are included in n_samples.
            self%mean_value(column) = sum_value / real(input%nrow, dp)
            if (scale) then
                variance = sum_square / real(input%nrow, dp) - &
                    self%mean_value(column)**2
                if (variance > 0.0_dp) self%scale_value(column) = sqrt(variance)
            end if
        end do
        self%fitted_flag = .true.
        call status_set(status, FORTSPARSE_OK, "")
    end subroutine sparse_scaler_fit

    subroutine sparse_scaler_transform(self, input, output, status)
        class(sparse_standard_scaler_t), intent(in) :: self
        type(csc_t), intent(in) :: input
        type(csc_t), intent(out) :: output
        type(fortsparse_status_t), intent(out) :: status

        call sparse_scaler_apply(self, input, output, .false., status)
    end subroutine sparse_scaler_transform

    subroutine sparse_scaler_inverse(self, input, output, status)
        class(sparse_standard_scaler_t), intent(in) :: self
        type(csc_t), intent(in) :: input
        type(csc_t), intent(out) :: output
        type(fortsparse_status_t), intent(out) :: status

        call sparse_scaler_apply(self, input, output, .true., status)
    end subroutine sparse_scaler_inverse

    !> JVP scales only stored values; structure is locally constant.
    subroutine sparse_scaler_transform_jvp(self, input_dot, output_dot, status)
        class(sparse_standard_scaler_t), intent(in) :: self
        type(csc_t), intent(in) :: input_dot
        type(csc_t), intent(out) :: output_dot
        type(fortsparse_status_t), intent(out) :: status

        call sparse_scaler_apply(self, input_dot, output_dot, .false., status)
    end subroutine sparse_scaler_transform_jvp

    !> VJP is the same diagonal scaling because the sparse map is linear.
    subroutine sparse_scaler_transform_vjp(self, output_bar, input_bar, status)
        class(sparse_standard_scaler_t), intent(in) :: self
        type(csc_t), intent(in) :: output_bar
        type(csc_t), intent(out) :: input_bar
        type(fortsparse_status_t), intent(out) :: status

        call sparse_scaler_apply(self, output_bar, input_bar, .false., status)
    end subroutine sparse_scaler_transform_vjp

    subroutine sparse_scaler_apply(self, input, output, inverse, status)
        class(sparse_standard_scaler_t), intent(in) :: self
        type(csc_t), intent(in) :: input
        type(csc_t), intent(out) :: output
        logical, intent(in) :: inverse
        type(fortsparse_status_t), intent(out) :: status
        integer :: column, entry
        real(dp) :: factor

        if (.not. sparse_scaler_valid(self) .or. .not. csc_is_valid(input) .or. &
                input%ncol /= size(self%scale_value)) then
            call status_set(status, FORTSPARSE_INVALID_MATRIX, &
                "sparse standard scaler transform: fitted state or shape is invalid")
            return
        end if
        output%nrow = input%nrow
        output%ncol = input%ncol
        output%nnz = input%nnz
        allocate(output%col_ptr(input%ncol + 1), output%row_idx(input%nnz), &
            output%val(input%nnz))
        output%col_ptr = input%col_ptr
        output%row_idx = input%row_idx
        do column = 1, input%ncol
            factor = self%scale_value(column)
            if (inverse) then
                do entry = input%col_ptr(column), input%col_ptr(column + 1) - 1
                    output%val(entry) = input%val(entry)*factor
                end do
            else
                do entry = input%col_ptr(column), input%col_ptr(column + 1) - 1
                    output%val(entry) = input%val(entry)/factor
                end do
            end if
        end do
        call status_set(status, FORTSPARSE_OK, "")
    end subroutine sparse_scaler_apply

    function sparse_scaler_means(self) result(values)
        class(sparse_standard_scaler_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%mean_value)) then
            values = self%mean_value
        else
            allocate(values(0))
        end if
    end function sparse_scaler_means

    function sparse_scaler_scales(self) result(values)
        class(sparse_standard_scaler_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%scale_value)) then
            values = self%scale_value
        else
            allocate(values(0))
        end if
    end function sparse_scaler_scales

    integer function sparse_scaler_feature_count(self) result(value)
        class(sparse_standard_scaler_t), intent(in) :: self

        value = 0
        if (allocated(self%scale_value)) value = size(self%scale_value)
    end function sparse_scaler_feature_count

    integer function sparse_scaler_sample_count(self) result(value)
        class(sparse_standard_scaler_t), intent(in) :: self

        value = self%n_samples
    end function sparse_scaler_sample_count

    logical function sparse_scaler_with_std(self) result(value)
        class(sparse_standard_scaler_t), intent(in) :: self

        value = self%with_std_flag
    end function sparse_scaler_with_std

    logical function sparse_scaler_fitted(self) result(value)
        class(sparse_standard_scaler_t), intent(in) :: self

        value = sparse_scaler_valid(self)
    end function sparse_scaler_fitted

    logical function sparse_scaler_valid(self) result(value)
        class(sparse_standard_scaler_t), intent(in) :: self

        value = self%fitted_flag .and. self%n_samples > 0 .and. &
            allocated(self%mean_value) .and. allocated(self%scale_value) .and. &
            size(self%mean_value) == size(self%scale_value)
    end function sparse_scaler_valid

    subroutine invalidate(self)
        class(sparse_standard_scaler_t), intent(inout) :: self

        if (allocated(self%mean_value)) deallocate(self%mean_value)
        if (allocated(self%scale_value)) deallocate(self%scale_value)
        self%n_samples = 0
        self%fitted_flag = .false.
        self%with_std_flag = .true.
    end subroutine invalidate

end module fortml_sparse_preprocessing
