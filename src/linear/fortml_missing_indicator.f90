!> Fitted binary indicators for IEEE-NaN feature entries.
module fortml_missing_indicator
    !! `missing_indicator_t` mirrors the dense scikit-learn MissingIndicator
    !! contract while retaining FortML's explicit derivative policy.  A mask
    !! is locally constant with respect to its input, so its JVP and VJP are
    !! exact zero products.  Infinities are rejected and NaNs are the only
    !! missing marker.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: MISSING_INDICATOR_ALL = 1
    integer, parameter, public :: MISSING_INDICATOR_MISSING_ONLY = 2

    type, public :: missing_indicator_t
        private
        integer, allocatable :: feature_index(:)
        integer :: n_inputs = 0
        integer :: mode_code = 0
        logical :: fitted_flag = .false.
    contains
        procedure, public :: fit => missing_indicator_fit
        procedure, public :: transform => missing_indicator_transform
        procedure, public :: transform_jvp => missing_indicator_transform_jvp
        procedure, public :: transform_vjp => missing_indicator_transform_vjp
        procedure, public :: feature_indices => missing_indicator_feature_indices
        procedure, public :: input_count => missing_indicator_input_count
        procedure, public :: output_count => missing_indicator_output_count
        procedure, public :: mode => missing_indicator_mode
        procedure, public :: fitted => missing_indicator_fitted
    end type missing_indicator_t

contains

    !> Fit either one indicator per input column or only columns missing in x.
    subroutine missing_indicator_fit(self, x, status, features)
        class(missing_indicator_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: features

        character(24) :: selected
        integer :: mode_code, i, n_selected
        logical :: has_missing

        call invalidate(self)
        if (.not. valid_input(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "missing indicator fit: input must be a nonempty finite/NaN matrix")
            return
        end if
        selected = "missing-only"
        if (present(features)) selected = adjustl(features)
        select case (trim(selected))
        case ("all")
            mode_code = MISSING_INDICATOR_ALL
        case ("missing-only", "missing_only")
            mode_code = MISSING_INDICATOR_MISSING_ONLY
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "missing indicator fit: features must be all or missing-only")
            return
        end select

        self%n_inputs = size(x, 2)
        if (mode_code == MISSING_INDICATOR_ALL) then
            allocate(self%feature_index(self%n_inputs))
            self%feature_index = [(i, i = 1, self%n_inputs)]
        else
            n_selected = 0
            do i = 1, self%n_inputs
                has_missing = any(ieee_is_nan(x(:, i)))
                if (has_missing) n_selected = n_selected + 1
            end do
            allocate(self%feature_index(n_selected))
            n_selected = 0
            do i = 1, self%n_inputs
                if (any(ieee_is_nan(x(:, i)))) then
                    n_selected = n_selected + 1
                    self%feature_index(n_selected) = i
                end if
            end do
        end if
        self%mode_code = mode_code
        self%fitted_flag = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine missing_indicator_fit

    !> Emit one in a selected feature column when its input is NaN, else zero.
    subroutine missing_indicator_transform(self, x, indicators, status)
        class(missing_indicator_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (.not. valid_transform(self, x, indicators, status, &
            "missing indicator transform")) return
        indicators = 0.0_dp
        do j = 1, size(self%feature_index)
            do i = 1, size(x, 1)
                if (ieee_is_nan(x(i, self%feature_index(j)))) indicators(i, j) = 1.0_dp
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine missing_indicator_transform

    !> The mask is locally constant; every input tangent maps to zero.
    subroutine missing_indicator_transform_jvp(self, x, x_dot, indicators_dot, status)
        class(missing_indicator_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: indicators_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        indicators_dot = 0.0_dp
        if (.not. valid_jvp(self, x, x_dot, indicators_dot, status, &
            "missing indicator JVP")) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine missing_indicator_transform_jvp

    !> The exact reverse product is zero because indicators are discrete masks.
    subroutine missing_indicator_transform_vjp(self, x, indicators_bar, x_bar, status)
        class(missing_indicator_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), indicators_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)

        type(fortnum_status_t), intent(out) :: status

        x_bar = 0.0_dp
        if (.not. valid_vjp(self, x, indicators_bar, x_bar, status, &
            "missing indicator VJP")) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine missing_indicator_transform_vjp

    function missing_indicator_feature_indices(self) result(indices)
        class(missing_indicator_t), intent(in) :: self
        integer, allocatable :: indices(:)

        if (allocated(self%feature_index)) then
            indices = self%feature_index
        else
            allocate(indices(0))
        end if
    end function missing_indicator_feature_indices

    integer function missing_indicator_input_count(self) result(count)
        class(missing_indicator_t), intent(in) :: self

        count = self%n_inputs
    end function missing_indicator_input_count

    integer function missing_indicator_output_count(self) result(count)
        class(missing_indicator_t), intent(in) :: self

        if (allocated(self%feature_index)) then
            count = size(self%feature_index)
        else
            count = 0
        end if
    end function missing_indicator_output_count

    integer function missing_indicator_mode(self) result(code)
        class(missing_indicator_t), intent(in) :: self

        code = self%mode_code
    end function missing_indicator_mode

    logical function missing_indicator_fitted(self) result(value)
        class(missing_indicator_t), intent(in) :: self

        value = self%fitted_flag .and. allocated(self%feature_index)
    end function missing_indicator_fitted

    subroutine invalidate(self)
        class(missing_indicator_t), intent(inout) :: self

        if (allocated(self%feature_index)) deallocate(self%feature_index)
        self%n_inputs = 0
        self%mode_code = 0
        self%fitted_flag = .false.
    end subroutine invalidate

    logical function valid_input(x) result(value)
        real(dp), intent(in) :: x(:, :)

        value = size(x, 1) > 0 .and. size(x, 2) > 0
        if (.not. value) return
        value = all(ieee_is_finite(x) .or. ieee_is_nan(x))
    end function valid_input

    logical function valid_transform(self, x, indicators, status, context) result(value)
        class(missing_indicator_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        value = self%fitted() .and. valid_input(x)
        if (.not. value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": model or input is invalid")
            return
        end if
        if (size(x, 2) /= self%n_inputs .or. any(shape(indicators) /= &
            [size(x, 1), self%output_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": shape is invalid")
            value = .false.
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function valid_transform

    logical function valid_jvp(self, x, x_dot, indicators_dot, status, context) result(value)
        class(missing_indicator_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: indicators_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        value = valid_transform(self, x, indicators_dot, status, context)
        if (.not. value) return
        if (any(shape(x_dot) /= shape(x)) .or. any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": tangent shape or values are invalid")
            value = .false.
        end if
    end function valid_jvp

    logical function valid_vjp(self, x, indicators_bar, x_bar, status, context) result(value)
        class(missing_indicator_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), indicators_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        value = self%fitted() .and. valid_input(x)
        if (.not. value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": model or input is invalid")
            return
        end if
        if (size(x, 2) /= self%n_inputs .or. any(shape(x_bar) /= shape(x)) .or. &
            any(shape(indicators_bar) /= [size(x, 1), self%output_count()]) .or. &
            any(.not. ieee_is_finite(indicators_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": shape or values are invalid")
            value = .false.
        end if
    end function valid_vjp

end module fortml_missing_indicator
