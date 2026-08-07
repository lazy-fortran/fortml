!> Fitted numerical imputation with explicit missing-value and derivative rules.
module fortml_simple_imputer
    !! `simple_imputer_t` follows the row-oriented estimator convention used by
    !! the preprocessing module.  Missing values are IEEE NaNs; infinities are
    !! always rejected.  The fitted statistic is state, not a trainable model
    !! parameter.  On an observed entry the transform is the identity; on a
    !! missing entry it is locally constant, so its input JVP/VJP is zero.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: SIMPLE_IMPUTER_MEAN = 1
    integer, parameter, public :: SIMPLE_IMPUTER_MEDIAN = 2
    integer, parameter, public :: SIMPLE_IMPUTER_CONSTANT = 3

    !> Fitted per-feature imputation statistics.
    type, public :: simple_imputer_t
        private
        real(dp), allocatable :: statistic_value(:)
        real(dp) :: fill_value = 0.0_dp
        integer :: strategy_code = 0
    contains
        procedure, public :: fit => simple_imputer_fit
        procedure, public :: transform => simple_imputer_transform
        procedure, public :: transform_jvp => simple_imputer_transform_jvp
        procedure, public :: transform_vjp => simple_imputer_transform_vjp
        procedure, public :: statistics => simple_imputer_statistics
        procedure, public :: feature_count => simple_imputer_feature_count
        procedure, public :: strategy => simple_imputer_strategy
        procedure, public :: fitted => simple_imputer_fitted
    end type simple_imputer_t

contains

    !> Fit mean, median, or constant statistics column by column.
    subroutine simple_imputer_fit(self, x, status, strategy, fill_value)
        class(simple_imputer_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: strategy
        real(dp), intent(in), optional :: fill_value
        character(16) :: selected
        integer :: code, i, j, n_observed
        real(dp), allocatable :: observed(:)

        if (.not. valid_missing_matrix(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "simple imputer fit: input must be a nonempty matrix without infinities")
            return
        end if
        selected = "mean"
        if (present(strategy)) selected = adjustl(strategy)
        select case (trim(selected))
        case ("mean")
            code = SIMPLE_IMPUTER_MEAN
        case ("median")
            code = SIMPLE_IMPUTER_MEDIAN
        case ("constant")
            code = SIMPLE_IMPUTER_CONSTANT
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "simple imputer fit: strategy must be mean, median, or constant")
            return
        end select

        self%strategy_code = code
        self%fill_value = 0.0_dp
        if (present(fill_value)) self%fill_value = fill_value
        if (.not. ieee_is_finite(self%fill_value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "simple imputer fit: fill_value must be finite")
            return
        end if
        allocate(self%statistic_value(size(x, 2)))
        self%statistic_value = self%fill_value

        do j = 1, size(x, 2)
            n_observed = 0
            do i = 1, size(x, 1)
                if (.not. ieee_is_nan(x(i, j))) n_observed = n_observed + 1
            end do
            if (code /= SIMPLE_IMPUTER_CONSTANT .and. n_observed < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "simple imputer fit: mean/median needs one observed value per feature")
                return
            end if
            if (code == SIMPLE_IMPUTER_MEAN) then
                self%statistic_value(j) = 0.0_dp
                do i = 1, size(x, 1)
                    if (.not. ieee_is_nan(x(i, j))) &
                        self%statistic_value(j) = self%statistic_value(j) + x(i, j)
                end do
                self%statistic_value(j) = self%statistic_value(j) / real(n_observed, dp)
            else if (code == SIMPLE_IMPUTER_MEDIAN) then
                allocate(observed(n_observed))
                n_observed = 0
                do i = 1, size(x, 1)
                    if (.not. ieee_is_nan(x(i, j))) then
                        n_observed = n_observed + 1
                        observed(n_observed) = x(i, j)
                    end if
                end do
                call sort_values(observed)
                if (mod(size(observed), 2) == 1) then
                    self%statistic_value(j) = observed((size(observed) + 1) / 2)
                else
                    self%statistic_value(j) = 0.5_dp * &
                        (observed(size(observed) / 2) + observed(size(observed) / 2 + 1))
                end if
                deallocate(observed)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine simple_imputer_fit

    !> Replace NaNs by the fitted per-feature statistics.
    subroutine simple_imputer_transform(self, x, transformed, status)
        class(simple_imputer_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (.not. simple_imputer_valid(self, x, transformed, status, &
            "simple imputer transform")) return
        transformed = x
        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                if (ieee_is_nan(x(i, j))) transformed(i, j) = self%statistic_value(j)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine simple_imputer_transform

    !> Input JVP of the piecewise transform.  Missing entries have zero tangent.
    subroutine simple_imputer_transform_jvp(self, x, x_dot, transformed_dot, status)
        class(simple_imputer_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: transformed_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (.not. simple_imputer_valid_jvp(self, x, x_dot, transformed_dot, &
            status, "simple imputer JVP")) return
        transformed_dot = x_dot
        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                if (ieee_is_nan(x(i, j))) transformed_dot(i, j) = 0.0_dp
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine simple_imputer_transform_jvp

    !> Input VJP of the piecewise transform.  Missing entries have zero cotangent.
    subroutine simple_imputer_transform_vjp(self, x, output_bar, input_bar, status)
        class(simple_imputer_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: input_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        input_bar = 0.0_dp
        if (.not. simple_imputer_valid_vjp(self, x, output_bar, input_bar, &
            status, "simple imputer VJP")) return
        input_bar = output_bar
        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                if (ieee_is_nan(x(i, j))) input_bar(i, j) = 0.0_dp
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine simple_imputer_transform_vjp

    function simple_imputer_statistics(self) result(values)
        class(simple_imputer_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%statistic_value)) then
            values = self%statistic_value
        else
            allocate(values(0))
        end if
    end function simple_imputer_statistics

    integer function simple_imputer_feature_count(self) result(count)
        class(simple_imputer_t), intent(in) :: self

        count = 0
        if (allocated(self%statistic_value)) count = size(self%statistic_value)
    end function simple_imputer_feature_count

    integer function simple_imputer_strategy(self) result(code)
        class(simple_imputer_t), intent(in) :: self

        code = self%strategy_code
    end function simple_imputer_strategy

    logical function simple_imputer_fitted(self) result(value)
        class(simple_imputer_t), intent(in) :: self

        value = allocated(self%statistic_value) .and. self%strategy_code > 0
    end function simple_imputer_fitted

    logical function valid_missing_matrix(x) result(value)
        real(dp), intent(in) :: x(:, :)

        value = size(x, 1) > 0 .and. size(x, 2) > 0
        if (.not. value) return
        value = all(ieee_is_finite(x) .or. ieee_is_nan(x))
    end function valid_missing_matrix

    logical function simple_imputer_valid(self, x, transformed, status, context) &
            result(value)
        class(simple_imputer_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        value = .false.
        if (.not. simple_imputer_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": model is not fitted")
            return
        end if
        if (.not. valid_missing_matrix(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": input has invalid values")
            return
        end if
        if (any(shape(transformed) /= shape(x)) .or. &
            size(x, 2) /= size(self%statistic_value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": shape is invalid")
            return
        end if
        value = .true.
        call status_set(status, FORTNUM_OK, "")
    end function simple_imputer_valid

    logical function simple_imputer_valid_jvp(self, x, x_dot, transformed_dot, &
            status, context) result(value)
        class(simple_imputer_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: transformed_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        value = .false.
        if (.not. simple_imputer_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": model is not fitted")
            return
        end if
        if (.not. valid_missing_matrix(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": input has invalid values")
            return
        end if
        if (any(shape(x_dot) /= shape(x)) .or. any(shape(transformed_dot) /= shape(x)) &
            .or. .not. all(ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": shape or tangent is invalid")
            return
        end if
        if (size(x, 2) /= size(self%statistic_value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": feature count is invalid")
            return
        end if
        value = .true.
        call status_set(status, FORTNUM_OK, "")
    end function simple_imputer_valid_jvp

    logical function simple_imputer_valid_vjp(self, x, output_bar, input_bar, &
            status, context) result(value)
        class(simple_imputer_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: input_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        value = .false.
        if (.not. simple_imputer_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": model is not fitted")
            return
        end if
        if (.not. valid_missing_matrix(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": input has invalid values")
            return
        end if
        if (any(shape(output_bar) /= shape(x)) .or. any(shape(input_bar) /= shape(x)) &
            .or. .not. all(ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": shape or cotangent is invalid")
            return
        end if
        if (size(x, 2) /= size(self%statistic_value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": feature count is invalid")
            return
        end if
        value = .true.
        call status_set(status, FORTNUM_OK, "")
    end function simple_imputer_valid_vjp

    subroutine sort_values(values)
        real(dp), intent(inout) :: values(:)
        integer :: i, j
        real(dp) :: key

        do i = 2, size(values)
            key = values(i)
            j = i - 1
            do while (j >= 1)
                if (values(j) <= key) exit
                values(j + 1) = values(j)
                j = j - 1
            end do
            values(j + 1) = key
        end do
    end subroutine sort_values

end module fortml_simple_imputer
