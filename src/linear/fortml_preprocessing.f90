module fortml_preprocessing
    !! Fitted numerical preprocessing with explicit sample-row semantics.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    type, public :: standard_scaler_t
        private
        real(dp), allocatable :: mean_value(:)
        real(dp), allocatable :: scale_value(:)
        logical :: with_mean = .true.
        logical :: with_std = .true.
    contains
        procedure, public :: fit => standard_scaler_fit
        procedure, public :: transform => standard_scaler_transform
        procedure, public :: inverse_transform => standard_scaler_inverse
        procedure, public :: transform_jvp => standard_scaler_transform_jvp
        procedure, public :: means => standard_scaler_means
        procedure, public :: scales => standard_scaler_scales
        procedure, public :: feature_count => standard_scaler_feature_count
        procedure, public :: fitted => standard_scaler_fitted
    end type standard_scaler_t

    type, public :: minmax_scaler_t
        private
        real(dp), allocatable :: data_min(:)
        real(dp), allocatable :: data_max(:)
        real(dp) :: lower = 0.0_dp
        real(dp) :: upper = 1.0_dp
    contains
        procedure, public :: fit => minmax_scaler_fit
        procedure, public :: transform => minmax_scaler_transform
        procedure, public :: inverse_transform => minmax_scaler_inverse
        procedure, public :: transform_jvp => minmax_scaler_transform_jvp
        procedure, public :: minimums => minmax_scaler_minimums
        procedure, public :: maximums => minmax_scaler_maximums
        procedure, public :: feature_count => minmax_scaler_feature_count
        procedure, public :: fitted => minmax_scaler_fitted
    end type minmax_scaler_t

    !> Fitted median/IQR preprocessing with row-oriented sample semantics.
    !>
    !> The default quantile range is the interquartile range (25, 75).  A
    !> constant feature receives unit scale, matching the other dense scaler
    !> contracts.  Fit is finite-only and the fitted center and scale are
    !> state, not differentiable parameters.
    type, public :: robust_scaler_t
        private
        real(dp), allocatable :: center_value(:)
        real(dp), allocatable :: scale_value(:)
        real(dp) :: lower_quantile = 25.0_dp
        real(dp) :: upper_quantile = 75.0_dp
        logical :: with_centering = .true.
        logical :: with_scaling = .true.
    contains
        procedure, public :: fit => robust_scaler_fit
        procedure, public :: transform => robust_scaler_transform
        procedure, public :: inverse_transform => robust_scaler_inverse
        procedure, public :: transform_jvp => robust_scaler_transform_jvp
        procedure, public :: centers => robust_scaler_centers
        procedure, public :: scales => robust_scaler_scales
        procedure, public :: feature_count => robust_scaler_feature_count
        procedure, public :: fitted => robust_scaler_fitted
    end type robust_scaler_t

contains

    subroutine standard_scaler_fit(self, x, status, with_mean, with_std)
        class(standard_scaler_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: with_mean, with_std
        real(dp) :: variance
        integer :: j

        if (.not. valid_matrix(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "standard scaler fit: input must be a finite nonempty matrix")
            return
        end if
        self%with_mean = .true.
        if (present(with_mean)) self%with_mean = with_mean
        self%with_std = .true.
        if (present(with_std)) self%with_std = with_std
        allocate(self%mean_value(size(x, 2)), self%scale_value(size(x, 2)))
        self%mean_value = 0.0_dp
        self%scale_value = 1.0_dp
        if (self%with_mean) self%mean_value = sum(x, dim=1)/real(size(x, 1), dp)
        if (self%with_std) then
            do j = 1, size(x, 2)
                variance = sum((x(:, j) - self%mean_value(j))**2)/ &
                    real(size(x, 1), dp)
                if (variance > 0.0_dp) self%scale_value(j) = sqrt(variance)
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine standard_scaler_fit

    subroutine standard_scaler_transform(self, x, transformed, status)
        class(standard_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. standard_scaler_valid(self, x, transformed, status, &
                "standard scaler transform")) return
        transformed = (x - spread(self%mean_value, dim=1, ncopies=size(x, 1)))/ &
            spread(self%scale_value, dim=1, ncopies=size(x, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine standard_scaler_transform

    subroutine standard_scaler_inverse(self, x, transformed, status)
        class(standard_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. standard_scaler_valid(self, x, transformed, status, &
                "standard scaler inverse_transform")) return
        transformed = x*spread(self%scale_value, dim=1, ncopies=size(x, 1)) + &
            spread(self%mean_value, dim=1, ncopies=size(x, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine standard_scaler_inverse

    subroutine standard_scaler_transform_jvp(self, x_dot, transformed_dot, status)
        class(standard_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x_dot(:, :)
        real(dp), intent(out) :: transformed_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. standard_scaler_valid(self, x_dot, transformed_dot, status, &
                "standard scaler JVP")) return
        transformed_dot = x_dot/spread(self%scale_value, dim=1, &
            ncopies=size(x_dot, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine standard_scaler_transform_jvp

    function standard_scaler_means(self) result(values)
        class(standard_scaler_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%mean_value)) then
            values = self%mean_value
        else
            allocate(values(0))
        end if
    end function standard_scaler_means

    function standard_scaler_scales(self) result(values)
        class(standard_scaler_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%scale_value)) then
            values = self%scale_value
        else
            allocate(values(0))
        end if
    end function standard_scaler_scales

    integer function standard_scaler_feature_count(self) result(count)
        class(standard_scaler_t), intent(in) :: self

        count = 0
        if (allocated(self%mean_value)) count = size(self%mean_value)
    end function standard_scaler_feature_count

    logical function standard_scaler_fitted(self) result(value)
        class(standard_scaler_t), intent(in) :: self

        value = allocated(self%mean_value) .and. allocated(self%scale_value)
    end function standard_scaler_fitted

    subroutine minmax_scaler_fit(self, x, status, feature_range)
        class(minmax_scaler_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: feature_range(2)

        if (.not. valid_matrix(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "minmax scaler fit: input must be a finite nonempty matrix")
            return
        end if
        self%lower = 0.0_dp
        self%upper = 1.0_dp
        if (present(feature_range)) then
            if (.not. all(ieee_is_finite(feature_range)) .or. &
                    feature_range(2) <= feature_range(1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "minmax scaler fit: feature range must be finite and increasing")
                return
            end if
            self%lower = feature_range(1)
            self%upper = feature_range(2)
        end if
        allocate(self%data_min(size(x, 2)), self%data_max(size(x, 2)))
        self%data_min = minval(x, dim=1)
        self%data_max = maxval(x, dim=1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine minmax_scaler_fit

    subroutine minmax_scaler_transform(self, x, transformed, status)
        class(minmax_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: span
        real(dp), allocatable :: denominator(:)

        if (.not. minmax_scaler_valid(self, x, transformed, status, &
                "minmax scaler transform")) return
        allocate(denominator(size(self%data_min)))
        denominator = self%data_max - self%data_min
        where (denominator == 0.0_dp) denominator = 1.0_dp
        span = self%upper - self%lower
        transformed = self%lower + span*(x - &
            spread(self%data_min, dim=1, ncopies=size(x, 1)))/ &
            spread(denominator, dim=1, ncopies=size(x, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine minmax_scaler_transform

    subroutine minmax_scaler_inverse(self, x, transformed, status)
        class(minmax_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: span
        real(dp), allocatable :: denominator(:)

        if (.not. minmax_scaler_valid(self, x, transformed, status, &
                "minmax scaler inverse_transform")) return
        allocate(denominator(size(self%data_min)))
        denominator = self%data_max - self%data_min
        where (denominator == 0.0_dp) denominator = 1.0_dp
        span = self%upper - self%lower
        transformed = spread(self%data_min, dim=1, ncopies=size(x, 1)) + &
            (x - self%lower)*spread(denominator, dim=1, ncopies=size(x, 1))/span
        call status_set(status, FORTNUM_OK, "")
    end subroutine minmax_scaler_inverse

    subroutine minmax_scaler_transform_jvp(self, x_dot, transformed_dot, status)
        class(minmax_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x_dot(:, :)
        real(dp), intent(out) :: transformed_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: span
        real(dp), allocatable :: denominator(:)

        if (.not. minmax_scaler_valid(self, x_dot, transformed_dot, status, &
                "minmax scaler JVP")) return
        allocate(denominator(size(self%data_min)))
        denominator = self%data_max - self%data_min
        where (denominator == 0.0_dp) denominator = 1.0_dp
        span = self%upper - self%lower
        transformed_dot = span*x_dot/spread(denominator, dim=1, &
            ncopies=size(x_dot, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine minmax_scaler_transform_jvp

    function minmax_scaler_minimums(self) result(values)
        class(minmax_scaler_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%data_min)) then
            values = self%data_min
        else
            allocate(values(0))
        end if
    end function minmax_scaler_minimums

    function minmax_scaler_maximums(self) result(values)
        class(minmax_scaler_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%data_max)) then
            values = self%data_max
        else
            allocate(values(0))
        end if
    end function minmax_scaler_maximums

    integer function minmax_scaler_feature_count(self) result(count)
        class(minmax_scaler_t), intent(in) :: self

        count = 0
        if (allocated(self%data_min)) count = size(self%data_min)
    end function minmax_scaler_feature_count

    logical function minmax_scaler_fitted(self) result(value)
        class(minmax_scaler_t), intent(in) :: self

        value = allocated(self%data_min) .and. allocated(self%data_max)
    end function minmax_scaler_fitted

    subroutine robust_scaler_fit(self, x, status, with_centering, with_scaling, &
            quantile_range)
        class(robust_scaler_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: with_centering, with_scaling
        real(dp), intent(in), optional :: quantile_range(2)
        real(dp), allocatable :: sorted(:)
        real(dp) :: lower, upper
        integer :: j

        if (.not. valid_matrix(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "robust scaler fit: input must be a finite nonempty matrix")
            return
        end if
        self%with_centering = .true.
        if (present(with_centering)) self%with_centering = with_centering
        self%with_scaling = .true.
        if (present(with_scaling)) self%with_scaling = with_scaling
        self%lower_quantile = 25.0_dp
        self%upper_quantile = 75.0_dp
        if (present(quantile_range)) then
            if (any(.not. ieee_is_finite(quantile_range)) .or. &
                    quantile_range(1) < 0.0_dp .or. &
                    quantile_range(2) > 100.0_dp .or. &
                    quantile_range(2) <= quantile_range(1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "robust scaler fit: quantile range must be finite, ordered, and within [0,100]")
                return
            end if
            self%lower_quantile = quantile_range(1)
            self%upper_quantile = quantile_range(2)
        end if

        allocate(self%center_value(size(x, 2)), self%scale_value(size(x, 2)))
        self%center_value = 0.0_dp
        self%scale_value = 1.0_dp
        allocate(sorted(size(x, 1)))
        do j = 1, size(x, 2)
            sorted = x(:, j)
            call sort_real_values(sorted)
            if (self%with_centering) then
                self%center_value(j) = percentile_value(sorted, 50.0_dp)
            end if
            if (self%with_scaling) then
                lower = percentile_value(sorted, self%lower_quantile)
                upper = percentile_value(sorted, self%upper_quantile)
                if (upper > lower) self%scale_value(j) = upper - lower
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_scaler_fit

    subroutine robust_scaler_transform(self, x, transformed, status)
        class(robust_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. robust_scaler_valid(self, x, transformed, status, &
                "robust scaler transform")) return
        transformed = (x - spread(self%center_value, dim=1, &
            ncopies=size(x, 1)))/spread(self%scale_value, dim=1, &
            ncopies=size(x, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_scaler_transform

    subroutine robust_scaler_inverse(self, x, transformed, status)
        class(robust_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. robust_scaler_valid(self, x, transformed, status, &
                "robust scaler inverse_transform")) return
        transformed = x*spread(self%scale_value, dim=1, ncopies=size(x, 1)) + &
            spread(self%center_value, dim=1, ncopies=size(x, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_scaler_inverse

    subroutine robust_scaler_transform_jvp(self, x_dot, transformed_dot, status)
        class(robust_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x_dot(:, :)
        real(dp), intent(out) :: transformed_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. robust_scaler_valid(self, x_dot, transformed_dot, status, &
                "robust scaler JVP")) return
        transformed_dot = x_dot/spread(self%scale_value, dim=1, &
            ncopies=size(x_dot, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine robust_scaler_transform_jvp

    function robust_scaler_centers(self) result(values)
        class(robust_scaler_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%center_value)) then
            values = self%center_value
        else
            allocate(values(0))
        end if
    end function robust_scaler_centers

    function robust_scaler_scales(self) result(values)
        class(robust_scaler_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%scale_value)) then
            values = self%scale_value
        else
            allocate(values(0))
        end if
    end function robust_scaler_scales

    integer function robust_scaler_feature_count(self) result(count)
        class(robust_scaler_t), intent(in) :: self

        count = 0
        if (allocated(self%center_value)) count = size(self%center_value)
    end function robust_scaler_feature_count

    logical function robust_scaler_fitted(self) result(value)
        class(robust_scaler_t), intent(in) :: self

        value = allocated(self%center_value) .and. allocated(self%scale_value)
    end function robust_scaler_fitted

    logical function robust_scaler_valid(self, x, transformed, status, context) &
            result(value)
        class(robust_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        transformed = 0.0_dp
        value = .false.
        if (.not. robust_scaler_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": model is not fitted")
            return
        end if
        if (.not. valid_matrix(x) .or. any(shape(transformed) /= shape(x)) .or. &
                size(x, 2) /= size(self%center_value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": shape or values are invalid")
            return
        end if
        value = .true.
        call status_set(status, FORTNUM_OK, "")
    end function robust_scaler_valid

    pure real(dp) function percentile_value(sorted, quantile) result(value)
        real(dp), intent(in) :: sorted(:), quantile
        real(dp) :: position, fraction
        integer :: lower_index, upper_index

        if (size(sorted) == 1) then
            value = sorted(1)
            return
        end if
        position = 1.0_dp + (real(size(sorted) - 1, dp)*quantile/100.0_dp)
        lower_index = int(floor(position))
        upper_index = min(lower_index + 1, size(sorted))
        fraction = position - real(lower_index, dp)
        value = (1.0_dp - fraction)*sorted(lower_index) + fraction*sorted(upper_index)
    end function percentile_value

    pure subroutine sort_real_values(values)
        real(dp), intent(inout) :: values(:)
        real(dp) :: key
        integer :: i, j

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
    end subroutine sort_real_values

    logical function valid_matrix(x) result(value)
        real(dp), intent(in) :: x(:, :)

        value = size(x, 1) > 0 .and. size(x, 2) > 0 .and. &
            all(ieee_is_finite(x))
    end function valid_matrix

    logical function standard_scaler_valid(self, x, transformed, status, context) &
            result(value)
        class(standard_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        transformed = 0.0_dp
        value = .false.
        if (.not. standard_scaler_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": model is not fitted")
            return
        end if
        if (.not. valid_matrix(x) .or. any(shape(transformed) /= shape(x)) .or. &
                size(x, 2) /= size(self%mean_value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": shape or values are invalid")
            return
        end if
        value = .true.
        call status_set(status, FORTNUM_OK, "")
    end function standard_scaler_valid

    logical function minmax_scaler_valid(self, x, transformed, status, context) &
            result(value)
        class(minmax_scaler_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        transformed = 0.0_dp
        value = .false.
        if (.not. minmax_scaler_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": model is not fitted")
            return
        end if
        if (.not. valid_matrix(x) .or. any(shape(transformed) /= shape(x)) .or. &
                size(x, 2) /= size(self%data_min)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(context)//": shape or values are invalid")
            return
        end if
        value = .true.
        call status_set(status, FORTNUM_OK, "")
    end function minmax_scaler_valid

end module fortml_preprocessing
