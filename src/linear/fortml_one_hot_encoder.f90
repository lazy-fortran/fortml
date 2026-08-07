!> Deterministic one-hot encoding for integer categorical columns.
module fortml_one_hot_encoder
    !! `one_hot_encoder_t` is the categorical counterpart to the numerical
    !! preprocessing transformers.  Samples are rows and categorical columns
    !! are represented by integer codes.  Categories are sorted independently
    !! for every feature and the fitted order is retained in packed metadata.
    !! Unknown values can be refused or mapped to an all-zero block.  A caller
    !! may designate one integer as missing and choose refusal, all-zero, or
    !! explicit-category handling for it.
    !!
    !! Integer categorical data has no canonical tangent space.  The JVP/VJP
    !! entry points therefore validate shapes and return
    !! `FORTNUM_NOT_IMPLEMENTED`; a derivative is never silently reported as
    !! zero.  This makes categorical derivative boundaries explicit when an
    !! encoder is used inside a differentiable pipeline.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    integer, parameter, public :: ONE_HOT_UNKNOWN_ERROR = 1
    integer, parameter, public :: ONE_HOT_UNKNOWN_IGNORE = 2
    integer, parameter, public :: ONE_HOT_MISSING_ERROR = 1
    integer, parameter, public :: ONE_HOT_MISSING_IGNORE = 2
    integer, parameter, public :: ONE_HOT_MISSING_CATEGORY = 3

    !> Fitted one-hot encoder with packed category and output offsets.
    type, public :: one_hot_encoder_t
        private
        integer, allocatable :: category_values(:)
        integer, allocatable :: category_offset(:)
        integer, allocatable :: output_offset(:)
        integer :: n_features = 0
        integer :: unknown_code = ONE_HOT_UNKNOWN_ERROR
        integer :: missing_code = ONE_HOT_MISSING_ERROR
        integer :: missing_value = 0
        logical :: has_missing_value = .false.
        logical :: drop_first = .false.
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => one_hot_fit
        procedure, public :: transform => one_hot_transform
        procedure, public :: transform_jvp => one_hot_transform_jvp
        procedure, public :: transform_vjp => one_hot_transform_vjp
        procedure, public :: categories => one_hot_categories
        procedure, public :: category_offsets => one_hot_category_offsets
        procedure, public :: output_offsets => one_hot_output_offsets
        procedure, public :: feature_category_count => &
            one_hot_feature_category_count
        procedure, public :: feature_output_count => one_hot_feature_output_count
        procedure, public :: feature_count => one_hot_feature_count
        procedure, public :: output_count => one_hot_output_count
        procedure, public :: unknown_policy => one_hot_unknown_policy
        procedure, public :: missing_policy => one_hot_missing_policy
        procedure, public :: missing_sentinel => one_hot_missing_sentinel
        procedure, public :: drop_first_category => one_hot_drop_first
        procedure, public :: fitted => one_hot_fitted
    end type one_hot_encoder_t

    public :: one_hot_fit
    public :: one_hot_transform

contains

    !> Fit sorted categories for each integer input column.
    subroutine one_hot_fit(self, x, status, handle_unknown, missing_value, &
            handle_missing, drop_first)
        class(one_hot_encoder_t), intent(out) :: self
        integer, intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: handle_unknown, handle_missing
        integer, intent(in), optional :: missing_value
        logical, intent(in), optional :: drop_first
        integer :: i, j, total_categories, n_categories
        integer :: unknown_code, missing_code, requested_missing
        integer, allocatable :: local_categories(:)
        integer, allocatable :: counts(:)
        logical :: has_missing, requested_drop

        unknown_code = ONE_HOT_UNKNOWN_ERROR
        if (present(handle_unknown)) then
            select case (trim(adjustl(handle_unknown)))
            case ("error")
                unknown_code = ONE_HOT_UNKNOWN_ERROR
            case ("ignore")
                unknown_code = ONE_HOT_UNKNOWN_IGNORE
            case default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "one-hot fit: handle_unknown must be error or ignore")
                return
            end select
        end if

        has_missing = present(missing_value)
        requested_missing = 0
        if (has_missing) requested_missing = missing_value
        missing_code = ONE_HOT_MISSING_ERROR
        if (present(handle_missing)) then
            select case (trim(adjustl(handle_missing)))
            case ("error")
                missing_code = ONE_HOT_MISSING_ERROR
            case ("ignore")
                missing_code = ONE_HOT_MISSING_IGNORE
            case ("category")
                missing_code = ONE_HOT_MISSING_CATEGORY
            case default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "one-hot fit: handle_missing must be error, ignore, or category")
                return
            end select
        else if (.not. has_missing) then
            missing_code = ONE_HOT_MISSING_IGNORE
        end if
        if (missing_code == ONE_HOT_MISSING_CATEGORY .and. .not. has_missing) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-hot fit: category missing policy needs missing_value")
            return
        end if

        requested_drop = .false.
        if (present(drop_first)) requested_drop = drop_first
        if (size(x, 1) < 1 .or. size(x, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-hot fit: input must be a nonempty integer matrix")
            return
        end if

        allocate(counts(size(x, 2)))
        counts = 0
        do j = 1, size(x, 2)
            call collect_categories(x(:, j), has_missing, requested_missing, &
                missing_code, local_categories, status)
            if (status%code /= FORTNUM_OK) return
            n_categories = size(local_categories)
            if (n_categories < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "one-hot fit: every feature needs at least one category")
                return
            end if
            counts(j) = n_categories
            deallocate(local_categories)
        end do

        self%n_features = size(x, 2)
        self%unknown_code = unknown_code
        self%missing_code = missing_code
        self%missing_value = requested_missing
        self%has_missing_value = has_missing
        self%drop_first = requested_drop
        allocate(self%category_offset(self%n_features + 1), &
            self%output_offset(self%n_features + 1))
        self%category_offset(1) = 1
        self%output_offset(1) = 1
        do j = 1, self%n_features
            self%category_offset(j + 1) = self%category_offset(j) + counts(j)
            self%output_offset(j + 1) = self%output_offset(j) + counts(j) - &
                merge(1, 0, requested_drop)
        end do
        total_categories = self%category_offset(self%n_features + 1) - 1
        allocate(self%category_values(total_categories))
        do j = 1, self%n_features
            call collect_categories(x(:, j), has_missing, requested_missing, &
                missing_code, local_categories, status)
            if (status%code /= FORTNUM_OK) return
            self%category_values(self%category_offset(j): &
                self%category_offset(j + 1) - 1) = local_categories
            deallocate(local_categories)
        end do
        deallocate(counts)
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine one_hot_fit

    !> Transform integer categories into a dense one-hot matrix.
    subroutine one_hot_transform(self, x, transformed, status)
        class(one_hot_encoder_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, category_index, output_index
        logical :: is_missing

        if (.not. valid_transform_shapes(self, x, transformed, status, &
            "one-hot transform")) return
        transformed = 0.0_dp
        do j = 1, self%n_features
            do i = 1, size(x, 1)
                is_missing = self%has_missing_value .and. &
                    x(i, j) == self%missing_value
                if (is_missing .and. self%missing_code == ONE_HOT_MISSING_ERROR) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "one-hot transform: missing category encountered")
                    return
                end if
                if (is_missing .and. self%missing_code == ONE_HOT_MISSING_IGNORE) cycle
                category_index = find_category(self, j, x(i, j))
                if (category_index < 1) then
                    if (self%unknown_code == ONE_HOT_UNKNOWN_ERROR) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "one-hot transform: unknown category encountered")
                        return
                    end if
                    cycle
                end if
                if (self%drop_first .and. category_index == 1) cycle
                output_index = self%output_offset(j) + category_index - 1 - &
                    merge(1, 0, self%drop_first)
                transformed(i, output_index) = 1.0_dp
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine one_hot_transform

    !> Refuse an input JVP: integer categories have no canonical tangent space.
    subroutine one_hot_transform_jvp(self, x, x_dot, transformed_dot, status)
        class(one_hot_encoder_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        real(dp), intent(in) :: x_dot(:, :)
        real(dp), intent(out) :: transformed_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        transformed_dot = 0.0_dp
        if (.not. valid_jvp_shapes(self, x, x_dot, transformed_dot, status, &
            "one-hot JVP")) return
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "one-hot JVP: categorical integer input has no tangent space")
    end subroutine one_hot_transform_jvp

    !> Refuse an input VJP for the same categorical derivative boundary.
    subroutine one_hot_transform_vjp(self, x, output_bar, input_bar, status)
        class(one_hot_encoder_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        real(dp), intent(in) :: output_bar(:, :)
        real(dp), intent(out) :: input_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        input_bar = 0.0_dp
        if (.not. valid_vjp_shapes(self, x, output_bar, input_bar, status, &
            "one-hot VJP")) return
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "one-hot VJP: categorical integer input has no cotangent space")
    end subroutine one_hot_transform_vjp

    function one_hot_categories(self) result(values)
        class(one_hot_encoder_t), intent(in) :: self
        integer, allocatable :: values(:)

        if (allocated(self%category_values)) then
            values = self%category_values
        else
            allocate(values(0))
        end if
    end function one_hot_categories

    function one_hot_category_offsets(self) result(values)
        class(one_hot_encoder_t), intent(in) :: self
        integer, allocatable :: values(:)

        if (allocated(self%category_offset)) then
            values = self%category_offset
        else
            allocate(values(0))
        end if
    end function one_hot_category_offsets

    function one_hot_output_offsets(self) result(values)
        class(one_hot_encoder_t), intent(in) :: self
        integer, allocatable :: values(:)

        if (allocated(self%output_offset)) then
            values = self%output_offset
        else
            allocate(values(0))
        end if
    end function one_hot_output_offsets

    integer function one_hot_feature_category_count(self, feature) result(count)
        class(one_hot_encoder_t), intent(in) :: self
        integer, intent(in) :: feature

        count = 0
        if (.not. self%is_fitted) return
        if (feature < 1 .or. feature > self%n_features) return
        count = self%category_offset(feature + 1) - self%category_offset(feature)
    end function one_hot_feature_category_count

    integer function one_hot_feature_output_count(self, feature) result(count)
        class(one_hot_encoder_t), intent(in) :: self
        integer, intent(in) :: feature

        count = 0
        if (.not. self%is_fitted) return
        if (feature < 1 .or. feature > self%n_features) return
        count = self%output_offset(feature + 1) - self%output_offset(feature)
    end function one_hot_feature_output_count

    integer function one_hot_feature_count(self) result(count)
        class(one_hot_encoder_t), intent(in) :: self

        count = self%n_features
        if (.not. self%is_fitted) count = 0
    end function one_hot_feature_count

    integer function one_hot_output_count(self) result(count)
        class(one_hot_encoder_t), intent(in) :: self

        count = 0
        if (self%is_fitted) count = self%output_offset(self%n_features + 1) - 1
    end function one_hot_output_count

    integer function one_hot_unknown_policy(self) result(code)
        class(one_hot_encoder_t), intent(in) :: self

        code = self%unknown_code
    end function one_hot_unknown_policy

    integer function one_hot_missing_policy(self) result(code)
        class(one_hot_encoder_t), intent(in) :: self

        code = self%missing_code
    end function one_hot_missing_policy

    integer function one_hot_missing_sentinel(self) result(value)
        class(one_hot_encoder_t), intent(in) :: self

        value = self%missing_value
    end function one_hot_missing_sentinel

    logical function one_hot_drop_first(self) result(value)
        class(one_hot_encoder_t), intent(in) :: self

        value = self%drop_first
    end function one_hot_drop_first

    logical function one_hot_fitted(self) result(value)
        class(one_hot_encoder_t), intent(in) :: self

        value = self%is_fitted .and. allocated(self%category_values) .and. &
            allocated(self%category_offset) .and. allocated(self%output_offset)
    end function one_hot_fitted

    subroutine collect_categories(values, has_missing, missing_value, missing_code, &
            unique, status)
        integer, intent(in) :: values(:)
        logical, intent(in) :: has_missing
        integer, intent(in) :: missing_value, missing_code
        integer, allocatable, intent(out) :: unique(:)
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: work(:)
        integer :: i, count, n_unique

        allocate(work(size(values) + merge(1, 0, &
            has_missing .and. missing_code == ONE_HOT_MISSING_CATEGORY)))
        count = 0
        do i = 1, size(values)
            if (has_missing .and. values(i) == missing_value) then
                if (missing_code == ONE_HOT_MISSING_ERROR) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "one-hot fit: missing category encountered")
                    return
                end if
                if (missing_code == ONE_HOT_MISSING_IGNORE) cycle
            end if
            count = count + 1
            work(count) = values(i)
        end do
        if (has_missing .and. missing_code == ONE_HOT_MISSING_CATEGORY) then
            count = count + 1
            work(count) = missing_value
        end if
        if (count < 1) then
            allocate(unique(0))
            deallocate(work)
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call sort_integers(work(1:count))
        n_unique = 1
        do i = 2, count
            if (work(i) /= work(n_unique)) then
                n_unique = n_unique + 1
                work(n_unique) = work(i)
            end if
        end do
        allocate(unique(n_unique))
        unique = work(1:n_unique)
        deallocate(work)
        call status_set(status, FORTNUM_OK, "")
    end subroutine collect_categories

    subroutine sort_integers(values)
        integer, intent(inout) :: values(:)
        integer :: i, j, key

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
    end subroutine sort_integers

    integer function find_category(self, feature, value) result(index)
        class(one_hot_encoder_t), intent(in) :: self
        integer, intent(in) :: feature, value
        integer :: i, first, last

        index = 0
        first = self%category_offset(feature)
        last = self%category_offset(feature + 1) - 1
        do i = first, last
            if (self%category_values(i) == value) then
                index = i - first + 1
                return
            end if
        end do
    end function find_category

    logical function valid_transform_shapes(self, x, transformed, status, context) &
            result(valid)
        class(one_hot_encoder_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        real(dp), intent(in) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        valid = one_hot_fitted(self) .and. size(x, 1) > 0 .and. &
            size(x, 2) == self%n_features .and. size(transformed, 1) == size(x, 1) &
            .and. size(transformed, 2) == one_hot_output_count(self)
        if (.not. valid) call status_set(status, FORTNUM_DOMAIN_ERROR, &
            trim(context)//": model or array shape is invalid")
    end function valid_transform_shapes

    logical function valid_jvp_shapes(self, x, x_dot, transformed_dot, status, &
            context) result(valid)
        class(one_hot_encoder_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        real(dp), intent(in) :: x_dot(:, :), transformed_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        valid = one_hot_fitted(self) .and. size(x, 1) > 0 .and. &
            size(x, 2) == self%n_features .and. size(x_dot, 1) == size(x, 1) .and. &
            size(x_dot, 2) == size(x, 2) .and. &
            size(transformed_dot, 1) == size(x, 1) .and. &
            size(transformed_dot, 2) == one_hot_output_count(self)
        if (.not. valid) call status_set(status, FORTNUM_DOMAIN_ERROR, &
            trim(context)//": model or array shape is invalid")
    end function valid_jvp_shapes

    logical function valid_vjp_shapes(self, x, output_bar, input_bar, status, &
            context) result(valid)
        class(one_hot_encoder_t), intent(in) :: self
        integer, intent(in) :: x(:, :)
        real(dp), intent(in) :: output_bar(:, :), input_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context

        valid = one_hot_fitted(self) .and. size(x, 1) > 0 .and. &
            size(x, 2) == self%n_features .and. size(output_bar, 1) == size(x, 1) &
            .and. size(output_bar, 2) == one_hot_output_count(self) .and. &
            size(input_bar, 1) == size(x, 1) .and. size(input_bar, 2) == &
            self%n_features
        if (.not. valid) call status_set(status, FORTNUM_DOMAIN_ERROR, &
            trim(context)//": model or array shape is invalid")
    end function valid_vjp_shapes

end module fortml_one_hot_encoder
