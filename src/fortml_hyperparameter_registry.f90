module fortml_hyperparameter_registry
    !! Transform-aware parameter blocks for optimizer-facing hyperparameters.
    !!
    !! `parameter_registry_t` owns the packing order for model parameters. This
    !! companion registry adds the contracts needed by an outer optimizer:
    !! physical and unconstrained coordinates, per-block transforms and bounds,
    !! trainable filtering, provenance/device labels, and a projected vector
    !! suitable for a bounded FortOpt L-BFGS-B callback. No finite differences
    !! are hidden in this layer; objective code remains responsible for values,
    !! gradients, and HVPs.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_parameter_registry, only: parameter_block_t, &
        parameter_get_proc, parameter_set_proc
    implicit none
    private

    integer, parameter, public :: HYPERPARAMETER_IDENTITY = 1
    integer, parameter, public :: HYPERPARAMETER_LOG = 2
    integer, parameter, public :: HYPERPARAMETER_LOGIT = 3

    ! Short aliases make declarations pleasant while retaining descriptive
    ! names for clients that prefer the long form.
    integer, parameter, public :: HP_TRANSFORM_IDENTITY = HYPERPARAMETER_IDENTITY
    integer, parameter, public :: HP_TRANSFORM_LOG = HYPERPARAMETER_LOG
    integer, parameter, public :: HP_TRANSFORM_LOGIT = HYPERPARAMETER_LOGIT

    type, public :: hyperparameter_block_t
        private
        type(parameter_block_t) :: base
        character(:), allocatable :: block_name
        real(dp), pointer :: owned_values(:) => null()
        real(dp), allocatable :: lower_bound(:)
        real(dp), allocatable :: upper_bound(:)
        character(:), allocatable :: transform_label
        character(:), allocatable :: provenance_name
        character(:), allocatable :: device_name
        integer :: transform_kind = HYPERPARAMETER_IDENTITY
        logical :: trainable_flag = .true.
        logical :: hvp_flag = .false.
    contains
        procedure, public :: initialize => hyperparameter_block_initialize
        procedure, public :: initialize_values => hyperparameter_block_initialize_values
        procedure, public :: initialized => hyperparameter_block_initialized
        procedure, public :: name => hyperparameter_block_name
        procedure, public :: size => hyperparameter_block_size
        procedure, public :: transform => hyperparameter_block_transform
        procedure, public :: transform_name => hyperparameter_block_transform_name
        procedure, public :: lower => hyperparameter_block_lower
        procedure, public :: upper => hyperparameter_block_upper
        procedure, public :: trainable => hyperparameter_block_trainable
        procedure, public :: hvp_available => hyperparameter_block_hvp_available
        procedure, public :: provenance => hyperparameter_block_provenance
        procedure, public :: device => hyperparameter_block_device
        procedure, public :: get_physical => hyperparameter_block_get_physical
        procedure, public :: set_physical => hyperparameter_block_set_physical
        procedure, public :: get_unconstrained => hyperparameter_block_get_unconstrained
        procedure, public :: set_unconstrained => hyperparameter_block_set_unconstrained
        procedure, public :: unconstrained_bounds => hyperparameter_block_unconstrained_bounds
        procedure, public :: project_unconstrained => hyperparameter_block_project
    end type hyperparameter_block_t

    type, public :: hyperparameter_registry_t
        private
        type(hyperparameter_block_t), allocatable :: blocks(:)
        integer :: n_blocks = 0
        integer :: n_parameters = 0
        integer :: n_trainable = 0
    contains
        procedure, public :: clear => hyperparameter_registry_clear
        procedure, public :: add => hyperparameter_registry_add
        procedure, public :: block_count => hyperparameter_registry_block_count
        procedure, public :: parameter_count => hyperparameter_registry_parameter_count
        procedure, public :: trainable_count => hyperparameter_registry_trainable_count
        procedure, public :: pack => hyperparameter_registry_pack
        procedure, public :: unpack => hyperparameter_registry_unpack
        procedure, public :: pack_unconstrained => hyperparameter_registry_pack_unconstrained
        procedure, public :: unpack_unconstrained => hyperparameter_registry_unpack_unconstrained
        procedure, public :: pack_trainable => hyperparameter_registry_pack_trainable
        procedure, public :: unpack_trainable => hyperparameter_registry_unpack_trainable
        procedure, public :: optimizer_bounds => hyperparameter_registry_optimizer_bounds
        procedure, public :: project => hyperparameter_registry_project
        procedure, public :: range => hyperparameter_registry_range
    end type hyperparameter_registry_t

    public :: hyperparameter_transform_name

contains

    subroutine hyperparameter_block_initialize(self, name, n_parameters, context, &
            getter, setter, status, transform, lower, upper, trainable, &
            provenance, device, hvp_available)
        class(hyperparameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        integer, intent(in) :: n_parameters
        class(*), target, intent(inout) :: context
        procedure(parameter_get_proc) :: getter
        procedure(parameter_set_proc) :: setter
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: transform
        real(dp), intent(in), optional :: lower(:), upper(:)
        logical, intent(in), optional :: trainable, hvp_available
        character(*), intent(in), optional :: provenance, device

        call initialize_metadata(self, name, n_parameters, status, transform, &
            lower, upper, trainable, provenance, device, hvp_available)
        if (status%code /= FORTNUM_OK) return
        call self%base%initialize(name, n_parameters, context, getter, setter, status)
    end subroutine hyperparameter_block_initialize

    subroutine hyperparameter_block_initialize_values(self, name, values, status, &
            transform, lower, upper, trainable, provenance, device, hvp_available)
        class(hyperparameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: transform
        real(dp), intent(in), optional :: lower(:), upper(:)
        logical, intent(in), optional :: trainable, hvp_available
        character(*), intent(in), optional :: provenance, device

        call initialize_metadata(self, name, size(values), status, transform, &
            lower, upper, trainable, provenance, device, hvp_available)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: values must be finite")
            return
        end if
        allocate(self%owned_values(size(values)))
        self%owned_values = values
        call self%set_physical(values, status)
    end subroutine hyperparameter_block_initialize_values

    subroutine initialize_metadata(self, name, n_parameters, status, transform, &
            lower, upper, trainable, provenance, device, hvp_available)
        class(hyperparameter_block_t), intent(inout) :: self
        character(*), intent(in) :: name
        integer, intent(in) :: n_parameters
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: transform
        real(dp), intent(in), optional :: lower(:), upper(:)
        logical, intent(in), optional :: trainable, hvp_available
        character(*), intent(in), optional :: provenance, device
        integer :: i, requested_transform

        requested_transform = HYPERPARAMETER_IDENTITY
        if (present(transform)) requested_transform = transform
        if (len_trim(name) == 0 .or. n_parameters < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: name or size is invalid")
            return
        end if
        self%block_name = trim(name)
        if (requested_transform < HYPERPARAMETER_IDENTITY .or. &
            requested_transform > HYPERPARAMETER_LOGIT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: transform is invalid")
            return
        end if
        if (present(lower)) then
            if (size(lower) /= n_parameters) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter block: lower bound shape is invalid")
                return
            end if
        end if
        if (present(upper)) then
            if (size(upper) /= n_parameters) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter block: upper bound shape is invalid")
                return
            end if
        end if
        allocate(self%lower_bound(n_parameters), self%upper_bound(n_parameters))
        self%lower_bound = -huge(1.0_dp)
        self%upper_bound = huge(1.0_dp)
        if (present(lower)) self%lower_bound = lower
        if (present(upper)) self%upper_bound = upper
        if (any(.not. ieee_is_finite(self%lower_bound)) .or. &
            any(.not. ieee_is_finite(self%upper_bound))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: bounds must be finite")
            return
        end if
        do i = 1, n_parameters
            if (self%lower_bound(i) > self%upper_bound(i)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter block: lower bound exceeds upper bound")
                return
            end if
        end do
        if (requested_transform == HYPERPARAMETER_LOG) then
            if (any(self%upper_bound <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter block: log bounds must have positive upper values")
                return
            end if
            if (any(self%lower_bound > self%upper_bound)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter block: log bounds are invalid")
                return
            end if
        else if (requested_transform == HYPERPARAMETER_LOGIT) then
            if (any(self%lower_bound <= -huge(1.0_dp)) .or. &
                any(self%upper_bound >= huge(1.0_dp))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter block: logit requires finite explicit bounds")
                return
            end if
            if (any(self%lower_bound == self%upper_bound)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter block: logit bounds must be distinct")
                return
            end if
        end if
        self%transform_kind = requested_transform
        self%transform_label = hyperparameter_transform_name(requested_transform)
        self%trainable_flag = .true.
        if (present(trainable)) self%trainable_flag = trainable
        self%hvp_flag = .false.
        if (present(hvp_available)) self%hvp_flag = hvp_available
        self%provenance_name = "unspecified"
        if (present(provenance)) self%provenance_name = trim(provenance)
        self%device_name = "cpu"
        if (present(device)) self%device_name = trim(device)
        if (len_trim(self%device_name) == 0 .or. len_trim(self%provenance_name) == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: metadata labels must be nonempty")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine initialize_metadata

    logical function hyperparameter_block_initialized(self) result(yes)
        class(hyperparameter_block_t), intent(in) :: self

        yes = .false.
        if (.not. allocated(self%transform_label)) return
        if (.not. allocated(self%lower_bound)) return
        if (.not. allocated(self%upper_bound)) return
        if (associated(self%owned_values)) then
            yes = size(self%owned_values) > 0 .and. &
                size(self%lower_bound) == size(self%owned_values)
        else
            yes = self%base%initialized() .and. self%base%size() == size(self%lower_bound)
        end if
    end function hyperparameter_block_initialized

    function hyperparameter_block_name(self) result(name)
        class(hyperparameter_block_t), intent(in) :: self
        character(:), allocatable :: name

        name = ""
        if (allocated(self%block_name)) then
            name = self%block_name
        else if (self%base%initialized()) then
            name = self%base%name()
        end if
    end function hyperparameter_block_name

    integer function hyperparameter_block_size(self) result(n)
        class(hyperparameter_block_t), intent(in) :: self

        n = 0
        if (associated(self%owned_values)) then
            n = size(self%owned_values)
        else if (self%base%initialized()) then
            n = self%base%size()
        else if (allocated(self%lower_bound)) then
            n = size(self%lower_bound)
        end if
    end function hyperparameter_block_size

    integer function hyperparameter_block_transform(self) result(kind)
        class(hyperparameter_block_t), intent(in) :: self

        kind = self%transform_kind
    end function hyperparameter_block_transform

    function hyperparameter_block_transform_name(self) result(name)
        class(hyperparameter_block_t), intent(in) :: self
        character(:), allocatable :: name

        name = hyperparameter_transform_name(self%transform_kind)
    end function hyperparameter_block_transform_name

    function hyperparameter_block_lower(self) result(value)
        class(hyperparameter_block_t), intent(in) :: self
        real(dp), allocatable :: value(:)

        if (allocated(self%lower_bound)) then
            value = self%lower_bound
        else
            allocate(value(0))
        end if
    end function hyperparameter_block_lower

    function hyperparameter_block_upper(self) result(value)
        class(hyperparameter_block_t), intent(in) :: self
        real(dp), allocatable :: value(:)

        if (allocated(self%upper_bound)) then
            value = self%upper_bound
        else
            allocate(value(0))
        end if
    end function hyperparameter_block_upper

    logical function hyperparameter_block_trainable(self) result(yes)
        class(hyperparameter_block_t), intent(in) :: self

        yes = self%trainable_flag
    end function hyperparameter_block_trainable

    logical function hyperparameter_block_hvp_available(self) result(yes)
        class(hyperparameter_block_t), intent(in) :: self

        yes = self%hvp_flag
    end function hyperparameter_block_hvp_available

    function hyperparameter_block_provenance(self) result(value)
        class(hyperparameter_block_t), intent(in) :: self
        character(:), allocatable :: value

        value = ""
        if (allocated(self%provenance_name)) value = self%provenance_name
    end function hyperparameter_block_provenance

    function hyperparameter_block_device(self) result(value)
        class(hyperparameter_block_t), intent(in) :: self
        character(:), allocatable :: value

        value = ""
        if (allocated(self%device_name)) value = self%device_name
    end function hyperparameter_block_device

    subroutine hyperparameter_block_get_physical(self, values, status)
        class(hyperparameter_block_t), intent(in) :: self
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: block is not initialized")
            return
        end if
        if (size(values) /= self%size()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: physical output shape is invalid")
            return
        end if
        if (associated(self%owned_values)) then
            values = self%owned_values
        else
            call self%base%get(values, status)
            if (status%code /= FORTNUM_OK) return
        end if
        call validate_physical_values(self, values, status)
    end subroutine hyperparameter_block_get_physical

    subroutine hyperparameter_block_set_physical(self, values, status)
        class(hyperparameter_block_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: block is not initialized")
            return
        end if
        if (size(values) /= self%size()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: physical input shape is invalid")
            return
        end if
        call validate_physical_values(self, values, status)
        if (status%code /= FORTNUM_OK) return
        if (associated(self%owned_values)) then
            self%owned_values = values
        else
            call self%base%set(values, status)
            if (status%code /= FORTNUM_OK) return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine hyperparameter_block_set_physical

    subroutine validate_physical_values(self, values, status)
        class(hyperparameter_block_t), intent(in) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: physical values are not finite")
            return
        end if
        if (any(values < self%lower_bound) .or. any(values > self%upper_bound)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: physical values violate bounds")
            return
        end if
        if (self%transform_kind == HYPERPARAMETER_LOG .and. any(values <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: log values must be positive")
            return
        end if
        if (self%transform_kind == HYPERPARAMETER_LOGIT) then
            if (any(values <= self%lower_bound) .or. any(values >= self%upper_bound)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter block: logit values must be strictly interior")
                return
            end if
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_physical_values

    subroutine hyperparameter_block_get_unconstrained(self, values, status)
        class(hyperparameter_block_t), intent(in) :: self
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: physical(:)

        if (size(values) /= self%size()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: unconstrained output shape is invalid")
            return
        end if
        allocate(physical(size(values)))
        call self%get_physical(physical, status)
        if (status%code /= FORTNUM_OK) return
        call physical_to_unconstrained(self%transform_kind, self%lower_bound, &
            self%upper_bound, physical, values, status)
    end subroutine hyperparameter_block_get_unconstrained

    subroutine hyperparameter_block_set_unconstrained(self, values, status)
        class(hyperparameter_block_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: physical(:)

        if (size(values) /= self%size()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: unconstrained input shape is invalid")
            return
        end if
        allocate(physical(size(values)))
        call unconstrained_to_physical(self%transform_kind, self%lower_bound, &
            self%upper_bound, values, physical, status)
        if (status%code /= FORTNUM_OK) return
        call self%set_physical(physical, status)
    end subroutine hyperparameter_block_set_unconstrained

    subroutine hyperparameter_block_unconstrained_bounds(self, lower, upper, status)
        class(hyperparameter_block_t), intent(in) :: self
        real(dp), intent(out) :: lower(:), upper(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: block is not initialized")
            return
        end if
        if (size(lower) /= self%size() .or. size(upper) /= self%size()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: bound output shape is invalid")
            return
        end if
        do i = 1, self%size()
            select case (self%transform_kind)
            case (HYPERPARAMETER_IDENTITY)
                lower(i) = self%lower_bound(i)
                upper(i) = self%upper_bound(i)
            case (HYPERPARAMETER_LOG)
                lower(i) = -huge(1.0_dp)
                upper(i) = huge(1.0_dp)
                if (self%lower_bound(i) > 0.0_dp) then
                    lower(i) = log(self%lower_bound(i))
                end if
                if (self%upper_bound(i) < huge(1.0_dp)) then
                    upper(i) = log(self%upper_bound(i))
                end if
            case (HYPERPARAMETER_LOGIT)
                lower(i) = -huge(1.0_dp)
                upper(i) = huge(1.0_dp)
            end select
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine hyperparameter_block_unconstrained_bounds

    subroutine hyperparameter_block_project(self, values, status)
        class(hyperparameter_block_t), intent(in) :: self
        real(dp), intent(inout) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: lower(:), upper(:)

        if (size(values) /= self%size()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: project shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter block: projected values must be finite")
            return
        end if
        allocate(lower(size(values)), upper(size(values)))
        call self%unconstrained_bounds(lower, upper, status)
        if (status%code /= FORTNUM_OK) return
        values = max(lower, min(upper, values))
        call status_set(status, FORTNUM_OK, "")
    end subroutine hyperparameter_block_project

    subroutine hyperparameter_registry_clear(self)
        class(hyperparameter_registry_t), intent(inout) :: self

        if (allocated(self%blocks)) deallocate(self%blocks)
        self%n_blocks = 0
        self%n_parameters = 0
        self%n_trainable = 0
    end subroutine hyperparameter_registry_clear

    subroutine hyperparameter_registry_add(self, block, status)
        class(hyperparameter_registry_t), intent(inout) :: self
        type(hyperparameter_block_t), intent(in) :: block
        type(fortnum_status_t), intent(out) :: status
        type(hyperparameter_block_t), allocatable :: extended(:)
        integer :: i

        if (.not. block%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter registry: block is not initialized")
            return
        end if
        do i = 1, self%n_blocks
            if (self%blocks(i)%name() == block%name()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter registry: block names must be unique")
                return
            end if
        end do
        allocate(extended(self%n_blocks + 1))
        if (self%n_blocks > 0) extended(:self%n_blocks) = self%blocks
        extended(self%n_blocks + 1) = block
        call move_alloc(extended, self%blocks)
        self%n_blocks = self%n_blocks + 1
        self%n_parameters = self%n_parameters + block%size()
        if (block%trainable()) self%n_trainable = self%n_trainable + block%size()
        call status_set(status, FORTNUM_OK, "")
    end subroutine hyperparameter_registry_add

    integer function hyperparameter_registry_block_count(self) result(n)
        class(hyperparameter_registry_t), intent(in) :: self

        n = self%n_blocks
    end function hyperparameter_registry_block_count

    integer function hyperparameter_registry_parameter_count(self) result(n)
        class(hyperparameter_registry_t), intent(in) :: self

        n = self%n_parameters
    end function hyperparameter_registry_parameter_count

    integer function hyperparameter_registry_trainable_count(self) result(n)
        class(hyperparameter_registry_t), intent(in) :: self

        n = self%n_trainable
    end function hyperparameter_registry_trainable_count

    subroutine hyperparameter_registry_pack(self, values, status)
        class(hyperparameter_registry_t), intent(in) :: self
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        call pack_all(self, values, status, .false.)
    end subroutine hyperparameter_registry_pack

    subroutine hyperparameter_registry_unpack(self, values, status)
        class(hyperparameter_registry_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        call unpack_all(self, values, status, .false.)
    end subroutine hyperparameter_registry_unpack

    subroutine hyperparameter_registry_pack_unconstrained(self, values, status)
        class(hyperparameter_registry_t), intent(in) :: self
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        call pack_all(self, values, status, .true.)
    end subroutine hyperparameter_registry_pack_unconstrained

    subroutine hyperparameter_registry_unpack_unconstrained(self, values, status)
        class(hyperparameter_registry_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        call unpack_all(self, values, status, .true.)
    end subroutine hyperparameter_registry_unpack_unconstrained

    subroutine hyperparameter_registry_pack_trainable(self, values, status)
        class(hyperparameter_registry_t), intent(in) :: self
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        call pack_trainable_impl(self, values, status, .true.)
    end subroutine hyperparameter_registry_pack_trainable

    subroutine hyperparameter_registry_unpack_trainable(self, values, status)
        class(hyperparameter_registry_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        call unpack_trainable_impl(self, values, status, .true.)
    end subroutine hyperparameter_registry_unpack_trainable

    subroutine hyperparameter_registry_optimizer_bounds(self, lower, upper, status)
        class(hyperparameter_registry_t), intent(in) :: self
        real(dp), intent(out) :: lower(:), upper(:)
        type(fortnum_status_t), intent(out) :: status

        call bounds_trainable_impl(self, lower, upper, status)
    end subroutine hyperparameter_registry_optimizer_bounds

    subroutine hyperparameter_registry_project(self, values, status)
        class(hyperparameter_registry_t), intent(in) :: self
        real(dp), intent(inout) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: lower(:), upper(:)

        if (size(values) /= self%n_trainable) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter registry: project shape is invalid")
            return
        end if
        allocate(lower(size(values)), upper(size(values)))
        call self%optimizer_bounds(lower, upper, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter registry: projected values must be finite")
            return
        end if
        values = max(lower, min(upper, values))
        call status_set(status, FORTNUM_OK, "")
    end subroutine hyperparameter_registry_project

    subroutine hyperparameter_registry_range(self, name, first, last, found, trainable_only)
        class(hyperparameter_registry_t), intent(in) :: self
        character(*), intent(in) :: name
        integer, intent(out) :: first, last
        logical, intent(out) :: found
        logical, intent(in), optional :: trainable_only
        logical :: only_trainable
        integer :: i

        only_trainable = .false.
        if (present(trainable_only)) only_trainable = trainable_only
        first = 1
        found = .false.
        do i = 1, self%n_blocks
            if (.not. only_trainable .or. self%blocks(i)%trainable()) then
                last = first + self%blocks(i)%size() - 1
                if (self%blocks(i)%name() == trim(name)) then
                    found = .true.
                    return
                end if
                first = last + 1
            end if
        end do
        first = 0
        last = -1
    end subroutine hyperparameter_registry_range

    subroutine pack_all(self, values, status, unconstrained)
        class(hyperparameter_registry_t), intent(in) :: self
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in) :: unconstrained
        integer :: i, first, last

        if (size(values) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter registry: packed output shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_blocks
            last = first + self%blocks(i)%size() - 1
            if (unconstrained) then
                call self%blocks(i)%get_unconstrained(values(first:last), status)
            else
                call self%blocks(i)%get_physical(values(first:last), status)
            end if
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pack_all

    subroutine unpack_all(self, values, status, unconstrained)
        class(hyperparameter_registry_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in) :: unconstrained
        integer :: i, first, last

        if (size(values) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter registry: packed input shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_blocks
            last = first + self%blocks(i)%size() - 1
            if (unconstrained) then
                call self%blocks(i)%set_unconstrained(values(first:last), status)
            else
                call self%blocks(i)%set_physical(values(first:last), status)
            end if
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine unpack_all

    subroutine pack_trainable_impl(self, values, status, unconstrained)
        class(hyperparameter_registry_t), intent(in) :: self
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in) :: unconstrained
        integer :: i, first, last

        if (size(values) /= self%n_trainable) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter registry: optimizer output shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_blocks
            if (.not. self%blocks(i)%trainable()) cycle
            last = first + self%blocks(i)%size() - 1
            if (unconstrained) then
                call self%blocks(i)%get_unconstrained(values(first:last), status)
            else
                call self%blocks(i)%get_physical(values(first:last), status)
            end if
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pack_trainable_impl

    subroutine unpack_trainable_impl(self, values, status, unconstrained)
        class(hyperparameter_registry_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in) :: unconstrained
        integer :: i, first, last

        if (size(values) /= self%n_trainable) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter registry: optimizer input shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_blocks
            if (.not. self%blocks(i)%trainable()) cycle
            last = first + self%blocks(i)%size() - 1
            if (unconstrained) then
                call self%blocks(i)%set_unconstrained(values(first:last), status)
            else
                call self%blocks(i)%set_physical(values(first:last), status)
            end if
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine unpack_trainable_impl

    subroutine bounds_trainable_impl(self, lower, upper, status)
        class(hyperparameter_registry_t), intent(in) :: self
        real(dp), intent(out) :: lower(:), upper(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last

        if (size(lower) /= self%n_trainable .or. size(upper) /= self%n_trainable) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter registry: optimizer bound shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_blocks
            if (.not. self%blocks(i)%trainable()) cycle
            last = first + self%blocks(i)%size() - 1
            call self%blocks(i)%unconstrained_bounds(lower(first:last), &
                upper(first:last), status)
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bounds_trainable_impl

    function hyperparameter_transform_name(kind) result(name)
        integer, intent(in) :: kind
        character(:), allocatable :: name

        select case (kind)
        case (HYPERPARAMETER_IDENTITY)
            name = "identity"
        case (HYPERPARAMETER_LOG)
            name = "log"
        case (HYPERPARAMETER_LOGIT)
            name = "logit"
        case default
            name = "invalid"
        end select
    end function hyperparameter_transform_name

    subroutine physical_to_unconstrained(kind, lower, upper, physical, unconstrained, status)
        integer, intent(in) :: kind
        real(dp), intent(in) :: lower(:), upper(:), physical(:)
        real(dp), intent(out) :: unconstrained(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: scaled
        integer :: i

        if (size(lower) /= size(physical) .or. size(upper) /= size(physical) .or. &
            size(unconstrained) /= size(physical)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter transform: shape is invalid")
            return
        end if
        do i = 1, size(physical)
            select case (kind)
            case (HYPERPARAMETER_IDENTITY)
                unconstrained(i) = physical(i)
            case (HYPERPARAMETER_LOG)
                if (physical(i) <= 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "hyperparameter transform: log requires positive values")
                    return
                end if
                unconstrained(i) = log(physical(i))
            case (HYPERPARAMETER_LOGIT)
                if (physical(i) <= lower(i) .or. physical(i) >= upper(i)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "hyperparameter transform: logit requires interior values")
                    return
                end if
                scaled = (physical(i)-lower(i))/(upper(i)-physical(i))
                unconstrained(i) = log(scaled)
            end select
        end do
        if (any(.not. ieee_is_finite(unconstrained))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter transform: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine physical_to_unconstrained

    subroutine unconstrained_to_physical(kind, lower, upper, unconstrained, physical, status)
        integer, intent(in) :: kind
        real(dp), intent(in) :: lower(:), upper(:), unconstrained(:)
        real(dp), intent(out) :: physical(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: logistic
        integer :: i

        if (size(lower) /= size(unconstrained) .or. size(upper) /= size(unconstrained) .or. &
            size(physical) /= size(unconstrained)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter transform: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(unconstrained))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter transform: input is not finite")
            return
        end if
        do i = 1, size(unconstrained)
            select case (kind)
            case (HYPERPARAMETER_IDENTITY)
                physical(i) = unconstrained(i)
            case (HYPERPARAMETER_LOG)
                physical(i) = exp(unconstrained(i))
            case (HYPERPARAMETER_LOGIT)
                if (unconstrained(i) >= 0.0_dp) then
                    logistic = 1.0_dp/(1.0_dp + exp(-unconstrained(i)))
                else
                    logistic = exp(unconstrained(i))/(1.0_dp + exp(unconstrained(i)))
                end if
                physical(i) = lower(i) + (upper(i)-lower(i))*logistic
            end select
        end do
        if (any(.not. ieee_is_finite(physical))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter transform: physical result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine unconstrained_to_physical

end module fortml_hyperparameter_registry
