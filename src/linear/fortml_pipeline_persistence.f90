!> Transactional, versioned persistence for fitted horizontal basis pipelines.
module fortml_pipeline_persistence
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_pipeline, only: basis_pipeline_t
    implicit none
    private

    integer, parameter, public :: FORTML_PIPELINE_STATE_VERSION = 1
    integer, parameter :: STATE_NAME_LENGTH = 128

    !> A dictionary-like snapshot of a fitted `basis_pipeline_t`.
    !>
    !> The snapshot deliberately stores derived names and one-based offsets in
    !> addition to the packed parameters.  A loader therefore validates the
    !> complete routing contract before changing a target pipeline.  The
    !> current format is host text; device requests return a typed refusal
    !> rather than hiding a host round trip.
    type, public :: basis_pipeline_state_t
        integer :: version = 0
        integer :: n_inputs = 0
        integer :: n_stages = 0
        integer :: n_features = 0
        integer :: n_parameters = 0
        logical :: fitted = .false.
        character(len=STATE_NAME_LENGTH), allocatable :: input_names(:)
        character(len=STATE_NAME_LENGTH), allocatable :: stage_names(:)
        character(len=STATE_NAME_LENGTH), allocatable :: feature_names(:)
        character(len=STATE_NAME_LENGTH), allocatable :: parameter_names(:)
        integer, allocatable :: feature_offsets(:)
        integer, allocatable :: parameter_offsets(:)
        real(dp), allocatable :: parameters(:)
    contains
        procedure, public :: clear => pipeline_state_clear
        procedure, public :: valid => pipeline_state_valid
    end type basis_pipeline_state_t

    public :: capture_basis_pipeline_state
    public :: restore_basis_pipeline_state
    public :: save_basis_pipeline_text
    public :: load_basis_pipeline_text
    public :: save_basis_pipeline_device
    public :: load_basis_pipeline_device

contains

    !> Capture a complete fitted pipeline snapshot without mutating the model.
    subroutine capture_basis_pipeline_state(pipeline, state, status)
        type(basis_pipeline_t), intent(in) :: pipeline
        type(basis_pipeline_state_t), intent(out) :: state
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        call state%clear()
        if (.not. pipeline%valid() .or. .not. pipeline%is_fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state capture: fitted pipeline is required")
            return
        end if

        state%version = FORTML_PIPELINE_STATE_VERSION
        state%n_inputs = pipeline%input_count()
        state%n_stages = pipeline%stage_count()
        state%n_features = pipeline%feature_count()
        state%n_parameters = pipeline%parameter_count()
        state%fitted = pipeline%is_fitted()
        allocate(state%input_names(state%n_inputs))
        allocate(state%stage_names(state%n_stages))
        allocate(state%feature_names(state%n_features))
        allocate(state%parameter_names(state%n_parameters))
        allocate(state%feature_offsets(state%n_stages))
        allocate(state%parameter_offsets(state%n_stages))
        state%parameters = pipeline%parameters()

        do i = 1, state%n_inputs
            state%input_names(i) = pipeline%input_schema_name(i)
        end do
        do i = 1, state%n_stages
            state%stage_names(i) = pipeline%stage_name(i)
            state%feature_offsets(i) = pipeline%stage_feature_offset(i)
            state%parameter_offsets(i) = pipeline%stage_parameter_offset(i)
        end do
        do i = 1, state%n_features
            state%feature_names(i) = pipeline%feature_name(i)
        end do
        do i = 1, state%n_parameters
            state%parameter_names(i) = pipeline%parameter_name(i)
        end do
        if (.not. state%valid()) then
            call state%clear()
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state capture: generated snapshot is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine capture_basis_pipeline_state

    !> Restore a snapshot atomically onto a preconfigured, fitted pipeline.
    !>
    !> A pipeline is preconfigured because basis maps contain their structural
    !> descriptors (polynomial degree, knots, frequencies, and so on).  The
    !> state file restores the fitted values and metadata, not executable
    !> callback procedures.  All shape, name, offset, and finiteness checks run
    !> before a candidate copy is committed to the caller's object.
    subroutine restore_basis_pipeline_state(pipeline, state, status)
        type(basis_pipeline_t), intent(inout) :: pipeline
        type(basis_pipeline_state_t), intent(in) :: state
        type(fortnum_status_t), intent(out) :: status
        type(basis_pipeline_t) :: candidate
        type(fortnum_status_t) :: local_status
        integer :: i

        if (.not. state%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state restore: snapshot is invalid")
            return
        end if
        if (.not. pipeline%valid() .or. .not. pipeline%is_fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state restore: fitted target is required")
            return
        end if
        if (pipeline%input_count() /= state%n_inputs .or. &
            pipeline%stage_count() /= state%n_stages .or. &
            pipeline%feature_count() /= state%n_features .or. &
            pipeline%parameter_count() /= state%n_parameters .or. &
            pipeline%is_fitted() .neqv. state%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state restore: target shape or fit state differs")
            return
        end if
        do i = 1, state%n_inputs
            if (pipeline%input_schema_name(i) /= trim(state%input_names(i))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "pipeline state restore: input schema differs")
                return
            end if
        end do
        do i = 1, state%n_stages
            if (pipeline%stage_name(i) /= trim(state%stage_names(i)) .or. &
                pipeline%stage_feature_offset(i) /= state%feature_offsets(i) .or. &
                pipeline%stage_parameter_offset(i) /= state%parameter_offsets(i)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "pipeline state restore: stage metadata differs")
                return
            end if
        end do
        do i = 1, state%n_features
            if (pipeline%feature_name(i) /= trim(state%feature_names(i))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "pipeline state restore: feature metadata differs")
                return
            end if
        end do
        do i = 1, state%n_parameters
            if (pipeline%parameter_name(i) /= trim(state%parameter_names(i))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "pipeline state restore: parameter metadata differs")
                return
            end if
        end do

        ! Intrinsic assignment deep-copies the allocatable polymorphic basis
        ! implementations.  Mutating only this candidate makes an invalid
        ! parameter update unable to partially alter the caller's model.
        candidate = pipeline
        call candidate%set_parameters(state%parameters, local_status)
        if (local_status%code /= FORTNUM_OK) then
            call status_set(status, local_status%code, &
                "pipeline state restore: parameter update failed")
            return
        end if
        pipeline = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine restore_basis_pipeline_state

    !> Save a fitted pipeline dictionary to the versioned host text format.
    subroutine save_basis_pipeline_text(pipeline, path, status)
        type(basis_pipeline_t), intent(in) :: pipeline
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        type(basis_pipeline_state_t) :: state
        type(fortnum_status_t) :: local_status
        integer :: unit, ios, i
        character(len=256) :: iomsg

        call capture_basis_pipeline_state(pipeline, state, local_status)
        if (local_status%code /= FORTNUM_OK) then
            call status_set(status, local_status%code, &
                "pipeline state save: capture failed")
            return
        end if
        open (newunit=unit, file=trim(path), status="replace", action="write", &
            form="formatted", iostat=ios, iomsg=iomsg)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state save: cannot open text path")
            return
        end if
        write (unit, '(a,i0)', iostat=ios) "FORTML_BASIS_PIPELINE_STATE ", state%version
        write (unit, '(a,4(1x,i0))', iostat=ios) "counts", state%n_inputs, &
            state%n_stages, state%n_features, state%n_parameters
        write (unit, '(a,1x,i0)', iostat=ios) "fitted", merge(1, 0, state%fitted)
        write (unit, '(a,1x,i0)', iostat=ios) "input_names", state%n_inputs
        do i = 1, state%n_inputs
            write (unit, '(a)', iostat=ios) trim(state%input_names(i))
        end do
        write (unit, '(a,1x,i0)', iostat=ios) "stage_names", state%n_stages
        do i = 1, state%n_stages
            write (unit, '(a)', iostat=ios) trim(state%stage_names(i))
        end do
        write (unit, '(a,1x,i0)', iostat=ios) "feature_names", state%n_features
        do i = 1, state%n_features
            write (unit, '(a)', iostat=ios) trim(state%feature_names(i))
        end do
        write (unit, '(a,1x,i0)', iostat=ios) "parameter_names", state%n_parameters
        do i = 1, state%n_parameters
            write (unit, '(a)', iostat=ios) trim(state%parameter_names(i))
        end do
        write (unit, '(a,1x,i0)', iostat=ios) "feature_offsets", state%n_stages
        do i = 1, state%n_stages
            write (unit, '(i0)', iostat=ios) state%feature_offsets(i)
        end do
        write (unit, '(a,1x,i0)', iostat=ios) "parameter_offsets", state%n_stages
        do i = 1, state%n_stages
            write (unit, '(i0)', iostat=ios) state%parameter_offsets(i)
        end do
        write (unit, '(a,1x,i0)', iostat=ios) "parameters", state%n_parameters
        do i = 1, state%n_parameters
            write (unit, '(es26.18e3)', iostat=ios) state%parameters(i)
        end do
        write (unit, '(a)', iostat=ios) "end"
        close (unit)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state save: text write failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine save_basis_pipeline_text

    !> Load the text format into a dictionary, then restore it transactionally.
    subroutine load_basis_pipeline_text(pipeline, path, status)
        type(basis_pipeline_t), intent(inout) :: pipeline
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        type(basis_pipeline_state_t) :: state
        type(fortnum_status_t) :: local_status
        integer :: unit, ios, i, version, n_inputs, n_stages, n_features
        integer :: n_parameters, fitted_int, value
        character(len=512) :: line, tag
        character(len=256) :: iomsg
        real(dp) :: real_value

        call state%clear()
        open (newunit=unit, file=trim(path), status="old", action="read", &
            form="formatted", iostat=ios, iomsg=iomsg)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: cannot open text path")
            return
        end if
        read (unit, '(a)', iostat=ios) line
        if (ios /= 0) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: missing version header")
            return
        end if
        read (line, *, iostat=ios) tag, version
        if (ios /= 0 .or. trim(tag) /= "FORTML_BASIS_PIPELINE_STATE" .or. &
            version /= FORTML_PIPELINE_STATE_VERSION) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: unsupported version header")
            return
        end if
        read (unit, '(a)', iostat=ios) line
        read (line, *, iostat=ios) tag, n_inputs, n_stages, n_features, n_parameters
        if (ios /= 0 .or. trim(tag) /= "counts" .or. n_inputs < 1 .or. &
            n_stages < 1 .or. n_features < 1 .or. n_parameters < 0) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: invalid counts")
            return
        end if
        read (unit, '(a)', iostat=ios) line
        read (line, *, iostat=ios) tag, fitted_int
        if (ios /= 0 .or. trim(tag) /= "fitted" .or. &
            (fitted_int /= 0 .and. fitted_int /= 1)) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: invalid fit flag")
            return
        end if
        state%version = version
        state%n_inputs = n_inputs
        state%n_stages = n_stages
        state%n_features = n_features
        state%n_parameters = n_parameters
        state%fitted = fitted_int == 1
        allocate(state%input_names(n_inputs), state%stage_names(n_stages))
        allocate(state%feature_names(n_features), state%parameter_names(n_parameters))
        allocate(state%feature_offsets(n_stages), state%parameter_offsets(n_stages))
        allocate(state%parameters(n_parameters))

        call read_names(unit, "input_names", state%input_names, ios)
        if (ios /= 0) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: invalid input names")
            return
        end if
        call read_names(unit, "stage_names", state%stage_names, ios)
        if (ios /= 0) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: invalid stage names")
            return
        end if
        call read_names(unit, "feature_names", state%feature_names, ios)
        if (ios /= 0) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: invalid feature names")
            return
        end if
        call read_names(unit, "parameter_names", state%parameter_names, ios)
        if (ios /= 0) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: invalid parameter names")
            return
        end if
        call read_integers(unit, "feature_offsets", state%feature_offsets, ios)
        if (ios /= 0) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: invalid feature offsets")
            return
        end if
        call read_integers(unit, "parameter_offsets", state%parameter_offsets, ios)
        if (ios /= 0) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: invalid parameter offsets")
            return
        end if
        read (unit, '(a)', iostat=ios) line
        read (line, *, iostat=ios) tag, value
        if (ios /= 0 .or. trim(tag) /= "parameters" .or. value /= n_parameters) then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: invalid parameter section")
            return
        end if
        do i = 1, n_parameters
            read (unit, '(a)', iostat=ios) line
            read (line, *, iostat=ios) real_value
            if (ios /= 0 .or. .not. ieee_is_finite(real_value)) then
                close (unit)
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "pipeline state load: nonfinite parameter")
                return
            end if
            state%parameters(i) = real_value
        end do
        read (unit, '(a)', iostat=ios) line
        if (ios /= 0 .or. trim(line) /= "end") then
            close (unit)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: malformed or invalid dictionary")
            return
        end if
        read (unit, '(a)', iostat=ios) line
        close (unit)
        if (ios == 0 .or. .not. state%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: malformed or invalid dictionary")
            return
        end if
        call restore_basis_pipeline_state(pipeline, state, local_status)
        call status_set(status, local_status%code, local_status%msg)
    end subroutine load_basis_pipeline_text

    !> Device-aware save boundary.  Only host text persistence is implemented.
    subroutine save_basis_pipeline_device(pipeline, path, device_kind, status)
        type(basis_pipeline_t), intent(in) :: pipeline
        character(*), intent(in) :: path
        integer, intent(in) :: device_kind
        type(fortnum_status_t), intent(out) :: status

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            call save_basis_pipeline_text(pipeline, path, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "pipeline state save: resident CUDA serialization is unavailable")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state save: device kind is invalid")
        end select
    end subroutine save_basis_pipeline_device

    !> Device-aware load boundary.  CUDA never falls back to a host mutation.
    subroutine load_basis_pipeline_device(pipeline, path, device_kind, status)
        type(basis_pipeline_t), intent(inout) :: pipeline
        character(*), intent(in) :: path
        integer, intent(in) :: device_kind
        type(fortnum_status_t), intent(out) :: status

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            call load_basis_pipeline_text(pipeline, path, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "pipeline state load: resident CUDA deserialization is unavailable")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pipeline state load: device kind is invalid")
        end select
    end subroutine load_basis_pipeline_device

    subroutine pipeline_state_clear(self)
        class(basis_pipeline_state_t), intent(inout) :: self

        self%version = 0
        self%n_inputs = 0
        self%n_stages = 0
        self%n_features = 0
        self%n_parameters = 0
        self%fitted = .false.
        if (allocated(self%input_names)) deallocate(self%input_names)
        if (allocated(self%stage_names)) deallocate(self%stage_names)
        if (allocated(self%feature_names)) deallocate(self%feature_names)
        if (allocated(self%parameter_names)) deallocate(self%parameter_names)
        if (allocated(self%feature_offsets)) deallocate(self%feature_offsets)
        if (allocated(self%parameter_offsets)) deallocate(self%parameter_offsets)
        if (allocated(self%parameters)) deallocate(self%parameters)
    end subroutine pipeline_state_clear

    logical function pipeline_state_valid(self) result(valid)
        class(basis_pipeline_state_t), intent(in) :: self
        integer :: i, j

        valid = self%version == FORTML_PIPELINE_STATE_VERSION .and. &
            self%n_inputs > 0 .and. self%n_stages > 0 .and. &
            self%n_features > 0 .and. self%n_parameters >= 0
        if (.not. valid) return
        valid = allocated(self%input_names) .and. allocated(self%stage_names) .and. &
            allocated(self%feature_names) .and. allocated(self%parameter_names) .and. &
            allocated(self%feature_offsets) .and. allocated(self%parameter_offsets) .and. &
            allocated(self%parameters)
        if (.not. valid) return
        valid = size(self%input_names) == self%n_inputs .and. &
            size(self%stage_names) == self%n_stages .and. &
            size(self%feature_names) == self%n_features .and. &
            size(self%parameter_names) == self%n_parameters .and. &
            size(self%feature_offsets) == self%n_stages .and. &
            size(self%parameter_offsets) == self%n_stages .and. &
            size(self%parameters) == self%n_parameters
        if (.not. valid) return
        if (any(.not. ieee_is_finite(self%parameters))) then
            valid = .false.
            return
        end if
        do i = 1, self%n_inputs
            if (len_trim(self%input_names(i)) < 1 .or. &
                len_trim(self%input_names(i)) > STATE_NAME_LENGTH) then
                valid = .false.
                return
            end if
            do j = 1, i - 1
                if (trim(self%input_names(i)) == trim(self%input_names(j))) then
                    valid = .false.
                    return
                end if
            end do
        end do
        do i = 1, self%n_stages
            if (len_trim(self%stage_names(i)) < 1 .or. &
                self%feature_offsets(i) < 1 .or. &
                self%parameter_offsets(i) < 1) then
                valid = .false.
                return
            end if
            do j = 1, i - 1
                if (trim(self%stage_names(i)) == trim(self%stage_names(j))) then
                    valid = .false.
                    return
                end if
            end do
        end do
        do i = 1, self%n_features
            if (len_trim(self%feature_names(i)) < 1) then
                valid = .false.
                return
            end if
        end do
        do i = 1, self%n_parameters
            if (len_trim(self%parameter_names(i)) < 1) then
                valid = .false.
                return
            end if
        end do
    end function pipeline_state_valid

    subroutine read_names(unit, expected, names, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        character(len=*), intent(out) :: names(:)
        integer, intent(out) :: ios
        character(len=512) :: line, tag
        integer :: count, i

        read (unit, '(a)', iostat=ios) line
        if (ios /= 0) return
        read (line, *, iostat=ios) tag, count
        if (ios /= 0 .or. trim(tag) /= trim(expected) .or. count /= size(names)) then
            ios = 1
            return
        end if
        do i = 1, size(names)
            read (unit, '(a)', iostat=ios) line
            if (ios /= 0) return
            if (len_trim(line) > len(names(i))) then
                ios = 1
                return
            end if
            names(i) = trim(line)
        end do
    end subroutine read_names

    subroutine read_integers(unit, expected, values, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        integer, intent(out) :: values(:)
        integer, intent(out) :: ios
        character(len=512) :: line, tag
        integer :: count, i

        read (unit, '(a)', iostat=ios) line
        if (ios /= 0) return
        read (line, *, iostat=ios) tag, count
        if (ios /= 0 .or. trim(tag) /= trim(expected) .or. count /= size(values)) then
            ios = 1
            return
        end if
        do i = 1, size(values)
            read (unit, '(a)', iostat=ios) line
            if (ios /= 0) return
            read (line, *, iostat=ios) values(i)
            if (ios /= 0) return
        end do
    end subroutine read_integers

end module fortml_pipeline_persistence
