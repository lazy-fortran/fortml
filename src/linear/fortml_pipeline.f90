!> Composable feature maps built from the differentiable basis API.
module fortml_pipeline
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_basis, only: basis_map_t
    implicit none
    private

    integer, parameter :: PIPELINE_NAME_LENGTH = 128

    !> A horizontal pipeline of basis maps.
    !>
    !> Each stage sees the original input matrix and contributes a block of
    !> columns to the output.  This is the useful common denominator for
    !> polynomial/Fourier/spline/RBF feature unions: it is composable without
    !> hiding a stage's parameters and retains exact JVP/VJP products.  A
    !> sequential transform (where one stage consumes the previous stage's
    !> output) is deliberately a separate future abstraction because its
    !> feature shape must be known at construction time.
    type, public :: basis_pipeline_t
        private
        integer :: n_inputs = 0
        integer :: n_stages = 0
        logical :: fitted = .false.
        type(basis_map_t), allocatable :: stages(:)
        character(len=PIPELINE_NAME_LENGTH), allocatable :: stage_names(:)
    contains
        procedure, public :: initialize => pipeline_initialize
        procedure, public :: append => pipeline_append
        procedure, public :: fit => pipeline_fit
        procedure, public :: transform => pipeline_transform
        procedure, public :: evaluate => pipeline_transform
        procedure, public :: jvp => pipeline_jvp
        procedure, public :: vjp => pipeline_vjp
        procedure, public :: hvp => pipeline_hvp
        procedure, public :: input_count => pipeline_input_count
        procedure, public :: stage_count => pipeline_stage_count
        procedure, public :: feature_count => pipeline_feature_count
        procedure, public :: parameter_count => pipeline_parameter_count
        procedure, public :: parameters => pipeline_parameters
        procedure, public :: set_parameters => pipeline_set_parameters
        procedure, public :: stage_name => pipeline_stage_name
        procedure, public :: feature_name => pipeline_feature_name
        procedure, public :: parameter_name => pipeline_parameter_name
        procedure, public :: stage_feature_offset => pipeline_stage_feature_offset
        procedure, public :: stage_parameter_offset => &
            pipeline_stage_parameter_offset
        procedure, public :: static_lowering_eligible => &
            pipeline_static_lowering_eligible
        procedure, public :: valid => pipeline_valid
        procedure, public :: is_fitted => pipeline_is_fitted
    end type basis_pipeline_t

    type :: pipeline_matrix_buffer_t
        real(dp), allocatable :: values(:, :)
    end type pipeline_matrix_buffer_t

    !> A sequential composition of basis maps.
    !>
    !> The output feature block of one stage is the input matrix of the next
    !> stage.  Parameter blocks remain in stage order and JVP/VJP products
    !> propagate through the complete chain.  A stage must be initialized for
    !> exactly the feature count produced by its predecessor.
    type, public :: sequential_basis_pipeline_t
        private
        integer :: n_inputs = 0
        integer :: n_features = 0
        integer :: n_stages = 0
        logical :: fitted = .false.
        type(basis_map_t), allocatable :: stages(:)
        character(len=PIPELINE_NAME_LENGTH), allocatable :: stage_names(:)
    contains
        procedure, public :: initialize => sequential_pipeline_initialize
        procedure, public :: append => sequential_pipeline_append
        procedure, public :: fit => sequential_pipeline_fit
        procedure, public :: transform => sequential_pipeline_transform
        procedure, public :: evaluate => sequential_pipeline_transform
        procedure, public :: jvp => sequential_pipeline_jvp
        procedure, public :: vjp => sequential_pipeline_vjp
        procedure, public :: hvp => sequential_pipeline_hvp
        procedure, public :: input_count => sequential_pipeline_input_count
        procedure, public :: stage_count => sequential_pipeline_stage_count
        procedure, public :: feature_count => sequential_pipeline_feature_count
        procedure, public :: parameter_count => sequential_pipeline_parameter_count
        procedure, public :: parameters => sequential_pipeline_parameters
        procedure, public :: set_parameters => sequential_pipeline_set_parameters
        procedure, public :: stage_name => sequential_pipeline_stage_name
        procedure, public :: feature_name => sequential_pipeline_feature_name
        procedure, public :: parameter_name => sequential_pipeline_parameter_name
        procedure, public :: stage_feature_offset => &
            sequential_pipeline_stage_feature_offset
        procedure, public :: stage_parameter_offset => &
            sequential_pipeline_stage_parameter_offset
        procedure, public :: static_lowering_eligible => &
            sequential_pipeline_static_lowering_eligible
        procedure, public :: valid => sequential_pipeline_valid
        procedure, public :: is_fitted => sequential_pipeline_is_fitted
    end type sequential_basis_pipeline_t

    public :: make_basis_pipeline
    public :: make_sequential_basis_pipeline

contains

    !> Construct an empty pipeline.  Append one or more basis maps before fit.
    function make_basis_pipeline(n_inputs, status) result(pipeline)
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status
        type(basis_pipeline_t) :: pipeline

        call pipeline%initialize(n_inputs, status)
    end function make_basis_pipeline

    !> Construct an empty sequential pipeline.
    function make_sequential_basis_pipeline(n_inputs, status) result(pipeline)
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status
        type(sequential_basis_pipeline_t) :: pipeline

        call pipeline%initialize(n_inputs, status)
    end function make_sequential_basis_pipeline

    subroutine pipeline_initialize(self, n_inputs, status)
        class(basis_pipeline_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status

        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline: n_inputs must be positive")
            return
        end if
        self%n_inputs = n_inputs
        self%n_stages = 0
        self%fitted = .false.
        allocate(self%stages(0))
        allocate(self%stage_names(0))
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_initialize

    subroutine pipeline_append(self, stage, status, name)
        class(basis_pipeline_t), intent(inout) :: self
        type(basis_map_t), intent(in) :: stage
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: name
        type(basis_map_t), allocatable :: new_stages(:)
        character(len=PIPELINE_NAME_LENGTH), allocatable :: new_names(:)
        character(:), allocatable :: stage_name
        integer :: old_count, i

        if (self%n_inputs < 1 .or. .not. allocated(self%stages)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline append: pipeline is not initialized")
            return
        end if
        if (.not. stage%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline append: stage is not initialized")
            return
        end if
        if (stage%input_count() /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline append: input dimensions do not match")
            return
        end if
        stage_name = default_stage_name(self%n_stages + 1)
        if (present(name)) then
            if (len_trim(name) < 1 .or. len_trim(name) > PIPELINE_NAME_LENGTH) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis pipeline append: stage name is invalid")
                return
            end if
            stage_name = trim(name)
        end if
        if (allocated(self%stage_names)) then
            do i = 1, self%n_stages
                if (trim(self%stage_names(i)) == stage_name) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "basis pipeline append: stage names must be unique")
                    return
                end if
            end do
        end if

        old_count = self%n_stages
        allocate(new_stages(old_count + 1))
        allocate(new_names(old_count + 1))
        if (old_count > 0) new_stages(1:old_count) = self%stages
        if (old_count > 0) new_names(1:old_count) = self%stage_names
        new_stages(old_count + 1) = stage
        new_names(old_count + 1) = stage_name
        call move_alloc(new_stages, self%stages)
        call move_alloc(new_names, self%stage_names)
        self%n_stages = old_count + 1
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_append

    !> Mark the fixed feature union as fitted after checking a sample matrix.
    !>
    !> Basis maps are configured at construction time and therefore have no
    !> data-dependent state to estimate.  Keeping a fit entry point makes the
    !> object usable by generic fit/transform workflows while preserving that
    !> explicit contract.
    subroutine pipeline_fit(self, x, status)
        class(basis_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. pipeline_valid(self) .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline fit: model or input shape is invalid")
            return
        end if
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_fit

    subroutine pipeline_transform(self, x, phi, status)
        class(basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, offset, n_features

        if (.not. pipeline_valid(self) .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%n_inputs .or. size(phi, 1) /= size(x, 1) .or. &
            size(phi, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline transform: model or array shape is invalid")
            return
        end if
        offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            call self%stages(i)%evaluate(x, phi(:, offset + 1:offset + n_features), &
                status)
            if (status%code /= FORTNUM_OK) return
            offset = offset + n_features
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_transform

    subroutine pipeline_jvp(self, x, theta_dot, x_dot, phi, phi_dot, status)
        class(basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_theta_dot(:)
        integer :: i, feature_offset, parameter_offset
        integer :: n_features, n_parameters

        if (.not. pipeline_valid(self) .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%n_inputs .or. any(shape(x_dot) /= shape(x)) .or. &
            size(phi, 1) /= size(x, 1) .or. &
            size(phi, 2) /= self%feature_count() .or. &
            any(shape(phi_dot) /= shape(phi)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline jvp: model or array shape is invalid")
            return
        end if

        phi = 0.0_dp
        phi_dot = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            n_parameters = self%stages(i)%parameter_count()
            if (n_parameters > 0) then
                allocate(local_theta_dot(n_parameters))
                local_theta_dot = theta_dot(parameter_offset + 1: &
                    parameter_offset + n_parameters)
            else
                allocate(local_theta_dot(0))
            end if
            call self%stages(i)%jvp(x, local_theta_dot, x_dot, &
                phi(:, feature_offset + 1:feature_offset + n_features), &
                phi_dot(:, feature_offset + 1:feature_offset + n_features), &
                status)
            deallocate(local_theta_dot)
            if (status%code /= FORTNUM_OK) return
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_jvp

    subroutine pipeline_vjp(self, x, u, theta_bar, x_bar, status)
        class(basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_theta_bar(:), local_x_bar(:, :)
        integer :: i, feature_offset, parameter_offset
        integer :: n_features, n_parameters

        if (.not. pipeline_valid(self) .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%n_inputs .or. size(u, 1) /= size(x, 1) .or. &
            size(u, 2) /= self%feature_count() .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline vjp: model or array shape is invalid")
            return
        end if

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            n_parameters = self%stages(i)%parameter_count()
            allocate(local_theta_bar(n_parameters))
            allocate(local_x_bar(size(x, 1), size(x, 2)))
            call self%stages(i)%vjp(x, u(:, feature_offset + 1: &
                feature_offset + n_features), local_theta_bar, local_x_bar, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_theta_bar, local_x_bar)
                return
            end if
            if (n_parameters > 0) theta_bar(parameter_offset + 1: &
                parameter_offset + n_parameters) = local_theta_bar
            x_bar = x_bar + local_x_bar
            deallocate(local_theta_bar, local_x_bar)
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_vjp

    subroutine pipeline_hvp(self, x, u, theta_dot, x_dot, theta_hvp, x_hvp, &
            status)
        !! HVP of a horizontal feature union.  Each stage contributes an
        !! independent parameter block; input curvature is accumulated because
        !! all stages consume the same original input matrix.
        class(basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_theta_dot(:), local_theta_hvp(:)
        real(dp), allocatable :: local_x_hvp(:, :)
        integer :: i, feature_offset, parameter_offset
        integer :: n_features, n_parameters

        if (.not. pipeline_valid(self) .or. size(x, 1) < 1 .or. &
                size(x, 2) /= self%n_inputs .or. size(u, 1) /= size(x, 1) .or. &
                size(u, 2) /= self%feature_count() .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                size(theta_dot) /= self%parameter_count() .or. &
                size(theta_hvp) /= self%parameter_count() .or. &
                any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline hvp: model or array shape is invalid")
            return
        end if
        theta_hvp = 0.0_dp
        x_hvp = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            n_parameters = self%stages(i)%parameter_count()
            allocate(local_theta_dot(n_parameters), local_theta_hvp(n_parameters))
            if (n_parameters > 0) local_theta_dot = theta_dot(parameter_offset + 1: &
                parameter_offset + n_parameters)
            allocate(local_x_hvp(size(x, 1), size(x, 2)))
            call self%stages(i)%hvp(x, u(:, feature_offset + 1: &
                feature_offset + n_features), local_theta_dot, x_dot, &
                local_theta_hvp, local_x_hvp, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_theta_dot, local_theta_hvp, local_x_hvp)
                return
            end if
            if (n_parameters > 0) theta_hvp(parameter_offset + 1: &
                parameter_offset + n_parameters) = local_theta_hvp
            x_hvp = x_hvp + local_x_hvp
            deallocate(local_theta_dot, local_theta_hvp, local_x_hvp)
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_hvp

    integer function pipeline_input_count(self) result(count)
        class(basis_pipeline_t), intent(in) :: self
        count = self%n_inputs
    end function pipeline_input_count

    integer function pipeline_stage_count(self) result(count)
        class(basis_pipeline_t), intent(in) :: self
        count = self%n_stages
    end function pipeline_stage_count

    integer function pipeline_feature_count(self) result(count)
        class(basis_pipeline_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%stages)) return
        do i = 1, self%n_stages
            count = count + self%stages(i)%feature_count()
        end do
    end function pipeline_feature_count

    integer function pipeline_parameter_count(self) result(count)
        class(basis_pipeline_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%stages)) return
        do i = 1, self%n_stages
            count = count + self%stages(i)%parameter_count()
        end do
    end function pipeline_parameter_count

    function pipeline_stage_name(self, stage) result(name)
        class(basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        character(:), allocatable :: name

        name = ""
        if (.not. allocated(self%stage_names)) return
        if (stage < 1 .or. stage > self%n_stages) return
        name = trim(self%stage_names(stage))
    end function pipeline_stage_name

    integer function pipeline_stage_feature_offset(self, stage) result(offset)
        class(basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        integer :: i

        offset = 0
        if (stage < 1 .or. stage > self%n_stages) return
        if (.not. allocated(self%stages)) return
        offset = 1
        do i = 1, stage - 1
            offset = offset + self%stages(i)%feature_count()
        end do
    end function pipeline_stage_feature_offset

    integer function pipeline_stage_parameter_offset(self, stage) result(offset)
        class(basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        integer :: i

        offset = 0
        if (stage < 1 .or. stage > self%n_stages) return
        if (.not. allocated(self%stages)) return
        offset = 1
        do i = 1, stage - 1
            offset = offset + self%stages(i)%parameter_count()
        end do
    end function pipeline_stage_parameter_offset

    function pipeline_feature_name(self, feature) result(name)
        class(basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name
        integer :: i, offset, n_features

        name = ""
        if (feature < 1 .or. feature > self%feature_count()) return
        offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            if (feature <= offset + n_features) then
                name = qualified_stage_name(trim(self%stage_names(i)), &
                    "feature", feature - offset)
                return
            end if
            offset = offset + n_features
        end do
    end function pipeline_feature_name

    function pipeline_parameter_name(self, parameter) result(name)
        class(basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: parameter
        character(:), allocatable :: name
        integer :: i, offset, n_parameters

        name = ""
        if (parameter < 1 .or. parameter > self%parameter_count()) return
        offset = 0
        do i = 1, self%n_stages
            n_parameters = self%stages(i)%parameter_count()
            if (parameter <= offset + n_parameters) then
                name = qualified_stage_name(trim(self%stage_names(i)), &
                    "parameter", parameter - offset)
                return
            end if
            offset = offset + n_parameters
        end do
    end function pipeline_parameter_name

    function pipeline_parameters(self) result(theta)
        class(basis_pipeline_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        real(dp), allocatable :: local_theta(:)
        integer :: i, offset, n_parameters

        allocate(theta(self%parameter_count()))
        theta = 0.0_dp
        offset = 0
        do i = 1, self%n_stages
            n_parameters = self%stages(i)%parameter_count()
            if (n_parameters > 0) then
                local_theta = self%stages(i)%parameters()
                theta(offset + 1:offset + n_parameters) = local_theta
            end if
            offset = offset + n_parameters
        end do
    end function pipeline_parameters

    subroutine pipeline_set_parameters(self, theta, status)
        class(basis_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, offset, n_parameters

        if (.not. pipeline_valid(self) .or. size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline set_parameters: model or shape is invalid")
            return
        end if
        offset = 0
        do i = 1, self%n_stages
            n_parameters = self%stages(i)%parameter_count()
            if (n_parameters > 0) then
                call self%stages(i)%set_parameters(theta(offset + 1: &
                    offset + n_parameters), status)
                if (status%code /= FORTNUM_OK) return
            end if
            offset = offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_set_parameters

    logical function pipeline_static_lowering_eligible(self) result(eligible)
        class(basis_pipeline_t), intent(in) :: self
        integer :: i

        eligible = pipeline_valid(self)
        if (.not. eligible) return
        do i = 1, self%n_stages
            if (.not. self%stages(i)%static_lowering_eligible()) then
                eligible = .false.
                return
            end if
        end do
    end function pipeline_static_lowering_eligible

    logical function pipeline_valid(self) result(valid)
        class(basis_pipeline_t), intent(in) :: self
        integer :: i

        valid = self%n_inputs > 0 .and. self%n_stages > 0 .and. &
            allocated(self%stages) .and. allocated(self%stage_names)
        if (.not. valid) return
        if (size(self%stages) < self%n_stages .or. &
            size(self%stage_names) < self%n_stages) then
            valid = .false.
            return
        end if
        do i = 1, self%n_stages
            if (.not. self%stages(i)%valid() .or. &
                self%stages(i)%input_count() /= self%n_inputs) then
                valid = .false.
                return
            end if
        end do
    end function pipeline_valid

    logical function pipeline_is_fitted(self) result(fitted)
        class(basis_pipeline_t), intent(in) :: self
        fitted = self%fitted .and. pipeline_valid(self)
    end function pipeline_is_fitted

    subroutine sequential_pipeline_initialize(self, n_inputs, status)
        class(sequential_basis_pipeline_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status

        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline: n_inputs must be positive")
            return
        end if
        self%n_inputs = n_inputs
        self%n_features = n_inputs
        self%n_stages = 0
        self%fitted = .false.
        allocate(self%stages(0))
        allocate(self%stage_names(0))
        call status_set(status, FORTNUM_OK, "")
    end subroutine sequential_pipeline_initialize

    subroutine sequential_pipeline_append(self, stage, status, name)
        class(sequential_basis_pipeline_t), intent(inout) :: self
        type(basis_map_t), intent(in) :: stage
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: name
        type(basis_map_t), allocatable :: new_stages(:)
        character(len=PIPELINE_NAME_LENGTH), allocatable :: new_names(:)
        character(:), allocatable :: stage_name
        integer :: old_count, i

        if (self%n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline append: pipeline is not initialized")
            return
        end if
        if (.not. allocated(self%stages)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline append: pipeline is not initialized")
            return
        end if
        if (.not. stage%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline append: stage is not initialized")
            return
        end if
        if (stage%input_count() /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline append: stage shape does not match")
            return
        end if
        stage_name = default_stage_name(self%n_stages + 1)
        if (present(name)) then
            if (len_trim(name) < 1 .or. len_trim(name) > PIPELINE_NAME_LENGTH) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "sequential basis pipeline append: stage name is invalid")
                return
            end if
            stage_name = trim(name)
        end if
        if (allocated(self%stage_names)) then
            do i = 1, self%n_stages
                if (trim(self%stage_names(i)) == stage_name) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "sequential basis pipeline append: stage names must be unique")
                    return
                end if
            end do
        end if

        old_count = self%n_stages
        allocate(new_stages(old_count + 1))
        allocate(new_names(old_count + 1))
        if (old_count > 0) new_stages(1:old_count) = self%stages
        if (old_count > 0) new_names(1:old_count) = self%stage_names
        new_stages(old_count + 1) = stage
        new_names(old_count + 1) = stage_name
        call move_alloc(new_stages, self%stages)
        call move_alloc(new_names, self%stage_names)
        self%n_stages = old_count + 1
        self%n_features = stage%feature_count()
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine sequential_pipeline_append

    subroutine sequential_pipeline_fit(self, x, status)
        class(sequential_basis_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: current(:, :), next(:, :)
        integer :: i, n_features

        if (.not. sequential_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline fit: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline fit: input shape is invalid")
            return
        end if

        allocate(current(size(x, 1), size(x, 2)))
        current = x
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            allocate(next(size(x, 1), n_features))
            call self%stages(i)%evaluate(current, next, status)
            if (status%code /= FORTNUM_OK) return
            call move_alloc(next, current)
        end do
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine sequential_pipeline_fit

    subroutine sequential_pipeline_transform(self, x, y, status)
        class(sequential_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: current(:, :), next(:, :)
        integer :: i, n_features

        if (.not. sequential_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline transform: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(y, 1) /= size(x, 1) .or. size(y, 2) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline transform: array shape is invalid")
            return
        end if

        allocate(current(size(x, 1), size(x, 2)))
        current = x
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            allocate(next(size(x, 1), n_features))
            call self%stages(i)%evaluate(current, next, status)
            if (status%code /= FORTNUM_OK) return
            call move_alloc(next, current)
        end do
        y = current
        call status_set(status, FORTNUM_OK, "")
    end subroutine sequential_pipeline_transform

    subroutine sequential_pipeline_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(sequential_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: current(:, :), current_dot(:, :)
        real(dp), allocatable :: next(:, :), next_dot(:, :), local_theta_dot(:)
        integer :: i, n_features, n_parameters, offset

        if (.not. sequential_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline jvp: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y, 1) /= size(x, 1) .or. &
            size(y, 2) /= self%n_features .or. any(shape(y_dot) /= shape(y)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline jvp: array shape is invalid")
            return
        end if

        allocate(current(size(x, 1), size(x, 2)))
        allocate(current_dot(size(x, 1), size(x, 2)))
        current = x
        current_dot = x_dot
        offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            n_parameters = self%stages(i)%parameter_count()
            allocate(next(size(x, 1), n_features))
            allocate(next_dot(size(x, 1), n_features))
            allocate(local_theta_dot(n_parameters))
            if (n_parameters > 0) local_theta_dot = theta_dot(offset + 1: &
                offset + n_parameters)
            call self%stages(i)%jvp(current, local_theta_dot, current_dot, next, &
                next_dot, status)
            deallocate(local_theta_dot)
            if (status%code /= FORTNUM_OK) return
            call move_alloc(next, current)
            call move_alloc(next_dot, current_dot)
            offset = offset + n_parameters
        end do
        y = current
        y_dot = current_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine sequential_pipeline_jvp

    subroutine sequential_pipeline_vjp(self, x, u, theta_bar, x_bar, status)
        class(sequential_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(pipeline_matrix_buffer_t), allocatable :: inputs(:), outputs(:)
        real(dp), allocatable :: current_u(:, :), local_theta_bar(:)
        real(dp), allocatable :: local_x_bar(:, :)
        integer :: i, n_features, n_parameters, offset

        if (.not. sequential_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline vjp: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(u, 1) /= size(x, 1) .or. size(u, 2) /= self%n_features .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline vjp: array shape is invalid")
            return
        end if

        allocate(inputs(self%n_stages), outputs(self%n_stages))
        allocate(inputs(1)%values(size(x, 1), size(x, 2)))
        inputs(1)%values = x
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            allocate(outputs(i)%values(size(x, 1), n_features))
            call self%stages(i)%evaluate(inputs(i)%values, outputs(i)%values, &
                status)
            if (status%code /= FORTNUM_OK) return
            if (i < self%n_stages) then
                allocate(inputs(i + 1)%values(size(x, 1), n_features))
                inputs(i + 1)%values = outputs(i)%values
            end if
        end do

        allocate(current_u(size(u, 1), size(u, 2)))
        current_u = u
        theta_bar = 0.0_dp
        offset = self%parameter_count()
        do i = self%n_stages, 1, -1
            n_parameters = self%stages(i)%parameter_count()
            offset = offset - n_parameters
            allocate(local_theta_bar(n_parameters))
            allocate(local_x_bar(size(x, 1), size(inputs(i)%values, 2)))
            call self%stages(i)%vjp(inputs(i)%values, current_u, local_theta_bar, &
                local_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            if (n_parameters > 0) theta_bar(offset + 1:offset + n_parameters) = &
                local_theta_bar
            deallocate(local_theta_bar)
            if (i > 1) then
                call move_alloc(local_x_bar, current_u)
            else
                x_bar = local_x_bar
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sequential_pipeline_vjp

    subroutine sequential_pipeline_hvp(self, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        !! Forward-over-reverse HVP for a sequential basis composition.
        !!
        !! The reverse cotangent itself has a directional component when a
        !! downstream stage is parameterized.  We therefore combine each
        !! stage's fixed-cotangent HVP with a VJP of that cotangent tangent,
        !! which is the standard compositional second-order chain rule.
        class(sequential_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(pipeline_matrix_buffer_t), allocatable :: inputs(:), outputs(:)
        type(pipeline_matrix_buffer_t), allocatable :: input_dots(:), output_dots(:)
        real(dp), allocatable :: current_u(:, :), current_u_dot(:, :)
        real(dp), allocatable :: local_theta_dot(:), local_theta_hvp(:)
        real(dp), allocatable :: local_theta_bar(:), local_x_hvp(:, :), local_x_bar(:, :)
        real(dp), allocatable :: local_x_bar_fixed(:, :)
        integer :: i, n_features, n_parameters, offset

        if (.not. sequential_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline hvp: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
                size(u, 1) /= size(x, 1) .or. size(u, 2) /= self%n_features .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                size(theta_dot) /= self%parameter_count() .or. &
                size(theta_hvp) /= self%parameter_count() .or. &
                any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline hvp: array shape is invalid")
            return
        end if

        allocate(inputs(self%n_stages), outputs(self%n_stages))
        allocate(input_dots(self%n_stages), output_dots(self%n_stages))
        allocate(inputs(1)%values(size(x, 1), size(x, 2)))
        allocate(input_dots(1)%values(size(x, 1), size(x, 2)))
        inputs(1)%values = x
        input_dots(1)%values = x_dot
        offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            n_parameters = self%stages(i)%parameter_count()
            allocate(outputs(i)%values(size(x, 1), n_features))
            allocate(output_dots(i)%values(size(x, 1), n_features))
            allocate(local_theta_dot(n_parameters))
            if (n_parameters > 0) local_theta_dot = theta_dot(offset + 1: &
                offset + n_parameters)
            call self%stages(i)%jvp(inputs(i)%values, local_theta_dot, &
                input_dots(i)%values, outputs(i)%values, output_dots(i)%values, &
                status)
            deallocate(local_theta_dot)
            if (status%code /= FORTNUM_OK) return
            if (i < self%n_stages) then
                allocate(inputs(i + 1)%values(size(x, 1), n_features))
                allocate(input_dots(i + 1)%values(size(x, 1), n_features))
                inputs(i + 1)%values = outputs(i)%values
                input_dots(i + 1)%values = output_dots(i)%values
            end if
            offset = offset + n_parameters
        end do

        allocate(current_u(size(u, 1), size(u, 2)))
        allocate(current_u_dot(size(u, 1), size(u, 2)))
        current_u = u
        current_u_dot = 0.0_dp
        theta_hvp = 0.0_dp
        offset = self%parameter_count()
        do i = self%n_stages, 1, -1
            n_features = self%stages(i)%feature_count()
            n_parameters = self%stages(i)%parameter_count()
            offset = offset - n_parameters
            allocate(local_theta_dot(n_parameters), local_theta_hvp(n_parameters))
            if (n_parameters > 0) local_theta_dot = theta_dot(offset + 1: &
                offset + n_parameters)
            allocate(local_theta_bar(n_parameters))
            allocate(local_x_hvp(size(x, 1), size(inputs(i)%values, 2)))
            allocate(local_x_bar(size(x, 1), size(inputs(i)%values, 2)))
            allocate(local_x_bar_fixed(size(x, 1), size(inputs(i)%values, 2)))
            call self%stages(i)%hvp(inputs(i)%values, current_u, local_theta_dot, &
                input_dots(i)%values, local_theta_hvp, local_x_hvp, status)
            if (status%code /= FORTNUM_OK) return
            call self%stages(i)%vjp(inputs(i)%values, current_u, &
                local_theta_bar, local_x_bar_fixed, status)
            if (status%code /= FORTNUM_OK) return
            call self%stages(i)%vjp(inputs(i)%values, current_u_dot, &
                local_theta_bar, local_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            if (n_parameters > 0) theta_hvp(offset + 1:offset + n_parameters) = &
                local_theta_hvp + local_theta_bar
            if (i > 1) then
                current_u = local_x_bar_fixed
                current_u_dot = local_x_hvp + local_x_bar
            else
                x_hvp = local_x_hvp + local_x_bar
            end if
            deallocate(local_theta_dot, local_theta_hvp, local_theta_bar, &
                local_x_hvp, local_x_bar, local_x_bar_fixed)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sequential_pipeline_hvp

    integer function sequential_pipeline_input_count(self) result(count)
        class(sequential_basis_pipeline_t), intent(in) :: self
        count = self%n_inputs
    end function sequential_pipeline_input_count

    integer function sequential_pipeline_stage_count(self) result(count)
        class(sequential_basis_pipeline_t), intent(in) :: self
        count = self%n_stages
    end function sequential_pipeline_stage_count

    integer function sequential_pipeline_feature_count(self) result(count)
        class(sequential_basis_pipeline_t), intent(in) :: self
        count = self%n_features
        if (self%n_inputs < 1) count = 0
    end function sequential_pipeline_feature_count

    integer function sequential_pipeline_parameter_count(self) result(count)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%stages)) return
        do i = 1, self%n_stages
            count = count + self%stages(i)%parameter_count()
        end do
    end function sequential_pipeline_parameter_count

    function sequential_pipeline_stage_name(self, stage) result(name)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        character(:), allocatable :: name

        name = ""
        if (.not. allocated(self%stage_names)) return
        if (stage < 1 .or. stage > self%n_stages) return
        name = trim(self%stage_names(stage))
    end function sequential_pipeline_stage_name

    integer function sequential_pipeline_stage_feature_offset(self, stage) &
            result(offset)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        integer :: i

        offset = 0
        if (stage < 1 .or. stage > self%n_stages) return
        if (.not. allocated(self%stages)) return
        offset = 1
        do i = 1, stage - 1
            offset = offset + self%stages(i)%feature_count()
        end do
    end function sequential_pipeline_stage_feature_offset

    integer function sequential_pipeline_stage_parameter_offset(self, stage) &
            result(offset)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        integer :: i

        offset = 0
        if (stage < 1 .or. stage > self%n_stages) return
        if (.not. allocated(self%stages)) return
        offset = 1
        do i = 1, stage - 1
            offset = offset + self%stages(i)%parameter_count()
        end do
    end function sequential_pipeline_stage_parameter_offset

    function sequential_pipeline_feature_name(self, feature) result(name)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name

        name = ""
        if (self%n_stages < 1 .or. feature < 1 .or. &
            feature > self%feature_count()) return
        name = qualified_stage_name(trim(self%stage_names(self%n_stages)), &
            "feature", feature)
    end function sequential_pipeline_feature_name

    function sequential_pipeline_parameter_name(self, parameter) result(name)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: parameter
        character(:), allocatable :: name
        integer :: i, offset, n_parameters

        name = ""
        if (parameter < 1 .or. parameter > self%parameter_count()) return
        offset = 0
        do i = 1, self%n_stages
            n_parameters = self%stages(i)%parameter_count()
            if (parameter <= offset + n_parameters) then
                name = qualified_stage_name(trim(self%stage_names(i)), &
                    "parameter", parameter - offset)
                return
            end if
            offset = offset + n_parameters
        end do
    end function sequential_pipeline_parameter_name

    function sequential_pipeline_parameters(self) result(theta)
        class(sequential_basis_pipeline_t), intent(in) :: self
        real(dp), allocatable :: theta(:), local_theta(:)
        integer :: i, offset, n_parameters

        allocate(theta(self%parameter_count()))
        theta = 0.0_dp
        offset = 0
        if (.not. allocated(self%stages)) return
        do i = 1, self%n_stages
            n_parameters = self%stages(i)%parameter_count()
            if (n_parameters > 0) then
                local_theta = self%stages(i)%parameters()
                theta(offset + 1:offset + n_parameters) = local_theta
            end if
            offset = offset + n_parameters
        end do
    end function sequential_pipeline_parameters

    subroutine sequential_pipeline_set_parameters(self, theta, status)
        class(sequential_basis_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, offset, n_parameters

        if (.not. sequential_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline set_parameters: model is invalid")
            return
        end if
        if (size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline set_parameters: shape is invalid")
            return
        end if
        offset = 0
        do i = 1, self%n_stages
            n_parameters = self%stages(i)%parameter_count()
            if (n_parameters > 0) then
                call self%stages(i)%set_parameters(theta(offset + 1: &
                    offset + n_parameters), status)
                if (status%code /= FORTNUM_OK) return
            end if
            offset = offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sequential_pipeline_set_parameters

    logical function sequential_pipeline_static_lowering_eligible(self) result(eligible)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer :: i

        eligible = sequential_pipeline_valid(self)
        if (.not. eligible) return
        do i = 1, self%n_stages
            if (.not. self%stages(i)%static_lowering_eligible()) then
                eligible = .false.
                return
            end if
        end do
    end function sequential_pipeline_static_lowering_eligible

    logical function sequential_pipeline_valid(self) result(valid)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer :: i, previous_features

        valid = .true.
        if (self%n_inputs < 1) valid = .false.
        if (.not. allocated(self%stages) .or. &
            .not. allocated(self%stage_names)) valid = .false.
        if (self%n_stages < 1) valid = .false.
        if (.not. valid) return
        if (size(self%stages) < self%n_stages .or. &
            size(self%stage_names) < self%n_stages) then
            valid = .false.
            return
        end if
        previous_features = self%n_inputs
        do i = 1, self%n_stages
            if (.not. self%stages(i)%valid()) then
                valid = .false.
                return
            end if
            if (self%stages(i)%input_count() /= previous_features) then
                valid = .false.
                return
            end if
            previous_features = self%stages(i)%feature_count()
        end do
        if (self%n_features /= previous_features) valid = .false.
    end function sequential_pipeline_valid

    logical function sequential_pipeline_is_fitted(self) result(fitted)
        class(sequential_basis_pipeline_t), intent(in) :: self
        fitted = self%fitted
        if (.not. fitted) return
        fitted = sequential_pipeline_valid(self)
    end function sequential_pipeline_is_fitted

    function default_stage_name(index) result(name)
        integer, intent(in) :: index
        character(:), allocatable :: name
        character(len=32) :: buffer

        write (buffer, '("stage_",i0)') index
        name = trim(buffer)
    end function default_stage_name

    function qualified_stage_name(stage, kind, index) result(name)
        character(*), intent(in) :: stage, kind
        integer, intent(in) :: index
        character(:), allocatable :: name
        character(len=32) :: buffer

        write (buffer, '(i0)') index
        name = trim(stage)//"."//trim(kind)//"_"//trim(buffer)
    end function qualified_stage_name

end module fortml_pipeline
