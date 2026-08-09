!> Composable feature maps built from the differentiable basis API.
module fortml_pipeline
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_basis, only: basis_map_t
    use fortml_estimator_capabilities, only: estimator_capability_t, &
        make_transformer_capabilities
    implicit none
    private

    integer, parameter :: PIPELINE_NAME_LENGTH = 128

    !> Names and validation rules for the dense input columns of a pipeline.
    !>
    !> All feature maps in the current API use real(dp) dense columns.  This
    !> value object carries the missing schema part of the transformer
    !> contract: callers may install stable names once and validate a later
    !> batch before consuming it.  Failed updates are transactional.
    type, public :: basis_input_schema_t
        private
        integer :: n_features = 0
        character(len=PIPELINE_NAME_LENGTH), allocatable :: names(:)
    contains
        procedure, public :: initialize => input_schema_initialize
        procedure, public :: set_names => input_schema_set_names
        procedure, public :: name => input_schema_name
        procedure, public :: validate_names => input_schema_validate_names
        procedure, public :: count => input_schema_count
        procedure, public :: valid => input_schema_valid
    end type basis_input_schema_t

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
        type(basis_input_schema_t) :: input_schema
    contains
        procedure, public :: initialize => pipeline_initialize
        procedure, public :: append => pipeline_append
        procedure, public :: clone => pipeline_clone
        procedure, public :: clone_device => pipeline_clone_device
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
        procedure, public :: set_input_schema => pipeline_set_input_schema
        procedure, public :: input_schema_name => pipeline_input_schema_name
        procedure, public :: validate_input_schema => &
            pipeline_validate_input_schema
        procedure, public :: static_lowering_eligible => &
            pipeline_static_lowering_eligible
        procedure, public :: capabilities => pipeline_capabilities
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
        type(basis_input_schema_t) :: input_schema
    contains
        procedure, public :: initialize => sequential_pipeline_initialize
        procedure, public :: append => sequential_pipeline_append
        procedure, public :: fit => sequential_pipeline_fit
        procedure, public :: transform => sequential_pipeline_transform
        procedure, public :: evaluate => sequential_pipeline_transform
        procedure, public :: transform_device => &
            sequential_pipeline_transform_device
        procedure, public :: jvp => sequential_pipeline_jvp
        procedure, public :: jvp_device => sequential_pipeline_jvp_device
        procedure, public :: vjp => sequential_pipeline_vjp
        procedure, public :: vjp_device => sequential_pipeline_vjp_device
        procedure, public :: hvp => sequential_pipeline_hvp
        procedure, public :: hvp_device => sequential_pipeline_hvp_device
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
        procedure, public :: set_input_schema => sequential_pipeline_set_input_schema
        procedure, public :: input_schema_name => &
            sequential_pipeline_input_schema_name
        procedure, public :: validate_input_schema => &
            sequential_pipeline_validate_input_schema
        procedure, public :: static_lowering_eligible => &
            sequential_pipeline_static_lowering_eligible
        procedure, public :: capabilities => sequential_pipeline_capabilities
        procedure, public :: device_supported => &
            sequential_pipeline_device_supported
        procedure, public :: valid => sequential_pipeline_valid
        procedure, public :: is_fitted => sequential_pipeline_is_fitted
    end type sequential_basis_pipeline_t

    !> A fan-out/fan-in DAG of sequential basis pipelines.
    !>
    !> The input matrix is fanned out to independent named branches.  Each
    !> branch may contain an arbitrary sequential basis composition, and the
    !> branch outputs are concatenated in append order.  Reverse products fan
    !> the output cotangent back through each branch and sum input cotangents;
    !> parameter blocks remain branch-major and therefore have deterministic
    !> names and offsets.  This is intentionally one acyclic graph layer: a
    !> branch cannot consume another branch, so cycles and ambiguous routing
    !> are rejected by construction.
    type, public :: basis_fanout_pipeline_t
        private
        integer :: n_inputs = 0
        integer :: n_branches = 0
        logical :: fitted = .false.
        type(sequential_basis_pipeline_t), allocatable :: branches(:)
        character(len=PIPELINE_NAME_LENGTH), allocatable :: branch_names(:)
        type(basis_input_schema_t) :: input_schema
    contains
        procedure, public :: initialize => fanout_pipeline_initialize
        procedure, public :: append => fanout_pipeline_append
        procedure, public :: fit => fanout_pipeline_fit
        procedure, public :: transform => fanout_pipeline_transform
        procedure, public :: evaluate => fanout_pipeline_transform
        procedure, public :: transform_device => fanout_pipeline_transform_device
        procedure, public :: jvp => fanout_pipeline_jvp
        procedure, public :: jvp_device => fanout_pipeline_jvp_device
        procedure, public :: vjp => fanout_pipeline_vjp
        procedure, public :: vjp_device => fanout_pipeline_vjp_device
        procedure, public :: hvp => fanout_pipeline_hvp
        procedure, public :: hvp_device => fanout_pipeline_hvp_device
        procedure, public :: input_count => fanout_pipeline_input_count
        procedure, public :: branch_count => fanout_pipeline_branch_count
        procedure, public :: feature_count => fanout_pipeline_feature_count
        procedure, public :: parameter_count => fanout_pipeline_parameter_count
        procedure, public :: parameters => fanout_pipeline_parameters
        procedure, public :: set_parameters => fanout_pipeline_set_parameters
        procedure, public :: branch_name => fanout_pipeline_branch_name
        procedure, public :: feature_name => fanout_pipeline_feature_name
        procedure, public :: parameter_name => fanout_pipeline_parameter_name
        procedure, public :: branch_feature_offset => &
            fanout_pipeline_branch_feature_offset
        procedure, public :: branch_parameter_offset => &
            fanout_pipeline_branch_parameter_offset
        procedure, public :: set_input_schema => fanout_pipeline_set_input_schema
        procedure, public :: input_schema_name => fanout_pipeline_input_schema_name
        procedure, public :: validate_input_schema => &
            fanout_pipeline_validate_input_schema
        procedure, public :: static_lowering_eligible => &
            fanout_pipeline_static_lowering_eligible
        procedure, public :: capabilities => fanout_pipeline_capabilities
        procedure, public :: device_supported => fanout_pipeline_device_supported
        procedure, public :: valid => fanout_pipeline_valid
        procedure, public :: is_fitted => fanout_pipeline_is_fitted
    end type basis_fanout_pipeline_t

    !> A named residual-sum DAG with one main and one residual branch.
    !>
    !> Both branches consume the same dense input and produce the same feature
    !> shape.  The forward map is ``main(x) + residual(x)``; parameter blocks
    !> are packed main-major followed by residual-major.  Reverse products sum
    !> input cotangents, while feature and parameter metadata retain the branch
    !> names.  This bounded layer is intentionally acyclic and rejects shape
    !> mismatches at configuration time.
    type, public :: basis_residual_pipeline_t
        private
        integer :: n_inputs = 0
        integer :: n_features = 0
        logical :: main_configured = .false.
        logical :: residual_configured = .false.
        logical :: configured = .false.
        logical :: fitted = .false.
        type(sequential_basis_pipeline_t) :: main_branch
        type(sequential_basis_pipeline_t) :: residual_branch
        character(len=PIPELINE_NAME_LENGTH) :: main_name = "main"
        character(len=PIPELINE_NAME_LENGTH) :: residual_name = "residual"
        type(basis_input_schema_t) :: input_schema
    contains
        procedure, public :: initialize => residual_pipeline_initialize
        procedure, public :: set_main => residual_pipeline_set_main
        procedure, public :: set_residual => residual_pipeline_set_residual
        procedure, public :: fit => residual_pipeline_fit
        procedure, public :: transform => residual_pipeline_transform
        procedure, public :: evaluate => residual_pipeline_transform
        procedure, public :: transform_device => residual_pipeline_transform_device
        procedure, public :: jvp => residual_pipeline_jvp
        procedure, public :: jvp_device => residual_pipeline_jvp_device
        procedure, public :: vjp => residual_pipeline_vjp
        procedure, public :: vjp_device => residual_pipeline_vjp_device
        procedure, public :: hvp => residual_pipeline_hvp
        procedure, public :: hvp_device => residual_pipeline_hvp_device
        procedure, public :: input_count => residual_pipeline_input_count
        procedure, public :: feature_count => residual_pipeline_feature_count
        procedure, public :: parameter_count => residual_pipeline_parameter_count
        procedure, public :: parameters => residual_pipeline_parameters
        procedure, public :: set_parameters => residual_pipeline_set_parameters
        procedure, public :: branch_name => residual_pipeline_branch_name
        procedure, public :: feature_name => residual_pipeline_feature_name
        procedure, public :: parameter_name => residual_pipeline_parameter_name
        procedure, public :: main_feature_offset => residual_pipeline_main_feature_offset
        procedure, public :: residual_feature_offset => residual_pipeline_residual_feature_offset
        procedure, public :: main_parameter_offset => residual_pipeline_main_parameter_offset
        procedure, public :: residual_parameter_offset => residual_pipeline_residual_parameter_offset
        procedure, public :: set_input_schema => residual_pipeline_set_input_schema
        procedure, public :: input_schema_name => residual_pipeline_input_schema_name
        procedure, public :: validate_input_schema => &
            residual_pipeline_validate_input_schema
        procedure, public :: static_lowering_eligible => &
            residual_pipeline_static_lowering_eligible
        procedure, public :: capabilities => residual_pipeline_capabilities
        procedure, public :: device_supported => residual_pipeline_device_supported
        procedure, public :: valid => residual_pipeline_valid
        procedure, public :: is_fitted => residual_pipeline_is_fitted
    end type basis_residual_pipeline_t

    public :: make_basis_pipeline
    public :: make_sequential_basis_pipeline
    public :: make_basis_fanout_pipeline
    public :: make_basis_residual_pipeline

contains

    subroutine input_schema_initialize(self, n_features, status, names)
        class(basis_input_schema_t), intent(out) :: self
        integer, intent(in) :: n_features
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: names(:)
        integer :: i

        self%n_features = 0
        if (allocated(self%names)) deallocate(self%names)
        if (n_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "input schema: feature count must be positive")
            return
        end if
        allocate(self%names(n_features))
        self%n_features = n_features
        do i = 1, n_features
            self%names(i) = "feature_"//integer_text(i)
        end do
        if (present(names)) then
            call self%set_names(names, status)
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine input_schema_initialize

    subroutine input_schema_set_names(self, names, status)
        class(basis_input_schema_t), intent(inout) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status
        character(len=PIPELINE_NAME_LENGTH), allocatable :: candidate(:)
        integer :: i, j

        if (self%n_features < 1 .or. .not. allocated(self%names) .or. &
            size(names) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "input schema: name count does not match feature count")
            return
        end if
        allocate(candidate(self%n_features))
        do i = 1, self%n_features
            if (len_trim(names(i)) < 1 .or. len_trim(names(i)) > &
                PIPELINE_NAME_LENGTH) then
                deallocate(candidate)
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "input schema: names must be nonempty and bounded")
                return
            end if
            candidate(i) = trim(names(i))
            do j = 1, i - 1
                if (trim(candidate(j)) == trim(candidate(i))) then
                    deallocate(candidate)
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "input schema: names must be unique")
                    return
                end if
            end do
        end do
        call move_alloc(candidate, self%names)
        call status_set(status, FORTNUM_OK, "")
    end subroutine input_schema_set_names

    function input_schema_name(self, feature) result(name)
        class(basis_input_schema_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name

        name = ""
        if (.not. self%valid()) return
        if (feature < 1 .or. feature > self%n_features) return
        name = trim(self%names(feature))
    end function input_schema_name

    subroutine input_schema_validate_names(self, names, status)
        class(basis_input_schema_t), intent(in) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. self%valid() .or. size(names) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "input schema: candidate shape is invalid")
            return
        end if
        do i = 1, self%n_features
            if (trim(names(i)) /= trim(self%names(i))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "input schema: candidate names do not match")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine input_schema_validate_names

    integer function input_schema_count(self) result(count)
        class(basis_input_schema_t), intent(in) :: self
        count = self%n_features
    end function input_schema_count

    logical function input_schema_valid(self) result(valid)
        class(basis_input_schema_t), intent(in) :: self
        integer :: i, j

        valid = self%n_features > 0 .and. allocated(self%names)
        if (.not. valid) return
        if (size(self%names) /= self%n_features) then
            valid = .false.
            return
        end if
        do i = 1, self%n_features
            if (len_trim(self%names(i)) < 1) then
                valid = .false.
                return
            end if
            do j = 1, i - 1
                if (trim(self%names(i)) == trim(self%names(j))) then
                    valid = .false.
                    return
                end if
            end do
        end do
    end function input_schema_valid

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

    !> Construct an empty fan-out/fan-in basis DAG.
    function make_basis_fanout_pipeline(n_inputs, status) result(pipeline)
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status
        type(basis_fanout_pipeline_t) :: pipeline

        call pipeline%initialize(n_inputs, status)
    end function make_basis_fanout_pipeline

    !> Construct an empty named residual-sum pipeline.
    function make_basis_residual_pipeline(n_inputs, status) result(pipeline)
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status
        type(basis_residual_pipeline_t) :: pipeline

        call pipeline%initialize(n_inputs, status)
    end function make_basis_residual_pipeline

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
        call self%input_schema%initialize(n_inputs, status)
        if (status%code /= FORTNUM_OK) return
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

    subroutine pipeline_clone(self, clone, status)
        !! Deep-copy a configured pipeline without sharing stage state.
        !!
        !! Intrinsic assignment of the allocatable polymorphic stage maps is
        !! used only after validating the source.  The candidate is committed
        !! to ``clone`` in one assignment, so malformed sources leave an
        !! existing destination untouched.  This is the host-side clone/reset
        !! seam used by model-selection and cross-validation callers.
        class(basis_pipeline_t), intent(in) :: self
        type(basis_pipeline_t), intent(inout) :: clone
        type(basis_pipeline_t) :: candidate
        type(fortnum_status_t), intent(out) :: status

        if (.not. pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline clone: source pipeline is invalid")
            return
        end if
        select type (source => self)
            type is (basis_pipeline_t)
            candidate = source
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline clone: source dynamic type is unsupported")
            return
        end select
        clone = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_clone

    subroutine pipeline_clone_device(self, device, clone, status)
        !! Device-aware clone boundary; CUDA requires resident graph support.
        class(basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        type(basis_pipeline_t), intent(inout) :: clone
        type(fortnum_status_t), intent(out) :: status

        if (.not. pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline clone device: source pipeline is invalid")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline clone device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%clone(clone, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis pipeline clone device: resident CUDA graph is unavailable")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis pipeline clone device: device kind is invalid")
        end select
    end subroutine pipeline_clone_device

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

    subroutine pipeline_set_input_schema(self, names, status)
        class(basis_pipeline_t), intent(inout) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%set_names(names, status)
    end subroutine pipeline_set_input_schema

    function pipeline_input_schema_name(self, feature) result(name)
        class(basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name

        name = self%input_schema%name(feature)
    end function pipeline_input_schema_name

    subroutine pipeline_validate_input_schema(self, names, status)
        class(basis_pipeline_t), intent(in) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%validate_names(names, status)
    end subroutine pipeline_validate_input_schema

    function pipeline_feature_name(self, feature) result(name)
        class(basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name
        integer :: i, offset, n_features
        character(:), allocatable :: local_name

        name = ""
        if (feature < 1 .or. feature > self%feature_count()) return
        offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%feature_count()
            if (feature <= offset + n_features) then
                local_name = self%stages(i)%feature_name(feature - offset)
                if (len_trim(local_name) > 0) then
                    name = trim(self%stage_names(i))//"."//trim(local_name)
                else
                    name = qualified_stage_name(trim(self%stage_names(i)), &
                        "feature", feature - offset)
                end if
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
            allocated(self%stages) .and. allocated(self%stage_names) .and. &
            self%input_schema%valid()
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

    !> Return the generic transformer contract for this feature union.
    subroutine pipeline_capabilities(self, report, status)
        class(basis_pipeline_t), intent(in) :: self
        type(estimator_capability_t), intent(out) :: report
        type(fortnum_status_t), intent(out) :: status

        if (.not. pipeline_valid(self)) then
            call report%initialize("basis_pipeline", 1, 1, status)
            if (status%code == FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis pipeline capabilities: pipeline is invalid")
            end if
            return
        end if
        report = make_transformer_capabilities("basis_pipeline", self%n_inputs, &
            self%feature_count(), status, self%is_fitted())
        if (status%code /= FORTNUM_OK) return
        report%supports_input_jvp = .true.
        report%supports_input_vjp = .true.
        report%supports_input_hvp = .true.
        report%supports_parameter_jvp = .true.
        report%supports_parameter_vjp = .true.
        report%supports_parameter_hvp = .true.
    end subroutine pipeline_capabilities

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
        call self%input_schema%initialize(n_inputs, status)
        if (status%code /= FORTNUM_OK) return
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

    subroutine sequential_pipeline_transform_device(self, device, x, y, status)
        class(sequential_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. sequential_pipeline_device_ready(self, device, status, &
            "sequential basis pipeline transform")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%transform(x, y, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "sequential basis pipeline transform: no resident CUDA basis "// &
                "kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline transform: device kind is invalid")
        end select
    end subroutine sequential_pipeline_transform_device

    subroutine sequential_pipeline_jvp_device(self, device, x, theta_dot, x_dot, &
            y, y_dot, status)
        class(sequential_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(inout) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. sequential_pipeline_device_ready(self, device, status, &
            "sequential basis pipeline JVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%jvp(x, theta_dot, x_dot, y, y_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "sequential basis pipeline JVP: no resident CUDA basis kernel "// &
                "is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline JVP: device kind is invalid")
        end select
    end subroutine sequential_pipeline_jvp_device

    subroutine sequential_pipeline_vjp_device(self, device, x, u, theta_bar, &
            x_bar, status)
        class(sequential_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(inout) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. sequential_pipeline_device_ready(self, device, status, &
            "sequential basis pipeline VJP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%vjp(x, u, theta_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "sequential basis pipeline VJP: no resident CUDA basis kernel "// &
                "is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline VJP: device kind is invalid")
        end select
    end subroutine sequential_pipeline_vjp_device

    subroutine sequential_pipeline_hvp_device(self, device, x, u, theta_dot, &
            x_dot, theta_hvp, x_hvp, status)
        class(sequential_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(inout) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. sequential_pipeline_device_ready(self, device, status, &
            "sequential basis pipeline HVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "sequential basis pipeline HVP: no resident CUDA basis kernel "// &
                "is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential basis pipeline HVP: device kind is invalid")
        end select
    end subroutine sequential_pipeline_hvp_device

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

    subroutine sequential_pipeline_set_input_schema(self, names, status)
        class(sequential_basis_pipeline_t), intent(inout) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%set_names(names, status)
    end subroutine sequential_pipeline_set_input_schema

    function sequential_pipeline_input_schema_name(self, feature) result(name)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name

        name = self%input_schema%name(feature)
    end function sequential_pipeline_input_schema_name

    subroutine sequential_pipeline_validate_input_schema(self, names, status)
        class(sequential_basis_pipeline_t), intent(in) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%validate_names(names, status)
    end subroutine sequential_pipeline_validate_input_schema

    function sequential_pipeline_feature_name(self, feature) result(name)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name
        character(:), allocatable :: local_name

        name = ""
        if (self%n_stages < 1 .or. feature < 1 .or. &
            feature > self%feature_count()) return
        local_name = self%stages(self%n_stages)%feature_name(feature)
        if (len_trim(local_name) > 0) then
            name = trim(self%stage_names(self%n_stages))//"."//trim(local_name)
        else
            name = qualified_stage_name(trim(self%stage_names(self%n_stages)), &
                "feature", feature)
        end if
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
        if (.not. self%input_schema%valid()) valid = .false.
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

    !> Return the generic transformer contract for this sequential pipeline.
    subroutine sequential_pipeline_capabilities(self, report, status)
        class(sequential_basis_pipeline_t), intent(in) :: self
        type(estimator_capability_t), intent(out) :: report
        type(fortnum_status_t), intent(out) :: status

        if (.not. sequential_pipeline_valid(self)) then
            call report%initialize("sequential_basis_pipeline", 1, 1, status)
            if (status%code == FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "sequential basis pipeline capabilities: pipeline is invalid")
            end if
            return
        end if
        report = make_transformer_capabilities("sequential_basis_pipeline", &
            self%n_inputs, self%feature_count(), status, self%is_fitted())
        if (status%code /= FORTNUM_OK) return
        report%supports_input_jvp = .true.
        report%supports_input_vjp = .true.
        report%supports_input_hvp = .true.
        report%supports_parameter_jvp = .true.
        report%supports_parameter_vjp = .true.
        report%supports_parameter_hvp = .true.
    end subroutine sequential_pipeline_capabilities

    logical function sequential_pipeline_device_supported(self, device_kind) &
            result(supported)
        class(sequential_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = .false.
        if (.not. sequential_pipeline_valid(self)) return
        if (.not. self%is_fitted()) return
        if (device_kind == FORTML_DEVICE_CPU) supported = .true.
    end function sequential_pipeline_device_supported

    logical function sequential_pipeline_device_ready(self, device, status, &
            operation) result(ready)
        class(sequential_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        ready = .false.
        if (.not. sequential_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model is invalid")
            return
        end if
        if (.not. self%is_fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model is not fitted")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": device is not selected")
            return
        end if
        ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end function sequential_pipeline_device_ready

    subroutine fanout_pipeline_initialize(self, n_inputs, status)
        class(basis_fanout_pipeline_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status

        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline: n_inputs must be positive")
            return
        end if
        self%n_inputs = n_inputs
        self%n_branches = 0
        self%fitted = .false.
        allocate(self%branches(0))
        allocate(self%branch_names(0))
        call self%input_schema%initialize(n_inputs, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine fanout_pipeline_initialize

    subroutine fanout_pipeline_append(self, branch, status, name)
        class(basis_fanout_pipeline_t), intent(inout) :: self
        type(sequential_basis_pipeline_t), intent(in) :: branch
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: name
        type(sequential_basis_pipeline_t), allocatable :: new_branches(:)
        character(len=PIPELINE_NAME_LENGTH), allocatable :: new_names(:)
        character(:), allocatable :: branch_name
        integer :: old_count, i

        if (self%n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline append: pipeline is not initialized")
            return
        end if
        if (.not. allocated(self%branches)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline append: pipeline is not initialized")
            return
        end if
        if (.not. branch%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline append: branch is invalid")
            return
        end if
        if (branch%input_count() /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline append: branch input dimensions do not match")
            return
        end if
        branch_name = default_branch_name(self%n_branches + 1)
        if (present(name)) then
            if (len_trim(name) < 1 .or. len_trim(name) > PIPELINE_NAME_LENGTH) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis fanout pipeline append: branch name is invalid")
                return
            end if
            branch_name = trim(name)
        end if
        do i = 1, self%n_branches
            if (trim(self%branch_names(i)) == branch_name) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis fanout pipeline append: branch names must be unique")
                return
            end if
        end do

        old_count = self%n_branches
        allocate(new_branches(old_count + 1))
        allocate(new_names(old_count + 1))
        if (old_count > 0) new_branches(1:old_count) = self%branches
        if (old_count > 0) new_names(1:old_count) = self%branch_names
        new_branches(old_count + 1) = branch
        new_names(old_count + 1) = branch_name
        call move_alloc(new_branches, self%branches)
        call move_alloc(new_names, self%branch_names)
        self%n_branches = old_count + 1
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine fanout_pipeline_append

    subroutine fanout_pipeline_fit(self, x, status)
        class(basis_fanout_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. fanout_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline fit: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline fit: input shape is invalid")
            return
        end if
        do i = 1, self%n_branches
            call self%branches(i)%fit(x, status)
            if (status%code /= FORTNUM_OK) return
        end do
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine fanout_pipeline_fit

    subroutine fanout_pipeline_transform(self, x, phi, status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, feature_offset, n_features

        if (.not. fanout_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline transform: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(phi, 1) /= size(x, 1) .or. &
            size(phi, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline transform: array shape is invalid")
            return
        end if
        feature_offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%feature_count()
            call self%branches(i)%transform(x, &
                phi(:, feature_offset + 1:feature_offset + n_features), status)
            if (status%code /= FORTNUM_OK) return
            feature_offset = feature_offset + n_features
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fanout_pipeline_transform

    subroutine fanout_pipeline_jvp(self, x, theta_dot, x_dot, phi, phi_dot, &
            status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_theta_dot(:)
        integer :: i, feature_offset, parameter_offset
        integer :: n_features, n_parameters

        if (.not. fanout_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline jvp: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            size(phi, 1) /= size(x, 1) .or. &
            size(phi, 2) /= self%feature_count() .or. &
            any(shape(phi_dot) /= shape(phi)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline jvp: array shape is invalid")
            return
        end if
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%feature_count()
            n_parameters = self%branches(i)%parameter_count()
            allocate(local_theta_dot(n_parameters))
            if (n_parameters > 0) local_theta_dot = theta_dot(parameter_offset + 1: &
                parameter_offset + n_parameters)
            call self%branches(i)%jvp(x, local_theta_dot, x_dot, &
                phi(:, feature_offset + 1:feature_offset + n_features), &
                phi_dot(:, feature_offset + 1:feature_offset + n_features), &
                status)
            deallocate(local_theta_dot)
            if (status%code /= FORTNUM_OK) return
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fanout_pipeline_jvp

    subroutine fanout_pipeline_vjp(self, x, u, theta_bar, x_bar, status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_theta_bar(:), local_x_bar(:, :)
        integer :: i, feature_offset, parameter_offset
        integer :: n_features, n_parameters

        if (.not. fanout_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline vjp: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(u, 1) /= size(x, 1) .or. &
            size(u, 2) /= self%feature_count() .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline vjp: array shape is invalid")
            return
        end if
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%feature_count()
            n_parameters = self%branches(i)%parameter_count()
            allocate(local_theta_bar(n_parameters))
            allocate(local_x_bar(size(x, 1), size(x, 2)))
            call self%branches(i)%vjp(x, &
                u(:, feature_offset + 1:feature_offset + n_features), &
                local_theta_bar, local_x_bar, status)
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
    end subroutine fanout_pipeline_vjp

    subroutine fanout_pipeline_hvp(self, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_theta_dot(:), local_theta_hvp(:)
        real(dp), allocatable :: local_x_hvp(:, :)
        integer :: i, feature_offset, parameter_offset
        integer :: n_features, n_parameters

        if (.not. fanout_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline hvp: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(u, 1) /= size(x, 1) .or. &
            size(u, 2) /= self%feature_count() .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            size(theta_dot) /= self%parameter_count() .or. &
            size(theta_hvp) /= self%parameter_count() .or. &
            any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline hvp: array shape is invalid")
            return
        end if
        theta_hvp = 0.0_dp
        x_hvp = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%feature_count()
            n_parameters = self%branches(i)%parameter_count()
            allocate(local_theta_dot(n_parameters), local_theta_hvp(n_parameters))
            allocate(local_x_hvp(size(x, 1), size(x, 2)))
            if (n_parameters > 0) local_theta_dot = theta_dot(parameter_offset + 1: &
                parameter_offset + n_parameters)
            call self%branches(i)%hvp(x, &
                u(:, feature_offset + 1:feature_offset + n_features), &
                local_theta_dot, x_dot, local_theta_hvp, local_x_hvp, status)
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
    end subroutine fanout_pipeline_hvp

    subroutine fanout_pipeline_transform_device(self, device, x, phi, status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. fanout_pipeline_device_ready(self, device, status, &
            "basis fanout pipeline transform")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%transform(x, phi, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis fanout pipeline transform: no resident CUDA basis kernel "// &
                "is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline transform: device kind is invalid")
        end select
    end subroutine fanout_pipeline_transform_device

    subroutine fanout_pipeline_jvp_device(self, device, x, theta_dot, x_dot, &
            phi, phi_dot, status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. fanout_pipeline_device_ready(self, device, status, &
            "basis fanout pipeline JVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis fanout pipeline JVP: no resident CUDA basis kernel is "// &
                "linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline JVP: device kind is invalid")
        end select
    end subroutine fanout_pipeline_jvp_device

    subroutine fanout_pipeline_vjp_device(self, device, x, u, theta_bar, x_bar, &
            status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. fanout_pipeline_device_ready(self, device, status, &
            "basis fanout pipeline VJP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%vjp(x, u, theta_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis fanout pipeline VJP: no resident CUDA basis kernel is "// &
                "linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline VJP: device kind is invalid")
        end select
    end subroutine fanout_pipeline_vjp_device

    subroutine fanout_pipeline_hvp_device(self, device, x, u, theta_dot, x_dot, &
            theta_hvp, x_hvp, status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. fanout_pipeline_device_ready(self, device, status, &
            "basis fanout pipeline HVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis fanout pipeline HVP: no resident CUDA basis kernel is "// &
                "linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline HVP: device kind is invalid")
        end select
    end subroutine fanout_pipeline_hvp_device

    integer function fanout_pipeline_input_count(self) result(count)
        class(basis_fanout_pipeline_t), intent(in) :: self

        count = self%n_inputs
    end function fanout_pipeline_input_count

    integer function fanout_pipeline_branch_count(self) result(count)
        class(basis_fanout_pipeline_t), intent(in) :: self

        count = self%n_branches
    end function fanout_pipeline_branch_count

    integer function fanout_pipeline_feature_count(self) result(count)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%branches)) return
        if (self%n_branches < 1 .or. size(self%branches) < self%n_branches) return
        do i = 1, self%n_branches
            count = count + self%branches(i)%feature_count()
        end do
    end function fanout_pipeline_feature_count

    integer function fanout_pipeline_parameter_count(self) result(count)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%branches)) return
        if (self%n_branches < 1 .or. size(self%branches) < self%n_branches) return
        do i = 1, self%n_branches
            count = count + self%branches(i)%parameter_count()
        end do
    end function fanout_pipeline_parameter_count

    function fanout_pipeline_parameters(self) result(theta)
        class(basis_fanout_pipeline_t), intent(in) :: self
        real(dp), allocatable :: theta(:), local_theta(:)
        integer :: i, offset, n_parameters

        allocate(theta(self%parameter_count()))
        theta = 0.0_dp
        offset = 0
        if (.not. allocated(self%branches)) return
        do i = 1, self%n_branches
            n_parameters = self%branches(i)%parameter_count()
            if (n_parameters > 0) then
                local_theta = self%branches(i)%parameters()
                theta(offset + 1:offset + n_parameters) = local_theta
            end if
            offset = offset + n_parameters
        end do
    end function fanout_pipeline_parameters

    subroutine fanout_pipeline_set_parameters(self, theta, status)
        class(basis_fanout_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, offset, n_parameters

        if (.not. fanout_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline set_parameters: model is invalid")
            return
        end if
        if (size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis fanout pipeline set_parameters: shape is invalid")
            return
        end if
        offset = 0
        do i = 1, self%n_branches
            n_parameters = self%branches(i)%parameter_count()
            if (n_parameters > 0) then
                call self%branches(i)%set_parameters(theta(offset + 1: &
                    offset + n_parameters), status)
                if (status%code /= FORTNUM_OK) return
            end if
            offset = offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fanout_pipeline_set_parameters

    function fanout_pipeline_branch_name(self, branch) result(name)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch
        character(:), allocatable :: name

        name = ""
        if (.not. allocated(self%branch_names)) return
        if (branch < 1 .or. branch > self%n_branches) return
        name = trim(self%branch_names(branch))
    end function fanout_pipeline_branch_name

    integer function fanout_pipeline_branch_feature_offset(self, branch) &
            result(offset)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch
        integer :: i

        offset = 0
        if (branch < 1 .or. branch > self%n_branches) return
        if (.not. allocated(self%branches)) return
        offset = 1
        do i = 1, branch - 1
            offset = offset + self%branches(i)%feature_count()
        end do
    end function fanout_pipeline_branch_feature_offset

    integer function fanout_pipeline_branch_parameter_offset(self, branch) &
            result(offset)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch
        integer :: i

        offset = 0
        if (branch < 1 .or. branch > self%n_branches) return
        if (.not. allocated(self%branches)) return
        offset = 1
        do i = 1, branch - 1
            offset = offset + self%branches(i)%parameter_count()
        end do
    end function fanout_pipeline_branch_parameter_offset

    subroutine fanout_pipeline_set_input_schema(self, names, status)
        class(basis_fanout_pipeline_t), intent(inout) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%set_names(names, status)
    end subroutine fanout_pipeline_set_input_schema

    function fanout_pipeline_input_schema_name(self, feature) result(name)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name

        name = self%input_schema%name(feature)
    end function fanout_pipeline_input_schema_name

    subroutine fanout_pipeline_validate_input_schema(self, names, status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%validate_names(names, status)
    end subroutine fanout_pipeline_validate_input_schema

    function fanout_pipeline_feature_name(self, feature) result(name)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name
        integer :: i, offset, n_features
        character(:), allocatable :: local_name

        name = ""
        if (feature < 1 .or. feature > self%feature_count()) return
        offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%feature_count()
            if (feature <= offset + n_features) then
                local_name = self%branches(i)%feature_name(feature - offset)
                if (len_trim(local_name) > 0) then
                    name = trim(self%branch_names(i))//"."//trim(local_name)
                else
                    name = trim(self%branch_names(i))//".feature_"// &
                        integer_text(feature - offset)
                end if
                return
            end if
            offset = offset + n_features
        end do
    end function fanout_pipeline_feature_name

    function fanout_pipeline_parameter_name(self, parameter) result(name)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer, intent(in) :: parameter
        character(:), allocatable :: name
        integer :: i, offset, n_parameters
        character(:), allocatable :: local_name

        name = ""
        if (parameter < 1 .or. parameter > self%parameter_count()) return
        offset = 0
        do i = 1, self%n_branches
            n_parameters = self%branches(i)%parameter_count()
            if (parameter <= offset + n_parameters) then
                local_name = self%branches(i)%parameter_name(parameter - offset)
                if (len_trim(local_name) > 0) then
                    name = trim(self%branch_names(i))//"."//trim(local_name)
                else
                    name = trim(self%branch_names(i))//".parameter_"// &
                        integer_text(parameter - offset)
                end if
                return
            end if
            offset = offset + n_parameters
        end do
    end function fanout_pipeline_parameter_name

    logical function fanout_pipeline_static_lowering_eligible(self) result(eligible)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer :: i

        eligible = fanout_pipeline_valid(self)
        if (.not. eligible) return
        do i = 1, self%n_branches
            if (.not. self%branches(i)%static_lowering_eligible()) then
                eligible = .false.
                return
            end if
        end do
    end function fanout_pipeline_static_lowering_eligible

    logical function fanout_pipeline_valid(self) result(valid)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer :: i

        valid = .true.
        if (self%n_inputs < 1) valid = .false.
        if (.not. self%input_schema%valid()) valid = .false.
        if (.not. allocated(self%branches)) valid = .false.
        if (.not. allocated(self%branch_names)) valid = .false.
        if (self%n_branches < 1) valid = .false.
        if (.not. valid) return
        if (size(self%branches) < self%n_branches .or. &
            size(self%branch_names) < self%n_branches) then
            valid = .false.
            return
        end if
        do i = 1, self%n_branches
            if (.not. self%branches(i)%valid()) then
                valid = .false.
                return
            end if
            if (self%branches(i)%input_count() /= self%n_inputs) then
                valid = .false.
                return
            end if
            if (len_trim(self%branch_names(i)) < 1) then
                valid = .false.
                return
            end if
        end do
    end function fanout_pipeline_valid

    logical function fanout_pipeline_is_fitted(self) result(fitted)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer :: i

        fitted = self%fitted
        if (.not. fitted) return
        if (.not. fanout_pipeline_valid(self)) then
            fitted = .false.
            return
        end if
        do i = 1, self%n_branches
            if (.not. self%branches(i)%is_fitted()) then
                fitted = .false.
                return
            end if
        end do
    end function fanout_pipeline_is_fitted

    subroutine fanout_pipeline_capabilities(self, report, status)
        class(basis_fanout_pipeline_t), intent(in) :: self
        type(estimator_capability_t), intent(out) :: report
        type(fortnum_status_t), intent(out) :: status

        if (.not. fanout_pipeline_valid(self)) then
            call report%initialize("basis_fanout_pipeline", 1, 1, status)
            if (status%code == FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis fanout pipeline capabilities: pipeline is invalid")
            end if
            return
        end if
        report = make_transformer_capabilities("basis_fanout_pipeline", &
            self%n_inputs, self%feature_count(), status, self%is_fitted())
        if (status%code /= FORTNUM_OK) return
        report%supports_input_jvp = .true.
        report%supports_input_vjp = .true.
        report%supports_input_hvp = .true.
        report%supports_parameter_jvp = .true.
        report%supports_parameter_vjp = .true.
        report%supports_parameter_hvp = .true.
        report%supports_cuda = .false.
        report%supports_resident = .false.
    end subroutine fanout_pipeline_capabilities

    logical function fanout_pipeline_device_supported(self, device_kind) &
            result(supported)
        class(basis_fanout_pipeline_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = .false.
        if (.not. fanout_pipeline_valid(self)) return
        if (.not. self%is_fitted()) return
        if (device_kind == FORTML_DEVICE_CPU) supported = .true.
    end function fanout_pipeline_device_supported

    logical function fanout_pipeline_device_ready(self, device, status, operation) &
            result(ready)
        class(basis_fanout_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        ready = .false.
        if (.not. fanout_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model is invalid")
            return
        end if
        if (.not. self%is_fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model is not fitted")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": device is not selected")
            return
        end if
        ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end function fanout_pipeline_device_ready

    subroutine residual_pipeline_initialize(self, n_inputs, status)
        class(basis_residual_pipeline_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status

        self%n_inputs = 0
        self%n_features = 0
        self%main_configured = .false.
        self%residual_configured = .false.
        self%configured = .false.
        self%fitted = .false.
        self%main_name = "main"
        self%residual_name = "residual"
        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline: n_inputs must be positive")
            return
        end if
        self%n_inputs = n_inputs
        call self%input_schema%initialize(n_inputs, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine residual_pipeline_initialize

    subroutine residual_pipeline_set_main(self, branch, status, name)
        class(basis_residual_pipeline_t), intent(inout) :: self
        type(sequential_basis_pipeline_t), intent(in) :: branch
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: name
        character(:), allocatable :: candidate_name

        if (self%n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_main: pipeline is not initialized")
            return
        end if
        if (.not. branch%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_main: branch is invalid")
            return
        end if
        if (branch%input_count() /= self%n_inputs .or. &
            branch%feature_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_main: branch shape is invalid")
            return
        end if
        if (self%residual_configured .and. branch%feature_count() /= &
            self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_main: feature count mismatch")
            return
        end if
        candidate_name = trim(self%main_name)
        if (present(name)) then
            if (len_trim(name) < 1 .or. len_trim(name) > PIPELINE_NAME_LENGTH) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis residual pipeline set_main: name is invalid")
                return
            end if
            candidate_name = trim(name)
        end if
        if (candidate_name == trim(self%residual_name)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_main: branch names must differ")
            return
        end if
        self%main_branch = branch
        self%main_name = candidate_name
        self%main_configured = .true.
        if (.not. self%residual_configured) self%n_features = branch%feature_count()
        self%configured = self%main_configured .and. self%residual_configured
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine residual_pipeline_set_main

    subroutine residual_pipeline_set_residual(self, branch, status, name)
        class(basis_residual_pipeline_t), intent(inout) :: self
        type(sequential_basis_pipeline_t), intent(in) :: branch
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: name
        character(:), allocatable :: candidate_name

        if (self%n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_residual: pipeline is not initialized")
            return
        end if
        if (.not. branch%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_residual: branch is invalid")
            return
        end if
        if (branch%input_count() /= self%n_inputs .or. &
            branch%feature_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_residual: branch shape is invalid")
            return
        end if
        if (self%main_configured .and. branch%feature_count() /= &
            self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_residual: feature count mismatch")
            return
        end if
        candidate_name = trim(self%residual_name)
        if (present(name)) then
            if (len_trim(name) < 1 .or. len_trim(name) > PIPELINE_NAME_LENGTH) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis residual pipeline set_residual: name is invalid")
                return
            end if
            candidate_name = trim(name)
        end if
        if (candidate_name == trim(self%main_name)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_residual: branch names must differ")
            return
        end if
        self%residual_branch = branch
        self%residual_name = candidate_name
        self%residual_configured = .true.
        if (.not. self%main_configured) self%n_features = branch%feature_count()
        self%configured = self%main_configured .and. self%residual_configured
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine residual_pipeline_set_residual

    subroutine residual_pipeline_fit(self, x, status)
        class(basis_residual_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. residual_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline fit: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline fit: input shape is invalid")
            return
        end if
        call self%main_branch%fit(x, status)
        if (status%code /= FORTNUM_OK) return
        call self%residual_branch%fit(x, status)
        if (status%code /= FORTNUM_OK) return
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine residual_pipeline_fit

    subroutine residual_pipeline_transform(self, x, phi, status)
        class(basis_residual_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: main_phi(:, :), residual_phi(:, :)

        if (.not. residual_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline transform: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(phi, 1) /= size(x, 1) .or. &
            size(phi, 2) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline transform: array shape is invalid")
            return
        end if
        allocate(main_phi(size(x, 1), self%n_features))
        allocate(residual_phi(size(x, 1), self%n_features))
        call self%main_branch%transform(x, main_phi, status)
        if (status%code /= FORTNUM_OK) then
            deallocate(main_phi, residual_phi)
            return
        end if
        call self%residual_branch%transform(x, residual_phi, status)
        if (status%code /= FORTNUM_OK) then
            deallocate(main_phi, residual_phi)
            return
        end if
        phi = main_phi + residual_phi
        deallocate(main_phi, residual_phi)
        call status_set(status, FORTNUM_OK, "")
    end subroutine residual_pipeline_transform

    subroutine residual_pipeline_jvp(self, x, theta_dot, x_dot, phi, phi_dot, &
            status)
        class(basis_residual_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: main_phi(:, :), residual_phi(:, :)
        real(dp), allocatable :: main_dot(:, :), residual_dot(:, :)
        real(dp), allocatable :: main_theta_dot(:), residual_theta_dot(:)
        integer :: n_main, n_residual, n_samples

        if (.not. residual_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline jvp: model is invalid")
            return
        end if
        n_main = self%main_branch%parameter_count()
        n_residual = self%residual_branch%parameter_count()
        n_samples = size(x, 1)
        if (n_samples < 1 .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            size(phi, 1) /= n_samples .or. size(phi, 2) /= self%n_features .or. &
            any(shape(phi_dot) /= shape(phi)) .or. &
            size(theta_dot) /= n_main + n_residual) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline jvp: array shape is invalid")
            return
        end if
        allocate(main_phi(n_samples, self%n_features), &
            residual_phi(n_samples, self%n_features))
        allocate(main_dot(n_samples, self%n_features), &
            residual_dot(n_samples, self%n_features))
        allocate(main_theta_dot(n_main), residual_theta_dot(n_residual))
        if (n_main > 0) main_theta_dot = theta_dot(1:n_main)
        if (n_residual > 0) residual_theta_dot = theta_dot(n_main + 1:n_main + n_residual)
        call self%main_branch%jvp(x, main_theta_dot, x_dot, main_phi, main_dot, status)
        if (status%code /= FORTNUM_OK) then
            deallocate(main_phi, residual_phi, main_dot, residual_dot, &
                main_theta_dot, residual_theta_dot)
            return
        end if
        call self%residual_branch%jvp(x, residual_theta_dot, x_dot, residual_phi, &
            residual_dot, status)
        if (status%code /= FORTNUM_OK) then
            deallocate(main_phi, residual_phi, main_dot, residual_dot, &
                main_theta_dot, residual_theta_dot)
            return
        end if
        phi = main_phi + residual_phi
        phi_dot = main_dot + residual_dot
        deallocate(main_phi, residual_phi, main_dot, residual_dot, &
            main_theta_dot, residual_theta_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine residual_pipeline_jvp

    subroutine residual_pipeline_vjp(self, x, u, theta_bar, x_bar, status)
        class(basis_residual_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: main_theta_bar(:), residual_theta_bar(:)
        real(dp), allocatable :: main_x_bar(:, :), residual_x_bar(:, :)
        integer :: n_main, n_residual, n_samples

        if (.not. residual_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline vjp: model is invalid")
            return
        end if
        n_main = self%main_branch%parameter_count()
        n_residual = self%residual_branch%parameter_count()
        n_samples = size(x, 1)
        if (n_samples < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(u, 1) /= n_samples .or. size(u, 2) /= self%n_features .or. &
            size(theta_bar) /= n_main + n_residual .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline vjp: array shape is invalid")
            return
        end if
        allocate(main_theta_bar(n_main), residual_theta_bar(n_residual))
        allocate(main_x_bar(n_samples, self%n_inputs), &
            residual_x_bar(n_samples, self%n_inputs))
        call self%main_branch%vjp(x, u, main_theta_bar, main_x_bar, status)
        if (status%code /= FORTNUM_OK) then
            deallocate(main_theta_bar, residual_theta_bar, main_x_bar, residual_x_bar)
            return
        end if
        call self%residual_branch%vjp(x, u, residual_theta_bar, residual_x_bar, status)
        if (status%code /= FORTNUM_OK) then
            deallocate(main_theta_bar, residual_theta_bar, main_x_bar, residual_x_bar)
            return
        end if
        theta_bar = 0.0_dp
        if (n_main > 0) theta_bar(1:n_main) = main_theta_bar
        if (n_residual > 0) theta_bar(n_main + 1:n_main + n_residual) = residual_theta_bar
        x_bar = main_x_bar + residual_x_bar
        deallocate(main_theta_bar, residual_theta_bar, main_x_bar, residual_x_bar)
        call status_set(status, FORTNUM_OK, "")
    end subroutine residual_pipeline_vjp

    subroutine residual_pipeline_hvp(self, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        class(basis_residual_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: main_theta_dot(:), residual_theta_dot(:)
        real(dp), allocatable :: main_theta_hvp(:), residual_theta_hvp(:)
        real(dp), allocatable :: main_x_hvp(:, :), residual_x_hvp(:, :)
        integer :: n_main, n_residual, n_samples

        if (.not. residual_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline hvp: model is invalid")
            return
        end if
        n_main = self%main_branch%parameter_count()
        n_residual = self%residual_branch%parameter_count()
        n_samples = size(x, 1)
        if (n_samples < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(u, 1) /= n_samples .or. size(u, 2) /= self%n_features .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            size(theta_dot) /= n_main + n_residual .or. &
            size(theta_hvp) /= n_main + n_residual .or. &
            any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline hvp: array shape is invalid")
            return
        end if
        allocate(main_theta_dot(n_main), residual_theta_dot(n_residual))
        allocate(main_theta_hvp(n_main), residual_theta_hvp(n_residual))
        allocate(main_x_hvp(n_samples, self%n_inputs), &
            residual_x_hvp(n_samples, self%n_inputs))
        if (n_main > 0) main_theta_dot = theta_dot(1:n_main)
        if (n_residual > 0) residual_theta_dot = theta_dot(n_main + 1:n_main + n_residual)
        call self%main_branch%hvp(x, u, main_theta_dot, x_dot, main_theta_hvp, &
            main_x_hvp, status)
        if (status%code /= FORTNUM_OK) then
            deallocate(main_theta_dot, residual_theta_dot, main_theta_hvp, &
                residual_theta_hvp, main_x_hvp, residual_x_hvp)
            return
        end if
        call self%residual_branch%hvp(x, u, residual_theta_dot, x_dot, &
            residual_theta_hvp, residual_x_hvp, status)
        if (status%code /= FORTNUM_OK) then
            deallocate(main_theta_dot, residual_theta_dot, main_theta_hvp, &
                residual_theta_hvp, main_x_hvp, residual_x_hvp)
            return
        end if
        theta_hvp = 0.0_dp
        if (n_main > 0) theta_hvp(1:n_main) = main_theta_hvp
        if (n_residual > 0) theta_hvp(n_main + 1:n_main + n_residual) = residual_theta_hvp
        x_hvp = main_x_hvp + residual_x_hvp
        deallocate(main_theta_dot, residual_theta_dot, main_theta_hvp, &
            residual_theta_hvp, main_x_hvp, residual_x_hvp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine residual_pipeline_hvp

    subroutine residual_pipeline_transform_device(self, device, x, phi, status)
        class(basis_residual_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. residual_pipeline_device_ready(self, device, status, &
            "basis residual pipeline transform")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%transform(x, phi, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis residual pipeline transform: no resident CUDA basis kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline transform: device kind is invalid")
        end select
    end subroutine residual_pipeline_transform_device

    subroutine residual_pipeline_jvp_device(self, device, x, theta_dot, x_dot, &
            phi, phi_dot, status)
        class(basis_residual_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(inout) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. residual_pipeline_device_ready(self, device, status, &
            "basis residual pipeline JVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis residual pipeline JVP: no resident CUDA basis kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline JVP: device kind is invalid")
        end select
    end subroutine residual_pipeline_jvp_device

    subroutine residual_pipeline_vjp_device(self, device, x, u, theta_bar, x_bar, &
            status)
        class(basis_residual_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(inout) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. residual_pipeline_device_ready(self, device, status, &
            "basis residual pipeline VJP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%vjp(x, u, theta_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis residual pipeline VJP: no resident CUDA basis kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline VJP: device kind is invalid")
        end select
    end subroutine residual_pipeline_vjp_device

    subroutine residual_pipeline_hvp_device(self, device, x, u, theta_dot, x_dot, &
            theta_hvp, x_hvp, status)
        class(basis_residual_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(inout) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. residual_pipeline_device_ready(self, device, status, &
            "basis residual pipeline HVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis residual pipeline HVP: no resident CUDA basis kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline HVP: device kind is invalid")
        end select
    end subroutine residual_pipeline_hvp_device

    integer function residual_pipeline_input_count(self) result(count)
        class(basis_residual_pipeline_t), intent(in) :: self

        count = self%n_inputs
    end function residual_pipeline_input_count

    integer function residual_pipeline_feature_count(self) result(count)
        class(basis_residual_pipeline_t), intent(in) :: self

        count = self%n_features
        if (.not. self%configured) count = 0
    end function residual_pipeline_feature_count

    integer function residual_pipeline_parameter_count(self) result(count)
        class(basis_residual_pipeline_t), intent(in) :: self

        count = 0
        if (.not. self%configured) return
        count = self%main_branch%parameter_count() + &
            self%residual_branch%parameter_count()
    end function residual_pipeline_parameter_count

    function residual_pipeline_parameters(self) result(theta)
        class(basis_residual_pipeline_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        real(dp), allocatable :: main_theta(:), residual_theta(:)
        integer :: n_main, n_residual

        allocate(theta(self%parameter_count()))
        theta = 0.0_dp
        if (.not. self%configured) return
        n_main = self%main_branch%parameter_count()
        n_residual = self%residual_branch%parameter_count()
        if (n_main > 0) then
            main_theta = self%main_branch%parameters()
            theta(1:n_main) = main_theta
        end if
        if (n_residual > 0) then
            residual_theta = self%residual_branch%parameters()
            theta(n_main + 1:n_main + n_residual) = residual_theta
        end if
    end function residual_pipeline_parameters

    subroutine residual_pipeline_set_parameters(self, theta, status)
        class(basis_residual_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_main, n_residual

        if (.not. residual_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_parameters: model is invalid")
            return
        end if
        n_main = self%main_branch%parameter_count()
        n_residual = self%residual_branch%parameter_count()
        if (size(theta) /= n_main + n_residual) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis residual pipeline set_parameters: shape is invalid")
            return
        end if
        if (n_main > 0) then
            call self%main_branch%set_parameters(theta(1:n_main), status)
            if (status%code /= FORTNUM_OK) return
        end if
        if (n_residual > 0) then
            call self%residual_branch%set_parameters(theta(n_main + 1:n_main + n_residual), &
                status)
            if (status%code /= FORTNUM_OK) return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine residual_pipeline_set_parameters

    function residual_pipeline_branch_name(self, branch) result(name)
        class(basis_residual_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch
        character(:), allocatable :: name

        name = ""
        select case (branch)
        case (1)
            name = trim(self%main_name)
        case (2)
            name = trim(self%residual_name)
        end select
    end function residual_pipeline_branch_name

    function residual_pipeline_feature_name(self, feature) result(name)
        class(basis_residual_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name

        name = ""
        if (feature < 1 .or. feature > self%feature_count()) return
        name = "residual_sum.feature_"//integer_text(feature)
    end function residual_pipeline_feature_name

    function residual_pipeline_parameter_name(self, parameter) result(name)
        class(basis_residual_pipeline_t), intent(in) :: self
        integer, intent(in) :: parameter
        character(:), allocatable :: name
        integer :: n_main
        character(:), allocatable :: local_name

        name = ""
        if (parameter < 1 .or. parameter > self%parameter_count()) return
        n_main = self%main_branch%parameter_count()
        if (parameter <= n_main) then
            local_name = self%main_branch%parameter_name(parameter)
            name = trim(self%main_name)//"."//trim(local_name)
        else
            local_name = self%residual_branch%parameter_name(parameter - n_main)
            name = trim(self%residual_name)//"."//trim(local_name)
        end if
    end function residual_pipeline_parameter_name

    integer function residual_pipeline_main_feature_offset(self) result(offset)
        class(basis_residual_pipeline_t), intent(in) :: self

        offset = 0
        if (self%configured) offset = 1
    end function residual_pipeline_main_feature_offset

    integer function residual_pipeline_residual_feature_offset(self) result(offset)
        class(basis_residual_pipeline_t), intent(in) :: self

        offset = 0
        if (self%configured) offset = 1
    end function residual_pipeline_residual_feature_offset

    integer function residual_pipeline_main_parameter_offset(self) result(offset)
        class(basis_residual_pipeline_t), intent(in) :: self

        offset = 0
        if (self%configured) offset = 1
    end function residual_pipeline_main_parameter_offset

    integer function residual_pipeline_residual_parameter_offset(self) result(offset)
        class(basis_residual_pipeline_t), intent(in) :: self

        offset = 0
        if (self%configured) offset = self%main_branch%parameter_count() + 1
    end function residual_pipeline_residual_parameter_offset

    subroutine residual_pipeline_set_input_schema(self, names, status)
        class(basis_residual_pipeline_t), intent(inout) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%set_names(names, status)
    end subroutine residual_pipeline_set_input_schema

    function residual_pipeline_input_schema_name(self, feature) result(name)
        class(basis_residual_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name

        name = self%input_schema%name(feature)
    end function residual_pipeline_input_schema_name

    subroutine residual_pipeline_validate_input_schema(self, names, status)
        class(basis_residual_pipeline_t), intent(in) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%validate_names(names, status)
    end subroutine residual_pipeline_validate_input_schema

    logical function residual_pipeline_static_lowering_eligible(self) result(eligible)
        class(basis_residual_pipeline_t), intent(in) :: self

        eligible = residual_pipeline_valid(self)
        if (.not. eligible) return
        eligible = self%main_branch%static_lowering_eligible() .and. &
            self%residual_branch%static_lowering_eligible()
    end function residual_pipeline_static_lowering_eligible

    logical function residual_pipeline_valid(self) result(valid)
        class(basis_residual_pipeline_t), intent(in) :: self

        valid = self%n_inputs > 0 .and. self%configured .and. &
            self%main_configured .and. self%residual_configured .and. &
            self%input_schema%valid()
        if (.not. valid) return
        if (.not. self%main_branch%valid() .or. &
            .not. self%residual_branch%valid()) then
            valid = .false.
            return
        end if
        if (self%main_branch%input_count() /= self%n_inputs .or. &
            self%residual_branch%input_count() /= self%n_inputs .or. &
            self%main_branch%feature_count() /= self%n_features .or. &
            self%residual_branch%feature_count() /= self%n_features) then
            valid = .false.
            return
        end if
        if (trim(self%main_name) == trim(self%residual_name)) valid = .false.
    end function residual_pipeline_valid

    logical function residual_pipeline_is_fitted(self) result(fitted)
        class(basis_residual_pipeline_t), intent(in) :: self

        fitted = self%fitted
        if (.not. fitted) return
        if (.not. residual_pipeline_valid(self)) then
            fitted = .false.
            return
        end if
        fitted = self%main_branch%is_fitted() .and. &
            self%residual_branch%is_fitted()
    end function residual_pipeline_is_fitted

    subroutine residual_pipeline_capabilities(self, report, status)
        class(basis_residual_pipeline_t), intent(in) :: self
        type(estimator_capability_t), intent(out) :: report
        type(fortnum_status_t), intent(out) :: status

        if (.not. residual_pipeline_valid(self)) then
            call report%initialize("basis_residual_pipeline", 1, 1, status)
            if (status%code == FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis residual pipeline capabilities: pipeline is invalid")
            end if
            return
        end if
        report = make_transformer_capabilities("basis_residual_pipeline", &
            self%n_inputs, self%n_features, status, self%is_fitted())
        if (status%code /= FORTNUM_OK) return
        report%supports_input_jvp = .true.
        report%supports_input_vjp = .true.
        report%supports_input_hvp = .true.
        report%supports_parameter_jvp = .true.
        report%supports_parameter_vjp = .true.
        report%supports_parameter_hvp = .true.
        report%supports_cuda = .false.
        report%supports_resident = .false.
    end subroutine residual_pipeline_capabilities

    logical function residual_pipeline_device_supported(self, device_kind) &
            result(supported)
        class(basis_residual_pipeline_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = .false.
        if (.not. residual_pipeline_valid(self)) return
        if (.not. self%is_fitted()) return
        if (device_kind == FORTML_DEVICE_CPU) supported = .true.
    end function residual_pipeline_device_supported

    logical function residual_pipeline_device_ready(self, device, status, operation) &
            result(ready)
        class(basis_residual_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        ready = .false.
        if (.not. residual_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model is invalid")
            return
        end if
        if (.not. self%is_fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model is not fitted")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": device is not selected")
            return
        end if
        ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end function residual_pipeline_device_ready

    function default_branch_name(index) result(name)
        integer, intent(in) :: index
        character(:), allocatable :: name
        character(len=32) :: buffer

        write (buffer, '("branch_",i0)') index
        name = trim(buffer)
    end function default_branch_name

    function integer_text(value) result(text)
        integer, intent(in) :: value
        character(:), allocatable :: text
        character(len=32) :: buffer

        write (buffer, '(i0)') value
        text = trim(buffer)
    end function integer_text

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
