!> Differentiable learned fan-in for same-shape basis pipeline branches.
module fortml_basis_blend_pipeline
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_estimator_capabilities, only: estimator_capability_t, &
        make_transformer_capabilities, FORTML_CAPABILITY_DEVICE_OPENACC
    use fortml_pipeline, only: sequential_basis_pipeline_t, basis_input_schema_t
    implicit none
    private

    integer, parameter :: NAME_LENGTH = 128

    !> A learnable weighted sum of named sequential basis pipelines.
    !>
    !> Every branch consumes the same input matrix and returns the same feature
    !> shape. The output is `sum_i weight_i * branch_i(x)`. Mixing weights and
    !> branch parameters share one stable packed parameter vector. All updates
    !> are validated on a deep candidate before they replace the live graph.
    type, public :: basis_blend_pipeline_t
        private
        integer :: n_inputs = 0
        integer :: n_features = 0
        integer :: n_branches = 0
        logical :: fitted = .false.
        type(sequential_basis_pipeline_t), allocatable :: branches(:)
        real(dp), allocatable :: weights(:)
        character(len=NAME_LENGTH), allocatable :: branch_names(:)
        type(basis_input_schema_t) :: input_schema
    contains
        procedure, public :: initialize => blend_initialize
        procedure, public :: append => blend_append
        procedure, public :: fit => blend_fit
        procedure, public :: transform => blend_transform
        procedure, public :: evaluate => blend_transform
        procedure, public :: transform_device => blend_transform_device
        procedure, public :: jvp => blend_jvp
        procedure, public :: jvp_device => blend_jvp_device
        procedure, public :: vjp => blend_vjp
        procedure, public :: vjp_device => blend_vjp_device
        procedure, public :: hvp => blend_hvp
        procedure, public :: hvp_device => blend_hvp_device
        procedure, public :: input_count => blend_input_count
        procedure, public :: feature_count => blend_feature_count
        procedure, public :: branch_count => blend_branch_count
        procedure, public :: parameter_count => blend_parameter_count
        procedure, public :: parameters => blend_parameters
        procedure, public :: set_parameters => blend_set_parameters
        procedure, public :: branch_name => blend_branch_name
        procedure, public :: feature_name => blend_feature_name
        procedure, public :: parameter_name => blend_parameter_name
        procedure, public :: branch_parameter_offset => &
            blend_branch_parameter_offset
        procedure, public :: branch_weight => blend_branch_weight
        procedure, public :: set_input_schema => blend_set_input_schema
        procedure, public :: input_schema_name => blend_input_schema_name
        procedure, public :: validate_input_schema => blend_validate_input_schema
        procedure, public :: static_lowering_eligible => &
            blend_static_lowering_eligible
        procedure, public :: capabilities => blend_capabilities
        procedure, public :: device_supported => blend_device_supported
        procedure, public :: valid => blend_valid
        procedure, public :: is_fitted => blend_is_fitted
    end type basis_blend_pipeline_t

    public :: make_basis_blend_pipeline

contains

    function make_basis_blend_pipeline(n_inputs, status) result(pipeline)
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status
        type(basis_blend_pipeline_t) :: pipeline

        call pipeline%initialize(n_inputs, status)
    end function make_basis_blend_pipeline

    subroutine blend_initialize(self, n_inputs, status)
        class(basis_blend_pipeline_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status

        self%n_inputs = 0
        self%n_features = 0
        self%n_branches = 0
        self%fitted = .false.
        allocate(self%branches(0), self%weights(0), self%branch_names(0))
        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline: n_inputs must be positive")
            return
        end if
        call self%input_schema%initialize(n_inputs, status)
        if (status%code /= FORTNUM_OK) return
        self%n_inputs = n_inputs
        call status_set(status, FORTNUM_OK, "")
    end subroutine blend_initialize

    !> Append one branch transactionally.
    subroutine blend_append(self, branch, weight, status, name)
        class(basis_blend_pipeline_t), intent(inout) :: self
        type(sequential_basis_pipeline_t), intent(in) :: branch
        real(dp), intent(in) :: weight
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: name
        type(sequential_basis_pipeline_t), allocatable :: candidate_branches(:)
        real(dp), allocatable :: candidate_weights(:)
        character(len=NAME_LENGTH), allocatable :: candidate_names(:)
        character(len=NAME_LENGTH) :: candidate_name
        integer :: i, old_count

        if (.not. blend_initialized(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline append: pipeline is not initialized")
            return
        end if
        if (.not. branch%valid() .or. branch%input_count() /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline append: branch shape is invalid")
            return
        end if
        if (self%n_branches > 0) then
            if (branch%feature_count() /= self%n_features) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis blend pipeline append: feature count must match")
                return
            end if
        end if
        if (.not. ieee_is_finite(weight)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline append: weight must be finite")
            return
        end if
        write (candidate_name, '(a,i0)') "branch_", self%n_branches + 1
        if (present(name)) then
            if (len_trim(name) < 1 .or. len_trim(name) > NAME_LENGTH) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis blend pipeline append: branch name is invalid")
                return
            end if
            candidate_name = trim(name)
        end if
        do i = 1, self%n_branches
            if (trim(self%branch_names(i)) == trim(candidate_name)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis blend pipeline append: branch names must be unique")
                return
            end if
        end do

        old_count = self%n_branches
        allocate(candidate_branches(old_count + 1))
        allocate(candidate_weights(old_count + 1))
        allocate(candidate_names(old_count + 1))
        if (old_count > 0) then
            candidate_branches(1:old_count) = self%branches
            candidate_weights(1:old_count) = self%weights
            candidate_names(1:old_count) = self%branch_names
        end if
        candidate_branches(old_count + 1) = branch
        candidate_weights(old_count + 1) = weight
        candidate_names(old_count + 1) = candidate_name
        call move_alloc(candidate_branches, self%branches)
        call move_alloc(candidate_weights, self%weights)
        call move_alloc(candidate_names, self%branch_names)
        self%n_branches = old_count + 1
        self%n_features = branch%feature_count()
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine blend_append

    subroutine blend_fit(self, x, status)
        class(basis_blend_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(basis_blend_pipeline_t) :: candidate
        integer :: i

        if (.not. self%valid() .or. .not. valid_input(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline fit: model or input shape is invalid")
            return
        end if
        candidate = self
        do i = 1, candidate%n_branches
            call candidate%branches(i)%fit(x, status)
            if (status%code /= FORTNUM_OK) return
        end do
        candidate%fitted = .true.
        select type (self)
            type is (basis_blend_pipeline_t)
            self = candidate
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine blend_fit

    subroutine blend_transform(self, x, y, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), local_y(:, :)
        integer :: i

        if (.not. blend_arrays_ready(self, x, y, status, "transform")) return
        allocate(candidate(size(y, 1), size(y, 2)))
        allocate(local_y(size(y, 1), size(y, 2)))
        candidate = 0.0_dp
        do i = 1, self%n_branches
            call self%branches(i)%transform(x, local_y, status)
            if (status%code /= FORTNUM_OK) return
            candidate = candidate + self%weights(i)*local_y
        end do
        y = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine blend_transform

    subroutine blend_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), candidate_dot(:, :)
        real(dp), allocatable :: local_y(:, :), local_dot(:, :), local_theta(:)
        integer :: i, offset, n_local

        if (.not. blend_products_ready(self, x, status, "JVP")) return
        if (size(theta_dot) /= self%parameter_count() .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            size(y, 1) /= size(x, 1) .or. &
            size(y, 2) /= self%n_features .or. &
            any(shape(y_dot) /= shape(y)) .or. &
            .not. all(ieee_is_finite(theta_dot)) .or. &
            .not. all(ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline JVP: array shape or tangent is invalid")
            return
        end if
        allocate(candidate(size(y, 1), size(y, 2)), &
            candidate_dot(size(y, 1), size(y, 2)))
        allocate(local_y(size(y, 1), size(y, 2)), &
            local_dot(size(y, 1), size(y, 2)))
        candidate = 0.0_dp
        candidate_dot = 0.0_dp
        offset = 0
        do i = 1, self%n_branches
            n_local = self%branches(i)%parameter_count()
            allocate(local_theta(n_local))
            if (n_local > 0) local_theta = theta_dot(offset + 2: &
                offset + 1 + n_local)
            call self%branches(i)%jvp(x, local_theta, x_dot, local_y, local_dot, &
                status)
            deallocate(local_theta)
            if (status%code /= FORTNUM_OK) return
            candidate = candidate + self%weights(i)*local_y
            candidate_dot = candidate_dot + self%weights(i)*local_dot + &
                theta_dot(offset + 1)*local_y
            offset = offset + 1 + n_local
        end do
        y = candidate
        y_dot = candidate_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine blend_jvp

    subroutine blend_vjp(self, x, u, theta_bar, x_bar, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate_theta(:), candidate_x(:, :)
        real(dp), allocatable :: local_y(:, :), local_theta(:), local_x(:, :)
        integer :: i, offset, n_local

        if (.not. blend_products_ready(self, x, status, "VJP")) return
        if (size(u, 1) /= size(x, 1) .or. size(u, 2) /= self%n_features .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            .not. all(ieee_is_finite(u))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline VJP: array shape or cotangent is invalid")
            return
        end if
        allocate(candidate_theta(size(theta_bar)), candidate_x(size(x, 1), &
            size(x, 2)), local_y(size(x, 1), self%n_features), &
            local_x(size(x, 1), size(x, 2)))
        candidate_theta = 0.0_dp
        candidate_x = 0.0_dp
        offset = 0
        do i = 1, self%n_branches
            n_local = self%branches(i)%parameter_count()
            allocate(local_theta(n_local))
            call self%branches(i)%transform(x, local_y, status)
            if (status%code /= FORTNUM_OK) return
            call self%branches(i)%vjp(x, self%weights(i)*u, local_theta, &
                local_x, status)
            if (status%code /= FORTNUM_OK) return
            candidate_theta(offset + 1) = sum(u*local_y)
            if (n_local > 0) candidate_theta(offset + 2:offset + 1 + n_local) = &
                local_theta
            candidate_x = candidate_x + local_x
            deallocate(local_theta)
            offset = offset + 1 + n_local
        end do
        theta_bar = candidate_theta
        x_bar = candidate_x
        call status_set(status, FORTNUM_OK, "")
    end subroutine blend_vjp

    subroutine blend_hvp(self, x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate_theta(:), candidate_x(:, :)
        real(dp), allocatable :: local_theta_dot(:), local_theta_bar(:)
        real(dp), allocatable :: local_theta_hvp(:), local_x_bar(:, :)
        real(dp), allocatable :: local_x_hvp(:, :), local_y(:, :), local_dot(:, :)
        integer :: i, offset, n_local
        real(dp) :: weight_dot

        if (.not. blend_products_ready(self, x, status, "HVP")) return
        if (size(u, 1) /= size(x, 1) .or. size(u, 2) /= self%n_features .or. &
            size(theta_dot) /= self%parameter_count() .or. &
            size(theta_hvp) /= self%parameter_count() .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(x_hvp) /= shape(x)) .or. &
            .not. all(ieee_is_finite(u)) .or. &
            .not. all(ieee_is_finite(theta_dot)) .or. &
            .not. all(ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline HVP: array shape or direction is invalid")
            return
        end if
        allocate(candidate_theta(size(theta_hvp)), candidate_x(size(x, 1), &
            size(x, 2)), local_y(size(x, 1), self%n_features), &
            local_dot(size(x, 1), self%n_features), &
            local_x_bar(size(x, 1), size(x, 2)), &
            local_x_hvp(size(x, 1), size(x, 2)))
        candidate_theta = 0.0_dp
        candidate_x = 0.0_dp
        offset = 0
        do i = 1, self%n_branches
            n_local = self%branches(i)%parameter_count()
            allocate(local_theta_dot(n_local), local_theta_bar(n_local), &
                local_theta_hvp(n_local))
            if (n_local > 0) local_theta_dot = theta_dot(offset + 2: &
                offset + 1 + n_local)
            weight_dot = theta_dot(offset + 1)
            call self%branches(i)%jvp(x, local_theta_dot, x_dot, local_y, &
                local_dot, status)
            if (status%code /= FORTNUM_OK) return
            call self%branches(i)%vjp(x, u, local_theta_bar, local_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            call self%branches(i)%hvp(x, u, local_theta_dot, x_dot, &
                local_theta_hvp, local_x_hvp, status)
            if (status%code /= FORTNUM_OK) return
            candidate_theta(offset + 1) = sum(u*local_dot)
            if (n_local > 0) candidate_theta(offset + 2:offset + 1 + n_local) = &
                self%weights(i)*local_theta_hvp + weight_dot*local_theta_bar
            candidate_x = candidate_x + self%weights(i)*local_x_hvp + &
                weight_dot*local_x_bar
            deallocate(local_theta_dot, local_theta_bar, local_theta_hvp)
            offset = offset + 1 + n_local
        end do
        theta_hvp = candidate_theta
        x_hvp = candidate_x
        call status_set(status, FORTNUM_OK, "")
    end subroutine blend_hvp

    subroutine blend_transform_device(self, device, x, y, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. blend_device_ready(self, device, status, &
            "basis blend pipeline transform")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%transform(x, y, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis blend pipeline transform: resident CUDA kernel is unavailable")
        case (FORTML_CAPABILITY_DEVICE_OPENACC)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis blend pipeline transform: OpenACC backend is unavailable")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline transform: device kind is invalid")
        end select
    end subroutine blend_transform_device

    subroutine blend_jvp_device(self, device, x, theta_dot, x_dot, y, y_dot, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(inout) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. blend_device_ready(self, device, status, &
            "basis blend pipeline JVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%jvp(x, theta_dot, x_dot, y, y_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis blend pipeline JVP: resident CUDA kernel is unavailable")
        case (FORTML_CAPABILITY_DEVICE_OPENACC)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis blend pipeline JVP: OpenACC backend is unavailable")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline JVP: device kind is invalid")
        end select
    end subroutine blend_jvp_device

    subroutine blend_vjp_device(self, device, x, u, theta_bar, x_bar, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(inout) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. blend_device_ready(self, device, status, &
            "basis blend pipeline VJP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%vjp(x, u, theta_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis blend pipeline VJP: resident CUDA kernel is unavailable")
        case (FORTML_CAPABILITY_DEVICE_OPENACC)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis blend pipeline VJP: OpenACC backend is unavailable")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline VJP: device kind is invalid")
        end select
    end subroutine blend_vjp_device

    subroutine blend_hvp_device(self, device, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(inout) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. blend_device_ready(self, device, status, &
            "basis blend pipeline HVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis blend pipeline HVP: resident CUDA kernel is unavailable")
        case (FORTML_CAPABILITY_DEVICE_OPENACC)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis blend pipeline HVP: OpenACC backend is unavailable")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline HVP: device kind is invalid")
        end select
    end subroutine blend_hvp_device

    integer function blend_input_count(self) result(count)
        class(basis_blend_pipeline_t), intent(in) :: self
        count = self%n_inputs
    end function blend_input_count

    integer function blend_feature_count(self) result(count)
        class(basis_blend_pipeline_t), intent(in) :: self
        count = self%n_features
        if (self%n_branches < 1) count = 0
    end function blend_feature_count

    integer function blend_branch_count(self) result(count)
        class(basis_blend_pipeline_t), intent(in) :: self
        count = self%n_branches
    end function blend_branch_count

    integer function blend_parameter_count(self) result(count)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer :: i

        count = self%n_branches
        do i = 1, self%n_branches
            count = count + self%branches(i)%parameter_count()
        end do
    end function blend_parameter_count

    function blend_parameters(self) result(theta)
        class(basis_blend_pipeline_t), intent(in) :: self
        real(dp), allocatable :: theta(:), local_theta(:)
        integer :: i, offset, n_local

        allocate(theta(self%parameter_count()))
        theta = 0.0_dp
        offset = 0
        do i = 1, self%n_branches
            theta(offset + 1) = self%weights(i)
            n_local = self%branches(i)%parameter_count()
            if (n_local > 0) then
                local_theta = self%branches(i)%parameters()
                theta(offset + 2:offset + 1 + n_local) = local_theta
            end if
            offset = offset + 1 + n_local
        end do
    end function blend_parameters

    subroutine blend_set_parameters(self, theta, status)
        class(basis_blend_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        type(basis_blend_pipeline_t) :: candidate
        integer :: i, offset, n_local

        if (.not. self%valid() .or. size(theta) /= self%parameter_count() .or. &
            .not. all(ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline set_parameters: parameter vector is invalid")
            return
        end if
        candidate = self
        offset = 0
        do i = 1, candidate%n_branches
            candidate%weights(i) = theta(offset + 1)
            n_local = candidate%branches(i)%parameter_count()
            if (n_local > 0) then
                call candidate%branches(i)%set_parameters(theta(offset + 2: &
                    offset + 1 + n_local), status)
                if (status%code /= FORTNUM_OK) return
            end if
            offset = offset + 1 + n_local
        end do
        select type (self)
            type is (basis_blend_pipeline_t)
            self = candidate
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine blend_set_parameters

    function blend_branch_name(self, branch) result(name)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch
        character(:), allocatable :: name

        name = ""
        if (branch < 1 .or. branch > self%n_branches) return
        name = trim(self%branch_names(branch))
    end function blend_branch_name

    function blend_feature_name(self, feature) result(name)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name
        character(len=32) :: index_text

        name = ""
        if (feature < 1 .or. feature > self%feature_count()) return
        write (index_text, '(i0)') feature
        name = "blend.feature_"//trim(index_text)
    end function blend_feature_name

    function blend_parameter_name(self, parameter) result(name)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer, intent(in) :: parameter
        character(:), allocatable :: name
        character(len=32) :: index_text
        integer :: i, offset, n_local

        name = ""
        if (parameter < 1 .or. parameter > self%parameter_count()) return
        offset = 0
        do i = 1, self%n_branches
            n_local = self%branches(i)%parameter_count()
            if (parameter == offset + 1) then
                name = trim(self%branch_names(i))//".weight"
                return
            end if
            if (parameter <= offset + 1 + n_local) then
                write (index_text, '(i0)') parameter - offset - 1
                name = trim(self%branch_names(i))//".parameter_"// &
                    trim(index_text)
                return
            end if
            offset = offset + 1 + n_local
        end do
    end function blend_parameter_name

    integer function blend_branch_parameter_offset(self, branch) result(offset)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch
        integer :: i

        offset = 0
        if (branch < 1 .or. branch > self%n_branches) return
        offset = 1
        do i = 1, branch - 1
            offset = offset + 1 + self%branches(i)%parameter_count()
        end do
    end function blend_branch_parameter_offset

    real(dp) function blend_branch_weight(self, branch) result(weight)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch

        weight = 0.0_dp
        if (branch < 1 .or. branch > self%n_branches) return
        weight = self%weights(branch)
    end function blend_branch_weight

    subroutine blend_set_input_schema(self, names, status)
        class(basis_blend_pipeline_t), intent(inout) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%set_names(names, status)
    end subroutine blend_set_input_schema

    function blend_input_schema_name(self, feature) result(name)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name

        name = self%input_schema%name(feature)
    end function blend_input_schema_name

    subroutine blend_validate_input_schema(self, names, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%validate_names(names, status)
    end subroutine blend_validate_input_schema

    logical function blend_static_lowering_eligible(self) result(eligible)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer :: i

        eligible = self%valid()
        if (.not. eligible) return
        do i = 1, self%n_branches
            if (.not. self%branches(i)%static_lowering_eligible()) then
                eligible = .false.
                return
            end if
        end do
    end function blend_static_lowering_eligible

    subroutine blend_capabilities(self, report, status)
        class(basis_blend_pipeline_t), intent(in) :: self
        type(estimator_capability_t), intent(out) :: report
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%valid()) then
            call report%initialize("basis_blend_pipeline", 1, 1, status)
            if (status%code == FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis blend pipeline capabilities: pipeline is invalid")
            end if
            return
        end if
        report = make_transformer_capabilities("basis_blend_pipeline", &
            self%n_inputs, self%n_features, status, self%is_fitted())
        if (status%code /= FORTNUM_OK) return
        report%supports_input_jvp = .true.
        report%supports_input_vjp = .true.
        report%supports_input_hvp = .true.
        report%supports_parameter_jvp = .true.
        report%supports_parameter_vjp = .true.
        report%supports_parameter_hvp = .true.
        report%supports_cuda = .false.
        report%supports_openacc = .false.
        report%supports_resident = .false.
    end subroutine blend_capabilities

    logical function blend_device_supported(self, device_kind) result(supported)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%is_fitted() .and. device_kind == FORTML_DEVICE_CPU
    end function blend_device_supported

    logical function blend_valid(self) result(valid)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer :: i, j

        valid = blend_initialized(self)
        if (.not. valid) return
        valid = self%n_branches > 0 .and. self%n_features > 0
        if (.not. valid) return
        if (size(self%branches) /= self%n_branches .or. &
            size(self%weights) /= self%n_branches .or. &
            size(self%branch_names) /= self%n_branches) then
            valid = .false.
            return
        end if
        if (.not. all(ieee_is_finite(self%weights))) then
            valid = .false.
            return
        end if
        do i = 1, self%n_branches
            if (.not. self%branches(i)%valid()) then
                valid = .false.
                return
            end if
            if (self%branches(i)%input_count() /= self%n_inputs .or. &
                self%branches(i)%feature_count() /= self%n_features .or. &
                len_trim(self%branch_names(i)) < 1) then
                valid = .false.
                return
            end if
            do j = 1, i - 1
                if (trim(self%branch_names(j)) == trim(self%branch_names(i))) then
                    valid = .false.
                    return
                end if
            end do
        end do
    end function blend_valid

    logical function blend_is_fitted(self) result(fitted)
        class(basis_blend_pipeline_t), intent(in) :: self
        integer :: i

        fitted = self%fitted .and. self%valid()
        if (.not. fitted) return
        do i = 1, self%n_branches
            if (.not. self%branches(i)%is_fitted()) then
                fitted = .false.
                return
            end if
        end do
    end function blend_is_fitted

    logical function blend_initialized(self) result(initialized)
        class(basis_blend_pipeline_t), intent(in) :: self

        initialized = self%n_inputs > 0 .and. self%input_schema%valid() .and. &
            allocated(self%branches) .and. allocated(self%weights) .and. &
            allocated(self%branch_names)
    end function blend_initialized

    logical function valid_input(self, x) result(valid)
        class(basis_blend_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)

        valid = size(x, 1) > 0 .and. size(x, 2) == self%n_inputs
        if (.not. valid) return
        valid = all(ieee_is_finite(x))
    end function valid_input

    logical function blend_products_ready(self, x, status, operation) result(ready)
        class(basis_blend_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        ready = .false.
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline "//trim(operation)//": model is invalid")
            return
        end if
        if (.not. self%is_fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline "//trim(operation)//": model is not fitted")
            return
        end if
        if (.not. valid_input(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline "//trim(operation)//": input is invalid")
            return
        end if
        ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end function blend_products_ready

    logical function blend_arrays_ready(self, x, y, status, operation) result(ready)
        class(basis_blend_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(in) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        ready = blend_products_ready(self, x, status, operation)
        if (.not. ready) return
        if (size(y, 1) /= size(x, 1) .or. size(y, 2) /= self%n_features) then
            ready = .false.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis blend pipeline "//trim(operation)//": output shape is invalid")
        end if
    end function blend_arrays_ready

    logical function blend_device_ready(self, device, status, operation) result(ready)
        class(basis_blend_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        ready = .false.
        if (.not. self%is_fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": pipeline is invalid or not fitted")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": device is not selected")
            return
        end if
        ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end function blend_device_ready

end module fortml_basis_blend_pipeline
