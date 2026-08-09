!> Piecewise conditional feature unions built from column basis pipelines.
module fortml_conditional_pipeline
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_basis, only: basis_map_t
    use fortml_column_pipeline, only: column_basis_pipeline_t, &
        make_column_basis_pipeline
    use fortml_pipeline, only: basis_input_schema_t
    use fortml_estimator_capabilities, only: estimator_capability_t, &
        make_transformer_capabilities
    implicit none
    private

    integer, parameter :: NAME_LENGTH = 128

    !> One half-open interval-routed column feature branch.
    !>
    !> The branch receives the original dense matrix through the existing
    !> column-basis pipeline.  It contributes zeros for rows outside
    !> ``[lower_bound,upper_bound)``.  The route column may also be consumed by
    !> the basis stage; in that case its ordinary stage derivative is retained.
    type :: conditional_branch_t
        type(column_basis_pipeline_t) :: pipeline
        integer :: route_column = 0
        real(dp) :: lower_bound = 0.0_dp
        real(dp) :: upper_bound = 0.0_dp
        character(len=NAME_LENGTH) :: name = ""
    end type conditional_branch_t

    !> A named parallel feature union with piecewise row routing.
    !>
    !> Branches are evaluated independently and concatenated in append order.
    !> Each branch owns one selected-column basis pipeline and a half-open
    !> interval on one original input column.  Intervals may overlap or leave
    !> gaps; this permits mixture-of-experts and explicit fallback branches.
    !> At an interval endpoint the value is still defined, but derivatives are
    !> refused because the route is discontinuous.  Configuration and packed
    !> parameter updates are transactional.
    type, public :: conditional_basis_pipeline_t
        private
        integer :: n_inputs = 0
        integer :: n_branches = 0
        logical :: fitted = .false.
        type(conditional_branch_t), allocatable :: branches(:)
        type(basis_input_schema_t) :: input_schema
    contains
        procedure, public :: initialize => conditional_pipeline_initialize
        procedure, public :: append => conditional_pipeline_append
        procedure, public :: fit => conditional_pipeline_fit
        procedure, public :: transform => conditional_pipeline_transform
        procedure, public :: evaluate => conditional_pipeline_transform
        procedure, public :: transform_device => &
            conditional_pipeline_transform_device
        procedure, public :: jvp => conditional_pipeline_jvp
        procedure, public :: jvp_device => conditional_pipeline_jvp_device
        procedure, public :: vjp => conditional_pipeline_vjp
        procedure, public :: vjp_device => conditional_pipeline_vjp_device
        procedure, public :: hvp => conditional_pipeline_hvp
        procedure, public :: hvp_device => conditional_pipeline_hvp_device
        procedure, public :: input_count => conditional_pipeline_input_count
        procedure, public :: branch_count => conditional_pipeline_branch_count
        procedure, public :: feature_count => conditional_pipeline_feature_count
        procedure, public :: parameter_count => &
            conditional_pipeline_parameter_count
        procedure, public :: parameters => conditional_pipeline_parameters
        procedure, public :: set_parameters => conditional_pipeline_set_parameters
        procedure, public :: branch_name => conditional_pipeline_branch_name
        procedure, public :: feature_name => conditional_pipeline_feature_name
        procedure, public :: parameter_name => conditional_pipeline_parameter_name
        procedure, public :: branch_feature_offset => &
            conditional_pipeline_branch_feature_offset
        procedure, public :: branch_parameter_offset => &
            conditional_pipeline_branch_parameter_offset
        procedure, public :: branch_route_column => &
            conditional_pipeline_branch_route_column
        procedure, public :: branch_lower_bound => &
            conditional_pipeline_branch_lower_bound
        procedure, public :: branch_upper_bound => &
            conditional_pipeline_branch_upper_bound
        procedure, public :: set_input_schema => &
            conditional_pipeline_set_input_schema
        procedure, public :: input_schema_name => &
            conditional_pipeline_input_schema_name
        procedure, public :: validate_input_schema => &
            conditional_pipeline_validate_input_schema
        procedure, public :: static_lowering_eligible => &
            conditional_pipeline_static_lowering_eligible
        procedure, public :: capabilities => conditional_pipeline_capabilities
        procedure, public :: device_supported => &
            conditional_pipeline_device_supported
        procedure, public :: valid => conditional_pipeline_valid
        procedure, public :: is_fitted => conditional_pipeline_is_fitted
    end type conditional_basis_pipeline_t

    public :: make_conditional_basis_pipeline

contains

    function make_conditional_basis_pipeline(n_inputs, status) result(pipeline)
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status
        type(conditional_basis_pipeline_t) :: pipeline

        call pipeline%initialize(n_inputs, status)
    end function make_conditional_basis_pipeline

    subroutine conditional_pipeline_initialize(self, n_inputs, status)
        class(conditional_basis_pipeline_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status

        self%n_inputs = 0
        self%n_branches = 0
        self%fitted = .false.
        allocate(self%branches(0))
        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline: n_inputs must be positive")
            return
        end if
        call self%input_schema%initialize(n_inputs, status)
        if (status%code /= FORTNUM_OK) return
        self%n_inputs = n_inputs
        call status_set(status, FORTNUM_OK, "")
    end subroutine conditional_pipeline_initialize

    !> Append one interval-routed branch without modifying self on failure.
    subroutine conditional_pipeline_append(self, stage, columns, route_column, &
            lower_bound, upper_bound, status, name)
        class(conditional_basis_pipeline_t), intent(inout) :: self
        type(basis_map_t), intent(in) :: stage
        integer, intent(in) :: columns(:)
        integer, intent(in) :: route_column
        real(dp), intent(in) :: lower_bound, upper_bound
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: name
        type(conditional_branch_t), allocatable :: candidate(:)
        type(column_basis_pipeline_t) :: branch_pipeline
        character(len=NAME_LENGTH) :: branch_name
        integer :: old_count, i

        if (.not. conditional_pipeline_configured(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline append: pipeline is not initialized")
            return
        end if
        if (.not. stage%valid() .or. route_column < 1 .or. &
                route_column > self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline append: stage or route column is invalid")
            return
        end if
        if (.not. ieee_is_finite(lower_bound) .or. &
                .not. ieee_is_finite(upper_bound) .or. &
                lower_bound >= upper_bound) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline append: interval must be finite and ordered")
            return
        end if
        branch_name = default_branch_name(self%n_branches + 1)
        if (present(name)) then
            if (len_trim(name) < 1 .or. len_trim(name) > NAME_LENGTH) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "conditional basis pipeline append: branch name is invalid")
                return
            end if
            branch_name = trim(name)
        end if
        do i = 1, self%n_branches
            if (trim(self%branches(i)%name) == trim(branch_name)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "conditional basis pipeline append: branch names must be unique")
                return
            end if
        end do

        branch_pipeline = make_column_basis_pipeline(self%n_inputs, status)
        if (status%code /= FORTNUM_OK) return
        call branch_pipeline%append(stage, columns, status, name=trim(branch_name))
        if (status%code /= FORTNUM_OK) return

        old_count = self%n_branches
        allocate(candidate(old_count + 1))
        if (old_count > 0) candidate(1:old_count) = self%branches
        candidate(old_count + 1)%pipeline = branch_pipeline
        candidate(old_count + 1)%route_column = route_column
        candidate(old_count + 1)%lower_bound = lower_bound
        candidate(old_count + 1)%upper_bound = upper_bound
        candidate(old_count + 1)%name = branch_name
        call move_alloc(candidate, self%branches)
        self%n_branches = old_count + 1
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine conditional_pipeline_append

    subroutine conditional_pipeline_fit(self, x, status)
        class(conditional_basis_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. conditional_pipeline_valid(self) .or. size(x, 1) < 1 .or. &
                size(x, 2) /= self%n_inputs .or. .not. all(ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline fit: model or input shape is invalid")
            return
        end if
        do i = 1, self%n_branches
            call self%branches(i)%pipeline%fit(x, status)
            if (status%code /= FORTNUM_OK) return
        end do
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine conditional_pipeline_fit

    subroutine conditional_pipeline_transform(self, x, phi, status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), local_phi(:, :)
        logical, allocatable :: active(:)
        integer :: i, feature_offset, n_features

        if (.not. conditional_pipeline_valid(self) .or. .not. self%fitted .or. &
                size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
                size(phi, 1) /= size(x, 1) .or. &
                size(phi, 2) /= self%feature_count() .or. .not. all(ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline transform: model or array shape is invalid")
            return
        end if
        allocate(candidate(size(phi, 1), size(phi, 2)))
        candidate = 0.0_dp
        feature_offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%pipeline%feature_count()
            allocate(local_phi(size(x, 1), n_features), active(size(x, 1)))
            call self%branches(i)%pipeline%transform(x, local_phi, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_phi, active, candidate)
                return
            end if
            call route_mask(self%branches(i), x, active, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_phi, active, candidate)
                return
            end if
            call mask_rows(local_phi, active)
            candidate(:, feature_offset + 1:feature_offset + n_features) = local_phi
            feature_offset = feature_offset + n_features
            deallocate(local_phi, active)
        end do
        phi = candidate
        deallocate(candidate)
        call status_set(status, FORTNUM_OK, "")
    end subroutine conditional_pipeline_transform

    subroutine conditional_pipeline_jvp(self, x, theta_dot, x_dot, phi, phi_dot, &
            status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), candidate_dot(:, :)
        real(dp), allocatable :: local_phi(:, :), local_phi_dot(:, :)
        real(dp), allocatable :: local_theta_dot(:)
        logical, allocatable :: active(:)
        integer :: i, feature_offset, parameter_offset, n_features, n_parameters

        if (.not. conditional_products_ready(self, x, status, "JVP")) return
        if (any(shape(x_dot) /= shape(x)) .or. size(theta_dot) /= &
                self%parameter_count() .or. size(phi, 1) /= size(x, 1) .or. &
                size(phi, 2) /= self%feature_count() .or. &
                any(shape(phi_dot) /= shape(phi))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline JVP: array shape is invalid")
            return
        end if
        if (.not. all(ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline JVP: tangent contains nonfinite values")
            return
        end if
        allocate(candidate(size(phi, 1), size(phi, 2)), &
            candidate_dot(size(phi_dot, 1), size(phi_dot, 2)))
        candidate = 0.0_dp
        candidate_dot = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%pipeline%feature_count()
            n_parameters = self%branches(i)%pipeline%parameter_count()
            allocate(local_phi(size(x, 1), n_features), &
                local_phi_dot(size(x, 1), n_features), active(size(x, 1)), &
                local_theta_dot(n_parameters))
            if (n_parameters > 0) local_theta_dot = theta_dot(parameter_offset + 1: &
                parameter_offset + n_parameters)
            call self%branches(i)%pipeline%jvp(x, local_theta_dot, x_dot, &
                local_phi, local_phi_dot, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_phi, local_phi_dot, active, local_theta_dot, &
                    candidate, candidate_dot)
                return
            end if
            call route_mask(self%branches(i), x, active, status, derivatives=.true.)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_phi, local_phi_dot, active, local_theta_dot, &
                    candidate, candidate_dot)
                return
            end if
            call mask_rows(local_phi, active)
            call mask_rows(local_phi_dot, active)
            candidate(:, feature_offset + 1:feature_offset + n_features) = local_phi
            candidate_dot(:, feature_offset + 1:feature_offset + n_features) = &
                local_phi_dot
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
            deallocate(local_phi, local_phi_dot, active, local_theta_dot)
        end do
        phi = candidate
        phi_dot = candidate_dot
        deallocate(candidate, candidate_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine conditional_pipeline_jvp

    subroutine conditional_pipeline_vjp(self, x, u, theta_bar, x_bar, status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate_theta(:), candidate_x(:, :)
        real(dp), allocatable :: local_u(:, :), local_theta_bar(:), local_x_bar(:, :)
        logical, allocatable :: active(:)
        integer :: i, feature_offset, parameter_offset, n_features, n_parameters

        if (.not. conditional_products_ready(self, x, status, "VJP")) return
        if (size(u, 1) /= size(x, 1) .or. size(u, 2) /= self%feature_count() .or. &
                size(theta_bar) /= self%parameter_count() .or. &
                any(shape(x_bar) /= shape(x)) .or. .not. all(ieee_is_finite(u))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline VJP: array shape is invalid")
            return
        end if
        allocate(candidate_theta(size(theta_bar)), candidate_x(size(x, 1), size(x, 2)))
        candidate_theta = 0.0_dp
        candidate_x = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%pipeline%feature_count()
            n_parameters = self%branches(i)%pipeline%parameter_count()
            allocate(local_u(size(x, 1), n_features), &
                local_theta_bar(n_parameters), local_x_bar(size(x, 1), size(x, 2)), &
                active(size(x, 1)))
            local_u = u(:, feature_offset + 1:feature_offset + n_features)
            call route_mask(self%branches(i), x, active, status, derivatives=.true.)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_u, local_theta_bar, local_x_bar, active, &
                    candidate_theta, candidate_x)
                return
            end if
            call mask_rows(local_u, active)
            call self%branches(i)%pipeline%vjp(x, local_u, local_theta_bar, &
                local_x_bar, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_u, local_theta_bar, local_x_bar, active, &
                    candidate_theta, candidate_x)
                return
            end if
            if (n_parameters > 0) candidate_theta(parameter_offset + 1: &
                parameter_offset + n_parameters) = local_theta_bar
            candidate_x = candidate_x + local_x_bar
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
            deallocate(local_u, local_theta_bar, local_x_bar, active)
        end do
        theta_bar = candidate_theta
        x_bar = candidate_x
        deallocate(candidate_theta, candidate_x)
        call status_set(status, FORTNUM_OK, "")
    end subroutine conditional_pipeline_vjp

    subroutine conditional_pipeline_hvp(self, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate_theta(:), candidate_x(:, :)
        real(dp), allocatable :: local_u(:, :), local_theta_dot(:), local_theta_hvp(:)
        real(dp), allocatable :: local_x_hvp(:, :)
        logical, allocatable :: active(:)
        integer :: i, feature_offset, parameter_offset, n_features, n_parameters

        if (.not. conditional_products_ready(self, x, status, "HVP")) return
        if (size(u, 1) /= size(x, 1) .or. size(u, 2) /= self%feature_count() .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                size(theta_dot) /= self%parameter_count() .or. &
                size(theta_hvp) /= self%parameter_count() .or. &
                any(shape(x_hvp) /= shape(x)) .or. .not. all(ieee_is_finite(u)) .or. &
                .not. all(ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline HVP: array shape is invalid")
            return
        end if
        allocate(candidate_theta(size(theta_hvp)), candidate_x(size(x, 1), size(x, 2)))
        candidate_theta = 0.0_dp
        candidate_x = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%pipeline%feature_count()
            n_parameters = self%branches(i)%pipeline%parameter_count()
            allocate(local_u(size(x, 1), n_features), local_theta_dot(n_parameters), &
                local_theta_hvp(n_parameters), local_x_hvp(size(x, 1), size(x, 2)), &
                active(size(x, 1)))
            local_u = u(:, feature_offset + 1:feature_offset + n_features)
            if (n_parameters > 0) local_theta_dot = theta_dot(parameter_offset + 1: &
                parameter_offset + n_parameters)
            call route_mask(self%branches(i), x, active, status, derivatives=.true.)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_u, local_theta_dot, local_theta_hvp, local_x_hvp, &
                    active, candidate_theta, candidate_x)
                return
            end if
            call mask_rows(local_u, active)
            call self%branches(i)%pipeline%hvp(x, local_u, local_theta_dot, x_dot, &
                local_theta_hvp, local_x_hvp, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(local_u, local_theta_dot, local_theta_hvp, local_x_hvp, &
                    active, candidate_theta, candidate_x)
                return
            end if
            if (n_parameters > 0) candidate_theta(parameter_offset + 1: &
                parameter_offset + n_parameters) = local_theta_hvp
            candidate_x = candidate_x + local_x_hvp
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
            deallocate(local_u, local_theta_dot, local_theta_hvp, local_x_hvp, active)
        end do
        theta_hvp = candidate_theta
        x_hvp = candidate_x
        deallocate(candidate_theta, candidate_x)
        call status_set(status, FORTNUM_OK, "")
    end subroutine conditional_pipeline_hvp

    subroutine conditional_pipeline_transform_device(self, device, x, phi, status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. conditional_device_ready(self, device, status, "transform")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%transform(x, phi, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "conditional basis pipeline transform: no resident CUDA route kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline transform: device kind is invalid")
        end select
    end subroutine conditional_pipeline_transform_device

    subroutine conditional_pipeline_jvp_device(self, device, x, theta_dot, x_dot, &
            phi, phi_dot, status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. conditional_device_ready(self, device, status, "JVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "conditional basis pipeline JVP: no resident CUDA route kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline JVP: device kind is invalid")
        end select
    end subroutine conditional_pipeline_jvp_device

    subroutine conditional_pipeline_vjp_device(self, device, x, u, theta_bar, &
            x_bar, status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. conditional_device_ready(self, device, status, "VJP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%vjp(x, u, theta_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "conditional basis pipeline VJP: no resident CUDA route kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline VJP: device kind is invalid")
        end select
    end subroutine conditional_pipeline_vjp_device

    subroutine conditional_pipeline_hvp_device(self, device, x, u, theta_dot, &
            x_dot, theta_hvp, x_hvp, status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. conditional_device_ready(self, device, status, "HVP")) return
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "conditional basis pipeline HVP: no resident CUDA route kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline HVP: device kind is invalid")
        end select
    end subroutine conditional_pipeline_hvp_device

    integer function conditional_pipeline_input_count(self) result(count)
        class(conditional_basis_pipeline_t), intent(in) :: self
        count = self%n_inputs
    end function conditional_pipeline_input_count

    integer function conditional_pipeline_branch_count(self) result(count)
        class(conditional_basis_pipeline_t), intent(in) :: self
        count = self%n_branches
    end function conditional_pipeline_branch_count

    integer function conditional_pipeline_feature_count(self) result(count)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%branches)) return
        do i = 1, self%n_branches
            count = count + self%branches(i)%pipeline%feature_count()
        end do
    end function conditional_pipeline_feature_count

    integer function conditional_pipeline_parameter_count(self) result(count)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%branches)) return
        do i = 1, self%n_branches
            count = count + self%branches(i)%pipeline%parameter_count()
        end do
    end function conditional_pipeline_parameter_count

    function conditional_pipeline_parameters(self) result(theta)
        class(conditional_basis_pipeline_t), intent(in) :: self
        real(dp), allocatable :: theta(:), local_theta(:)
        integer :: i, offset, n_parameters

        allocate(theta(self%parameter_count()))
        theta = 0.0_dp
        offset = 0
        if (.not. allocated(self%branches)) return
        do i = 1, self%n_branches
            n_parameters = self%branches(i)%pipeline%parameter_count()
            if (n_parameters > 0) then
                local_theta = self%branches(i)%pipeline%parameters()
                theta(offset + 1:offset + n_parameters) = local_theta
                deallocate(local_theta)
            end if
            offset = offset + n_parameters
        end do
    end function conditional_pipeline_parameters

    subroutine conditional_pipeline_set_parameters(self, theta, status)
        class(conditional_basis_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        type(conditional_branch_t), allocatable :: candidate(:)
        integer :: i, offset, n_parameters

        if (.not. conditional_pipeline_valid(self) .or. size(theta) /= &
                self%parameter_count() .or. .not. all(ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline set_parameters: model or shape is invalid")
            return
        end if
        allocate(candidate(self%n_branches))
        candidate = self%branches
        offset = 0
        do i = 1, self%n_branches
            n_parameters = candidate(i)%pipeline%parameter_count()
            if (n_parameters > 0) then
                call candidate(i)%pipeline%set_parameters(theta(offset + 1: &
                    offset + n_parameters), status)
                if (status%code /= FORTNUM_OK) then
                    deallocate(candidate)
                    return
                end if
            end if
            offset = offset + n_parameters
        end do
        call move_alloc(candidate, self%branches)
        call status_set(status, FORTNUM_OK, "")
    end subroutine conditional_pipeline_set_parameters

    function conditional_pipeline_branch_name(self, branch) result(name)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch
        character(:), allocatable :: name

        name = ""
        if (.not. allocated(self%branches)) return
        if (branch < 1 .or. branch > self%n_branches) return
        name = trim(self%branches(branch)%name)
    end function conditional_pipeline_branch_name

    integer function conditional_pipeline_branch_feature_offset(self, branch) &
            result(offset)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch
        integer :: i

        offset = 0
        if (branch < 1 .or. branch > self%n_branches) return
        offset = 1
        do i = 1, branch - 1
            offset = offset + self%branches(i)%pipeline%feature_count()
        end do
    end function conditional_pipeline_branch_feature_offset

    integer function conditional_pipeline_branch_parameter_offset(self, branch) &
            result(offset)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch
        integer :: i

        offset = 0
        if (branch < 1 .or. branch > self%n_branches) return
        offset = 1
        do i = 1, branch - 1
            offset = offset + self%branches(i)%pipeline%parameter_count()
        end do
    end function conditional_pipeline_branch_parameter_offset

    function conditional_pipeline_feature_name(self, feature) result(name)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name
        integer :: i, offset, n_features

        name = ""
        if (feature < 1 .or. feature > self%feature_count()) return
        offset = 0
        do i = 1, self%n_branches
            n_features = self%branches(i)%pipeline%feature_count()
            if (feature <= offset + n_features) then
                name = trim(self%branches(i)%name)//"."// &
                    self%branches(i)%pipeline%feature_name(feature - offset)
                return
            end if
            offset = offset + n_features
        end do
    end function conditional_pipeline_feature_name

    function conditional_pipeline_parameter_name(self, parameter) result(name)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: parameter
        character(:), allocatable :: name
        integer :: i, offset, n_parameters

        name = ""
        if (parameter < 1 .or. parameter > self%parameter_count()) return
        offset = 0
        do i = 1, self%n_branches
            n_parameters = self%branches(i)%pipeline%parameter_count()
            if (parameter <= offset + n_parameters) then
                name = trim(self%branches(i)%name)//"."// &
                    self%branches(i)%pipeline%parameter_name(parameter - offset)
                return
            end if
            offset = offset + n_parameters
        end do
    end function conditional_pipeline_parameter_name

    integer function conditional_pipeline_branch_route_column(self, branch) &
            result(column)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch

        column = 0
        if (.not. allocated(self%branches)) return
        if (branch < 1 .or. branch > self%n_branches) return
        column = self%branches(branch)%route_column
    end function conditional_pipeline_branch_route_column

    real(dp) function conditional_pipeline_branch_lower_bound(self, branch) &
            result(bound)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch

        bound = 0.0_dp
        if (.not. allocated(self%branches)) return
        if (branch < 1 .or. branch > self%n_branches) return
        bound = self%branches(branch)%lower_bound
    end function conditional_pipeline_branch_lower_bound

    real(dp) function conditional_pipeline_branch_upper_bound(self, branch) &
            result(bound)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: branch

        bound = 0.0_dp
        if (.not. allocated(self%branches)) return
        if (branch < 1 .or. branch > self%n_branches) return
        bound = self%branches(branch)%upper_bound
    end function conditional_pipeline_branch_upper_bound

    subroutine conditional_pipeline_set_input_schema(self, names, status)
        class(conditional_basis_pipeline_t), intent(inout) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%set_names(names, status)
    end subroutine conditional_pipeline_set_input_schema

    function conditional_pipeline_input_schema_name(self, feature) result(name)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name

        name = self%input_schema%name(feature)
    end function conditional_pipeline_input_schema_name

    subroutine conditional_pipeline_validate_input_schema(self, names, status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        character(*), intent(in) :: names(:)
        type(fortnum_status_t), intent(out) :: status

        call self%input_schema%validate_names(names, status)
    end subroutine conditional_pipeline_validate_input_schema

    logical function conditional_pipeline_static_lowering_eligible(self) &
            result(eligible)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer :: i

        eligible = conditional_pipeline_valid(self)
        if (.not. eligible) return
        do i = 1, self%n_branches
            if (.not. self%branches(i)%pipeline%static_lowering_eligible()) then
                eligible = .false.
                return
            end if
        end do
    end function conditional_pipeline_static_lowering_eligible

    subroutine conditional_pipeline_capabilities(self, report, status)
        class(conditional_basis_pipeline_t), intent(in) :: self
        type(estimator_capability_t), intent(out) :: report
        type(fortnum_status_t), intent(out) :: status

        if (.not. conditional_pipeline_valid(self)) then
            call report%initialize("conditional_basis_pipeline", 1, 1, status)
            if (status%code == FORTNUM_OK) call status_set(status, &
                FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline capabilities: pipeline is invalid")
            return
        end if
        report = make_transformer_capabilities("conditional_basis_pipeline", &
            self%n_inputs, self%feature_count(), status, self%is_fitted())
        if (status%code /= FORTNUM_OK) return
        report%supports_input_jvp = .true.
        report%supports_input_vjp = .true.
        report%supports_input_hvp = .true.
        report%supports_parameter_jvp = .true.
        report%supports_parameter_vjp = .true.
        report%supports_parameter_hvp = .true.
    end subroutine conditional_pipeline_capabilities

    logical function conditional_pipeline_device_supported(self, device_kind) &
            result(supported)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%is_fitted() .and. conditional_pipeline_valid(self)
        case default
            supported = .false.
        end select
    end function conditional_pipeline_device_supported

    logical function conditional_pipeline_valid(self) result(valid)
        class(conditional_basis_pipeline_t), intent(in) :: self
        integer :: i

        valid = conditional_pipeline_configured(self) .and. self%n_branches > 0
        if (.not. valid) return
        if (.not. self%input_schema%valid()) then
            valid = .false.
            return
        end if
        do i = 1, self%n_branches
            if (.not. self%branches(i)%pipeline%valid() .or. &
                    self%branches(i)%pipeline%input_count() /= self%n_inputs .or. &
                    self%branches(i)%route_column < 1 .or. &
                    self%branches(i)%route_column > self%n_inputs .or. &
                    .not. ieee_is_finite(self%branches(i)%lower_bound) .or. &
                    .not. ieee_is_finite(self%branches(i)%upper_bound) .or. &
                    self%branches(i)%lower_bound >= self%branches(i)%upper_bound .or. &
                    len_trim(self%branches(i)%name) < 1) then
                valid = .false.
                return
            end if
            if (i > 1) then
                if (any(trimmed_name_equals(self%branches(1:i-1), &
                        trim(self%branches(i)%name)))) then
                    valid = .false.
                    return
                end if
            end if
        end do
    end function conditional_pipeline_valid

    logical function conditional_pipeline_is_fitted(self) result(fitted)
        class(conditional_basis_pipeline_t), intent(in) :: self

        fitted = self%fitted .and. conditional_pipeline_valid(self)
    end function conditional_pipeline_is_fitted

    logical function conditional_pipeline_configured(self) result(configured)
        class(conditional_basis_pipeline_t), intent(in) :: self

        configured = self%n_inputs > 0 .and. self%n_branches >= 0 .and. &
            allocated(self%branches)
        if (.not. configured) return
        configured = size(self%branches) >= self%n_branches
    end function conditional_pipeline_configured

    logical function conditional_device_ready(self, device, status, operation) &
            result(ready)
        class(conditional_basis_pipeline_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        ready = .false.
        if (.not. conditional_pipeline_valid(self) .or. .not. self%is_fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline "//trim(operation)//": model is invalid or unfitted")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline "//trim(operation)//": device is not selected")
            return
        end if
        ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end function conditional_device_ready

    logical function conditional_products_ready(self, x, status, operation) &
            result(ready)
        class(conditional_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        ready = .false.
        if (.not. conditional_pipeline_valid(self) .or. .not. self%is_fitted() .or. &
                size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
                .not. all(ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline "//trim(operation)//": model or input is invalid")
            return
        end if
        ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end function conditional_products_ready

    subroutine route_mask(branch, x, active, status, derivatives)
        type(conditional_branch_t), intent(in) :: branch
        real(dp), intent(in) :: x(:, :)
        logical, intent(out) :: active(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: derivatives
        logical :: need_derivatives
        integer :: i, column

        need_derivatives = .false.
        if (present(derivatives)) need_derivatives = derivatives
        if (size(active) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "conditional basis pipeline route: mask shape is invalid")
            return
        end if
        column = branch%route_column
        do i = 1, size(x, 1)
            if (.not. ieee_is_finite(x(i, column))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "conditional basis pipeline route: route value is nonfinite")
                return
            end if
            if (need_derivatives .and. (x(i, column) == branch%lower_bound .or. &
                    x(i, column) == branch%upper_bound)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "conditional basis pipeline route: derivative at interval endpoint is undefined")
                return
            end if
            active(i) = x(i, column) >= branch%lower_bound .and. &
                x(i, column) < branch%upper_bound
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine route_mask

    subroutine mask_rows(values, active)
        real(dp), intent(inout) :: values(:, :)
        logical, intent(in) :: active(:)
        integer :: i

        do i = 1, size(active)
            if (.not. active(i)) values(i, :) = 0.0_dp
        end do
    end subroutine mask_rows

    function trimmed_name_equals(branches, candidate) result(matches)
        type(conditional_branch_t), intent(in) :: branches(:)
        character(*), intent(in) :: candidate
        logical :: matches(size(branches))
        integer :: i

        do i = 1, size(branches)
            matches(i) = trim(branches(i)%name) == trim(candidate)
        end do
    end function trimmed_name_equals

    function default_branch_name(index) result(name)
        integer, intent(in) :: index
        character(len=NAME_LENGTH) :: name
        character(len=32) :: buffer

        name = ""
        write (buffer, '("branch_",i0)') index
        name = trim(buffer)
    end function default_branch_name

end module fortml_conditional_pipeline
