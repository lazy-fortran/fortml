!> Composable feature maps built from the differentiable basis API.
module fortml_pipeline
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_basis, only: basis_map_t
    implicit none
    private

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
    contains
        procedure, public :: initialize => pipeline_initialize
        procedure, public :: append => pipeline_append
        procedure, public :: fit => pipeline_fit
        procedure, public :: transform => pipeline_transform
        procedure, public :: evaluate => pipeline_transform
        procedure, public :: jvp => pipeline_jvp
        procedure, public :: vjp => pipeline_vjp
        procedure, public :: input_count => pipeline_input_count
        procedure, public :: stage_count => pipeline_stage_count
        procedure, public :: feature_count => pipeline_feature_count
        procedure, public :: parameter_count => pipeline_parameter_count
        procedure, public :: parameters => pipeline_parameters
        procedure, public :: set_parameters => pipeline_set_parameters
        procedure, public :: static_lowering_eligible => &
            pipeline_static_lowering_eligible
        procedure, public :: valid => pipeline_valid
        procedure, public :: is_fitted => pipeline_is_fitted
    end type basis_pipeline_t

    public :: make_basis_pipeline

contains

    !> Construct an empty pipeline.  Append one or more basis maps before fit.
    function make_basis_pipeline(n_inputs, status) result(pipeline)
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status
        type(basis_pipeline_t) :: pipeline

        call pipeline%initialize(n_inputs, status)
    end function make_basis_pipeline

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
        call status_set(status, FORTNUM_OK, "")
    end subroutine pipeline_initialize

    subroutine pipeline_append(self, stage, status)
        class(basis_pipeline_t), intent(inout) :: self
        type(basis_map_t), intent(in) :: stage
        type(fortnum_status_t), intent(out) :: status
        type(basis_map_t), allocatable :: new_stages(:)
        integer :: old_count

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

        old_count = self%n_stages
        allocate(new_stages(old_count + 1))
        if (old_count > 0) new_stages(1:old_count) = self%stages
        new_stages(old_count + 1) = stage
        call move_alloc(new_stages, self%stages)
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
            allocated(self%stages)
        if (.not. valid) return
        if (size(self%stages) < self%n_stages) then
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

end module fortml_pipeline
