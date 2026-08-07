!> Column-selecting feature unions built from differentiable basis maps.
module fortml_column_pipeline
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_basis, only: basis_map_t
    use fortml_estimator_capabilities, only: estimator_capability_t, &
        make_transformer_capabilities
    implicit none
    private

    integer, parameter :: PIPELINE_NAME_LENGTH = 128

    !> One basis map and the original input columns it consumes.
    type :: column_pipeline_stage_t
        type(basis_map_t) :: map
        integer, allocatable :: columns(:)
    end type column_pipeline_stage_t

    !> A horizontal basis union with explicit per-stage column selection.
    !>
    !> Unlike `basis_pipeline_t`, each stage receives only the selected columns
    !> from the original matrix.  Stage parameters remain packed in append
    !> order, while input VJPs are scattered back into the full input matrix.
    !> Column indices are one-based, strictly unique within each stage, and
    !> are checked at append time.  Cross-stage reuse is allowed.
    type, public :: column_basis_pipeline_t
        private
        integer :: n_inputs = 0
        integer :: n_stages = 0
        logical :: fitted = .false.
        type(column_pipeline_stage_t), allocatable :: stages(:)
        character(len=PIPELINE_NAME_LENGTH), allocatable :: stage_names(:)
    contains
        procedure, public :: initialize => column_pipeline_initialize
        procedure, public :: append => column_pipeline_append
        procedure, public :: fit => column_pipeline_fit
        procedure, public :: transform => column_pipeline_transform
        procedure, public :: evaluate => column_pipeline_transform
        procedure, public :: jvp => column_pipeline_jvp
        procedure, public :: vjp => column_pipeline_vjp
        procedure, public :: hvp => column_pipeline_hvp
        procedure, public :: input_count => column_pipeline_input_count
        procedure, public :: stage_count => column_pipeline_stage_count
        procedure, public :: feature_count => column_pipeline_feature_count
        procedure, public :: parameter_count => column_pipeline_parameter_count
        procedure, public :: parameters => column_pipeline_parameters
        procedure, public :: set_parameters => column_pipeline_set_parameters
        procedure, public :: stage_name => column_pipeline_stage_name
        procedure, public :: feature_name => column_pipeline_feature_name
        procedure, public :: parameter_name => column_pipeline_parameter_name
        procedure, public :: stage_feature_offset => &
            column_pipeline_stage_feature_offset
        procedure, public :: stage_parameter_offset => &
            column_pipeline_stage_parameter_offset
        procedure, public :: stage_columns => column_pipeline_stage_columns
        procedure, public :: static_lowering_eligible => &
            column_pipeline_static_lowering_eligible
        procedure, public :: capabilities => column_pipeline_capabilities
        procedure, public :: valid => column_pipeline_valid
        procedure, public :: is_fitted => column_pipeline_is_fitted
    end type column_basis_pipeline_t

    public :: make_column_basis_pipeline

contains

    !> Construct an empty column-selecting pipeline.
    function make_column_basis_pipeline(n_inputs, status) result(pipeline)
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status
        type(column_basis_pipeline_t) :: pipeline

        call pipeline%initialize(n_inputs, status)
    end function make_column_basis_pipeline

    subroutine column_pipeline_initialize(self, n_inputs, status)
        class(column_basis_pipeline_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        type(fortnum_status_t), intent(out) :: status

        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline: n_inputs must be positive")
            return
        end if
        self%n_inputs = n_inputs
        self%n_stages = 0
        self%fitted = .false.
        allocate(self%stages(0))
        allocate(self%stage_names(0))
        call status_set(status, FORTNUM_OK, "")
    end subroutine column_pipeline_initialize

    subroutine column_pipeline_append(self, stage, columns, status, name)
        class(column_basis_pipeline_t), intent(inout) :: self
        type(basis_map_t), intent(in) :: stage
        integer, intent(in) :: columns(:)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: name
        type(column_pipeline_stage_t), allocatable :: new_stages(:)
        character(len=PIPELINE_NAME_LENGTH), allocatable :: new_names(:)
        character(:), allocatable :: stage_name
        integer :: old_count, i, j

        if (self%n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline append: pipeline is not initialized")
            return
        end if
        if (.not. allocated(self%stages)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline append: pipeline is not initialized")
            return
        end if
        if (.not. stage%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline append: stage is not initialized")
            return
        end if
        if (size(columns) < 1 .or. stage%input_count() /= size(columns)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline append: stage and columns do not match")
            return
        end if
        do i = 1, size(columns)
            if (columns(i) < 1 .or. columns(i) > self%n_inputs) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "column basis pipeline append: column index is out of range")
                return
            end if
            do j = 1, i - 1
                if (columns(j) == columns(i)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "column basis pipeline append: duplicate column index")
                    return
                end if
            end do
        end do
        stage_name = default_stage_name(self%n_stages + 1)
        if (present(name)) then
            if (len_trim(name) < 1 .or. len_trim(name) > PIPELINE_NAME_LENGTH) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "column basis pipeline append: stage name is invalid")
                return
            end if
            stage_name = trim(name)
        end if
        if (allocated(self%stage_names)) then
            do i = 1, self%n_stages
                if (trim(self%stage_names(i)) == stage_name) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "column basis pipeline append: stage names must be unique")
                    return
                end if
            end do
        end if

        old_count = self%n_stages
        allocate(new_stages(old_count + 1))
        allocate(new_names(old_count + 1))
        if (old_count > 0) new_stages(1:old_count) = self%stages
        if (old_count > 0) new_names(1:old_count) = self%stage_names
        new_stages(old_count + 1)%map = stage
        new_stages(old_count + 1)%columns = columns
        new_names(old_count + 1) = stage_name
        call move_alloc(new_stages, self%stages)
        call move_alloc(new_names, self%stage_names)
        self%n_stages = old_count + 1
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine column_pipeline_append

    subroutine column_pipeline_fit(self, x, status)
        class(column_basis_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. column_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline fit: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline fit: input shape is invalid")
            return
        end if
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine column_pipeline_fit

    subroutine column_pipeline_transform(self, x, phi, status)
        class(column_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: x_local(:, :)
        integer :: i, feature_offset, n_features

        if (.not. column_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline transform: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(phi, 1) /= size(x, 1) .or. &
            size(phi, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline transform: array shape is invalid")
            return
        end if

        feature_offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%map%feature_count()
            allocate(x_local(size(x, 1), size(self%stages(i)%columns)))
            call gather_columns(x, self%stages(i)%columns, x_local)
            call self%stages(i)%map%evaluate(x_local, &
                phi(:, feature_offset + 1:feature_offset + n_features), status)
            deallocate(x_local)
            if (status%code /= FORTNUM_OK) return
            feature_offset = feature_offset + n_features
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine column_pipeline_transform

    subroutine column_pipeline_jvp(self, x, theta_dot, x_dot, phi, phi_dot, &
            status)
        class(column_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: x_local(:, :), x_dot_local(:, :), theta_local(:)
        integer :: i, feature_offset, parameter_offset, n_features, n_parameters

        if (.not. column_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline jvp: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            size(phi, 1) /= size(x, 1) .or. &
            size(phi, 2) /= self%feature_count() .or. &
            any(shape(phi_dot) /= shape(phi)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline jvp: array shape is invalid")
            return
        end if

        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%map%feature_count()
            n_parameters = self%stages(i)%map%parameter_count()
            allocate(x_local(size(x, 1), size(self%stages(i)%columns)))
            allocate(x_dot_local(size(x, 1), size(self%stages(i)%columns)))
            allocate(theta_local(n_parameters))
            call gather_columns(x, self%stages(i)%columns, x_local)
            call gather_columns(x_dot, self%stages(i)%columns, x_dot_local)
            if (n_parameters > 0) theta_local = theta_dot(parameter_offset + 1: &
                parameter_offset + n_parameters)
            call self%stages(i)%map%jvp(x_local, theta_local, x_dot_local, &
                phi(:, feature_offset + 1:feature_offset + n_features), &
                phi_dot(:, feature_offset + 1:feature_offset + n_features), &
                status)
            deallocate(x_local, x_dot_local, theta_local)
            if (status%code /= FORTNUM_OK) return
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine column_pipeline_jvp

    subroutine column_pipeline_vjp(self, x, u, theta_bar, x_bar, status)
        class(column_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: x_local(:, :), local_x_bar(:, :)
        real(dp), allocatable :: local_theta_bar(:)
        integer :: i, feature_offset, parameter_offset, n_features, n_parameters
        integer :: j

        if (.not. column_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline vjp: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            size(u, 1) /= size(x, 1) .or. size(u, 2) /= self%feature_count() .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline vjp: array shape is invalid")
            return
        end if

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%map%feature_count()
            n_parameters = self%stages(i)%map%parameter_count()
            allocate(x_local(size(x, 1), size(self%stages(i)%columns)))
            allocate(local_x_bar(size(x, 1), size(self%stages(i)%columns)))
            allocate(local_theta_bar(n_parameters))
            call gather_columns(x, self%stages(i)%columns, x_local)
            call self%stages(i)%map%vjp(x_local, &
                u(:, feature_offset + 1:feature_offset + n_features), &
                local_theta_bar, local_x_bar, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(x_local, local_x_bar, local_theta_bar)
                return
            end if
            if (n_parameters > 0) theta_bar(parameter_offset + 1: &
                parameter_offset + n_parameters) = local_theta_bar
            do j = 1, size(self%stages(i)%columns)
                x_bar(:, self%stages(i)%columns(j)) = x_bar(:, &
                    self%stages(i)%columns(j)) + local_x_bar(:, j)
            end do
            deallocate(x_local, local_x_bar, local_theta_bar)
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine column_pipeline_vjp

    subroutine column_pipeline_hvp(self, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        !! HVP for a column-selecting union.  Each stage computes its local
        !! second-order product on gathered columns and the result is scattered
        !! back, accumulating contributions when stages reuse an input column.
        class(column_basis_pipeline_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: x_local(:, :), x_dot_local(:, :)
        real(dp), allocatable :: local_theta_dot(:), local_theta_hvp(:)
        real(dp), allocatable :: local_x_hvp(:, :)
        integer :: i, j, feature_offset, parameter_offset, n_features, n_parameters

        if (.not. column_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline hvp: model is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
                size(u, 1) /= size(x, 1) .or. size(u, 2) /= self%feature_count() .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                size(theta_dot) /= self%parameter_count() .or. &
                size(theta_hvp) /= self%parameter_count() .or. &
                any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline hvp: array shape is invalid")
            return
        end if
        theta_hvp = 0.0_dp
        x_hvp = 0.0_dp
        feature_offset = 0
        parameter_offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%map%feature_count()
            n_parameters = self%stages(i)%map%parameter_count()
            allocate(x_local(size(x, 1), size(self%stages(i)%columns)))
            allocate(x_dot_local(size(x, 1), size(self%stages(i)%columns)))
            allocate(local_theta_dot(n_parameters), local_theta_hvp(n_parameters))
            allocate(local_x_hvp(size(x, 1), size(self%stages(i)%columns)))
            call gather_columns(x, self%stages(i)%columns, x_local)
            call gather_columns(x_dot, self%stages(i)%columns, x_dot_local)
            if (n_parameters > 0) local_theta_dot = theta_dot(parameter_offset + 1: &
                parameter_offset + n_parameters)
            call self%stages(i)%map%hvp(x_local, &
                u(:, feature_offset + 1:feature_offset + n_features), &
                local_theta_dot, x_dot_local, local_theta_hvp, local_x_hvp, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(x_local, x_dot_local, local_theta_dot, local_theta_hvp, &
                    local_x_hvp)
                return
            end if
            if (n_parameters > 0) theta_hvp(parameter_offset + 1: &
                parameter_offset + n_parameters) = local_theta_hvp
            do j = 1, size(self%stages(i)%columns)
                x_hvp(:, self%stages(i)%columns(j)) = x_hvp(:, &
                    self%stages(i)%columns(j)) + local_x_hvp(:, j)
            end do
            deallocate(x_local, x_dot_local, local_theta_dot, local_theta_hvp, &
                local_x_hvp)
            feature_offset = feature_offset + n_features
            parameter_offset = parameter_offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine column_pipeline_hvp

    integer function column_pipeline_input_count(self) result(count)
        class(column_basis_pipeline_t), intent(in) :: self
        count = self%n_inputs
    end function column_pipeline_input_count

    integer function column_pipeline_stage_count(self) result(count)
        class(column_basis_pipeline_t), intent(in) :: self
        count = self%n_stages
    end function column_pipeline_stage_count

    integer function column_pipeline_feature_count(self) result(count)
        class(column_basis_pipeline_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%stages)) return
        do i = 1, self%n_stages
            count = count + self%stages(i)%map%feature_count()
        end do
    end function column_pipeline_feature_count

    integer function column_pipeline_parameter_count(self) result(count)
        class(column_basis_pipeline_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%stages)) return
        do i = 1, self%n_stages
            count = count + self%stages(i)%map%parameter_count()
        end do
    end function column_pipeline_parameter_count

    function column_pipeline_stage_name(self, stage) result(name)
        class(column_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        character(:), allocatable :: name

        name = ""
        if (.not. allocated(self%stage_names)) return
        if (stage < 1 .or. stage > self%n_stages) return
        name = trim(self%stage_names(stage))
    end function column_pipeline_stage_name

    integer function column_pipeline_stage_feature_offset(self, stage) &
            result(offset)
        class(column_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        integer :: i

        offset = 0
        if (stage < 1 .or. stage > self%n_stages) return
        if (.not. allocated(self%stages)) return
        offset = 1
        do i = 1, stage - 1
            offset = offset + self%stages(i)%map%feature_count()
        end do
    end function column_pipeline_stage_feature_offset

    integer function column_pipeline_stage_parameter_offset(self, stage) &
            result(offset)
        class(column_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        integer :: i

        offset = 0
        if (stage < 1 .or. stage > self%n_stages) return
        if (.not. allocated(self%stages)) return
        offset = 1
        do i = 1, stage - 1
            offset = offset + self%stages(i)%map%parameter_count()
        end do
    end function column_pipeline_stage_parameter_offset

    function column_pipeline_feature_name(self, feature) result(name)
        class(column_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: feature
        character(:), allocatable :: name
        integer :: i, offset, n_features

        name = ""
        if (feature < 1 .or. feature > self%feature_count()) return
        offset = 0
        do i = 1, self%n_stages
            n_features = self%stages(i)%map%feature_count()
            if (feature <= offset + n_features) then
                name = qualified_stage_name(trim(self%stage_names(i)), &
                    "feature", feature - offset)
                return
            end if
            offset = offset + n_features
        end do
    end function column_pipeline_feature_name

    function column_pipeline_parameter_name(self, parameter) result(name)
        class(column_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: parameter
        character(:), allocatable :: name
        integer :: i, offset, n_parameters

        name = ""
        if (parameter < 1 .or. parameter > self%parameter_count()) return
        offset = 0
        do i = 1, self%n_stages
            n_parameters = self%stages(i)%map%parameter_count()
            if (parameter <= offset + n_parameters) then
                name = qualified_stage_name(trim(self%stage_names(i)), &
                    "parameter", parameter - offset)
                return
            end if
            offset = offset + n_parameters
        end do
    end function column_pipeline_parameter_name

    function column_pipeline_stage_columns(self, stage) result(columns)
        class(column_basis_pipeline_t), intent(in) :: self
        integer, intent(in) :: stage
        integer, allocatable :: columns(:)

        allocate(columns(0))
        if (.not. allocated(self%stages)) return
        if (stage < 1 .or. stage > self%n_stages) return
        columns = self%stages(stage)%columns
    end function column_pipeline_stage_columns

    function column_pipeline_parameters(self) result(theta)
        class(column_basis_pipeline_t), intent(in) :: self
        real(dp), allocatable :: theta(:), local_theta(:)
        integer :: i, offset, n_parameters

        allocate(theta(self%parameter_count()))
        theta = 0.0_dp
        offset = 0
        if (.not. allocated(self%stages)) return
        do i = 1, self%n_stages
            n_parameters = self%stages(i)%map%parameter_count()
            if (n_parameters > 0) then
                local_theta = self%stages(i)%map%parameters()
                theta(offset + 1:offset + n_parameters) = local_theta
            end if
            offset = offset + n_parameters
        end do
    end function column_pipeline_parameters

    subroutine column_pipeline_set_parameters(self, theta, status)
        class(column_basis_pipeline_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, offset, n_parameters

        if (.not. column_pipeline_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline set_parameters: model is invalid")
            return
        end if
        if (size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "column basis pipeline set_parameters: shape is invalid")
            return
        end if
        offset = 0
        do i = 1, self%n_stages
            n_parameters = self%stages(i)%map%parameter_count()
            if (n_parameters > 0) then
                call self%stages(i)%map%set_parameters(theta(offset + 1: &
                    offset + n_parameters), status)
                if (status%code /= FORTNUM_OK) return
            end if
            offset = offset + n_parameters
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine column_pipeline_set_parameters

    logical function column_pipeline_static_lowering_eligible(self) result(eligible)
        class(column_basis_pipeline_t), intent(in) :: self
        integer :: i

        eligible = column_pipeline_valid(self)
        if (.not. eligible) return
        do i = 1, self%n_stages
            if (.not. self%stages(i)%map%static_lowering_eligible()) then
                eligible = .false.
                return
            end if
        end do
    end function column_pipeline_static_lowering_eligible

    logical function column_pipeline_valid(self) result(valid)
        class(column_basis_pipeline_t), intent(in) :: self
        integer :: i, j, k

        valid = self%n_inputs > 0 .and. self%n_stages > 0 .and. &
            allocated(self%stages) .and. allocated(self%stage_names)
        if (.not. valid) return
        if (size(self%stages) < self%n_stages .or. &
            size(self%stage_names) < self%n_stages) then
            valid = .false.
            return
        end if
        do i = 1, self%n_stages
            if (.not. self%stages(i)%map%valid()) then
                valid = .false.
                return
            end if
            if (.not. allocated(self%stages(i)%columns)) then
                valid = .false.
                return
            end if
            if (size(self%stages(i)%columns) < 1 .or. &
                self%stages(i)%map%input_count() /= &
                size(self%stages(i)%columns)) then
                valid = .false.
                return
            end if
            do j = 1, size(self%stages(i)%columns)
                if (self%stages(i)%columns(j) < 1 .or. &
                    self%stages(i)%columns(j) > self%n_inputs) then
                    valid = .false.
                    return
                end if
                do k = 1, j - 1
                    if (self%stages(i)%columns(j) == &
                        self%stages(i)%columns(k)) then
                        valid = .false.
                        return
                    end if
                end do
            end do
        end do
    end function column_pipeline_valid

    logical function column_pipeline_is_fitted(self) result(fitted)
        class(column_basis_pipeline_t), intent(in) :: self

        fitted = self%fitted
        if (.not. fitted) return
        fitted = column_pipeline_valid(self)
    end function column_pipeline_is_fitted

    !> Return the generic transformer contract for this column union.
    subroutine column_pipeline_capabilities(self, report, status)
        class(column_basis_pipeline_t), intent(in) :: self
        type(estimator_capability_t), intent(out) :: report
        type(fortnum_status_t), intent(out) :: status

        if (.not. column_pipeline_valid(self)) then
            call report%initialize("column_basis_pipeline", 1, 1, status)
            if (status%code == FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "column basis pipeline capabilities: pipeline is invalid")
            end if
            return
        end if
        report = make_transformer_capabilities("column_basis_pipeline", &
            self%n_inputs, self%feature_count(), status, self%is_fitted())
        if (status%code /= FORTNUM_OK) return
        report%supports_input_jvp = .true.
        report%supports_input_vjp = .true.
        report%supports_input_hvp = .true.
        report%supports_parameter_jvp = .true.
        report%supports_parameter_vjp = .true.
        report%supports_parameter_hvp = .true.
    end subroutine column_pipeline_capabilities

    subroutine gather_columns(x, columns, selected)
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: columns(:)
        real(dp), intent(out) :: selected(:, :)
        integer :: j

        do j = 1, size(columns)
            selected(:, j) = x(:, columns(j))
        end do
    end subroutine gather_columns

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

end module fortml_column_pipeline
