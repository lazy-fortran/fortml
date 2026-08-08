module fortml_mlp_chain
    !! Composable sequential MLP modules with one optimizer-facing parameter tree.
    !!
    !! A chain owns an ordered set of dense `mlp_t` children.  The children
    !! remain ordinary MLPs (and therefore retain their established value,
    !! JVP, VJP, and HVP products), while this module supplies the missing
    !! composition rule: a deterministic stage-name/offset tree and exact
    !! chain-rule products over all child parameters.  The objective adapter
    !! below uses the same products for FortOpt L-BFGS-B; it never estimates a
    !! derivative by perturbing a composed model.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_mlp, only: mlp_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type :: name_holder_t
        character(:), allocatable :: value
    end type name_holder_t

    type :: matrix_holder_t
        real(dp), allocatable :: value(:, :)
    end type matrix_holder_t

    type, public :: mlp_chain_t
        !! Ordered composition `stage_n(...stage_2(stage_1(x)))`.
        private
        type(mlp_t), allocatable :: stage(:)
        type(name_holder_t), allocatable :: stage_name(:)
        integer :: n_stages = 0
        integer :: input_features = 0
    contains
        procedure, public :: initialize => mlp_chain_initialize
        procedure, public :: append => mlp_chain_append
        procedure, public :: clear => mlp_chain_clear
        procedure, public :: stage_count => mlp_chain_stage_count
        procedure, public :: input_count => mlp_chain_input_count
        procedure, public :: output_count => mlp_chain_output_count
        procedure, public :: parameter_count => mlp_chain_parameter_count
        procedure, public :: parameter_range => mlp_chain_parameter_range
        procedure, public :: parameters => mlp_chain_parameters
        procedure, public :: set_parameters => mlp_chain_set_parameters
        procedure, public :: predict => mlp_chain_predict
        procedure, public :: jvp => mlp_chain_jvp
        procedure, public :: vjp => mlp_chain_vjp
        procedure, public :: hvp => mlp_chain_hvp
        procedure, public :: device_supported => mlp_chain_device_supported
    end type mlp_chain_t

    type, public :: mlp_chain_objective_t
        !! Mean squared-error objective over a composed chain.
        private
        type(mlp_chain_t), pointer :: model => null()
        real(dp), allocatable :: features(:, :), targets(:, :)
        real(dp) :: l2 = 0.0_dp
        logical :: optimize_l2 = .false.
    contains
        procedure, public :: initialize => mlp_chain_objective_initialize
        procedure, public :: parameter_count => mlp_chain_objective_parameter_count
        procedure, public :: parameters => mlp_chain_objective_parameters
        procedure, public :: value_gradient => mlp_chain_objective_value_gradient
        procedure, public :: jvp => mlp_chain_objective_jvp
        procedure, public :: vjp => mlp_chain_objective_vjp
        procedure, public :: hvp => mlp_chain_objective_hvp
        procedure, public :: fortopt => mlp_chain_objective_fortopt
    end type mlp_chain_objective_t

    type, public :: mlp_chain_lbfgsb_options_t
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: l2_lower_bound = 0.0_dp
        real(dp) :: l2_upper_bound = 20.0_dp
        logical :: optimize_l2 = .false.
        integer :: device_kind = FORTML_DEVICE_CPU
    end type mlp_chain_lbfgsb_options_t

    type, public :: mlp_chain_lbfgsb_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: l2 = 0.0_dp
    end type mlp_chain_lbfgsb_result_t

    public :: mlp_chain_optimize_lbfgsb

contains

    subroutine mlp_chain_initialize(self, input_features, status)
        class(mlp_chain_t), intent(out) :: self
        integer, intent(in) :: input_features
        type(fortnum_status_t), intent(out) :: status

        if (input_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain: input feature count must be positive")
            return
        end if
        self%input_features = input_features
        self%n_stages = 0
        if (allocated(self%stage)) deallocate(self%stage)
        if (allocated(self%stage_name)) deallocate(self%stage_name)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_initialize

    subroutine mlp_chain_append(self, model, status, name)
        class(mlp_chain_t), intent(inout) :: self
        type(mlp_t), intent(in) :: model
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in), optional :: name
        type(mlp_t), allocatable :: extended(:)
        type(name_holder_t), allocatable :: extended_names(:)
        integer :: i, input_count
        character(:), allocatable :: stage_name

        if (self%input_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain: initialize the chain before appending a stage")
            return
        end if
        if (.not. allocated(model%layer_sizes) .or. &
                size(model%layer_sizes) < 2 .or. model%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain: appended stage is not initialized")
            return
        end if
        input_count = self%input_features
        if (self%n_stages > 0) then
            input_count = self%stage(self%n_stages)%layer_sizes(&
                size(self%stage(self%n_stages)%layer_sizes))
        end if
        if (model%layer_sizes(1) /= input_count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain: stage input does not match previous output")
            return
        end if
        stage_name = default_stage_name(self%n_stages + 1)
        if (present(name)) then
            if (len_trim(name) == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP chain: stage name must not be empty")
                return
            end if
            stage_name = trim(name)
        end if
        do i = 1, self%n_stages
            if (self%stage_name(i)%value == stage_name) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP chain: stage names must be unique")
                return
            end if
        end do

        allocate(extended(self%n_stages + 1))
        allocate(extended_names(self%n_stages + 1))
        do i = 1, self%n_stages
            extended(i) = self%stage(i)
            extended_names(i)%value = self%stage_name(i)%value
        end do
        extended(self%n_stages + 1) = model
        extended_names(self%n_stages + 1)%value = stage_name
        call move_alloc(extended, self%stage)
        call move_alloc(extended_names, self%stage_name)
        self%n_stages = self%n_stages + 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_append

    subroutine mlp_chain_clear(self)
        class(mlp_chain_t), intent(inout) :: self

        if (allocated(self%stage)) deallocate(self%stage)
        if (allocated(self%stage_name)) deallocate(self%stage_name)
        self%n_stages = 0
        self%input_features = 0
    end subroutine mlp_chain_clear

    integer function mlp_chain_stage_count(self) result(count)
        class(mlp_chain_t), intent(in) :: self

        count = self%n_stages
    end function mlp_chain_stage_count

    integer function mlp_chain_input_count(self) result(count)
        class(mlp_chain_t), intent(in) :: self

        count = self%input_features
    end function mlp_chain_input_count

    integer function mlp_chain_output_count(self) result(count)
        class(mlp_chain_t), intent(in) :: self

        count = 0
        if (self%n_stages > 0) count = self%stage(self%n_stages)%layer_sizes(&
            size(self%stage(self%n_stages)%layer_sizes))
    end function mlp_chain_output_count

    integer function mlp_chain_parameter_count(self) result(count)
        class(mlp_chain_t), intent(in) :: self
        integer :: i

        count = 0
        do i = 1, self%n_stages
            count = count + self%stage(i)%parameter_count()
        end do
    end function mlp_chain_parameter_count

    subroutine mlp_chain_parameter_range(self, name, first, last, found)
        class(mlp_chain_t), intent(in) :: self
        character(*), intent(in) :: name
        integer, intent(out) :: first, last
        logical, intent(out) :: found
        integer :: i

        first = 1
        found = .false.
        do i = 1, self%n_stages
            last = first + self%stage(i)%parameter_count() - 1
            if (self%stage_name(i)%value == trim(name)) then
                found = .true.
                return
            end if
            first = last + 1
        end do
        first = 0
        last = -1
    end subroutine mlp_chain_parameter_range

    function mlp_chain_parameters(self) result(theta)
        class(mlp_chain_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        integer :: i, first, last

        allocate(theta(self%parameter_count()))
        theta = 0.0_dp
        first = 1
        do i = 1, self%n_stages
            last = first + self%stage(i)%parameter_count() - 1
            theta(first:last) = self%stage(i)%parameters()
            first = last + 1
        end do
    end function mlp_chain_parameters

    subroutine mlp_chain_set_parameters(self, theta, status)
        class(mlp_chain_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last

        if (self%n_stages < 1 .or. size(theta) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain: packed parameter shape or values are invalid")
            return
        end if
        first = 1
        do i = 1, self%n_stages
            last = first + self%stage(i)%parameter_count() - 1
            call self%stage(i)%set_parameters(theta(first:last), status)
            if (.not. status_ok(status)) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_set_parameters

    subroutine mlp_chain_predict(self, x, y, status)
        class(mlp_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(matrix_holder_t) :: current, next
        integer :: i

        if (.not. valid_chain(self, x, y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain predict: chain or batch shape is invalid")
            return
        end if
        allocate(current%value, source=x)
        do i = 1, self%n_stages
            allocate(next%value(size(x, 1), self%stage(i)%layer_sizes( &
                size(self%stage(i)%layer_sizes))))
            call self%stage(i)%predict(current%value, next%value, status)
            if (.not. status_ok(status)) return
            call move_alloc(next%value, current%value)
        end do
        y = current%value
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_predict

    subroutine mlp_chain_jvp(self, x, dtheta, dx, y, dy, status)
        class(mlp_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), dtheta(:), dx(:, :)
        real(dp), intent(out) :: y(:, :), dy(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(matrix_holder_t) :: current, current_tangent, next, next_tangent
        integer :: i, first, last

        if (.not. valid_chain(self, x, y) .or. any(shape(dx) /= shape(x)) .or. &
                any(shape(dy) /= shape(y)) .or. size(dtheta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain JVP: chain, tangent, or batch shape is invalid")
            return
        end if
        allocate(current%value, source=x)
        allocate(current_tangent%value, source=dx)
        first = 1
        do i = 1, self%n_stages
            last = first + self%stage(i)%parameter_count() - 1
            allocate(next%value(size(x, 1), self%stage(i)%layer_sizes( &
                size(self%stage(i)%layer_sizes))))
            allocate(next_tangent%value, mold=next%value)
            call self%stage(i)%jvp(current%value, dtheta(first:last), &
                current_tangent%value, next%value, next_tangent%value, status)
            if (.not. status_ok(status)) return
            call move_alloc(next%value, current%value)
            call move_alloc(next_tangent%value, current_tangent%value)
            first = last + 1
        end do
        y = current%value
        dy = current_tangent%value
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_jvp

    subroutine mlp_chain_vjp(self, x, u, parameter_bar, x_bar, status)
        class(mlp_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: parameter_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(matrix_holder_t), allocatable :: values(:)
        real(dp), allocatable :: cotangent(:, :), previous(:, :), stage_bar(:)
        integer :: i, first, last

        if (.not. valid_chain(self, x, u) .or. size(parameter_bar) /= self%parameter_count() .or. &
                any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain VJP: chain, cotangent, or output shape is invalid")
            return
        end if
        call chain_forward(self, x, values, status)
        if (.not. status_ok(status)) return
        allocate(cotangent, source=u)
        parameter_bar = 0.0_dp
        last = self%parameter_count()
        do i = self%n_stages, 1, -1
            first = last - self%stage(i)%parameter_count() + 1
            allocate(stage_bar(self%stage(i)%parameter_count()))
            allocate(previous(size(x, 1), self%stage(i)%layer_sizes(1)))
            call self%stage(i)%vjp(values(i - 1)%value, cotangent, stage_bar, previous, status)
            if (.not. status_ok(status)) return
            parameter_bar(first:last) = stage_bar
            deallocate(stage_bar, cotangent)
            call move_alloc(previous, cotangent)
            last = first - 1
        end do
        x_bar = cotangent
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_vjp

    subroutine mlp_chain_hvp(self, x, u, dtheta, dx, parameter_hvp, x_hvp, status)
        !! Differentiate the chain VJP for fixed output cotangent `u`.
        class(mlp_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), dtheta(:), dx(:, :)
        real(dp), intent(out) :: parameter_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(matrix_holder_t), allocatable :: values(:), tangents(:)
        real(dp), allocatable :: cotangent(:, :), cotangent_tangent(:, :)
        real(dp), allocatable :: previous(:, :), previous_tangent(:, :)
        real(dp), allocatable :: cotangent_tangent_bar(:, :)
        real(dp), allocatable :: local_hvp(:), tangent_bar(:), normal_bar(:)
        real(dp), allocatable :: dtheta_stage(:)
        integer :: i, first, last

        if (.not. valid_chain(self, x, u) .or. any(shape(dx) /= shape(x)) .or. &
                size(dtheta) /= self%parameter_count() .or. &
                size(parameter_hvp) /= self%parameter_count() .or. &
                any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain HVP: chain, direction, or output shape is invalid")
            return
        end if
        call chain_forward(self, x, values, status)
        if (.not. status_ok(status)) return
        allocate(tangents(0:self%n_stages))
        allocate(tangents(0)%value, source=dx)
        do i = 1, self%n_stages
            allocate(tangents(i)%value, mold=values(i)%value)
            allocate(dtheta_stage(self%stage(i)%parameter_count()))
            call stage_slice(dtheta, i, self, dtheta_stage)
            call self%stage(i)%jvp(values(i - 1)%value, dtheta_stage, &
                tangents(i - 1)%value, values(i)%value, tangents(i)%value, status)
            deallocate(dtheta_stage)
            if (.not. status_ok(status)) return
        end do

        allocate(cotangent, source=u)
        allocate(cotangent_tangent, mold=u)
        cotangent_tangent = 0.0_dp
        parameter_hvp = 0.0_dp
        last = self%parameter_count()
        do i = self%n_stages, 1, -1
            first = last - self%stage(i)%parameter_count() + 1
            allocate(dtheta_stage(self%stage(i)%parameter_count()))
            call stage_slice(dtheta, i, self, dtheta_stage)
            allocate(local_hvp(size(dtheta_stage)))
            allocate(previous_tangent(size(x, 1), self%stage(i)%layer_sizes(1)))
            call self%stage(i)%hvp(values(i - 1)%value, cotangent, dtheta_stage, &
                tangents(i - 1)%value, local_hvp, previous_tangent, status)
            if (.not. status_ok(status)) return
            allocate(normal_bar(size(dtheta_stage)))
            allocate(previous(size(x, 1), self%stage(i)%layer_sizes(1)))
            call self%stage(i)%vjp(values(i - 1)%value, cotangent, normal_bar, &
                previous, status)
            if (.not. status_ok(status)) return
            allocate(tangent_bar(size(dtheta_stage)))
            allocate(cotangent_tangent_bar(size(x, 1), self%stage(i)%layer_sizes(1)))
            call self%stage(i)%vjp(values(i - 1)%value, cotangent_tangent, tangent_bar, &
                cotangent_tangent_bar, status)
            if (.not. status_ok(status)) return
            parameter_hvp(first:last) = local_hvp + tangent_bar
            ! The reverse cotangent tangent has two contributions: the
            ! stage-local HVP and the VJP of the downstream cotangent
            ! tangent.  Add them before replacing the old reverse state.
            previous_tangent = previous_tangent + cotangent_tangent_bar
            deallocate(cotangent, cotangent_tangent, local_hvp, tangent_bar, &
                normal_bar, cotangent_tangent_bar, dtheta_stage)
            call move_alloc(previous, cotangent)
            call move_alloc(previous_tangent, cotangent_tangent)
            last = first - 1
        end do
        x_hvp = cotangent_tangent
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_hvp

    logical function mlp_chain_device_supported(self, device_kind) result(supported)
        class(mlp_chain_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%n_stages > 0 .and. device_kind == FORTML_DEVICE_CPU
    end function mlp_chain_device_supported

    subroutine mlp_chain_objective_initialize(self, model, x, target, l2, status, &
            optimize_l2, device_kind)
        class(mlp_chain_objective_t), intent(out) :: self
        type(mlp_chain_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :), l2
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: optimize_l2
        integer, intent(in), optional :: device_kind
        integer :: requested_device

        requested_device = FORTML_DEVICE_CPU
        if (present(device_kind)) requested_device = device_kind
        self%optimize_l2 = .false.
        if (present(optimize_l2)) self%optimize_l2 = optimize_l2
        if (.not. model%device_supported(requested_device)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP chain objective: requested device has no resident lowering")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= model%input_count() .or. &
                size(target, 1) /= size(x, 1) .or. size(target, 2) /= model%output_count() .or. &
                .not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain objective: model, data, or L2 coefficient is invalid")
            return
        end if
        self%model => model
        allocate(self%features, source=x)
        allocate(self%targets, source=target)
        self%l2 = l2
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_objective_initialize

    integer function mlp_chain_objective_parameter_count(self) result(count)
        class(mlp_chain_objective_t), intent(in) :: self

        count = 0
        if (.not. associated(self%model)) return
        count = self%model%parameter_count()
        if (self%optimize_l2) count = count + 1
    end function mlp_chain_objective_parameter_count

    function mlp_chain_objective_parameters(self) result(parameters)
        class(mlp_chain_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        real(dp), allocatable :: stage_parameters(:)
        integer :: n_model, i, first, last

        allocate(parameters(self%parameter_count()))
        parameters = 0.0_dp
        if (.not. associated(self%model)) return
        n_model = self%model%parameter_count()
        first = 1
        do i = 1, self%model%n_stages
            last = first + self%model%stage(i)%parameter_count() - 1
            stage_parameters = self%model%stage(i)%parameters()
            parameters(first:last) = stage_parameters
            first = last + 1
        end do
        if (self%optimize_l2) parameters(n_model + 1) = self%l2
    end function mlp_chain_objective_parameters

    subroutine mlp_chain_objective_value_gradient(self, parameters, value, gradient, status)
        class(mlp_chain_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: prediction(:, :), residual(:, :), x_bar(:, :), theta(:)
        real(dp) :: l2, l2_gradient
        integer :: n_model, n_samples

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain objective: adapter is not initialized")
            return
        end if
        if (size(parameters) /= self%parameter_count() .or. size(gradient) /= size(parameters) .or. &
                any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain objective: parameter or gradient shape/value is invalid")
            return
        end if
        n_model = self%model%parameter_count()
        l2 = self%l2
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            if (l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP chain objective: optimized L2 coefficient is negative")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        n_samples = size(self%features, 1)
        allocate(prediction(size(self%features, 1), size(self%targets, 2)))
        allocate(residual, mold=prediction)
        allocate(x_bar, mold=self%features)
        call self%model%predict(self%features, prediction, status)
        if (.not. status_ok(status)) return
        residual = prediction - self%targets
        call self%model%vjp(self%features, residual/real(n_samples, dp), &
            gradient(:n_model), x_bar, status)
        if (.not. status_ok(status)) return
        theta = self%model%parameters()
        value = 0.5_dp*sum(residual*residual)/real(n_samples, dp) + &
            0.5_dp*l2*sum(theta*theta)
        l2_gradient = 0.5_dp*sum(theta*theta)
        gradient(:n_model) = gradient(:n_model) + l2*theta
        if (self%optimize_l2) gradient(n_model + 1) = l2_gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_objective_value_gradient

    subroutine mlp_chain_objective_jvp(self, parameters, direction, value, tangent, status)
        class(mlp_chain_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        tangent = 0.0_dp
        if (size(direction) /= size(parameters) .or. any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain objective JVP: direction shape/value is invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        tangent = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_objective_jvp

    subroutine mlp_chain_objective_vjp(self, parameters, output_bar, gradient, status)
        class(mlp_chain_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain objective VJP: output cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_objective_vjp

    subroutine mlp_chain_objective_hvp(self, parameters, direction, product, status)
        class(mlp_chain_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: prediction(:, :), residual(:, :), output_tangent(:, :)
        real(dp), allocatable :: x_bar(:, :), x_hvp(:, :), jtj(:), curvature(:), theta(:)
        real(dp) :: l2, l2_direction, l2_hvp
        integer :: n_model, n_samples

        product = 0.0_dp
        if (.not. associated(self%model) .or. size(parameters) /= self%parameter_count() .or. &
                size(direction) /= size(parameters) .or. size(product) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain objective HVP: adapter or direction shape is invalid")
            return
        end if
        n_model = self%model%parameter_count()
        l2 = self%l2
        l2_direction = 0.0_dp
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            l2_direction = direction(n_model + 1)
        end if
        if (l2 < 0.0_dp .or. .not. ieee_is_finite(l2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain objective HVP: L2 coefficient is invalid")
            return
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        n_samples = size(self%features, 1)
        allocate(prediction(size(self%features, 1), size(self%targets, 2)))
        allocate(residual, mold=prediction)
        allocate(output_tangent, mold=prediction)
        allocate(x_bar, mold=self%features)
        allocate(x_hvp, mold=self%features)
        allocate(jtj(n_model), curvature(n_model), theta(n_model))
        call self%model%predict(self%features, prediction, status)
        if (.not. status_ok(status)) return
        residual = prediction - self%targets
        ! Differentiate the chain prediction with a zero input tangent.
        call self%model%jvp(self%features, direction(:n_model), 0.0_dp*self%features, &
            prediction, output_tangent, status)
        if (.not. status_ok(status)) return
        call self%model%vjp(self%features, output_tangent/real(n_samples, dp), &
            jtj, x_bar, status)
        if (.not. status_ok(status)) return
        call self%model%hvp(self%features, residual/real(n_samples, dp), &
            direction(:n_model), 0.0_dp*self%features, curvature, x_hvp, status)
        if (.not. status_ok(status)) return
        theta = self%model%parameters()
        product(:n_model) = jtj + curvature + l2*direction(:n_model) + &
            l2_direction*theta
        if (self%optimize_l2) product(n_model + 1) = dot_product(theta, direction(:n_model))
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_objective_hvp

    subroutine mlp_chain_objective_fortopt(self, objective, status)
        class(mlp_chain_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            mlp_chain_objective_context_callback, status)
    end subroutine mlp_chain_objective_fortopt

    subroutine mlp_chain_objective_context_callback(context, parameters, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_chain_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain objective: FortOpt context has wrong type")
        end select
    end subroutine mlp_chain_objective_context_callback

    subroutine mlp_chain_optimize_lbfgsb(model, x, target, options, result, status)
        type(mlp_chain_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(mlp_chain_lbfgsb_options_t), intent(in) :: options
        type(mlp_chain_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(mlp_chain_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_model, n_parameters
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_chain_lbfgsb_result_t) :: mlp_chain_lbfgsb_result_t_default

        result = mlp_chain_lbfgsb_result_t_default
        if (options%device_kind /= FORTML_DEVICE_CPU .or. options%memory < 1 .or. &
                options%max_iterations < 1 .or. options%max_line_search < 1 .or. &
                options%lower_bound > options%upper_bound .or. options%l2 < 0.0_dp .or. &
                options%l2_lower_bound > options%l2_upper_bound .or. &
                .not. model%device_supported(options%device_kind)) then
            if (options%device_kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "MLP chain L-BFGS-B: no resident device lowering")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP chain L-BFGS-B: model or options are invalid")
            end if
            return
        end if
        n_model = model%parameter_count()
        if (n_model < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain L-BFGS-B: chain has no parameters")
            return
        end if
        call adapter%initialize(model, x, target, options%l2, status, &
            optimize_l2=options%optimize_l2)
        if (.not. status_ok(status)) return
        n_parameters = adapter%parameter_count()
        parameters = adapter%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower(:n_model) = options%lower_bound
        upper(:n_model) = options%upper_bound
        if (options%optimize_l2) then
            lower(n_model + 1) = options%l2_lower_bound
            upper(n_model + 1) = options%l2_upper_bound
        end if
        call adapter%fortopt(objective, status)
        if (.not. status_ok(status)) return
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call optimizer%minimize(objective, parameters, lower, upper, optimizer_options, &
            optimizer_result, status)
        if (.not. status_ok(status)) return
        call model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) return
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = sqrt(sum(gradient*gradient))
        result%l2 = options%l2
        if (options%optimize_l2) result%l2 = parameters(n_model + 1)
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP chain L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_chain_optimize_lbfgsb

    subroutine chain_forward(self, x, values, status)
        class(mlp_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(matrix_holder_t), allocatable, intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (self%n_stages < 1 .or. size(x, 2) /= self%input_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain forward: chain or input shape is invalid")
            return
        end if
        allocate(values(0:self%n_stages))
        allocate(values(0)%value, source=x)
        do i = 1, self%n_stages
            allocate(values(i)%value(size(x, 1), self%stage(i)%layer_sizes( &
                size(self%stage(i)%layer_sizes))))
            call self%stage(i)%predict(values(i - 1)%value, values(i)%value, status)
            if (.not. status_ok(status)) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine chain_forward

    subroutine stage_slice(theta, index, self, slice)
        real(dp), intent(in) :: theta(:)
        integer, intent(in) :: index
        class(mlp_chain_t), intent(in) :: self
        real(dp), intent(out) :: slice(:)
        integer :: i, first, last

        first = 1
        do i = 1, index
            last = first + self%stage(i)%parameter_count() - 1
            if (i == index) then
                slice = theta(first:last)
                return
            end if
            first = last + 1
        end do
    end subroutine stage_slice

    logical function valid_chain(self, x, y) result(valid)
        class(mlp_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)

        valid = self%n_stages > 0 .and. size(x, 1) > 0 .and. &
            size(x, 2) == self%input_features .and. &
            all(shape(y) == [size(x, 1), self%output_count()])
    end function valid_chain

    function default_stage_name(index) result(name)
        integer, intent(in) :: index
        character(:), allocatable :: name

        character(32) :: buffer
        write(buffer, '(a,i0)') "stage_", index
        name = trim(buffer)
    end function default_stage_name

end module fortml_mlp_chain
