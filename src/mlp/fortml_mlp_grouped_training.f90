module fortml_mlp_grouped_training
    !! Group-wise MLP regularization and exact hyperparameter products.
    !!
    !! `mlp_grouped_training_objective_t` extends the ordinary MSE training
    !! objective with one independently differentiable log-L2 coefficient per
    !! parameter group.  Its packed variable is
    !!
    !!     [ network parameters, log(lambda_1), ..., log(lambda_g) ].
    !!
    !! Group coefficients use an exponential parameterization, so every
    !! L-BFGS-B iterate has a strictly positive regularization coefficient.
    !! Value, JVP, VJP, and HVP products are analytic.  The HVP includes the
    !! mixed network/log-coefficient blocks and can therefore be used by an
    !! outer hyperparameter optimizer without finite differences.
    !!
    !! This module intentionally has no hidden host fallback.  Its current
    !! dense MLP graph is CPU-only; requesting CUDA returns
    !! `FORTNUM_NOT_IMPLEMENTED` until a resident CUDA MLP and derivative graph
    !! are available.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient, mlp_loss_hvp
    use fortopt_objective, only: objective_t
    implicit none
    private

    real(dp), parameter :: LOG_L2_MIN = -50.0_dp
    real(dp), parameter :: LOG_L2_MAX = 50.0_dp

    type, public :: mlp_parameter_group_t
        !! A contiguous, non-overlapping network parameter slice.
        character(len=64) :: name = ""
        integer :: first = 0
        integer :: last = -1
        real(dp) :: log_l2 = -4.0_dp
    contains
        procedure, public :: initialize => mlp_parameter_group_initialize
        procedure, public :: size => mlp_parameter_group_size
        procedure, public :: initialized => mlp_parameter_group_initialized
    end type mlp_parameter_group_t

    type, public :: mlp_grouped_training_objective_t
        !! MSE plus independently tunable group penalties.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: features(:, :), targets(:, :)
        type(mlp_parameter_group_t), allocatable :: groups(:)
        integer :: device_kind = FORTML_DEVICE_CPU
    contains
        procedure, public :: initialize => mlp_grouped_objective_initialize
        procedure, public :: initialized => mlp_grouped_objective_initialized
        procedure, public :: device_supported => mlp_grouped_device_supported
        procedure, public :: parameter_count => mlp_grouped_parameter_count
        procedure, public :: group_count => mlp_grouped_group_count
        procedure, public :: parameters => mlp_grouped_parameters
        procedure, public :: group_name => mlp_grouped_group_name
        procedure, public :: group_range => mlp_grouped_group_range
        procedure, public :: value_gradient => mlp_grouped_value_gradient
        procedure, public :: jvp => mlp_grouped_jvp
        procedure, public :: vjp => mlp_grouped_vjp
        procedure, public :: hvp => mlp_grouped_hvp
        procedure, public :: fortopt => mlp_grouped_fortopt
    end type mlp_grouped_training_objective_t

contains

    subroutine mlp_parameter_group_initialize(self, name, first, last, log_l2, status)
        class(mlp_parameter_group_t), intent(out) :: self
        character(*), intent(in) :: name
        integer, intent(in) :: first, last
        real(dp), intent(in) :: log_l2
        type(fortnum_status_t), intent(out) :: status

        self%name = ""
        self%first = 0
        self%last = -1
        self%log_l2 = -4.0_dp
        if (len_trim(name) == 0 .or. len_trim(name) > len(self%name) .or. &
                first < 1 .or. last < first .or. &
                .not. ieee_is_finite(log_l2) .or. log_l2 < LOG_L2_MIN .or. &
                log_l2 > LOG_L2_MAX) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP parameter group: invalid name, range, or log L2")
            return
        end if
        self%name = trim(name)
        self%first = first
        self%last = last
        self%log_l2 = log_l2
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_parameter_group_initialize

    integer function mlp_parameter_group_size(self) result(n)
        class(mlp_parameter_group_t), intent(in) :: self

        n = 0
        if (self%initialized()) n = self%last - self%first + 1
    end function mlp_parameter_group_size

    logical function mlp_parameter_group_initialized(self) result(yes)
        class(mlp_parameter_group_t), intent(in) :: self

        yes = len_trim(self%name) > 0 .and. self%first >= 1 .and. &
            self%last >= self%first .and. ieee_is_finite(self%log_l2) .and. &
            self%log_l2 >= LOG_L2_MIN .and. self%log_l2 <= LOG_L2_MAX
    end function mlp_parameter_group_initialized

    subroutine mlp_grouped_objective_initialize(self, model, x, target, groups, status, &
            device_kind)
        class(mlp_grouped_training_objective_t), intent(out) :: self
        type(mlp_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(mlp_parameter_group_t), intent(in) :: groups(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: device_kind
        integer :: requested_device, i, j, n_model

        self%device_kind = FORTML_DEVICE_CPU
        if (present(device_kind)) self%device_kind = device_kind
        if (self%device_kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP grouped objective: CUDA resident derivative graph is not implemented")
            return
        end if
        if (self%device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: device kind is invalid")
            return
        end if
        requested_device = self%device_kind
        n_model = model%parameter_count()
        if (.not. valid_data(model, x, target) .or. n_model < 1 .or. &
                size(groups) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: model, data, or groups are invalid")
            return
        end if
        do i = 1, size(groups)
            if (.not. groups(i)%initialized() .or. groups(i)%last > n_model) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP grouped objective: group range or coefficient is invalid")
                return
            end if
            do j = 1, i - 1
                if (trim(groups(i)%name) == trim(groups(j)%name) .or. &
                        ranges_overlap(groups(i), groups(j))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "MLP grouped objective: group names and ranges must be unique")
                    return
                end if
            end do
        end do
        self%model => model
        allocate(self%features, source=x)
        allocate(self%targets, source=target)
        allocate(self%groups(size(groups)))
        self%groups = groups
        if (requested_device /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: device selection was not retained")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_grouped_objective_initialize

    logical function mlp_grouped_objective_initialized(self) result(yes)
        class(mlp_grouped_training_objective_t), intent(in) :: self

        yes = associated(self%model) .and. allocated(self%features) .and. &
            allocated(self%targets) .and. allocated(self%groups)
        if (.not. yes) return
        yes = size(self%groups) > 0
    end function mlp_grouped_objective_initialized

    logical function mlp_grouped_device_supported(self, device_kind) result(yes)
        class(mlp_grouped_training_objective_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = requested == FORTML_DEVICE_CPU
    end function mlp_grouped_device_supported

    integer function mlp_grouped_parameter_count(self) result(n)
        class(mlp_grouped_training_objective_t), intent(in) :: self

        n = 0
        if (.not. associated(self%model)) return
        n = self%model%parameter_count()
        if (allocated(self%groups)) n = n + size(self%groups)
    end function mlp_grouped_parameter_count

    integer function mlp_grouped_group_count(self) result(n)
        class(mlp_grouped_training_objective_t), intent(in) :: self

        n = 0
        if (allocated(self%groups)) n = size(self%groups)
    end function mlp_grouped_group_count

    function mlp_grouped_parameters(self) result(parameters)
        class(mlp_grouped_training_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: n_model, i

        allocate(parameters(self%parameter_count()))
        parameters = 0.0_dp
        if (.not. self%initialized()) return
        n_model = self%model%parameter_count()
        parameters(:n_model) = self%model%parameters()
        do i = 1, self%group_count()
            parameters(n_model + i) = self%groups(i)%log_l2
        end do
    end function mlp_grouped_parameters

    function mlp_grouped_group_name(self, index) result(name)
        class(mlp_grouped_training_objective_t), intent(in) :: self
        integer, intent(in) :: index
        character(len=64) :: name

        name = ""
        if (index < 1 .or. index > self%group_count()) return
        name = self%groups(index)%name
    end function mlp_grouped_group_name

    subroutine mlp_grouped_group_range(self, index, first, last, status)
        class(mlp_grouped_training_objective_t), intent(in) :: self
        integer, intent(in) :: index
        integer, intent(out) :: first, last
        type(fortnum_status_t), intent(out) :: status

        first = 0
        last = -1
        if (index < 1 .or. index > self%group_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: group index is invalid")
            return
        end if
        first = self%groups(index)%first
        last = self%groups(index)%last
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_grouped_group_range

    subroutine mlp_grouped_value_gradient(self, parameters, value, gradient, status)
        class(mlp_grouped_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: base_gradient(:), theta(:)
        real(dp) :: base_value, ignored_l2, lambda, norm2
        integer :: n_model, i, first, last

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
                size(gradient) /= size(parameters) .or. &
                any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: parameter or gradient shape/value is invalid")
            return
        end if
        if (any(parameters(n_model + 1:) < LOG_L2_MIN) .or. &
                any(parameters(n_model + 1:) > LOG_L2_MAX)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: log L2 is outside the safe range")
            return
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        allocate(base_gradient(n_model))
        call mlp_loss_value_gradient(self%model, self%features, self%targets, 0.0_dp, &
            base_value, base_gradient, ignored_l2, status)
        if (status%code /= FORTNUM_OK) return
        theta = self%model%parameters()
        value = base_value
        gradient(:n_model) = base_gradient
        do i = 1, self%group_count()
            first = self%groups(i)%first
            last = self%groups(i)%last
            lambda = exp(parameters(n_model + i))
            norm2 = dot_product(theta(first:last), theta(first:last))
            value = value + 0.5_dp*lambda*norm2
            gradient(first:last) = gradient(first:last) + lambda*theta(first:last)
            gradient(n_model + i) = 0.5_dp*lambda*norm2
        end do
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_grouped_value_gradient

    subroutine mlp_grouped_jvp(self, parameters, direction, value, tangent, status)
        class(mlp_grouped_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters) .or. &
                any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective JVP: direction shape/value is invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective JVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_grouped_jvp

    subroutine mlp_grouped_vjp(self, parameters, output_bar, gradient, status)
        class(mlp_grouped_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective VJP: output cotangent is not finite")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_grouped_vjp

    subroutine mlp_grouped_hvp(self, parameters, direction, product, status)
        class(mlp_grouped_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), base_hvp(:)
        real(dp) :: ignored_hvp, lambda, norm2
        integer :: n_model, i, first, last

        product = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective HVP: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
                size(direction) /= size(parameters) .or. &
                size(product) /= size(parameters) .or. &
                any(.not. ieee_is_finite(parameters)) .or. &
                any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective HVP: parameter or direction is invalid")
            return
        end if
        if (any(parameters(n_model + 1:) < LOG_L2_MIN) .or. &
                any(parameters(n_model + 1:) > LOG_L2_MAX)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective HVP: log L2 is outside the safe range")
            return
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        allocate(base_hvp(n_model), theta(n_model))
        call mlp_loss_hvp(self%model, self%features, self%targets, 0.0_dp, &
            direction(:n_model), 0.0_dp, base_hvp, ignored_hvp, status)
        if (status%code /= FORTNUM_OK) return
        theta = self%model%parameters()
        product(:n_model) = base_hvp
        do i = 1, self%group_count()
            first = self%groups(i)%first
            last = self%groups(i)%last
            lambda = exp(parameters(n_model + i))
            norm2 = dot_product(theta(first:last), theta(first:last))
            product(first:last) = product(first:last) + lambda* &
                (direction(first:last) + theta(first:last)*direction(n_model + i))
            product(n_model + i) = lambda*dot_product(theta(first:last), &
                direction(first:last)) + 0.5_dp*lambda*norm2*direction(n_model + i)
        end do
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_grouped_hvp

    subroutine mlp_grouped_fortopt(self, objective, status)
        class(mlp_grouped_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective

        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            mlp_grouped_context_callback, status)
    end subroutine mlp_grouped_fortopt

    subroutine mlp_grouped_context_callback(context, parameters, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_grouped_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP grouped objective: context has the wrong type")
        end select
    end subroutine mlp_grouped_context_callback

    logical function ranges_overlap(a, b) result(overlap)
        type(mlp_parameter_group_t), intent(in) :: a, b

        overlap = a%first <= b%last .and. b%first <= a%last
    end function ranges_overlap

    logical function valid_data(model, x, target) result(valid)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = model%parameter_count() > 0 .and. size(x, 1) > 0 .and. &
            size(x, 2) == model%layer_sizes(1) .and. &
            size(target, 1) == size(x, 1) .and. &
            size(target, 2) == model%layer_sizes(size(model%layer_sizes)) .and. &
            all(ieee_is_finite(x)) .and. all(ieee_is_finite(target))
    end function valid_data

end module fortml_mlp_grouped_training
