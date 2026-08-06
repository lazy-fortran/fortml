module fortml_parameter_products
    !! Optimizer-facing value and derivative products over one packed registry.
    !!
    !! The model keeps its live parameters in the registered object. Directions
    !! and cotangents use the registry's packed ordering, so optimizers never
    !! need model-family dispatch or object-specific offsets. Inputs are fixed
    !! data for this contract; input derivatives remain available on the model
    !! APIs where they are declared.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_parameter_registry, only: parameter_registry_t, &
        parameter_block_t, parameter_block_from_mlp, parameter_block_from_gp
    use fortml_mlp, only: mlp_t
    use fortml_gaussian_process, only: gp_regression_t
    implicit none
    private

    abstract interface
        subroutine parameter_value_proc(context, x, y, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: x(:, :)
            real(dp), intent(out) :: y(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine parameter_value_proc

        subroutine parameter_jvp_proc(context, x, theta_dot, y, y_dot, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: x(:, :), theta_dot(:)
            real(dp), intent(out) :: y(:, :), y_dot(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine parameter_jvp_proc

        subroutine parameter_vjp_proc(context, x, y_bar, theta_bar, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: x(:, :), y_bar(:, :)
            real(dp), intent(out) :: theta_bar(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine parameter_vjp_proc

        subroutine parameter_hvp_proc(context, x, y_bar, theta_dot, &
                theta_hvp, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: x(:, :), y_bar(:, :), theta_dot(:)
            real(dp), intent(out) :: theta_hvp(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine parameter_hvp_proc
    end interface

    type, public :: parameter_products_t
        private
        type(parameter_registry_t) :: registry
        class(*), pointer :: context => null()
        procedure(parameter_value_proc), pointer, nopass :: value_proc => null()
        procedure(parameter_jvp_proc), pointer, nopass :: jvp_proc => null()
        procedure(parameter_vjp_proc), pointer, nopass :: vjp_proc => null()
        procedure(parameter_hvp_proc), pointer, nopass :: hvp_proc => null()
    contains
        procedure, public :: initialized => parameter_products_initialized
        procedure, public :: parameter_count => parameter_products_parameter_count
        procedure, public :: pack => parameter_products_pack
        procedure, public :: unpack => parameter_products_unpack
        procedure, public :: range => parameter_products_range
        procedure, public :: value => parameter_products_value
        procedure, public :: jvp => parameter_products_jvp
        procedure, public :: vjp => parameter_products_vjp
        procedure, public :: hvp => parameter_products_hvp
        procedure, public :: has_hvp => parameter_products_has_hvp
    end type parameter_products_t

    public :: parameter_value_proc, parameter_jvp_proc, parameter_vjp_proc
    public :: parameter_hvp_proc
    public :: parameter_products_from_mlp, parameter_products_from_gp

contains

    subroutine parameter_products_from_mlp(self, name, model, status)
        type(parameter_products_t), intent(out) :: self
        character(*), intent(in) :: name
        type(mlp_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status
        type(parameter_block_t) :: block

        call parameter_block_from_mlp(block, name, model, status)
        if (status%code /= FORTNUM_OK) return
        call self%registry%add(block, status)
        if (status%code /= FORTNUM_OK) return
        self%context => model
        self%value_proc => mlp_parameter_value
        self%jvp_proc => mlp_parameter_jvp
        self%vjp_proc => mlp_parameter_vjp
        self%hvp_proc => mlp_parameter_hvp
    end subroutine parameter_products_from_mlp

    subroutine parameter_products_from_gp(self, name, model, status)
        type(parameter_products_t), intent(out) :: self
        character(*), intent(in) :: name
        type(gp_regression_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status
        type(parameter_block_t) :: block

        call parameter_block_from_gp(block, name, model, status)
        if (status%code /= FORTNUM_OK) return
        call self%registry%add(block, status)
        if (status%code /= FORTNUM_OK) return
        self%context => model
        self%value_proc => gp_parameter_value
        self%jvp_proc => gp_parameter_jvp
        self%vjp_proc => gp_parameter_vjp
    end subroutine parameter_products_from_gp

    logical function parameter_products_initialized(self) result(yes)
        class(parameter_products_t), intent(in) :: self

        yes = associated(self%context) .and. self%registry%parameter_count() > 0
        yes = yes .and. associated(self%value_proc) .and. &
            associated(self%jvp_proc) .and. associated(self%vjp_proc)
    end function parameter_products_initialized

    integer function parameter_products_parameter_count(self) result(count)
        class(parameter_products_t), intent(in) :: self

        count = self%registry%parameter_count()
    end function parameter_products_parameter_count

    subroutine parameter_products_pack(self, theta, status)
        class(parameter_products_t), intent(in) :: self
        real(dp), intent(out) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: object is not initialized")
            return
        end if
        call self%registry%pack(theta, status)
    end subroutine parameter_products_pack

    subroutine parameter_products_unpack(self, theta, status)
        class(parameter_products_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: object is not initialized")
            return
        end if
        call self%registry%unpack(theta, status)
    end subroutine parameter_products_unpack

    subroutine parameter_products_range(self, name, first, last, found)
        class(parameter_products_t), intent(in) :: self
        character(*), intent(in) :: name
        integer, intent(out) :: first, last
        logical, intent(out) :: found

        call self%registry%range(name, first, last, found)
    end subroutine parameter_products_range

    subroutine parameter_products_value(self, x, y, status)
        class(parameter_products_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: object is not initialized")
            return
        end if
        call self%value_proc(self%context, x, y, status)
    end subroutine parameter_products_value

    subroutine parameter_products_jvp(self, x, theta_dot, y, y_dot, status)
        class(parameter_products_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: object is not initialized")
            return
        end if
        if (size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: JVP direction shape is invalid")
            return
        end if
        call self%jvp_proc(self%context, x, theta_dot, y, y_dot, status)
    end subroutine parameter_products_jvp

    subroutine parameter_products_vjp(self, x, y_bar, theta_bar, status)
        class(parameter_products_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: object is not initialized")
            return
        end if
        if (size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: VJP output shape is invalid")
            return
        end if
        call self%vjp_proc(self%context, x, y_bar, theta_bar, status)
    end subroutine parameter_products_vjp

    subroutine parameter_products_hvp(self, x, y_bar, theta_dot, theta_hvp, status)
        class(parameter_products_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y_bar(:, :), theta_dot(:)
        real(dp), intent(out) :: theta_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: object is not initialized")
            return
        end if
        if (.not. associated(self%hvp_proc)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: HVP is not declared for this model")
            return
        end if
        if (size(theta_dot) /= self%parameter_count() .or. &
            size(theta_hvp) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter products: HVP direction or output shape is invalid")
            return
        end if
        call self%hvp_proc(self%context, x, y_bar, theta_dot, theta_hvp, status)
    end subroutine parameter_products_hvp

    logical function parameter_products_has_hvp(self) result(yes)
        class(parameter_products_t), intent(in) :: self

        yes = self%initialized() .and. associated(self%hvp_proc)
    end function parameter_products_has_hvp

    subroutine mlp_parameter_value(context, x, y, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (mlp_t)
            call model%predict(x, y, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP parameter products: context has the wrong type")
        end select
    end subroutine mlp_parameter_value

    subroutine mlp_parameter_jvp(context, x, theta_dot, y, y_dot, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: x_dot(:, :)

        select type (model => context)
            type is (mlp_t)
            allocate(x_dot, source=x)
            x_dot = 0.0_dp
            call model%jvp(x, theta_dot, x_dot, y, y_dot, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP parameter products: context has the wrong type")
        end select
    end subroutine mlp_parameter_jvp

    subroutine mlp_parameter_vjp(context, x, y_bar, theta_bar, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: x(:, :), y_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: x_bar(:, :)

        select type (model => context)
            type is (mlp_t)
            allocate(x_bar, source=x)
            call model%vjp(x, y_bar, theta_bar, x_bar, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP parameter products: context has the wrong type")
        end select
    end subroutine mlp_parameter_vjp

    subroutine mlp_parameter_hvp(context, x, y_bar, theta_dot, theta_hvp, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: x(:, :), y_bar(:, :), theta_dot(:)
        real(dp), intent(out) :: theta_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: x_dot(:, :), x_hvp(:, :)

        select type (model => context)
            type is (mlp_t)
            allocate(x_dot, source=x)
            allocate(x_hvp, source=x)
            x_dot = 0.0_dp
            call model%hvp(x, y_bar, theta_dot, x_dot, theta_hvp, x_hvp, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP parameter products: context has the wrong type")
        end select
    end subroutine mlp_parameter_hvp

    subroutine gp_parameter_value(context, x, y, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: variance(:)

        select type (model => context)
            type is (gp_regression_t)
            allocate(variance(size(x, 1)))
            call model%predict(x, y, variance, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP parameter products: context has the wrong type")
        end select
    end subroutine gp_parameter_value

    subroutine gp_parameter_jvp(context, x, theta_dot, y, y_dot, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: variance(:), variance_dot(:)

        select type (model => context)
            type is (gp_regression_t)
            allocate(variance(size(x, 1)), variance_dot(size(x, 1)))
            call model%predict_jvp(x, theta_dot, y, y_dot, variance, &
                variance_dot, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP parameter products: context has the wrong type")
        end select
    end subroutine gp_parameter_jvp

    subroutine gp_parameter_vjp(context, x, y_bar, theta_bar, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: x(:, :), y_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: variance_bar(:)

        select type (model => context)
            type is (gp_regression_t)
            allocate(variance_bar(size(x, 1)), source=0.0_dp)
            call model%predict_vjp(x, y_bar, variance_bar, theta_bar, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP parameter products: context has the wrong type")
        end select
    end subroutine gp_parameter_vjp

end module fortml_parameter_products
