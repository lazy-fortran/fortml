module fortml_parameter_registry
    !! One optimizer-facing packing contract for heterogeneous model parameters.
    !!
    !! A block keeps a pointer to the live model object and procedure pointers
    !! for packing and unpacking it. The registry therefore owns no duplicate
    !! model state: optimizer updates are applied directly to the registered
    !! MLP, kernel, likelihood, inducing, or variational object. New model
    !! families provide two small callbacks instead of adding dispatch cases
    !! to the registry.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_mlp, only: mlp_t
    use fortml_kernels, only: kernel_t
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_ridge_regression, only: ridge_regression_t
    use fortml_elastic_net_regression, only: elastic_net_regression_t
    use fortml_logistic_regression, only: logistic_regression_t
    use fortml_softmax_regression, only: softmax_regression_t
    use fortml_pipeline, only: basis_pipeline_t, sequential_basis_pipeline_t
    use fortml_mlp_chain, only: mlp_chain_t
    implicit none
    private

    abstract interface
        subroutine parameter_get_proc(context, values, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(out) :: values(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine parameter_get_proc

        subroutine parameter_set_proc(context, values, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(inout) :: context
            real(dp), intent(in) :: values(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine parameter_set_proc
    end interface

    type, public :: parameter_block_t
        private
        character(:), allocatable :: block_name
        integer :: n_parameters = 0
        class(*), pointer :: context => null()
        procedure(parameter_get_proc), pointer, nopass :: getter => null()
        procedure(parameter_set_proc), pointer, nopass :: setter => null()
    contains
        procedure, public :: initialize => parameter_block_initialize
        procedure, public :: name => parameter_block_name
        procedure, public :: size => parameter_block_size
        procedure, public :: get => parameter_block_get
        procedure, public :: set => parameter_block_set
        procedure, public :: initialized => parameter_block_initialized
    end type parameter_block_t

    type, public :: parameter_registry_t
        private
        type(parameter_block_t), allocatable :: block(:)
        integer :: n_blocks = 0
        integer :: n_parameters = 0
    contains
        procedure, public :: clear => parameter_registry_clear
        procedure, public :: add => parameter_registry_add
        procedure, public :: block_count => parameter_registry_block_count
        procedure, public :: parameter_count => parameter_registry_parameter_count
        procedure, public :: pack => parameter_registry_pack
        procedure, public :: unpack => parameter_registry_unpack
        procedure, public :: range => parameter_registry_range
    end type parameter_registry_t

    public :: parameter_get_proc, parameter_set_proc
    public :: parameter_block_from_mlp, parameter_block_from_kernel, &
        parameter_block_from_gp, parameter_block_from_ridge, &
        parameter_block_from_elastic_net, parameter_block_from_logistic, &
        parameter_block_from_softmax, parameter_block_from_basis_pipeline, &
        parameter_block_from_sequential_pipeline, parameter_block_from_mlp_chain

contains

    subroutine parameter_block_initialize(self, name, n_parameters, context, &
            getter, setter, status)
        class(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        integer, intent(in) :: n_parameters
        class(*), target, intent(inout) :: context
        procedure(parameter_get_proc) :: getter
        procedure(parameter_set_proc) :: setter
        type(fortnum_status_t), intent(out) :: status

        if (len_trim(name) == 0 .or. n_parameters < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter block: name or size is invalid")
            return
        end if

        self%block_name = trim(name)
        self%n_parameters = n_parameters
        self%context => context
        self%getter => getter
        self%setter => setter
        call status_set(status, FORTNUM_OK, "")
    end subroutine parameter_block_initialize

    function parameter_block_name(self) result(name)
        class(parameter_block_t), intent(in) :: self
        character(:), allocatable :: name

        name = ""
        if (allocated(self%block_name)) name = self%block_name
    end function parameter_block_name

    integer function parameter_block_size(self) result(n_parameters)
        class(parameter_block_t), intent(in) :: self

        n_parameters = self%n_parameters
    end function parameter_block_size

    logical function parameter_block_initialized(self) result(yes)
        class(parameter_block_t), intent(in) :: self

        yes = .false.
        if (.not. allocated(self%block_name)) return
        if (.not. associated(self%context)) return
        if (.not. associated(self%getter)) return
        if (.not. associated(self%setter)) return
        yes = self%n_parameters > 0
    end function parameter_block_initialized

    subroutine parameter_block_get(self, values, status)
        class(parameter_block_t), intent(in) :: self
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter block: block is not initialized")
            return
        end if
        if (size(values) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter block: output shape is invalid")
            return
        end if
        call self%getter(self%context, values, status)
    end subroutine parameter_block_get

    subroutine parameter_block_set(self, values, status)
        class(parameter_block_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter block: block is not initialized")
            return
        end if
        if (size(values) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter block: input shape is invalid")
            return
        end if
        call self%setter(self%context, values, status)
    end subroutine parameter_block_set

    subroutine parameter_registry_clear(self)
        class(parameter_registry_t), intent(inout) :: self

        if (allocated(self%block)) deallocate(self%block)
        self%n_blocks = 0
        self%n_parameters = 0
    end subroutine parameter_registry_clear

    subroutine parameter_registry_add(self, block, status)
        class(parameter_registry_t), intent(inout) :: self
        type(parameter_block_t), intent(in) :: block
        type(fortnum_status_t), intent(out) :: status
        type(parameter_block_t), allocatable :: extended(:)
        integer :: i

        if (.not. block%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter registry: block is not initialized")
            return
        end if
        do i = 1, self%n_blocks
            if (self%block(i)%name() == block%name()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "parameter registry: block names must be unique")
                return
            end if
        end do

        allocate(extended(self%n_blocks + 1))
        do i = 1, self%n_blocks
            extended(i) = self%block(i)
        end do
        extended(self%n_blocks + 1) = block
        call move_alloc(extended, self%block)
        self%n_blocks = self%n_blocks + 1
        self%n_parameters = self%n_parameters + block%size()
        call status_set(status, FORTNUM_OK, "")
    end subroutine parameter_registry_add

    integer function parameter_registry_block_count(self) result(count)
        class(parameter_registry_t), intent(in) :: self

        count = self%n_blocks
    end function parameter_registry_block_count

    integer function parameter_registry_parameter_count(self) result(count)
        class(parameter_registry_t), intent(in) :: self

        count = self%n_parameters
    end function parameter_registry_parameter_count

    subroutine parameter_registry_pack(self, values, status)
        class(parameter_registry_t), intent(in) :: self
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last

        if (size(values) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter registry: packed output shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_blocks
            last = first + self%block(i)%size() - 1
            call self%block(i)%get(values(first:last), status)
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine parameter_registry_pack

    subroutine parameter_registry_unpack(self, values, status)
        class(parameter_registry_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last

        if (size(values) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "parameter registry: packed input shape is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_blocks
            last = first + self%block(i)%size() - 1
            call self%block(i)%set(values(first:last), status)
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine parameter_registry_unpack

    subroutine parameter_registry_range(self, name, first, last, found)
        class(parameter_registry_t), intent(in) :: self
        character(*), intent(in) :: name
        integer, intent(out) :: first, last
        logical, intent(out) :: found
        integer :: i

        first = 0
        last = -1
        found = .false.
        first = 1
        do i = 1, self%n_blocks
            last = first + self%block(i)%size() - 1
            if (self%block(i)%name() == trim(name)) then
                found = .true.
                return
            end if
            first = last + 1
        end do
        first = 0
        last = -1
    end subroutine parameter_registry_range

    subroutine parameter_block_from_mlp(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(mlp_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            mlp_parameter_get, mlp_parameter_set, status)
    end subroutine parameter_block_from_mlp

    subroutine parameter_block_from_mlp_chain(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(mlp_chain_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            mlp_chain_parameter_get, mlp_chain_parameter_set, status)
    end subroutine parameter_block_from_mlp_chain

    subroutine parameter_block_from_kernel(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(kernel_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            kernel_parameter_get, kernel_parameter_set, status)
    end subroutine parameter_block_from_kernel

    subroutine parameter_block_from_gp(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(gp_regression_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            gp_parameter_get, gp_parameter_set, status)
    end subroutine parameter_block_from_gp

    subroutine parameter_block_from_ridge(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(ridge_regression_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            ridge_parameter_get, ridge_parameter_set, status)
    end subroutine parameter_block_from_ridge

    subroutine parameter_block_from_elastic_net(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(elastic_net_regression_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            elastic_net_parameter_get, elastic_net_parameter_set, status)
    end subroutine parameter_block_from_elastic_net

    subroutine parameter_block_from_logistic(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(logistic_regression_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            logistic_parameter_get, logistic_parameter_set, status)
    end subroutine parameter_block_from_logistic

    subroutine parameter_block_from_softmax(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(softmax_regression_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            softmax_parameter_get, softmax_parameter_set, status)
    end subroutine parameter_block_from_softmax

    subroutine parameter_block_from_basis_pipeline(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(basis_pipeline_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            basis_pipeline_parameter_get, basis_pipeline_parameter_set, status)
    end subroutine parameter_block_from_basis_pipeline

    subroutine parameter_block_from_sequential_pipeline(self, name, model, status)
        type(parameter_block_t), intent(out) :: self
        character(*), intent(in) :: name
        type(sequential_basis_pipeline_t), target, intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call self%initialize(name, model%parameter_count(), model, &
            sequential_pipeline_parameter_get, sequential_pipeline_parameter_set, status)
    end subroutine parameter_block_from_sequential_pipeline

    subroutine mlp_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (mlp_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP parameter block: output shape is invalid")
                return
            end if
            values = model%parameters()
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP parameter block: context has the wrong type")
        end select
    end subroutine mlp_parameter_get

    subroutine mlp_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (mlp_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP parameter block: context has the wrong type")
        end select
    end subroutine mlp_parameter_set

    subroutine mlp_chain_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: parameters(:)

        select type (model => context)
            type is (mlp_chain_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP chain parameter block: output shape is invalid")
                return
            end if
            parameters = model%parameters()
            values = parameters
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain parameter block: context has the wrong type")
        end select
    end subroutine mlp_chain_parameter_get

    subroutine mlp_chain_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (mlp_chain_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP chain parameter block: context has the wrong type")
        end select
    end subroutine mlp_chain_parameter_set

    subroutine kernel_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (kernel_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel parameter block: output shape is invalid")
                return
            end if
            values = model%parameters()
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel parameter block: context has the wrong type")
        end select
    end subroutine kernel_parameter_get

    subroutine kernel_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (kernel_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel parameter block: context has the wrong type")
        end select
    end subroutine kernel_parameter_set

    subroutine gp_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (gp_regression_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP parameter block: output shape is invalid")
                return
            end if
            values = model%parameters()
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP parameter block: context has the wrong type")
        end select
    end subroutine gp_parameter_get

    subroutine gp_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (gp_regression_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP parameter block: context has the wrong type")
        end select
    end subroutine gp_parameter_set

    subroutine ridge_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (ridge_regression_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ridge parameter block: output shape is invalid")
                return
            end if
            values = model%parameters()
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge parameter block: context has the wrong type")
        end select
    end subroutine ridge_parameter_get

    subroutine ridge_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (ridge_regression_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge parameter block: context has the wrong type")
        end select
    end subroutine ridge_parameter_set

    subroutine elastic_net_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (elastic_net_regression_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "elastic-net parameter block: output shape is invalid")
                return
            end if
            values = model%parameters()
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net parameter block: context has the wrong type")
        end select
    end subroutine elastic_net_parameter_get

    subroutine elastic_net_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (elastic_net_regression_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net parameter block: context has the wrong type")
        end select
    end subroutine elastic_net_parameter_set

    subroutine logistic_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (logistic_regression_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic parameter block: output shape is invalid")
                return
            end if
            values = model%parameters()
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic parameter block: context has the wrong type")
        end select
    end subroutine logistic_parameter_get

    subroutine logistic_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (logistic_regression_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic parameter block: context has the wrong type")
        end select
    end subroutine logistic_parameter_set

    subroutine softmax_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (softmax_regression_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "softmax parameter block: output shape is invalid")
                return
            end if
            values = model%parameters()
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax parameter block: context has the wrong type")
        end select
    end subroutine softmax_parameter_get

    subroutine softmax_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (softmax_regression_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax parameter block: context has the wrong type")
        end select
    end subroutine softmax_parameter_set

    subroutine basis_pipeline_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (basis_pipeline_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "basis-pipeline parameter block: output shape is invalid")
                return
            end if
            values = model%parameters()
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis-pipeline parameter block: context has the wrong type")
        end select
    end subroutine basis_pipeline_parameter_get

    subroutine basis_pipeline_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (basis_pipeline_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis-pipeline parameter block: context has the wrong type")
        end select
    end subroutine basis_pipeline_parameter_set

    subroutine sequential_pipeline_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (sequential_basis_pipeline_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "sequential-pipeline parameter block: output shape is invalid")
                return
            end if
            values = model%parameters()
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential-pipeline parameter block: context has the wrong type")
        end select
    end subroutine sequential_pipeline_parameter_get

    subroutine sequential_pipeline_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (sequential_basis_pipeline_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sequential-pipeline parameter block: context has the wrong type")
        end select
    end subroutine sequential_pipeline_parameter_set

end module fortml_parameter_registry
