module fortml_device
    !! Explicit device selection and residency accounting for FortML.
    !!
    !! This module is a control-plane contract.  It does not allocate device
    !! buffers and it does not make a host compiler pretend to be a CUDA
    !! runtime.  Operators may register their explicit data regions here so a
    !! caller can report ownership, residency, and transfer costs without
    !! hiding a host fallback.
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    integer, parameter, public :: FORTML_DEVICE_CPU = 0
    integer, parameter, public :: FORTML_DEVICE_CUDA = 1
    integer, parameter, public :: FORTML_DEVICE_INVALID = -1

    integer, parameter, public :: FORTML_BACKEND_HOST = 0
    integer, parameter, public :: FORTML_BACKEND_CUDA = 1
    integer, parameter, public :: FORTML_BACKEND_NONE = -1

    integer, parameter, public :: FORTML_TRANSFER_HOST_TO_DEVICE = 1
    integer, parameter, public :: FORTML_TRANSFER_DEVICE_TO_HOST = 2

    type, public :: fortml_device_capability_t
        logical :: available = .false.
        logical :: host_accessible = .true.
        logical :: supports_residency = .false.
        logical :: supports_async = .false.
        logical :: supports_streams = .false.
        logical :: supports_cuda_kernels = .false.
    end type fortml_device_capability_t

    type, public :: fortml_device_t
        integer :: kind = FORTML_DEVICE_INVALID
        integer :: backend = FORTML_BACKEND_NONE
        integer :: device_index = 0
        integer :: stream_id = 0
        logical :: selected = .false.
        logical :: available = .false.
        logical :: resident = .false.
        logical :: owns_residency = .false.
        integer(int64) :: resident_bytes = 0_int64
        integer(int64) :: host_to_device_bytes = 0_int64
        integer(int64) :: device_to_host_bytes = 0_int64
        integer :: host_to_device_transfers = 0
        integer :: device_to_host_transfers = 0
        type(fortml_device_capability_t) :: capability
    contains
        procedure, public :: select => fortml_device_select
        procedure, public :: clear => fortml_device_clear
        procedure, public :: query => fortml_device_query
        procedure, public :: begin_residency => fortml_device_begin_residency
        procedure, public :: end_residency => fortml_device_end_residency
        procedure, public :: record_host_to_device => &
            fortml_device_record_host_to_device
        procedure, public :: record_device_to_host => &
            fortml_device_record_device_to_host
        procedure, public :: reset_transfer_counters => &
            fortml_device_reset_transfer_counters
        procedure, public :: backend_name => fortml_device_backend_name
        procedure, public :: device_name => fortml_device_name
    end type fortml_device_t

    public :: fortml_device_available
    public :: fortml_query_device
    public :: fortml_cuda_knn_available
    public :: fortml_cuda_knn_plan_create
    public :: fortml_cuda_knn_plan_destroy
    public :: fortml_cuda_knn_plan_predict
    public :: fortml_cuda_mse_available
    public :: fortml_cuda_forest_available
    public :: fortml_cuda_dense_available
    public :: fortml_cuda_knn_predict

    interface
        function fortml_cuda_kernel_available() bind(C, &
                name="fortml_cuda_kernel_available") result(available)
            import :: c_int
            integer(c_int) :: available
        end function fortml_cuda_kernel_available

        function fortml_cuda_rbf_available() bind(C, &
                name="fortml_cuda_rbf_available") result(available)
            import :: c_int
            integer(c_int) :: available
        end function fortml_cuda_rbf_available

        function fortml_cuda_knn_available() bind(C, &
                name="fortml_cuda_knn_available") result(available)
            import :: c_int
            integer(c_int) :: available
        end function fortml_cuda_knn_available

        function fortml_cuda_mse_available() bind(C, &
                name="fortml_cuda_mse_available") result(available)
            import :: c_int
            integer(c_int) :: available
        end function fortml_cuda_mse_available

        function fortml_cuda_forest_available() bind(C, &
                name="fortml_cuda_forest_available") result(available)
            import :: c_int
            integer(c_int) :: available
        end function fortml_cuda_forest_available

        function fortml_cuda_dense_available() bind(C, &
                name="fortml_cuda_dense_available") result(available)
            import :: c_int
            integer(c_int) :: available
        end function fortml_cuda_dense_available

        function fortml_cuda_knn_predict( &
                train_x, train_class, sample_weight, query_x, class_label, &
                output, n_train, n_features, n_query, n_classes, &
                n_neighbors, weighting_code) bind(C, &
                name="fortml_cuda_knn_predict") result(status)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: train_x, train_class, sample_weight
            type(c_ptr), value :: query_x, class_label, output
            integer(c_int), value :: n_train, n_features, n_query, n_classes
            integer(c_int), value :: n_neighbors, weighting_code
            integer(c_int) :: status
        end function fortml_cuda_knn_predict

        function fortml_cuda_knn_plan_create( &
                train_x, train_class, sample_weight, class_label, n_train, &
                n_features, n_classes, n_neighbors, weighting_code, &
                device_index, plan) bind(C, &
                name="fortml_cuda_knn_plan_create") result(status)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: train_x, train_class, sample_weight
            type(c_ptr), value :: class_label
            integer(c_int), value :: n_train, n_features, n_classes
            integer(c_int), value :: n_neighbors, weighting_code, device_index
            type(c_ptr) :: plan
            integer(c_int) :: status
        end function fortml_cuda_knn_plan_create

        function fortml_cuda_knn_plan_destroy(plan) bind(C, &
                name="fortml_cuda_knn_plan_destroy") result(status)
            import :: c_int, c_ptr
            type(c_ptr), value :: plan
            integer(c_int) :: status
        end function fortml_cuda_knn_plan_destroy

        function fortml_cuda_knn_plan_predict( &
                plan, query_x, n_query, output) bind(C, &
                name="fortml_cuda_knn_plan_predict") result(status)
            import :: c_int, c_ptr
            type(c_ptr), value :: plan, query_x, output
            integer(c_int), value :: n_query
            integer(c_int) :: status
        end function fortml_cuda_knn_plan_predict
    end interface

contains

    subroutine fortml_device_select(self, kind, status, device_index, stream_id)
        class(fortml_device_t), intent(inout) :: self
        integer, intent(in) :: kind
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: device_index, stream_id

        type(fortml_device_capability_t) :: capability
        integer :: requested_index, requested_stream
        logical :: available

        requested_index = 0
        if (present(device_index)) requested_index = device_index
        requested_stream = 0
        if (present(stream_id)) requested_stream = stream_id
        if (requested_index < 0 .or. requested_stream < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: device and stream indices must be nonnegative")
            return
        end if
        if (self%resident) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: selection cannot change while residency is active")
            return
        end if
        call fortml_query_device(kind, capability, status)
        if (status%code /= FORTNUM_OK) return
        available = capability%available
        if (.not. available) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "device: requested backend is unavailable in this build")
            return
        end if
        if (kind == FORTML_DEVICE_CPU .and. requested_stream /= 0) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "device: CPU selection has no asynchronous stream")
            return
        end if
        if (kind == FORTML_DEVICE_CUDA .and. requested_stream /= 0 .and. &
            .not. capability%supports_streams) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "device: CUDA stream ownership is not implemented")
            return
        end if

        self%kind = kind
        self%backend = merge(FORTML_BACKEND_CUDA, FORTML_BACKEND_HOST, &
            kind == FORTML_DEVICE_CUDA)
        self%device_index = requested_index
        self%stream_id = requested_stream
        self%selected = .true.
        self%available = .true.
        self%capability = capability
        call fortml_device_reset_transfer_counters(self)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortml_device_select

    subroutine fortml_device_clear(self, status)
        class(fortml_device_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status

        if (self%resident) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: clear requires end_residency first")
            return
        end if
        self%kind = FORTML_DEVICE_INVALID
        self%backend = FORTML_BACKEND_NONE
        self%device_index = 0
        self%stream_id = 0
        self%selected = .false.
        self%available = .false.
        self%capability = fortml_device_capability_t()
        call fortml_device_reset_transfer_counters(self)
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortml_device_clear

    subroutine fortml_device_query(self, capability, status)
        class(fortml_device_t), intent(in) :: self
        type(fortml_device_capability_t), intent(out) :: capability
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%selected) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: select a device before querying its capability")
            return
        end if
        capability = self%capability
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortml_device_query

    subroutine fortml_device_begin_residency(self, bytes, status, owns_data)
        class(fortml_device_t), intent(inout) :: self
        integer(int64), intent(in) :: bytes
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: owns_data

        logical :: owner

        if (.not. self%selected .or. .not. self%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: select an available device before residency")
            return
        end if
        if (bytes < 0_int64) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: resident byte count must be nonnegative")
            return
        end if
        if (self%resident) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: residency is already active")
            return
        end if
        owner = .false.
        if (present(owns_data)) owner = owns_data
        self%resident = .true.
        self%owns_residency = owner
        self%resident_bytes = bytes
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortml_device_begin_residency

    subroutine fortml_device_end_residency(self, status)
        class(fortml_device_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status

        self%resident = .false.
        self%owns_residency = .false.
        self%resident_bytes = 0_int64
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortml_device_end_residency

    subroutine fortml_device_record_host_to_device(self, bytes, status)
        class(fortml_device_t), intent(inout) :: self
        integer(int64), intent(in) :: bytes
        type(fortnum_status_t), intent(out) :: status

        call record_transfer(self, FORTML_TRANSFER_HOST_TO_DEVICE, bytes, status)
    end subroutine fortml_device_record_host_to_device

    subroutine fortml_device_record_device_to_host(self, bytes, status)
        class(fortml_device_t), intent(inout) :: self
        integer(int64), intent(in) :: bytes
        type(fortnum_status_t), intent(out) :: status

        call record_transfer(self, FORTML_TRANSFER_DEVICE_TO_HOST, bytes, status)
    end subroutine fortml_device_record_device_to_host

    subroutine record_transfer(self, direction, bytes, status)
        class(fortml_device_t), intent(inout) :: self
        integer, intent(in) :: direction
        integer(int64), intent(in) :: bytes
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%selected .or. .not. self%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: select an available device before recording transfer")
            return
        end if
        if (bytes < 0_int64) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: transfer byte count must be nonnegative")
            return
        end if
        if (.not. self%resident) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: transfer requires active residency")
            return
        end if
        if (self%kind == FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "device: CPU selection has no host-device transfer")
            return
        end if
        select case (direction)
        case (FORTML_TRANSFER_HOST_TO_DEVICE)
            self%host_to_device_bytes = self%host_to_device_bytes + bytes
            self%host_to_device_transfers = self%host_to_device_transfers + 1
        case (FORTML_TRANSFER_DEVICE_TO_HOST)
            self%device_to_host_bytes = self%device_to_host_bytes + bytes
            self%device_to_host_transfers = self%device_to_host_transfers + 1
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: transfer direction is invalid")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine record_transfer

    subroutine fortml_device_reset_transfer_counters(self)
        class(fortml_device_t), intent(inout) :: self

        self%host_to_device_bytes = 0_int64
        self%device_to_host_bytes = 0_int64
        self%host_to_device_transfers = 0
        self%device_to_host_transfers = 0
    end subroutine fortml_device_reset_transfer_counters

    character(len=16) function fortml_device_backend_name(self) result(name)
        class(fortml_device_t), intent(in) :: self

        select case (self%backend)
        case (FORTML_BACKEND_HOST)
            name = "CPU"
        case (FORTML_BACKEND_CUDA)
            name = "CUDA"
        case default
            name = "NONE"
        end select
    end function fortml_device_backend_name

    character(len=24) function fortml_device_name(self) result(name)
        class(fortml_device_t), intent(in) :: self

        select case (self%kind)
        case (FORTML_DEVICE_CPU)
            write (name, '(a,i0)') "cpu:", self%device_index
        case (FORTML_DEVICE_CUDA)
            write (name, '(a,i0)') "cuda:", self%device_index
        case default
            name = "unselected"
        end select
    end function fortml_device_name

    subroutine fortml_device_available(kind, available, status)
        integer, intent(in) :: kind
        logical, intent(out) :: available
        type(fortnum_status_t), intent(out) :: status

        type(fortml_device_capability_t) :: capability

        call fortml_query_device(kind, capability, status)
        available = .false.
        if (status%code == FORTNUM_OK) available = capability%available
    end subroutine fortml_device_available

    subroutine fortml_query_device(kind, capability, status)
        integer, intent(in) :: kind
        type(fortml_device_capability_t), intent(out) :: capability
        type(fortnum_status_t), intent(out) :: status

        integer(c_int) :: cuda_kernel, cuda_rbf, cuda_knn, cuda_mse, cuda_forest, &
            cuda_dense

        capability = fortml_device_capability_t()
        select case (kind)
        case (FORTML_DEVICE_CPU)
            capability%available = .true.
            capability%host_accessible = .true.
            capability%supports_residency = .true.
        case (FORTML_DEVICE_CUDA)
            cuda_kernel = fortml_cuda_kernel_available()
            cuda_rbf = fortml_cuda_rbf_available()
            cuda_knn = fortml_cuda_knn_available()
            cuda_mse = fortml_cuda_mse_available()
            cuda_forest = fortml_cuda_forest_available()
            cuda_dense = fortml_cuda_dense_available()
            capability%supports_cuda_kernels = &
                cuda_kernel /= 0_c_int .or. cuda_rbf /= 0_c_int .or. &
                cuda_knn /= 0_c_int .or. cuda_mse /= 0_c_int .or. &
                cuda_forest /= 0_c_int .or. cuda_dense /= 0_c_int
            capability%available = capability%supports_cuda_kernels
            capability%host_accessible = .false.
            capability%supports_residency = capability%available
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "device: kind must be FORTML_DEVICE_CPU or FORTML_DEVICE_CUDA")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine fortml_query_device

end module fortml_device
