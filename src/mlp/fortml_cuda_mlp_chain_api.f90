module fortml_cuda_mlp_chain_api
    !! Typed wrapper for a resident, no-autodiff CUDA dense MLP chain.
    !!
    !! The chain owns every layer's weights and biases on one selected device.
    !! Prediction and fixed-state input/parameter JVP/VJP products explicitly
    !! transfer their batches; the ordinary build links the typed unavailable
    !! stub and never executes a CPU fallback.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int64_t, c_loc, &
        c_null_ptr, c_ptr, c_associated
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    public :: fortml_cuda_mlp_chain_available
    public :: cuda_mlp_chain_plan_t

    interface
        function fortml_cuda_mlp_chain_available() bind(C, &
                name="fortml_cuda_mlp_chain_available") result(value)
            import :: c_int
            integer(c_int) :: value
        end function fortml_cuda_mlp_chain_available

        function fortml_cuda_mlp_chain_create(layer_sizes, activations, &
                weights, biases, n_layers, device_index, handle) bind(C, &
                name="fortml_cuda_mlp_chain_create") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: layer_sizes, activations, weights, biases
            integer(c_int), value :: n_layers, device_index
            type(c_ptr) :: handle
            integer(c_int) :: value
        end function fortml_cuda_mlp_chain_create

        function fortml_cuda_mlp_chain_predict(handle, query_x, n_query, &
                outputs) bind(C, name="fortml_cuda_mlp_chain_predict") &
                result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: handle, query_x, outputs
            integer(c_int), value :: n_query
            integer(c_int) :: value
        end function fortml_cuda_mlp_chain_predict

        function fortml_cuda_mlp_chain_jvp(handle, query_x, query_x_dot, &
                weights_dot, biases_dot, n_query, outputs, outputs_dot) &
                bind(C, name="fortml_cuda_mlp_chain_jvp") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: handle, query_x, query_x_dot, weights_dot, &
                biases_dot, outputs, outputs_dot
            integer(c_int), value :: n_query
            integer(c_int) :: value
        end function fortml_cuda_mlp_chain_jvp

        function fortml_cuda_mlp_chain_vjp(handle, query_x, output_bar, &
                n_query, query_x_bar, weights_bar, biases_bar) bind(C, &
                name="fortml_cuda_mlp_chain_vjp") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: handle, query_x, output_bar
            integer(c_int), value :: n_query
            type(c_ptr), value :: query_x_bar, weights_bar, biases_bar
            integer(c_int) :: value
        end function fortml_cuda_mlp_chain_vjp

        function fortml_cuda_mlp_chain_transfer_stats(handle, &
                host_to_device_bytes, device_to_host_bytes, resident_bytes) &
                bind(C, name="fortml_cuda_mlp_chain_transfer_stats") &
                result(value)
            import :: c_int, c_int64_t, c_ptr
            type(c_ptr), value :: handle
            integer(c_int64_t) :: host_to_device_bytes, device_to_host_bytes
            integer(c_int64_t) :: resident_bytes
            integer(c_int) :: value
        end function fortml_cuda_mlp_chain_transfer_stats

        function fortml_cuda_mlp_chain_destroy(handle) bind(C, &
                name="fortml_cuda_mlp_chain_destroy") result(value)
            import :: c_int, c_ptr
            type(c_ptr), value :: handle
            integer(c_int) :: value
        end function fortml_cuda_mlp_chain_destroy
    end interface

    type, public :: cuda_mlp_chain_plan_t
        private
        type(c_ptr) :: handle = c_null_ptr
        integer :: n_layers = 0
        integer :: input_width = 0
        integer :: output_width = 0
        integer :: weight_count_value = 0
        integer :: bias_count_value = 0
        integer :: parameter_count_value = 0
        integer :: device_index = -1
    contains
        procedure, public :: create => cuda_mlp_chain_create
        procedure, public :: predict => cuda_mlp_chain_predict
        procedure, public :: jvp => cuda_mlp_chain_jvp
        procedure, public :: vjp => cuda_mlp_chain_vjp
        procedure, public :: transfer_stats => cuda_mlp_chain_transfer_stats
        procedure, public :: destroy => cuda_mlp_chain_destroy
        procedure, public :: fitted => cuda_mlp_chain_fitted
        procedure, public :: stage_count => cuda_mlp_chain_stage_count
        procedure, public :: input_count => cuda_mlp_chain_input_count
        procedure, public :: output_count => cuda_mlp_chain_output_count
        procedure, public :: parameter_count => cuda_mlp_chain_parameter_count
        procedure, public :: device => cuda_mlp_chain_device
    end type cuda_mlp_chain_plan_t

contains

    subroutine cuda_mlp_chain_create(self, layer_sizes, activations, weights, &
            biases, device_index, status)
        class(cuda_mlp_chain_plan_t), intent(out) :: self
        integer, intent(in) :: layer_sizes(:), activations(:), device_index
        real(dp), intent(in), target, contiguous :: weights(:), biases(:)
        type(fortnum_status_t), intent(out) :: status
        integer(c_int), allocatable, target :: layer_sizes_c(:), activations_c(:)
        type(c_ptr) :: handle
        integer(c_int) :: code
        integer :: n_layers, expected_weights, expected_biases, layer

        self%handle = c_null_ptr
        self%n_layers = 0
        self%input_width = 0
        self%output_width = 0
        self%weight_count_value = 0
        self%bias_count_value = 0
        self%parameter_count_value = 0
        self%device_index = -1
        n_layers = size(activations)
        if (n_layers < 1 .or. size(layer_sizes) /= n_layers + 1 .or. &
                device_index < 0 .or. any(layer_sizes < 1) .or. &
                any(activations < 1) .or. any(activations > 8)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA MLP chain: topology or activation is invalid")
            return
        end if
        expected_weights = 0
        expected_biases = 0
        do layer = 1, n_layers
            expected_weights = expected_weights + &
                layer_sizes(layer) * layer_sizes(layer + 1)
            expected_biases = expected_biases + layer_sizes(layer + 1)
        end do
        if (size(weights) /= expected_weights .or. &
                size(biases) /= expected_biases .or. &
                any(.not. ieee_is_finite(weights)) .or. &
                any(.not. ieee_is_finite(biases))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA MLP chain: packed parameters are invalid")
            return
        end if
        allocate(layer_sizes_c(n_layers + 1), activations_c(n_layers))
        layer_sizes_c = int(layer_sizes, c_int)
        activations_c = int(activations, c_int)
        handle = c_null_ptr
        code = fortml_cuda_mlp_chain_create(c_loc(layer_sizes_c), &
            c_loc(activations_c), c_loc(weights), c_loc(biases), &
            int(n_layers, c_int), int(device_index, c_int), handle)
        if (code /= 0_c_int .or. .not. c_associated(handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: resident native plan is unavailable")
            return
        end if
        self%handle = handle
        self%n_layers = n_layers
        self%input_width = layer_sizes(1)
        self%output_width = layer_sizes(n_layers + 1)
        self%weight_count_value = expected_weights
        self%bias_count_value = expected_biases
        self%parameter_count_value = expected_weights + expected_biases
        self%device_index = device_index
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_mlp_chain_create

    subroutine cuda_mlp_chain_predict(self, query_x, outputs, status)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        real(dp), intent(in), target, contiguous :: query_x(:, :)
        real(dp), intent(inout), target, contiguous :: outputs(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code
        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: plan is not fitted")
            return
        end if
        if (size(query_x, 2) /= self%input_width .or. &
                size(outputs, 1) /= size(query_x, 1) .or. &
                size(outputs, 2) /= self%output_width .or. size(query_x, 1) < 1 .or. &
                any(.not. ieee_is_finite(query_x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA MLP chain: query or output shape is invalid")
            return
        end if
        code = fortml_cuda_mlp_chain_predict(self%handle, c_loc(query_x), &
            int(size(query_x, 1), c_int), c_loc(outputs))
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: resident prediction failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_mlp_chain_predict

    subroutine cuda_mlp_chain_jvp(self, query_x, query_x_dot, weights_dot, &
            biases_dot, outputs, outputs_dot, status)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        real(dp), intent(in), target, contiguous :: query_x(:, :), query_x_dot(:, :)
        real(dp), intent(in), target, contiguous :: weights_dot(:), biases_dot(:)
        real(dp), intent(inout), target, contiguous :: outputs(:, :), outputs_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code
        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: plan is not fitted")
            return
        end if
        if (size(query_x, 2) /= self%input_width .or. &
                any(shape(query_x_dot) /= shape(query_x)) .or. &
                size(weights_dot) /= self%weight_count_value .or. &
                size(biases_dot) /= self%bias_count_value .or. &
                size(outputs, 1) /= size(query_x, 1) .or. &
                size(outputs, 2) /= self%output_width .or. &
                any(shape(outputs_dot) /= shape(outputs)) .or. size(query_x, 1) < 1 .or. &
                any(.not. ieee_is_finite(query_x)) .or. &
                any(.not. ieee_is_finite(query_x_dot)) .or. &
                any(.not. ieee_is_finite(weights_dot)) .or. &
                any(.not. ieee_is_finite(biases_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA MLP chain: JVP query, tangent, or shape is invalid")
            return
        end if
        code = fortml_cuda_mlp_chain_jvp(self%handle, c_loc(query_x), &
            c_loc(query_x_dot), c_loc(weights_dot), c_loc(biases_dot), &
            int(size(query_x, 1), c_int), c_loc(outputs), c_loc(outputs_dot))
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: resident JVP failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_mlp_chain_jvp

    subroutine cuda_mlp_chain_vjp(self, query_x, output_bar, query_x_bar, &
            weights_bar, biases_bar, status)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        real(dp), intent(in), target, contiguous :: query_x(:, :), output_bar(:, :)
        real(dp), intent(inout), target, contiguous :: query_x_bar(:, :), &
            weights_bar(:), biases_bar(:)
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code
        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: plan is not fitted")
            return
        end if
        if (size(query_x, 2) /= self%input_width .or. &
                size(output_bar, 1) /= size(query_x, 1) .or. &
                size(output_bar, 2) /= self%output_width .or. &
                any(shape(query_x_bar) /= shape(query_x)) .or. &
                size(weights_bar) /= self%weight_count_value .or. &
                size(biases_bar) /= self%bias_count_value .or. &
                size(query_x, 1) < 1 .or. any(.not. ieee_is_finite(query_x)) .or. &
                any(.not. ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA MLP chain: VJP query, cotangent, or shape is invalid")
            return
        end if
        code = fortml_cuda_mlp_chain_vjp(self%handle, c_loc(query_x), &
            c_loc(output_bar), int(size(query_x, 1), c_int), c_loc(query_x_bar), &
            c_loc(weights_bar), c_loc(biases_bar))
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: resident VJP failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_mlp_chain_vjp

    subroutine cuda_mlp_chain_transfer_stats(self, host_to_device_bytes, &
            device_to_host_bytes, resident_bytes, status)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        integer(c_int64_t), intent(out) :: host_to_device_bytes, device_to_host_bytes
        integer(c_int64_t), intent(out) :: resident_bytes
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code
        host_to_device_bytes = 0_c_int64_t
        device_to_host_bytes = 0_c_int64_t
        resident_bytes = 0_c_int64_t
        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: plan is not fitted")
            return
        end if
        code = fortml_cuda_mlp_chain_transfer_stats(self%handle, &
            host_to_device_bytes, device_to_host_bytes, resident_bytes)
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: transfer statistics failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_mlp_chain_transfer_stats

    subroutine cuda_mlp_chain_destroy(self, status)
        class(cuda_mlp_chain_plan_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code
        code = 0_c_int
        if (c_associated(self%handle)) code = &
            fortml_cuda_mlp_chain_destroy(self%handle)
        self%handle = c_null_ptr
        self%n_layers = 0
        self%input_width = 0
        self%output_width = 0
        self%weight_count_value = 0
        self%bias_count_value = 0
        self%parameter_count_value = 0
        self%device_index = -1
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MLP chain: destroy failed")
        else
            call status_set(status, FORTNUM_OK, "")
        end if
    end subroutine cuda_mlp_chain_destroy

    logical function cuda_mlp_chain_fitted(self) result(value)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        value = c_associated(self%handle)
    end function cuda_mlp_chain_fitted

    integer function cuda_mlp_chain_stage_count(self) result(value)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        value = self%n_layers
    end function cuda_mlp_chain_stage_count

    integer function cuda_mlp_chain_input_count(self) result(value)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        value = self%input_width
    end function cuda_mlp_chain_input_count

    integer function cuda_mlp_chain_output_count(self) result(value)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        value = self%output_width
    end function cuda_mlp_chain_output_count

    integer function cuda_mlp_chain_parameter_count(self) result(value)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        value = self%parameter_count_value
    end function cuda_mlp_chain_parameter_count

    integer function cuda_mlp_chain_device(self) result(value)
        class(cuda_mlp_chain_plan_t), intent(in) :: self
        value = self%device_index
    end function cuda_mlp_chain_device

end module fortml_cuda_mlp_chain_api
