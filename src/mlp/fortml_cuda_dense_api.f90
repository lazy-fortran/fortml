module fortml_cuda_dense_api
    !! Typed Fortran control-plane wrapper for the resident CUDA dense ABI.
    !!
    !! This product is intentionally prediction-only and no-autodiff.  A
    !! native CUDA build uploads one affine layer once, then copies only each
    !! query batch and result.  The ordinary build links the unavailable stub;
    !! it never labels a CPU execution as CUDA or silently falls back.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_loc, c_null_ptr, &
        c_ptr, c_associated
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: MLP_LINEAR, MLP_LEAKY_RELU
    implicit none
    private

    public :: fortml_cuda_dense_available
    public :: cuda_dense_plan_t

    interface
        function fortml_cuda_dense_available() bind(C, &
                name="fortml_cuda_dense_available") result(value)
            import :: c_int
            integer(c_int) :: value
        end function fortml_cuda_dense_available

        function fortml_cuda_dense_plan_create( &
                weights, bias, n_inputs, n_outputs, activation, device_index, &
                handle) bind(C, name="fortml_cuda_dense_plan_create") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: weights, bias
            integer(c_int), value :: n_inputs, n_outputs, activation, device_index
            type(c_ptr) :: handle
            integer(c_int) :: value
        end function fortml_cuda_dense_plan_create

        function fortml_cuda_dense_plan_predict( &
                handle, query_x, n_query, outputs) bind(C, &
                name="fortml_cuda_dense_plan_predict") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: handle, query_x, outputs
            integer(c_int), value :: n_query
            integer(c_int) :: value
        end function fortml_cuda_dense_plan_predict

        function fortml_cuda_dense_plan_jvp( &
                handle, query_x, query_x_dot, weights_dot, bias_dot, n_query, &
                outputs, outputs_dot) bind(C, &
                name="fortml_cuda_dense_plan_jvp") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: handle, query_x, query_x_dot, weights_dot, bias_dot
            integer(c_int), value :: n_query
            type(c_ptr), value :: outputs, outputs_dot
            integer(c_int) :: value
        end function fortml_cuda_dense_plan_jvp

        function fortml_cuda_dense_plan_destroy(handle) bind(C, &
                name="fortml_cuda_dense_plan_destroy") result(value)
            import :: c_int, c_ptr
            type(c_ptr), value :: handle
            integer(c_int) :: value
        end function fortml_cuda_dense_plan_destroy
    end interface

    type, public :: cuda_dense_plan_t
        private
        type(c_ptr) :: handle = c_null_ptr
        integer :: n_inputs = 0
        integer :: n_outputs = 0
        integer :: activation = MLP_LINEAR
        integer :: device_index = -1
    contains
        procedure, public :: create => cuda_dense_plan_create
        procedure, public :: predict => cuda_dense_plan_predict
        procedure, public :: jvp => cuda_dense_plan_jvp
        procedure, public :: destroy => cuda_dense_plan_destroy
        procedure, public :: fitted => cuda_dense_plan_fitted
        procedure, public :: input_count => cuda_dense_plan_input_count
        procedure, public :: output_count => cuda_dense_plan_output_count
        procedure, public :: activation_kind => cuda_dense_plan_activation
        procedure, public :: device => cuda_dense_plan_device
    end type cuda_dense_plan_t

contains

    subroutine cuda_dense_plan_create(self, weights, bias, activation, &
            device_index, status)
        class(cuda_dense_plan_t), intent(out) :: self
        real(dp), intent(in), target, contiguous :: weights(:, :), bias(:)
        integer, intent(in) :: activation, device_index
        type(fortnum_status_t), intent(out) :: status
        real(c_double), allocatable, target :: weights_c(:), bias_c(:)
        type(c_ptr) :: handle
        integer(c_int) :: code
        integer :: n_inputs, n_outputs, input, output, position

        self%handle = c_null_ptr
        self%n_inputs = 0
        self%n_outputs = 0
        self%activation = MLP_LINEAR
        self%device_index = -1
        n_inputs = size(weights, 1)
        n_outputs = size(weights, 2)
        if (n_inputs < 1 .or. n_outputs < 1 .or. size(bias) /= n_outputs .or. &
            device_index < 0 .or. .not. valid_activation(activation) .or. &
            any(.not. ieee_is_finite(weights)) .or. &
            any(.not. ieee_is_finite(bias))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA dense wrapper: weights, bias, or activation is invalid")
            return
        end if

        allocate(weights_c(n_inputs*n_outputs), bias_c(n_outputs))
        ! The native ABI is output-major; this loop makes the conversion
        ! explicit instead of relying on a compiler-specific transpose view.
        position = 0
        do output = 1, n_outputs
            do input = 1, n_inputs
                position = position + 1
                weights_c(position) = weights(input, output)
            end do
        end do
        bias_c = bias
        handle = c_null_ptr
        code = fortml_cuda_dense_plan_create(c_loc(weights_c), c_loc(bias_c), &
            int(n_inputs, c_int), int(n_outputs, c_int), int(activation, c_int), &
            int(device_index, c_int), handle)
        if (code /= 0_c_int .or. .not. c_associated(handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA dense wrapper: resident native plan is unavailable")
            return
        end if
        self%handle = handle
        self%n_inputs = n_inputs
        self%n_outputs = n_outputs
        self%activation = activation
        self%device_index = device_index
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_dense_plan_create

    subroutine cuda_dense_plan_predict(self, query_x, outputs, status)
        class(cuda_dense_plan_t), intent(in) :: self
        real(dp), intent(in), target, contiguous :: query_x(:, :)
        real(dp), intent(inout), target, contiguous :: outputs(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code

        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA dense wrapper: plan is not fitted")
            return
        end if
        if (size(query_x, 2) /= self%n_inputs .or. &
            size(outputs, 1) /= size(query_x, 1) .or. &
            size(outputs, 2) /= self%n_outputs .or. size(query_x, 1) < 1 .or. &
            any(.not. ieee_is_finite(query_x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA dense wrapper: query or output shape is invalid")
            return
        end if
        code = fortml_cuda_dense_plan_predict(self%handle, c_loc(query_x), &
            int(size(query_x, 1), c_int), c_loc(outputs))
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA dense wrapper: resident prediction failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_dense_plan_predict

    subroutine cuda_dense_plan_jvp(self, query_x, query_x_dot, weights_dot, &
            bias_dot, outputs, outputs_dot, status)
        class(cuda_dense_plan_t), intent(in) :: self
        real(dp), intent(in), target, contiguous :: query_x(:, :), query_x_dot(:, :)
        real(dp), intent(in), target, contiguous :: weights_dot(:, :), bias_dot(:)
        real(dp), intent(inout), target, contiguous :: outputs(:, :), outputs_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(c_double), allocatable, target :: weights_dot_c(:), bias_dot_c(:)
        type(c_ptr) :: query_ptr, query_dot_ptr, weights_dot_ptr, bias_dot_ptr
        type(c_ptr) :: outputs_ptr, outputs_dot_ptr
        integer(c_int) :: code
        integer :: input, output, position

        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA dense wrapper: plan is not fitted")
            return
        end if
        if (size(query_x, 2) /= self%n_inputs .or. &
            any(shape(query_x_dot) /= shape(query_x)) .or. &
            size(weights_dot, 1) /= self%n_inputs .or. &
            size(weights_dot, 2) /= self%n_outputs .or. &
            size(bias_dot) /= self%n_outputs .or. &
            size(outputs, 1) /= size(query_x, 1) .or. &
            size(outputs, 2) /= self%n_outputs .or. &
            any(shape(outputs_dot) /= shape(outputs)) .or. &
            size(query_x, 1) < 1 .or. any(.not. ieee_is_finite(query_x)) .or. &
            any(.not. ieee_is_finite(query_x_dot)) .or. &
            any(.not. ieee_is_finite(weights_dot)) .or. &
            any(.not. ieee_is_finite(bias_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA dense wrapper: JVP query, tangent, or output shape is invalid")
            return
        end if
        allocate(weights_dot_c(self%n_inputs*self%n_outputs), &
            bias_dot_c(self%n_outputs))
        position = 0
        do output = 1, self%n_outputs
            do input = 1, self%n_inputs
                position = position + 1
                weights_dot_c(position) = weights_dot(input, output)
            end do
        end do
        bias_dot_c = bias_dot
        query_ptr = c_loc(query_x)
        query_dot_ptr = c_loc(query_x_dot)
        weights_dot_ptr = c_loc(weights_dot_c)
        bias_dot_ptr = c_loc(bias_dot_c)
        outputs_ptr = c_loc(outputs)
        outputs_dot_ptr = c_loc(outputs_dot)
        code = fortml_cuda_dense_plan_jvp(self%handle, query_ptr, query_dot_ptr, &
            weights_dot_ptr, bias_dot_ptr, int(size(query_x, 1), c_int), &
            outputs_ptr, outputs_dot_ptr)
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA dense wrapper: resident JVP failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_dense_plan_jvp

    subroutine cuda_dense_plan_destroy(self, status)
        class(cuda_dense_plan_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code

        if (c_associated(self%handle)) then
            code = fortml_cuda_dense_plan_destroy(self%handle)
            if (code /= 0_c_int) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "CUDA dense wrapper: plan destruction failed")
                return
            end if
        end if
        self%handle = c_null_ptr
        self%n_inputs = 0
        self%n_outputs = 0
        self%activation = MLP_LINEAR
        self%device_index = -1
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_dense_plan_destroy

    logical function cuda_dense_plan_fitted(self) result(value)
        class(cuda_dense_plan_t), intent(in) :: self
        value = c_associated(self%handle)
    end function cuda_dense_plan_fitted

    integer function cuda_dense_plan_input_count(self) result(value)
        class(cuda_dense_plan_t), intent(in) :: self
        value = self%n_inputs
    end function cuda_dense_plan_input_count

    integer function cuda_dense_plan_output_count(self) result(value)
        class(cuda_dense_plan_t), intent(in) :: self
        value = self%n_outputs
    end function cuda_dense_plan_output_count

    integer function cuda_dense_plan_activation(self) result(value)
        class(cuda_dense_plan_t), intent(in) :: self
        value = self%activation
    end function cuda_dense_plan_activation

    integer function cuda_dense_plan_device(self) result(value)
        class(cuda_dense_plan_t), intent(in) :: self
        value = self%device_index
    end function cuda_dense_plan_device

    logical function valid_activation(activation) result(value)
        integer, intent(in) :: activation
        value = activation >= MLP_LINEAR .and. activation <= MLP_LEAKY_RELU
    end function valid_activation

end module fortml_cuda_dense_api
