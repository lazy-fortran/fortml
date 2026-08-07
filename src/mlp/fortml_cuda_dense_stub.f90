function fortml_cuda_dense_available() bind(C, &
        name="fortml_cuda_dense_available") result(value)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: value
    value = 0_c_int
end function fortml_cuda_dense_available

function fortml_cuda_dense_plan_create(weights, bias, n_inputs, n_outputs, &
        activation, device_index, handle) bind(C, &
        name="fortml_cuda_dense_plan_create") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_null_ptr
    implicit none
    type(c_ptr), value :: weights, bias
    integer(c_int), value :: n_inputs, n_outputs, activation, device_index
    type(c_ptr) :: handle
    integer(c_int) :: value
    handle = c_null_ptr
    value = 1_c_int
end function fortml_cuda_dense_plan_create

function fortml_cuda_dense_plan_predict(handle, query_x, n_query, outputs) &
        bind(C, name="fortml_cuda_dense_plan_predict") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle, query_x, outputs
    integer(c_int), value :: n_query
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_dense_plan_predict

function fortml_cuda_dense_plan_jvp(handle, query_x, query_x_dot, weights_dot, &
        bias_dot, n_query, outputs, outputs_dot) bind(C, &
        name="fortml_cuda_dense_plan_jvp") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle, query_x, query_x_dot, weights_dot, bias_dot
    integer(c_int), value :: n_query
    type(c_ptr), value :: outputs, outputs_dot
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_dense_plan_jvp

function fortml_cuda_dense_plan_vjp(handle, query_x, output_bar, n_query, &
        query_x_bar, weights_bar, bias_bar) bind(C, &
        name="fortml_cuda_dense_plan_vjp") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle, query_x, output_bar
    integer(c_int), value :: n_query
    type(c_ptr), value :: query_x_bar, weights_bar, bias_bar
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_dense_plan_vjp

function fortml_cuda_dense_plan_destroy(handle) bind(C, &
        name="fortml_cuda_dense_plan_destroy") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle
    integer(c_int) :: value
    value = 0_c_int
end function fortml_cuda_dense_plan_destroy
