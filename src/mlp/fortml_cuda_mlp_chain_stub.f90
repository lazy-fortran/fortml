function fortml_cuda_mlp_chain_available() bind(C, &
        name="fortml_cuda_mlp_chain_available") result(value)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: value
    value = 0_c_int
end function fortml_cuda_mlp_chain_available

function fortml_cuda_mlp_chain_create(layer_sizes, activations, weights, biases, &
        n_layers, device_index, handle) bind(C, &
        name="fortml_cuda_mlp_chain_create") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_null_ptr
    implicit none
    type(c_ptr), value :: layer_sizes, activations, weights, biases
    integer(c_int), value :: n_layers, device_index
    type(c_ptr) :: handle
    integer(c_int) :: value
    handle = c_null_ptr
    value = 1_c_int
end function fortml_cuda_mlp_chain_create

function fortml_cuda_mlp_chain_predict(handle, query_x, n_query, outputs) &
        bind(C, name="fortml_cuda_mlp_chain_predict") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle, query_x, outputs
    integer(c_int), value :: n_query
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_mlp_chain_predict

function fortml_cuda_mlp_chain_jvp(handle, query_x, query_x_dot, weights_dot, &
        biases_dot, n_query, outputs, outputs_dot) bind(C, &
        name="fortml_cuda_mlp_chain_jvp") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle, query_x, query_x_dot, weights_dot, biases_dot
    integer(c_int), value :: n_query
    type(c_ptr), value :: outputs, outputs_dot
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_mlp_chain_jvp

function fortml_cuda_mlp_chain_vjp(handle, query_x, output_bar, n_query, &
        query_x_bar, weights_bar, biases_bar) bind(C, &
        name="fortml_cuda_mlp_chain_vjp") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle, query_x, output_bar
    integer(c_int), value :: n_query
    type(c_ptr), value :: query_x_bar, weights_bar, biases_bar
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_mlp_chain_vjp

function fortml_cuda_mlp_chain_transfer_stats(handle, host_to_device_bytes, &
        device_to_host_bytes, resident_bytes) bind(C, &
        name="fortml_cuda_mlp_chain_transfer_stats") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_ptr
    implicit none
    type(c_ptr), value :: handle
    integer(c_int64_t) :: host_to_device_bytes, device_to_host_bytes
    integer(c_int64_t) :: resident_bytes
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_mlp_chain_transfer_stats

function fortml_cuda_mlp_chain_destroy(handle) bind(C, &
        name="fortml_cuda_mlp_chain_destroy") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle
    integer(c_int) :: value
    value = 0_c_int
end function fortml_cuda_mlp_chain_destroy
