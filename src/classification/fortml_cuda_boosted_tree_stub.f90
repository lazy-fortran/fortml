function fortml_cuda_boosted_tree_available() bind(C, &
        name="fortml_cuda_boosted_tree_available") result(value)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: value
    value = 0_c_int
end function fortml_cuda_boosted_tree_available

function fortml_cuda_boosted_tree_plan_create( &
        tree_offset, node_feature, node_left, node_right, node_threshold, &
        node_weight, node_missing_left, tree_scale, n_trees, n_nodes, n_inputs, &
        base_score, learning_rate, device_index, opaque_plan) bind(C, &
        name="fortml_cuda_boosted_tree_plan_create") result(value)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr, c_null_ptr
    implicit none
    type(c_ptr), value :: tree_offset, node_feature, node_left, node_right
    type(c_ptr), value :: node_threshold, node_weight, node_missing_left
    type(c_ptr), value :: tree_scale
    integer(c_int), value :: n_trees, n_nodes, n_inputs, device_index
    real(c_double), value :: base_score, learning_rate
    type(c_ptr) :: opaque_plan
    integer(c_int) :: value
    opaque_plan = c_null_ptr
    value = 1_c_int
end function fortml_cuda_boosted_tree_plan_create

function fortml_cuda_boosted_tree_plan_predict( &
        opaque_plan, query_x, n_query, margin) bind(C, &
        name="fortml_cuda_boosted_tree_plan_predict") result(value)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    implicit none
    type(c_ptr), value :: opaque_plan, query_x, margin
    integer(c_int), value :: n_query
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_boosted_tree_plan_predict

function fortml_cuda_boosted_tree_plan_predict_jvp( &
        opaque_plan, query_x, query_x_dot, n_query, margin, margin_dot) bind(C, &
        name="fortml_cuda_boosted_tree_plan_predict_jvp") result(value)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    implicit none
    type(c_ptr), value :: opaque_plan, query_x, query_x_dot, margin, margin_dot
    integer(c_int), value :: n_query
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_boosted_tree_plan_predict_jvp

function fortml_cuda_boosted_tree_plan_transfer_stats(opaque_plan, &
        host_to_device_bytes, device_to_host_bytes, resident_bytes) bind(C, &
        name="fortml_cuda_boosted_tree_plan_transfer_stats") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_ptr
    implicit none
    type(c_ptr), value :: opaque_plan
    integer(c_int64_t) :: host_to_device_bytes, device_to_host_bytes
    integer(c_int64_t) :: resident_bytes
    integer(c_int) :: value
    host_to_device_bytes = 0_c_int64_t
    device_to_host_bytes = 0_c_int64_t
    resident_bytes = 0_c_int64_t
    value = 1_c_int
end function fortml_cuda_boosted_tree_plan_transfer_stats

function fortml_cuda_boosted_tree_plan_destroy(opaque_plan) bind(C, &
        name="fortml_cuda_boosted_tree_plan_destroy") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: opaque_plan
    integer(c_int) :: value
    value = 0_c_int
end function fortml_cuda_boosted_tree_plan_destroy
