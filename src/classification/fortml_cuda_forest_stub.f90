function fortml_cuda_forest_available() bind(C, &
        name="fortml_cuda_forest_available") result(value)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: value
    value = 0_c_int
end function fortml_cuda_forest_available

function fortml_cuda_forest_plan_create( &
        tree_offset, node_feature, node_left, node_right, node_threshold, &
        node_probability, class_label, n_trees, n_nodes, n_inputs, n_classes, &
        device_index, handle) bind(C, &
        name="fortml_cuda_forest_plan_create") result(value)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr, c_null_ptr
    implicit none
    type(c_ptr), value :: tree_offset, node_feature, node_left, node_right
    type(c_ptr), value :: node_threshold, node_probability, class_label
    integer(c_int), value :: n_trees, n_nodes, n_inputs, n_classes, device_index
    type(c_ptr) :: handle
    integer(c_int) :: value
    handle = c_null_ptr
    value = 1_c_int
end function fortml_cuda_forest_plan_create

function fortml_cuda_forest_plan_predict_proba( &
        handle, query_x, n_query, probabilities) bind(C, &
        name="fortml_cuda_forest_plan_predict_proba") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle, query_x, probabilities
    integer(c_int), value :: n_query
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_forest_plan_predict_proba

function fortml_cuda_forest_plan_predict(handle, query_x, n_query, labels) bind(C, &
        name="fortml_cuda_forest_plan_predict") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle, query_x, labels
    integer(c_int), value :: n_query
    integer(c_int) :: value
    value = 1_c_int
end function fortml_cuda_forest_plan_predict

function fortml_cuda_forest_plan_destroy(handle) bind(C, &
        name="fortml_cuda_forest_plan_destroy") result(value)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: handle
    integer(c_int) :: value
    value = 0_c_int
end function fortml_cuda_forest_plan_destroy
