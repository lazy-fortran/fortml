function fortml_cuda_knn_available() bind(C, &
        name="fortml_cuda_knn_available") result(available)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none

    integer(c_int) :: available

    available = 0_c_int
end function fortml_cuda_knn_available

function fortml_cuda_knn_predict( &
        train_x, train_class, sample_weight, query_x, class_label, output, &
        n_train, n_features, n_query, n_classes, n_neighbors, &
        weighting_code) bind(C, name="fortml_cuda_knn_predict") result(status)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none

    type(c_ptr), value :: train_x, train_class, sample_weight, query_x
    type(c_ptr), value :: class_label, output
    integer(c_int), value :: n_train, n_features, n_query, n_classes
    integer(c_int), value :: n_neighbors, weighting_code
    integer(c_int) :: status

    status = 1_c_int
end function fortml_cuda_knn_predict

function fortml_cuda_knn_plan_create( &
        train_x, train_class, sample_weight, class_label, n_train, &
        n_features, n_classes, n_neighbors, weighting_code, device_index, &
        plan) bind(C, name="fortml_cuda_knn_plan_create") result(status)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_null_ptr
    implicit none

    type(c_ptr), value :: train_x, train_class, sample_weight, class_label
    integer(c_int), value :: n_train, n_features, n_classes, n_neighbors
    integer(c_int), value :: weighting_code, device_index
    type(c_ptr) :: plan
    integer(c_int) :: status

    plan = c_null_ptr
    status = 1_c_int
end function fortml_cuda_knn_plan_create

function fortml_cuda_knn_plan_destroy(plan) bind(C, &
        name="fortml_cuda_knn_plan_destroy") result(status)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none

    type(c_ptr), value :: plan
    integer(c_int) :: status

    status = 1_c_int
end function fortml_cuda_knn_plan_destroy

function fortml_cuda_knn_plan_predict( &
        plan, query_x, n_query, output) bind(C, &
        name="fortml_cuda_knn_plan_predict") result(status)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none

    type(c_ptr), value :: plan, query_x, output
    integer(c_int), value :: n_query
    integer(c_int) :: status

    status = 1_c_int
end function fortml_cuda_knn_plan_predict
