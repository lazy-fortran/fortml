module fortml_cuda_forest_api
    !! Fortran control-plane wrapper for the resident CUDA forest ABI.
    !!
    !! The wrapper accepts an explicit flattened model.  It does not flatten
    !! `cart_classifier_t` or `random_forest_classifier_t`, whose node arrays
    !! remain private; callers therefore cannot accidentally trigger a hidden
    !! host copy.  The ordinary Fortran build links a typed unavailable stub,
    !! while a CUDA application may link the native `.cu` implementation.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_loc, c_null_ptr, &
        c_ptr, c_associated
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    public :: fortml_cuda_forest_available
    public :: cuda_forest_plan_t

    type, public :: cuda_forest_plan_t
        private
        type(c_ptr) :: handle = c_null_ptr
        integer :: n_inputs = 0
        integer :: n_classes = 0
        integer :: n_trees = 0
        integer :: n_nodes = 0
        integer :: device_index = -1
    contains
        procedure, public :: create => cuda_forest_plan_create
        procedure, public :: predict_proba => cuda_forest_plan_predict_proba
        procedure, public :: predict => cuda_forest_plan_predict
        procedure, public :: destroy => cuda_forest_plan_destroy
        procedure, public :: fitted => cuda_forest_plan_fitted
        procedure, public :: feature_count => cuda_forest_plan_feature_count
        procedure, public :: class_count => cuda_forest_plan_class_count
        procedure, public :: tree_count => cuda_forest_plan_tree_count
        procedure, public :: node_count => cuda_forest_plan_node_count
        procedure, public :: device => cuda_forest_plan_device
    end type cuda_forest_plan_t

    interface
        function fortml_cuda_forest_available() bind(C, &
                name="fortml_cuda_forest_available") result(value)
            import :: c_int
            integer(c_int) :: value
        end function fortml_cuda_forest_available

        function fortml_cuda_forest_plan_create( &
                tree_offset, node_feature, node_left, node_right, &
                node_threshold, node_probability, class_label, n_trees, &
                n_nodes, n_inputs, n_classes, device_index, handle) bind(C, &
                name="fortml_cuda_forest_plan_create") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: tree_offset, node_feature, node_left, node_right
            type(c_ptr), value :: node_threshold, node_probability, class_label
            integer(c_int), value :: n_trees, n_nodes, n_inputs, n_classes
            integer(c_int), value :: device_index
            type(c_ptr) :: handle
            integer(c_int) :: value
        end function fortml_cuda_forest_plan_create

        function fortml_cuda_forest_plan_predict_proba( &
                handle, query_x, n_query, probabilities) bind(C, &
                name="fortml_cuda_forest_plan_predict_proba") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: handle, query_x, probabilities
            integer(c_int), value :: n_query
            integer(c_int) :: value
        end function fortml_cuda_forest_plan_predict_proba

        function fortml_cuda_forest_plan_predict( &
                handle, query_x, n_query, labels) bind(C, &
                name="fortml_cuda_forest_plan_predict") result(value)
            import :: c_int, c_ptr
            type(c_ptr), value :: handle, query_x, labels
            integer(c_int), value :: n_query
            integer(c_int) :: value
        end function fortml_cuda_forest_plan_predict

        function fortml_cuda_forest_plan_destroy(handle) bind(C, &
                name="fortml_cuda_forest_plan_destroy") result(value)
            import :: c_int, c_ptr
            type(c_ptr), value :: handle
            integer(c_int) :: value
        end function fortml_cuda_forest_plan_destroy
    end interface

contains

    subroutine cuda_forest_plan_create(self, tree_offset, node_feature, &
            node_left, node_right, node_threshold, node_probability, &
            class_label, device_index, status)
        class(cuda_forest_plan_t), intent(out) :: self
        integer, intent(in), target, contiguous :: tree_offset(:), node_feature(:), &
            node_left(:), node_right(:), class_label(:)
        real(dp), intent(in), target, contiguous :: node_threshold(:), &
            node_probability(:)
        integer, intent(in) :: device_index
        type(fortnum_status_t), intent(out) :: status
        integer(c_int), allocatable, target :: tree_offset_c(:), node_feature_c(:), &
            node_left_c(:), node_right_c(:), class_label_c(:)
        real(c_double), allocatable, target :: node_threshold_c(:), &
            node_probability_c(:)
        type(c_ptr) :: handle
        integer(c_int) :: code
        integer :: n_trees, n_nodes, n_inputs, n_classes

        self%handle = c_null_ptr
        self%n_inputs = 0
        self%n_classes = 0
        self%n_trees = 0
        self%n_nodes = 0
        self%device_index = -1
        n_trees = size(tree_offset) - 1
        n_nodes = size(node_feature)
        n_inputs = 0
        if (size(node_feature) > 0) n_inputs = max(1, maxval(node_feature) + 1)
        n_classes = size(class_label)
        if (n_trees < 1 .or. n_nodes < 1 .or. n_inputs < 1 .or. &
                n_classes < 1 .or. size(node_left) /= n_nodes .or. &
                size(node_right) /= n_nodes .or. size(node_threshold) /= n_nodes .or. &
                size(node_probability) /= n_nodes*n_classes .or. &
                device_index < 0 .or. any(.not. ieee_is_finite(node_threshold)) .or. &
                any(.not. ieee_is_finite(node_probability))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA forest wrapper: flattened model shape or values are invalid")
            return
        end if
        allocate(tree_offset_c(size(tree_offset)), node_feature_c(n_nodes), &
            node_left_c(n_nodes), node_right_c(n_nodes), class_label_c(n_classes), &
            node_threshold_c(n_nodes), node_probability_c(n_nodes*n_classes))
        tree_offset_c = int(tree_offset, c_int)
        node_feature_c = int(node_feature, c_int)
        node_left_c = int(node_left, c_int)
        node_right_c = int(node_right, c_int)
        class_label_c = int(class_label, c_int)
        node_threshold_c = node_threshold
        node_probability_c = node_probability
        handle = c_null_ptr
        code = fortml_cuda_forest_plan_create(c_loc(tree_offset_c), &
            c_loc(node_feature_c), c_loc(node_left_c), c_loc(node_right_c), &
            c_loc(node_threshold_c), c_loc(node_probability_c), &
            c_loc(class_label_c), int(n_trees, c_int), int(n_nodes, c_int), &
            int(n_inputs, c_int), int(n_classes, c_int), &
            int(device_index, c_int), handle)
        if (code /= 0_c_int .or. .not. c_associated(handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA forest wrapper: resident native plan is unavailable")
            return
        end if
        self%handle = handle
        self%n_inputs = n_inputs
        self%n_classes = n_classes
        self%n_trees = n_trees
        self%n_nodes = n_nodes
        self%device_index = device_index
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_forest_plan_create

    subroutine cuda_forest_plan_predict_proba(self, query_x, probabilities, status)
        class(cuda_forest_plan_t), intent(in) :: self
        real(dp), intent(in), target, contiguous :: query_x(:, :)
        real(dp), intent(inout), target, contiguous :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code

        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA forest wrapper: plan is not fitted")
            return
        end if
        if (size(query_x, 2) /= self%n_inputs .or. &
                any(shape(probabilities) /= [size(query_x, 1), self%n_classes]) .or. &
                size(query_x, 1) < 1 .or. any(.not. ieee_is_finite(query_x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA forest wrapper: query or output shape is invalid")
            return
        end if
        code = fortml_cuda_forest_plan_predict_proba(self%handle, c_loc(query_x), &
            int(size(query_x, 1), c_int), c_loc(probabilities))
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA forest wrapper: resident probability prediction failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_forest_plan_predict_proba

    subroutine cuda_forest_plan_predict(self, query_x, labels, status)
        class(cuda_forest_plan_t), intent(in) :: self
        real(dp), intent(in), target, contiguous :: query_x(:, :)
        integer, intent(inout), target, contiguous :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer(c_int), allocatable, target :: labels_c(:)
        integer(c_int) :: code

        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA forest wrapper: plan is not fitted")
            return
        end if
        if (size(query_x, 2) /= self%n_inputs .or. size(labels) /= size(query_x, 1) .or. &
                size(query_x, 1) < 1 .or. any(.not. ieee_is_finite(query_x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA forest wrapper: query or label shape is invalid")
            return
        end if
        allocate(labels_c(size(labels)))
        code = fortml_cuda_forest_plan_predict(self%handle, c_loc(query_x), &
            int(size(query_x, 1), c_int), c_loc(labels_c))
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA forest wrapper: resident label prediction failed")
            return
        end if
        labels = int(labels_c)
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_forest_plan_predict

    subroutine cuda_forest_plan_destroy(self, status)
        class(cuda_forest_plan_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code

        if (c_associated(self%handle)) then
            code = fortml_cuda_forest_plan_destroy(self%handle)
            if (code /= 0_c_int) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "CUDA forest wrapper: resident plan destruction failed")
                return
            end if
        end if
        self%handle = c_null_ptr
        self%n_inputs = 0
        self%n_classes = 0
        self%n_trees = 0
        self%n_nodes = 0
        self%device_index = -1
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_forest_plan_destroy

    logical function cuda_forest_plan_fitted(self) result(value)
        class(cuda_forest_plan_t), intent(in) :: self
        value = c_associated(self%handle)
    end function cuda_forest_plan_fitted

    integer function cuda_forest_plan_feature_count(self) result(value)
        class(cuda_forest_plan_t), intent(in) :: self
        value = self%n_inputs
    end function cuda_forest_plan_feature_count

    integer function cuda_forest_plan_class_count(self) result(value)
        class(cuda_forest_plan_t), intent(in) :: self
        value = self%n_classes
    end function cuda_forest_plan_class_count

    integer function cuda_forest_plan_tree_count(self) result(value)
        class(cuda_forest_plan_t), intent(in) :: self
        value = self%n_trees
    end function cuda_forest_plan_tree_count

    integer function cuda_forest_plan_node_count(self) result(value)
        class(cuda_forest_plan_t), intent(in) :: self
        value = self%n_nodes
    end function cuda_forest_plan_node_count

    integer function cuda_forest_plan_device(self) result(value)
        class(cuda_forest_plan_t), intent(in) :: self
        value = self%device_index
    end function cuda_forest_plan_device

end module fortml_cuda_forest_api
