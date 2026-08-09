!> Fortran control plane for a resident fixed-topology additive-tree CUDA ABI.
module fortml_cuda_boosted_tree_api
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_loc, c_null_ptr, &
        c_ptr, c_associated
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    public :: fortml_cuda_boosted_tree_available
    public :: cuda_boosted_tree_plan_t

    interface
        function fortml_cuda_boosted_tree_available() bind(C, &
                name="fortml_cuda_boosted_tree_available") result(value)
            import :: c_int
            integer(c_int) :: value
        end function fortml_cuda_boosted_tree_available

        function fortml_cuda_boosted_tree_plan_create( &
                tree_offset, node_feature, node_left, node_right, &
                node_threshold, node_weight, node_missing_left, tree_scale, &
                n_trees, n_nodes, n_inputs, base_score, learning_rate, &
                device_index, opaque_plan) bind(C, &
                name="fortml_cuda_boosted_tree_plan_create") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: tree_offset, node_feature, node_left, node_right
            type(c_ptr), value :: node_threshold, node_weight, node_missing_left
            type(c_ptr), value :: tree_scale
            integer(c_int), value :: n_trees, n_nodes, n_inputs, device_index
            real(c_double), value :: base_score, learning_rate
            type(c_ptr) :: opaque_plan
            integer(c_int) :: value
        end function fortml_cuda_boosted_tree_plan_create

        function fortml_cuda_boosted_tree_plan_predict( &
                opaque_plan, query_x, n_query, margin) bind(C, &
                name="fortml_cuda_boosted_tree_plan_predict") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: opaque_plan, query_x, margin
            integer(c_int), value :: n_query
            integer(c_int) :: value
        end function fortml_cuda_boosted_tree_plan_predict

        function fortml_cuda_boosted_tree_plan_predict_jvp( &
                opaque_plan, query_x, query_x_dot, n_query, margin, &
                margin_dot) bind(C, &
                name="fortml_cuda_boosted_tree_plan_predict_jvp") result(value)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: opaque_plan, query_x, query_x_dot
            integer(c_int), value :: n_query
            type(c_ptr), value :: margin, margin_dot
            integer(c_int) :: value
        end function fortml_cuda_boosted_tree_plan_predict_jvp

        function fortml_cuda_boosted_tree_plan_destroy(opaque_plan) bind(C, &
                name="fortml_cuda_boosted_tree_plan_destroy") result(value)
            import :: c_int, c_ptr
            type(c_ptr), value :: opaque_plan
            integer(c_int) :: value
        end function fortml_cuda_boosted_tree_plan_destroy
    end interface

    type, public :: cuda_boosted_tree_plan_t
        private
        type(c_ptr) :: handle = c_null_ptr
        integer :: n_inputs = 0
        integer :: n_trees = 0
        integer :: n_nodes = 0
        integer :: device_index = -1
    contains
        procedure, public :: create => cuda_boosted_tree_plan_create
        procedure, public :: predict => cuda_boosted_tree_plan_predict
        procedure, public :: predict_jvp => cuda_boosted_tree_plan_predict_jvp
        procedure, public :: destroy => cuda_boosted_tree_plan_destroy
        procedure, public :: fitted => cuda_boosted_tree_plan_fitted
        procedure, public :: feature_count => cuda_boosted_tree_plan_feature_count
        procedure, public :: tree_count => cuda_boosted_tree_plan_tree_count
        procedure, public :: node_count => cuda_boosted_tree_plan_node_count
        procedure, public :: device => cuda_boosted_tree_plan_device
    end type cuda_boosted_tree_plan_t

contains

    subroutine cuda_boosted_tree_plan_create(self, tree_offset, node_feature, &
            node_left, node_right, node_threshold, node_weight, &
            node_missing_left, tree_scale, base_score, learning_rate, &
            device_index, status, n_inputs)
        class(cuda_boosted_tree_plan_t), intent(out) :: self
        integer, intent(in), target, contiguous :: tree_offset(:), node_feature(:), &
            node_left(:), node_right(:), node_missing_left(:)
        real(dp), intent(in), target, contiguous :: node_threshold(:), &
            node_weight(:), tree_scale(:)
        real(dp), intent(in) :: base_score, learning_rate
        integer, intent(in) :: device_index
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_inputs
        integer(c_int), allocatable, target :: tree_offset_c(:), node_feature_c(:), &
            node_left_c(:), node_right_c(:), node_missing_left_c(:)
        real(c_double), allocatable, target :: node_threshold_c(:), node_weight_c(:), &
            tree_scale_c(:)
        type(c_ptr) :: handle
        integer(c_int) :: code
        integer :: n_trees, n_nodes, input_count, i

        self%handle = c_null_ptr
        self%n_inputs = 0
        self%n_trees = 0
        self%n_nodes = 0
        self%device_index = -1
        n_trees = size(tree_offset) - 1
        n_nodes = size(node_feature)
        input_count = 0
        if (present(n_inputs)) input_count = n_inputs
        if (n_trees < 1 .or. n_nodes < n_trees .or. input_count < 1 .or. &
                size(tree_offset) /= n_trees + 1 .or. tree_offset(1) /= 0 .or. &
                tree_offset(n_trees + 1) /= n_nodes .or. &
                size(node_left) /= n_nodes .or. size(node_right) /= n_nodes .or. &
                size(node_threshold) /= n_nodes .or. size(node_weight) /= n_nodes .or. &
                size(node_missing_left) /= n_nodes .or. size(tree_scale) /= n_trees .or. &
                device_index < 0 .or. .not. ieee_is_finite(base_score) .or. &
                .not. ieee_is_finite(learning_rate) .or. learning_rate <= 0.0_dp .or. &
                any(.not. ieee_is_finite(node_threshold)) .or. &
                any(.not. ieee_is_finite(node_weight)) .or. &
                any(.not. ieee_is_finite(tree_scale)) .or. &
                any(node_missing_left < 0) .or. any(node_missing_left > 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA boosted tree wrapper: model shape or values are invalid")
            return
        end if
        do i = 1, n_trees
            if (tree_offset(i) < 0 .or. tree_offset(i + 1) <= tree_offset(i) .or. &
                    tree_offset(i + 1) > n_nodes) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CUDA boosted tree wrapper: tree offsets are invalid")
                return
            end if
        end do
        if (any(node_feature < -1) .or. any(node_feature >= input_count)) then
            ! Leaf nodes use -1; split features are zero based.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA boosted tree wrapper: node feature index is invalid")
            return
        end if

        allocate(tree_offset_c(size(tree_offset)), node_feature_c(n_nodes), &
            node_left_c(n_nodes), node_right_c(n_nodes), node_missing_left_c(n_nodes), &
            node_threshold_c(n_nodes), node_weight_c(n_nodes), tree_scale_c(n_trees))
        tree_offset_c = int(tree_offset, c_int)
        node_feature_c = int(node_feature, c_int)
        node_left_c = int(node_left, c_int)
        node_right_c = int(node_right, c_int)
        node_missing_left_c = int(node_missing_left, c_int)
        node_threshold_c = node_threshold
        node_weight_c = node_weight
        tree_scale_c = tree_scale
        handle = c_null_ptr
        code = fortml_cuda_boosted_tree_plan_create(c_loc(tree_offset_c), &
            c_loc(node_feature_c), c_loc(node_left_c), c_loc(node_right_c), &
            c_loc(node_threshold_c), c_loc(node_weight_c), &
            c_loc(node_missing_left_c), c_loc(tree_scale_c), int(n_trees, c_int), &
            int(n_nodes, c_int), int(input_count, c_int), real(base_score, c_double), &
            real(learning_rate, c_double), int(device_index, c_int), handle)
        if (code /= 0_c_int .or. .not. c_associated(handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA boosted tree wrapper: resident native plan is unavailable")
            return
        end if
        self%handle = handle
        self%n_inputs = input_count
        self%n_trees = n_trees
        self%n_nodes = n_nodes
        self%device_index = device_index
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_boosted_tree_plan_create

    subroutine cuda_boosted_tree_plan_predict(self, query_x, margin, status)
        class(cuda_boosted_tree_plan_t), intent(in) :: self
        real(dp), intent(in), target, contiguous :: query_x(:, :)
        real(dp), intent(inout), target, contiguous :: margin(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable, target :: candidate(:)
        integer(c_int) :: code

        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA boosted tree wrapper: plan is not fitted")
            return
        end if
        if (size(query_x, 2) /= self%n_inputs .or. size(margin) /= size(query_x, 1) .or. &
                size(query_x, 1) < 1 .or. &
                any(.not. ieee_is_nan(query_x) .and. .not. ieee_is_finite(query_x))) then
            ! NaNs are valid missing values; infinities are not.
            if (size(query_x, 2) == self%n_inputs .and. &
                    size(margin) == size(query_x, 1) .and. size(query_x, 1) >= 1 .and. &
                    any(.not. ieee_is_nan(query_x) .and. .not. ieee_is_finite(query_x))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CUDA boosted tree wrapper: query contains infinity")
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CUDA boosted tree wrapper: query or output shape is invalid")
            end if
            return
        end if
        allocate(candidate(size(margin)))
        code = fortml_cuda_boosted_tree_plan_predict(self%handle, c_loc(query_x), &
            int(size(query_x, 1), c_int), c_loc(candidate))
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA boosted tree wrapper: resident prediction failed")
            return
        end if
        margin = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_boosted_tree_plan_predict

    subroutine cuda_boosted_tree_plan_predict_jvp(self, query_x, query_x_dot, &
            margin, margin_dot, status)
        class(cuda_boosted_tree_plan_t), intent(in) :: self
        real(dp), intent(in), target, contiguous :: query_x(:, :), query_x_dot(:, :)
        real(dp), intent(inout), target, contiguous :: margin(:), margin_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable, target :: candidate(:), candidate_dot(:)
        integer(c_int) :: code

        if (.not. c_associated(self%handle)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA boosted tree wrapper: plan is not fitted")
            return
        end if
        if (size(query_x, 2) /= self%n_inputs .or. any(shape(query_x_dot) /= shape(query_x)) .or. &
                size(margin) /= size(query_x, 1) .or. size(margin_dot) /= size(margin) .or. &
                size(query_x, 1) < 1 .or. any(.not. ieee_is_finite(query_x)) .or. &
                any(.not. ieee_is_finite(query_x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA boosted tree wrapper: JVP query or output shape is invalid")
            return
        end if
        allocate(candidate(size(margin)), candidate_dot(size(margin_dot)))
        code = fortml_cuda_boosted_tree_plan_predict_jvp(self%handle, c_loc(query_x), &
            c_loc(query_x_dot), int(size(query_x, 1), c_int), c_loc(candidate), &
            c_loc(candidate_dot))
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA boosted tree wrapper: JVP is undefined on a split boundary")
            return
        end if
        margin = candidate
        margin_dot = candidate_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_boosted_tree_plan_predict_jvp

    subroutine cuda_boosted_tree_plan_destroy(self, status)
        class(cuda_boosted_tree_plan_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer(c_int) :: code

        if (c_associated(self%handle)) then
            code = fortml_cuda_boosted_tree_plan_destroy(self%handle)
            if (code /= 0_c_int) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "CUDA boosted tree wrapper: resident plan destruction failed")
                return
            end if
        end if
        self%handle = c_null_ptr
        self%n_inputs = 0
        self%n_trees = 0
        self%n_nodes = 0
        self%device_index = -1
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_boosted_tree_plan_destroy

    logical function cuda_boosted_tree_plan_fitted(self) result(value)
        class(cuda_boosted_tree_plan_t), intent(in) :: self
        value = c_associated(self%handle)
    end function cuda_boosted_tree_plan_fitted

    integer function cuda_boosted_tree_plan_feature_count(self) result(value)
        class(cuda_boosted_tree_plan_t), intent(in) :: self
        value = self%n_inputs
    end function cuda_boosted_tree_plan_feature_count

    integer function cuda_boosted_tree_plan_tree_count(self) result(value)
        class(cuda_boosted_tree_plan_t), intent(in) :: self
        value = self%n_trees
    end function cuda_boosted_tree_plan_tree_count

    integer function cuda_boosted_tree_plan_node_count(self) result(value)
        class(cuda_boosted_tree_plan_t), intent(in) :: self
        value = self%n_nodes
    end function cuda_boosted_tree_plan_node_count

    integer function cuda_boosted_tree_plan_device(self) result(value)
        class(cuda_boosted_tree_plan_t), intent(in) :: self
        value = self%device_index
    end function cuda_boosted_tree_plan_device

end module fortml_cuda_boosted_tree_api
