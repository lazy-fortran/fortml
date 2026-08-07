!> Deterministic tree primitives and a small gradient-boosting foundation.
module fortml_tree
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    !> A one-level regression tree selected by exhaustive squared-error splits.
    !>
    !> The stump is intentionally deterministic: features are visited in
    !> ascending order and thresholds are midpoints between distinct sorted
    !> observations.  This gives a stable, auditable primitive for the future
    !> CART/XGBoost implementation.  Tree predictions are piecewise constant;
    !> their input derivative is zero away from a split and undefined on it.
    type, public :: decision_stump_t
        private
        integer :: n_inputs = 0
        integer :: feature_index = 0
        integer :: min_samples_leaf = 1
        real(dp) :: threshold = 0.0_dp
        real(dp) :: left_value = 0.0_dp
        real(dp) :: right_value = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: fit => stump_fit
        procedure, public :: predict_matrix => stump_predict_matrix
        procedure, public :: predict_vector => stump_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: jvp => stump_jvp
        procedure, public :: input_count => stump_input_count
        procedure, public :: split_feature => stump_split_feature
        procedure, public :: split_threshold => stump_split_threshold
        procedure, public :: is_initialized => stump_is_initialized
    end type decision_stump_t

    !> Squared-loss gradient boosting using deterministic regression stumps.
    !>
    !> This is a deliberately small first implementation of the tree-family
    !> API.  It has no differentiable surrogate for split selection: fit is a
    !> discrete operation, while prediction JVPs are available only away from
    !> split boundaries.  Histogram/quantile proposals, subsampling, feature
    !> metadata, and GPU kernels belong to the full tree backend.
    type, public :: gradient_boosting_regressor_t
        private
        integer :: n_inputs = 0
        integer :: n_estimators = 0
        integer :: min_samples_leaf = 1
        real(dp) :: learning_rate = 0.1_dp
        real(dp) :: base_value = 0.0_dp
        type(decision_stump_t), allocatable :: estimators(:)
        logical :: initialized = .false.
    contains
        procedure, public :: fit => boosting_fit
        procedure, public :: predict_matrix => boosting_predict_matrix
        procedure, public :: predict_vector => boosting_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_jvp => boosting_predict_jvp
        procedure, public :: input_count => boosting_input_count
        procedure, public :: estimator_count => boosting_estimator_count
        procedure, public :: is_initialized => boosting_is_initialized
    end type gradient_boosting_regressor_t

    !> A deterministic, depth-limited CART regression tree.
    !>
    !> Splits are selected by exhaustive numeric threshold search with weighted
    !> squared error.  Features are visited in ascending order and thresholds
    !> in ascending sorted order; strict improvement therefore gives a stable
    !> tie rule.  The fit is finite-only and prediction is piecewise constant.
    type, public :: cart_regressor_t
        private
        integer :: n_inputs = 0
        integer :: max_depth = 0
        integer :: min_samples_leaf = 1
        integer :: n_nodes = 0
        integer, allocatable :: feature(:), left_child(:), right_child(:)
        real(dp), allocatable :: threshold(:), value(:)
        logical, allocatable :: leaf(:)
        logical :: initialized = .false.
    contains
        procedure, public :: fit => cart_fit
        procedure, public :: predict_matrix => cart_predict_matrix
        procedure, public :: predict_vector => cart_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_jvp => cart_predict_jvp
        procedure, public :: input_count => cart_input_count
        procedure, public :: node_count => cart_node_count
        procedure, public :: depth => cart_depth
        procedure, public :: is_initialized => cart_is_initialized
    end type cart_regressor_t

contains

    subroutine stump_fit(self, x, y, status, min_samples_leaf)
        class(decision_stump_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: min_samples_leaf
        integer, allocatable :: order(:)
        integer :: n_samples, n_features, j, i, k, leaf_size
        integer :: best_feature
        real(dp) :: best_threshold, best_left, best_right, best_sse
        real(dp) :: sum_left, sum_right, mean_left, mean_right, sse
        real(dp) :: candidate_threshold

        n_samples = size(x, 1)
        n_features = size(x, 2)
        leaf_size = 1
        if (present(min_samples_leaf)) leaf_size = min_samples_leaf
        if (n_samples < 2 .or. n_features < 1 .or. size(y) /= n_samples .or. &
            leaf_size < 1 .or. leaf_size*2 > n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "decision stump fit: invalid dimensions or leaf size")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "decision stump fit: inputs and targets must be finite")
            return
        end if

        allocate(order(n_samples))
        best_sse = huge(1.0_dp)
        best_feature = 1
        best_threshold = 0.0_dp
        best_left = sum(y)/real(n_samples, dp)
        best_right = best_left

        do j = 1, n_features
            call sort_feature_indices(x(:, j), order)
            do k = leaf_size, n_samples - leaf_size
                if (x(order(k), j) >= x(order(k + 1), j)) cycle
                candidate_threshold = 0.5_dp*(x(order(k), j) + &
                    x(order(k + 1), j))
                sum_left = 0.0_dp
                do i = 1, k
                    sum_left = sum_left + y(order(i))
                end do
                sum_right = sum(y) - sum_left
                mean_left = sum_left/real(k, dp)
                mean_right = sum_right/real(n_samples - k, dp)
                sse = 0.0_dp
                do i = 1, k
                    sse = sse + (y(order(i)) - mean_left)**2
                end do
                do i = k + 1, n_samples
                    sse = sse + (y(order(i)) - mean_right)**2
                end do

                if (sse < best_sse) then
                    best_sse = sse
                    best_feature = j
                    best_threshold = candidate_threshold
                    best_left = mean_left
                    best_right = mean_right
                end if
            end do
        end do

        self%n_inputs = n_features
        self%feature_index = best_feature
        self%min_samples_leaf = leaf_size
        self%threshold = best_threshold
        self%left_value = best_left
        self%right_value = best_right
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine stump_fit

    subroutine stump_predict_matrix(self, x, y, status)
        class(decision_stump_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:,:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(y) /= [size(x, 1), 1]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "decision stump predict: model, input, or array shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            if (x(i, self%feature_index) < self%threshold) then
                y(i, 1) = self%left_value
            else
                y(i, 1) = self%right_value
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine stump_predict_matrix

    subroutine stump_predict_vector(self, x, y, status)
        class(decision_stump_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: ym(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "decision stump predict: output shape is invalid")
            return
        end if
        allocate(ym(size(y), 1))
        call self%predict_matrix(x, ym, status)
        if (status%code == FORTNUM_OK) y = ym(:, 1)
    end subroutine stump_predict_vector

    !> Product through prediction with respect to x at differentiable points.
    subroutine stump_jvp(self, x, x_dot, y, y_dot, status)
        class(decision_stump_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: y(:), y_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y) /= size(x, 1) .or. &
            size(y_dot) /= size(y) .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "decision stump jvp: model, input, tangent, or shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            if (x(i, self%feature_index) == self%threshold) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "decision stump jvp: derivative is undefined on split")
                return
            end if
        end do
        call self%predict_vector(x, y, status)
        if (status%code /= FORTNUM_OK) return
        y_dot = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine stump_jvp

    integer function stump_input_count(self) result(count)
        class(decision_stump_t), intent(in) :: self
        count = self%n_inputs
    end function stump_input_count

    integer function stump_split_feature(self) result(feature)
        class(decision_stump_t), intent(in) :: self
        feature = self%feature_index
    end function stump_split_feature

    real(dp) function stump_split_threshold(self) result(threshold)
        class(decision_stump_t), intent(in) :: self
        threshold = self%threshold
    end function stump_split_threshold

    logical function stump_is_initialized(self) result(initialized)
        class(decision_stump_t), intent(in) :: self
        initialized = self%initialized
    end function stump_is_initialized

    subroutine boosting_fit(self, x, y, status, n_estimators, learning_rate, &
            min_samples_leaf)
        class(gradient_boosting_regressor_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_estimators, min_samples_leaf
        real(dp), intent(in), optional :: learning_rate
        integer :: n_samples, n_features, n_trees, leaf_size, i
        real(dp) :: rate
        real(dp), allocatable :: prediction(:), residual(:), correction(:)

        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_trees = 20
        if (present(n_estimators)) n_trees = n_estimators
        rate = 0.1_dp
        if (present(learning_rate)) rate = learning_rate
        leaf_size = 1
        if (present(min_samples_leaf)) leaf_size = min_samples_leaf
        if (n_samples < 2 .or. n_features < 1 .or. size(y) /= n_samples .or. &
            n_trees < 1 .or. rate <= 0.0_dp .or. rate > 1.0_dp .or. &
            leaf_size < 1 .or. leaf_size*2 > n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "gradient boosting fit: invalid dimensions or hyperparameters")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "gradient boosting fit: inputs and targets must be finite")
            return
        end if

        allocate(self%estimators(n_trees))
        allocate(prediction(n_samples), residual(n_samples), correction(n_samples))
        self%base_value = sum(y)/real(n_samples, dp)
        prediction = self%base_value
        do i = 1, n_trees
            residual = y - prediction
            call self%estimators(i)%fit(x, residual, status, leaf_size)
            if (status%code /= FORTNUM_OK) return
            call self%estimators(i)%predict(x, correction, status)
            if (status%code /= FORTNUM_OK) return
            prediction = prediction + rate*correction
        end do
        self%n_inputs = n_features
        self%n_estimators = n_trees
        self%min_samples_leaf = leaf_size
        self%learning_rate = rate
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosting_fit

    subroutine cart_fit(self, x, y, status, max_depth, min_samples_leaf, &
            sample_weight)
        class(cart_regressor_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: max_depth, min_samples_leaf
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: weights(:)
        integer, allocatable :: indices(:)
        integer :: tree_depth, leaf_size, max_nodes, i

        tree_depth = 3
        if (present(max_depth)) tree_depth = max_depth
        leaf_size = 1
        if (present(min_samples_leaf)) leaf_size = min_samples_leaf
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(y) /= size(x, 1) .or. &
            tree_depth < 0 .or. tree_depth > 12 .or. leaf_size < 1 .or. &
            leaf_size > size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cart regression fit: invalid dimensions or hyperparameters")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cart regression fit: inputs and targets must be finite")
            return
        end if
        allocate(weights(size(y)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(y) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "cart regression fit: sample weights must be finite and positive")
                return
            end if
            weights = sample_weight
        end if

        max_nodes = 2**(tree_depth + 1) - 1
        allocate(self%feature(max_nodes), self%left_child(max_nodes), &
            self%right_child(max_nodes), self%threshold(max_nodes), &
            self%value(max_nodes), self%leaf(max_nodes))
        self%feature = 0
        self%left_child = 0
        self%right_child = 0
        self%threshold = 0.0_dp
        self%value = 0.0_dp
        self%leaf = .true.
        allocate(indices(size(y)))
        do i = 1, size(y)
            indices(i) = i
        end do
        self%n_inputs = size(x, 2)
        self%max_depth = tree_depth
        self%min_samples_leaf = leaf_size
        self%n_nodes = 0
        self%initialized = .false.
        call cart_build_node(self, x, y, weights, indices, 0, 1, status)
        if (status%code /= FORTNUM_OK) return
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine cart_fit

    recursive subroutine cart_build_node(self, x, y, weights, indices, depth, &
            node, status)
        class(cart_regressor_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:), weights(:)
        integer, intent(in) :: indices(:), depth, node
        type(fortnum_status_t), intent(out) :: status
        integer :: best_feature, i, left_count, right_count
        integer, allocatable :: left_indices(:), right_indices(:)
        real(dp) :: best_threshold, best_sse, parent_sse, node_value
        logical :: found

        if (node > size(self%leaf)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cart regression fit: internal node capacity exceeded")
            return
        end if
        self%n_nodes = max(self%n_nodes, node)
        node_value = cart_weighted_mean(y, weights, indices)
        self%value(node) = node_value
        self%leaf(node) = .true.
        if (depth >= self%max_depth .or. size(indices) < 2*self%min_samples_leaf) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        parent_sse = cart_weighted_sse(y, weights, indices)
        call cart_find_best_split(x, y, weights, indices, self%min_samples_leaf, &
            best_feature, best_threshold, best_sse, found)
        if (.not. found .or. best_sse >= parent_sse) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        left_count = 0
        do i = 1, size(indices)
            if (x(indices(i), best_feature) < best_threshold) left_count = left_count + 1
        end do
        right_count = size(indices) - left_count
        if (left_count < self%min_samples_leaf .or. &
            right_count < self%min_samples_leaf) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cart regression fit: split partition violates leaf size")
            return
        end if
        allocate(left_indices(left_count), right_indices(right_count))
        left_count = 0
        right_count = 0
        do i = 1, size(indices)
            if (x(indices(i), best_feature) < best_threshold) then
                left_count = left_count + 1
                left_indices(left_count) = indices(i)
            else
                right_count = right_count + 1
                right_indices(right_count) = indices(i)
            end if
        end do
        self%leaf(node) = .false.
        self%feature(node) = best_feature
        self%threshold(node) = best_threshold
        self%left_child(node) = 2*node
        self%right_child(node) = 2*node + 1
        call cart_build_node(self, x, y, weights, left_indices, depth + 1, &
            2*node, status)
        if (status%code /= FORTNUM_OK) return
        call cart_build_node(self, x, y, weights, right_indices, depth + 1, &
            2*node + 1, status)
    end subroutine cart_build_node

    subroutine cart_find_best_split(x, y, weights, indices, min_leaf, &
            best_feature, best_threshold, best_sse, found)
        real(dp), intent(in) :: x(:, :), y(:), weights(:)
        integer, intent(in) :: indices(:), min_leaf
        integer, intent(out) :: best_feature
        real(dp), intent(out) :: best_threshold, best_sse
        logical, intent(out) :: found
        integer, allocatable :: order(:)
        integer :: j, k, i, left_count, n
        real(dp) :: left_weight, right_weight, left_sum, right_sum
        real(dp) :: left_mean, right_mean, candidate_sse, threshold

        n = size(indices)
        allocate(order(n))
        found = .false.
        best_feature = 0
        best_threshold = 0.0_dp
        best_sse = huge(1.0_dp)
        do j = 1, size(x, 2)
            call sort_subset_indices(x(:, j), indices, order)
            left_weight = 0.0_dp
            left_sum = 0.0_dp
            do k = 1, n - 1
                i = indices(order(k))
                left_weight = left_weight + weights(i)
                left_sum = left_sum + weights(i)*y(i)
                left_count = k
                if (left_count < min_leaf .or. n - left_count < min_leaf) cycle
                if (x(i, j) >= x(indices(order(k + 1)), j)) cycle
                right_weight = sum(weights(indices(order(k + 1:n))))
                right_sum = sum(weights(indices(order(k + 1:n))) * &
                    y(indices(order(k + 1:n))))
                left_mean = left_sum/left_weight
                right_mean = right_sum/right_weight
                candidate_sse = 0.0_dp
                do i = 1, k
                    candidate_sse = candidate_sse + weights(indices(order(i)))* &
                        (y(indices(order(i))) - left_mean)**2
                end do
                do i = k + 1, n
                    candidate_sse = candidate_sse + weights(indices(order(i)))* &
                        (y(indices(order(i))) - right_mean)**2
                end do
                threshold = 0.5_dp*x(indices(order(k)), j) + &
                    0.5_dp*x(indices(order(k + 1)), j)
                if (.not. found .or. candidate_sse < best_sse) then
                    found = .true.
                    best_feature = j
                    best_threshold = threshold
                    best_sse = candidate_sse
                end if
            end do
        end do
    end subroutine cart_find_best_split

    real(dp) function cart_weighted_mean(y, weights, indices) result(mean)
        real(dp), intent(in) :: y(:), weights(:)
        integer, intent(in) :: indices(:)
        real(dp) :: total_weight

        total_weight = sum(weights(indices))
        mean = sum(weights(indices)*y(indices))/total_weight
    end function cart_weighted_mean

    real(dp) function cart_weighted_sse(y, weights, indices) result(sse)
        real(dp), intent(in) :: y(:), weights(:)
        integer, intent(in) :: indices(:)
        real(dp) :: mean

        mean = cart_weighted_mean(y, weights, indices)
        sse = sum(weights(indices)*(y(indices) - mean)**2)
    end function cart_weighted_sse

    subroutine sort_subset_indices(values, subset, order)
        real(dp), intent(in) :: values(:)
        integer, intent(in) :: subset(:)
        integer, intent(out) :: order(:)
        integer :: i, j, key

        do i = 1, size(subset)
            order(i) = i
        end do
        do i = 2, size(subset)
            key = order(i)
            j = i - 1
            do while (j >= 1)
                if (values(subset(order(j))) <= values(subset(key))) exit
                order(j + 1) = order(j)
                j = j - 1
            end do
            order(j + 1) = key
        end do
    end subroutine sort_subset_indices

    subroutine cart_predict_matrix(self, x, y, status)
        class(cart_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, node

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(y) /= [size(x, 1), 1]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cart regression predict: model, input, or array shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            node = 1
            do while (.not. self%leaf(node))
                if (x(i, self%feature(node)) < self%threshold(node)) then
                    node = self%left_child(node)
                else
                    node = self%right_child(node)
                end if
            end do
            y(i, 1) = self%value(node)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine cart_predict_matrix

    subroutine cart_predict_vector(self, x, y, status)
        class(cart_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: ym(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cart regression predict: output shape is invalid")
            return
        end if
        allocate(ym(size(y), 1))
        call self%predict_matrix(x, ym, status)
        if (status%code == FORTNUM_OK) y = ym(:, 1)
    end subroutine cart_predict_vector

    subroutine cart_predict_jvp(self, x, x_dot, y, y_dot, status)
        class(cart_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: y(:), y_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, node

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y) /= size(x, 1) .or. &
            size(y_dot) /= size(y) .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cart regression jvp: model, input, tangent, or shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            node = 1
            do while (.not. self%leaf(node))
                if (x(i, self%feature(node)) == self%threshold(node)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "cart regression jvp: derivative is undefined on split")
                    return
                end if
                if (x(i, self%feature(node)) < self%threshold(node)) then
                    node = self%left_child(node)
                else
                    node = self%right_child(node)
                end if
            end do
            y(i) = self%value(node)
        end do
        y_dot = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine cart_predict_jvp

    integer function cart_input_count(self) result(count)
        class(cart_regressor_t), intent(in) :: self
        count = self%n_inputs
    end function cart_input_count

    integer function cart_node_count(self) result(count)
        class(cart_regressor_t), intent(in) :: self
        count = self%n_nodes
    end function cart_node_count

    integer function cart_depth(self) result(depth)
        class(cart_regressor_t), intent(in) :: self
        depth = self%max_depth
    end function cart_depth

    logical function cart_is_initialized(self) result(initialized)
        class(cart_regressor_t), intent(in) :: self
        initialized = self%initialized
    end function cart_is_initialized

    subroutine boosting_predict_matrix(self, x, y, status)
        class(gradient_boosting_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(y) /= [size(x, 1), 1]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "gradient boosting predict: model, input, or array shape is invalid")
            return
        end if
        allocate(correction(size(x, 1)))
        y(:, 1) = self%base_value
        do i = 1, self%n_estimators
            call self%estimators(i)%predict(x, correction, status)
            if (status%code /= FORTNUM_OK) return
            y(:, 1) = y(:, 1) + self%learning_rate*correction
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosting_predict_matrix

    subroutine boosting_predict_vector(self, x, y, status)
        class(gradient_boosting_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: ym(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "gradient boosting predict: output shape is invalid")
            return
        end if
        allocate(ym(size(y), 1))
        call self%predict_matrix(x, ym, status)
        if (status%code == FORTNUM_OK) y = ym(:, 1)
    end subroutine boosting_predict_vector

    subroutine boosting_predict_jvp(self, x, x_dot, y, y_dot, status)
        class(gradient_boosting_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: y(:), y_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:), correction_dot(:)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y) /= size(x, 1) .or. &
            size(y_dot) /= size(y) .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "gradient boosting jvp: model, input, tangent, or shape is invalid")
            return
        end if
        allocate(correction(size(x, 1)), correction_dot(size(x, 1)))
        y = self%base_value
        y_dot = 0.0_dp
        do i = 1, self%n_estimators
            call self%estimators(i)%jvp(x, x_dot, correction, correction_dot, &
                status)
            if (status%code /= FORTNUM_OK) return
            y = y + self%learning_rate*correction
            y_dot = y_dot + self%learning_rate*correction_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosting_predict_jvp

    integer function boosting_input_count(self) result(count)
        class(gradient_boosting_regressor_t), intent(in) :: self
        count = self%n_inputs
    end function boosting_input_count

    integer function boosting_estimator_count(self) result(count)
        class(gradient_boosting_regressor_t), intent(in) :: self
        count = self%n_estimators
    end function boosting_estimator_count

    logical function boosting_is_initialized(self) result(initialized)
        class(gradient_boosting_regressor_t), intent(in) :: self
        initialized = self%initialized
    end function boosting_is_initialized

    subroutine sort_feature_indices(values, order)
        real(dp), intent(in) :: values(:)
        integer, intent(out) :: order(:)
        integer :: i, j, key

        do i = 1, size(values)
            order(i) = i
        end do
        do i = 2, size(values)
            key = order(i)
            j = i - 1
            do while (j >= 1)
                if (values(order(j)) <= values(key)) exit
                order(j + 1) = order(j)
                j = j - 1
            end do
            order(j + 1) = key
        end do
    end subroutine sort_feature_indices

end module fortml_tree
