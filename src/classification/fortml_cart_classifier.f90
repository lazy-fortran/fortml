!> Deterministic finite-only CART classification.
module fortml_cart_classifier
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: CART_CRITERION_GINI = 1
    integer, parameter, public :: CART_CRITERION_ENTROPY = 2

    !> A deterministic numeric CART classifier with weighted impurity splits.
    !>
    !> Features are visited in ascending order and each feature's distinct
    !> sorted midpoint is considered in ascending order.  A split is accepted
    !> only on strict impurity improvement, so ties retain the first feature
    !> and threshold. Leaves store weighted class frequencies as empirical
    !> piecewise-constant probabilities for this tree.
    !> The implementation is dense and finite-only; missing-value routing and
    !> input derivatives are intentionally outside this contract.
    type, public :: cart_classifier_t
        private
        integer :: n_inputs = 0
        integer :: n_classes = 0
        integer :: max_depth = 0
        integer :: min_samples_leaf = 1
        integer :: criterion_code = CART_CRITERION_GINI
        integer :: n_nodes = 0
        integer, allocatable :: class_label(:)
        integer, allocatable :: feature(:), left_child(:), right_child(:)
        real(dp), allocatable :: threshold(:), probability(:, :)
        logical, allocatable :: leaf(:)
        logical :: initialized = .false.
    contains
        procedure, public :: fit => cart_classifier_fit
        procedure, public :: predict_proba => cart_classifier_predict_proba
        procedure, public :: predict => cart_classifier_predict
        procedure, public :: classes => cart_classifier_classes
        procedure, public :: feature_count => cart_classifier_feature_count
        procedure, public :: class_count => cart_classifier_class_count
        procedure, public :: node_count => cart_classifier_node_count
        procedure, public :: depth => cart_classifier_depth
        procedure, public :: criterion => cart_classifier_criterion
        procedure, public :: fitted => cart_classifier_fitted
    end type cart_classifier_t

contains

    subroutine cart_classifier_fit(self, x, labels, status, max_depth, &
            min_samples_leaf, sample_weight, criterion)
        class(cart_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: max_depth, min_samples_leaf, criterion
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: indices(:), classes(:)
        real(dp), allocatable :: weights(:)
        integer :: tree_depth, leaf_size, criterion_value, max_nodes, i

        tree_depth = 3
        if (present(max_depth)) tree_depth = max_depth
        leaf_size = 1
        if (present(min_samples_leaf)) leaf_size = min_samples_leaf
        criterion_value = CART_CRITERION_GINI
        if (present(criterion)) criterion_value = criterion
        if (size(x, 1) < 2 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1) .or. tree_depth < 0 .or. &
            tree_depth > 12 .or. leaf_size < 1 .or. &
            leaf_size > size(x, 1) .or. (criterion_value /= CART_CRITERION_GINI &
            .and. criterion_value /= CART_CRITERION_ENTROPY)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier fit: invalid dimensions or hyperparameters")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier fit: inputs must be finite")
            return
        end if
        call unique_sorted_labels(labels, classes)
        if (size(classes) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier fit: at least two classes are required")
            return
        end if

        allocate(weights(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CART classifier fit: sample weights must be finite and positive")
                return
            end if
            weights = sample_weight
        end if

        max_nodes = 2**(tree_depth + 1) - 1
        allocate(self%class_label(size(classes)), self%feature(max_nodes), &
            self%left_child(max_nodes), self%right_child(max_nodes), &
            self%threshold(max_nodes), self%probability(max_nodes, size(classes)), &
            self%leaf(max_nodes))
        self%class_label = classes
        self%feature = 0
        self%left_child = 0
        self%right_child = 0
        self%threshold = 0.0_dp
        self%probability = 0.0_dp
        self%leaf = .true.
        allocate(indices(size(labels)))
        do i = 1, size(labels)
            indices(i) = i
        end do

        self%n_inputs = size(x, 2)
        self%n_classes = size(classes)
        self%max_depth = tree_depth
        self%min_samples_leaf = leaf_size
        self%criterion_code = criterion_value
        self%n_nodes = 0
        self%initialized = .false.
        call cart_classifier_build_node(self, x, labels, weights, indices, 0, 1, &
            status)
        if (status%code /= FORTNUM_OK) return
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine cart_classifier_fit

    recursive subroutine cart_classifier_build_node(self, x, labels, weights, &
            indices, depth, node, status)
        class(cart_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), weights(:)
        integer, intent(in) :: labels(:), indices(:), depth, node
        type(fortnum_status_t), intent(out) :: status
        integer :: best_feature, i, left_count, right_count
        integer, allocatable :: left_indices(:), right_indices(:)
        real(dp) :: best_threshold, best_impurity, parent_impurity
        logical :: found

        if (node > size(self%leaf)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier fit: internal node capacity exceeded")
            return
        end if
        self%n_nodes = max(self%n_nodes, node)
        call cart_classifier_leaf_probability(labels, weights, indices, &
            self%class_label, self%probability(node, :))
        self%leaf(node) = .true.
        if (depth >= self%max_depth .or. size(indices) < 2*self%min_samples_leaf) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        parent_impurity = cart_classifier_weighted_impurity(labels, weights, &
            indices, self%class_label, self%criterion_code)
        call cart_classifier_find_best_split(x, labels, weights, indices, &
            self%class_label, self%criterion_code, self%min_samples_leaf, &
            best_feature, &
            best_threshold, best_impurity, found)
        if (.not. found .or. best_impurity >= parent_impurity) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        left_count = 0
        do i = 1, size(indices)
            if (x(indices(i), best_feature) < best_threshold) then
                left_count = left_count + 1
            end if
        end do
        right_count = size(indices) - left_count
        if (left_count < self%min_samples_leaf .or. &
            right_count < self%min_samples_leaf) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier fit: split partition violates leaf size")
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
        call cart_classifier_build_node(self, x, labels, weights, left_indices, &
            depth + 1, 2*node, status)
        if (status%code /= FORTNUM_OK) return
        call cart_classifier_build_node(self, x, labels, weights, right_indices, &
            depth + 1, 2*node + 1, status)
    end subroutine cart_classifier_build_node

    subroutine cart_classifier_find_best_split(x, labels, weights, indices, &
            classes, criterion, min_leaf, best_feature, best_threshold, &
            best_impurity, found)
        real(dp), intent(in) :: x(:, :), weights(:)
        integer, intent(in) :: labels(:), indices(:), classes(:), criterion, min_leaf
        integer, intent(out) :: best_feature
        real(dp), intent(out) :: best_threshold, best_impurity
        logical, intent(out) :: found
        integer, allocatable :: order(:), left_indices(:), right_indices(:)
        integer :: j, k, n, left_count, right_count, i
        real(dp) :: threshold, candidate_impurity

        n = size(indices)
        allocate(order(n), left_indices(n), right_indices(n))
        found = .false.
        best_feature = 0
        best_threshold = 0.0_dp
        best_impurity = huge(1.0_dp)
        do j = 1, size(x, 2)
            call sort_subset_indices(x(:, j), indices, order)
            do k = min_leaf, n - min_leaf
                if (x(indices(order(k)), j) >= &
                    x(indices(order(k + 1)), j)) cycle
                threshold = 0.5_dp*x(indices(order(k)), j) + &
                    0.5_dp*x(indices(order(k + 1)), j)
                left_count = 0
                right_count = 0
                do i = 1, n
                    if (x(indices(i), j) < threshold) then
                        left_count = left_count + 1
                        left_indices(left_count) = indices(i)
                    else
                        right_count = right_count + 1
                        right_indices(right_count) = indices(i)
                    end if
                end do
                candidate_impurity = &
                    cart_classifier_weighted_impurity(labels, weights, &
                    left_indices(:left_count), classes, criterion) + &
                    cart_classifier_weighted_impurity(labels, weights, &
                    right_indices(:right_count), classes, criterion)
                if (.not. found .or. candidate_impurity < best_impurity) then
                    found = .true.
                    best_feature = j
                    best_threshold = threshold
                    best_impurity = candidate_impurity
                end if
            end do
        end do
    end subroutine cart_classifier_find_best_split

    real(dp) function cart_classifier_weighted_impurity(labels, weights, &
            indices, classes, criterion) result(impurity)
        integer, intent(in) :: labels(:), indices(:), classes(:), criterion
        real(dp), intent(in) :: weights(:)
        real(dp) :: total_weight, class_weight, probability
        integer :: i, j

        total_weight = sum(weights(indices))
        impurity = 0.0_dp
        if (total_weight <= 0.0_dp) return
        do j = 1, size(classes)
            class_weight = 0.0_dp
            do i = 1, size(indices)
                if (labels(indices(i)) == classes(j)) then
                    class_weight = class_weight + weights(indices(i))
                end if
            end do
            probability = class_weight/total_weight
            if (criterion == CART_CRITERION_GINI) then
                impurity = impurity + probability*(1.0_dp - probability)* &
                    total_weight
            else
                if (probability > 0.0_dp) impurity = impurity - &
                    class_weight*log(probability)
            end if
        end do
    end function cart_classifier_weighted_impurity

    subroutine cart_classifier_leaf_probability(labels, weights, indices, classes, &
            probabilities)
        integer, intent(in) :: labels(:), indices(:), classes(:)
        real(dp), intent(in) :: weights(:)
        real(dp), intent(out) :: probabilities(:)
        real(dp) :: total_weight
        integer :: i, j

        total_weight = sum(weights(indices))
        probabilities = 0.0_dp
        do j = 1, size(classes)
            do i = 1, size(indices)
                if (labels(indices(i)) == classes(j)) then
                    probabilities(j) = probabilities(j) + weights(indices(i))
                end if
            end do
        end do
        if (total_weight > 0.0_dp) probabilities = probabilities/total_weight
    end subroutine cart_classifier_leaf_probability

    subroutine cart_classifier_predict_proba(self, x, probabilities, status)
        class(cart_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, node

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier predict_proba: model, input, or shape is invalid")
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
            probabilities(i, :) = self%probability(node, :)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine cart_classifier_predict_proba

    subroutine cart_classifier_predict(self, x, labels, status)
        class(cart_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, class_index

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            class_index = maxloc(probabilities(i, :), dim=1)
            labels(i) = self%class_label(class_index)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine cart_classifier_predict

    function cart_classifier_classes(self) result(classes)
        class(cart_classifier_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function cart_classifier_classes

    integer function cart_classifier_feature_count(self) result(count)
        class(cart_classifier_t), intent(in) :: self
        count = self%n_inputs
    end function cart_classifier_feature_count

    integer function cart_classifier_class_count(self) result(count)
        class(cart_classifier_t), intent(in) :: self
        count = self%n_classes
    end function cart_classifier_class_count

    integer function cart_classifier_node_count(self) result(count)
        class(cart_classifier_t), intent(in) :: self
        count = self%n_nodes
    end function cart_classifier_node_count

    integer function cart_classifier_depth(self) result(depth)
        class(cart_classifier_t), intent(in) :: self
        depth = self%max_depth
    end function cart_classifier_depth

    integer function cart_classifier_criterion(self) result(criterion)
        class(cart_classifier_t), intent(in) :: self
        criterion = self%criterion_code
    end function cart_classifier_criterion

    logical function cart_classifier_fitted(self) result(fitted)
        class(cart_classifier_t), intent(in) :: self
        fitted = self%initialized
    end function cart_classifier_fitted

    subroutine unique_sorted_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer :: i, j, n_unique, temporary

        allocate(classes(size(labels)))
        classes = labels
        do i = 2, size(classes)
            temporary = classes(i)
            j = i - 1
            do while (j >= 1)
                if (classes(j) <= temporary) exit
                classes(j + 1) = classes(j)
                j = j - 1
            end do
            classes(j + 1) = temporary
        end do
        n_unique = 1
        do i = 2, size(classes)
            if (classes(i) /= classes(n_unique)) then
                n_unique = n_unique + 1
                classes(n_unique) = classes(i)
            end if
        end do
        if (n_unique < size(classes)) classes = classes(:n_unique)
    end subroutine unique_sorted_labels

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

end module fortml_cart_classifier
