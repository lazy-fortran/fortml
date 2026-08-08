!> Deterministic CART classification with optional NaN routing.
module fortml_cart_classifier
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: CART_CRITERION_GINI = 1
    integer, parameter, public :: CART_CRITERION_ENTROPY = 2
    integer, parameter, public :: CART_MISSING_ERROR = 0
    integer, parameter, public :: CART_MISSING_LEARN = 1
    integer, parameter, public :: CART_MISSING_LEFT = 2
    integer, parameter, public :: CART_MISSING_RIGHT = 3

    !> A deterministic numeric CART classifier with weighted impurity splits.
    !>
    !> Features are visited in ascending order and each feature's distinct
    !> sorted midpoint is considered in ascending order.  A split is accepted
    !> only on strict impurity improvement, so ties retain the first feature
    !> and threshold. Leaves store weighted class frequencies as empirical
    !> piecewise-constant probabilities for this tree.
    !> The implementation is dense.  The default `missing_policy="error"`
    !> retains the finite-only behavior; `"learn"` compares both default
    !> branches at each split and stores the strictly best one (left wins an
    !> exact tie), while `"left"` and `"right"` force a deterministic branch.
    type, public :: cart_classifier_t
        private
        integer :: n_inputs = 0
        integer :: n_classes = 0
        integer :: max_depth = 0
        integer :: min_samples_leaf = 1
        integer :: criterion_code = CART_CRITERION_GINI
        integer :: missing_code = CART_MISSING_ERROR
        integer :: n_nodes = 0
        integer, allocatable :: class_label(:)
        integer, allocatable :: feature(:), left_child(:), right_child(:)
        real(dp), allocatable :: threshold(:), probability(:, :)
        logical, allocatable :: leaf(:), missing_left(:)
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
        procedure, public :: missing_policy => cart_classifier_missing_policy
        procedure, public :: accepts_missing => cart_classifier_accepts_missing
        procedure, public :: fitted => cart_classifier_fitted
    end type cart_classifier_t

contains

    subroutine cart_classifier_fit(self, x, labels, status, max_depth, &
            min_samples_leaf, sample_weight, criterion, missing_policy)
        class(cart_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: max_depth, min_samples_leaf, criterion
        real(dp), intent(in), optional :: sample_weight(:)
        character(len=*), intent(in), optional :: missing_policy
        integer, allocatable :: indices(:), classes(:)
        real(dp), allocatable :: weights(:)
        integer :: tree_depth, leaf_size, criterion_value, missing_code_value
        integer :: max_nodes, i
        character(len=16) :: missing_policy_value

        tree_depth = 3
        if (present(max_depth)) tree_depth = max_depth
        leaf_size = 1
        if (present(min_samples_leaf)) leaf_size = min_samples_leaf
        criterion_value = CART_CRITERION_GINI
        if (present(criterion)) criterion_value = criterion
        missing_policy_value = "error"
        if (present(missing_policy)) missing_policy_value = trim(adjustl(missing_policy))
        missing_code_value = cart_classifier_parse_missing_policy(missing_policy_value)
        if (size(x, 1) < 2 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1) .or. tree_depth < 0 .or. &
            tree_depth > 12 .or. leaf_size < 1 .or. &
            leaf_size > size(x, 1) .or. (criterion_value /= CART_CRITERION_GINI &
            .and. criterion_value /= CART_CRITERION_ENTROPY)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier fit: invalid dimensions or hyperparameters")
            return
        end if
        if (missing_code_value < CART_MISSING_ERROR .or. &
            missing_code_value > CART_MISSING_RIGHT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier fit: missing_policy must be error, learn, left, or right")
            return
        end if
        if (any((.not. ieee_is_finite(x)) .and. (.not. ieee_is_nan(x)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier fit: inputs must be finite or IEEE NaN")
            return
        end if
        if (missing_code_value == CART_MISSING_ERROR .and. any(ieee_is_nan(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier fit: NaN inputs require a missing-value policy")
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
            self%leaf(max_nodes), self%missing_left(max_nodes))
        self%class_label = classes
        self%feature = 0
        self%left_child = 0
        self%right_child = 0
        self%threshold = 0.0_dp
        self%probability = 0.0_dp
        self%leaf = .true.
        self%missing_left = .false.
        allocate(indices(size(labels)))
        do i = 1, size(labels)
            indices(i) = i
        end do

        self%n_inputs = size(x, 2)
        self%n_classes = size(classes)
        self%max_depth = tree_depth
        self%min_samples_leaf = leaf_size
        self%criterion_code = criterion_value
        self%missing_code = missing_code_value
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
        logical :: found, best_missing_left, is_missing

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
            self%missing_code, best_feature, best_threshold, best_impurity, &
            best_missing_left, found)
        if (.not. found .or. best_impurity >= parent_impurity) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        left_count = 0
        do i = 1, size(indices)
            is_missing = ieee_is_nan(x(indices(i), best_feature))
            if ((is_missing .and. best_missing_left) .or. &
                (.not. is_missing .and. x(indices(i), best_feature) < best_threshold)) then
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
            is_missing = ieee_is_nan(x(indices(i), best_feature))
            if ((is_missing .and. best_missing_left) .or. &
                (.not. is_missing .and. x(indices(i), best_feature) < best_threshold)) then
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
        self%missing_left(node) = best_missing_left
        self%left_child(node) = 2*node
        self%right_child(node) = 2*node + 1
        call cart_classifier_build_node(self, x, labels, weights, left_indices, &
            depth + 1, 2*node, status)
        if (status%code /= FORTNUM_OK) return
        call cart_classifier_build_node(self, x, labels, weights, right_indices, &
            depth + 1, 2*node + 1, status)
    end subroutine cart_classifier_build_node

    subroutine cart_classifier_find_best_split(x, labels, weights, indices, &
            classes, criterion, min_leaf, missing_code, best_feature, &
            best_threshold, best_impurity, best_missing_left, found)
        real(dp), intent(in) :: x(:, :), weights(:)
        integer, intent(in) :: labels(:), indices(:), classes(:), criterion, min_leaf
        integer, intent(in) :: missing_code
        integer, intent(out) :: best_feature
        real(dp), intent(out) :: best_threshold, best_impurity
        logical, intent(out) :: best_missing_left, found
        integer, allocatable :: order(:), finite_indices(:), left_indices(:), right_indices(:)
        integer :: j, k, n, n_finite, n_missing, left_count, right_count, i, branch
        real(dp) :: threshold, candidate_impurity
        logical :: missing_left, is_missing

        n = size(indices)
        allocate(order(n), finite_indices(n), left_indices(n), right_indices(n))
        found = .false.
        best_feature = 0
        best_threshold = 0.0_dp
        best_impurity = huge(1.0_dp)
        best_missing_left = .false.
        do j = 1, size(x, 2)
            n_finite = 0
            n_missing = 0
            do i = 1, n
                if (ieee_is_nan(x(indices(i), j))) then
                    n_missing = n_missing + 1
                else
                    n_finite = n_finite + 1
                    finite_indices(n_finite) = indices(i)
                end if
            end do
            if (n_finite < 2*min_leaf) cycle
            call sort_subset_indices(x(:, j), finite_indices(:n_finite), &
                order(:n_finite))
            do k = min_leaf, n_finite - min_leaf
                if (x(finite_indices(order(k)), j) >= &
                    x(finite_indices(order(k + 1)), j)) cycle
                threshold = 0.5_dp*x(finite_indices(order(k)), j) + &
                    0.5_dp*x(finite_indices(order(k + 1)), j)
                do branch = 1, 2
                    if (missing_code == CART_MISSING_LEARN) then
                        missing_left = branch == 1
                    else if (missing_code == CART_MISSING_LEFT) then
                        if (branch == 2) cycle
                        missing_left = .true.
                    else if (missing_code == CART_MISSING_RIGHT) then
                        if (branch == 1) cycle
                        missing_left = .false.
                    else
                        if (branch == 1) cycle
                        missing_left = .false.
                    end if
                    left_count = 0
                    right_count = 0
                    do i = 1, n
                        is_missing = ieee_is_nan(x(indices(i), j))
                        if ((is_missing .and. missing_left) .or. &
                            (.not. is_missing .and. x(indices(i), j) < threshold)) then
                            left_count = left_count + 1
                            left_indices(left_count) = indices(i)
                        else
                            right_count = right_count + 1
                            right_indices(right_count) = indices(i)
                        end if
                    end do
                    if (left_count < min_leaf .or. right_count < min_leaf) cycle
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
                        best_missing_left = missing_left
                    end if
                end do
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
            any((.not. ieee_is_finite(x)) .and. (.not. ieee_is_nan(x)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier predict_proba: model, input, or shape is invalid")
            return
        end if
        if (self%missing_code == CART_MISSING_ERROR .and. any(ieee_is_nan(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CART classifier predict_proba: NaN input requires a missing-value policy")
            return
        end if
        do i = 1, size(x, 1)
            node = 1
            do while (.not. self%leaf(node))
                if (ieee_is_nan(x(i, self%feature(node)))) then
                    if (self%missing_left(node)) then
                        node = self%left_child(node)
                    else
                        node = self%right_child(node)
                    end if
                else if (x(i, self%feature(node)) < self%threshold(node)) then
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

    character(len=16) function cart_classifier_missing_policy(self) result(policy)
        class(cart_classifier_t), intent(in) :: self

        select case (self%missing_code)
        case (CART_MISSING_ERROR)
            policy = "error"
        case (CART_MISSING_LEARN)
            policy = "learn"
        case (CART_MISSING_LEFT)
            policy = "left"
        case (CART_MISSING_RIGHT)
            policy = "right"
        case default
            policy = "unfitted"
        end select
    end function cart_classifier_missing_policy

    logical function cart_classifier_accepts_missing(self) result(accepts)
        class(cart_classifier_t), intent(in) :: self

        accepts = self%initialized .and. self%missing_code /= CART_MISSING_ERROR
    end function cart_classifier_accepts_missing

    logical function cart_classifier_fitted(self) result(fitted)
        class(cart_classifier_t), intent(in) :: self
        fitted = self%initialized
    end function cart_classifier_fitted

    integer function cart_classifier_parse_missing_policy(policy) result(code)
        character(len=*), intent(in) :: policy

        select case (trim(adjustl(policy)))
        case ("error")
            code = CART_MISSING_ERROR
        case ("learn")
            code = CART_MISSING_LEARN
        case ("left")
            code = CART_MISSING_LEFT
        case ("right")
            code = CART_MISSING_RIGHT
        case default
            code = -1
        end select
    end function cart_classifier_parse_missing_policy

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
