!> Extremely randomized, finite-only tree ensemble classification.
module fortml_extra_trees_classifier
    !! A deterministic Extra-Trees style classifier.
    !!
    !! Trees use the complete training set (no bootstrap) and choose the best
    !! impurity reduction among seeded random feature/threshold candidates at
    !! each node.  This is deliberately a dense CPU contract: split routing is
    !! discrete, and CUDA requests return a typed refusal until a resident tree
    !! kernel is linked.
    use, intrinsic :: iso_fortran_env, only: int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_cart_classifier, only: CART_CRITERION_GINI, CART_CRITERION_ENTROPY
    implicit none
    private

    integer, parameter, public :: EXTRA_TREES_MAX_TREES = 256
    integer, parameter, public :: EXTRA_TREES_DEFAULT_SEED = 5489
    integer, parameter, public :: EXTRA_TREES_DEFAULT_RANDOM_SPLITS = 16

    type :: extra_tree_t
        integer :: n_nodes = 0
        integer :: max_depth = 0
        integer :: n_classes = 0
        integer, allocatable :: feature(:), left_child(:), right_child(:)
        real(dp), allocatable :: threshold(:), probability(:, :)
        logical, allocatable :: leaf(:)
    end type extra_tree_t

    type, public :: extra_trees_classifier_t
        private
        type(extra_tree_t), allocatable :: trees(:)
        integer, allocatable :: class_label(:)
        integer :: n_inputs = 0
        integer :: n_classes = 0
        integer :: n_trees = 0
        integer :: max_depth = 0
        integer :: min_samples_leaf = 1
        integer :: max_features = 0
        integer :: random_splits = EXTRA_TREES_DEFAULT_RANDOM_SPLITS
        integer :: criterion_code = CART_CRITERION_GINI
        integer :: seed = EXTRA_TREES_DEFAULT_SEED
        logical :: initialized = .false.
    contains
        procedure, public :: fit => extra_trees_classifier_fit
        procedure, public :: predict_proba => extra_trees_classifier_predict_proba
        procedure, public :: predict_proba_device => &
            extra_trees_classifier_predict_proba_device
        procedure, public :: predict => extra_trees_classifier_predict
        procedure, public :: predict_device => extra_trees_classifier_predict_device
        procedure, public :: classes => extra_trees_classifier_classes
        procedure, public :: feature_count => extra_trees_classifier_feature_count
        procedure, public :: class_count => extra_trees_classifier_class_count
        procedure, public :: tree_count => extra_trees_classifier_tree_count
        procedure, public :: depth => extra_trees_classifier_depth
        procedure, public :: min_leaf => extra_trees_classifier_min_leaf
        procedure, public :: feature_subsample_count => &
            extra_trees_classifier_feature_subsample_count
        procedure, public :: random_split_count => &
            extra_trees_classifier_random_split_count
        procedure, public :: criterion => extra_trees_classifier_criterion
        procedure, public :: random_seed => extra_trees_classifier_seed
        procedure, public :: fitted => extra_trees_classifier_fitted
        procedure, public :: device_supported => &
            extra_trees_classifier_device_supported
    end type extra_trees_classifier_t

    public :: extra_trees_classifier_fit
    public :: extra_trees_classifier_predict_proba
    public :: extra_trees_classifier_predict_proba_device
    public :: extra_trees_classifier_predict
    public :: extra_trees_classifier_predict_device

contains

    subroutine extra_trees_classifier_fit(self, x, labels, status, n_trees, &
            max_depth, min_samples_leaf, max_features, random_splits, criterion, seed, &
            sample_weight)
        class(extra_trees_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_trees, max_depth, min_samples_leaf, &
            max_features, random_splits, criterion, seed
        real(dp), intent(in), optional :: sample_weight(:)
        integer :: nt, md, ml, mf, ns, crit, sd, n, i, max_nodes
        integer, allocatable :: classes(:), indices(:)
        real(dp), allocatable :: weights(:)
        integer(int64) :: state

        nt = 100; if (present(n_trees)) nt = n_trees
        md = 6; if (present(max_depth)) md = max_depth
        ml = 1; if (present(min_samples_leaf)) ml = min_samples_leaf
        mf = max(1, int(sqrt(real(max(1, size(x, 2)), dp))))
        if (present(max_features)) mf = max_features
        ns = EXTRA_TREES_DEFAULT_RANDOM_SPLITS; if (present(random_splits)) ns = random_splits
        crit = CART_CRITERION_GINI; if (present(criterion)) crit = criterion
        sd = EXTRA_TREES_DEFAULT_SEED; if (present(seed)) sd = seed
        n = size(x, 1)
        if (n < 2 .or. size(x, 2) < 1 .or. size(labels) /= n .or. &
            nt < 1 .or. nt > EXTRA_TREES_MAX_TREES .or. md < 0 .or. md > 12 .or. &
            ml < 1 .or. ml > n .or. mf < 1 .or. mf > size(x, 2) .or. ns < 1 .or. &
            (crit /= CART_CRITERION_GINI .and. crit /= CART_CRITERION_ENTROPY) .or. sd < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "extra trees classifier fit: invalid dimensions or hyperparameters")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "extra trees classifier fit: inputs must be finite")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= n .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "extra trees classifier fit: sample weights must be finite and positive")
                return
            end if
        end if
        call unique_sorted_labels(labels, classes)
        if (size(classes) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "extra trees classifier fit: at least two classes are required")
            return
        end if
        allocate(weights(n), indices(n), self%trees(nt), self%class_label(size(classes)))
        weights = 1.0_dp
        if (present(sample_weight)) weights = sample_weight
        indices = [(i, i=1,n)]
        self%class_label = classes
        self%n_inputs = size(x, 2); self%n_classes = size(classes)
        self%n_trees = nt; self%max_depth = md; self%min_samples_leaf = ml
        self%max_features = mf; self%random_splits = ns; self%criterion_code = crit
        self%seed = sd; self%initialized = .false.
        max_nodes = 2**(md + 1) - 1
        state = int(sd, int64)
        do i = 1, nt
            self%trees(i)%max_depth = md
            self%trees(i)%n_classes = size(classes)
            self%trees(i)%n_nodes = 0
            allocate(self%trees(i)%feature(max_nodes), self%trees(i)%left_child(max_nodes), &
                self%trees(i)%right_child(max_nodes), self%trees(i)%threshold(max_nodes), &
                self%trees(i)%probability(max_nodes, size(classes)), self%trees(i)%leaf(max_nodes))
            self%trees(i)%feature = 0; self%trees(i)%left_child = 0
            self%trees(i)%right_child = 0; self%trees(i)%threshold = 0.0_dp
            self%trees(i)%probability = 0.0_dp; self%trees(i)%leaf = .true.
            call extra_tree_build_node(self%trees(i), x, labels, weights, indices, 0, 1, &
                classes, ml, mf, ns, crit, state, status)
            if (status%code /= FORTNUM_OK) return
        end do
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine extra_trees_classifier_fit

    recursive subroutine extra_tree_build_node(tree, x, labels, weights, indices, depth, &
            node, classes, min_leaf, max_features, random_splits, criterion, state, status)
        type(extra_tree_t), intent(inout) :: tree
        real(dp), intent(in) :: x(:, :), weights(:)
        integer, intent(in) :: labels(:), indices(:), depth, node, classes(:), min_leaf
        integer, intent(in) :: max_features, random_splits, criterion
        integer(int64), intent(inout) :: state
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, f, left_count, right_count, best_feature, n_features, n
        integer, allocatable :: left_indices(:), right_indices(:), features(:)
        real(dp) :: best_threshold, best_impurity, parent_impurity, lo, hi, threshold, impurity
        logical :: found

        n = size(indices)
        if (node > size(tree%leaf)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "extra trees classifier fit: internal node capacity exceeded")
            return
        end if
        tree%n_nodes = max(tree%n_nodes, node)
        call extra_leaf_probability(labels, weights, indices, classes, tree%probability(node, :))
        tree%leaf(node) = .true.
        if (depth >= tree%max_depth .or. n < 2*min_leaf) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        parent_impurity = extra_impurity(labels, weights, indices, classes, criterion)
        n_features = size(x, 2)
        allocate(features(max_features))
        do i = 1, max_features
            features(i) = 1 + int(extra_uniform(state)*real(n_features, dp))
            if (i > 1) then
                do while (any(features(:i-1) == features(i)))
                    features(i) = 1 + int(extra_uniform(state)*real(n_features, dp))
                end do
            end if
        end do
        found = .false.; best_impurity = huge(1.0_dp); best_feature = 0; best_threshold = 0.0_dp
        allocate(left_indices(n), right_indices(n))
        do i = 1, max_features
            f = features(i); lo = minval(x(indices, f)); hi = maxval(x(indices, f))
            if (hi <= lo) cycle
            do j = 1, random_splits
                threshold = lo + extra_uniform(state)*(hi - lo)
                left_count = count(x(indices, f) < threshold); right_count = n - left_count
                if (left_count < min_leaf .or. right_count < min_leaf) cycle
                impurity = extra_impurity_partition(x, labels, weights, indices, f, threshold, &
                    classes, criterion, left_indices, right_indices, left_count, right_count)
                if (.not. found .or. impurity < best_impurity) then
                    found = .true.; best_impurity = impurity; best_feature = f
                    best_threshold = threshold
                end if
            end do
        end do
        if (.not. found .or. best_impurity >= parent_impurity) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        left_count = 0; right_count = 0
        do i = 1, n
            if (x(indices(i), best_feature) < best_threshold) then
                left_count = left_count + 1; left_indices(left_count) = indices(i)
            else
                right_count = right_count + 1; right_indices(right_count) = indices(i)
            end if
        end do
        tree%leaf(node) = .false.; tree%feature(node) = best_feature
        tree%threshold(node) = best_threshold; tree%left_child(node) = 2*node
        tree%right_child(node) = 2*node + 1
        call extra_tree_build_node(tree, x, labels, weights, left_indices(:left_count), depth+1, &
            2*node, classes, min_leaf, max_features, random_splits, criterion, state, status)
        if (status%code /= FORTNUM_OK) return
        call extra_tree_build_node(tree, x, labels, weights, right_indices(:right_count), depth+1, &
            2*node+1, classes, min_leaf, max_features, random_splits, criterion, state, status)
    end subroutine extra_tree_build_node

    real(dp) function extra_impurity_partition(x, labels, weights, indices, feature, threshold, &
            classes, criterion, left_indices, right_indices, left_count, right_count) result(value)
        real(dp), intent(in) :: x(:, :), weights(:), threshold
        integer, intent(in) :: labels(:), indices(:), feature, classes(:), criterion
        integer, intent(out) :: left_indices(:), right_indices(:), left_count, right_count
        integer :: i
        left_count = 0; right_count = 0
        do i = 1, size(indices)
            if (x(indices(i), feature) < threshold) then
                left_count = left_count + 1; left_indices(left_count) = indices(i)
            else
                right_count = right_count + 1; right_indices(right_count) = indices(i)
            end if
        end do
        value = extra_impurity(labels, weights, left_indices(:left_count), classes, criterion) + &
            extra_impurity(labels, weights, right_indices(:right_count), classes, criterion)
    end function extra_impurity_partition

    real(dp) function extra_impurity(labels, weights, indices, classes, criterion) result(value)
        integer, intent(in) :: labels(:), indices(:), classes(:), criterion
        real(dp), intent(in) :: weights(:)
        integer :: i, j
        real(dp) :: total, mass, p
        total = 0.0_dp; value = 0.0_dp
        do i = 1, size(indices); total = total + weights(indices(i)); end do
        if (total <= 0.0_dp) return
        do j = 1, size(classes)
            mass = 0.0_dp
            do i = 1, size(indices)
                if (labels(indices(i)) == classes(j)) mass = mass + weights(indices(i))
            end do
            p = mass/total
            if (criterion == CART_CRITERION_ENTROPY) then
                if (p > 0.0_dp) value = value - total*p*log(p)
            else
                value = value + total*p*(1.0_dp - p)
            end if
        end do
    end function extra_impurity

    subroutine extra_leaf_probability(labels, weights, indices, classes, probability)
        integer, intent(in) :: labels(:), indices(:), classes(:)
        real(dp), intent(in) :: weights(:)
        real(dp), intent(out) :: probability(:)
        integer :: i, j
        real(dp) :: total
        probability = 0.0_dp; total = sum(weights(indices))
        if (total <= 0.0_dp) return
        do j = 1, size(classes)
            do i = 1, size(indices)
                if (labels(indices(i)) == classes(j)) probability(j) = probability(j) + weights(indices(i))
            end do
        end do
        probability = probability/total
    end subroutine extra_leaf_probability

    subroutine extra_trees_classifier_predict_proba(self, x, probabilities, status)
        class(extra_trees_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, node
        probabilities = 0.0_dp
        if (.not. self%initialized .or. size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "extra trees classifier predict_proba: model, input, or shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            do j = 1, self%n_trees
                node = 1
                do while (.not. self%trees(j)%leaf(node))
                    if (x(i, self%trees(j)%feature(node)) < self%trees(j)%threshold(node)) then
                        node = self%trees(j)%left_child(node)
                    else
                        node = self%trees(j)%right_child(node)
                    end if
                end do
                probabilities(i, :) = probabilities(i, :) + self%trees(j)%probability(node, :)
            end do
        end do
        probabilities = probabilities/real(self%n_trees, dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine extra_trees_classifier_predict_proba

    subroutine extra_trees_classifier_predict_proba_device(self, device, x, probabilities, status)
        class(extra_trees_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "extra trees device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "extra trees device prediction: no resident CUDA tree ensemble is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, "extra trees device prediction: device kind is invalid")
        end select
    end subroutine extra_trees_classifier_predict_proba_device

    subroutine extra_trees_classifier_predict(self, x, labels, status)
        class(extra_trees_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "extra trees classifier predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels); labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1)); end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine extra_trees_classifier_predict

    subroutine extra_trees_classifier_predict_device(self, device, x, labels, status)
        class(extra_trees_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "extra trees label prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU); call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "extra trees label prediction: no resident CUDA tree ensemble is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, "extra trees label prediction: device kind is invalid")
        end select
    end subroutine extra_trees_classifier_predict_device

    function extra_trees_classifier_classes(self) result(classes)
        class(extra_trees_classifier_t), intent(in) :: self
        integer, allocatable :: classes(:)
        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function extra_trees_classifier_classes
    integer function extra_trees_classifier_feature_count(self) result(count)
        class(extra_trees_classifier_t), intent(in) :: self
        count = self%n_inputs
    end function extra_trees_classifier_feature_count
    integer function extra_trees_classifier_class_count(self) result(count)
        class(extra_trees_classifier_t), intent(in) :: self
        count = self%n_classes
    end function extra_trees_classifier_class_count
    integer function extra_trees_classifier_tree_count(self) result(count)
        class(extra_trees_classifier_t), intent(in) :: self
        count = self%n_trees
    end function extra_trees_classifier_tree_count
    integer function extra_trees_classifier_depth(self) result(depth)
        class(extra_trees_classifier_t), intent(in) :: self
        depth = self%max_depth
    end function extra_trees_classifier_depth
    integer function extra_trees_classifier_min_leaf(self) result(value)
        class(extra_trees_classifier_t), intent(in) :: self
        value = self%min_samples_leaf
    end function extra_trees_classifier_min_leaf
    integer function extra_trees_classifier_feature_subsample_count(self) result(value)
        class(extra_trees_classifier_t), intent(in) :: self
        value = self%max_features
    end function extra_trees_classifier_feature_subsample_count
    integer function extra_trees_classifier_random_split_count(self) result(value)
        class(extra_trees_classifier_t), intent(in) :: self
        value = self%random_splits
    end function extra_trees_classifier_random_split_count
    integer function extra_trees_classifier_criterion(self) result(value)
        class(extra_trees_classifier_t), intent(in) :: self
        value = self%criterion_code
    end function extra_trees_classifier_criterion
    integer function extra_trees_classifier_seed(self) result(value)
        class(extra_trees_classifier_t), intent(in) :: self
        value = self%seed
    end function extra_trees_classifier_seed
    logical function extra_trees_classifier_fitted(self) result(value)
        class(extra_trees_classifier_t), intent(in) :: self
        value = self%initialized .and. allocated(self%trees) .and. self%n_trees > 0
    end function extra_trees_classifier_fitted
    logical function extra_trees_classifier_device_supported(self, device_kind) result(value)
        class(extra_trees_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = device_kind == FORTML_DEVICE_CPU .and. self%fitted()
    end function extra_trees_classifier_device_supported

    real(dp) function extra_uniform(state) result(value)
        integer(int64), intent(inout) :: state
        state = modulo(48271_int64*state, 2147483647_int64)
        value = real(state, dp)/2147483647.0_dp
    end function extra_uniform

    subroutine unique_sorted_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer :: i, j, count, temporary
        allocate(classes(size(labels))); classes = labels
        do i = 2, size(classes)
            temporary = classes(i); j = i - 1
            do while (j >= 1)
                if (classes(j) <= temporary) exit
                classes(j + 1) = classes(j); j = j - 1
            end do
            classes(j + 1) = temporary
        end do
        count = 1
        do i = 2, size(classes)
            if (classes(i) /= classes(count)) then; count = count + 1; classes(count) = classes(i); end if
        end do
        if (count < size(classes)) classes = classes(:count)
    end subroutine unique_sorted_labels

end module fortml_extra_trees_classifier
