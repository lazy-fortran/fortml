module fortml_bagging_classifier
    !! Deterministic bagging classifier built from weighted CART trees.
    !!
    !! The estimator samples rows for each tree with a seeded, reproducible
    !! stream. Bootstrap and without-replacement subsets are supported, with
    !! one observation from every class forced into each subset so that the
    !! wrapped CART probability columns remain aligned. The tree routing is
    !! discrete: input JVP/VJP products return a typed refusal rather than a
    !! misleading zero. CUDA requests are also explicit refusals until a
    !! resident ensemble representation is linked.
    use, intrinsic :: iso_fortran_env, only: int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_cart_classifier, only: cart_classifier_t, CART_CRITERION_GINI, &
        CART_CRITERION_ENTROPY
    implicit none
    private

    integer, parameter, public :: BAGGING_MAX_ESTIMATORS = 256

    type, public :: bagging_classifier_t
        private
        type(cart_classifier_t), allocatable :: trees(:)
        integer, allocatable :: class_label(:)
        integer :: n_inputs = 0
        integer :: n_classes = 0
        integer :: n_trees = 0
        integer :: max_samples_value = 0
        integer :: max_depth_value = 0
        integer :: min_samples_leaf_value = 1
        integer :: criterion_code = CART_CRITERION_GINI
        integer :: seed_value = 1
        logical :: bootstrap_value = .true.
        logical :: initialized = .false.
    contains
        procedure, public :: fit => bagging_classifier_fit
        procedure, public :: predict_proba => bagging_classifier_predict_proba
        procedure, public :: predict_proba_device => &
            bagging_classifier_predict_proba_device
        procedure, public :: predict => bagging_classifier_predict
        procedure, public :: predict_device => bagging_classifier_predict_device
        procedure, public :: predict_proba_jvp => bagging_classifier_predict_proba_jvp
        procedure, public :: predict_proba_vjp => bagging_classifier_predict_proba_vjp
        procedure, public :: classes => bagging_classifier_classes
        procedure, public :: feature_count => bagging_classifier_feature_count
        procedure, public :: class_count => bagging_classifier_class_count
        procedure, public :: tree_count => bagging_classifier_tree_count
        procedure, public :: max_samples => bagging_classifier_max_samples
        procedure, public :: max_depth => bagging_classifier_max_depth
        procedure, public :: min_leaf => bagging_classifier_min_leaf
        procedure, public :: criterion => bagging_classifier_criterion
        procedure, public :: random_seed => bagging_classifier_seed
        procedure, public :: bootstrap => bagging_classifier_bootstrap
        procedure, public :: fitted => bagging_classifier_fitted
        procedure, public :: device_supported => bagging_classifier_device_supported
    end type bagging_classifier_t

    public :: bagging_classifier_fit
    public :: bagging_classifier_predict_proba
    public :: bagging_classifier_predict_proba_device
    public :: bagging_classifier_predict
    public :: bagging_classifier_predict_device

contains

    subroutine bagging_classifier_fit(self, x, labels, status, n_trees, max_depth, &
            min_samples_leaf, max_samples, bootstrap, criterion, seed, sample_weight, &
            missing_policy)
        class(bagging_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_trees, max_depth, min_samples_leaf, &
            max_samples, criterion, seed
        logical, intent(in), optional :: bootstrap
        real(dp), intent(in), optional :: sample_weight(:)
        character(len=*), intent(in), optional :: missing_policy
        integer, allocatable :: classes(:), indices(:), sampled_labels(:)
        real(dp), allocatable :: sampled_x(:, :), sampled_weight(:), weights(:)
        integer(int64) :: state
        integer :: n, p, nt, md, ml, ms, crit, sd, t
        logical :: use_bootstrap

        self%initialized = .false.
        n = size(x, 1)
        p = size(x, 2)
        nt = 10
        if (present(n_trees)) nt = n_trees
        md = 3
        if (present(max_depth)) md = max_depth
        ml = 1
        if (present(min_samples_leaf)) ml = min_samples_leaf
        ms = n
        if (present(max_samples)) ms = max_samples
        use_bootstrap = .true.
        if (present(bootstrap)) use_bootstrap = bootstrap
        crit = CART_CRITERION_GINI
        if (present(criterion)) crit = criterion
        sd = 1
        if (present(seed)) sd = seed
        if (n < 2 .or. p < 1 .or. size(labels) /= n .or. &
            nt < 1 .or. nt > BAGGING_MAX_ESTIMATORS .or. md < 0 .or. md > 12 .or. &
            ml < 1 .or. ml > ms .or. ms < 1 .or. ms > n .or. sd < 1 .or. &
            (crit /= CART_CRITERION_GINI .and. crit /= CART_CRITERION_ENTROPY)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging classifier fit: invalid dimensions or hyperparameters")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging classifier fit: inputs must be finite")
            return
        end if
        call sorted_unique_labels(labels, classes)
        if (size(classes) < 2 .or. ms < size(classes)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging classifier fit: every subset needs every class")
            return
        end if
        allocate(weights(n))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "bagging classifier fit: sample weights must be finite and positive")
                return
            end if
            weights = sample_weight
        end if

        allocate(self%trees(nt), self%class_label(size(classes)))
        self%class_label = classes
        self%n_inputs = p
        self%n_classes = size(classes)
        self%n_trees = nt
        self%max_samples_value = ms
        self%max_depth_value = md
        self%min_samples_leaf_value = ml
        self%criterion_code = crit
        self%seed_value = sd
        self%bootstrap_value = use_bootstrap
        state = int(sd, int64)
        allocate(indices(ms), sampled_x(ms, p), sampled_labels(ms), sampled_weight(ms))
        do t = 1, nt
            call sample_subset(labels, classes, ms, use_bootstrap, state, indices, status)
            if (status%code /= FORTNUM_OK) return
            sampled_x = x(indices, :) 
            sampled_labels = labels(indices)
            sampled_weight = weights(indices)
            if (present(missing_policy)) then
                call self%trees(t)%fit(sampled_x, sampled_labels, status, &
                    max_depth=md, min_samples_leaf=ml, criterion=crit, &
                    sample_weight=sampled_weight, missing_policy=missing_policy)
            else
                call self%trees(t)%fit(sampled_x, sampled_labels, status, &
                    max_depth=md, min_samples_leaf=ml, criterion=crit, &
                    sample_weight=sampled_weight)
            end if
            if (status%code /= FORTNUM_OK) return
        end do
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine bagging_classifier_fit

    subroutine bagging_classifier_predict_proba(self, x, probabilities, status)
        class(bagging_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: tree_probabilities(:, :)
        integer :: t

        probabilities = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging classifier predict_proba: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_inputs .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging classifier predict_proba: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging classifier predict_proba: inputs must be finite")
            return
        end if
        allocate(tree_probabilities(size(x, 1), self%n_classes))
        do t = 1, self%n_trees
            call self%trees(t)%predict_proba(x, tree_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            probabilities = probabilities + tree_probabilities
        end do
        probabilities = probabilities/real(self%n_trees, dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine bagging_classifier_predict_proba

    subroutine bagging_classifier_predict(self, x, labels, status)
        class(bagging_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging classifier predict: model is not fitted")
            return
        end if
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging classifier predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bagging_classifier_predict

    subroutine bagging_classifier_predict_proba_device(self, device, x, &
            probabilities, status)
        class(bagging_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging probability prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "bagging probability prediction: no resident CUDA ensemble is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging probability prediction: device kind is invalid")
        end select
    end subroutine bagging_classifier_predict_proba_device

    subroutine bagging_classifier_predict_device(self, device, x, labels, status)
        class(bagging_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging label prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "bagging label prediction: no resident CUDA ensemble is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging label prediction: device kind is invalid")
        end select
    end subroutine bagging_classifier_predict_device

    subroutine bagging_classifier_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(bagging_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging probability JVP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_inputs .or. any(shape(x_dot) /= shape(x)) .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging probability JVP: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging probability JVP: inputs must be finite")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "bagging probability JVP: discrete tree routing has no smooth product")
    end subroutine bagging_classifier_predict_proba_jvp

    subroutine bagging_classifier_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(bagging_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        x_bar = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging probability VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_inputs .or. &
            any(shape(probabilities_bar) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging probability VJP: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging probability VJP: inputs must be finite")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "bagging probability VJP: discrete tree routing has no smooth product")
    end subroutine bagging_classifier_predict_proba_vjp

    function bagging_classifier_classes(self) result(classes)
        class(bagging_classifier_t), intent(in) :: self
        integer, allocatable :: classes(:)
        if (allocated(self%class_label)) then
            allocate(classes, source=self%class_label)
        else
            allocate(classes(0))
        end if
    end function bagging_classifier_classes

    integer function bagging_classifier_feature_count(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%n_inputs
    end function bagging_classifier_feature_count

    integer function bagging_classifier_class_count(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%n_classes
    end function bagging_classifier_class_count

    integer function bagging_classifier_tree_count(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%n_trees
    end function bagging_classifier_tree_count

    integer function bagging_classifier_max_samples(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%max_samples_value
    end function bagging_classifier_max_samples

    integer function bagging_classifier_max_depth(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%max_depth_value
    end function bagging_classifier_max_depth

    integer function bagging_classifier_min_leaf(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%min_samples_leaf_value
    end function bagging_classifier_min_leaf

    integer function bagging_classifier_criterion(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%criterion_code
    end function bagging_classifier_criterion

    integer function bagging_classifier_seed(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%seed_value
    end function bagging_classifier_seed

    logical function bagging_classifier_bootstrap(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%bootstrap_value
    end function bagging_classifier_bootstrap

    logical function bagging_classifier_fitted(self) result(value)
        class(bagging_classifier_t), intent(in) :: self
        value = self%initialized .and. self%n_trees > 0 .and. self%n_classes > 1 .and. &
            allocated(self%trees) .and. allocated(self%class_label)
    end function bagging_classifier_fitted

    logical function bagging_classifier_device_supported(self, device_kind) result(value)
        class(bagging_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = device_kind == FORTML_DEVICE_CPU .and. self%fitted()
    end function bagging_classifier_device_supported

    subroutine sample_subset(labels, classes, n_draws, bootstrap, state, indices, status)
        integer, intent(in) :: labels(:), classes(:), n_draws
        logical, intent(in) :: bootstrap
        integer(int64), intent(inout) :: state
        integer, intent(out) :: indices(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: c, i, candidate, n_classes

        n_classes = size(classes)
        if (size(indices) /= n_draws .or. n_draws < n_classes) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "bagging classifier sampling: subset shape or class coverage is invalid")
            return
        end if
        do c = 1, n_classes
            indices(c) = first_index(labels, classes(c))
            if (indices(c) < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "bagging classifier sampling: class index is missing")
                return
            end if
        end do
        if (bootstrap) then
            do i = n_classes + 1, n_draws
                indices(i) = 1 + int(next_uniform(state)*real(size(labels), dp))
            end do
        else
            do i = n_classes + 1, n_draws
                candidate = 1 + int(next_uniform(state)*real(size(labels), dp))
                do while (contains_index(indices(:i-1), candidate))
                    candidate = 1 + int(next_uniform(state)*real(size(labels), dp))
                end do
                indices(i) = candidate
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine sample_subset

    integer function first_index(labels, value) result(index)
        integer, intent(in) :: labels(:), value
        integer :: i
        index = 0
        do i = 1, size(labels)
            if (labels(i) == value) then
                index = i
                return
            end if
        end do
    end function first_index

    logical function contains_index(values, value) result(found)
        integer, intent(in) :: values(:), value
        found = any(values == value)
    end function contains_index

    real(dp) function next_uniform(state) result(value)
        integer(int64), intent(inout) :: state
        state = modulo(48271_int64*state, 2147483647_int64)
        value = real(state, dp)/2147483647.0_dp
    end function next_uniform

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, count, key

        allocate(work(size(labels)))
        work = labels
        do i = 2, size(work)
            key = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= key) exit
                work(j + 1) = work(j)
                j = j - 1
            end do
            work(j + 1) = key
        end do
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(count)) then
                count = count + 1
                work(count) = work(i)
            end if
        end do
        allocate(classes(count))
        classes = work(:count)
    end subroutine sorted_unique_labels

end module fortml_bagging_classifier
