!> Deterministic bootstrap-ensemble classification built from CART trees.
module fortml_random_forest_classifier
    !! A compact, production-oriented random-forest classifier.
    !!
    !! Every tree is fit on a deterministic bootstrap sample.  Bootstrap draws
    !! are stratified only enough to retain at least one sample of each global
    !! class, which keeps the probability columns stable for small data sets.
    !! The implementation is intentionally CPU-only: hard tree routing is
    !! piecewise constant and has no resident CUDA or derivative contract yet.
    use, intrinsic :: iso_fortran_env, only: int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_cart_classifier, only: cart_classifier_t, &
        CART_CRITERION_GINI, CART_CRITERION_ENTROPY
    implicit none
    private

    integer, parameter, public :: RANDOM_FOREST_MAX_TREES = 256
    integer, parameter, public :: RANDOM_FOREST_DEFAULT_SEED = 5489
    integer, parameter, public :: RANDOM_FOREST_CUDA_PLAN_ABI_VERSION = 1
    ! The generic status vocabulary has no separate coverage code.  A forest
    ! with a row that was never out-of-bag reports convergence failure rather
    ! than returning an in-bag substitute.
    integer, parameter, public :: RANDOM_FOREST_OOB_INSUFFICIENT = &
        FORTNUM_CONVERGENCE_ERROR

    type, public :: random_forest_classifier_t
        private
        type(cart_classifier_t), allocatable :: trees(:)
        integer, allocatable :: class_label(:)
        logical, allocatable :: bootstrap_included(:, :)
        integer :: n_inputs = 0
        integer :: n_classes = 0
        integer :: n_samples = 0
        integer :: n_trees = 0
        integer :: max_depth = 0
        integer :: min_samples_leaf = 1
        integer :: criterion_code = CART_CRITERION_GINI
        integer :: seed = RANDOM_FOREST_DEFAULT_SEED
        logical :: initialized = .false.
    contains
        procedure, public :: fit => random_forest_classifier_fit
        procedure, public :: predict_proba => random_forest_classifier_predict_proba
        procedure, public :: oob_decision_function => &
            random_forest_classifier_oob_decision_function
        procedure, public :: oob_predict_proba => &
            random_forest_classifier_oob_decision_function
        procedure, public :: predict_proba_oob => &
            random_forest_classifier_oob_decision_function
        procedure, public :: oob_decision_function_device => &
            random_forest_classifier_oob_decision_function_device
        procedure, public :: oob_predict_proba_device => &
            random_forest_classifier_oob_decision_function_device
        procedure, public :: predict_proba_oob_device => &
            random_forest_classifier_oob_decision_function_device
        procedure, public :: oob_score => random_forest_classifier_oob_score
        procedure, public :: oob_score_device => random_forest_classifier_oob_score_device
        procedure, public :: predict_proba_device => &
            random_forest_classifier_predict_proba_device
        procedure, public :: predict => random_forest_classifier_predict
        procedure, public :: predict_device => &
            random_forest_classifier_predict_device
        procedure, public :: classes => random_forest_classifier_classes
        procedure, public :: feature_count => random_forest_classifier_feature_count
        procedure, public :: class_count => random_forest_classifier_class_count
        procedure, public :: tree_count => random_forest_classifier_tree_count
        procedure, public :: depth => random_forest_classifier_depth
        procedure, public :: criterion => random_forest_classifier_criterion
        procedure, public :: random_seed => random_forest_classifier_seed
        procedure, public :: fitted => random_forest_classifier_fitted
        procedure, public :: oob_coverage => random_forest_classifier_oob_coverage
        procedure, public :: bootstrap_inclusion => &
            random_forest_classifier_bootstrap_inclusion
        procedure, public :: device_supported => &
            random_forest_classifier_device_supported
    end type random_forest_classifier_t

    !> Typed contract for a future resident CUDA forest.
    !>
    !> The plan deliberately does not expose host tree arrays or silently copy
    !> them during prediction.  Until a generated/native ensemble kernel is
    !> linked, create returns FORTNUM_NOT_IMPLEMENTED after recording the
    !> immutable model/device shape.  Keeping this shape as a public ABI makes
    !> the eventual CUDA lowering explicit without weakening the CPU oracle.
    type, public :: random_forest_cuda_plan_t
        private
        integer :: abi_version = RANDOM_FOREST_CUDA_PLAN_ABI_VERSION
        integer :: n_inputs = 0
        integer :: n_classes = 0
        integer :: n_trees = 0
        integer :: device_index = -1
        logical :: initialized = .false.
    contains
        procedure, public :: create => random_forest_cuda_plan_create
        procedure, public :: destroy => random_forest_cuda_plan_destroy
        procedure, public :: predict_proba => random_forest_cuda_plan_predict_proba
        procedure, public :: predict => random_forest_cuda_plan_predict
        procedure, public :: abi => random_forest_cuda_plan_abi
        procedure, public :: feature_count => random_forest_cuda_plan_feature_count
        procedure, public :: class_count => random_forest_cuda_plan_class_count
        procedure, public :: tree_count => random_forest_cuda_plan_tree_count
        procedure, public :: device => random_forest_cuda_plan_device
        procedure, public :: fitted => random_forest_cuda_plan_fitted
    end type random_forest_cuda_plan_t

    public :: random_forest_classifier_fit
    public :: random_forest_classifier_predict_proba
    public :: random_forest_classifier_predict_proba_device
    public :: random_forest_classifier_predict
    public :: random_forest_classifier_predict_device

contains

    subroutine random_forest_classifier_fit(self, x, labels, status, n_trees, &
            max_depth, min_samples_leaf, criterion, seed)
        class(random_forest_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_trees, max_depth, min_samples_leaf, &
            criterion, seed
        integer :: requested_trees, requested_depth, requested_leaf, &
            requested_criterion, requested_seed
        integer, allocatable :: classes(:), bootstrap(:), labels_boot(:)
        real(dp), allocatable :: x_boot(:, :)
        integer :: i, j, n, n_classes
        integer(int64) :: rng_state

        requested_trees = 100
        if (present(n_trees)) requested_trees = n_trees
        requested_depth = 6
        if (present(max_depth)) requested_depth = max_depth
        requested_leaf = 1
        if (present(min_samples_leaf)) requested_leaf = min_samples_leaf
        requested_criterion = CART_CRITERION_GINI
        if (present(criterion)) requested_criterion = criterion
        requested_seed = RANDOM_FOREST_DEFAULT_SEED
        if (present(seed)) requested_seed = seed
        n = size(x, 1)
        if (n < 2 .or. size(x, 2) < 1 .or. size(labels) /= n .or. &
            requested_trees < 1 .or. requested_trees > RANDOM_FOREST_MAX_TREES .or. &
            requested_depth < 0 .or. requested_depth > 12 .or. &
            requested_leaf < 1 .or. requested_leaf > n .or. &
            (requested_criterion /= CART_CRITERION_GINI .and. &
            requested_criterion /= CART_CRITERION_ENTROPY)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest classifier fit: invalid dimensions or hyperparameters")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest classifier fit: inputs must be finite")
            return
        end if
        call unique_sorted_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest classifier fit: at least two classes are required")
            return
        end if
        if (requested_seed < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest classifier fit: seed must be positive")
            return
        end if

        allocate(self%trees(requested_trees), self%class_label(n_classes), &
            self%bootstrap_included(n, requested_trees))
        allocate(bootstrap(n), labels_boot(n), x_boot(n, size(x, 2)))
        self%class_label = classes
        self%n_inputs = size(x, 2)
        self%n_classes = n_classes
        self%n_samples = n
        self%n_trees = requested_trees
        self%max_depth = requested_depth
        self%min_samples_leaf = requested_leaf
        self%criterion_code = requested_criterion
        self%seed = requested_seed
        self%initialized = .false.
        self%bootstrap_included = .false.
        rng_state = int(requested_seed, int64)
        do i = 1, requested_trees
            call bootstrap_indices(labels, classes, rng_state, bootstrap)
            do j = 1, n
                self%bootstrap_included(j, i) = .false.
            end do
            do j = 1, n
                x_boot(j, :) = x(bootstrap(j), :)
                labels_boot(j) = labels(bootstrap(j))
                self%bootstrap_included(bootstrap(j), i) = .true.
            end do
            call self%trees(i)%fit(x_boot, labels_boot, status, requested_depth, &
                requested_leaf, criterion=requested_criterion)
            if (status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "random forest classifier fit: CART tree failed")
                return
            end if
        end do
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_classifier_fit

    subroutine random_forest_classifier_predict_proba(self, x, probabilities, status)
        class(random_forest_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local(:, :)
        integer, allocatable :: local_classes(:)
        integer :: i, j, k, class_index

        probabilities = 0.0_dp
        if (.not. self%initialized .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%n_inputs .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest classifier predict_proba: model, input, or shape is invalid")
            return
        end if
        allocate(local(size(x, 1), self%trees(1)%class_count()))
        do i = 1, self%n_trees
            call self%trees(i)%predict_proba(x, local, status)
            if (status%code /= FORTNUM_OK) return
            local_classes = self%trees(i)%classes()
            do j = 1, size(local_classes)
                class_index = 0
                do k = 1, self%n_classes
                    if (self%class_label(k) == local_classes(j)) then
                        class_index = k
                        exit
                    end if
                end do
                if (class_index == 0) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "random forest classifier predict_proba: tree class mismatch")
                    return
                end if
                probabilities(:, class_index) = probabilities(:, class_index) + local(:, j)
            end do
        end do
        probabilities = probabilities/real(self%n_trees, dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_classifier_predict_proba

    subroutine random_forest_classifier_oob_decision_function(self, x, probabilities, &
            status)
        class(random_forest_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        ! INOUT makes the refusal transactional: no row receives an in-bag
        ! prediction when OOB coverage is incomplete or a tree fails.
        real(dp), intent(inout) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local(:, :), accumulated(:, :)
        integer, allocatable :: local_classes(:), counts(:)
        integer :: i, j, k, class_index, n_rows

        n_rows = size(x, 1)
        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB prediction: model, input, or shape is invalid")
            return
        end if
        if (.not. allocated(self%bootstrap_included)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB prediction: bootstrap state is missing")
            return
        end if
        if (n_rows < 1 .or. n_rows /= self%n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB prediction: input row count is invalid")
            return
        end if
        if (size(x, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB prediction: input feature count is invalid")
            return
        end if
        if (any(shape(probabilities) /= [n_rows, self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB prediction: output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB prediction: inputs must be finite")
            return
        end if
        allocate(counts(n_rows), accumulated(n_rows, self%n_classes))
        counts = 0
        accumulated = 0.0_dp
        do i = 1, n_rows
            counts(i) = count(.not. self%bootstrap_included(i, :))
            if (counts(i) == 0) then
                call status_set(status, RANDOM_FOREST_OOB_INSUFFICIENT, &
                    "random forest OOB prediction: insufficient out-of-bag coverage")
                return
            end if
        end do
        allocate(local(n_rows, self%trees(1)%class_count()))
        do i = 1, self%n_trees
            call self%trees(i)%predict_proba(x, local, status)
            if (status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "random forest OOB prediction: CART tree failed")
                return
            end if
            local_classes = self%trees(i)%classes()
            do j = 1, n_rows
                if (self%bootstrap_included(j, i)) cycle
                do k = 1, size(local_classes)
                    class_index = 0
                    do while (class_index < self%n_classes)
                        class_index = class_index + 1
                        if (self%class_label(class_index) == local_classes(k)) then
                            exit
                        end if
                    end do
                    if (self%class_label(class_index) /= local_classes(k)) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "random forest OOB prediction: tree class mismatch")
                        return
                    end if
                    accumulated(j, class_index) = accumulated(j, class_index) + local(j, k)
                end do
            end do
        end do
        do i = 1, n_rows
            probabilities(i, :) = accumulated(i, :)/real(counts(i), dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_classifier_oob_decision_function

    subroutine random_forest_classifier_oob_decision_function_device(self, device, x, &
            probabilities, status)
        class(random_forest_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%oob_decision_function(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "random forest OOB device prediction: resident CUDA OOB kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB device prediction: device kind is invalid")
        end select
    end subroutine random_forest_classifier_oob_decision_function_device

    subroutine random_forest_classifier_oob_score(self, x, labels, score, status)
        class(random_forest_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(inout) :: score
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, predicted

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB score: label shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%oob_decision_function(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            predicted = self%class_label(maxloc(probabilities(i, :), dim=1))
            if (i == 1) score = 0.0_dp
            if (predicted == labels(i)) score = score + 1.0_dp
        end do
        score = score/real(size(labels), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_classifier_oob_score

    subroutine random_forest_classifier_oob_score_device(self, device, x, labels, score, &
            status)
        class(random_forest_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(inout) :: score
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB score device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%oob_score(x, labels, score, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "random forest OOB score device: resident CUDA OOB kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest OOB score device: device kind is invalid")
        end select
    end subroutine random_forest_classifier_oob_score_device

    subroutine random_forest_classifier_predict_proba_device(self, device, x, &
            probabilities, status)
        class(random_forest_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "random forest device prediction: no resident CUDA tree ensemble is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest device prediction: device kind is invalid")
        end select
    end subroutine random_forest_classifier_predict_proba_device

    subroutine random_forest_classifier_predict(self, x, labels, status)
        class(random_forest_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest classifier predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_classifier_predict

    subroutine random_forest_classifier_predict_device(self, device, x, labels, status)
        class(random_forest_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest device label prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "random forest device label prediction: no resident CUDA tree ensemble is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest device label prediction: device kind is invalid")
        end select
    end subroutine random_forest_classifier_predict_device

    function random_forest_classifier_classes(self) result(classes)
        class(random_forest_classifier_t), intent(in) :: self
        integer, allocatable :: classes(:)
        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function random_forest_classifier_classes

    integer function random_forest_classifier_feature_count(self) result(count)
        class(random_forest_classifier_t), intent(in) :: self
        count = self%n_inputs
    end function random_forest_classifier_feature_count

    integer function random_forest_classifier_class_count(self) result(count)
        class(random_forest_classifier_t), intent(in) :: self
        count = self%n_classes
    end function random_forest_classifier_class_count

    integer function random_forest_classifier_tree_count(self) result(count)
        class(random_forest_classifier_t), intent(in) :: self
        count = self%n_trees
    end function random_forest_classifier_tree_count

    integer function random_forest_classifier_depth(self) result(depth)
        class(random_forest_classifier_t), intent(in) :: self
        depth = self%max_depth
    end function random_forest_classifier_depth

    integer function random_forest_classifier_criterion(self) result(criterion)
        class(random_forest_classifier_t), intent(in) :: self
        criterion = self%criterion_code
    end function random_forest_classifier_criterion

    integer function random_forest_classifier_seed(self) result(seed)
        class(random_forest_classifier_t), intent(in) :: self
        seed = self%seed
    end function random_forest_classifier_seed

    logical function random_forest_classifier_fitted(self) result(fitted)
        class(random_forest_classifier_t), intent(in) :: self
        fitted = self%initialized .and. allocated(self%trees) .and. &
            allocated(self%class_label) .and. allocated(self%bootstrap_included) .and. &
            self%n_samples > 0 .and. self%n_trees > 0
    end function random_forest_classifier_fitted

    real(dp) function random_forest_classifier_oob_coverage(self) result(coverage)
        class(random_forest_classifier_t), intent(in) :: self
        integer :: i

        coverage = 0.0_dp
        if (.not. allocated(self%bootstrap_included)) return
        if (self%n_samples < 1) return
        do i = 1, self%n_samples
            if (count(.not. self%bootstrap_included(i, :)) > 0) then
                coverage = coverage + 1.0_dp
            end if
        end do
        coverage = coverage/real(self%n_samples, dp)
    end function random_forest_classifier_oob_coverage

    function random_forest_classifier_bootstrap_inclusion(self) result(inclusion)
        class(random_forest_classifier_t), intent(in) :: self
        logical, allocatable :: inclusion(:, :)

        if (allocated(self%bootstrap_included)) then
            allocate(inclusion(size(self%bootstrap_included, 1), &
                size(self%bootstrap_included, 2)))
            inclusion = self%bootstrap_included
        else
            allocate(inclusion(0, 0))
        end if
    end function random_forest_classifier_bootstrap_inclusion

    logical function random_forest_classifier_device_supported(self, device_kind) &
            result(supported)
        class(random_forest_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%fitted()
        case default
            supported = .false.
        end select
    end function random_forest_classifier_device_supported

    subroutine random_forest_cuda_plan_create(self, model, device, status)
        class(random_forest_cuda_plan_t), intent(out) :: self
        class(random_forest_classifier_t), intent(in) :: model
        type(fortml_device_t), intent(in) :: device
        type(fortnum_status_t), intent(out) :: status

        self%abi_version = RANDOM_FOREST_CUDA_PLAN_ABI_VERSION
        self%n_inputs = 0
        self%n_classes = 0
        self%n_trees = 0
        self%device_index = -1
        self%initialized = .false.
        if (.not. model%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest CUDA plan: a fitted model is required")
            return
        end if
        if (.not. device%selected .or. .not. device%available .or. &
            device%kind /= FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest CUDA plan: a selected CUDA device is required")
            return
        end if
        self%n_inputs = model%feature_count()
        self%n_classes = model%class_count()
        self%n_trees = model%tree_count()
        self%device_index = device%device_index
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "random forest CUDA plan: resident ensemble kernel is not linked")
    end subroutine random_forest_cuda_plan_create

    subroutine random_forest_cuda_plan_destroy(self, status)
        class(random_forest_cuda_plan_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status

        self%abi_version = RANDOM_FOREST_CUDA_PLAN_ABI_VERSION
        self%n_inputs = 0
        self%n_classes = 0
        self%n_trees = 0
        self%device_index = -1
        self%initialized = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_cuda_plan_destroy

    subroutine random_forest_cuda_plan_predict_proba(self, x, probabilities, status)
        class(random_forest_cuda_plan_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        ! `probabilities` is intentionally INOUT: a refused device operation
        ! must not make callers lose a valid sentinel or previous result.
        if (.not. self%initialized) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "random forest CUDA plan: no resident ensemble is available")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "random forest CUDA plan: prediction kernel is not linked")
    end subroutine random_forest_cuda_plan_predict_proba

    subroutine random_forest_cuda_plan_predict(self, x, labels, status)
        class(random_forest_cuda_plan_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(inout) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "random forest CUDA plan: no resident ensemble is available")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "random forest CUDA plan: label kernel is not linked")
    end subroutine random_forest_cuda_plan_predict

    integer function random_forest_cuda_plan_abi(self) result(version)
        class(random_forest_cuda_plan_t), intent(in) :: self
        version = self%abi_version
    end function random_forest_cuda_plan_abi

    integer function random_forest_cuda_plan_feature_count(self) result(count)
        class(random_forest_cuda_plan_t), intent(in) :: self
        count = self%n_inputs
    end function random_forest_cuda_plan_feature_count

    integer function random_forest_cuda_plan_class_count(self) result(count)
        class(random_forest_cuda_plan_t), intent(in) :: self
        count = self%n_classes
    end function random_forest_cuda_plan_class_count

    integer function random_forest_cuda_plan_tree_count(self) result(count)
        class(random_forest_cuda_plan_t), intent(in) :: self
        count = self%n_trees
    end function random_forest_cuda_plan_tree_count

    integer function random_forest_cuda_plan_device(self) result(device_index)
        class(random_forest_cuda_plan_t), intent(in) :: self
        device_index = self%device_index
    end function random_forest_cuda_plan_device

    logical function random_forest_cuda_plan_fitted(self) result(fitted)
        class(random_forest_cuda_plan_t), intent(in) :: self
        fitted = self%initialized
    end function random_forest_cuda_plan_fitted

    subroutine bootstrap_indices(labels, classes, state, indices)
        integer, intent(in) :: labels(:), classes(:)
        integer(int64), intent(inout) :: state
        integer, intent(out) :: indices(:)
        integer :: i, j, class_index, n, class_size, target, seen

        n = size(labels)
        ! One deterministic draw from each class gives stable output columns
        ! without making the first row of every class permanently in-bag.
        do class_index = 1, size(classes)
            class_size = count(labels == classes(class_index))
            state = modulo(48271_int64*state, 2147483647_int64)
            target = 1 + int(modulo(state, int(class_size, int64)))
            seen = 0
            do j = 1, n
                if (labels(j) /= classes(class_index)) cycle
                seen = seen + 1
                if (seen == target) exit
            end do
            indices(class_index) = j
        end do
        do i = size(classes) + 1, n
            state = modulo(48271_int64*state, 2147483647_int64)
            indices(i) = 1 + int(modulo(state, int(n, int64)))
        end do
    end subroutine bootstrap_indices

    subroutine unique_sorted_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer :: i, j, count, temporary
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
        count = 1
        do i = 2, size(classes)
            if (classes(i) /= classes(count)) then
                count = count + 1
                classes(count) = classes(i)
            end if
        end do
        if (count < size(classes)) classes = classes(:count)
    end subroutine unique_sorted_labels

end module fortml_random_forest_classifier
