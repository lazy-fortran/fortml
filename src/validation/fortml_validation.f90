module fortml_validation
    !! Deterministic train/test split iterators for estimator workflows.
    !!
    !! The iterators own only indices. They never inspect feature values, so
    !! callers can route the same folds through a fitted-transformer pipeline
    !! without leaking validation statistics. A split is returned as one-based
    !! indices in the original sample order, or in a seeded permutation when
    !! `shuffle` is enabled.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_estimator_capabilities, only: estimator_capability_t
    implicit none
    private

    type, public :: kfold_splitter_t
        private
        integer :: n_samples = 0
        integer :: n_splits = 0
        integer :: next_fold = 1
        logical :: do_shuffle = .false.
        integer, allocatable :: order(:)
    contains
        procedure, public :: initialize => kfold_initialize
        procedure, public :: reset => kfold_reset
        procedure, public :: next_split => kfold_next_split
        procedure, public :: sample_count => kfold_sample_count
        procedure, public :: fold_count => kfold_fold_count
        procedure, public :: shuffled => kfold_shuffled
    end type kfold_splitter_t

    type, public :: stratified_kfold_splitter_t
        private
        integer :: n_samples = 0
        integer :: n_splits = 0
        integer :: next_fold = 1
        logical :: do_shuffle = .false.
        integer, allocatable :: fold_assignment(:)
    contains
        procedure, public :: initialize => stratified_initialize
        procedure, public :: reset => stratified_reset
        procedure, public :: next_split => stratified_next_split
        procedure, public :: sample_count => stratified_sample_count
        procedure, public :: fold_count => stratified_fold_count
        procedure, public :: shuffled => stratified_shuffled
    end type stratified_kfold_splitter_t

    public :: validate_estimator_capability
    public :: require_estimator_capability

contains

    !> Validate a model capability record in the same status style as splits.
    subroutine validate_estimator_capability(capability, status)
        type(estimator_capability_t), intent(in) :: capability
        type(fortnum_status_t), intent(out) :: status

        call capability%validate(status)
    end subroutine validate_estimator_capability

    !> Check a requested capability before entering a validation workflow.
    subroutine require_estimator_capability(capability, requirement, status)
        type(estimator_capability_t), intent(in) :: capability, requirement
        type(fortnum_status_t), intent(out) :: status

        call capability%require(requirement, status)
    end subroutine require_estimator_capability

    subroutine kfold_initialize(self, n_samples, n_splits, status, shuffle, seed)
        class(kfold_splitter_t), intent(out) :: self
        integer, intent(in) :: n_samples, n_splits
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: shuffle
        integer, intent(in), optional :: seed
        integer :: random_seed, i

        if (n_samples < 2 .or. n_splits < 2 .or. n_splits > n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kfold: sample and fold counts are incompatible")
            return
        end if
        self%n_samples = n_samples
        self%n_splits = n_splits
        self%next_fold = 1
        self%do_shuffle = .false.
        if (present(shuffle)) self%do_shuffle = shuffle
        random_seed = 17
        if (present(seed)) random_seed = seed
        if (self%do_shuffle .and. random_seed <= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kfold: shuffled splits require a positive seed")
            return
        end if
        allocate(self%order(n_samples))
        self%order = [(i, i=1, n_samples)]
        if (self%do_shuffle) call shuffle_indices(self%order, random_seed)
        call status_set(status, FORTNUM_OK, "")
    end subroutine kfold_initialize

    subroutine kfold_reset(self)
        class(kfold_splitter_t), intent(inout) :: self

        self%next_fold = 1
    end subroutine kfold_reset

    subroutine kfold_next_split(self, train_indices, test_indices, has_split, &
            status)
        class(kfold_splitter_t), intent(inout) :: self
        integer, allocatable, intent(out) :: train_indices(:), test_indices(:)
        logical, intent(out) :: has_split
        type(fortnum_status_t), intent(out) :: status
        integer :: fold, base_size, remainder, test_start, test_count
        integer :: position, train_position, test_position

        has_split = .false.
        allocate(train_indices(0), test_indices(0))
        if (.not. allocated(self%order)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kfold: splitter is not initialized")
            return
        end if
        if (self%next_fold > self%n_splits) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        fold = self%next_fold
        base_size = self%n_samples/self%n_splits
        remainder = mod(self%n_samples, self%n_splits)
        test_count = base_size
        if (fold <= remainder) test_count = test_count + 1
        test_start = (fold - 1)*base_size + min(fold - 1, remainder) + 1
        deallocate(train_indices, test_indices)
        allocate(train_indices(self%n_samples - test_count), test_indices(test_count))
        train_position = 0
        test_position = 0
        do position = 1, self%n_samples
            if (position >= test_start .and. position < test_start + test_count) then
                test_position = test_position + 1
                test_indices(test_position) = self%order(position)
            else
                train_position = train_position + 1
                train_indices(train_position) = self%order(position)
            end if
        end do
        self%next_fold = self%next_fold + 1
        has_split = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine kfold_next_split

    integer function kfold_sample_count(self) result(count)
        class(kfold_splitter_t), intent(in) :: self

        count = self%n_samples
    end function kfold_sample_count

    integer function kfold_fold_count(self) result(count)
        class(kfold_splitter_t), intent(in) :: self

        count = self%n_splits
    end function kfold_fold_count

    logical function kfold_shuffled(self) result(value)
        class(kfold_splitter_t), intent(in) :: self

        value = self%do_shuffle
    end function kfold_shuffled

    subroutine stratified_initialize(self, labels, n_splits, status, shuffle, seed)
        class(stratified_kfold_splitter_t), intent(out) :: self
        integer, intent(in) :: labels(:), n_splits
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: shuffle
        integer, intent(in), optional :: seed
        integer, allocatable :: order(:), class_seen(:)
        integer :: i, j, class_position, rank, random_seed

        if (size(labels) < 2 .or. n_splits < 2 .or. n_splits > size(labels)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "stratified kfold: sample and fold counts are incompatible")
            return
        end if
        self%n_samples = size(labels)
        self%n_splits = n_splits
        self%next_fold = 1
        self%do_shuffle = .false.
        if (present(shuffle)) self%do_shuffle = shuffle
        random_seed = 17
        if (present(seed)) random_seed = seed
        if (self%do_shuffle .and. random_seed <= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "stratified kfold: shuffled splits require a positive seed")
            return
        end if
        allocate(order(size(labels)), self%fold_assignment(size(labels)))
        order = [(i, i=1, size(labels))]
        if (self%do_shuffle) call shuffle_indices(order, random_seed)
        self%fold_assignment = 0
        allocate(class_seen(size(labels)))
        class_seen = 0
        class_position = 0
        do i = 1, size(labels)
            if (class_position > 0) then
                if (any(class_seen(:class_position) == labels(order(i)))) cycle
            end if
            class_position = class_position + 1
            class_seen(class_position) = labels(order(i))
            rank = 0
            do j = 1, size(labels)
                if (labels(order(j)) == labels(order(i))) then
                    rank = rank + 1
                    self%fold_assignment(order(j)) = 1 + mod(rank - 1, n_splits)
                end if
            end do
        end do
        deallocate(order, class_seen)
        call status_set(status, FORTNUM_OK, "")
    end subroutine stratified_initialize

    subroutine stratified_reset(self)
        class(stratified_kfold_splitter_t), intent(inout) :: self

        self%next_fold = 1
    end subroutine stratified_reset

    subroutine stratified_next_split(self, train_indices, test_indices, has_split, &
            status)
        class(stratified_kfold_splitter_t), intent(inout) :: self
        integer, allocatable, intent(out) :: train_indices(:), test_indices(:)
        logical, intent(out) :: has_split
        type(fortnum_status_t), intent(out) :: status
        integer :: fold, test_count, train_count, i, train_position, test_position

        has_split = .false.
        allocate(train_indices(0), test_indices(0))
        if (.not. allocated(self%fold_assignment)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "stratified kfold: splitter is not initialized")
            return
        end if
        if (self%next_fold > self%n_splits) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        fold = self%next_fold
        test_count = count(self%fold_assignment == fold)
        train_count = self%n_samples - test_count
        deallocate(train_indices, test_indices)
        allocate(train_indices(train_count), test_indices(test_count))
        train_position = 0
        test_position = 0
        do i = 1, self%n_samples
            if (self%fold_assignment(i) == fold) then
                test_position = test_position + 1
                test_indices(test_position) = i
            else
                train_position = train_position + 1
                train_indices(train_position) = i
            end if
        end do
        self%next_fold = self%next_fold + 1
        has_split = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine stratified_next_split

    integer function stratified_sample_count(self) result(count)
        class(stratified_kfold_splitter_t), intent(in) :: self

        count = self%n_samples
    end function stratified_sample_count

    integer function stratified_fold_count(self) result(count)
        class(stratified_kfold_splitter_t), intent(in) :: self

        count = self%n_splits
    end function stratified_fold_count

    logical function stratified_shuffled(self) result(value)
        class(stratified_kfold_splitter_t), intent(in) :: self

        value = self%do_shuffle
    end function stratified_shuffled

    subroutine shuffle_indices(indices, seed)
        integer, intent(inout) :: indices(:)
        integer, intent(in) :: seed
        integer(int64) :: state
        integer :: i, j, temporary

        state = int(seed, int64)
        do i = size(indices), 2, -1
            state = mod(1103515245_int64*state + 12345_int64, &
                2147483647_int64)
            j = 1 + int(mod(state, int(i, int64)))
            temporary = indices(i)
            indices(i) = indices(j)
            indices(j) = temporary
        end do
    end subroutine shuffle_indices

end module fortml_validation
