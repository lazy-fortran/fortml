module fortml_validation
    !! Deterministic train/test split iterators for estimator workflows.
    !!
    !! The iterators own only indices. They never inspect feature values, so
    !! callers can route the same folds through a fitted-transformer pipeline
    !! without leaking validation statistics. A split is returned as one-based
    !! indices in the original sample order, or in a seeded permutation when
    !! `shuffle` is enabled.
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_estimator_capabilities, only: estimator_capability_t
    implicit none
    private

    integer, parameter, public :: FORTML_SCORE_INPUT_LABELS = 1
    integer, parameter, public :: FORTML_SCORE_INPUT_DECISION = 2
    integer, parameter, public :: FORTML_SCORE_INPUT_PROBABILITY = 3

    integer, parameter, public :: FORTML_SCORE_ACCURACY = 1
    integer, parameter, public :: FORTML_SCORE_LOG_LOSS = 2
    integer, parameter, public :: FORTML_SCORE_R2 = 3
    integer, parameter, public :: FORTML_SCORE_CUSTOM = 4

    !> Metadata needed to route an estimator output to a validation scorer.
    !>
    !> The record is deliberately independent of a model implementation.  It
    !> describes which prediction representation a scorer consumes and whether
    !> larger raw scores are preferable.  `oriented_value` maps every scorer
    !> to a common maximize convention without changing the original metric.
    type, public :: estimator_score_metadata_t
        character(len=64) :: name = ""
        integer :: kind = 0
        integer :: input_kind = 0
        logical :: higher_is_better = .true.
        logical :: supports_sample_weight = .false.
        logical :: differentiable = .false.
    contains
        procedure, public :: initialize => score_metadata_initialize
        procedure, public :: valid => score_metadata_valid
        procedure, public :: oriented_value => score_metadata_oriented_value
        procedure, public :: prefer => score_metadata_prefer
    end type estimator_score_metadata_t

    !> Clone/reset and scoring metadata carried by a validation workflow.
    !>
    !> Fortran model types expose their own strongly typed clone/reset method;
    !> this value object records whether a generic search may invoke that
    !> protocol and the parameter topology expected by the scorer.  A false
    !> `cloneable` or `resettable` flag is a hard leakage guard, not a request
    !> to reuse a fitted model.
    type, public :: estimator_validation_metadata_t
        character(len=96) :: estimator_name = ""
        type(estimator_capability_t) :: capability
        type(estimator_score_metadata_t) :: score
        logical :: cloneable = .false.
        logical :: resettable = .false.
        integer :: parameter_count = 0
    contains
        procedure, public :: initialize => validation_metadata_initialize
        procedure, public :: valid => validation_metadata_valid
        procedure, public :: can_clone => validation_metadata_can_clone
        procedure, public :: can_reset => validation_metadata_can_reset
    end type estimator_validation_metadata_t

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

    type, public :: group_kfold_splitter_t
        !! Deterministic group-isolated K-fold iterator.
        !!
        !! Every sample carrying the same group identifier is assigned to one
        !! fold.  Groups are greedily packed from largest to smallest into
        !! the currently lightest fold, which keeps test-row counts balanced
        !! without ever splitting a group.  The optional seed only changes
        !! equal-size group tie ordering.
        private
        integer :: n_samples = 0
        integer :: n_groups = 0
        integer :: n_splits = 0
        integer :: next_fold = 1
        logical :: do_shuffle = .false.
        integer, allocatable :: fold_assignment(:)
    contains
        procedure, public :: initialize => group_initialize
        procedure, public :: reset => group_reset
        procedure, public :: next_split => group_next_split
        procedure, public :: sample_count => group_sample_count
        procedure, public :: group_count => group_group_count
        procedure, public :: fold_count => group_fold_count
        procedure, public :: shuffled => group_shuffled
    end type group_kfold_splitter_t

    type, public :: time_series_splitter_t
        !! Deterministic expanding or rolling-window time-series splitter.
        !!
        !! Test windows are contiguous and always occur strictly after the
        !! training window.  `gap` rows immediately before each test window
        !! are omitted from training.  A positive `max_train_size` turns the
        !! expanding window into a rolling window.  The splitter owns only
        !! integer bounds and is therefore an index-only CPU operation.
        private
        integer :: n_samples = 0
        integer :: n_splits = 0
        integer :: test_size = 0
        integer :: gap = 0
        integer :: max_train_size = 0
        integer :: initial_train_size = 0
        integer :: next_split_index = 1
    contains
        procedure, public :: initialize => time_series_initialize
        procedure, public :: reset => time_series_reset
        procedure, public :: next_split => time_series_next_split
        procedure, public :: sample_count => time_series_sample_count
        procedure, public :: fold_count => time_series_fold_count
        procedure, public :: test_window => time_series_test_window
        procedure, public :: gap_size => time_series_gap_size
        procedure, public :: rolling => time_series_rolling
    end type time_series_splitter_t

    public :: validate_estimator_capability
    public :: require_estimator_capability

contains

    subroutine score_metadata_initialize(self, name, input_kind, status, kind, &
            higher_is_better, supports_sample_weight, differentiable)
        class(estimator_score_metadata_t), intent(out) :: self
        character(*), intent(in) :: name
        integer, intent(in) :: input_kind
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: kind
        logical, intent(in), optional :: higher_is_better
        logical, intent(in), optional :: supports_sample_weight, differentiable

        self%name = ""
        self%kind = 0
        self%input_kind = 0
        self%higher_is_better = .true.
        self%supports_sample_weight = .false.
        self%differentiable = .false.
        if (len_trim(name) < 1 .or. len_trim(name) > len(self%name)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "score metadata: name is empty or too long")
            return
        end if
        if (input_kind < FORTML_SCORE_INPUT_LABELS .or. &
                input_kind > FORTML_SCORE_INPUT_PROBABILITY) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "score metadata: input kind is invalid")
            return
        end if
        self%name = trim(name)
        self%input_kind = input_kind
        if (present(kind)) self%kind = kind
        if (present(higher_is_better)) self%higher_is_better = higher_is_better
        if (present(supports_sample_weight)) then
            self%supports_sample_weight = supports_sample_weight
        end if
        if (present(differentiable)) self%differentiable = differentiable
        if (self%kind < FORTML_SCORE_ACCURACY .or. &
                self%kind > FORTML_SCORE_CUSTOM) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "score metadata: scorer kind is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine score_metadata_initialize

    logical function score_metadata_valid(self) result(valid)
        class(estimator_score_metadata_t), intent(in) :: self

        valid = len_trim(self%name) > 0 .and. len_trim(self%name) <= len(self%name) &
            .and. self%kind >= FORTML_SCORE_ACCURACY .and. &
            self%kind <= FORTML_SCORE_CUSTOM .and. &
            self%input_kind >= FORTML_SCORE_INPUT_LABELS .and. &
            self%input_kind <= FORTML_SCORE_INPUT_PROBABILITY
    end function score_metadata_valid

    real(real64) function score_metadata_oriented_value(self, raw_value) result(value)
        class(estimator_score_metadata_t), intent(in) :: self
        real(real64), intent(in) :: raw_value

        if (self%higher_is_better) then
            value = raw_value
        else
            value = -raw_value
        end if
    end function score_metadata_oriented_value

    logical function score_metadata_prefer(self, candidate, incumbent) result(value)
        class(estimator_score_metadata_t), intent(in) :: self
        real(real64), intent(in) :: candidate, incumbent

        if (candidate /= candidate .or. incumbent /= incumbent) then
            value = .false.
            return
        end if
        value = self%oriented_value(candidate) > self%oriented_value(incumbent)
    end function score_metadata_prefer

    subroutine validation_metadata_initialize(self, estimator_name, capability, &
            score, status, cloneable, resettable, parameter_count)
        class(estimator_validation_metadata_t), intent(out) :: self
        character(*), intent(in) :: estimator_name
        type(estimator_capability_t), intent(in) :: capability
        type(estimator_score_metadata_t), intent(in) :: score
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: cloneable, resettable
        integer, intent(in), optional :: parameter_count

        self%estimator_name = ""
        self%capability = capability
        self%score = score
        self%cloneable = .false.
        self%resettable = .false.
        self%parameter_count = 0
        if (len_trim(estimator_name) < 1 .or. &
                len_trim(estimator_name) > len(self%estimator_name)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "validation metadata: estimator name is empty or too long")
            return
        end if
        if (.not. capability%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "validation metadata: capability is invalid")
            return
        end if
        if (.not. score%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "validation metadata: scorer is invalid")
            return
        end if
        if (present(cloneable)) self%cloneable = cloneable
        if (present(resettable)) self%resettable = resettable
        if (present(parameter_count)) self%parameter_count = parameter_count
        if (self%parameter_count < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "validation metadata: parameter count must be nonnegative")
            return
        end if
        self%estimator_name = trim(estimator_name)
        call status_set(status, FORTNUM_OK, "")
    end subroutine validation_metadata_initialize

    logical function validation_metadata_valid(self) result(valid)
        class(estimator_validation_metadata_t), intent(in) :: self

        valid = len_trim(self%estimator_name) > 0 .and. &
            len_trim(self%estimator_name) <= len(self%estimator_name) .and. &
            self%capability%valid() .and. self%score%valid() .and. &
            self%parameter_count >= 0
        if (.not. valid) return
        valid = trim(self%estimator_name) == trim(self%capability%name)
    end function validation_metadata_valid

    logical function validation_metadata_can_clone(self) result(value)
        class(estimator_validation_metadata_t), intent(in) :: self

        value = self%valid() .and. self%cloneable
    end function validation_metadata_can_clone

    logical function validation_metadata_can_reset(self) result(value)
        class(estimator_validation_metadata_t), intent(in) :: self

        value = self%valid() .and. self%resettable
    end function validation_metadata_can_reset

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

    subroutine time_series_initialize(self, n_samples, n_splits, status, &
            test_size, gap, max_train_size)
        class(time_series_splitter_t), intent(out) :: self
        integer, intent(in) :: n_samples, n_splits
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: test_size, gap, max_train_size
        integer :: requested_test_size, requested_gap, requested_max_train

        self%n_samples = 0
        self%n_splits = 0
        self%test_size = 0
        self%gap = 0
        self%max_train_size = 0
        self%initial_train_size = 0
        self%next_split_index = 1
        if (n_samples < 2 .or. n_splits < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "time-series split: sample and split counts are incompatible")
            return
        end if
        requested_test_size = n_samples/(n_splits + 1)
        if (present(test_size)) requested_test_size = test_size
        requested_gap = 0
        if (present(gap)) requested_gap = gap
        requested_max_train = 0
        if (present(max_train_size)) requested_max_train = max_train_size
        if (requested_test_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "time-series split: test_size must be positive")
            return
        end if
        if (requested_gap < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "time-series split: gap must be nonnegative")
            return
        end if
        if (requested_max_train < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "time-series split: max_train_size must be nonnegative")
            return
        end if
        self%initial_train_size = n_samples - n_splits*requested_test_size - &
            requested_gap
        if (self%initial_train_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "time-series split: not enough samples for requested windows")
            return
        end if
        self%n_samples = n_samples
        self%n_splits = n_splits
        self%test_size = requested_test_size
        self%gap = requested_gap
        self%max_train_size = requested_max_train
        self%next_split_index = 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine time_series_initialize

    subroutine time_series_reset(self)
        class(time_series_splitter_t), intent(inout) :: self

        self%next_split_index = 1
    end subroutine time_series_reset

    subroutine time_series_next_split(self, train_indices, test_indices, has_split, &
            status)
        class(time_series_splitter_t), intent(inout) :: self
        integer, allocatable, intent(out) :: train_indices(:), test_indices(:)
        logical, intent(out) :: has_split
        type(fortnum_status_t), intent(out) :: status
        integer :: test_start, test_end, train_start, train_end
        integer :: train_count, test_count, i

        has_split = .false.
        allocate(train_indices(0), test_indices(0))
        if (self%n_samples < 2 .or. self%n_splits < 2 .or. &
                self%test_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "time-series split: splitter is not initialized")
            return
        end if
        if (self%next_split_index > self%n_splits) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        test_start = self%initial_train_size + self%gap + &
            (self%next_split_index - 1)*self%test_size + 1
        test_end = test_start + self%test_size - 1
        train_end = test_start - self%gap - 1
        train_start = 1
        if (self%max_train_size > 0) then
            train_start = max(1, train_end - self%max_train_size + 1)
        end if
        train_count = train_end - train_start + 1
        test_count = test_end - test_start + 1
        if (train_count < 1 .or. test_count < 1 .or. test_end > self%n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "time-series split: generated window is out of bounds")
            return
        end if
        deallocate(train_indices, test_indices)
        allocate(train_indices(train_count), test_indices(test_count))
        train_indices = [(i, i=train_start, train_end)]
        test_indices = [(i, i=test_start, test_end)]
        self%next_split_index = self%next_split_index + 1
        has_split = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine time_series_next_split

    integer function time_series_sample_count(self) result(count)
        class(time_series_splitter_t), intent(in) :: self

        count = self%n_samples
    end function time_series_sample_count

    integer function time_series_fold_count(self) result(count)
        class(time_series_splitter_t), intent(in) :: self

        count = self%n_splits
    end function time_series_fold_count

    integer function time_series_test_window(self) result(count)
        class(time_series_splitter_t), intent(in) :: self

        count = self%test_size
    end function time_series_test_window

    integer function time_series_gap_size(self) result(count)
        class(time_series_splitter_t), intent(in) :: self

        count = self%gap
    end function time_series_gap_size

    logical function time_series_rolling(self) result(value)
        class(time_series_splitter_t), intent(in) :: self

        value = self%max_train_size > 0
    end function time_series_rolling

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

    subroutine group_initialize(self, groups, n_splits, status, shuffle, seed)
        class(group_kfold_splitter_t), intent(out) :: self
        integer, intent(in) :: groups(:), n_splits
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: shuffle
        integer, intent(in), optional :: seed
        integer, allocatable :: unique_groups(:), group_counts(:)
        integer, allocatable :: first_index(:), order(:), tie_rank(:)
        integer, allocatable :: fold_sizes(:)
        integer :: i, j, k, n_unique, random_seed, group_id, best_fold
        integer :: temporary, best_order

        if (size(groups) < 2 .or. n_splits < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "group kfold: sample and fold counts are incompatible")
            return
        end if
        self%n_samples = size(groups)
        self%n_splits = n_splits
        self%next_fold = 1
        self%do_shuffle = .false.
        if (present(shuffle)) self%do_shuffle = shuffle
        random_seed = 17
        if (present(seed)) random_seed = seed
        if (self%do_shuffle .and. random_seed <= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "group kfold: shuffled splits require a positive seed")
            return
        end if

        allocate(unique_groups(size(groups)), group_counts(size(groups)), &
            first_index(size(groups)))
        n_unique = 0
        do i = 1, size(groups)
            group_id = 0
            do j = 1, n_unique
                if (unique_groups(j) == groups(i)) then
                    group_id = j
                    exit
                end if
            end do
            if (group_id == 0) then
                n_unique = n_unique + 1
                unique_groups(n_unique) = groups(i)
                group_counts(n_unique) = 1
                first_index(n_unique) = i
            else
                group_counts(group_id) = group_counts(group_id) + 1
            end if
        end do
        if (n_splits > n_unique) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "group kfold: n_splits cannot exceed the number of groups")
            return
        end if

        self%n_groups = n_unique
        allocate(self%fold_assignment(size(groups)), order(n_unique), &
            tie_rank(n_unique), fold_sizes(n_splits))
        order = [(i, i=1, n_unique)]
        tie_rank = first_index(:n_unique)
        if (self%do_shuffle) then
            call shuffle_indices(order, random_seed)
            do i = 1, n_unique
                tie_rank(order(i)) = i
            end do
        end if

        ! Stable descending-size ordering, with first occurrence (or the
        ! seeded permutation rank) as the deterministic tie breaker.
        do i = 1, n_unique - 1
            best_order = i
            do j = i + 1, n_unique
                if (group_counts(order(j)) > group_counts(order(best_order))) then
                    best_order = j
                else if (group_counts(order(j)) == group_counts(order(best_order))) then
                    if (tie_rank(order(j)) < tie_rank(order(best_order))) then
                        best_order = j
                    end if
                end if
            end do
            if (best_order /= i) then
                temporary = order(i)
                order(i) = order(best_order)
                order(best_order) = temporary
            end if
        end do

        fold_sizes = 0
        self%fold_assignment = 0
        do i = 1, n_unique
            group_id = order(i)
            best_fold = 1
            do k = 2, n_splits
                if (fold_sizes(k) < fold_sizes(best_fold)) best_fold = k
            end do
            do j = 1, size(groups)
                if (groups(j) == unique_groups(group_id)) then
                    self%fold_assignment(j) = best_fold
                end if
            end do
            fold_sizes(best_fold) = fold_sizes(best_fold) + group_counts(group_id)
        end do
        deallocate(unique_groups, group_counts, first_index, order, tie_rank, &
            fold_sizes)
        call status_set(status, FORTNUM_OK, "")
    end subroutine group_initialize

    subroutine group_reset(self)
        class(group_kfold_splitter_t), intent(inout) :: self

        self%next_fold = 1
    end subroutine group_reset

    subroutine group_next_split(self, train_indices, test_indices, has_split, &
            status)
        class(group_kfold_splitter_t), intent(inout) :: self
        integer, allocatable, intent(out) :: train_indices(:), test_indices(:)
        logical, intent(out) :: has_split
        type(fortnum_status_t), intent(out) :: status
        integer :: fold, test_count, train_count, i, train_position, test_position

        has_split = .false.
        allocate(train_indices(0), test_indices(0))
        if (.not. allocated(self%fold_assignment)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "group kfold: splitter is not initialized")
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
    end subroutine group_next_split

    integer function group_sample_count(self) result(count)
        class(group_kfold_splitter_t), intent(in) :: self

        count = self%n_samples
    end function group_sample_count

    integer function group_group_count(self) result(count)
        class(group_kfold_splitter_t), intent(in) :: self

        count = self%n_groups
    end function group_group_count

    integer function group_fold_count(self) result(count)
        class(group_kfold_splitter_t), intent(in) :: self

        count = self%n_splits
    end function group_fold_count

    logical function group_shuffled(self) result(value)
        class(group_kfold_splitter_t), intent(in) :: self

        value = self%do_shuffle
    end function group_shuffled

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
