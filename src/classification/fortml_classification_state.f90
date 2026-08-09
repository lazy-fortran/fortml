module fortml_classification_state
    !! Transactional metadata shared by incremental classifiers.
    !!
    !! The state owns the sorted integer class vocabulary and the counters that
    !! describe a replayable stream of batches.  It deliberately contains no
    !! estimator parameters: callers can build a candidate estimator and only
    !! assign it to the live model after both fitting and this metadata update
    !! succeed.  This makes malformed ``partial_fit`` calls observationally
    !! atomic.
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    type, public :: classification_state_t
        private
        integer, allocatable :: class_label(:)
        integer :: n_features = 0
        integer :: n_samples = 0
        integer :: n_batches = 0
        logical :: is_initialized = .false.
    contains
        procedure, public :: initialize => classification_state_initialize
        procedure, public :: validate_batch => classification_state_validate_batch
        procedure, public :: append_batch => classification_state_append_batch
        procedure, public :: classes => classification_state_classes
        procedure, public :: class_count => classification_state_class_count
        procedure, public :: feature_count => classification_state_feature_count
        procedure, public :: sample_count => classification_state_sample_count
        procedure, public :: batch_count => classification_state_batch_count
        procedure, public :: initialized => classification_state_initialized
    end type classification_state_t

contains

    subroutine classification_state_initialize(self, classes, n_features, status)
        class(classification_state_t), intent(inout) :: self
        integer, intent(in) :: classes(:), n_features
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: candidate(:)
        integer :: i

        if (size(classes) < 2 .or. n_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classification state: at least two classes and one feature are required")
            return
        end if
        do i = 2, size(classes)
            if (classes(i) <= classes(i - 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "classification state: classes must be strictly increasing")
                return
            end if
        end do
        allocate(candidate(size(classes)))
        candidate = classes
        if (allocated(self%class_label)) deallocate(self%class_label)
        call move_alloc(candidate, self%class_label)
        self%n_features = n_features
        self%n_samples = 0
        self%n_batches = 0
        self%is_initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_state_initialize

    subroutine classification_state_validate_batch(self, labels, n_features, status)
        class(classification_state_t), intent(in) :: self
        integer, intent(in) :: labels(:), n_features
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j
        logical :: found

        if (.not. self%is_initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classification state: metadata is not initialized")
            return
        end if
        if (n_features /= self%n_features .or. size(labels) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classification state: batch feature count or size is invalid")
            return
        end if
        do i = 1, size(labels)
            found = .false.
            do j = 1, size(self%class_label)
                if (labels(i) == self%class_label(j)) then
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "classification state: batch contains an unseen class")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_state_validate_batch

    subroutine classification_state_append_batch(self, n_samples, status)
        class(classification_state_t), intent(inout) :: self
        integer, intent(in) :: n_samples
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_initialized .or. n_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classification state: batch append is invalid")
            return
        end if
        if (self%n_samples > huge(self%n_samples) - n_samples .or. &
            self%n_batches == huge(self%n_batches)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classification state: batch counters overflow")
            return
        end if
        self%n_samples = self%n_samples + n_samples
        self%n_batches = self%n_batches + 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine classification_state_append_batch

    function classification_state_classes(self) result(classes)
        class(classification_state_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function classification_state_classes

    integer function classification_state_class_count(self) result(count)
        class(classification_state_t), intent(in) :: self

        if (allocated(self%class_label)) then
            count = size(self%class_label)
        else
            count = 0
        end if
    end function classification_state_class_count

    integer function classification_state_feature_count(self) result(count)
        class(classification_state_t), intent(in) :: self

        count = self%n_features
    end function classification_state_feature_count

    integer function classification_state_sample_count(self) result(count)
        class(classification_state_t), intent(in) :: self

        count = self%n_samples
    end function classification_state_sample_count

    integer function classification_state_batch_count(self) result(count)
        class(classification_state_t), intent(in) :: self

        count = self%n_batches
    end function classification_state_batch_count

    logical function classification_state_initialized(self) result(value)
        class(classification_state_t), intent(in) :: self

        value = self%is_initialized
    end function classification_state_initialized

end module fortml_classification_state
