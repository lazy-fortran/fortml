module fortml_radius_neighbors_classifier
    !! Dense radius-neighbor classification with deterministic CPU semantics.
    !!
    !! Training rows are retained verbatim.  A query selects every row whose
    !! squared Euclidean distance is at most ``radius**2``; ties therefore do
    !! not depend on a requested neighbor count.  Uniform and inverse-distance
    !! votes, sample weights, arbitrary integer labels, and an optional
    !! in-training outlier label are supported.  Neighbor selection is a
    !! discontinuous operation, so input JVP/VJP products return a typed
    !! refusal.  CUDA is deliberately a refusal until a resident radius
    !! search kernel is linked; no hidden host fallback is permitted.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    integer, parameter, public :: RADIUS_WEIGHTS_UNIFORM = 1
    integer, parameter, public :: RADIUS_WEIGHTS_DISTANCE = 2

    type, public :: radius_neighbors_classifier_t
        private
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: sample_weight(:)
        integer, allocatable :: train_class(:)
        integer, allocatable :: class_label(:)
        real(dp) :: radius_value = 0.0_dp
        integer :: weighting_code = RADIUS_WEIGHTS_UNIFORM
        integer :: outlier_class = 0
        integer :: n_features = 0
        integer :: n_samples = 0
        integer :: n_classes = 0
        logical :: has_outlier = .false.
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => radius_neighbors_fit
        procedure, public :: predict_proba => radius_neighbors_predict_proba
        procedure, public :: predict => radius_neighbors_predict
        procedure, public :: predict_device => radius_neighbors_predict_device
        procedure, public :: predict_proba_jvp => radius_neighbors_predict_proba_jvp
        procedure, public :: predict_proba_vjp => radius_neighbors_predict_proba_vjp
        procedure, public :: device_supported => radius_neighbors_device_supported
        procedure, public :: classes => radius_neighbors_classes
        procedure, public :: radius => radius_neighbors_radius
        procedure, public :: weighting => radius_neighbors_weighting
        procedure, public :: feature_count => radius_neighbors_feature_count
        procedure, public :: sample_count => radius_neighbors_sample_count
        procedure, public :: class_count => radius_neighbors_class_count
        procedure, public :: fitted => radius_neighbors_fitted
    end type radius_neighbors_classifier_t

    public :: radius_neighbors_fit
    public :: radius_neighbors_predict_proba
    public :: radius_neighbors_predict
    public :: radius_neighbors_predict_device

contains

    subroutine radius_neighbors_fit(self, x, labels, status, radius, weights, &
            sample_weight, outlier_label)
        class(radius_neighbors_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: radius, sample_weight(:)
        integer, intent(in), optional :: weights, outlier_label
        integer, allocatable :: classes(:)
        integer :: i, requested_weight, requested_outlier
        real(dp) :: requested_radius, total_weight

        self%is_fitted = .false.
        requested_radius = 1.0_dp
        if (present(radius)) requested_radius = radius
        requested_weight = RADIUS_WEIGHTS_UNIFORM
        if (present(weights)) requested_weight = weights
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. .not. ieee_is_finite(requested_radius) .or. &
            requested_radius <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor fit: inputs and radius must be finite and positive")
            return
        end if
        if (requested_weight /= RADIUS_WEIGHTS_UNIFORM .and. &
            requested_weight /= RADIUS_WEIGHTS_DISTANCE) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor fit: weights must be uniform or distance")
            return
        end if
        call sorted_unique_labels(labels, classes)
        allocate(self%x_train(size(x, 1), size(x, 2)), &
            self%sample_weight(size(x, 1)), self%train_class(size(x, 1)), &
            self%class_label(size(classes)))
        self%x_train = x
        self%class_label = classes
        do i = 1, size(labels)
            self%train_class(i) = class_index(classes, labels(i))
        end do
        self%sample_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor fit: sample weights must be finite and nonnegative")
                return
            end if
            total_weight = sum(sample_weight)
            if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor fit: sample weights need positive total mass")
                return
            end if
            self%sample_weight = sample_weight
        end if
        self%radius_value = requested_radius
        self%weighting_code = requested_weight
        self%has_outlier = .false.
        self%outlier_class = 0
        if (present(outlier_label)) then
            self%outlier_class = class_index(classes, outlier_label)
            if (self%outlier_class == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor fit: outlier_label must be a fitted class")
                return
            end if
            self%has_outlier = .true.
        end if
        self%n_features = size(x, 2)
        self%n_samples = size(x, 1)
        self%n_classes = size(classes)
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine radius_neighbors_fit

    subroutine radius_neighbors_predict_proba(self, x, probabilities, status)
        class(radius_neighbors_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: d2, delta, denominator, weight, radius_squared
        real(dp), allocatable :: scores(:)
        integer :: i, j, k, class_index_value, selected
        logical :: exact_neighbor

        probabilities = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor predict_proba: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor predict_proba: input or output shape is invalid")
            return
        end if
        allocate(scores(self%n_classes))
        radius_squared = self%radius_value*self%radius_value
        do i = 1, size(x, 1)
            scores = 0.0_dp
            selected = 0
            exact_neighbor = .false.
            do j = 1, self%n_samples
                d2 = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                end do
                if (.not. ieee_is_finite(d2)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "radius-neighbor predict_proba: distance overflow")
                    return
                end if
                if (d2 <= radius_squared) then
                    selected = selected + 1
                    if (d2 == 0.0_dp) exact_neighbor = .true.
                end if
            end do
            if (selected == 0) then
                if (.not. self%has_outlier) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "radius-neighbor predict_proba: query has no neighbors")
                    return
                end if
                probabilities(i, self%outlier_class) = 1.0_dp
                cycle
            end if
            do j = 1, self%n_samples
                d2 = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                end do
                if (d2 <= radius_squared) then
                    if (self%weighting_code == RADIUS_WEIGHTS_UNIFORM) then
                        weight = self%sample_weight(j)
                    else if (exact_neighbor) then
                        weight = 0.0_dp
                        if (d2 == 0.0_dp) weight = self%sample_weight(j)
                    else
                        weight = self%sample_weight(j)/sqrt(d2)
                    end if
                    class_index_value = self%train_class(j)
                    scores(class_index_value) = scores(class_index_value) + weight
                end if
            end do
            denominator = sum(scores)
            if (.not. ieee_is_finite(denominator) .or. denominator <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor predict_proba: selected neighbors have zero weight")
                return
            end if
            probabilities(i, :) = scores/denominator
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine radius_neighbors_predict_proba

    subroutine radius_neighbors_predict(self, x, labels, status)
        class(radius_neighbors_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i
        if (.not. self%is_fitted .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor predict: model or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine radius_neighbors_predict

    subroutine radius_neighbors_predict_device(self, device, x, labels, status)
        class(radius_neighbors_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "radius-neighbor device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor device prediction: device kind is invalid")
        end select
    end subroutine radius_neighbors_predict_device

    subroutine radius_neighbors_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(radius_neighbors_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "radius-neighbor input JVP is undefined across selection boundaries")
    end subroutine radius_neighbors_predict_proba_jvp

    subroutine radius_neighbors_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(radius_neighbors_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        x_bar = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "radius-neighbor input VJP is undefined across selection boundaries")
    end subroutine radius_neighbors_predict_proba_vjp

    logical function radius_neighbors_device_supported(self, device_kind) result(supported)
        class(radius_neighbors_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function radius_neighbors_device_supported

    function radius_neighbors_classes(self) result(values)
        class(radius_neighbors_classifier_t), intent(in) :: self
        integer, allocatable :: values(:)
        if (allocated(self%class_label)) then
            values = self%class_label
        else
            allocate(values(0))
        end if
    end function radius_neighbors_classes

    real(dp) function radius_neighbors_radius(self) result(value)
        class(radius_neighbors_classifier_t), intent(in) :: self
        value = self%radius_value
    end function radius_neighbors_radius

    integer function radius_neighbors_weighting(self) result(value)
        class(radius_neighbors_classifier_t), intent(in) :: self
        value = self%weighting_code
    end function radius_neighbors_weighting

    integer function radius_neighbors_feature_count(self) result(value)
        class(radius_neighbors_classifier_t), intent(in) :: self
        value = self%n_features
    end function radius_neighbors_feature_count

    integer function radius_neighbors_sample_count(self) result(value)
        class(radius_neighbors_classifier_t), intent(in) :: self
        value = self%n_samples
    end function radius_neighbors_sample_count

    integer function radius_neighbors_class_count(self) result(value)
        class(radius_neighbors_classifier_t), intent(in) :: self
        value = self%n_classes
    end function radius_neighbors_class_count

    logical function radius_neighbors_fitted(self) result(value)
        class(radius_neighbors_classifier_t), intent(in) :: self
        value = self%is_fitted
    end function radius_neighbors_fitted

    integer function class_index(classes, label) result(index)
        integer, intent(in) :: classes(:), label
        integer :: i
        index = 0
        do i = 1, size(classes)
            if (classes(i) == label) then
                index = i
                return
            end if
        end do
    end function class_index

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, count, value
        allocate(work, source=labels)
        do i = 2, size(work)
            value = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= value) exit
                work(j + 1) = work(j)
                j = j - 1
            end do
            work(j + 1) = value
        end do
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(i - 1)) count = count + 1
        end do
        allocate(classes(count))
        classes(1) = work(1)
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(i - 1)) then
                count = count + 1
                classes(count) = work(i)
            end if
        end do
    end subroutine sorted_unique_labels

end module fortml_radius_neighbors_classifier
