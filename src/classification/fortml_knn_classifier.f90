module fortml_knn_classifier
    !! Deterministic dense k-nearest-neighbor classification.
    !!
    !! Samples are rows and features are columns.  The estimator uses squared
    !! Euclidean distance and stores the training rows verbatim.  Neighbor
    !! order is stable: smaller distance wins, and an exact distance tie is
    !! resolved by the original training-row index.  Class columns are sorted
    !! by their integer labels, so ties in a vote resolve to the smallest
    !! label.  Uniform and inverse-distance votes are supported.
    !!
    !! Neighbor selection is discrete.  Consequently input JVP and VJP
    !! products are explicitly refused rather than returning a misleading
    !! zero derivative.  The fixed-state prediction itself is deterministic
    !! and suitable for use inside a piecewise-constant objective.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    integer, parameter, public :: KNN_WEIGHTS_UNIFORM = 1
    integer, parameter, public :: KNN_WEIGHTS_DISTANCE = 2
    ! Singular aliases make the option names convenient in code that uses the
    ! singular sklearn-style spelling while retaining the documented names.
    integer, parameter, public :: KNN_WEIGHT_UNIFORM = KNN_WEIGHTS_UNIFORM
    integer, parameter, public :: KNN_WEIGHT_DISTANCE = KNN_WEIGHTS_DISTANCE

    type, public :: knn_classifier_t
        private
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: sample_weight(:)
        integer, allocatable :: train_class(:)
        integer, allocatable :: class_label(:)
        integer :: n_neighbors = 0
        integer :: n_features = 0
        integer :: n_samples = 0
        integer :: n_classes = 0
        integer :: weighting_code = KNN_WEIGHTS_UNIFORM
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => knn_classifier_fit
        procedure, public :: predict_proba => knn_classifier_predict_proba
        procedure, public :: predict => knn_classifier_predict
        procedure, public :: predict_proba_jvp => knn_classifier_predict_proba_jvp
        procedure, public :: predict_proba_vjp => knn_classifier_predict_proba_vjp
        procedure, public :: classes => knn_classifier_classes
        procedure, public :: n_neighbors_value => knn_classifier_n_neighbors
        procedure, public :: weighting => knn_classifier_weighting
        procedure, public :: feature_count => knn_classifier_feature_count
        procedure, public :: sample_count => knn_classifier_sample_count
        procedure, public :: class_count => knn_classifier_class_count
        procedure, public :: fitted => knn_classifier_fitted
    end type knn_classifier_t

    public :: knn_classifier_fit
    public :: knn_classifier_predict_proba
    public :: knn_classifier_predict

contains

    subroutine knn_classifier_fit(self, x, labels, status, n_neighbors, weights, &
            sample_weight)
        class(knn_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_neighbors, weights
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: classes(:)
        integer :: requested_neighbors, requested_weights, i
        real(dp) :: total_weight

        self%is_fitted = .false.
        requested_neighbors = 5
        if (present(n_neighbors)) requested_neighbors = n_neighbors
        requested_weights = KNN_WEIGHTS_UNIFORM
        if (present(weights)) requested_weights = weights

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier fit: input dimensions are invalid")
            return
        end if
        if (requested_neighbors < 1 .or. requested_neighbors > size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier fit: n_neighbors must be in [1,n_samples]")
            return
        end if
        if (requested_weights /= KNN_WEIGHTS_UNIFORM .and. &
            requested_weights /= KNN_WEIGHTS_DISTANCE) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier fit: weights must be uniform or distance")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier fit: inputs must be finite")
            return
        end if
        call sorted_unique_labels(labels, classes)
        if (size(classes) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier fit: at least one class is required")
            return
        end if

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
            if (size(sample_weight) /= size(x, 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "KNN classifier fit: sample-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "KNN classifier fit: sample weights must be finite and nonnegative")
                return
            end if
            total_weight = sum(sample_weight)
            if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "KNN classifier fit: sample weights need positive total mass")
                return
            end if
            self%sample_weight = sample_weight
        end if

        self%n_neighbors = requested_neighbors
        self%n_features = size(x, 2)
        self%n_samples = size(x, 1)
        self%n_classes = size(classes)
        self%weighting_code = requested_weights
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine knn_classifier_fit

    subroutine knn_classifier_predict_proba(self, x, probabilities, status)
        class(knn_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: order(:)
        real(dp), allocatable :: distances(:), scores(:)
        real(dp) :: delta, distance, denominator, weight
        logical :: exact_neighbor
        integer :: i, j, k, c, index, n_query

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier predict_proba: model is not fitted")
            return
        end if
        n_query = size(x, 1)
        if (size(x, 2) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier predict_proba: input feature shape is invalid")
            return
        end if
        if (any(shape(probabilities) /= [n_query, self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier predict_proba: output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier predict_proba: inputs must be finite")
            return
        end if

        allocate(order(self%n_samples), distances(self%n_samples), &
            scores(self%n_classes))
        do i = 1, n_query
            do j = 1, self%n_samples
                distance = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    distance = distance + delta*delta
                    if (.not. ieee_is_finite(distance)) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "KNN classifier predict_proba: distance overflow")
                        return
                    end if
                end do
                distances(j) = distance
                order(j) = j
            end do
            call sort_neighbor_order(order, distances)
            scores = 0.0_dp
            exact_neighbor = .false.
            if (self%weighting_code == KNN_WEIGHTS_DISTANCE) then
                do k = 1, self%n_neighbors
                    if (distances(order(k)) == 0.0_dp) exact_neighbor = .true.
                end do
            end if
            do k = 1, self%n_neighbors
                index = order(k)
                if (self%weighting_code == KNN_WEIGHTS_UNIFORM) then
                    weight = self%sample_weight(index)
                else if (exact_neighbor) then
                    weight = 0.0_dp
                    if (distances(index) == 0.0_dp) weight = self%sample_weight(index)
                else
                    weight = self%sample_weight(index)/sqrt(distances(index))
                end if
                c = self%train_class(index)
                scores(c) = scores(c) + weight
            end do
            denominator = sum(scores)
            if (.not. ieee_is_finite(denominator) .or. denominator <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "KNN classifier predict_proba: selected neighbors have zero weight")
                return
            end if
            probabilities(i, :) = scores/denominator
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine knn_classifier_predict_proba

    subroutine knn_classifier_predict(self, x, labels, status)
        class(knn_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier predict: model is not fitted")
            return
        end if
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "KNN classifier predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine knn_classifier_predict

    subroutine knn_classifier_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(knn_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "KNN classifier input JVP is undefined across neighbor selection")
    end subroutine knn_classifier_predict_proba_jvp

    subroutine knn_classifier_predict_proba_vjp(self, x, probabilities_bar, x_bar, &
            status)
        class(knn_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        x_bar = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "KNN classifier input VJP is undefined across neighbor selection")
    end subroutine knn_classifier_predict_proba_vjp

    function knn_classifier_classes(self) result(values)
        class(knn_classifier_t), intent(in) :: self
        integer, allocatable :: values(:)

        if (allocated(self%class_label)) then
            allocate(values, source=self%class_label)
        else
            allocate(values(0))
        end if
    end function knn_classifier_classes

    integer function knn_classifier_n_neighbors(self) result(value)
        class(knn_classifier_t), intent(in) :: self
        value = self%n_neighbors
    end function knn_classifier_n_neighbors

    integer function knn_classifier_weighting(self) result(value)
        class(knn_classifier_t), intent(in) :: self
        value = self%weighting_code
    end function knn_classifier_weighting

    integer function knn_classifier_feature_count(self) result(value)
        class(knn_classifier_t), intent(in) :: self
        value = self%n_features
    end function knn_classifier_feature_count

    integer function knn_classifier_sample_count(self) result(value)
        class(knn_classifier_t), intent(in) :: self
        value = self%n_samples
    end function knn_classifier_sample_count

    integer function knn_classifier_class_count(self) result(value)
        class(knn_classifier_t), intent(in) :: self
        value = self%n_classes
    end function knn_classifier_class_count

    logical function knn_classifier_fitted(self) result(value)
        class(knn_classifier_t), intent(in) :: self
        value = self%is_fitted
    end function knn_classifier_fitted

    integer function class_index(classes, label) result(index)
        integer, intent(in) :: classes(:), label
        integer :: left, right, middle

        left = 1
        right = size(classes)
        index = 0
        do while (left <= right)
            middle = left + (right-left)/2
            if (classes(middle) == label) then
                index = middle
                return
            else if (classes(middle) < label) then
                left = middle + 1
            else
                right = middle - 1
            end if
        end do
    end function class_index

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, count

        allocate(work(size(labels)))
        work = labels
        call sort_in_place(work)
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(i-1)) count = count + 1
        end do
        allocate(classes(count))
        classes(1) = work(1)
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(i-1)) then
                count = count + 1
                classes(count) = work(i)
            end if
        end do
    end subroutine sorted_unique_labels

    subroutine sort_in_place(values)
        integer, intent(inout) :: values(:)
        integer :: i, j, value

        do i = 2, size(values)
            value = values(i)
            j = i - 1
            do while (j >= 1)
                if (values(j) <= value) exit
                values(j+1) = values(j)
                j = j - 1
            end do
            values(j+1) = value
        end do
    end subroutine sort_in_place

    subroutine sort_neighbor_order(order, distances)
        integer, intent(inout) :: order(:)
        real(dp), intent(in) :: distances(:)
        integer :: i, j, value

        do i = 2, size(order)
            value = order(i)
            j = i - 1
            do while (j >= 1)
                if (neighbor_precedes(order(j), value, distances)) exit
                order(j+1) = order(j)
                j = j - 1
            end do
            order(j+1) = value
        end do
    end subroutine sort_neighbor_order

    logical function neighbor_precedes(left, right, distances) result(precedes)
        integer, intent(in) :: left, right
        real(dp), intent(in) :: distances(:)

        if (distances(left) < distances(right)) then
            precedes = .true.
        else if (distances(left) > distances(right)) then
            precedes = .false.
        else
            precedes = left < right
        end if
    end function neighbor_precedes

end module fortml_knn_classifier
