module fortml_kmeans
    !! Deterministic dense k-means clustering with explicit capability bounds.
    !!
    !! Samples are rows.  Initialization is a seeded cyclic sample of distinct
    !! rows, followed by Lloyd updates with stable lowest-index tie breaking.
    !! Empty clusters are reported as a convergence error rather than silently
    !! reseeded, because an implicit active-set change would make a persisted
    !! fit irreproducible.  Fixed-center Euclidean distances expose input JVP
    !! and VJP products away from zero distance; labels remain discrete.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU
    implicit none
    private

    type, public :: kmeans_t
        private
        real(dp), allocatable :: center_value(:, :)
        integer, allocatable :: label_value(:)
        real(dp) :: inertia_value = huge(1.0_dp)
        real(dp) :: tolerance_value = 1.0e-4_dp
        integer :: cluster_count_value = 0
        integer :: feature_count_value = 0
        integer :: sample_count_value = 0
        integer :: iteration_count_value = 0
        integer :: max_iteration_value = 300
        integer :: initialization_seed_value = 1
        integer :: device_kind_value = FORTML_DEVICE_CPU
    contains
        procedure, public :: fit => kmeans_fit
        procedure, public :: fit_transform => kmeans_fit_transform
        procedure, public :: predict => kmeans_predict
        procedure, public :: transform => kmeans_transform
        procedure, public :: transform_jvp => kmeans_transform_jvp
        procedure, public :: transform_vjp => kmeans_transform_vjp
        procedure, public :: cluster_centers => kmeans_cluster_centers
        procedure, public :: labels => kmeans_labels
        procedure, public :: inertia => kmeans_inertia
        procedure, public :: n_clusters => kmeans_n_clusters
        procedure, public :: feature_count => kmeans_feature_count
        procedure, public :: sample_count => kmeans_sample_count
        procedure, public :: n_iter => kmeans_n_iter
        procedure, public :: tolerance => kmeans_tolerance
        procedure, public :: initialization_seed => kmeans_initialization_seed
        procedure, public :: device_kind => kmeans_device_kind
        procedure, public :: fitted => kmeans_fitted
    end type kmeans_t

contains

    subroutine kmeans_fit(self, x, status, n_clusters, max_iter, tolerance, &
            initialization_seed, device_kind)
        class(kmeans_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_clusters, max_iter, initialization_seed
        real(dp), intent(in), optional :: tolerance
        integer, intent(in), optional :: device_kind
        integer :: requested_clusters, requested_iterations, seed, backend
        real(dp) :: requested_tolerance
        integer :: n_samples, n_features, iteration, i, j, nearest, count
        integer, allocatable :: counts(:), labels(:)
        real(dp), allocatable :: centers(:, :), next_centers(:, :), distances(:)
        real(dp) :: shift, squared_distance, best_distance, scale

        call invalidate_kmeans(self)
        requested_clusters = 8
        if (present(n_clusters)) requested_clusters = n_clusters
        requested_iterations = 300
        if (present(max_iter)) requested_iterations = max_iter
        requested_tolerance = 1.0e-4_dp
        if (present(tolerance)) requested_tolerance = tolerance
        seed = 1
        if (present(initialization_seed)) seed = initialization_seed
        backend = FORTML_DEVICE_CPU
        if (present(device_kind)) backend = device_kind
        if (backend /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "kmeans fit: CUDA/device-resident clustering is not implemented")
            return
        end if
        if (.not. valid_matrix(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kmeans fit: input must be a finite nonempty matrix")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (requested_clusters < 1 .or. requested_clusters > n_samples .or. &
                requested_iterations < 1 .or. requested_tolerance < 0.0_dp .or. &
                .not. ieee_is_finite(requested_tolerance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kmeans fit: cluster count, iteration limit, or tolerance is invalid")
            return
        end if

        allocate(centers(requested_clusters, n_features), next_centers(requested_clusters, n_features))
        allocate(counts(requested_clusters), labels(n_samples), distances(n_samples))
        do j = 1, requested_clusters
            i = modulo(seed - 1 + j - 1, n_samples) + 1
            centers(j, :) = x(i, :)
        end do
        shift = huge(1.0_dp)
        do iteration = 1, requested_iterations
            counts = 0
            next_centers = 0.0_dp
            do i = 1, n_samples
                nearest = 1
                best_distance = squared_distance_row(x(i, :), centers(1, :))
                do j = 2, requested_clusters
                    squared_distance = squared_distance_row(x(i, :), centers(j, :))
                    if (squared_distance < best_distance) then
                        nearest = j
                        best_distance = squared_distance
                    end if
                end do
                labels(i) = nearest
                distances(i) = best_distance
                counts(nearest) = counts(nearest) + 1
                next_centers(nearest, :) = next_centers(nearest, :) + x(i, :)
            end do
            if (any(counts == 0)) then
                call invalidate_kmeans(self)
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "kmeans fit: empty cluster encountered; no implicit reseed was applied")
                return
            end if
            do j = 1, requested_clusters
                next_centers(j, :) = next_centers(j, :) / real(counts(j), dp)
            end do
            shift = maxval(sqrt(sum((next_centers-centers)**2, dim=2)))
            centers = next_centers
            if (shift <= requested_tolerance) exit
        end do
        if (iteration > requested_iterations) iteration = requested_iterations
        ! Reassign once to the final centers so labels and inertia describe the
        ! returned state rather than the preceding Lloyd iterate.
        counts = 0
        do i = 1, n_samples
            nearest = 1
            best_distance = squared_distance_row(x(i, :), centers(1, :))
            do j = 2, requested_clusters
                squared_distance = squared_distance_row(x(i, :), centers(j, :))
                if (squared_distance < best_distance) then
                    nearest = j
                    best_distance = squared_distance
                end if
            end do
            labels(i) = nearest
            distances(i) = best_distance
            counts(nearest) = counts(nearest) + 1
        end do
        if (any(counts == 0)) then
            call invalidate_kmeans(self)
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "kmeans fit: final assignment produced an empty cluster")
            return
        end if
        self%center_value = centers
        self%label_value = labels
        self%inertia_value = sum(distances)
        self%tolerance_value = requested_tolerance
        self%cluster_count_value = requested_clusters
        self%feature_count_value = n_features
        self%sample_count_value = n_samples
        self%iteration_count_value = iteration
        self%max_iteration_value = requested_iterations
        self%initialization_seed_value = seed
        self%device_kind_value = backend
        if (shift > requested_tolerance) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "kmeans fit: iteration limit reached before tolerance")
        else
            call status_set(status, FORTNUM_OK, "")
        end if
    end subroutine kmeans_fit

    subroutine kmeans_fit_transform(self, x, transformed, status, n_clusters, &
            max_iter, tolerance, initialization_seed, device_kind)
        class(kmeans_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_clusters, max_iter, initialization_seed
        real(dp), intent(in), optional :: tolerance
        integer, intent(in), optional :: device_kind

        call self%fit(x, status, n_clusters, max_iter, tolerance, initialization_seed, device_kind)
        if (status%code /= FORTNUM_OK) return
        call self%transform(x, transformed, status)
    end subroutine kmeans_fit_transform

    subroutine kmeans_predict(self, x, labels, status, device_kind)
        class(kmeans_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: device_kind
        integer :: i, j, nearest
        real(dp) :: distance, best_distance

        if (.not. valid_predict_input(self, x, labels, device_kind)) then
            if (present(device_kind)) then
                if (device_kind /= FORTML_DEVICE_CPU) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "kmeans predict: CUDA/device-resident prediction is not implemented")
                    return
                end if
            end if
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kmeans predict: model, input, or output shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            nearest = 1
            best_distance = squared_distance_row(x(i, :), self%center_value(1, :))
            do j = 2, self%cluster_count_value
                distance = squared_distance_row(x(i, :), self%center_value(j, :))
                if (distance < best_distance) then
                    nearest = j
                    best_distance = distance
                end if
            end do
            labels(i) = nearest
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine kmeans_predict

    subroutine kmeans_transform(self, x, distances, status, device_kind)
        class(kmeans_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: distances(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: device_kind
        integer :: i, j

        if (.not. valid_transform_input(self, x, distances, device_kind)) then
            if (present(device_kind)) then
                if (device_kind /= FORTML_DEVICE_CPU) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "kmeans transform: CUDA/device-resident transform is not implemented")
                    return
                end if
            end if
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kmeans transform: model, input, or output shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            do j = 1, self%cluster_count_value
                distances(i, j) = sqrt(max(0.0_dp, squared_distance_row(x(i, :), &
                    self%center_value(j, :))))
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine kmeans_transform

    subroutine kmeans_transform_jvp(self, x, x_dot, distances_dot, status)
        class(kmeans_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: distances_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: difference(self%feature_count_value), distance
        integer :: i, j

        if (.not. valid_transform_jvp_input(self, x, x_dot, distances_dot)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kmeans transform_jvp: model, input, or output shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            do j = 1, self%cluster_count_value
                difference = x(i, :) - self%center_value(j, :)
                distance = sqrt(max(0.0_dp, sum(difference*difference)))
                if (distance <= tiny(1.0_dp)) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "kmeans transform_jvp: zero-distance norm has no smooth derivative")
                    return
                end if
                distances_dot(i, j) = dot_product(difference, x_dot(i, :))/distance
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine kmeans_transform_jvp

    subroutine kmeans_transform_vjp(self, x, distances_bar, x_bar, status)
        class(kmeans_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), distances_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: difference(self%feature_count_value), distance
        integer :: i, j

        if (.not. valid_transform_vjp_input(self, x, distances_bar, x_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kmeans transform_vjp: model, input, or output shape is invalid")
            return
        end if
        x_bar = 0.0_dp
        do i = 1, size(x, 1)
            do j = 1, self%cluster_count_value
                difference = x(i, :) - self%center_value(j, :)
                distance = sqrt(max(0.0_dp, sum(difference*difference)))
                if (distance <= tiny(1.0_dp)) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "kmeans transform_vjp: zero-distance norm has no smooth derivative")
                    return
                end if
                x_bar(i, :) = x_bar(i, :) + distances_bar(i, j)*difference/distance
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine kmeans_transform_vjp

    function kmeans_cluster_centers(self) result(values)
        class(kmeans_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%center_value)) then
            values = self%center_value
        else
            allocate(values(0, 0))
        end if
    end function kmeans_cluster_centers

    function kmeans_labels(self) result(values)
        class(kmeans_t), intent(in) :: self
        integer, allocatable :: values(:)

        if (allocated(self%label_value)) then
            values = self%label_value
        else
            allocate(values(0))
        end if
    end function kmeans_labels

    real(dp) function kmeans_inertia(self) result(value)
        class(kmeans_t), intent(in) :: self

        value = self%inertia_value
        if (.not. self%fitted()) value = huge(1.0_dp)
    end function kmeans_inertia

    integer function kmeans_n_clusters(self) result(value)
        class(kmeans_t), intent(in) :: self

        value = self%cluster_count_value
    end function kmeans_n_clusters

    integer function kmeans_feature_count(self) result(value)
        class(kmeans_t), intent(in) :: self

        value = self%feature_count_value
    end function kmeans_feature_count

    integer function kmeans_sample_count(self) result(value)
        class(kmeans_t), intent(in) :: self

        value = self%sample_count_value
    end function kmeans_sample_count

    integer function kmeans_n_iter(self) result(value)
        class(kmeans_t), intent(in) :: self

        value = self%iteration_count_value
    end function kmeans_n_iter

    real(dp) function kmeans_tolerance(self) result(value)
        class(kmeans_t), intent(in) :: self

        value = self%tolerance_value
    end function kmeans_tolerance

    integer function kmeans_initialization_seed(self) result(value)
        class(kmeans_t), intent(in) :: self

        value = self%initialization_seed_value
    end function kmeans_initialization_seed

    integer function kmeans_device_kind(self) result(value)
        class(kmeans_t), intent(in) :: self

        value = self%device_kind_value
    end function kmeans_device_kind

    logical function kmeans_fitted(self) result(value)
        class(kmeans_t), intent(in) :: self

        value = allocated(self%center_value)
        if (.not. value) return
        value = allocated(self%label_value)
        if (.not. value) return
        value = self%cluster_count_value > 0 .and. self%feature_count_value > 0
        if (.not. value) return
        value = size(self%label_value) == self%sample_count_value
    end function kmeans_fitted

    real(dp) function squared_distance_row(left, right) result(value)
        real(dp), intent(in) :: left(:), right(:)

        value = sum((left-right)**2)
    end function squared_distance_row

    logical function valid_matrix(x) result(valid)
        real(dp), intent(in) :: x(:, :)

        valid = size(x, 1) > 0
        if (.not. valid) return
        valid = size(x, 2) > 0
        if (.not. valid) return
        valid = all(ieee_is_finite(x))
    end function valid_matrix

    logical function valid_predict_input(self, x, labels, device_kind) result(valid)
        class(kmeans_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        integer, intent(in), optional :: device_kind

        valid = self%fitted()
        if (.not. valid) return
        if (present(device_kind)) then
            valid = device_kind == FORTML_DEVICE_CPU
            if (.not. valid) return
        end if
        valid = valid_matrix(x)
        if (.not. valid) return
        valid = size(x, 2) == self%feature_count_value
        if (.not. valid) return
        valid = size(labels) == size(x, 1)
    end function valid_predict_input

    logical function valid_transform_input(self, x, distances, device_kind) result(valid)
        class(kmeans_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), distances(:, :)
        integer, intent(in), optional :: device_kind

        valid = self%fitted()
        if (.not. valid) return
        if (present(device_kind)) then
            valid = device_kind == FORTML_DEVICE_CPU
            if (.not. valid) return
        end if
        valid = valid_matrix(x)
        if (.not. valid) return
        valid = size(x, 2) == self%feature_count_value
        if (.not. valid) return
        valid = all(shape(distances) == [size(x, 1), self%cluster_count_value])
    end function valid_transform_input

    logical function valid_transform_jvp_input(self, x, x_dot, distances_dot) result(valid)
        class(kmeans_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :), distances_dot(:, :)

        valid = valid_matrix(x)
        if (.not. valid) return
        valid = valid_matrix(x_dot)
        if (.not. valid) return
        valid = self%fitted()
        if (.not. valid) return
        valid = all(shape(x_dot) == shape(x))
        if (.not. valid) return
        valid = size(x, 2) == self%feature_count_value
        if (.not. valid) return
        valid = all(shape(distances_dot) == [size(x, 1), self%cluster_count_value])
    end function valid_transform_jvp_input

    logical function valid_transform_vjp_input(self, x, distances_bar, x_bar) result(valid)
        class(kmeans_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), distances_bar(:, :), x_bar(:, :)

        valid = valid_matrix(x)
        if (.not. valid) return
        valid = valid_matrix(distances_bar)
        if (.not. valid) return
        valid = self%fitted()
        if (.not. valid) return
        valid = size(x, 2) == self%feature_count_value
        if (.not. valid) return
        valid = all(shape(distances_bar) == [size(x, 1), self%cluster_count_value])
        if (.not. valid) return
        valid = all(shape(x_bar) == shape(x))
    end function valid_transform_vjp_input

    subroutine invalidate_kmeans(self)
        class(kmeans_t), intent(inout) :: self

        if (allocated(self%center_value)) deallocate(self%center_value)
        if (allocated(self%label_value)) deallocate(self%label_value)
        self%inertia_value = huge(1.0_dp)
        self%cluster_count_value = 0
        self%feature_count_value = 0
        self%sample_count_value = 0
        self%iteration_count_value = 0
        self%device_kind_value = FORTML_DEVICE_CPU
    end subroutine invalidate_kmeans

end module fortml_kmeans
