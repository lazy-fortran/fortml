module fortml_local_experts
    !! Local approximations: naive local experts and the aggregation family.
    !!
    !! Following Liu, Ong, Shen and Cai, "When Gaussian Process Meets Big Data"
    !! (IEEE TNNLS 31(11):4405-4423, 2020), Section IV. The training data is
    !! partitioned into `M` subsets, each fitted by its own exact GP with
    !! shared hyperparameters, and the experts' predictions are aggregated:
    !!
    !!   NLE    the single expert owning the test point answers  (Sec. IV-A)
    !!   PoE    product of experts, precision `sum_i s_i^{-2}`   (25)-(27)
    !!   GPoE   generalized PoE, weights `beta_i` from the drop  (27)
    !!          in differential entropy, normalized to sum to one
    !!   BCM    Bayesian committee machine, PoE plus the prior   (28)
    !!          correction `(1 - M) k_**^{-1}`
    !!   RBCM   robust BCM, BCM with the GPoE weights            (28)
    !!   GRBCM  generalized RBCM with a communication subset     (29)
    !!          `D_c` and enhanced experts trained on `D_c U D_i`;
    !!          the first weight is one and later weights measure
    !!          the entropy drop from the communication expert
    !!   MoE    mixture of experts, a softmax-gated Gaussian     (21)-(22)
    !!          mixture rather than a product
    !!
    !! Cost is `O(M m0^3)` for training with `m0 = n/M` points per expert, so
    !! `O(n m0^2)` overall, against `O(n^3)` for the exact GP.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortml_kernels, only: kernel_t
    implicit none
    private

    integer, parameter, public :: AGGREGATE_NLE = 1
    integer, parameter, public :: AGGREGATE_POE = 2
    integer, parameter, public :: AGGREGATE_GPOE = 3
    integer, parameter, public :: AGGREGATE_BCM = 4
    integer, parameter, public :: AGGREGATE_RBCM = 5
    integer, parameter, public :: AGGREGATE_GRBCM = 6
    integer, parameter, public :: AGGREGATE_MOE = 7
    integer, parameter :: DEFAULT_COMMUNICATION_SEED = 104729

    type :: expert_t
        real(dp), allocatable :: inputs(:, :)
        real(dp), allocatable :: alpha(:)
        type(cholesky_factorization_t) :: factorization
        integer :: n_points = 0
    end type expert_t

    type, public :: local_expert_gp_t
        type(kernel_t) :: kernel
        type(expert_t), allocatable :: experts(:)
        !! The communication expert of GRBCM, trained on disjoint `D_c`.
        type(expert_t) :: global_expert
        real(dp) :: noise_variance = 1.0_dp
        integer :: n_experts = 0
        integer :: n_enhanced_experts = 0
        integer :: method = AGGREGATE_POE
        logical :: fitted = .false.
    contains
        procedure, public :: initialize => local_initialize
        procedure, public :: fit => local_fit
        procedure, public :: fit_clustered => local_fit_clustered
        procedure, public :: predict => local_predict
        procedure, public :: expert_count => local_expert_count
    end type local_expert_gp_t

    public :: aggregation_name

contains

    function aggregation_name(method) result(name)
        integer, intent(in) :: method
        character(len=:), allocatable :: name

        select case (method)
        case (AGGREGATE_NLE)
            name = "NLE"
        case (AGGREGATE_POE)
            name = "PoE"
        case (AGGREGATE_GPOE)
            name = "GPoE"
        case (AGGREGATE_BCM)
            name = "BCM"
        case (AGGREGATE_RBCM)
            name = "RBCM"
        case (AGGREGATE_GRBCM)
            name = "GRBCM"
        case (AGGREGATE_MOE)
            name = "MoE"
        case default
            name = "unknown"
        end select
    end function aggregation_name

    subroutine local_initialize(self, kernel, noise_variance, method, status)
        class(local_expert_gp_t), intent(out) :: self
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: noise_variance
        integer, intent(in) :: method
        type(fortnum_status_t), intent(out) :: status

        if (noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "local experts: the noise variance must be positive")
            return
        end if
        if (method < AGGREGATE_NLE .or. method > AGGREGATE_MOE) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "local experts: unknown aggregation")
            return
        end if
        self%kernel = kernel
        self%noise_variance = noise_variance
        self%method = method
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine local_initialize

    subroutine local_fit(self, x, y, n_experts, status, communication_seed)
        !! Partition by contiguous blocks of the given ordering, which keeps
        !! the split reproducible; a caller that wants clustered experts sorts
        !! its data first. GRBCM instead draws its disjoint communication set
        !! without replacement from `communication_seed` (default 104729),
        !! then partitions the remainder into `M - 1` blocks.
        class(local_expert_gp_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        integer, intent(in) :: n_experts
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: communication_seed
        integer, allocatable :: assignment(:), counts(:)
        real(dp), allocatable :: communication_x(:, :), communication_y(:)
        real(dp), allocatable :: remainder_x(:, :), remainder_y(:)
        integer :: base_size, extra, i, first, last, block_size, n, seed

        self%fitted = .false.
        n = size(x, 1)
        if (n_experts < 1 .or. n < n_experts .or. size(y) /= n .or. &
            size(x, 2) /= self%kernel%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "local experts: training shape is invalid")
            return
        end if
        if (self%method == AGGREGATE_GRBCM) then
            if (n_experts < 2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "local experts: GRBCM requires a communication expert")
                return
            end if
            seed = DEFAULT_COMMUNICATION_SEED
            if (present(communication_seed)) seed = communication_seed
            call split_communication_subset(x, y, n_experts, communication_x, &
                communication_y, remainder_x, remainder_y, seed, status)
            if (status%code /= FORTNUM_OK) return
            call contiguous_assignments(size(remainder_y), n_experts - 1, &
                assignment, counts)
            call fit_grbcm_experts(self, communication_x, communication_y, &
                remainder_x, remainder_y, assignment, counts, n_experts, status)
            return
        end if
        if (allocated(self%experts)) deallocate(self%experts)
        allocate(self%experts(n_experts))
        self%n_experts = n_experts
        self%n_enhanced_experts = 0
        base_size = n/n_experts
        extra = mod(n, n_experts)
        first = 1
        do i = 1, n_experts
            block_size = base_size
            if (i <= extra) block_size = block_size + 1
            last = first + block_size - 1
            call fit_expert(self, x(first:last, :), y(first:last), &
                self%experts(i), status)
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do

        self%fitted = .true.
    end subroutine local_fit

    subroutine local_fit_clustered(self, x, y, n_experts, status, &
            max_iterations, communication_seed)
        !! Partition observations with deterministic Lloyd k-means before
        !! fitting the local GPs. Farthest-point initialization makes the
        !! initial centers across separated groups and avoids a random-state
        !! contract in benchmark runs. For GRBCM, only the observations left
        !! after seeded communication-set selection are clustered.
        class(local_expert_gp_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        integer, intent(in) :: n_experts
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: max_iterations
        integer, intent(in), optional :: communication_seed
        real(dp), allocatable :: cluster_x(:, :), cluster_y(:)
        real(dp), allocatable :: communication_x(:, :), communication_y(:)
        real(dp), allocatable :: remainder_x(:, :), remainder_y(:)
        integer, allocatable :: assignment(:), counts(:)
        integer :: n, d, limit, i, cluster, seed, selected

        self%fitted = .false.
        n = size(x, 1)
        d = size(x, 2)
        limit = 50
        if (present(max_iterations)) limit = max_iterations
        if (n_experts < 1 .or. n < n_experts .or. size(y) /= n .or. &
            d /= self%kernel%input_dim .or. limit < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "local experts: clustered training shape is invalid")
            return
        end if

        if (self%method == AGGREGATE_GRBCM) then
            if (n_experts < 2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "local experts: GRBCM requires a communication expert")
                return
            end if
            seed = DEFAULT_COMMUNICATION_SEED
            if (present(communication_seed)) seed = communication_seed
            call split_communication_subset(x, y, n_experts, communication_x, &
                communication_y, remainder_x, remainder_y, seed, status)
            if (status%code /= FORTNUM_OK) return
            call cluster_assignments(remainder_x, n_experts - 1, limit, &
                assignment, counts)
            call fit_grbcm_experts(self, communication_x, communication_y, &
                remainder_x, remainder_y, assignment, counts, n_experts, status)
            return
        end if

        call cluster_assignments(x, n_experts, limit, assignment, counts)

        if (allocated(self%experts)) deallocate(self%experts)
        allocate(self%experts(n_experts))
        self%n_experts = n_experts
        self%n_enhanced_experts = 0
        do cluster = 1, n_experts
            allocate(cluster_x(counts(cluster), d), cluster_y(counts(cluster)))
            selected = 0
            do i = 1, n
                if (assignment(i) /= cluster) cycle
                selected = selected + 1
                cluster_x(selected, :) = x(i, :)
                cluster_y(selected) = y(i)
            end do
            call fit_expert(self, cluster_x, cluster_y, self%experts(cluster), &
                status)
            deallocate(cluster_x, cluster_y)
            if (status%code /= FORTNUM_OK) return
        end do
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine local_fit_clustered

    subroutine split_communication_subset(x, y, n_experts, communication_x, &
            communication_y, remainder_x, remainder_y, seed, status)
        !! Select the published random `D_c` without replacement. The explicit
        !! seed makes the random subset reproducible without global RNG state.
        real(dp), intent(in) :: x(:, :), y(:)
        integer, intent(in) :: n_experts, seed
        real(dp), allocatable, intent(out) :: communication_x(:, :)
        real(dp), allocatable, intent(out) :: communication_y(:)
        real(dp), allocatable, intent(out) :: remainder_x(:, :), remainder_y(:)
        type(fortnum_status_t), intent(out) :: status
        type(rng_t) :: generator
        logical, allocatable :: selected(:)
        integer, allocatable :: permutation(:)
        real(dp) :: draw
        integer :: candidate, communication_count, i, index, remainder_count

        communication_count = size(y)/n_experts
        if (mod(size(y), n_experts) /= 0) then
            communication_count = communication_count + 1
        end if
        remainder_count = size(y) - communication_count
        allocate(communication_x(communication_count, size(x, 2)))
        allocate(communication_y(communication_count))
        allocate(remainder_x(remainder_count, size(x, 2)))
        allocate(remainder_y(remainder_count), selected(size(y)))
        allocate(permutation(size(y)))
        permutation = [(i, i=1, size(y))]
        selected = .false.
        call rng_seed(generator, int(seed, int64), status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, communication_count
            call rng_uniform(generator, draw)
            candidate = i + min(int(draw*real(size(y) - i + 1, dp)), &
                size(y) - i)
            index = permutation(candidate)
            permutation(candidate) = permutation(i)
            permutation(i) = index
            selected(index) = .true.
            communication_x(i, :) = x(index, :)
            communication_y(i) = y(index)
        end do
        remainder_count = 0
        do i = 1, size(y)
            if (selected(i)) cycle
            remainder_count = remainder_count + 1
            remainder_x(remainder_count, :) = x(i, :)
            remainder_y(remainder_count) = y(i)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine split_communication_subset

    subroutine contiguous_assignments(n_points, n_clusters, assignment, counts)
        integer, intent(in) :: n_points, n_clusters
        integer, allocatable, intent(out) :: assignment(:), counts(:)
        integer :: base_size, cluster, extra, first, last

        allocate(assignment(n_points), counts(n_clusters))
        base_size = n_points/n_clusters
        extra = mod(n_points, n_clusters)
        first = 1
        do cluster = 1, n_clusters
            counts(cluster) = base_size
            if (cluster <= extra) counts(cluster) = counts(cluster) + 1
            last = first + counts(cluster) - 1
            assignment(first:last) = cluster
            first = last + 1
        end do
    end subroutine contiguous_assignments

    subroutine cluster_assignments(x, n_clusters, iteration_limit, assignment, &
            counts)
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: n_clusters, iteration_limit
        integer, allocatable, intent(out) :: assignment(:), counts(:)
        real(dp), allocatable :: centers(:, :), distances(:)
        integer, allocatable :: previous(:)
        integer :: cluster, i, iteration, j, selected

        allocate(centers(n_clusters, size(x, 2)), distances(size(x, 1)))
        allocate(assignment(size(x, 1)), previous(size(x, 1)))
        allocate(counts(n_clusters))
        centers(1, :) = x(1, :)
        distances = huge(1.0_dp)
        do cluster = 2, n_clusters
            do i = 1, size(x, 1)
                distances(i) = min(distances(i), &
                    sum((x(i, :) - centers(cluster - 1, :))**2))
            end do
            selected = maxloc(distances, dim=1)
            centers(cluster, :) = x(selected, :)
        end do

        assignment = 0
        previous = -1
        do iteration = 1, iteration_limit
            previous = assignment
            counts = 0
            do i = 1, size(x, 1)
                cluster = 1
                distances(1) = sum((x(i, :) - centers(1, :))**2)
                do j = 2, n_clusters
                    distances(j) = sum((x(i, :) - centers(j, :))**2)
                    if (distances(j) < distances(cluster)) cluster = j
                end do
                assignment(i) = cluster
                counts(cluster) = counts(cluster) + 1
            end do

            call fill_empty_clusters(x, centers, assignment, counts)
            centers = 0.0_dp
            do i = 1, size(x, 1)
                centers(assignment(i), :) = centers(assignment(i), :) + x(i, :)
            end do
            do cluster = 1, n_clusters
                centers(cluster, :) = centers(cluster, :)/real(counts(cluster), dp)
            end do
            if (all(assignment == previous)) exit
        end do
    end subroutine cluster_assignments

    subroutine fit_grbcm_experts(self, communication_x, communication_y, &
            remainder_x, remainder_y, assignment, counts, n_experts, status)
        class(local_expert_gp_t), intent(inout) :: self
        real(dp), intent(in) :: communication_x(:, :), communication_y(:)
        real(dp), intent(in) :: remainder_x(:, :), remainder_y(:)
        integer, intent(in) :: assignment(:), counts(:), n_experts
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: enhanced_x(:, :), enhanced_y(:)
        integer :: cluster, communication_count, i, selected

        call fit_expert(self, communication_x, communication_y, &
            self%global_expert, status)
        if (status%code /= FORTNUM_OK) return
        if (allocated(self%experts)) deallocate(self%experts)
        allocate(self%experts(size(counts)))
        self%n_experts = n_experts
        self%n_enhanced_experts = size(counts)
        communication_count = size(communication_y)
        do cluster = 1, self%n_enhanced_experts
            allocate(enhanced_x(communication_count + counts(cluster), &
                size(communication_x, 2)))
            allocate(enhanced_y(communication_count + counts(cluster)))
            enhanced_x(:communication_count, :) = communication_x
            enhanced_y(:communication_count) = communication_y
            selected = communication_count
            do i = 1, size(remainder_y)
                if (assignment(i) /= cluster) cycle
                selected = selected + 1
                enhanced_x(selected, :) = remainder_x(i, :)
                enhanced_y(selected) = remainder_y(i)
            end do
            call fit_expert(self, enhanced_x, enhanced_y, &
                self%experts(cluster), status)
            deallocate(enhanced_x, enhanced_y)
            if (status%code /= FORTNUM_OK) return
        end do
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine fit_grbcm_experts

    subroutine fill_empty_clusters(x, centers, assignment, counts)
        real(dp), intent(in) :: x(:, :), centers(:, :)
        integer, intent(inout) :: assignment(:), counts(:)
        real(dp) :: distance, farthest
        integer :: empty, i, donor, selected

        do empty = 1, size(counts)
            if (counts(empty) /= 0) cycle
            selected = 0
            farthest = -1.0_dp
            do i = 1, size(assignment)
                donor = assignment(i)
                if (counts(donor) <= 1) cycle
                distance = sum((x(i, :) - centers(donor, :))**2)
                if (distance > farthest) then
                    farthest = distance
                    selected = i
                end if
            end do
            if (selected == 0) cycle
            donor = assignment(selected)
            counts(donor) = counts(donor) - 1
            assignment(selected) = empty
            counts(empty) = 1
        end do
    end subroutine fill_empty_clusters

    subroutine fit_expert(self, x, y, expert, status)
        class(local_expert_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(expert_t), intent(out) :: expert
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: matrix(:, :)
        integer :: i

        expert%n_points = size(x, 1)
        allocate(expert%inputs, source=x)
        allocate(matrix(expert%n_points, expert%n_points))
        call self%kernel%matrix(x, x, matrix, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, expert%n_points
            matrix(i, i) = matrix(i, i) + self%noise_variance
        end do
        call expert%factorization%factorize(matrix, status)
        if (status%code /= FORTNUM_OK) return
        allocate(expert%alpha, source=y)
        call expert%factorization%solve(expert%alpha, status)
    end subroutine fit_expert

    integer function local_expert_count(self) result(count)
        class(local_expert_gp_t), intent(in) :: self

        count = self%n_experts
    end function local_expert_count

    subroutine expert_predict(self, expert, query, mean, variance, status)
        class(local_expert_gp_t), intent(in) :: self
        type(expert_t), intent(in) :: expert
        real(dp), intent(in) :: query(:)
        real(dp), intent(out) :: mean, variance
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:), solved(:)
        integer :: j

        allocate(cross(expert%n_points), solved(expert%n_points))
        do j = 1, expert%n_points
            cross(j) = self%kernel%value(query, expert%inputs(j, :))
        end do
        mean = sum(cross*expert%alpha)
        solved = cross
        call expert%factorization%solve(solved, status)
        if (status%code /= FORTNUM_OK) return
        variance = self%kernel%value(query, query) - sum(cross*solved)
        variance = max(variance, 1.0e-12_dp)
    end subroutine expert_predict

    subroutine local_predict(self, query, mean, variance, status)
        class(local_expert_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: expert_mean(:), expert_variance(:), weight(:)
        real(dp) :: prior, precision, weighted_mean, global_mean, global_variance
        integer :: i, j, nearest, prediction_experts

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "local experts: predict before fit")
            return
        end if
        if (size(query, 2) /= self%kernel%input_dim .or. &
            size(mean) /= size(query, 1) .or. size(variance) /= size(query, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "local experts: prediction shape is invalid")
            return
        end if

        prediction_experts = self%n_experts
        if (self%method == AGGREGATE_GRBCM) then
            prediction_experts = self%n_enhanced_experts
        end if
        allocate(expert_mean(prediction_experts))
        allocate(expert_variance(prediction_experts), weight(prediction_experts))
        do i = 1, size(query, 1)
            prior = self%kernel%value(query(i, :), query(i, :))
            do j = 1, prediction_experts
                call expert_predict(self, self%experts(j), query(i, :), &
                    expert_mean(j), expert_variance(j), status)
                if (status%code /= FORTNUM_OK) return
            end do

            if (self%method == AGGREGATE_GRBCM) then
                call expert_predict(self, self%global_expert, query(i, :), &
                    global_mean, global_variance, status)
                if (status%code /= FORTNUM_OK) return
            end if

            if (self%method == AGGREGATE_MOE) then
                ! A gated Gaussian mixture, not a product. Following the
                ! review's Fig. 5 setting, the gate is a softmax over the same
                ! differential-entropy scores, and the mixture moments are
                ! taken exactly. A mixture is never sharper than its sharpest
                ! component, which is what separates it from PoE.
                do j = 1, prediction_experts
                    weight(j) = 0.5_dp*(log(prior) - log(expert_variance(j)))
                end do
                weight = weight - maxval(weight)
                weight = exp(weight)
                weight = weight/sum(weight)
                mean(i) = sum(weight*expert_mean)
                variance(i) = sum(weight*(expert_variance + expert_mean**2)) &
                    - mean(i)*mean(i)
                cycle
            end if

            select case (self%method)
            case (AGGREGATE_NLE)
                nearest = nearest_expert(self, query(i, :))
                mean(i) = expert_mean(nearest)
                variance(i) = expert_variance(nearest)
                cycle
            case (AGGREGATE_POE, AGGREGATE_BCM)
                weight = 1.0_dp
            case (AGGREGATE_GPOE, AGGREGATE_RBCM)
                ! The drop in differential entropy from prior to posterior.
                do j = 1, prediction_experts
                    weight(j) = 0.5_dp*(log(prior) - log(expert_variance(j)))
                    weight(j) = max(weight(j), 1.0e-12_dp)
                end do
                if (self%method == AGGREGATE_GPOE) then
                    ! Normalizing recovers the prior when leaving the data.
                    weight = weight/sum(weight)
                end if
            case (AGGREGATE_GRBCM)
                ! Liu et al. (29): the first enhanced expert receives unit
                ! weight; later weights are entropy drops relative to `M_c`.
                weight(1) = 1.0_dp
                do j = 2, prediction_experts
                    weight(j) = 0.5_dp*(log(global_variance) - &
                        log(expert_variance(j)))
                end do
            end select

            if (self%method == AGGREGATE_GRBCM) then
                precision = sum(weight/expert_variance) - &
                    (sum(weight) - 1.0_dp)/global_variance
                weighted_mean = sum(weight*expert_mean/expert_variance) - &
                    (sum(weight) - 1.0_dp)*global_mean/global_variance
            else
                precision = 0.0_dp
                weighted_mean = 0.0_dp
                do j = 1, prediction_experts
                    precision = precision + weight(j)/expert_variance(j)
                    weighted_mean = weighted_mean + &
                        weight(j)*expert_mean(j)/expert_variance(j)
                end do
                if (self%method == AGGREGATE_BCM .or. &
                    self%method == AGGREGATE_RBCM) then
                    ! The prior correction of the committee machine.
                    precision = precision + (1.0_dp - sum(weight))/prior
                end if
            end if

            if (precision <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "local experts: the aggregated precision is not positive")
                return
            end if
            variance(i) = 1.0_dp/precision
            mean(i) = weighted_mean/precision
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine local_predict

    integer function nearest_expert(self, query) result(index)
        !! The expert whose points contain the closest training input.
        class(local_expert_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:)
        real(dp) :: best, distance
        integer :: i, j

        index = 1
        best = huge(1.0_dp)
        do i = 1, self%n_experts
            do j = 1, self%experts(i)%n_points
                distance = sum((self%experts(i)%inputs(j, :) - query)**2)
                if (distance < best) then
                    best = distance
                    index = i
                end if
            end do
        end do
    end function nearest_expert

end module fortml_local_experts
