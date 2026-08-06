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
    !!   GRBCM  generalized RBCM with a global communication     (29)
    !!          expert trained on a shared random subset
    !!   MoE    mixture of experts, a softmax-gated Gaussian     (21)-(22)
    !!          mixture rather than a product
    !!
    !! Cost is `O(M m0^3)` for training with `m0 = n/M` points per expert, so
    !! `O(n m0^2)` overall, against `O(n^3)` for the exact GP.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
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

    type :: expert_t
        real(dp), allocatable :: inputs(:, :)
        real(dp), allocatable :: alpha(:)
        type(cholesky_factorization_t) :: factorization
        integer :: n_points = 0
    end type expert_t

    type, public :: local_expert_gp_t
        type(kernel_t) :: kernel
        type(expert_t), allocatable :: experts(:)
        !! The communication expert of GRBCM, trained on a shared subset.
        type(expert_t) :: global_expert
        real(dp) :: noise_variance = 1.0_dp
        integer :: n_experts = 0
        integer :: method = AGGREGATE_POE
        logical :: fitted = .false.
    contains
        procedure, public :: initialize => local_initialize
        procedure, public :: fit => local_fit
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

    subroutine local_fit(self, x, y, n_experts, status)
        !! Partition by contiguous blocks of the given ordering, which keeps
        !! the split reproducible; a caller that wants clustered experts sorts
        !! its data first.
        class(local_expert_gp_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        integer, intent(in) :: n_experts
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last, block_size, n

        self%fitted = .false.
        n = size(x, 1)
        if (n_experts < 1 .or. n < n_experts .or. size(y) /= n .or. &
            size(x, 2) /= self%kernel%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "local experts: training shape is invalid")
            return
        end if
        if (allocated(self%experts)) deallocate(self%experts)
        allocate(self%experts(n_experts))
        self%n_experts = n_experts
        block_size = (n + n_experts - 1)/n_experts

        do i = 1, n_experts
            first = (i - 1)*block_size + 1
            last = min(n, i*block_size)
            if (first > last) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "local experts: an expert received no data")
                return
            end if
            call fit_expert(self, x(first:last, :), y(first:last), &
                self%experts(i), status)
            if (status%code /= FORTNUM_OK) return
        end do

        if (self%method == AGGREGATE_GRBCM) then
            ! The communication expert sees one point from each block, which
            ! is the cheapest shared subset that still spans the input range.
            block
                real(dp), allocatable :: global_x(:, :), global_y(:)
                integer :: index

                allocate(global_x(n_experts, size(x, 2)), global_y(n_experts))
                do i = 1, n_experts
                    index = min(n, (i - 1)*block_size + 1)
                    global_x(i, :) = x(index, :)
                    global_y(i) = y(index)
                end do
                call fit_expert(self, global_x, global_y, self%global_expert, &
                    status)
                if (status%code /= FORTNUM_OK) return
            end block
        end if
        self%fitted = .true.
    end subroutine local_fit

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
        integer :: i, j, nearest

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

        allocate(expert_mean(self%n_experts), expert_variance(self%n_experts))
        allocate(weight(self%n_experts))
        do i = 1, size(query, 1)
            prior = self%kernel%value(query(i, :), query(i, :))
            do j = 1, self%n_experts
                call expert_predict(self, self%experts(j), query(i, :), &
                    expert_mean(j), expert_variance(j), status)
                if (status%code /= FORTNUM_OK) return
            end do

            if (self%method == AGGREGATE_MOE) then
                ! A gated Gaussian mixture, not a product. Following the
                ! review's Fig. 5 setting, the gate is a softmax over the same
                ! differential-entropy scores, and the mixture moments are
                ! taken exactly. A mixture is never sharper than its sharpest
                ! component, which is what separates it from PoE.
                do j = 1, self%n_experts
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
            case (AGGREGATE_GPOE, AGGREGATE_RBCM, AGGREGATE_GRBCM)
                ! The drop in differential entropy from prior to posterior.
                do j = 1, self%n_experts
                    weight(j) = 0.5_dp*(log(prior) - log(expert_variance(j)))
                    weight(j) = max(weight(j), 1.0e-12_dp)
                end do
                if (self%method == AGGREGATE_GPOE) then
                    ! Normalizing recovers the prior when leaving the data.
                    weight = weight/sum(weight)
                end if
            end select

            if (self%method == AGGREGATE_GRBCM) then
                call expert_predict(self, self%global_expert, query(i, :), &
                    global_mean, global_variance, status)
                if (status%code /= FORTNUM_OK) return
                precision = 1.0_dp/global_variance
                weighted_mean = global_mean/global_variance
                do j = 1, self%n_experts
                    precision = precision + &
                        weight(j)*(1.0_dp/expert_variance(j) - &
                        1.0_dp/global_variance)
                    weighted_mean = weighted_mean + &
                        weight(j)*(expert_mean(j)/expert_variance(j) - &
                        global_mean/global_variance)
                end do
            else
                precision = 0.0_dp
                weighted_mean = 0.0_dp
                do j = 1, self%n_experts
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
