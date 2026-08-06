program test_local_experts
    !! Oracles for the local-expert aggregations.
    !!
    !! Exact identities first: with a single expert, PoE, the normalized GPoE
    !! and BCM all carry unit total weight and so reproduce the exact GP, while
    !! RBCM's entropy weight exceeds one for a confident expert and sharpens
    !! the aggregate. Then the behaviours the review reports in its Fig. 5: PoE
    !! is overconfident, its aggregated precision growing with the number of
    !! experts, while the normalized GPoE returns to the prior away from the
    !! data instead of collapsing.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_local_experts, only: local_expert_gp_t, aggregation_name, &
        AGGREGATE_NLE, AGGREGATE_POE, AGGREGATE_GPOE, AGGREGATE_BCM, &
        AGGREGATE_RBCM, AGGREGATE_GRBCM, AGGREGATE_MOE
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n = 12, d = 1
    real(dp), parameter :: variance = 1.2_dp
    real(dp), parameter :: lengthscale = 0.5_dp
    real(dp), parameter :: noise = 0.2_dp
    real(dp) :: x(n, d), y(n)
    integer :: failures

    call build_data(x, y)
    failures = 0
    call test_single_expert_is_the_exact_gp(failures)
    call test_poe_is_overconfident(failures)
    call test_gpoe_returns_to_the_prior(failures)
    call test_nle_uses_one_expert(failures)
    call test_unbalanced_partition(failures)
    call test_clustered_partition(failures)
    call test_clustered_empty_cluster_repair(failures)
    call test_moe_is_a_gated_mixture(failures)
    call test_grbcm_formula(failures)
    call test_clustered_grbcm_formula(failures)
    call test_grbcm_seed_contract(failures)
    call test_names_and_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " local expert test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_data(inputs, targets)
        real(dp), intent(out) :: inputs(:, :), targets(:)
        integer :: i

        do i = 1, n
            inputs(i, 1) = -1.8_dp + 0.3_dp*real(i, dp)
            targets(i) = sin(2.2_dp*inputs(i, 1))
        end do
    end subroutine build_data

    real(dp) function rbf(a, b) result(value)
        real(dp), intent(in) :: a, b

        value = variance*exp(-0.5_dp*(a - b)**2/(lengthscale*lengthscale))
    end function rbf

    subroutine exact_reference(query, mean, variance_out)
        real(dp), intent(in) :: query(:, :)
        real(dp), intent(out) :: mean(:), variance_out(:)
        integer :: i

        do i = 1, size(query, 1)
            call dense_expert_reference(x, y, query(i, :), mean(i), &
                variance_out(i))
        end do
    end subroutine exact_reference

    subroutine dense_expert_reference(train_x, train_y, query, mean, &
            variance_out)
        !! A direct dense GP oracle, independent of the production expert type.
        real(dp), intent(in) :: train_x(:, :), train_y(:), query(:)
        real(dp), intent(out) :: mean, variance_out
        real(dp), allocatable :: alpha(:), column(:), cross(:), factor(:, :)
        real(dp), allocatable :: matrix(:, :)
        real(dp) :: total
        integer :: i, j, k, train_count

        train_count = size(train_y)
        allocate(alpha(train_count), column(train_count), cross(train_count))
        allocate(factor(train_count, train_count))
        allocate(matrix(train_count, train_count))
        do j = 1, train_count
            do i = 1, train_count
                matrix(i, j) = rbf(train_x(i, 1), train_x(j, 1))
            end do
            matrix(j, j) = matrix(j, j) + noise
        end do
        factor = 0.0_dp
        do i = 1, train_count
            do j = 1, i
                total = matrix(i, j)
                do k = 1, j - 1
                    total = total - factor(i, k)*factor(j, k)
                end do
                if (i == j) then
                    factor(i, i) = sqrt(total)
                else
                    factor(i, j) = total/factor(j, j)
                end if
            end do
        end do
        do j = 1, train_count
            cross(j) = rbf(query(1), train_x(j, 1))
        end do
        call solve(factor, train_y, alpha)
        mean = sum(cross*alpha)
        call solve(factor, cross, column)
        variance_out = max(rbf(query(1), query(1)) - sum(cross*column), &
            1.0e-12_dp)
    end subroutine dense_expert_reference

    subroutine solve(factor, rhs, solution)
        real(dp), intent(in) :: factor(:, :), rhs(:)
        real(dp), intent(out) :: solution(:)
        real(dp) :: total
        integer :: i, k

        do i = 1, size(rhs)
            total = rhs(i)
            do k = 1, i - 1
                total = total - factor(i, k)*solution(k)
            end do
            solution(i) = total/factor(i, i)
        end do
        do i = size(rhs), 1, -1
            total = solution(i)
            do k = i + 1, size(rhs)
                total = total - factor(k, i)*solution(k)
            end do
            solution(i) = total/factor(i, i)
        end do
    end subroutine solve

    subroutine test_single_expert_is_the_exact_gp(failures)
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: query(4, d), mean(4), variance_out(4)
        real(dp) :: reference_mean(4), reference_variance(4)
        integer :: methods(3), index, i

        do i = 1, 4
            query(i, 1) = -1.2_dp + 0.7_dp*real(i, dp)
        end do
        call exact_reference(query, reference_mean, reference_variance)

        ! PoE, the normalized GPoE and BCM all carry unit total weight, so a
        ! single expert reproduces the exact GP. RBCM deliberately does not:
        ! its entropy weight is not one, which is checked separately below.
        methods = [AGGREGATE_POE, AGGREGATE_GPOE, AGGREGATE_BCM]
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        do index = 1, 3
            call model%initialize(kernel, noise, methods(index), status)
            call model%fit(x, y, 1, status)
            call model%predict(query, mean, variance_out, status)
            if (.not. status_ok(status) .or. &
                maxval(abs(mean - reference_mean)) > 1.0e-10_dp .or. &
                maxval(abs(variance_out - reference_variance)) > 1.0e-10_dp) then
                write (error_unit, '(a,a,2es12.4)') "FAIL [single] ", &
                    aggregation_name(methods(index)), &
                    maxval(abs(mean - reference_mean)), &
                    maxval(abs(variance_out - reference_variance))
                failures = failures + 1
            end if
        end do
        call check_rbcm_stays_between(reference_variance(1), query(1:1, :), &
            failures)
    end subroutine test_single_expert_is_the_exact_gp

    subroutine check_rbcm_stays_between(exact_variance, query, failures)
        !! RBCM reweights each expert by its entropy drop, a weight that
        !! exceeds one for a confident expert. So even a single expert does not
        !! reproduce the exact GP: the aggregation is sharper. What must hold
        !! is that the variance stays positive and that the reweighting is
        !! actually applied, which is what separates RBCM from BCM.
        real(dp), intent(in) :: exact_variance
        real(dp), intent(in) :: query(:, :)
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: mean(1), variance_out(1)

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(kernel, noise, AGGREGATE_RBCM, status)
        call model%fit(x, y, 1, status)
        call model%predict(query, mean, variance_out, status)
        if (.not. status_ok(status) .or. variance_out(1) <= 0.0_dp .or. &
            variance_out(1) > variance) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [rbcm] variance left its bounds ", variance_out(1), variance
            failures = failures + 1
        end if
        if (abs(variance_out(1) - exact_variance) < 1.0e-12_dp) then
            write (error_unit, '(a)') &
                "FAIL [rbcm] the entropy reweighting was not applied"
            failures = failures + 1
        end if
        if (variance_out(1) >= exact_variance) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [rbcm] a confident expert should sharpen the aggregate ", &
                variance_out(1), exact_variance
            failures = failures + 1
        end if
    end subroutine check_rbcm_stays_between

    subroutine test_poe_is_overconfident(failures)
        !! Paper Sec. IV-C: the PoE precision is the sum of the experts', so it
        !! grows with M and the aggregated variance shrinks below every expert.
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: two, four
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: query(1, d), two_mean(1), two_variance(1)
        real(dp) :: four_mean(1), four_variance(1)

        query(1, 1) = 0.15_dp
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call two%initialize(kernel, noise, AGGREGATE_POE, status)
        call two%fit(x, y, 2, status)
        call two%predict(query, two_mean, two_variance, status)
        call four%initialize(kernel, noise, AGGREGATE_POE, status)
        call four%fit(x, y, 4, status)
        call four%predict(query, four_mean, four_variance, status)

        if (.not. status_ok(status) .or. four_variance(1) >= two_variance(1)) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [poe] variance does not shrink with more experts ", &
                two_variance(1), four_variance(1)
            failures = failures + 1
        end if
    end subroutine test_poe_is_overconfident

    subroutine test_gpoe_returns_to_the_prior(failures)
        !! Paper Sec. IV-C: with weights summing to one, GPoE recovers the
        !! prior variance far from the data, where PoE does not.
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: gpoe, poe
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: query(1, d), mean(1), gpoe_variance(1), poe_variance(1)

        query(1, 1) = 25.0_dp
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call gpoe%initialize(kernel, noise, AGGREGATE_GPOE, status)
        call gpoe%fit(x, y, 4, status)
        call gpoe%predict(query, mean, gpoe_variance, status)
        call poe%initialize(kernel, noise, AGGREGATE_POE, status)
        call poe%fit(x, y, 4, status)
        call poe%predict(query, mean, poe_variance, status)

        if (.not. status_ok(status) .or. &
            abs(gpoe_variance(1) - variance) > 1.0e-8_dp) then
            write (error_unit, '(a,es14.6)') &
                "FAIL [gpoe] does not recover the prior variance ", &
                gpoe_variance(1)
            failures = failures + 1
        end if
        if (poe_variance(1) >= gpoe_variance(1)) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [gpoe] PoE is not the more confident of the two ", &
                poe_variance(1), gpoe_variance(1)
            failures = failures + 1
        end if
    end subroutine test_gpoe_returns_to_the_prior

    subroutine test_nle_uses_one_expert(failures)
        !! The naive local expert answers with the owning subregion alone, so
        !! its prediction must equal that expert's own GP, and the aggregation
        !! is discontinuous across a boundary.
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: nle, single
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: query(1, d), mean(1), variance_out(1)
        real(dp) :: expert_mean(1), expert_variance(1)

        query(1, 1) = x(2, 1)
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call nle%initialize(kernel, noise, AGGREGATE_NLE, status)
        call nle%fit(x, y, 3, status)
        call nle%predict(query, mean, variance_out, status)

        ! The first expert owns the first quarter of the ordering.
        call single%initialize(kernel, noise, AGGREGATE_POE, status)
        call single%fit(x(1:4, :), y(1:4), 1, status)
        call single%predict(query, expert_mean, expert_variance, status)

        if (.not. status_ok(status) .or. &
            abs(mean(1) - expert_mean(1)) > 1.0e-10_dp .or. &
            abs(variance_out(1) - expert_variance(1)) > 1.0e-10_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [nle] does not equal the owning expert ", &
                mean(1) - expert_mean(1), variance_out(1) - expert_variance(1)
            failures = failures + 1
        end if
    end subroutine test_nle_uses_one_expert

    subroutine test_unbalanced_partition(failures)
        !! Five observations split four ways must form blocks [2,1,1,1].
        !! Compare the two edge predictions with dense GPs on the hand-known
        !! owning blocks; this also catches an accidentally empty last expert.
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: expected_mean(2), expected_variance(2), mean(2)
        real(dp) :: query(2, d), variance_out(2)

        query(:, 1) = [x(1, 1), x(5, 1)]
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(kernel, noise, AGGREGATE_NLE, status)
        call model%fit(x(:5, :), y(:5), 4, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [partition] valid unbalanced split was refused"
            failures = failures + 1
            return
        end if
        call model%predict(query, mean, variance_out, status)
        call dense_expert_reference(x(:2, :), y(:2), query(1, :), &
            expected_mean(1), expected_variance(1))
        call dense_expert_reference(x(5:5, :), y(5:5), query(2, :), &
            expected_mean(2), expected_variance(2))
        if (.not. status_ok(status) .or. model%expert_count() /= 4 .or. &
            maxval(abs(mean - expected_mean)) > 1.0e-11_dp .or. &
            maxval(abs(variance_out - expected_variance)) > 1.0e-11_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [partition] unbalanced blocks differ from dense GPs ", &
                maxval(abs(mean - expected_mean)), &
                maxval(abs(variance_out - expected_variance))
            failures = failures + 1
        end if
    end subroutine test_unbalanced_partition

    subroutine test_clustered_partition(failures)
        !! A shuffled two-cluster data set has alternating signs, so contiguous
        !! blocks mix both clusters. The clustered fit must recover the two
        !! hand-known sign groups and give the same NLE predictions as two
        !! independently fitted single-expert models.
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: clustered, negative, positive
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: shuffled_x(8, d), shuffled_y(8)
        real(dp) :: negative_x(4, d), positive_x(4, d)
        real(dp) :: negative_y(4), positive_y(4), query(2, d)
        real(dp) :: mean(2), variance_out(2), expected_mean(2)
        real(dp) :: expected_variance(2), one_mean(1), one_variance(1)

        shuffled_x(:, 1) = [-2.2_dp, 1.8_dp, -1.8_dp, 2.2_dp, &
            -2.0_dp, 2.0_dp, -1.9_dp, 1.9_dp]
        shuffled_y = sin(shuffled_x(:, 1))
        negative_x(:, 1) = [-2.2_dp, -1.8_dp, -2.0_dp, -1.9_dp]
        positive_x(:, 1) = [1.8_dp, 2.2_dp, 2.0_dp, 1.9_dp]
        negative_y = sin(negative_x(:, 1))
        positive_y = sin(positive_x(:, 1))
        query(:, 1) = [-2.05_dp, 2.05_dp]

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call clustered%initialize(kernel, noise, AGGREGATE_NLE, status)
        call clustered%fit_clustered(shuffled_x, shuffled_y, 2, status)
        call clustered%predict(query, mean, variance_out, status)

        call negative%initialize(kernel, noise, AGGREGATE_NLE, status)
        call negative%fit(negative_x, negative_y, 1, status)
        call negative%predict(query(1:1, :), one_mean, one_variance, status)
        expected_mean(1) = one_mean(1)
        expected_variance(1) = one_variance(1)
        call positive%initialize(kernel, noise, AGGREGATE_NLE, status)
        call positive%fit(positive_x, positive_y, 1, status)
        call positive%predict(query(2:2, :), one_mean, one_variance, status)
        expected_mean(2) = one_mean(1)
        expected_variance(2) = one_variance(1)

        if (.not. status_ok(status) .or. clustered%expert_count() /= 2 .or. &
            maxval(abs(mean - expected_mean)) > 1.0e-11_dp .or. &
            maxval(abs(variance_out - expected_variance)) > 1.0e-11_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [clustered] differs from hand-partitioned experts ", &
                maxval(abs(mean - expected_mean)), &
                maxval(abs(variance_out - expected_variance))
            failures = failures + 1
        end if

        call clustered%fit_clustered(shuffled_x, shuffled_y, 2, status, &
            max_iterations=0)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [clustered] zero k-means iterations accepted"
            failures = failures + 1
        end if
    end subroutine test_clustered_partition

    subroutine test_clustered_empty_cluster_repair(failures)
        !! Identical coordinates initially collapse into one Lloyd cluster.
        !! The deterministic repair moves rows 1 and 2 to the empty clusters,
        !! leaving rows 3:6 in expert one, which wins the all-distance tie.
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: inputs(6, d), targets(6), query(1, d)
        real(dp) :: expected_mean, expected_variance, mean(1), variance_out(1)

        inputs = 0.0_dp
        targets = [0.3_dp, -0.2_dp, 1.0_dp, 0.5_dp, -0.7_dp, 0.8_dp]
        query = 0.0_dp
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(kernel, noise, AGGREGATE_NLE, status)
        call model%fit_clustered(inputs, targets, 3, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [clustered] identical points left an empty cluster"
            failures = failures + 1
            return
        end if
        call model%predict(query, mean, variance_out, status)
        call dense_expert_reference(inputs(3:6, :), targets(3:6), query(1, :), &
            expected_mean, expected_variance)
        if (.not. status_ok(status) .or. model%expert_count() /= 3 .or. &
            abs(mean(1) - expected_mean) > 1.0e-11_dp .or. &
            abs(variance_out(1) - expected_variance) > 1.0e-11_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [clustered] empty-cluster repair differs from oracle ", &
                mean(1) - expected_mean, variance_out(1) - expected_variance
            failures = failures + 1
        end if
    end subroutine test_clustered_empty_cluster_repair

    subroutine test_moe_is_a_gated_mixture(failures)
        !! Paper Sec. IV-B and Fig. 5: the MoE is a weighted sum, so it can
        !! never be sharper than its sharpest expert, where the PoE product
        !! can. With one expert the gate is degenerate and the mixture is that
        !! expert exactly.
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: moe, poe
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        integer, parameter :: n_experts = 4
        real(dp) :: expert_mean(n_experts), expert_variance(n_experts)
        real(dp) :: expected_mean, expected_variance, gate(n_experts)
        real(dp) :: query(3, d), mean(3), variance_out(3)
        real(dp) :: reference_mean(3), reference_variance(3)
        real(dp) :: poe_mean(3), poe_variance(3), score(n_experts), sharpest
        integer :: block_size, first, i, j, last

        do i = 1, 3
            query(i, 1) = -0.9_dp + 0.9_dp*real(i, dp)
        end do
        kernel = make_rbf_kernel(d, variance, lengthscale, status)

        call moe%initialize(kernel, noise, AGGREGATE_MOE, status)
        call moe%fit(x, y, 1, status)
        call moe%predict(query, mean, variance_out, status)
        call exact_reference(query, reference_mean, reference_variance)
        if (.not. status_ok(status) .or. &
            maxval(abs(mean - reference_mean)) > 1.0e-10_dp .or. &
            maxval(abs(variance_out - reference_variance)) > 1.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [moe] one expert is not the exact GP"
            failures = failures + 1
        end if

        call moe%initialize(kernel, noise, AGGREGATE_MOE, status)
        call moe%fit(x, y, n_experts, status)
        call moe%predict(query, mean, variance_out, status)
        call poe%initialize(kernel, noise, AGGREGATE_POE, status)
        call poe%fit(x, y, n_experts, status)
        call poe%predict(query, poe_mean, poe_variance, status)
        block_size = (n + n_experts - 1)/n_experts
        do i = 1, 3
            do j = 1, n_experts
                first = (j - 1)*block_size + 1
                last = min(n, j*block_size)
                call dense_expert_reference(x(first:last, :), y(first:last), &
                    query(i, :), expert_mean(j), expert_variance(j))
            end do
            score = 0.5_dp*(log(variance) - log(expert_variance))
            gate = exp(score - maxval(score))
            gate = gate/sum(gate)
            expected_mean = sum(gate*expert_mean)
            expected_variance = sum(gate*(expert_variance + expert_mean**2)) &
                - expected_mean**2
            if (abs(mean(i) - expected_mean) > 1.0e-10_dp .or. &
                abs(variance_out(i) - expected_variance) > 1.0e-10_dp) then
                write (error_unit, '(a,2es12.4)') &
                    "FAIL [moe] differs from the gated-mixture formula ", &
                    mean(i) - expected_mean, variance_out(i) - expected_variance
                failures = failures + 1
            end if
            sharpest = minval(expert_variance)
            if (variance_out(i) < sharpest - 1.0e-12_dp) then
                write (error_unit, '(a,2es12.4)') &
                    "FAIL [moe] the mixture is sharper than its sharpest expert ", &
                    variance_out(i), sharpest
                failures = failures + 1
            end if
            if (poe_variance(i) > variance_out(i)) then
                write (error_unit, '(a)') &
                    "FAIL [moe] PoE is not the sharper of the two"
                failures = failures + 1
            end if
        end do
    end subroutine test_moe_is_a_gated_mixture

    subroutine test_grbcm_formula(failures)
        !! The M=2 case has beta_2=1 and is exactly its sole enhanced expert.
        !! The M=3 case independently assembles equations (14)-(15) from the
        !! communication and enhanced data selected by the seed-42 RNG KAT.
        integer, intent(inout) :: failures
        integer, parameter :: n_experts = 3
        type(local_expert_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        integer, parameter :: communication(4) = [2, 12, 6, 7]
        integer, parameter :: enhanced_one(8) = [2, 12, 6, 7, 1, 3, 4, 5]
        integer, parameter :: enhanced_two(8) = [2, 12, 6, 7, 8, 9, 10, 11]
        real(dp) :: expected_mean, expected_variance, mean(1), query(1, d)
        real(dp) :: reference_mean(1), reference_variance(1), variance_out(1)

        query(1, 1) = 0.2_dp
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(kernel, noise, AGGREGATE_GRBCM, status)
        call model%fit(x, y, 2, status, communication_seed=42)
        call model%predict(query, mean, variance_out, status)
        call exact_reference(query, reference_mean, reference_variance)
        if (.not. status_ok(status) .or. model%expert_count() /= 2 .or. &
            abs(mean(1) - reference_mean(1)) > 1.0e-10_dp .or. &
            abs(variance_out(1) - reference_variance(1)) > 1.0e-10_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [grbcm] M=2 does not equal its enhanced expert ", &
                mean(1) - reference_mean(1), &
                variance_out(1) - reference_variance(1)
            failures = failures + 1
        end if

        call model%initialize(kernel, noise, AGGREGATE_GRBCM, status)
        call model%fit(x, y, n_experts, status, communication_seed=42)
        call model%predict(query, mean, variance_out, status)
        call three_expert_grbcm_reference(x(communication, :), &
            y(communication), x(enhanced_one, :), y(enhanced_one), &
            x(enhanced_two, :), y(enhanced_two), query(1, :), expected_mean, &
            expected_variance)
        if (.not. status_ok(status) .or. &
            abs(mean(1) - expected_mean) > 1.0e-10_dp .or. &
            abs(variance_out(1) - expected_variance) > 1.0e-10_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [grbcm] differs from the communication-expert formula ", &
                mean(1) - expected_mean, variance_out(1) - expected_variance
            failures = failures + 1
        end if
    end subroutine test_grbcm_formula

    subroutine three_expert_grbcm_reference(communication_x, communication_y, &
            enhanced_one_x, enhanced_one_y, enhanced_two_x, enhanced_two_y, &
            query, mean, variance_out)
        !! Direct dense GP solves followed by the published GRBCM equations.
        real(dp), intent(in) :: communication_x(:, :), communication_y(:)
        real(dp), intent(in) :: enhanced_one_x(:, :), enhanced_one_y(:)
        real(dp), intent(in) :: enhanced_two_x(:, :), enhanced_two_y(:)
        real(dp), intent(in) :: query(:)
        real(dp), intent(out) :: mean, variance_out
        real(dp) :: beta(2), expert_mean(2), expert_variance(2)
        real(dp) :: global_mean, global_variance, precision, weighted_mean

        call dense_expert_reference(communication_x, communication_y, query, &
            global_mean, global_variance)
        call dense_expert_reference(enhanced_one_x, enhanced_one_y, query, &
            expert_mean(1), expert_variance(1))
        call dense_expert_reference(enhanced_two_x, enhanced_two_y, query, &
            expert_mean(2), expert_variance(2))
        beta(1) = 1.0_dp
        beta(2) = 0.5_dp*(log(global_variance) - &
            log(expert_variance(2)))
        precision = sum(beta/expert_variance) - &
            (sum(beta) - 1.0_dp)/global_variance
        weighted_mean = sum(beta*expert_mean/expert_variance) - &
            (sum(beta) - 1.0_dp)*global_mean/global_variance
        variance_out = 1.0_dp/precision
        mean = weighted_mean/precision
    end subroutine three_expert_grbcm_reference

    subroutine test_clustered_grbcm_formula(failures)
        !! Seed 42 chooses rows [1,8,5] for D_c. K-means then separates the
        !! positive remainder [2,4,6] from the negative remainder [3,7].
        integer, intent(inout) :: failures
        integer, parameter :: communication(3) = [1, 8, 5]
        integer, parameter :: enhanced_one(6) = [1, 8, 5, 2, 4, 6]
        integer, parameter :: enhanced_two(5) = [1, 8, 5, 3, 7]
        type(local_expert_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: inputs(8, d), targets(8), query(1, d)
        real(dp) :: expected_mean, expected_variance, mean(1), variance_out(1)

        inputs(:, 1) = [-2.2_dp, 1.8_dp, -1.8_dp, 2.2_dp, &
            -2.0_dp, 2.0_dp, -1.9_dp, 1.9_dp]
        targets = sin(inputs(:, 1))
        query(1, 1) = 0.35_dp
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call model%initialize(kernel, noise, AGGREGATE_GRBCM, status)
        call model%fit_clustered(inputs, targets, 3, status, &
            communication_seed=42)
        call model%predict(query, mean, variance_out, status)
        call three_expert_grbcm_reference(inputs(communication, :), &
            targets(communication), inputs(enhanced_one, :), &
            targets(enhanced_one), inputs(enhanced_two, :), &
            targets(enhanced_two), query(1, :), expected_mean, expected_variance)
        if (.not. status_ok(status) .or. model%expert_count() /= 3 .or. &
            abs(mean(1) - expected_mean) > 1.0e-10_dp .or. &
            abs(variance_out(1) - expected_variance) > 1.0e-10_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [grbcm] clustered formula differs ", &
                mean(1) - expected_mean, variance_out(1) - expected_variance
            failures = failures + 1
        end if
    end subroutine test_clustered_grbcm_formula

    subroutine test_grbcm_seed_contract(failures)
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: default_model, other, repeated
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: default_mean(3), default_variance(3), other_mean(3)
        real(dp) :: other_variance(3), query(3, d), repeated_mean(3)
        real(dp) :: repeated_variance(3)

        query(:, 1) = [-1.1_dp, 0.2_dp, 1.4_dp]
        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        call default_model%initialize(kernel, noise, AGGREGATE_GRBCM, status)
        call default_model%fit(x, y, 3, status)
        call default_model%predict(query, default_mean, default_variance, status)
        call repeated%initialize(kernel, noise, AGGREGATE_GRBCM, status)
        call repeated%fit(x, y, 3, status, communication_seed=104729)
        call repeated%predict(query, repeated_mean, repeated_variance, status)
        call other%initialize(kernel, noise, AGGREGATE_GRBCM, status)
        call other%fit(x, y, 3, status, communication_seed=1730)
        call other%predict(query, other_mean, other_variance, status)

        if (.not. status_ok(status) .or. &
            maxval(abs(default_mean - repeated_mean)) > 1.0e-13_dp .or. &
            maxval(abs(default_variance - repeated_variance)) > 1.0e-13_dp) then
            write (error_unit, '(a)') &
                "FAIL [grbcm] default communication seed is not reproducible"
            failures = failures + 1
        end if
        if (maxval(abs(default_mean - other_mean)) < 1.0e-10_dp .and. &
            maxval(abs(default_variance - other_variance)) < 1.0e-10_dp) then
            write (error_unit, '(a)') &
                "FAIL [grbcm] changing the communication seed changes nothing"
            failures = failures + 1
        end if
    end subroutine test_grbcm_seed_contract

    subroutine test_names_and_refusals(failures)
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: query(1, d), mean(1), variance_out(1)

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        if (aggregation_name(AGGREGATE_GRBCM) /= "GRBCM" .or. &
            aggregation_name(AGGREGATE_MOE) /= "MoE" .or. &
            aggregation_name(0) /= "unknown") then
            write (error_unit, '(a)') "FAIL [name] aggregation names"
            failures = failures + 1
        end if
        call model%initialize(kernel, -1.0_dp, AGGREGATE_POE, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] negative noise accepted"
            failures = failures + 1
        end if
        call model%initialize(kernel, noise, 99, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] unknown aggregation accepted"
            failures = failures + 1
        end if
        call model%initialize(kernel, noise, AGGREGATE_POE, status)
        call model%fit(x, y, n + 1, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] more experts than points"
            failures = failures + 1
        end if
        query = 0.0_dp
        call model%predict(query, mean, variance_out, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] predict before fit accepted"
            failures = failures + 1
        end if

        call model%initialize(kernel, noise, AGGREGATE_GRBCM, status)
        call model%fit(x, y, 1, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] GRBCM accepted no enhanced expert"
            failures = failures + 1
        end if
        call model%fit_clustered(x, y, 1, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [guard] clustered GRBCM accepted no enhanced expert"
            failures = failures + 1
        end if

    end subroutine test_names_and_refusals

end program test_local_experts
