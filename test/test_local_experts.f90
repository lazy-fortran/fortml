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
        AGGREGATE_RBCM, AGGREGATE_GRBCM
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
        real(dp) :: matrix(n, n), factor(n, n), alpha(n), cross(n), column(n)
        real(dp) :: total
        integer :: i, j, k

        do j = 1, n
            do i = 1, n
                matrix(i, j) = rbf(x(i, 1), x(j, 1))
            end do
            matrix(j, j) = matrix(j, j) + noise
        end do
        factor = 0.0_dp
        do i = 1, n
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
        do i = 1, size(query, 1)
            do j = 1, n
                cross(j) = rbf(query(i, 1), x(j, 1))
            end do
            call solve(factor, y, alpha)
            mean(i) = sum(cross*alpha)
            call solve(factor, cross, column)
            variance_out(i) = rbf(query(i, 1), query(i, 1)) - sum(cross*column)
        end do
    end subroutine exact_reference

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

    subroutine test_names_and_refusals(failures)
        integer, intent(inout) :: failures
        type(local_expert_gp_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: query(1, d), mean(1), variance_out(1)

        kernel = make_rbf_kernel(d, variance, lengthscale, status)
        if (aggregation_name(AGGREGATE_GRBCM) /= "GRBCM" .or. &
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

        ! GRBCM must still run and stay a valid variance.
        call model%initialize(kernel, noise, AGGREGATE_GRBCM, status)
        call model%fit(x, y, 3, status)
        query(1, 1) = 0.2_dp
        call model%predict(query, mean, variance_out, status)
        if (.not. status_ok(status) .or. variance_out(1) <= 0.0_dp .or. &
            variance_out(1) > variance) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [grbcm] variance left its bounds ", variance_out(1)
            failures = failures + 1
        end if
    end subroutine test_names_and_refusals

end program test_local_experts
