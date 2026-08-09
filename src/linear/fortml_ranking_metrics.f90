module fortml_ranking_metrics
    !! Query-grouped ranking metrics shared by the tree estimators.
    !!
    !! ranking_ndcg computes macro NDCG over arbitrary positive integer query
    !! IDs.  Score ties retain input order, ideal ordering is by weighted gain,
    !! and a positive cutoff is applied independently inside each query.  The
    !! implementation is deliberately a standalone reduction so ranking
    !! evaluation does not depend on a fitted booster or a host-language
    !! sorting callback.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    public :: ranking_ndcg
    public :: ranking_ndcg_device

contains

    subroutine ranking_ndcg(relevance, scores, group, value, status, k, &
            sample_weight, gain_base)
        real(dp), intent(in) :: relevance(:), scores(:)
        integer, intent(in) :: group(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: k
        real(dp), intent(in), optional :: sample_weight(:), gain_base
        real(dp), allocatable :: weights(:), query_gain(:), query_score(:)
        integer, allocatable :: query_ids(:), rows(:), score_order(:), ideal_order(:)
        real(dp) :: base, actual_dcg, ideal_dcg, query_value
        integer :: i, j, q, n_queries, n_rows, cutoff, n_used
        logical :: seen, have_query

        value = 0.0_dp
        if (size(relevance) < 1 .or. size(scores) /= size(relevance) .or. &
            size(group) /= size(relevance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ranking NDCG: input shapes are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(relevance)) .or. &
            any(.not. ieee_is_finite(scores)) .or. any(relevance < 0.0_dp) .or. &
            any(group <= 0)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ranking NDCG: relevance and scores must be finite with positive groups")
            return
        end if
        base = 2.0_dp
        if (present(gain_base)) base = gain_base
        if (.not. ieee_is_finite(base) .or. base <= 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ranking NDCG: gain_base must be finite and greater than one")
            return
        end if
        if (present(k)) then
            if (k < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ranking NDCG: cutoff k must be positive")
                return
            end if
        end if
        allocate(weights(size(relevance)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(relevance) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ranking NDCG: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if

        allocate(query_ids(size(group)))
        n_queries = 0
        do i = 1, size(group)
            seen = .false.
            if (n_queries > 0) then
                do j = 1, n_queries
                    if (query_ids(j) == group(i)) then
                        seen = .true.
                        exit
                    end if
                end do
            end if
            if (.not. seen) then
                n_queries = n_queries + 1
                query_ids(n_queries) = group(i)
            end if
        end do
        allocate(query_gain(size(relevance)), query_score(size(relevance)))
        have_query = .false.
        do q = 1, n_queries
            n_rows = count(group == query_ids(q))
            allocate(rows(n_rows), score_order(n_rows), ideal_order(n_rows))
            n_used = 0
            do i = 1, size(group)
                if (group(i) == query_ids(q)) then
                    n_used = n_used + 1
                    rows(n_used) = i
                end if
            end do
            do i = 1, n_rows
                query_score(i) = scores(rows(i))
                query_gain(i) = weights(rows(i))*(base**relevance(rows(i)) - 1.0_dp)
                score_order(i) = i
                ideal_order(i) = i
            end do
            call sort_descending(query_score(:n_rows), score_order)
            call sort_descending(query_gain(:n_rows), ideal_order)
            cutoff = n_rows
            if (present(k)) cutoff = min(k, n_rows)
            actual_dcg = 0.0_dp
            ideal_dcg = 0.0_dp
            do i = 1, cutoff
                j = rows(score_order(i))
                actual_dcg = actual_dcg + weights(j)*(base**relevance(j) - 1.0_dp)/ &
                    (log(real(i + 1, dp))/log(2.0_dp))
                j = rows(ideal_order(i))
                ideal_dcg = ideal_dcg + weights(j)*(base**relevance(j) - 1.0_dp)/ &
                    (log(real(i + 1, dp))/log(2.0_dp))
            end do
            if (ideal_dcg > tiny(1.0_dp)) then
                query_value = actual_dcg/ideal_dcg
                value = value + query_value
                have_query = .true.
            end if
            deallocate(rows, score_order, ideal_order)
        end do
        if (.not. have_query) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ranking NDCG: every query has zero ideal gain")
            return
        end if
        value = value/real(n_queries, dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine ranking_ndcg

    subroutine ranking_ndcg_device(device, relevance, scores, group, value, status, &
            k, sample_weight, gain_base)
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: relevance(:), scores(:)
        integer, intent(in) :: group(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: k
        real(dp), intent(in), optional :: sample_weight(:), gain_base

        value = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ranking NDCG device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            if (present(k)) then
                if (present(sample_weight)) then
                    if (present(gain_base)) then
                        call ranking_ndcg(relevance, scores, group, value, status, k, &
                            sample_weight, gain_base)
                    else
                        call ranking_ndcg(relevance, scores, group, value, status, k, &
                            sample_weight=sample_weight)
                    end if
                else
                    call ranking_ndcg(relevance, scores, group, value, status, k)
                end if
            else
                call ranking_ndcg(relevance, scores, group, value, status)
            end if
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ranking NDCG device: resident CUDA reduction is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ranking NDCG device: device kind is invalid")
        end select
    end subroutine ranking_ndcg_device

    subroutine sort_descending(values, order)
        real(dp), intent(in) :: values(:)
        integer, intent(inout) :: order(:)
        integer :: i, j, temporary

        do i = 2, size(order)
            temporary = order(i)
            j = i
            do while (j > 1)
                if (values(order(j - 1)) >= values(temporary)) exit
                order(j) = order(j - 1)
                j = j - 1
            end do
            order(j) = temporary
        end do
    end subroutine sort_descending

end module fortml_ranking_metrics
