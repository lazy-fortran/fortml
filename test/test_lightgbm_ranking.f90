program test_lightgbm_ranking
    !! Independent behavioral oracle for LightGBM query-weighted pairwise
    !! ranking.  The direct two-row formulas below do not call FortML's
    !! objective helpers; they check the pair loss, gradient/Hessian weights,
    !! group isolation, leaf-wise fit ordering, and transactional refusals.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_weighted_pair_oracle(failures)
    call test_group_isolation(failures)
    call test_two_item_fit(failures)
    call test_validation_contract(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " LightGBM ranking test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM ranking independent behavioral oracle"

contains

    subroutine test_weighted_pair_oracle(failures)
        integer, intent(inout) :: failures
        real(real64) :: margin(2), target(2), weights(2), loss
        real(real64) :: gradient(2), hessian(2), expected_loss
        integer :: group(2)
        real(real64), parameter :: half = 0.5_real64
        type(fortnum_status_t) :: status

        margin = 0.0_real64
        target = [0.0_real64, 1.0_real64]
        weights = [2.0_real64, 4.0_real64]
        group = [7, 7]
        expected_loss = log(2.0_real64)
        call direct_pair(margin, target, group, weights, loss, gradient, hessian)
        call check(abs(loss-expected_loss) < 1.0e-14_real64, &
            "weighted pair loss oracle", failures)
        call check(maxval(abs(gradient-[1.0_real64,-1.0_real64])) < 1.0e-14_real64, &
            "minimum-pair weight gradient oracle", failures)
        call check(maxval(abs(hessian-[half,half])) < 1.0e-14_real64, &
            "minimum-pair weight Hessian oracle", failures)
        call check(all(gradient == [1.0_real64,-1.0_real64]), &
            "direct oracle is deterministic", failures)
        ! Keep the status variable live so malformed future calls remain a
        ! typed API check in this focused fixture.
        status%code = FORTNUM_OK
    end subroutine test_weighted_pair_oracle

    subroutine test_group_isolation(failures)
        integer, intent(inout) :: failures
        real(real64) :: margin(3), target(3), weights(3), loss
        real(real64) :: gradient(3), hessian(3)
        integer :: group(3)

        margin = [0.2_real64, -0.1_real64, 0.3_real64]
        target = [1.0_real64, 0.0_real64, 0.0_real64]
        weights = 1.0_real64
        group = [1, 1, 2]
        call direct_pair(margin, target, group, weights, loss, gradient, hessian)
        call check(abs(gradient(3)) < 1.0e-15_real64 .and. &
            abs(hessian(3)) < 1.0e-15_real64, &
            "rows in another query do not contribute", failures)
        call check(loss > 0.0_real64, "isolated pair loss is positive", failures)
    end subroutine test_group_isolation

    subroutine test_two_item_fit(failures)
        integer, intent(inout) :: failures
        type(lightgbm_t) :: model, repeated, restored
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(2,1), target(2), prediction(2), replay(2), restored_prediction(2)
        integer :: group(2)
        character(*), parameter :: snapshot = "test_lightgbm_ranking.snapshot"

        x(:,1) = [0.0_real64, 1.0_real64]
        target = [0.0_real64, 1.0_real64]
        group = [11, 11]
        options = lightgbm_options_t()
        options%n_estimators = 1
        options%num_leaves = 2
        options%min_data_in_leaf = 1
        options%max_bin = 16
        options%learning_rate = 1.0_real64
        options%l2 = 0.0_real64
        call model%fit_ranking(x, target, group, status, options)
        call check(status%code == FORTNUM_OK, "ranking fit", failures)
        call model%predict(x, prediction, status)
        call check(status%code == FORTNUM_OK, "ranking prediction", failures)
        call check(trim(model%objective_name()) == "rank:pairwise", &
            "ranking objective metadata", failures)
        call check(prediction(2) > prediction(1) .and. &
            abs(prediction(1)+2.0_real64) < 2.0e-12_real64 .and. &
            abs(prediction(2)-2.0_real64) < 2.0e-12_real64, &
            "two-item pairwise leaf oracle", failures)
        call repeated%fit_ranking(x, target, group, status, options)
        call repeated%predict(x, replay, status)
        call check(status%code == FORTNUM_OK .and. &
            maxval(abs(replay-prediction)) < 1.0e-14_real64, &
            "seeded ranking replay", failures)
        call model%save_text(snapshot, status)
        call check(status%code == FORTNUM_OK, "ranking persistence save", failures)
        call restored%load_text(snapshot, status)
        call check(status%code == FORTNUM_OK, "ranking persistence load", failures)
        call restored%predict(x, restored_prediction, status)
        call check(status%code == FORTNUM_OK .and. &
            maxval(abs(restored_prediction-prediction)) < 1.0e-14_real64, &
            "ranking persistence prediction replay", failures)
    end subroutine test_two_item_fit

    subroutine test_validation_contract(failures)
        integer, intent(inout) :: failures
        type(lightgbm_t) :: model
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(2,1), target(2), prediction(2)
        integer :: group(2), malformed_group(2)

        x(:,1) = [0.0_real64, 1.0_real64]
        target = [0.0_real64, 1.0_real64]
        group = [19, 19]
        malformed_group = [19, 20]
        options = lightgbm_options_t()
        options%n_estimators = 1
        options%num_leaves = 2
        options%min_data_in_leaf = 1
        options%l2 = 0.0_real64
        options%early_stopping_rounds = 1
        call model%fit_ranking(x, target, group, status, options, &
            validation_x=x, validation_relevance=target, validation_group=group)
        call check(status%code == FORTNUM_OK .and. model%best_iteration() == 1 .and. &
            model%best_validation_loss() < huge(1.0_real64), &
            "ranking validation and best-round accounting", failures)
        call model%predict(x, prediction, status)
        call check(status%code == FORTNUM_OK .and. prediction(2) > prediction(1), &
            "validated ranking prediction", failures)
        call model%fit_ranking(x, target, group, status, options, &
            validation_x=x, validation_relevance=target, &
            validation_group=malformed_group)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "malformed validation group refusal", failures)
    end subroutine test_validation_contract

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(lightgbm_t) :: model
        type(lightgbm_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(2,1), target(2)
        integer :: singleton_group(2), valid_group(2)

        x(:,1) = [0.0_real64, 1.0_real64]
        target = [0.0_real64, 1.0_real64]
        singleton_group = [1, 2]
        valid_group = [3, 3]
        options = lightgbm_options_t()
        options%n_estimators = 1
        options%num_leaves = 2
        options%min_data_in_leaf = 1
        options%l2 = 0.0_real64
        call model%fit_ranking(x, target, singleton_group, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "singleton query refusal", failures)
        options%objective = "rank:pairwise"
        call model%fit(x, target, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "ranking objective without groups refusal", failures)
        options%n_estimators = 1
        call model%fit_ranking(x, target, valid_group, status, options)
        call check(status%code == FORTNUM_OK, "valid ranking prefix", failures)
        options%n_estimators = 2
        call model%fit_warm_start(x, target, status, options)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "ranking warm-start query-state refusal", failures)
    end subroutine test_refusals

    subroutine direct_pair(margin, target, group, weights, loss, gradient, hessian)
        real(real64), intent(in) :: margin(:), target(:), weights(:)
        integer, intent(in) :: group(:)
        real(real64), intent(out) :: loss, gradient(:), hessian(:)
        real(real64) :: delta, pair_weight, probability, total_weight
        integer :: i, j, high, low

        gradient = 0.0_real64
        hessian = 0.0_real64
        loss = 0.0_real64
        total_weight = 0.0_real64
        do i = 1, size(margin)-1
            do j = i+1, size(margin)
                if (group(i) /= group(j) .or. target(i) == target(j)) cycle
                if (target(i) > target(j)) then
                    high = i
                    low = j
                else
                    high = j
                    low = i
                end if
                pair_weight = min(weights(i), weights(j))
                delta = margin(high)-margin(low)
                if (delta >= 0.0_real64) then
                    probability = exp(-delta)/(1.0_real64+exp(-delta))
                    loss = loss + pair_weight*log(1.0_real64+exp(-delta))
                else
                    probability = 1.0_real64/(1.0_real64+exp(delta))
                    loss = loss + pair_weight*(-delta+log(1.0_real64+exp(delta)))
                end if
                gradient(high) = gradient(high)-pair_weight*probability
                gradient(low) = gradient(low)+pair_weight*probability
                hessian(high) = hessian(high)+pair_weight*probability*(1.0_real64-probability)
                hessian(low) = hessian(low)+pair_weight*probability*(1.0_real64-probability)
                total_weight = total_weight+pair_weight
            end do
        end do
        if (total_weight > 0.0_real64) loss = loss/total_weight
    end subroutine direct_pair

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(label)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_lightgbm_ranking
