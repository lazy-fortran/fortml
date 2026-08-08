program test_random_forest_classifier
    !! Independent behavior checks for the deterministic CART ensemble.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_random_forest_classifier, only: random_forest_classifier_t, &
        random_forest_cuda_plan_t, RANDOM_FOREST_MAX_TREES, &
        RANDOM_FOREST_CUDA_PLAN_ABI_VERSION, RANDOM_FOREST_OOB_INSUFFICIENT
    implicit none

    type(random_forest_classifier_t) :: model, repeat_model, other_model
    type(random_forest_cuda_plan_t) :: cuda_plan
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(12, 2), query(3, 2), probabilities(3, 3), &
        repeat_probabilities(3, 3), other_probabilities(3, 3), &
        cuda_probabilities(3, 3), oob_probabilities(12, 3), &
        repeat_oob_probabilities(12, 3), oob_score, sentinel_score, &
        x_small(2, 2), oob_small(2, 2)
    integer :: labels(12), labels_small(2), predictions(3), cuda_predictions(3), failures
    logical :: inclusion(12, 17)

    x = reshape([ &
        -3.0_real64, -2.8_real64, -3.2_real64, -2.9_real64, &
        0.0_real64,  0.2_real64, -0.2_real64,  0.1_real64, &
        3.0_real64,  2.8_real64,  3.2_real64,  2.9_real64, &
        -3.0_real64,  0.1_real64,  3.0_real64,  0.0_real64, &
        -2.8_real64,  0.2_real64,  2.8_real64, -0.1_real64, &
        -3.2_real64, -0.2_real64,  3.2_real64,  0.1_real64], shape(x))
    ! Reshape above is column-major: rows 1--4 are class -5, 5--8 class 0,
    ! and 9--12 class 8 with well-separated first coordinates.
    labels = [-5, -5, -5, -5, 0, 0, 0, 0, 8, 8, 8, 8]
    x_small = x([1, 5], :)
    labels_small = labels([1, 5])
    query(:, 1) = [-3.1_real64, 0.0_real64, 3.1_real64]
    query(:, 2) = [0.0_real64, 0.1_real64, 0.0_real64]
    failures = 0

    call model%fit(x, labels, status, n_trees=17, max_depth=2, seed=123)
    call check(status_ok(status) .and. model%fitted() .and. &
        model%tree_count() == 17 .and. model%class_count() == 3 .and. &
        model%feature_count() == 2 .and. model%random_seed() == 123, &
        "fit metadata", failures)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predictions, status)
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2) - 1.0_real64)) &
        < 2.0e-14_real64 .and. all(probabilities >= 0.0_real64), &
        "probability simplex oracle", failures)
    call check(all(predictions == [-5, 0, 8]), &
        "separated-cluster prediction oracle", failures)

    call repeat_model%fit(x, labels, status, n_trees=17, max_depth=2, seed=123)
    call repeat_model%predict_proba(query, repeat_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(repeat_probabilities - probabilities)) &
        < 2.0e-14_real64, "seeded bootstrap determinism", failures)

    call other_model%fit(x, labels, status, n_trees=17, max_depth=2, seed=124)
    call other_model%predict_proba(query, other_probabilities, status)
    call check(status_ok(status) .and. other_model%random_seed() == 124 .and. &
        maxval(abs(sum(other_probabilities, dim=2) - 1.0_real64)) < 2.0e-14_real64, &
        "alternate seed remains a valid bootstrap ensemble", failures)

    inclusion = model%bootstrap_inclusion()
    call check(count(inclusion) > 0 .and. count(.not. inclusion) > 0 .and. &
        model%oob_coverage() > 0.99_real64, &
        "stored bootstrap inclusion has complete OOB coverage", failures)
    oob_probabilities = -41.0_real64
    call model%oob_decision_function(x, oob_probabilities, status)
    call check(status_ok(status) .and. &
        maxval(abs(sum(oob_probabilities, dim=2) - 1.0_real64)) < 2.0e-14_real64 .and. &
        all(oob_probabilities >= 0.0_real64), &
        "OOB decision probabilities are a simplex", failures)
    call repeat_model%oob_decision_function(x, repeat_oob_probabilities, status)
    call check(status_ok(status) .and. &
        maxval(abs(repeat_oob_probabilities - oob_probabilities)) < 2.0e-14_real64, &
        "OOB predictions are deterministic from stored inclusion", failures)
    oob_score = -9.0_real64
    call model%oob_score(x, labels, oob_score, status)
    call check(status_ok(status) .and. oob_score > 0.90_real64 .and. &
        oob_score <= 1.0_real64, "independent separated-cluster OOB score oracle", failures)

    call model%fit(x, labels, status, n_trees=17, max_depth=2, &
        criterion=2, seed=123)
    call check(status_ok(status), "entropy criterion is accepted", failures)

    call model%fit(x, labels, status, n_trees=0)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "zero-tree refusal", failures)
    call model%fit(x, labels, status, n_trees=RANDOM_FOREST_MAX_TREES + 1)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "tree-count limit refusal", failures)
    call model%fit(x, labels, status, seed=0)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "nonpositive seed refusal", failures)

    ! Re-establish a fitted model after the invalid-fit checks.  The fit API
    ! intentionally uses INTENT(OUT), so a refused fit clears prior state.
    call model%fit(x, labels, status, n_trees=17, max_depth=2, seed=123)
    call check(status_ok(status) .and. model%fitted(), &
        "refit before device contract", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda_probabilities = -37.0_real64
    cuda_predictions = -91
    call model%predict_proba_device(cuda, query, cuda_probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA probability refusal", failures)
    call check(all(cuda_probabilities == -37.0_real64) .and. &
        .not. cuda%resident .and. cuda%host_to_device_transfers == 0 .and. &
        cuda%device_to_host_transfers == 0, &
        "CUDA probability refusal has no hidden host fallback", failures)
    call model%predict_device(cuda, query, cuda_predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA label refusal", failures)
    call check(all(cuda_predictions == -91), &
        "CUDA label refusal preserves the output buffer", failures)
    call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability refusal", failures)

    cuda_probabilities = -47.0_real64
    call model%oob_decision_function_device(cuda, x(1:3, :), cuda_probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(cuda_probabilities == -47.0_real64), &
        "CUDA OOB refusal preserves the output buffer", failures)
    sentinel_score = -13.0_real64
    call model%oob_score_device(cuda, x(1:3, :), labels(1:3), sentinel_score, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. sentinel_score == -13.0_real64, &
        "CUDA OOB score refusal preserves the scalar", failures)

    call other_model%fit(x_small, labels_small, status, n_trees=1, &
        max_depth=0, seed=123)
    oob_small = -53.0_real64
    call other_model%oob_decision_function(x_small, oob_small, status)
    call check(status%code == RANDOM_FOREST_OOB_INSUFFICIENT .and. &
        all(oob_small == -53.0_real64), &
        "insufficient OOB coverage refuses without in-bag fallback", failures)

    call cuda_plan%create(model, cuda, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        cuda_plan%abi() == RANDOM_FOREST_CUDA_PLAN_ABI_VERSION .and. &
        cuda_plan%feature_count() == model%feature_count() .and. &
        cuda_plan%class_count() == model%class_count() .and. &
        cuda_plan%tree_count() == model%tree_count() .and. &
        cuda_plan%device() == cuda%device_index .and. .not. cuda_plan%fitted(), &
        "typed CUDA plan records model shape before refusal", failures)
    cuda_probabilities = -19.0_real64
    cuda_predictions = -23
    call cuda_plan%predict_proba(query, cuda_probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(cuda_probabilities == -19.0_real64), &
        "typed CUDA probability plan preserves output on refusal", failures)
    call cuda_plan%predict(query, cuda_predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(cuda_predictions == -23), &
        "typed CUDA label plan preserves output on refusal", failures)
    call cuda_plan%destroy(status)
    call check(status_ok(status) .and. cuda_plan%abi() == &
        RANDOM_FOREST_CUDA_PLAN_ABI_VERSION .and. .not. cuda_plan%fitted(), &
        "typed CUDA plan destroy", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL random forest classifier cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS random forest classifier independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [random-forest] "//description
        end if
    end subroutine check

end program test_random_forest_classifier
