program test_extra_trees_classifier
    !! Independent behavioral and device-contract checks for Extra Trees.
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_extra_trees_classifier, only: extra_trees_classifier_t, &
        EXTRA_TREES_MAX_TREES
    implicit none

    type(extra_trees_classifier_t) :: model, repeat_model, other_model, leaf_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(12, 2), query(3, 2), probabilities(3, 3), repeat_probabilities(3, 3), &
        other_probabilities(3, 3), leaf_probabilities(1, 2), cuda_probabilities(3, 3)
    integer :: labels(12), predictions(3), cuda_predictions(3), leaf_labels(4), failures
    real(real64) :: nan_value

    x = reshape([ &
        -3.0_real64, -2.8_real64, -3.2_real64, -2.9_real64, &
         0.0_real64,  0.2_real64, -0.2_real64,  0.1_real64, &
         3.0_real64,  2.8_real64,  3.2_real64,  2.9_real64, &
        -3.0_real64,  0.1_real64,  3.0_real64,  0.0_real64, &
        -2.8_real64,  0.2_real64,  2.8_real64, -0.1_real64, &
        -3.2_real64, -0.2_real64,  3.2_real64,  0.1_real64], shape(x))
    labels = [-5, -5, -5, -5, 0, 0, 0, 0, 8, 8, 8, 8]
    query(:, 1) = [-3.1_real64, 0.0_real64, 3.1_real64]
    query(:, 2) = [0.0_real64, 0.1_real64, 0.0_real64]
    failures = 0

    call model%fit(x, labels, status, n_trees=31, max_depth=3, &
        max_features=2, random_splits=64, seed=123)
    call check(status_ok(status) .and. model%fitted() .and. &
        model%tree_count() == 31 .and. model%class_count() == 3 .and. &
        model%feature_count() == 2 .and. model%feature_subsample_count() == 2 .and. &
        model%random_split_count() == 64 .and. model%random_seed() == 123, &
        "fit metadata", failures)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predictions, status)
    call check(status_ok(status) .and. maxval(abs(sum(probabilities, dim=2) - 1.0_real64)) &
        < 2.0e-14_real64 .and. all(probabilities >= 0.0_real64), &
        "probability simplex oracle", failures)
    call check(all(predictions == [-5, 0, 8]), &
        "separated-cluster prediction oracle", failures)

    call repeat_model%fit(x, labels, status, n_trees=31, max_depth=3, &
        max_features=2, random_splits=64, seed=123)
    call repeat_model%predict_proba(query, repeat_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(repeat_probabilities - probabilities)) &
        < 2.0e-14_real64, "seeded randomization determinism", failures)
    call other_model%fit(x, labels, status, n_trees=31, max_depth=3, &
        max_features=2, random_splits=64, seed=124)
    call other_model%predict_proba(query, other_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(sum(other_probabilities, dim=2) - 1.0_real64)) &
        < 2.0e-14_real64, "alternate seed remains a valid ensemble", failures)

    leaf_labels = [2, 2, 9, 9]
    call leaf_model%fit(reshape([0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64], [4, 1]), &
        leaf_labels, status, n_trees=1, max_depth=0, max_features=1, random_splits=1)
    call leaf_model%predict_proba(reshape([1.5_real64], [1, 1]), leaf_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(leaf_probabilities(1, :) - [0.5_real64, 0.5_real64])) &
        < 2.0e-14_real64, "depth-zero empirical leaf oracle", failures)

    call model%fit(x, labels, status, n_trees=0)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero-tree refusal", failures)
    call model%fit(x, labels, status, n_trees=EXTRA_TREES_MAX_TREES + 1)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "tree-count limit refusal", failures)
    nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
    x(1, 1) = nan_value
    call model%fit(x, labels, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "nonfinite input refusal", failures)
    x(1, 1) = -3.0_real64

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.; cuda%available = .true.
    cuda_probabilities = -37.0_real64; cuda_predictions = -91
    call leaf_model%predict_proba_device(cuda, query, cuda_probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(cuda_probabilities == -37.0_real64), "CUDA probability refusal has no fallback", failures)
    call leaf_model%predict_device(cuda, query, cuda_predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(cuda_predictions == -91), &
        "CUDA label refusal preserves output", failures)
    call check(.not. leaf_model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL Extra Trees classifier cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS Extra Trees classifier independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [extra-trees] "//description
        end if
    end subroutine check

end program test_extra_trees_classifier
