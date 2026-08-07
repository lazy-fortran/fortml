program test_random_forest_classifier
    !! Independent behavior checks for the deterministic CART ensemble.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_random_forest_classifier, only: random_forest_classifier_t, &
        RANDOM_FOREST_MAX_TREES
    implicit none

    type(random_forest_classifier_t) :: model, repeat_model, other_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(12, 2), query(3, 2), probabilities(3, 3), &
        repeat_probabilities(3, 3), other_probabilities(3, 3)
    integer :: labels(12), predictions(3), failures

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

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, query, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA probability refusal", failures)
    call model%predict_device(cuda, query, predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA label refusal", failures)
    call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability refusal", failures)

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
