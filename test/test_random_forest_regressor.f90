program test_random_forest_regressor
    !! Independent behavioral and device-contract checks for RF regression.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_random_forest_regressor, only: random_forest_regressor_t, &
        RANDOM_FOREST_REGRESSION_MAX_TREES, &
        RANDOM_FOREST_REGRESSION_MODEL_SCHEMA_VERSION
    implicit none

    integer, parameter :: n_samples = 8, n_features = 2, n_outputs = 2
    integer, parameter :: n_trees = 25
    real(real64) :: x(n_samples, n_features), targets(n_samples, n_outputs)
    real(real64) :: query(4, n_features), predictions(4, n_outputs), &
        repeated(4, n_outputs), staged(4, n_trees, n_outputs)
    real(real64) :: x_dot(4, n_features), predictions_dot(4, n_outputs), &
        targets_bar(4, n_outputs), x_bar(4, n_features), importances(n_features), &
        cuda_predictions(4, n_outputs), weights(n_samples)
    logical, allocatable :: inclusion(:, :)
    type(random_forest_regressor_t) :: model, repeat_model
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status
    integer :: i, failures

    x(:, 1) = [-3.0_real64, -2.0_real64, -1.0_real64, 1.0_real64, &
        2.0_real64, 3.0_real64, 4.0_real64, 5.0_real64]
    x(:, 2) = 0.25_real64
    targets(:, 1) = [0.0_real64, 0.0_real64, 0.0_real64, 1.0_real64, &
        1.0_real64, 1.0_real64, 1.0_real64, 1.0_real64]
    targets(:, 2) = 0.5_real64 + 2.0_real64*targets(:, 1)
    query(:, 1) = [-2.7_real64, -0.123_real64, 0.123_real64, 2.7_real64]
    query(:, 2) = 0.25_real64
    x_dot = 1.0_real64
    targets_bar = 1.0_real64
    weights = [1.0_real64, 2.0_real64, 1.0_real64, 3.0_real64, &
        1.0_real64, 1.0_real64, 2.0_real64, 1.0_real64]
    failures = 0

    call model%fit(x, targets, status, n_trees=n_trees, max_depth=3, &
        min_samples_leaf=1, seed=5489, sample_weight=weights)
    call check(status_ok(status) .and. model%fitted() .and. &
        model%feature_count() == n_features .and. model%output_count() == n_outputs .and. &
        model%sample_count() == n_samples .and. model%tree_count() == n_trees .and. &
        model%depth() == 3 .and. model%min_leaf() == 1 .and. &
        model%random_seed() == 5489 .and. model%schema_version() == &
        RANDOM_FOREST_REGRESSION_MODEL_SCHEMA_VERSION, "fit metadata", failures)

    call model%predict(query, predictions, status)
    call check(status_ok(status) .and. all(predictions(:, 2) == &
        0.5_real64 + 2.0_real64*predictions(:, 1)), &
        "multi-output affine target oracle", failures)
    call model%predict_staged(query, staged, status)
    call check(status_ok(status) .and. maxval(abs(staged(:, n_trees, :) - predictions)) &
        < 2.0e-14_real64, "staged final-prefix oracle", failures)
    call check(all(staged(:, 1, 1) >= 0.0_real64) .and. &
        all(staged(:, 1, 1) <= 1.0_real64), &
        "bounded first prefix oracle", failures)

    call repeat_model%fit(x, targets, status, n_trees=n_trees, max_depth=3, &
        min_samples_leaf=1, seed=5489, sample_weight=weights)
    call repeat_model%predict(query, repeated, status)
    call check(status_ok(status) .and. maxval(abs(repeated - predictions)) < 2.0e-14_real64, &
        "seeded determinism oracle", failures)

    inclusion = model%bootstrap_inclusion()
    call check(all(shape(inclusion) == [n_samples, n_trees]) .and. &
        all([(count(inclusion(:, i)) > 0, i=1,n_trees)]), &
        "bootstrap audit state", failures)
    importances = -7.0_real64
    call model%feature_importances(importances, status)
    call check(status_ok(status) .and. importances(1) > 0.999999999999_real64 .and. &
        abs(sum(importances) - 1.0_real64) < 2.0e-14_real64, &
        "split-frequency feature importance oracle", failures)

    call model%predict_jvp(query, x_dot, predictions, predictions_dot, status)
    call check(status_ok(status) .and. maxval(abs(predictions_dot)) == 0.0_real64, &
        "piecewise-constant input JVP oracle", failures)
    call model%predict_vjp(query, targets_bar, x_bar, status)
    call check(status_ok(status) .and. maxval(abs(x_bar)) == 0.0_real64, &
        "piecewise-constant input VJP oracle", failures)

    ! With this seed the first bootstrap contains both sides of the exact
    ! step at x=0, whose best root threshold is zero.  The fixed-state
    ! derivative must therefore refuse the split surface, not guess a zero.
    call model%predict_jvp(reshape([0.0_real64, 0.25_real64], [1, 2]), &
        reshape([1.0_real64, 1.0_real64], [1, 2]), predictions(1:1, :), &
        predictions_dot(1:1, :), status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "split-boundary JVP refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda_predictions = -31.0_real64
    call model%predict_device(cuda, query, cuda_predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(cuda_predictions == -31.0_real64), &
        "typed CUDA refusal preserves output", failures)
    call check(.not. model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability refusal", failures)

    predictions = -19.0_real64
    call model%fit(x, targets, status, n_trees=0)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. &
        all(predictions == -19.0_real64), "transactional invalid fit", failures)
    call model%predict(query, predictions, status)
    call check(status_ok(status) .and. maxval(abs(predictions - repeated)) < 2.0e-14_real64, &
        "state retained after invalid fit", failures)
    call model%fit(x, targets, status, n_trees=RANDOM_FOREST_REGRESSION_MAX_TREES + 1)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "tree-count limit refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') &
            "FAIL random forest regression cases: ", failures
        error stop 1
    end if
    write (*, '(a)') &
        "PASS random forest regression independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [random-forest-regression] "//description
        end if
    end subroutine check

end program test_random_forest_regressor
