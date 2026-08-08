program test_bagging_classifier
    !! Independent behavioral and device-contract checks for bagging.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_bagging_classifier, only: bagging_classifier_t, &
        BAGGING_MAX_ESTIMATORS
    implicit none

    type(bagging_classifier_t) :: model, repeat_model, weighted_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(6, 2), query(3, 2), probabilities(3, 3), &
        repeat_probabilities(3, 3), weighted_probabilities(1, 3), &
        tangent(3, 2), probabilities_dot(3, 3), probabilities_bar(3, 3), &
        x_bar(3, 2), cuda_probabilities(3, 3), sample_weight(6)
    integer :: labels(6), predictions(3), cuda_predictions(3), failures
    integer, allocatable :: classes(:)

    x(:, 1) = [-3.0_real64, -2.0_real64, 0.0_real64, 0.1_real64, &
        2.0_real64, 3.0_real64]
    x(:, 2) = [0.0_real64, 0.1_real64, -0.1_real64, 0.2_real64, &
        -0.2_real64, 0.0_real64]
    labels = [-5, -5, 2, 2, 9, 9]
    query(:, 1) = [-2.5_real64, 0.05_real64, 2.5_real64]
    query(:, 2) = 0.0_real64
    failures = 0

    call model%fit(x, labels, status, n_trees=5, max_depth=0, &
        max_samples=6, bootstrap=.false., seed=1729)
    classes = model%classes()
    call check(status_ok(status) .and. model%fitted() .and. &
        model%tree_count() == 5 .and. model%class_count() == 3 .and. &
        model%feature_count() == 2 .and. model%max_samples() == 6 .and. &
        .not. model%bootstrap() .and. classes(1) == -5 .and. &
        classes(2) == 2 .and. classes(3) == 9, "fit metadata", failures)
    call model%predict_proba(query, probabilities, status)
    call model%predict(query, predictions, status)
    call check(status_ok(status) .and. maxval(abs(probabilities - 1.0_real64/3.0_real64)) &
        < 2.0e-14_real64 .and. all(predictions == -5) .and. &
        maxval(abs(sum(probabilities, dim=2) - 1.0_real64)) < 2.0e-14_real64, &
        "depth-zero empirical class oracle", failures)

    call repeat_model%fit(x, labels, status, n_trees=11, max_depth=2, &
        max_samples=4, bootstrap=.true., seed=1729)
    call repeat_model%predict_proba(query, repeat_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(sum(repeat_probabilities, dim=2) - &
        1.0_real64)) < 2.0e-14_real64, "bootstrap probability simplex", failures)
    call model%fit(x, labels, status, n_trees=11, max_depth=2, max_samples=4, &
        bootstrap=.true., seed=1729)
    call model%predict_proba(query, probabilities, status)
    call check(status_ok(status) .and. maxval(abs(probabilities - repeat_probabilities)) &
        < 2.0e-14_real64, "seeded bootstrap determinism", failures)

    sample_weight = [1.0_real64, 1.0_real64, 1.0_real64, 1.0_real64, &
        4.0_real64, 4.0_real64]
    call weighted_model%fit(x, labels, status, n_trees=3, max_depth=0, &
        max_samples=6, bootstrap=.false., sample_weight=sample_weight)
    call weighted_model%predict_proba(x(1:1, :), weighted_probabilities, status)
    call check(status_ok(status) .and. maxval(abs(weighted_probabilities(1, :) - &
        [1.0_real64/6.0_real64, 1.0_real64/6.0_real64, &
         2.0_real64/3.0_real64])) < 2.0e-14_real64, &
        "weighted empirical class oracle", failures)

    tangent = 1.0_real64
    call weighted_model%predict_proba_jvp(x(1:3, :), tangent, probabilities(1:3, :), &
        probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        maxval(abs(probabilities_dot)) == 0.0_real64, &
        "discrete routing JVP refusal", failures)
    probabilities_bar = 1.0_real64
    call weighted_model%predict_proba_vjp(x(1:3, :), probabilities_bar, x_bar, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        maxval(abs(x_bar)) == 0.0_real64, "discrete routing VJP refusal", failures)

    call model%fit(x, labels, status, n_trees=0)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero-tree refusal", failures)
    call model%fit(x, labels, status, n_trees=BAGGING_MAX_ESTIMATORS + 1)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "tree-count limit refusal", failures)
    call model%fit(x, labels, status, n_trees=2, max_samples=2)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "class-coverage refusal", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    cuda_probabilities = -37.0_real64
    cuda_predictions = -91
    call weighted_model%predict_proba_device(cuda, x(1:3, :), cuda_probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(cuda_probabilities == -37.0_real64), "CUDA probability refusal", failures)
    call weighted_model%predict_device(cuda, x(1:3, :), cuda_predictions, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        all(cuda_predictions == -91), "CUDA label refusal", failures)
    call check(.not. weighted_model%device_supported(FORTML_DEVICE_CUDA), &
        "CUDA capability refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL bagging classifier cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS bagging classifier independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [bagging] "//description
        end if
    end subroutine check

end program test_bagging_classifier
