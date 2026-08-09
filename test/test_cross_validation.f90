program test_cross_validation
    use fortml_cross_validation, only: cross_validation_evaluate, &
        cross_validation_result_t, cross_validation_options_t, &
        cross_validation_objective_t, &
        FORTML_CV_WEIGHTED_MEAN, FORTML_CV_WEIGHTED_SUM
    use fortml_validation, only: kfold_splitter_t, estimator_score_metadata_t, &
        estimator_validation_metadata_t, FORTML_SCORE_INPUT_LABELS, &
        FORTML_SCORE_ACCURACY
    use fortml_estimator_capabilities, only: estimator_capability_t, &
        make_regressor_capabilities
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    implicit none

    type :: oracle_context_t
        integer :: callback_count = 0
    end type oracle_context_t

    integer :: failures

    failures = 0
    call test_weighted_mean_and_sum(failures)
    call test_fortopt_adapter(failures)
    call test_leakage_and_device_refusals(failures)
    if (failures > 0) error stop "cross-validation tests failed"
    write (*, '(a)') "PASS cross-validation independent behavioral oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

    subroutine make_metadata(metadata, status, cloneable, resettable)
        type(estimator_validation_metadata_t), intent(out) :: metadata
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in) :: cloneable, resettable
        type(estimator_capability_t) :: capability
        type(estimator_score_metadata_t) :: score

        capability = make_regressor_capabilities("cv-oracle", 1, 1, status)
        if (.not. status_ok(status)) return
        call score%initialize("accuracy", FORTML_SCORE_INPUT_LABELS, status, &
            kind=FORTML_SCORE_ACCURACY, higher_is_better=.true., &
            supports_sample_weight=.true., differentiable=.true.)
        if (.not. status_ok(status)) return
        call metadata%initialize("cv-oracle", capability, score, status, &
            cloneable=cloneable, resettable=resettable, parameter_count=1)
    end subroutine make_metadata

    subroutine test_weighted_mean_and_sum(failures)
        integer, intent(inout) :: failures
        type(kfold_splitter_t), target :: splitter
        type(estimator_validation_metadata_t) :: metadata
        type(cross_validation_result_t) :: result, sum_result
        type(cross_validation_options_t) :: options
        type(oracle_context_t), target :: context
        type(fortnum_status_t) :: status

        call make_metadata(metadata, status, cloneable=.true., resettable=.false.)
        call check(status_ok(status), "metadata oracle initialization", failures)
        call splitter%initialize(7, 3, status)
        call check(status_ok(status), "splitter oracle initialization", failures)
        call cross_validation_evaluate(splitter, metadata, [0.5_dp], context, &
            oracle_fold, result, status)
        call check(status_ok(status) .and. result%complete, &
            "weighted mean evaluation", failures)
        call check(result%successful_folds == 3 .and. context%callback_count == 3, &
            "all folds and fresh callback invocations", failures)
        ! The held-out fold means are 2, 4.5, and 6.5 with weights 3, 2, 2.
        call check(abs(result%value - 4.5_dp) < 1.0e-13_dp, &
            "weighted mean independent oracle", failures)
        call check(abs(result%gradient(1) - 1.0_dp) < 1.0e-13_dp .and. &
            abs(result%objective_value + 4.5_dp) < 1.0e-13_dp .and. &
            abs(result%objective_gradient(1) + 1.0_dp) < 1.0e-13_dp, &
            "oriented FortOpt products", failures)
        call check(abs(result%fold_values(1) - 2.5_dp) < 1.0e-13_dp .and. &
            abs(result%fold_weights(1) - 3.0_dp) < 1.0e-13_dp, &
            "fold diagnostics", failures)

        options%aggregation = FORTML_CV_WEIGHTED_SUM
        context%callback_count = 0
        call cross_validation_evaluate(splitter, metadata, [0.5_dp], context, &
            oracle_fold, sum_result, status, options)
        call check(status_ok(status), "weighted sum evaluation", failures)
        call check(abs(sum_result%value - 31.5_dp) < 1.0e-13_dp, &
            "weighted sum independent oracle", failures)
    end subroutine test_weighted_mean_and_sum

    subroutine test_fortopt_adapter(failures)
        integer, intent(inout) :: failures
        type(kfold_splitter_t), target :: splitter
        type(estimator_validation_metadata_t) :: metadata
        type(cross_validation_objective_t), target :: context_objective
        type(oracle_context_t), target :: callback_context
        type(objective_t) :: objective
        type(fortnum_status_t) :: status
        real(dp) :: value, gradient(1)

        call make_metadata(metadata, status, cloneable=.true., resettable=.false.)
        call splitter%initialize(7, 3, status)
        call context_objective%initialize(splitter, metadata, callback_context, &
            oracle_fold, status)
        call check(status_ok(status) .and. context_objective%initialized(), &
            "cross-validation objective initialization", failures)
        call context_objective%as_objective(objective, status)
        call check(status_ok(status), "FortOpt objective adapter initialization", failures)
        call objective%value_gradient([0.5_dp], value, gradient, status)
        call check(status_ok(status), "FortOpt objective callback", failures)
        call check(abs(value + 4.5_dp) < 1.0e-13_dp .and. &
            abs(gradient(1) + 1.0_dp) < 1.0e-13_dp, &
            "FortOpt objective independent oracle", failures)
    end subroutine test_fortopt_adapter

    subroutine test_leakage_and_device_refusals(failures)
        integer, intent(inout) :: failures
        type(kfold_splitter_t) :: splitter
        type(estimator_validation_metadata_t) :: metadata
        type(cross_validation_result_t) :: result
        type(oracle_context_t) :: context
        type(fortml_device_t) :: cuda
        type(fortnum_status_t) :: status

        call make_metadata(metadata, status, cloneable=.false., resettable=.false.)
        call splitter%initialize(7, 3, status)
        call cross_validation_evaluate(splitter, metadata, [0.5_dp], context, &
            oracle_fold, result, status)
        call check(.not. status_ok(status) .and. context%callback_count == 0, &
            "undeclared clone/reset leakage refusal", failures)

        call make_metadata(metadata, status, cloneable=.true., resettable=.false.)
        cuda%kind = FORTML_DEVICE_CUDA
        call cross_validation_evaluate(splitter, metadata, [0.5_dp], context, &
            oracle_fold, result, status, device=cuda)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
            context%callback_count == 0, "typed CUDA refusal", failures)
    end subroutine test_leakage_and_device_refusals

    subroutine oracle_fold(context, parameters, train_indices, test_indices, &
            fold_index, score, gradient, fold_weight, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        integer, intent(in) :: train_indices(:), test_indices(:), fold_index
        real(dp), intent(out) :: score, gradient(:), fold_weight
        type(fortnum_status_t), intent(out) :: status
        score = 0.0_dp
        gradient = 0.0_dp
        fold_weight = 0.0_dp
        select type (typed_context => context)
            type is (oracle_context_t)
            typed_context%callback_count = typed_context%callback_count + 1
            if (size(parameters) /= 1 .or. size(gradient) /= 1 .or. &
                size(test_indices) < 1 .or. size(train_indices) < 1) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, "oracle shape")
                return
            end if
            score = parameters(1) + sum(real(test_indices, dp))/real(size(test_indices), dp)
            gradient(1) = 1.0_dp
            fold_weight = real(size(test_indices), dp)
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, "oracle context")
        end select
    end subroutine oracle_fold

end program test_cross_validation
