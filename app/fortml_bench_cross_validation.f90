program fortml_bench_cross_validation
    !! Release workload for weighted cross-validation scoring and FortOpt wiring.
    !! The companion Python lane independently derives fold scores from the
    !! emitted test indices and checks the aggregation before retaining timing.
    use, intrinsic :: iso_fortran_env, only: real64, int64
    use fortml_cross_validation, only: cross_validation_evaluate, &
        cross_validation_result_t
    use fortml_validation, only: kfold_splitter_t, estimator_score_metadata_t, &
        estimator_validation_metadata_t, FORTML_SCORE_INPUT_LABELS, &
        FORTML_SCORE_ACCURACY
    use fortml_estimator_capabilities, only: estimator_capability_t, &
        make_regressor_capabilities
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, FORTNUM_OK, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type :: callback_state_t
        integer :: calls = 0
    end type callback_state_t

    integer, parameter :: n_samples = 17, n_splits = 4, repetitions = 512
    type(kfold_splitter_t) :: splitter
    type(estimator_capability_t) :: capability
    type(estimator_score_metadata_t) :: score
    type(estimator_validation_metadata_t) :: metadata
    type(callback_state_t) :: callback_state
    type(cross_validation_result_t) :: result
    type(fortnum_status_t) :: status
    integer(int64) :: started, finished, clock_rate
    integer :: repetition, fold, environment_status, io_status, unit
    real(real64) :: elapsed
    character(len=1024) :: oracle_path

    capability = make_regressor_capabilities("cv-benchmark", 1, 1, status)
    if (.not. status_ok(status)) error stop "cross-validation capability failed"
    call score%initialize("accuracy", FORTML_SCORE_INPUT_LABELS, status, &
        kind=FORTML_SCORE_ACCURACY, higher_is_better=.true., &
        supports_sample_weight=.true., differentiable=.true.)
    if (.not. status_ok(status)) error stop "cross-validation score failed"
    call metadata%initialize("cv-benchmark", capability, score, status, &
        cloneable=.true., resettable=.false., parameter_count=1)
    if (.not. status_ok(status)) error stop "cross-validation metadata failed"
    call splitter%initialize(n_samples, n_splits, status)
    if (.not. status_ok(status)) error stop "cross-validation splitter failed"

    call system_clock(started, clock_rate)
    do repetition = 1, repetitions
        call cross_validation_evaluate(splitter, metadata, [0.25_dp], &
            callback_state, fold_score, result, status)
        if (.not. status_ok(status) .or. .not. result%complete) then
            error stop "cross-validation scoring failed"
        end if
    end do
    call system_clock(finished)
    elapsed = real(finished - started, real64)/real(clock_rate, real64) / &
        real(repetitions, real64)
    write (*, '(a,",",es24.16)') "cross_validation", elapsed
    write (*, '(a,",",es24.16)') "cross_validation_value", result%value
    write (*, '(a,",",es24.16)') "cross_validation_gradient", result%gradient(1)
    write (*, '(a,",",es24.16)') "cross_validation_objective", result%objective_value
    write (*, '(a,",",i0)') "cross_validation_folds", result%successful_folds

    call get_environment_variable("FORTML_BENCH_CROSS_VALIDATION_ORACLE", &
        oracle_path, status=environment_status)
    if (environment_status /= 0 .or. len_trim(oracle_path) == 0) stop
    open (newunit=unit, file=trim(oracle_path), status="replace", action="write", &
        iostat=io_status)
    if (io_status /= 0) error stop "cross-validation benchmark oracle open failed"
    write (unit, '(a)') "fold,score,gradient,weight"
    do fold = 1, result%fold_count
        write (unit, '(i0,",",es24.16,",",es24.16,",",es24.16)') fold, &
            result%fold_values(fold), result%fold_gradients(1, fold), &
            result%fold_weights(fold)
    end do
    close (unit)

contains

    subroutine fold_score(context, parameters, train_indices, test_indices, &
            fold_index, value, gradient, fold_weight, callback_status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        integer, intent(in) :: train_indices(:), test_indices(:), fold_index
        real(dp), intent(out) :: value, gradient(:), fold_weight
        type(fortnum_status_t), intent(out) :: callback_status
        value = 0.0_dp
        gradient = 0.0_dp
        fold_weight = 0.0_dp
        select type (typed_context => context)
            type is (callback_state_t)
            typed_context%calls = typed_context%calls + 1
            if (size(parameters) /= 1 .or. size(gradient) /= 1 .or. &
                size(train_indices) < 1 .or. size(test_indices) < 1 .or. &
                fold_index < 1) then
                call status_set(callback_status, FORTNUM_NOT_IMPLEMENTED, &
                    "cross-validation callback shape")
                return
            end if
            value = parameters(1) + sum(real(test_indices, dp))/ &
                real(size(test_indices), dp)
            gradient(1) = 1.0_dp
            fold_weight = real(size(test_indices), dp)
            call status_set(callback_status, FORTNUM_OK, "")
        class default
            call status_set(callback_status, FORTNUM_NOT_IMPLEMENTED, &
                "cross-validation callback context")
        end select
    end subroutine fold_score

end program fortml_bench_cross_validation
