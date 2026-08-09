module fortml_cross_validation
    !! Leakage-safe cross-validation scoring and FortOpt adapters.
    !!
    !! Splitters remain index-only and estimator implementations stay out of
    !! this module.  A typed callback receives one fresh train/test partition
    !! at a time and returns the raw fold score, its parameter gradient, and a
    !! positive fold weight.  The metadata record is the contract boundary:
    !! clone/reset declarations are checked before any callback is invoked,
    !! while scorer orientation and weighted aggregation are handled once in
    !! this module.  The callback owns fitting a fresh clone or resetting a
    !! model before each fold; reusing fitted state is never silently allowed.
    !!
    !! The control plane is CPU-only.  A CUDA request is refused explicitly;
    !! no host callback is hidden behind a device selection.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_validation, only: kfold_splitter_t, &
        stratified_kfold_splitter_t, group_kfold_splitter_t, &
        time_series_splitter_t, estimator_validation_metadata_t
    use fortopt_objective, only: objective_t
    implicit none
    private

    integer, parameter, public :: FORTML_CV_WEIGHTED_MEAN = 1
    integer, parameter, public :: FORTML_CV_WEIGHTED_SUM = 2

    !> One fold callback.  `score` is in the scorer's natural orientation;
    !> `gradient` is d(score)/d(parameters), and `fold_weight` is normally the
    !> number or total sample weight of the held-out rows.
    abstract interface
        subroutine cross_validation_fold_proc(context, parameters, &
                train_indices, test_indices, fold_index, score, gradient, &
                fold_weight, status)
            import :: dp, fortnum_status_t
            class(*), intent(inout) :: context
            real(dp), intent(in) :: parameters(:)
            integer, intent(in) :: train_indices(:), test_indices(:)
            integer, intent(in) :: fold_index
            real(dp), intent(out) :: score, gradient(:), fold_weight
            type(fortnum_status_t), intent(out) :: status
        end subroutine cross_validation_fold_proc
    end interface

    type, public :: cross_validation_options_t
        integer :: aggregation = FORTML_CV_WEIGHTED_MEAN
        logical :: require_clone = .false.
        logical :: require_reset = .false.
    end type cross_validation_options_t

    type, public :: cross_validation_result_t
        character(len=64) :: scorer_name = ""
        integer :: aggregation = FORTML_CV_WEIGHTED_MEAN
        integer :: parameter_count = 0
        integer :: fold_count = 0
        integer :: successful_folds = 0
        real(dp) :: value = huge(1.0_dp)
        real(dp) :: oriented_value = huge(1.0_dp)
        real(dp) :: objective_value = huge(1.0_dp)
        real(dp) :: weight_sum = 0.0_dp
        logical :: complete = .false.
        logical :: objective_compatible = .false.
        logical :: clone_guard_passed = .false.
        logical :: reset_guard_passed = .false.
        real(dp), allocatable :: gradient(:)
        real(dp), allocatable :: objective_gradient(:)
        real(dp), allocatable :: fold_values(:)
        real(dp), allocatable :: fold_weights(:)
        real(dp), allocatable :: fold_gradients(:, :)
    contains
        procedure, public :: clear => cross_validation_result_clear
        procedure, public :: best_value => cross_validation_result_best_value
    end type cross_validation_result_t

    !> Reusable objective context for FortOpt grid/random/L-BFGS-B search.
    !! The splitter and estimator callback are intentionally borrowed; callers
    !! must keep them alive for the lifetime of this object and any objective
    !! created from it.
    type, public :: cross_validation_objective_t
        private
        integer :: parameter_count_ = 0
        type(estimator_validation_metadata_t) :: metadata
        type(cross_validation_options_t) :: options
        class(*), pointer :: splitter => null()
        class(*), pointer :: context => null()
        procedure(cross_validation_fold_proc), pointer, nopass :: fold_proc => null()
        logical :: ready = .false.
    contains
        procedure, public :: initialize => cross_validation_objective_initialize
        procedure, public :: initialized => cross_validation_objective_initialized
        procedure, public :: evaluate => cross_validation_objective_evaluate
        procedure, public :: as_objective => cross_validation_objective_as_objective
        procedure, public :: parameter_count => &
            cross_validation_objective_parameter_count
    end type cross_validation_objective_t

    interface cross_validation_evaluate
        module procedure cross_validation_evaluate_kfold
        module procedure cross_validation_evaluate_stratified
        module procedure cross_validation_evaluate_group
        module procedure cross_validation_evaluate_time_series
    end interface cross_validation_evaluate

    public :: cross_validation_fold_proc
    public :: cross_validation_evaluate
    public :: cross_validation_objective_context_proc

    abstract interface
        subroutine cross_validation_objective_context_proc(context, parameters, &
                value, gradient, status)
            import :: dp, fortnum_status_t
            class(*), intent(inout) :: context
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: value, gradient(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine cross_validation_objective_context_proc
    end interface

contains

    subroutine cross_validation_result_clear(self)
        class(cross_validation_result_t), intent(inout) :: self

        self%scorer_name = ""
        self%aggregation = FORTML_CV_WEIGHTED_MEAN
        self%parameter_count = 0
        self%fold_count = 0
        self%successful_folds = 0
        self%value = huge(1.0_dp)
        self%oriented_value = huge(1.0_dp)
        self%objective_value = huge(1.0_dp)
        self%weight_sum = 0.0_dp
        self%complete = .false.
        self%objective_compatible = .false.
        self%clone_guard_passed = .false.
        self%reset_guard_passed = .false.
        if (allocated(self%gradient)) deallocate(self%gradient)
        if (allocated(self%objective_gradient)) deallocate(self%objective_gradient)
        if (allocated(self%fold_values)) deallocate(self%fold_values)
        if (allocated(self%fold_weights)) deallocate(self%fold_weights)
        if (allocated(self%fold_gradients)) deallocate(self%fold_gradients)
    end subroutine cross_validation_result_clear

    real(dp) function cross_validation_result_best_value(self) result(value)
        class(cross_validation_result_t), intent(in) :: self

        value = self%oriented_value
    end function cross_validation_result_best_value

    subroutine cross_validation_evaluate_kfold(splitter, metadata, parameters, &
            context, fold_proc, result, status, options, device)
        type(kfold_splitter_t), intent(inout) :: splitter
        type(estimator_validation_metadata_t), intent(in) :: metadata
        real(dp), intent(in) :: parameters(:)
        class(*), intent(inout) :: context
        procedure(cross_validation_fold_proc) :: fold_proc
        type(cross_validation_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(cross_validation_options_t), intent(in), optional :: options
        type(fortml_device_t), intent(in), optional :: device
        integer, allocatable :: train_indices(:), test_indices(:)
        logical :: has_split
        integer :: fold

        call begin_evaluation(metadata, parameters, result, status, options, device, &
            splitter%fold_count())
        if (status%code /= FORTNUM_OK) return
        call splitter%reset()
        do fold = 1, splitter%fold_count()
            call splitter%next_split(train_indices, test_indices, has_split, status)
            if (status%code /= FORTNUM_OK) return
            if (.not. has_split) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "cross-validation: splitter ended before fold count")
                return
            end if
            call accumulate_fold(metadata, result, parameters, context, fold_proc, &
                train_indices, test_indices, fold, status)
            if (status%code /= FORTNUM_OK) return
        end do
        call finish_evaluation(metadata, result, status)
    end subroutine cross_validation_evaluate_kfold

    subroutine cross_validation_evaluate_stratified(splitter, metadata, parameters, &
            context, fold_proc, result, status, options, device)
        type(stratified_kfold_splitter_t), intent(inout) :: splitter
        type(estimator_validation_metadata_t), intent(in) :: metadata
        real(dp), intent(in) :: parameters(:)
        class(*), intent(inout) :: context
        procedure(cross_validation_fold_proc) :: fold_proc
        type(cross_validation_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(cross_validation_options_t), intent(in), optional :: options
        type(fortml_device_t), intent(in), optional :: device
        integer, allocatable :: train_indices(:), test_indices(:)
        logical :: has_split
        integer :: fold

        call begin_evaluation(metadata, parameters, result, status, options, device, &
            splitter%fold_count())
        if (status%code /= FORTNUM_OK) return
        call splitter%reset()
        do fold = 1, splitter%fold_count()
            call splitter%next_split(train_indices, test_indices, has_split, status)
            if (status%code /= FORTNUM_OK) return
            if (.not. has_split) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "cross-validation: splitter ended before fold count")
                return
            end if
            call accumulate_fold(metadata, result, parameters, context, fold_proc, &
                train_indices, test_indices, fold, status)
            if (status%code /= FORTNUM_OK) return
        end do
        call finish_evaluation(metadata, result, status)
    end subroutine cross_validation_evaluate_stratified

    subroutine cross_validation_evaluate_group(splitter, metadata, parameters, &
            context, fold_proc, result, status, options, device)
        type(group_kfold_splitter_t), intent(inout) :: splitter
        type(estimator_validation_metadata_t), intent(in) :: metadata
        real(dp), intent(in) :: parameters(:)
        class(*), intent(inout) :: context
        procedure(cross_validation_fold_proc) :: fold_proc
        type(cross_validation_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(cross_validation_options_t), intent(in), optional :: options
        type(fortml_device_t), intent(in), optional :: device
        integer, allocatable :: train_indices(:), test_indices(:)
        logical :: has_split
        integer :: fold

        call begin_evaluation(metadata, parameters, result, status, options, device, &
            splitter%fold_count())
        if (status%code /= FORTNUM_OK) return
        call splitter%reset()
        do fold = 1, splitter%fold_count()
            call splitter%next_split(train_indices, test_indices, has_split, status)
            if (status%code /= FORTNUM_OK) return
            if (.not. has_split) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "cross-validation: splitter ended before fold count")
                return
            end if
            call accumulate_fold(metadata, result, parameters, context, fold_proc, &
                train_indices, test_indices, fold, status)
            if (status%code /= FORTNUM_OK) return
        end do
        call finish_evaluation(metadata, result, status)
    end subroutine cross_validation_evaluate_group

    subroutine cross_validation_evaluate_time_series(splitter, metadata, parameters, &
            context, fold_proc, result, status, options, device)
        type(time_series_splitter_t), intent(inout) :: splitter
        type(estimator_validation_metadata_t), intent(in) :: metadata
        real(dp), intent(in) :: parameters(:)
        class(*), intent(inout) :: context
        procedure(cross_validation_fold_proc) :: fold_proc
        type(cross_validation_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(cross_validation_options_t), intent(in), optional :: options
        type(fortml_device_t), intent(in), optional :: device
        integer, allocatable :: train_indices(:), test_indices(:)
        logical :: has_split
        integer :: fold

        call begin_evaluation(metadata, parameters, result, status, options, device, &
            splitter%fold_count())
        if (status%code /= FORTNUM_OK) return
        call splitter%reset()
        do fold = 1, splitter%fold_count()
            call splitter%next_split(train_indices, test_indices, has_split, status)
            if (status%code /= FORTNUM_OK) return
            if (.not. has_split) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "cross-validation: splitter ended before fold count")
                return
            end if
            call accumulate_fold(metadata, result, parameters, context, fold_proc, &
                train_indices, test_indices, fold, status)
            if (status%code /= FORTNUM_OK) return
        end do
        call finish_evaluation(metadata, result, status)
    end subroutine cross_validation_evaluate_time_series

    subroutine begin_evaluation(metadata, parameters, result, status, options, device, &
            fold_count)
        type(estimator_validation_metadata_t), intent(in) :: metadata
        real(dp), intent(in) :: parameters(:)
        type(cross_validation_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(cross_validation_options_t), intent(in), optional :: options
        type(fortml_device_t), intent(in), optional :: device
        integer, intent(in) :: fold_count
        type(cross_validation_options_t) :: settings
        type(cross_validation_options_t) :: cross_validation_options_t_default

        call result%clear()
        settings = cross_validation_options_t_default
        if (present(options)) settings = options
        if (.not. metadata%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: estimator metadata is invalid")
            return
        end if
        if (size(parameters) /= metadata%parameter_count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: parameter shape does not match metadata")
            return
        end if
        if (fold_count < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: at least two folds are required")
            return
        end if
        if (settings%aggregation /= FORTML_CV_WEIGHTED_MEAN .and. &
            settings%aggregation /= FORTML_CV_WEIGHTED_SUM) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: aggregation mode is invalid")
            return
        end if
        if (settings%require_clone .and. .not. metadata%can_clone()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: scorer requires a cloneable estimator")
            return
        end if
        if (settings%require_reset .and. .not. metadata%can_reset()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: scorer requires a resettable estimator")
            return
        end if
        if (.not. metadata%can_clone() .and. .not. metadata%can_reset()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: clone/reset leakage guard is not declared")
            return
        end if
        if (any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: parameters must be finite")
            return
        end if
        if (present(device)) then
            if (device%kind == FORTML_DEVICE_CUDA) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "cross-validation: CUDA index/scoring control plane is not implemented")
                return
            else if (device%kind /= FORTML_DEVICE_CPU .and. device%selected) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "cross-validation: device kind is invalid")
                return
            end if
        end if
        result%scorer_name = metadata%score%name
        result%aggregation = settings%aggregation
        result%parameter_count = metadata%parameter_count
        result%fold_count = fold_count
        result%clone_guard_passed = metadata%can_clone()
        result%reset_guard_passed = metadata%can_reset()
        result%objective_compatible = metadata%score%differentiable .and. &
            metadata%parameter_count > 0
        allocate(result%gradient(metadata%parameter_count), &
            result%objective_gradient(metadata%parameter_count), &
            result%fold_values(fold_count), result%fold_weights(fold_count), &
            result%fold_gradients(metadata%parameter_count, fold_count))
        result%gradient = 0.0_dp
        result%objective_gradient = 0.0_dp
        result%fold_values = 0.0_dp
        result%fold_weights = 0.0_dp
        result%fold_gradients = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine begin_evaluation

    subroutine accumulate_fold(metadata, result, parameters, context, fold_proc, &
            train_indices, test_indices, fold, status)
        type(estimator_validation_metadata_t), intent(in) :: metadata
        type(cross_validation_result_t), intent(inout) :: result
        real(dp), intent(in) :: parameters(:)
        class(*), intent(inout) :: context
        procedure(cross_validation_fold_proc) :: fold_proc
        integer, intent(in) :: train_indices(:), test_indices(:), fold
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: score, fold_weight
        real(dp), allocatable :: gradient(:)

        allocate(gradient(result%parameter_count))
        score = 0.0_dp
        fold_weight = 0.0_dp
        gradient = 0.0_dp
        call fold_proc(context, parameters, train_indices, test_indices, fold, score, &
            gradient, fold_weight, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. ieee_is_finite(score) .or. .not. ieee_is_finite(fold_weight) .or. &
            fold_weight <= 0.0_dp .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: fold callback returned a nonfinite value")
            return
        end if
        result%fold_values(fold) = score
        result%fold_weights(fold) = fold_weight
        result%fold_gradients(:, fold) = gradient
        result%successful_folds = result%successful_folds + 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine accumulate_fold

    subroutine finish_evaluation(metadata, result, status)
        type(estimator_validation_metadata_t), intent(in) :: metadata
        type(cross_validation_result_t), intent(inout) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: denominator, orientation
        integer :: fold

        if (result%successful_folds /= result%fold_count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: not all folds were evaluated")
            return
        end if
        result%weight_sum = sum(result%fold_weights)
        if (.not. ieee_is_finite(result%weight_sum) .or. result%weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation: fold weights must have a positive sum")
            return
        end if
        denominator = 1.0_dp
        if (result%aggregation == FORTML_CV_WEIGHTED_MEAN) denominator = result%weight_sum
        result%value = 0.0_dp
        result%gradient = 0.0_dp
        do fold = 1, result%fold_count
            result%value = result%value + result%fold_weights(fold)* &
                result%fold_values(fold)
            result%gradient = result%gradient + result%fold_weights(fold)* &
                result%fold_gradients(:, fold)
        end do
        result%value = result%value/denominator
        result%gradient = result%gradient/denominator
        orientation = 1.0_dp
        if (.not. metadata%score%higher_is_better) orientation = -1.0_dp
        result%oriented_value = orientation*result%value
        result%objective_value = -result%oriented_value
        result%objective_gradient = -orientation*result%gradient
        result%complete = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine finish_evaluation

    subroutine cross_validation_objective_initialize(self, splitter, metadata, &
            context, fold_proc, status, options)
        class(cross_validation_objective_t), intent(out) :: self
        class(*), target, intent(inout) :: splitter
        type(estimator_validation_metadata_t), intent(in) :: metadata
        class(*), target, intent(inout) :: context
        procedure(cross_validation_fold_proc) :: fold_proc
        type(fortnum_status_t), intent(out) :: status
        type(cross_validation_options_t), intent(in), optional :: options
        logical :: supported
        type(cross_validation_options_t) :: cross_validation_options_t_default

        self%parameter_count_ = 0
        nullify(self%splitter, self%context, self%fold_proc)
        self%ready = .false.
        supported = .false.
        select type (splitter)
            type is (kfold_splitter_t)
            supported = .true.
            type is (stratified_kfold_splitter_t)
            supported = .true.
            type is (group_kfold_splitter_t)
            supported = .true.
            type is (time_series_splitter_t)
            supported = .true.
        class default
            supported = .false.
        end select
        if (.not. supported) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation objective: unsupported splitter type")
            return
        end if
        if (.not. metadata%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation objective: estimator metadata is invalid")
            return
        end if
        if (.not. metadata%score%differentiable .or. metadata%parameter_count < 1) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "cross-validation objective: scorer has no differentiable parameter path")
            return
        end if
        if (.not. metadata%can_clone() .and. .not. metadata%can_reset()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation objective: clone/reset leakage guard is not declared")
            return
        end if
        self%parameter_count_ = metadata%parameter_count
        self%metadata = metadata
        self%options = cross_validation_options_t_default
        if (present(options)) self%options = options
        self%splitter => splitter
        self%context => context
        self%fold_proc => fold_proc
        self%ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine cross_validation_objective_initialize

    logical function cross_validation_objective_initialized(self) result(value)
        class(cross_validation_objective_t), intent(in) :: self

        value = self%ready .and. associated(self%splitter) .and. &
            associated(self%context) .and. associated(self%fold_proc)
    end function cross_validation_objective_initialized

    integer function cross_validation_objective_parameter_count(self) result(value)
        class(cross_validation_objective_t), intent(in) :: self

        value = self%parameter_count_
    end function cross_validation_objective_parameter_count

    subroutine cross_validation_objective_evaluate(self, parameters, result, status)
        class(cross_validation_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(cross_validation_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call result%clear()
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation objective: object is not initialized")
            return
        end if
        if (size(parameters) /= self%parameter_count_) then
            call result%clear()
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation objective: parameter shape is invalid")
            return
        end if
        select type (splitter => self%splitter)
            type is (kfold_splitter_t)
            call cross_validation_evaluate(splitter, self%metadata, parameters, &
                self%context, self%fold_proc, result, status, self%options)
            type is (stratified_kfold_splitter_t)
            call cross_validation_evaluate(splitter, self%metadata, parameters, &
                self%context, self%fold_proc, result, status, self%options)
            type is (group_kfold_splitter_t)
            call cross_validation_evaluate(splitter, self%metadata, parameters, &
                self%context, self%fold_proc, result, status, self%options)
            type is (time_series_splitter_t)
            call cross_validation_evaluate(splitter, self%metadata, parameters, &
                self%context, self%fold_proc, result, status, self%options)
        class default
            call result%clear()
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation objective: splitter type was lost")
        end select
    end subroutine cross_validation_objective_evaluate

    subroutine cross_validation_objective_as_objective(self, objective, status)
        class(cross_validation_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation objective: object is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count_, self, &
            cross_validation_objective_callback, status)
    end subroutine cross_validation_objective_as_objective

    subroutine cross_validation_objective_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        type(cross_validation_result_t) :: result

        value = huge(1.0_dp)
        gradient = 0.0_dp
        select type (context)
            type is (cross_validation_objective_t)
            call context%evaluate(parameters, result, status)
            if (status%code /= FORTNUM_OK) return
            if (.not. result%objective_compatible) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "cross-validation objective: result is not FortOpt-compatible")
                return
            end if
            value = result%objective_value
            gradient = result%objective_gradient
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "cross-validation objective: callback context has wrong type")
        end select
    end subroutine cross_validation_objective_callback

end module fortml_cross_validation
