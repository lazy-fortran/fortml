module fortml_calibrated_softmax_classifier
    !! Leakage-safe multiclass temperature calibration with stratified OOF logits.
    !!
    !! ``calibrated_softmax_classifier_t`` fits a softmax model on each
    !! stratified training fold, collects one held-out logit row per sample,
    !! fits a single positive temperature on those logits, and finally fits
    !! the deployment softmax model on all rows.  The calibration head is
    !! therefore never trained on in-sample logits.  The packed parameter
    !! vector is the softmax parameters followed by the positive temperature.
    !! CPU prediction and exact fixed-state input/parameter JVP and VJP
    !! products are available.  CUDA requests return a typed refusal until a
    !! resident softmax-plus-calibration kernel is linked.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_softmax_regression, only: softmax_regression_t
    use fortml_probability_calibration, only: &
        multiclass_probability_calibrator_t, probability_calibration_options_t, &
        probability_calibration_state_t, CALIBRATION_TEMPERATURE
    use fortml_validation, only: stratified_kfold_splitter_t
    implicit none
    private

    type, public :: calibrated_softmax_classifier_options_t
        !! Softmax, calibration, and deterministic OOF controls.
        real(dp) :: l2 = 1.0_dp
        real(dp) :: tolerance = 1.0e-8_dp
        integer :: max_iterations = 200
        logical :: fit_intercept = .true.
        integer :: cv_folds = 5
        logical :: cv_shuffle = .false.
        integer :: cv_seed = 17
        type(probability_calibration_options_t) :: calibration = &
            probability_calibration_options_t(method=CALIBRATION_TEMPERATURE)
    end type calibrated_softmax_classifier_options_t

    type, public :: calibrated_softmax_classifier_state_t
        integer :: cv_folds = 0
        integer :: cv_samples = 0
        integer :: class_count = 0
        integer :: calibration_iterations = 0
        logical :: cv_converged = .false.
        logical :: classifier_converged = .false.
        logical :: calibration_converged = .false.
        logical :: converged = .false.
        real(dp) :: oof_log_loss = huge(1.0_dp)
        real(dp) :: calibrated_oof_log_loss = huge(1.0_dp)
        real(dp) :: calibration_objective = huge(1.0_dp)
    end type calibrated_softmax_classifier_state_t

    type, public :: calibrated_softmax_classifier_t
        private
        type(softmax_regression_t) :: classifier
        type(multiclass_probability_calibrator_t) :: calibrator
        integer :: calibration_method_code = CALIBRATION_TEMPERATURE
        integer :: cv_fold_count = 0
        real(dp) :: oof_log_loss_value = huge(1.0_dp)
        real(dp) :: calibrated_oof_log_loss_value = huge(1.0_dp)
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => calibrated_softmax_fit
        procedure, public :: decision_function => calibrated_softmax_decision
        procedure, public :: decision_function_device => &
            calibrated_softmax_decision_device
        procedure, public :: decision_function_jvp => calibrated_softmax_decision_jvp
        procedure, public :: decision_function_vjp => calibrated_softmax_decision_vjp
        procedure, public :: predict_proba => calibrated_softmax_predict_proba
        procedure, public :: predict_proba_device => calibrated_softmax_predict_proba_device
        procedure, public :: predict_proba_jvp => calibrated_softmax_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            calibrated_softmax_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => calibrated_softmax_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            calibrated_softmax_predict_proba_parameter_vjp
        procedure, public :: predict => calibrated_softmax_predict
        procedure, public :: classes => calibrated_softmax_classes
        procedure, public :: feature_count => calibrated_softmax_feature_count
        procedure, public :: class_count => calibrated_softmax_class_count
        procedure, public :: parameter_count => calibrated_softmax_parameter_count
        procedure, public :: parameters => calibrated_softmax_parameters
        procedure, public :: set_parameters => calibrated_softmax_set_parameters
        procedure, public :: fitted => calibrated_softmax_fitted
        procedure, public :: device_supported => calibrated_softmax_device_supported
        procedure, public :: calibration_method => calibrated_softmax_method
        procedure, public :: cv_folds => calibrated_softmax_cv_folds
        procedure, public :: oof_log_loss => calibrated_softmax_oof_log_loss
        procedure, public :: calibrated_oof_log_loss => &
            calibrated_softmax_calibrated_oof_log_loss
        procedure, public :: temperature => calibrated_softmax_temperature
    end type calibrated_softmax_classifier_t

    public :: calibrated_softmax_fit
    public :: calibrated_softmax_decision
    public :: calibrated_softmax_decision_device
    public :: calibrated_softmax_decision_jvp
    public :: calibrated_softmax_decision_vjp
    public :: calibrated_softmax_predict_proba
    public :: calibrated_softmax_predict_proba_device
    public :: calibrated_softmax_predict_proba_jvp
    public :: calibrated_softmax_predict_proba_parameter_jvp
    public :: calibrated_softmax_predict_proba_vjp
    public :: calibrated_softmax_predict_proba_parameter_vjp
    public :: calibrated_softmax_predict

contains

    subroutine calibrated_softmax_fit(self, x, labels, status, options, state, &
            sample_weight, class_weight)
        class(calibrated_softmax_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(calibrated_softmax_classifier_options_t), intent(in), optional :: options
        type(calibrated_softmax_classifier_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        type(calibrated_softmax_classifier_options_t) :: config
        type(calibrated_softmax_classifier_state_t) :: result
        type(probability_calibration_state_t) :: calibration_state
        type(stratified_kfold_splitter_t) :: splitter
        type(softmax_regression_t) :: fold_model
        integer, allocatable :: train_indices(:), test_indices(:), classes(:)
        real(dp), allocatable :: train_x(:, :), test_x(:, :), fold_scores(:, :)
        real(dp), allocatable :: oof_scores(:, :), oof_weights(:), fold_weights(:)
        real(dp), allocatable :: oof_probabilities(:, :)
        logical :: has_split
        integer :: fold, n_classes
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(calibrated_softmax_classifier_options_t) :: calibrated_softmax_classifier_options_t_default
        type(calibrated_softmax_classifier_state_t) :: calibrated_softmax_classifier_state_t_default

        self%is_fitted = .false.
        config = calibrated_softmax_classifier_options_t_default
        if (present(options)) config = options
        result = calibrated_softmax_classifier_state_t_default
        if (present(state)) state = result
        if (.not. valid_options(config)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax fit: options are invalid")
            return
        end if
        ! This wrapper deliberately owns the leakage-safe temperature path.
        ! The generic multiclass calibrator also exposes Platt and isotonic
        ! methods, but those methods are not part of this classifier's packed
        ! parameter/derivative contract yet.  Refuse them explicitly rather
        ! than silently changing the public calibrated-softmax API.
        if (config%calibration%method /= CALIBRATION_TEMPERATURE) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "calibrated softmax fit: only temperature calibration is supported")
            return
        end if
        if (size(x, 1) < config%cv_folds .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax fit: input dimensions or values are invalid")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax fit: at least two classes are required")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp) .or. &
                .not. ieee_is_finite(sum(sample_weight)) .or. sum(sample_weight) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated softmax fit: sample weights are invalid")
                return
            end if
        end if
        if (.not. enough_class_support(labels, classes, config%cv_folds, sample_weight)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax fit: every class needs at least cv_folds positive rows")
            return
        end if
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes .or. &
                any(.not. ieee_is_finite(class_weight)) .or. any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated softmax fit: class weights need positive sorted-class values")
                return
            end if
        end if

        call splitter%initialize(labels, config%cv_folds, status, &
            shuffle=config%cv_shuffle, seed=config%cv_seed)
        if (status%code /= FORTNUM_OK) return
        allocate(oof_scores(size(labels), n_classes), oof_weights(size(labels)))
        oof_scores = 0.0_dp
        oof_weights = 1.0_dp
        if (present(sample_weight)) oof_weights = sample_weight
        do fold = 1, config%cv_folds
            call splitter%next_split(train_indices, test_indices, has_split, status)
            if (status%code /= FORTNUM_OK) return
            if (.not. has_split) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated softmax fit: splitter ended before all folds")
                return
            end if
            call gather_rows(x, train_indices, train_x)
            call gather_rows(x, test_indices, test_x)
            if (present(sample_weight)) then
                allocate(fold_weights(size(train_indices)))
                fold_weights = sample_weight(train_indices)
            end if
            if (present(sample_weight)) then
                if (present(class_weight)) then
                    call fit_base(fold_model, train_x, labels(train_indices), status, config, &
                        fold_weights, class_weight)
                else
                    call fit_base(fold_model, train_x, labels(train_indices), status, config, &
                        fold_weights)
                end if
            else if (present(class_weight)) then
                call fit_base(fold_model, train_x, labels(train_indices), status, config, &
                    class_weight=class_weight)
            else
                call fit_base(fold_model, train_x, labels(train_indices), status, config)
            end if
            if (allocated(fold_weights)) deallocate(fold_weights)
            if (status%code /= FORTNUM_OK) return
            allocate(fold_scores(size(test_indices), n_classes))
            call fold_model%decision_function(test_x, fold_scores, status)
            if (status%code /= FORTNUM_OK) return
            oof_scores(test_indices, :) = fold_scores
            deallocate(fold_scores, train_x, test_x)
        end do
        result%cv_folds = config%cv_folds
        result%cv_samples = size(labels)
        result%class_count = n_classes
        result%cv_converged = .true.
        allocate(oof_probabilities(size(labels), n_classes))
        call raw_probabilities(oof_scores, oof_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        result%oof_log_loss = weighted_log_loss(oof_probabilities, labels, classes, oof_weights)
        self%oof_log_loss_value = result%oof_log_loss
        if (present(sample_weight)) then
            call self%calibrator%fit(oof_scores, labels, status, &
                options=config%calibration, sample_weight=oof_weights, state=calibration_state)
        else
            call self%calibrator%fit(oof_scores, labels, status, &
                options=config%calibration, state=calibration_state)
        end if
        if (status%code /= FORTNUM_OK) return
        self%calibration_method_code = config%calibration%method
        result%calibration_iterations = calibration_state%iterations
        result%calibration_converged = calibration_state%converged
        result%calibration_objective = calibration_state%objective
        call self%calibrator%predict_proba(oof_scores, oof_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        result%calibrated_oof_log_loss = weighted_log_loss(oof_probabilities, labels, &
            classes, oof_weights)
        self%calibrated_oof_log_loss_value = result%calibrated_oof_log_loss
        if (present(sample_weight)) then
            if (present(class_weight)) then
                call fit_base(self%classifier, x, labels, status, config, &
                    sample_weight, class_weight)
            else
                call fit_base(self%classifier, x, labels, status, config, &
                    sample_weight=sample_weight)
            end if
        else if (present(class_weight)) then
            call fit_base(self%classifier, x, labels, status, config, class_weight=class_weight)
        else
            call fit_base(self%classifier, x, labels, status, config)
        end if
        if (status%code /= FORTNUM_OK) return
        result%classifier_converged = self%classifier%fitted()
        result%converged = result%cv_converged .and. result%calibration_converged .and. &
            result%classifier_converged
        self%cv_fold_count = config%cv_folds
        self%is_fitted = .true.
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine calibrated_softmax_fit

    subroutine calibrated_softmax_decision(self, x, scores, status)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax decision: model is not fitted")
            return
        end if
        call self%classifier%decision_function(x, scores, status)
    end subroutine calibrated_softmax_decision

    subroutine calibrated_softmax_decision_device(self, device, x, scores, status)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "calibrated softmax device decision: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax device decision: device kind is invalid")
        end select
    end subroutine calibrated_softmax_decision_device

    subroutine calibrated_softmax_decision_jvp(self, x, theta_dot, x_dot, scores, &
            scores_dot, status)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:, :), scores_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_base

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax decision JVP: model is not fitted")
            return
        end if
        n_base = self%classifier%parameter_count()
        if (size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax decision JVP: parameter tangent has invalid size")
            return
        end if
        call self%classifier%decision_function_jvp(x, theta_dot(:n_base), x_dot, &
            scores, scores_dot, status)
    end subroutine calibrated_softmax_decision_jvp

    subroutine calibrated_softmax_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_base
        real(dp), allocatable :: base_bar(:)

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax decision VJP: model is not fitted")
            return
        end if
        if (size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax decision VJP: parameter cotangent has invalid size")
            return
        end if
        n_base = self%classifier%parameter_count()
        allocate(base_bar(n_base))
        call self%classifier%decision_function_vjp(x, scores_bar, base_bar, x_bar, status)
        if (status%code == FORTNUM_OK) theta_bar(:n_base) = base_bar
    end subroutine calibrated_softmax_decision_vjp

    subroutine calibrated_softmax_predict_proba(self, x, probabilities, status)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax probability: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax probability: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%class_count()))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        call self%calibrator%predict_proba(scores, probabilities, status)
    end subroutine calibrated_softmax_predict_proba

    subroutine calibrated_softmax_predict_proba_device(self, device, x, probabilities, status)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "calibrated softmax device probability: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax device probability: device kind is invalid")
        end select
    end subroutine calibrated_softmax_predict_proba_device

    subroutine calibrated_softmax_predict_proba_jvp(self, x, theta_dot, x_dot, probabilities, &
            probabilities_dot, status)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), scores_dot(:, :), calibration_dot(:, :)
        integer :: n_base

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax probability JVP: model is not fitted")
            return
        end if
        if (size(theta_dot) /= self%parameter_count() .or. &
            any(shape(probabilities) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax probability JVP: input or output shape is invalid")
            return
        end if
        n_base = self%classifier%parameter_count()
        allocate(scores(size(x, 1), self%class_count()), scores_dot(size(x, 1), &
            self%class_count()), calibration_dot(size(x, 1), self%class_count()))
        call self%classifier%decision_function_jvp(x, theta_dot(:n_base), x_dot, &
            scores, scores_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%calibrator%predict_proba_jvp(scores, scores_dot, probabilities, &
            probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%calibrator%predict_proba_parameter_jvp(scores, theta_dot(n_base + 1:), &
            probabilities, calibration_dot, status)
        if (status%code /= FORTNUM_OK) return
        probabilities_dot = probabilities_dot + calibration_dot
    end subroutine calibrated_softmax_predict_proba_jvp

    subroutine calibrated_softmax_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status, x_dot)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: x_dot(:, :)
        real(dp), allocatable :: local_x_dot(:, :)

        allocate(local_x_dot(size(x, 1), size(x, 2)))
        local_x_dot = 0.0_dp
        if (present(x_dot)) then
            if (any(shape(x_dot) /= shape(x))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated softmax parameter JVP: input tangent shape is invalid")
                probabilities = 0.0_dp
                probabilities_dot = 0.0_dp
                return
            end if
            local_x_dot = x_dot
        end if
        call self%predict_proba_jvp(x, theta_dot, local_x_dot, probabilities, &
            probabilities_dot, status)
    end subroutine calibrated_softmax_predict_proba_parameter_jvp

    subroutine calibrated_softmax_predict_proba_vjp(self, x, probabilities_bar, theta_bar, &
            x_bar, status)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), scores_bar(:, :), base_bar(:), calibration_bar(:)
        integer :: n_base

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax probability VJP: model is not fitted")
            return
        end if
        if (size(theta_bar) /= self%parameter_count() .or. &
            any(shape(probabilities_bar) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax probability VJP: input or output shape is invalid")
            return
        end if
        n_base = self%classifier%parameter_count()
        allocate(scores(size(x, 1), self%class_count()), scores_bar(size(x, 1), &
            self%class_count()), base_bar(n_base), calibration_bar(1))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        call self%calibrator%predict_proba_vjp(scores, probabilities_bar, scores_bar, status)
        if (status%code /= FORTNUM_OK) return
        call self%classifier%decision_function_vjp(x, scores_bar, base_bar, x_bar, status)
        if (status%code /= FORTNUM_OK) return
        theta_bar(:n_base) = base_bar
        call self%calibrator%predict_proba_parameter_vjp(scores, probabilities_bar, &
            calibration_bar, status)
        if (status%code == FORTNUM_OK) theta_bar(n_base + 1:) = calibration_bar
    end subroutine calibrated_softmax_predict_proba_vjp

    subroutine calibrated_softmax_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status, x_bar)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(out), optional :: x_bar(:, :)
        real(dp), allocatable :: local_x_bar(:, :)

        allocate(local_x_bar(size(x, 1), size(x, 2)))
        call self%predict_proba_vjp(x, probabilities_bar, theta_bar, local_x_bar, status)
        if (present(x_bar)) then
            if (any(shape(x_bar) /= shape(x))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated softmax parameter VJP: input cotangent shape is invalid")
                return
            end if
            x_bar = local_x_bar
        end if
    end subroutine calibrated_softmax_predict_proba_parameter_vjp

    subroutine calibrated_softmax_predict(self, x, labels, status)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :)

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax predict: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%class_count()))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        call self%calibrator%predict(scores, labels, status)
    end subroutine calibrated_softmax_predict

    function calibrated_softmax_classes(self) result(labels)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        integer, allocatable :: labels(:)

        labels = self%classifier%classes()
    end function calibrated_softmax_classes

    integer function calibrated_softmax_feature_count(self) result(count)
        class(calibrated_softmax_classifier_t), intent(in) :: self

        count = self%classifier%feature_count()
    end function calibrated_softmax_feature_count

    integer function calibrated_softmax_class_count(self) result(count)
        class(calibrated_softmax_classifier_t), intent(in) :: self

        count = self%classifier%class_count()
    end function calibrated_softmax_class_count

    integer function calibrated_softmax_parameter_count(self) result(count)
        class(calibrated_softmax_classifier_t), intent(in) :: self

        if (.not. self%is_fitted) then
            count = 0
        else
            count = self%classifier%parameter_count() + self%calibrator%parameter_count()
        end if
    end function calibrated_softmax_parameter_count

    function calibrated_softmax_parameters(self) result(values)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:), base(:), calibration(:)
        integer :: n_base

        if (.not. self%is_fitted) then
            allocate(values(0))
            return
        end if
        base = self%classifier%parameters()
        calibration = self%calibrator%parameters()
        n_base = size(base)
        allocate(values(n_base + size(calibration)))
        values(:n_base) = base
        values(n_base + 1:) = calibration
    end function calibrated_softmax_parameters

    subroutine calibrated_softmax_set_parameters(self, values, status)
        class(calibrated_softmax_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_base

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax set_parameters: model or vector is invalid")
            return
        end if
        n_base = self%classifier%parameter_count()
        call self%classifier%set_parameters(values(:n_base), status)
        if (status%code /= FORTNUM_OK) return
        call self%calibrator%set_parameters(values(n_base + 1:), status)
    end subroutine calibrated_softmax_set_parameters

    logical function calibrated_softmax_fitted(self) result(value)
        class(calibrated_softmax_classifier_t), intent(in) :: self

        value = self%is_fitted .and. self%classifier%fitted() .and. self%calibrator%fitted()
    end function calibrated_softmax_fitted

    logical function calibrated_softmax_device_supported(self, device_kind) result(value)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            value = self%fitted()
        case (FORTML_DEVICE_CUDA)
            value = .false.
        case default
            value = .false.
        end select
    end function calibrated_softmax_device_supported

    integer function calibrated_softmax_method(self) result(value)
        class(calibrated_softmax_classifier_t), intent(in) :: self

        value = self%calibration_method_code
    end function calibrated_softmax_method

    integer function calibrated_softmax_cv_folds(self) result(value)
        class(calibrated_softmax_classifier_t), intent(in) :: self

        value = self%cv_fold_count
    end function calibrated_softmax_cv_folds

    real(dp) function calibrated_softmax_oof_log_loss(self) result(value)
        class(calibrated_softmax_classifier_t), intent(in) :: self

        value = self%oof_log_loss_value
    end function calibrated_softmax_oof_log_loss

    real(dp) function calibrated_softmax_calibrated_oof_log_loss(self) result(value)
        class(calibrated_softmax_classifier_t), intent(in) :: self

        value = self%calibrated_oof_log_loss_value
    end function calibrated_softmax_calibrated_oof_log_loss

    real(dp) function calibrated_softmax_temperature(self) result(value)
        class(calibrated_softmax_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        values = self%calibrator%parameters()
        if (size(values) == 1) then
            value = values(1)
        else
            value = 1.0_dp
        end if
    end function calibrated_softmax_temperature

    logical function valid_options(config) result(valid)
        type(calibrated_softmax_classifier_options_t), intent(in) :: config

        valid = ieee_is_finite(config%l2) .and. config%l2 >= 0.0_dp
        if (.not. valid) return
        valid = ieee_is_finite(config%tolerance) .and. config%tolerance > 0.0_dp
        if (.not. valid) return
        valid = config%max_iterations >= 1 .and. config%cv_folds >= 2
        if (.not. valid) return
        valid = config%cv_seed > 0
        if (.not. valid .and. config%cv_shuffle) return
        valid = config%calibration%max_iterations >= 1 .and. &
            ieee_is_finite(config%calibration%tolerance) .and. &
            config%calibration%tolerance > 0.0_dp
    end function valid_options

    subroutine fit_base(model, x, labels, status, config, sample_weight, class_weight)
        type(softmax_regression_t), intent(out) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(calibrated_softmax_classifier_options_t), intent(in) :: config
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)

        if (present(sample_weight)) then
            if (present(class_weight)) then
                call model%fit(x, labels, status, l2=config%l2, &
                    fit_intercept=config%fit_intercept, max_iterations=config%max_iterations, &
                    tolerance=config%tolerance, sample_weight=sample_weight, &
                    class_weight=class_weight)
            else
                call model%fit(x, labels, status, l2=config%l2, &
                    fit_intercept=config%fit_intercept, max_iterations=config%max_iterations, &
                    tolerance=config%tolerance, sample_weight=sample_weight)
            end if
        else if (present(class_weight)) then
            call model%fit(x, labels, status, l2=config%l2, &
                fit_intercept=config%fit_intercept, max_iterations=config%max_iterations, &
                tolerance=config%tolerance, class_weight=class_weight)
        else
            call model%fit(x, labels, status, l2=config%l2, &
                fit_intercept=config%fit_intercept, max_iterations=config%max_iterations, &
                tolerance=config%tolerance)
        end if
    end subroutine fit_base

    subroutine gather_rows(x, indices, result)
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: indices(:)
        real(dp), allocatable, intent(out) :: result(:, :)
        integer :: i

        allocate(result(size(indices), size(x, 2)))
        do i = 1, size(indices)
            result(i, :) = x(indices(i), :)
        end do
    end subroutine gather_rows

    subroutine raw_probabilities(scores, probabilities, status)
        real(dp), intent(in) :: scores(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: maximum, normalizer
        integer :: i

        if (any(shape(probabilities) /= shape(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated softmax raw probabilities: output shape is invalid")
            return
        end if
        do i = 1, size(scores, 1)
            maximum = maxval(scores(i, :))
            probabilities(i, :) = exp(scores(i, :) - maximum)
            normalizer = sum(probabilities(i, :))
            if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated softmax raw probabilities: normalizer is invalid")
                return
            end if
            probabilities(i, :) = probabilities(i, :)/normalizer
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine raw_probabilities

    real(dp) function weighted_log_loss(probabilities, labels, classes, weights) result(value)
        real(dp), intent(in) :: probabilities(:, :), weights(:)
        integer, intent(in) :: labels(:), classes(:)
        real(dp) :: denominator
        integer :: i, class_index

        value = 0.0_dp
        denominator = sum(weights)
        if (denominator <= 0.0_dp .or. .not. ieee_is_finite(denominator)) then
            value = huge(1.0_dp)
            return
        end if
        do i = 1, size(labels)
            class_index = find_class(labels(i), classes)
            if (class_index == 0) then
                value = huge(1.0_dp)
                return
            end if
            value = value - weights(i)*log(max(probabilities(i, class_index), 1.0e-15_dp))
        end do
        value = value/denominator
    end function weighted_log_loss

    logical function enough_class_support(labels, classes, n_folds, sample_weight) result(valid)
        integer, intent(in) :: labels(:), classes(:), n_folds
        real(dp), intent(in), optional :: sample_weight(:)
        integer :: i, j, positive_count

        valid = .true.
        do j = 1, size(classes)
            positive_count = 0
            do i = 1, size(labels)
                if (labels(i) /= classes(j)) cycle
                if (present(sample_weight)) then
                    if (sample_weight(i) <= 0.0_dp) cycle
                end if
                positive_count = positive_count + 1
            end do
            if (positive_count < n_folds) then
                valid = .false.
                return
            end if
        end do
    end function enough_class_support

    integer function find_class(label, classes) result(position)
        integer, intent(in) :: label, classes(:)

        position = 0
        if (size(classes) < 1) return
        do position = 1, size(classes)
            if (classes(position) == label) return
        end do
        position = 0
    end function find_class

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, count, temporary

        allocate(work, source=labels)
        do i = 2, size(work)
            temporary = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= temporary) exit
                work(j + 1) = work(j)
                j = j - 1
            end do
            work(j + 1) = temporary
        end do
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(count)) then
                count = count + 1
                work(count) = work(i)
            end if
        end do
        allocate(classes(count))
        classes = work(:count)
    end subroutine sorted_unique_labels

end module fortml_calibrated_softmax_classifier
