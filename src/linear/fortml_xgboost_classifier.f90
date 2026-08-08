!> Binary classification adapter for the deterministic XGBoost-style tree
!> estimator.
module fortml_xgboost_classifier
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    implicit none
    private

    !> A binary classifier facade over `xgboost_t`.
    !>
    !> The wrapped estimator always uses the logistic objective.  Integer
    !> labels are retained (sorted) at fit time, while the wrapped tree uses
    !> the conventional zero/one target.  This keeps the public contract
    !> aligned with classifier APIs: `predict` returns labels,
    !> `predict_proba` returns an `(n,2)` simplex, and `decision_function`
    !> returns the positive-class logit.  Fit is discrete; input products are
    !> zero away from split boundaries and refuse a query on a boundary.
    type, public :: xgboost_classifier_t
        private
        type(xgboost_t) :: booster
        integer, allocatable :: class_label(:)
        integer :: n_inputs = 0
        logical :: initialized = .false.
    contains
        procedure, public :: fit => xgb_classifier_fit
        procedure, public :: predict_proba => xgb_classifier_predict_proba
        procedure, public :: predict_log_proba => xgb_classifier_predict_log_proba
        procedure, public :: predict_proba_device => &
            xgb_classifier_predict_proba_device
        procedure, public :: predict_proba_staged => &
            xgb_classifier_predict_proba_staged
        procedure, public :: decision_function_staged => &
            xgb_classifier_decision_function_staged
        procedure, public :: decision_function => xgb_classifier_decision_function
        procedure, public :: predict => xgb_classifier_predict
        procedure, public :: predict_device => xgb_classifier_predict_device
        procedure, public :: predict_proba_jvp => xgb_classifier_predict_proba_jvp
        procedure, public :: predict_proba_vjp => xgb_classifier_predict_proba_vjp
        procedure, public :: predict_log_proba_jvp => xgb_classifier_predict_log_proba_jvp
        procedure, public :: predict_log_proba_vjp => xgb_classifier_predict_log_proba_vjp
        procedure, public :: feature_importance => &
            xgb_classifier_feature_importance
        procedure, public :: classes => xgb_classifier_classes
        procedure, public :: feature_count => xgb_classifier_feature_count
        procedure, public :: estimator_count => xgb_classifier_estimator_count
        procedure, public :: device_supported => xgb_classifier_device_supported
        procedure, public :: monotone_constraint => &
            xgb_classifier_monotone_constraint
        procedure, public :: missing_policy => xgb_classifier_missing_policy
        procedure, public :: accepts_missing => xgb_classifier_accepts_missing
        procedure, public :: tree_method => xgb_classifier_tree_method
        procedure, public :: categorical_policy => xgb_classifier_categorical_policy
        procedure, public :: categorical_max_categories => &
            xgb_classifier_categorical_max_categories
        procedure, public :: categorical_feature => xgb_classifier_categorical_feature
        procedure, public :: interaction_group => xgb_classifier_interaction_group
        procedure, public :: fitted => xgb_classifier_fitted
        procedure, public :: best_iteration => xgb_classifier_best_iteration
        procedure, public :: best_validation_loss => &
            xgb_classifier_best_validation_loss
        procedure, public :: early_stopped => xgb_classifier_early_stopped
    end type xgboost_classifier_t

contains

    !> Fit a weighted binary logistic booster with arbitrary integer labels.
    subroutine xgb_classifier_fit(self, x, labels, status, options, sample_weight, &
            validation_x, validation_labels, validation_weight)
        class(xgboost_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :)
        integer, intent(in), optional :: validation_labels(:)
        real(dp), intent(in), optional :: validation_weight(:)
        type(xgboost_options_t) :: settings
        type(xgboost_t) :: candidate
        integer, allocatable :: classes(:), binary_labels(:), validation_binary(:)
        integer :: n_samples, n_features, n_validation
        logical :: have_validation
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(xgboost_options_t) :: xgboost_options_t_default

        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            "XGBoost binary classifier fit: invalid input")
        settings = xgboost_options_t_default
        if (present(options)) settings = options
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 2 .or. n_features < 1 .or. size(labels) /= n_samples) return
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) return
        end if
        if (present(validation_x) .neqv. present(validation_labels)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier fit: validation_x and validation_labels must be supplied together")
            return
        end if
        if (present(validation_weight) .and. .not. present(validation_x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier fit: validation_weight requires validation data")
            return
        end if

        call sorted_unique_labels(labels, classes)
        if (size(classes) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier fit: exactly two labels are required")
            return
        end if
        allocate(binary_labels(n_samples))
        binary_labels = merge(1, 0, labels == classes(2))

        have_validation = present(validation_x)
        if (have_validation) then
            n_validation = size(validation_x, 1)
            if (size(validation_x, 2) /= n_features .or. n_validation < 1 .or. &
                size(validation_labels) /= n_validation) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "XGBoost binary classifier fit: validation dimensions are invalid")
                return
            end if
            if (present(validation_weight)) then
                if (size(validation_weight) /= n_validation .or. &
                    any(.not. ieee_is_finite(validation_weight)) .or. &
                    any(validation_weight <= 0.0_dp)) return
            end if
            if (any((validation_labels /= classes(1)) .and. &
                (validation_labels /= classes(2)))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "XGBoost binary classifier fit: validation labels are not in the fitted classes")
                return
            end if
            allocate(validation_binary(n_validation))
            validation_binary = merge(1, 0, validation_labels == classes(2))
        end if

        ! Forward optional arguments explicitly so a supplied validation or
        ! weight vector can never be silently dropped.
        if (have_validation) then
            if (present(sample_weight)) then
                if (present(validation_weight)) then
                    call candidate%fit_binary(x, real(binary_labels, dp), status, &
                        settings, sample_weight, validation_x, &
                        real(validation_binary, dp), validation_weight)
                else
                    call candidate%fit_binary(x, real(binary_labels, dp), status, &
                        settings, sample_weight, validation_x, &
                        real(validation_binary, dp))
                end if
            else if (present(validation_weight)) then
                call candidate%fit_binary(x, real(binary_labels, dp), status, &
                    settings, validation_x=validation_x, &
                    validation_y=real(validation_binary, dp), &
                    validation_weight=validation_weight)
            else
                call candidate%fit_binary(x, real(binary_labels, dp), status, &
                    settings, validation_x=validation_x, &
                    validation_y=real(validation_binary, dp))
            end if
        else if (present(sample_weight)) then
            call candidate%fit_binary(x, real(binary_labels, dp), status, &
                settings, sample_weight)
        else
            call candidate%fit_binary(x, real(binary_labels, dp), status, settings)
        end if
        if (status%code /= FORTNUM_OK) return
        self%booster = candidate
        call move_alloc(classes, self%class_label)
        self%n_inputs = n_features
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_classifier_fit

    subroutine xgb_classifier_predict_proba(self, x, probabilities, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(probabilities) /= [size(x, 1), 2])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier predict_proba: model, input, or output shape is invalid")
            return
        end if
        call self%booster%predict_proba(x, probabilities, status)
    end subroutine xgb_classifier_predict_proba

    !> Return per-class log probabilities in the fitted sorted class order.
    !! The log-sigmoid branches are evaluated from the margin directly so
    !! probabilities in either tail remain finite and do not underflow before
    !! taking the logarithm.
    subroutine xgb_classifier_predict_log_proba(self, x, log_probabilities, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:)

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(log_probabilities) /= [size(x, 1), 2])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier predict_log_proba: model, input, or output shape is invalid")
            return
        end if
        allocate(margin(size(x, 1)))
        call self%booster%decision_function(x, margin, status)
        if (status%code /= FORTNUM_OK) return
        call fill_log_probability_columns(margin, log_probabilities)
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_classifier_predict_log_proba

    subroutine xgb_classifier_predict_proba_device(self, device, x, probabilities, &
            status)
        class(xgboost_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "XGBoost binary classifier device prediction: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier device prediction: device kind is invalid")
        end select
    end subroutine xgb_classifier_predict_proba_device

    subroutine xgb_classifier_predict_proba_staged(self, x, probabilities, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= 2 .or. &
            size(probabilities, 3) /= self%estimator_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier staged probabilities: shape is invalid")
            return
        end if
        call self%booster%predict_proba_staged(x, probabilities, status)
    end subroutine xgb_classifier_predict_proba_staged

    subroutine xgb_classifier_decision_function_staged(self, x, margins, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(margins) /= [size(x, 1), self%estimator_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier staged margins: shape is invalid")
            return
        end if
        call self%booster%predict_staged_margin(x, margins, status)
    end subroutine xgb_classifier_decision_function_staged

    subroutine xgb_classifier_decision_function(self, x, margins, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(margins) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier decision_function: shape is invalid")
            return
        end if
        call self%booster%decision_function(x, margins, status)
    end subroutine xgb_classifier_decision_function

    subroutine xgb_classifier_predict(self, x, labels, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier predict: shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), 2))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            if (probabilities(i, 2) > probabilities(i, 1)) then
                labels(i) = self%class_label(2)
            else
                labels(i) = self%class_label(1)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_classifier_predict

    subroutine xgb_classifier_predict_device(self, device, x, labels, status)
        class(xgboost_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            if (.not. self%initialized .or. size(labels) /= size(x, 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "XGBoost binary classifier device prediction: shape is invalid")
                return
            end if
            allocate(probabilities(size(x, 1), 2))
            call self%predict_proba(x, probabilities, status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, size(labels)
                labels(i) = self%class_label(1)
                if (probabilities(i, 2) > probabilities(i, 1)) labels(i) = self%class_label(2)
            end do
            call status_set(status, FORTNUM_OK, "")
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "XGBoost binary classifier device prediction: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier device prediction: device kind is invalid")
        end select
    end subroutine xgb_classifier_predict_device

    subroutine xgb_classifier_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: positive(:), positive_dot(:)

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(probabilities) /= [size(x, 1), 2]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier probability JVP: shape is invalid")
            return
        end if
        allocate(positive(size(x, 1)), positive_dot(size(x, 1)))
        call self%booster%predict_jvp(x, x_dot, positive, positive_dot, status)
        if (status%code /= FORTNUM_OK) return
        probabilities(:, 2) = positive
        probabilities(:, 1) = 1.0_dp - positive
        probabilities_dot(:, 2) = positive_dot
        probabilities_dot(:, 1) = -positive_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_classifier_predict_proba_jvp

    subroutine xgb_classifier_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: positive_bar(:)

        x_bar = 0.0_dp
        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(probabilities_bar) /= [size(x, 1), 2]) .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier probability VJP: shape or cotangent is invalid")
            return
        end if
        allocate(positive_bar(size(x, 1)))
        positive_bar = probabilities_bar(:, 2) - probabilities_bar(:, 1)
        call self%booster%predict_vjp(x, positive_bar, x_bar, status)
    end subroutine xgb_classifier_predict_proba_vjp

    !> Forward product of the log-probability output and an input tangent.
    !! The fitted tree remains fixed, so the product inherits the tree's
    !! split-boundary and categorical-discrete refusal contract.
    subroutine xgb_classifier_predict_log_proba_jvp(self, x, x_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:), margin_dot(:), positive(:)

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(log_probabilities) /= [size(x, 1), 2]) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier log-probability JVP: shape is invalid")
            return
        end if
        allocate(margin(size(x, 1)), margin_dot(size(x, 1)), positive(size(x, 1)))
        call self%booster%predict_jvp(x, x_dot, margin, margin_dot, status)
        if (status%code /= FORTNUM_OK) return
        call fill_log_probability_columns(margin, log_probabilities)
        positive = exp(log_probabilities(:, 2))
        log_probabilities_dot(:, 2) = (1.0_dp - positive)*margin_dot
        log_probabilities_dot(:, 1) = -positive*margin_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_classifier_predict_log_proba_jvp

    !> Reverse product of the log-probability output with respect to inputs.
    subroutine xgb_classifier_predict_log_proba_vjp(self, x, log_probabilities_bar, &
            x_bar, status)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:), positive(:), margin_bar(:)

        x_bar = 0.0_dp
        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), 2]) .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier log-probability VJP: shape or cotangent is invalid")
            return
        end if
        allocate(margin(size(x, 1)), positive(size(x, 1)), margin_bar(size(x, 1)))
        call self%booster%decision_function(x, margin, status)
        if (status%code /= FORTNUM_OK) return
        positive = classifier_stable_sigmoid_array(margin)
        margin_bar = -positive*log_probabilities_bar(:, 1) + &
            (1.0_dp - positive)*log_probabilities_bar(:, 2)
        call self%booster%predict_vjp(x, margin_bar, x_bar, status)
    end subroutine xgb_classifier_predict_log_proba_vjp

    subroutine xgb_classifier_feature_importance(self, importance, status, kind, &
            normalize)
        class(xgboost_classifier_t), intent(in) :: self
        real(dp), intent(out) :: importance(:)
        type(fortnum_status_t), intent(out) :: status
        character(len=*), intent(in), optional :: kind
        logical, intent(in), optional :: normalize

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost binary classifier feature_importance: model is not initialized")
            return
        end if
        if (present(kind)) then
            if (present(normalize)) then
                call self%booster%feature_importance(importance, status, kind, normalize)
            else
                call self%booster%feature_importance(importance, status, kind)
            end if
        else if (present(normalize)) then
            call self%booster%feature_importance(importance, status, normalize=normalize)
        else
            call self%booster%feature_importance(importance, status)
        end if
    end subroutine xgb_classifier_feature_importance

    function xgb_classifier_classes(self) result(classes)
        class(xgboost_classifier_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function xgb_classifier_classes

    integer function xgb_classifier_feature_count(self) result(count)
        class(xgboost_classifier_t), intent(in) :: self
        count = self%n_inputs
    end function xgb_classifier_feature_count

    integer function xgb_classifier_estimator_count(self) result(count)
        class(xgboost_classifier_t), intent(in) :: self
        count = self%booster%estimator_count()
    end function xgb_classifier_estimator_count

    logical function xgb_classifier_device_supported(self, device_kind) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = self%initialized .and. device_kind == FORTML_DEVICE_CPU
    end function xgb_classifier_device_supported

    integer function xgb_classifier_monotone_constraint(self, feature_index) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        integer, intent(in) :: feature_index
        value = self%booster%monotone_constraint(feature_index)
    end function xgb_classifier_monotone_constraint

    character(len=16) function xgb_classifier_missing_policy(self) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        value = self%booster%missing_policy()
    end function xgb_classifier_missing_policy

    logical function xgb_classifier_accepts_missing(self) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        value = self%booster%accepts_missing()
    end function xgb_classifier_accepts_missing

    character(len=16) function xgb_classifier_tree_method(self) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        value = self%booster%tree_method()
    end function xgb_classifier_tree_method

    character(len=16) function xgb_classifier_categorical_policy(self) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        value = self%booster%categorical_policy()
    end function xgb_classifier_categorical_policy

    integer function xgb_classifier_categorical_max_categories(self) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        value = self%booster%categorical_max_categories()
    end function xgb_classifier_categorical_max_categories

    logical function xgb_classifier_categorical_feature(self, feature_index) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        integer, intent(in) :: feature_index
        value = self%booster%categorical_feature(feature_index)
    end function xgb_classifier_categorical_feature

    integer function xgb_classifier_interaction_group(self, feature_index) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        integer, intent(in) :: feature_index
        value = self%booster%interaction_group(feature_index)
    end function xgb_classifier_interaction_group

    logical function xgb_classifier_fitted(self) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        value = self%initialized
    end function xgb_classifier_fitted

    integer function xgb_classifier_best_iteration(self) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        value = self%booster%best_iteration()
    end function xgb_classifier_best_iteration

    real(dp) function xgb_classifier_best_validation_loss(self) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        value = self%booster%best_validation_loss()
    end function xgb_classifier_best_validation_loss

    logical function xgb_classifier_early_stopped(self) result(value)
        class(xgboost_classifier_t), intent(in) :: self
        value = self%booster%early_stopped()
    end function xgb_classifier_early_stopped

    pure real(dp) function classifier_log_sigmoid(value) result(log_probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            log_probability = -log(1.0_dp + exp(-value))
        else
            log_probability = value - log(1.0_dp + exp(value))
        end if
    end function classifier_log_sigmoid

    pure real(dp) function classifier_log_one_minus_sigmoid(value) result(log_probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            log_probability = -value - log(1.0_dp + exp(-value))
        else
            log_probability = -log(1.0_dp + exp(value))
        end if
    end function classifier_log_one_minus_sigmoid

    pure real(dp) function classifier_stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function classifier_stable_sigmoid

    function classifier_stable_sigmoid_array(values) result(probabilities)
        real(dp), intent(in) :: values(:)
        real(dp) :: probabilities(size(values))
        integer :: i

        do i = 1, size(values)
            probabilities(i) = classifier_stable_sigmoid(values(i))
        end do
    end function classifier_stable_sigmoid_array

    subroutine fill_log_probability_columns(margin, log_probabilities)
        real(dp), intent(in) :: margin(:)
        real(dp), intent(out) :: log_probabilities(:, :)
        integer :: i

        do i = 1, size(margin)
            log_probabilities(i, 1) = classifier_log_one_minus_sigmoid(margin(i))
            log_probabilities(i, 2) = classifier_log_sigmoid(margin(i))
        end do
    end subroutine fill_log_probability_columns

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer :: i, j, n_unique, temporary

        allocate(classes(size(labels)))
        classes = labels
        do i = 2, size(classes)
            temporary = classes(i)
            j = i - 1
            do while (j >= 1)
                if (classes(j) <= temporary) exit
                classes(j + 1) = classes(j)
                j = j - 1
            end do
            classes(j + 1) = temporary
        end do
        n_unique = 1
        do i = 2, size(classes)
            if (classes(i) /= classes(n_unique)) then
                n_unique = n_unique + 1
                classes(n_unique) = classes(i)
            end if
        end do
        if (n_unique < size(classes)) classes = classes(:n_unique)
    end subroutine sorted_unique_labels

end module fortml_xgboost_classifier
