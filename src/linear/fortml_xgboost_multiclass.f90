!> One-vs-rest multiclass classification over the exact XGBoost-style binary
!> logistic estimator.
module fortml_xgboost_multiclass
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    implicit none
    private

    !> Deterministic one-vs-rest multiclass XGBoost-style classifier.
    !>
    !> Each class owns a deterministic depth-limited second-order logistic
    !> booster.  Positive class probabilities are normalized across the
    !> one-vs-rest models, so
    !> arbitrary integer labels and a proper multiclass probability contract
    !> are available without changing the binary estimator.
    type, public :: xgboost_multiclass_t
        private
        type(xgboost_t), allocatable :: one_vs_rest(:)
        integer, allocatable :: class_label(:)
        integer :: n_inputs = 0
        logical :: initialized = .false.
    contains
        procedure, public :: fit => xgb_multiclass_fit
        procedure, public :: predict_proba => xgb_multiclass_predict_proba
        procedure, public :: predict_proba_staged => &
            xgb_multiclass_predict_proba_staged
        procedure, public :: decision_function_staged => &
            xgb_multiclass_decision_function_staged
        procedure, public :: predict_proba_jvp => xgb_multiclass_predict_proba_jvp
        procedure, public :: predict_proba_vjp => xgb_multiclass_predict_proba_vjp
        procedure, public :: decision_function => xgb_multiclass_decision_function
        procedure, public :: predict => xgb_multiclass_predict
        procedure, public :: feature_importance => &
            xgb_multiclass_feature_importance
        procedure, public :: classes => xgb_multiclass_classes
        procedure, public :: feature_count => xgb_multiclass_feature_count
        procedure, public :: class_count => xgb_multiclass_class_count
        procedure, public :: estimator_count => xgb_multiclass_estimator_count
        procedure, public :: fitted => xgb_multiclass_fitted
    end type xgboost_multiclass_t

contains

    subroutine xgb_multiclass_fit(self, x, labels, status, options)
        class(xgboost_multiclass_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        type(xgboost_options_t) :: settings
        integer, allocatable :: classes(:), binary_labels(:)
        integer :: i, n_classes, n_samples

        settings = xgboost_options_t()
        if (present(options)) settings = options
        n_samples = size(x, 1)
        if (n_samples < 2 .or. size(x, 2) < 1 .or. size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass fit: input dimensions are invalid")
            return
        end if
        call unique_sorted_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass fit: at least two classes are required")
            return
        end if

        allocate(self%one_vs_rest(n_classes), self%class_label(n_classes))
        self%class_label = classes
        allocate(binary_labels(n_samples))
        do i = 1, n_classes
            binary_labels = 0
            where (labels == classes(i)) binary_labels = 1
            call self%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                status, settings)
            if (status%code /= FORTNUM_OK) return
        end do
        self%n_inputs = size(x, 2)
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_fit

    subroutine xgb_multiclass_predict_proba(self, x, probabilities, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), totals(:)
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba: model or input is invalid")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba: output shape is invalid")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2), totals(size(x, 1)))
        probabilities = 0.0_dp
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_proba(x, binary_probabilities, &
                status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i) = binary_probabilities(:, 2)
        end do
        totals = sum(probabilities, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            probabilities(:, i) = probabilities(:, i)/totals
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict_proba

    !> Return normalized one-vs-rest probabilities after every boosting stage.
    !>
    !> The output has shape `(n_samples, n_classes, n_estimators)`.  Each
    !> stage is normalized across classes, just like `predict_proba`.
    subroutine xgb_multiclass_predict_proba_staged(self, x, probabilities, &
            status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_staged(:, :), totals(:, :)
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba_staged: model or input is invalid")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= self%class_count() .or. &
            size(probabilities, 3) /= self%estimator_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba_staged: output shape is invalid")
            return
        end if
        allocate(binary_staged(size(x, 1), self%estimator_count()), &
            totals(size(x, 1), self%estimator_count()))
        probabilities = 0.0_dp
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_staged(x, binary_staged, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i, :) = binary_staged
        end do
        totals = sum(probabilities, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba_staged: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            probabilities(:, i, :) = probabilities(:, i, :)/totals
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict_proba_staged

    !> Return one-vs-rest raw margins after every boosting stage.
    !>
    !> The output has shape `(n_samples, n_classes, n_estimators)` and is not
    !> normalized; use `predict_proba_staged` for class probabilities.
    subroutine xgb_multiclass_decision_function_staged(self, x, margins, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_margins(:, :)
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass decision_function_staged: model or "// &
                "input is invalid")
            return
        end if
        if (size(margins, 1) /= size(x, 1) .or. &
            size(margins, 2) /= self%class_count() .or. &
            size(margins, 3) /= self%estimator_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass decision_function_staged: output shape is invalid")
            return
        end if
        allocate(binary_margins(size(x, 1), self%estimator_count()))
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_staged_margin(x, binary_margins, status)
            if (status%code /= FORTNUM_OK) return
            margins(:, i, :) = binary_margins
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_decision_function_staged

    !> Aggregate feature diagnostics across all one-vs-rest estimators.
    !>
    !> The `kind` and normalization semantics match the binary estimator's
    !> `feature_importance` method.  Raw values are summed across classes.
    subroutine xgb_multiclass_feature_importance(self, importance, status, &
            kind, normalize)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(out) :: importance(:)
        type(fortnum_status_t), intent(out) :: status
        character(len=*), intent(in), optional :: kind
        logical, intent(in), optional :: normalize
        real(dp), allocatable :: contribution(:)
        logical :: should_normalize
        real(dp) :: total
        integer :: i

        if (.not. self%initialized .or. .not. allocated(self%one_vs_rest)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass feature_importance: model is not initialized")
            return
        end if
        if (size(importance) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass feature_importance: output shape is invalid")
            return
        end if
        allocate(contribution(self%n_inputs))
        importance = 0.0_dp
        do i = 1, self%class_count()
            if (present(kind)) then
                call self%one_vs_rest(i)%feature_importance(contribution, status, &
                    kind, .false.)
            else
                call self%one_vs_rest(i)%feature_importance(contribution, status)
            end if
            if (status%code /= FORTNUM_OK) return
            importance = importance + contribution
        end do
        should_normalize = .false.
        if (present(normalize)) should_normalize = normalize
        if (should_normalize) then
            total = sum(importance)
            if (total > 0.0_dp) importance = importance/total
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_feature_importance

    subroutine xgb_multiclass_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: positive(:), positive_dot(:), totals(:), &
            totals_dot(:)
        integer :: i

        if (.not. valid_query(self, x) .or. any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability JVP: model or input is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability JVP: tangent is not finite")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability JVP: output shape is invalid")
            return
        end if
        allocate(positive(size(x, 1)), positive_dot(size(x, 1)), &
            totals(size(x, 1)), totals_dot(size(x, 1)))
        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_jvp(x, x_dot, positive, &
                positive_dot, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i) = positive
            probabilities_dot(:, i) = positive_dot
        end do
        totals = sum(probabilities, dim=2)
        totals_dot = sum(probabilities_dot, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp) .or. &
            any(.not. ieee_is_finite(totals_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability JVP: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            probabilities_dot(:, i) = (probabilities_dot(:, i)*totals - &
                probabilities(:, i)*totals_dot)/(totals*totals)
            probabilities(:, i) = probabilities(:, i)/totals
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict_proba_jvp

    !> Reverse-mode product of normalized one-vs-rest probabilities with
    !> respect to query features.  It is zero away from tree split surfaces;
    !> the binary estimators provide the shared boundary refusal.
    subroutine xgb_multiclass_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        x_bar = 0.0_dp
        if (.not. valid_query(self, x) .or. &
            any(shape(probabilities_bar) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability VJP: model, cotangent, or output shape is invalid")
            return
        end if
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_vjp(x, probabilities_bar(:, i), &
                x_bar, status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict_proba_vjp

    subroutine xgb_multiclass_decision_function(self, x, margins, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass decision_function: model or input is invalid")
            return
        end if
        if (any(shape(margins) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass decision_function: output shape is invalid")
            return
        end if
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%decision_function(x, margins(:, i), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_decision_function

    subroutine xgb_multiclass_predict(self, x, labels, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%class_count()))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict

    function xgb_multiclass_classes(self) result(classes)
        class(xgboost_multiclass_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function xgb_multiclass_classes

    integer function xgb_multiclass_feature_count(self) result(count)
        class(xgboost_multiclass_t), intent(in) :: self

        count = self%n_inputs
    end function xgb_multiclass_feature_count

    integer function xgb_multiclass_class_count(self) result(count)
        class(xgboost_multiclass_t), intent(in) :: self

        count = 0
        if (allocated(self%class_label)) count = size(self%class_label)
    end function xgb_multiclass_class_count

    integer function xgb_multiclass_estimator_count(self) result(count)
        class(xgboost_multiclass_t), intent(in) :: self

        count = 0
        if (.not. allocated(self%one_vs_rest)) return
        if (size(self%one_vs_rest) < 1) return
        count = self%one_vs_rest(1)%estimator_count()
    end function xgb_multiclass_estimator_count

    logical function xgb_multiclass_fitted(self) result(fitted)
        class(xgboost_multiclass_t), intent(in) :: self

        fitted = self%initialized
    end function xgb_multiclass_fitted

    logical function valid_query(self, x) result(valid)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)

        valid = self%initialized
        if (.not. valid) return
        valid = size(x, 1) > 0 .and. size(x, 2) == self%n_inputs
        if (.not. valid) return
        valid = all(ieee_is_finite(x) .or. ieee_is_nan(x))
        if (.not. valid) return
        if (allocated(self%one_vs_rest)) then
            valid = self%one_vs_rest(1)%accepts_missing() .or. &
                .not. any(ieee_is_nan(x))
        else
            valid = .false.
        end if
    end function valid_query

    subroutine unique_sorted_labels(labels, classes)
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
        if (n_unique < size(classes)) then
            classes = classes(:n_unique)
        end if
    end subroutine unique_sorted_labels

end module fortml_xgboost_multiclass
