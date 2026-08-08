!> Deterministic binary AdaBoost over weighted CART weak learners.
module fortml_adaboost_classifier
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_cart_classifier, only: cart_classifier_t, CART_CRITERION_GINI
    implicit none
    private

    integer, parameter, public :: ADABOOST_MAX_ESTIMATORS = 256
    integer, parameter, public :: ADABOOST_DEFAULT_SEED = 1

    !> A discrete AdaBoost classifier with binary and multiclass SAMME modes.
    !>
    !> Each weak learner is a weighted CART classifier.  The fitted tree path
    !> is discrete, so input and parameter derivatives are explicit typed
    !> refusals.  The CPU probability map is the smooth logistic map of the
    !> accumulated signed stump margin and is useful for downstream metrics.
    type, public :: adaboost_classifier_t
        private
        type(cart_classifier_t), allocatable :: trees(:)
        real(dp), allocatable :: alpha(:)
        integer, allocatable :: class_label(:)
        integer :: n_inputs = 0
        integer :: n_estimators = 0
        integer :: n_classes = 0
        integer :: max_depth = 1
        integer :: min_samples_leaf = 1
        integer :: seed = ADABOOST_DEFAULT_SEED
        logical :: initialized = .false.
    contains
        procedure, public :: fit => adaboost_fit
        procedure, public :: predict_proba => adaboost_predict_proba
        procedure, public :: predict_proba_device => adaboost_predict_proba_device
        procedure, public :: predict => adaboost_predict
        procedure, public :: decision_function_binary => adaboost_decision_function
        procedure, public :: decision_function_multiclass => adaboost_decision_function_multiclass
        generic, public :: decision_function => decision_function_binary, &
            decision_function_multiclass
        procedure, public :: predict_proba_jvp => adaboost_predict_proba_jvp
        procedure, public :: classes => adaboost_classes
        procedure, public :: class_count => adaboost_class_count
        procedure, public :: stage_weights => adaboost_stage_weights
        procedure, public :: feature_count => adaboost_feature_count
        procedure, public :: estimator_count => adaboost_estimator_count
        procedure, public :: fitted => adaboost_fitted
        procedure, public :: device_supported => adaboost_device_supported
    end type adaboost_classifier_t

    public :: adaboost_fit
    public :: adaboost_predict_proba
    public :: adaboost_predict_proba_device
    public :: adaboost_predict
    public :: adaboost_decision_function
    public :: adaboost_decision_function_multiclass
    public :: adaboost_predict_proba_jvp

contains

    subroutine adaboost_fit(self, x, labels, status, n_estimators, max_depth, &
            min_samples_leaf, sample_weight, seed)
        class(adaboost_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_estimators, max_depth, min_samples_leaf, seed
        real(dp), intent(in), optional :: sample_weight(:)

        type(cart_classifier_t), allocatable :: candidate_trees(:)
        real(dp), allocatable :: candidate_alpha(:), weights(:), predictions(:)
        integer, allocatable :: classes(:), predicted_labels(:)
        integer :: requested_estimators, requested_depth, requested_leaf, requested_seed
        integer :: n, j, fitted_count, n_classes
        real(dp) :: weight_sum, error, learner_alpha, epsilon, chance_limit

        call status_set(status, FORTNUM_DOMAIN_ERROR, "AdaBoost fit: invalid input")
        requested_estimators = 50
        if (present(n_estimators)) requested_estimators = n_estimators
        requested_depth = 1
        if (present(max_depth)) requested_depth = max_depth
        requested_leaf = 1
        if (present(min_samples_leaf)) requested_leaf = min_samples_leaf
        requested_seed = ADABOOST_DEFAULT_SEED
        if (present(seed)) requested_seed = seed
        n = size(x, 1)
        if (n < 2) return
        if (size(x, 2) < 1) return
        if (size(labels) /= n) return
        if (requested_estimators < 1 .or. requested_estimators > ADABOOST_MAX_ESTIMATORS) return
        if (requested_depth < 0 .or. requested_depth > 12) return
        if (requested_leaf < 1 .or. requested_leaf*2 > n) return
        if (requested_seed < 0) return
        if (any(.not. ieee_is_finite(x))) return
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) return
        if (present(sample_weight)) then
            if (size(sample_weight) /= n) return
            if (any(.not. ieee_is_finite(sample_weight))) return
            if (any(sample_weight <= 0.0_dp)) return
        end if

        allocate(candidate_trees(requested_estimators), candidate_alpha(requested_estimators))
        allocate(weights(n), predicted_labels(n), predictions(n))
        weights = 1.0_dp
        if (present(sample_weight)) weights = sample_weight
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) return
        weights = weights/weight_sum
        epsilon = epsilon_dp()
        chance_limit = 1.0_dp - 1.0_dp/real(n_classes, dp)
        fitted_count = 0

        do j = 1, requested_estimators
            call candidate_trees(j)%fit(x, labels, status, max_depth=requested_depth, &
                min_samples_leaf=requested_leaf, sample_weight=weights, &
                criterion=CART_CRITERION_GINI)
            if (status%code /= FORTNUM_OK) return
            call candidate_trees(j)%predict(x, predicted_labels, status)
            if (status%code /= FORTNUM_OK) return
            error = sum(weights, mask=predicted_labels /= labels)
            if (error >= chance_limit) then
                if (fitted_count == 0) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "AdaBoost fit: first weak learner is no better than chance")
                    return
                end if
                exit
            end if
            if (error <= epsilon) then
                if (n_classes == 2) then
                    learner_alpha = 0.5_dp*log(1.0_dp/epsilon)
                else
                    learner_alpha = log((1.0_dp-epsilon)/epsilon) + &
                        log(real(n_classes-1, dp))
                end if
            else if (n_classes == 2) then
                learner_alpha = 0.5_dp*log((1.0_dp - error)/error)
            else
                learner_alpha = log((1.0_dp - error)/error) + &
                    log(real(n_classes-1, dp))
            end if
            candidate_alpha(j) = learner_alpha
            fitted_count = j
            if (n_classes == 2) then
                predictions = merge(1.0_dp, -1.0_dp, predicted_labels == classes(2))
                predictions = predictions*merge(1.0_dp, -1.0_dp, labels == classes(2))
                weights = weights*exp(-learner_alpha*predictions)
            else
                predictions = merge(1.0_dp, 0.0_dp, predicted_labels /= labels)
                weights = weights*exp(learner_alpha*predictions)
            end if
            weight_sum = sum(weights)
            if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "AdaBoost fit: weight normalization is nonfinite")
                return
            end if
            weights = weights/weight_sum
            if (error <= epsilon) exit
        end do
        if (fitted_count < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "AdaBoost fit: no weak learner was fitted")
            return
        end if

        call move_alloc(candidate_trees, self%trees)
        if (allocated(self%alpha)) deallocate(self%alpha)
        if (allocated(self%class_label)) deallocate(self%class_label)
        allocate(self%alpha(fitted_count))
        self%alpha = candidate_alpha(:fitted_count)
        allocate(self%class_label(n_classes))
        self%class_label = classes
        self%n_inputs = size(x, 2)
        self%n_estimators = fitted_count
        self%n_classes = n_classes
        self%max_depth = requested_depth
        self%min_samples_leaf = requested_leaf
        self%seed = requested_seed
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine adaboost_fit

    subroutine adaboost_decision_function(self, x, score, status)
        class(adaboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: score(:)
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: labels(:)
        integer :: j, i

        call validate_query(self, x, size(score), status, "decision_function")
        if (status%code /= FORTNUM_OK) return
        if (self%n_classes /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost decision_function: rank-one margins require exactly two classes")
            return
        end if
        allocate(labels(size(x, 1)))
        score = 0.0_dp
        do j = 1, self%n_estimators
            call self%trees(j)%predict(x, labels, status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, size(x, 1)
                if (labels(i) == self%class_label(2)) then
                    score(i) = score(i) + self%alpha(j)
                else
                    score(i) = score(i) - self%alpha(j)
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine adaboost_decision_function

    subroutine adaboost_decision_function_multiclass(self, x, margins, status)
        class(adaboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: labels(:)
        integer :: i, j, k

        call validate_query_multiclass(self, x, margins, status, "decision_function")
        if (status%code /= FORTNUM_OK) return
        allocate(labels(size(x, 1)))
        margins = 0.0_dp
        do j = 1, self%n_estimators
            call self%trees(j)%predict(x, labels, status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, size(x, 1)
                do k = 1, self%n_classes
                    if (labels(i) == self%class_label(k)) then
                        margins(i, k) = margins(i, k) + self%alpha(j)
                        exit
                    end if
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine adaboost_decision_function_multiclass

    subroutine adaboost_predict_proba(self, x, probabilities, status)
        class(adaboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: score(:)
        integer :: i

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost predict_proba: model is not fitted")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. size(probabilities, 2) /= self%n_classes) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost predict_proba: output shape is invalid")
            return
        end if
        if (self%n_classes == 2) then
            allocate(score(size(x, 1)))
            call self%decision_function_binary(x, score, status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, size(x, 1)
                probabilities(i, 2) = stable_sigmoid(2.0_dp*score(i))
                probabilities(i, 1) = 1.0_dp - probabilities(i, 2)
            end do
        else
            block
                real(dp), allocatable :: margins(:, :), shifted(:)
                real(dp) :: normalizer
                integer :: k
                allocate(margins(size(x, 1), self%n_classes), shifted(self%n_classes))
                call self%decision_function_multiclass(x, margins, status)
                if (status%code /= FORTNUM_OK) return
                do i = 1, size(x, 1)
                    shifted = exp(margins(i, :) - maxval(margins(i, :)))
                    normalizer = sum(shifted)
                    if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "AdaBoost predict_proba: nonfinite SAMME normalizer")
                        return
                    end if
                    do k = 1, self%n_classes
                        probabilities(i, k) = shifted(k)/normalizer
                    end do
                end do
            end block
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine adaboost_predict_proba

    subroutine adaboost_predict(self, x, labels, status)
        class(adaboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: score(:)
        integer :: i

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost predict: output shape is invalid")
            return
        end if
        if (self%n_classes == 2) then
            allocate(score(size(x, 1)))
            call self%decision_function_binary(x, score, status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, size(x, 1)
                if (score(i) >= 0.0_dp) then
                    labels(i) = self%class_label(2)
                else
                    labels(i) = self%class_label(1)
                end if
            end do
        else
            block
                real(dp), allocatable :: margins(:, :)
                integer :: k, best
                allocate(margins(size(x, 1), self%n_classes))
                call self%decision_function_multiclass(x, margins, status)
                if (status%code /= FORTNUM_OK) return
                do i = 1, size(x, 1)
                    best = 1
                    do k = 2, self%n_classes
                        if (margins(i, k) > margins(i, best)) best = k
                    end do
                    labels(i) = self%class_label(best)
                end do
            end block
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine adaboost_predict

    subroutine adaboost_predict_proba_device(self, device, x, probabilities, status)
        class(adaboost_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "AdaBoost device prediction: no resident CUDA tree ensemble is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost device prediction: device kind is invalid")
        end select
    end subroutine adaboost_predict_proba_device

    subroutine adaboost_predict_proba_jvp(self, x, x_dot, probabilities, probabilities_dot, status)
        class(adaboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
                any(shape(x_dot) /= shape(x)) .or. any(shape(probabilities) /= &
                [size(x, 1), self%n_classes]) .or. any(shape(probabilities_dot) /= &
                shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost probability JVP: model, input, or shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "AdaBoost probability JVP: split routing is nondifferentiable")
    end subroutine adaboost_predict_proba_jvp

    function adaboost_classes(self) result(classes)
        class(adaboost_classifier_t), intent(in) :: self
        integer, allocatable :: classes(:)
        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function adaboost_classes

    integer function adaboost_class_count(self) result(count)
        class(adaboost_classifier_t), intent(in) :: self
        count = self%n_classes
    end function adaboost_class_count

    function adaboost_stage_weights(self) result(weights)
        class(adaboost_classifier_t), intent(in) :: self
        real(dp), allocatable :: weights(:)
        if (allocated(self%alpha)) then
            allocate(weights(self%n_estimators))
            weights = self%alpha(:self%n_estimators)
        else
            allocate(weights(0))
        end if
    end function adaboost_stage_weights

    integer function adaboost_feature_count(self) result(count)
        class(adaboost_classifier_t), intent(in) :: self
        count = self%n_inputs
    end function adaboost_feature_count

    integer function adaboost_estimator_count(self) result(count)
        class(adaboost_classifier_t), intent(in) :: self
        count = self%n_estimators
    end function adaboost_estimator_count

    logical function adaboost_fitted(self) result(fitted)
        class(adaboost_classifier_t), intent(in) :: self
        fitted = self%initialized
    end function adaboost_fitted

    logical function adaboost_device_supported(self, kind) result(supported)
        class(adaboost_classifier_t), intent(in) :: self
        integer, intent(in) :: kind
        supported = kind == FORTML_DEVICE_CPU
    end function adaboost_device_supported

    subroutine validate_query(self, x, output_size, status, operation)
        class(adaboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: output_size
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost "//operation//": model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. output_size /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost "//operation//": model, input, or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost "//operation//": inputs must be finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_query

    subroutine validate_query_multiclass(self, x, margins, status, operation)
        class(adaboost_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), margins(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost "//operation//": model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
                size(margins, 1) /= size(x, 1) .or. &
                size(margins, 2) /= self%n_classes) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost "//operation//": model, input, or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "AdaBoost "//operation//": inputs must be finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_query_multiclass

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, n_unique

        allocate(work(size(labels)))
        work = labels
        call sort_in_place(work)
        n_unique = 1
        do i = 2, size(work)
            if (work(i) /= work(n_unique)) then
                n_unique = n_unique + 1
                work(n_unique) = work(i)
            end if
        end do
        allocate(classes(n_unique))
        classes = work(:n_unique)
    end subroutine sorted_unique_labels

    subroutine sort_in_place(values)
        integer, intent(inout) :: values(:)
        integer :: i, j, key
        do i = 2, size(values)
            key = values(i)
            j = i - 1
            do while (j >= 1)
                if (values(j) <= key) exit
                values(j + 1) = values(j)
                j = j - 1
            end do
            values(j + 1) = key
        end do
    end subroutine sort_in_place

    real(dp) function epsilon_dp() result(value)
        value = 1.0e-12_dp
    end function epsilon_dp

    real(dp) function stable_sigmoid(value) result(output)
        real(dp), intent(in) :: value
        if (value >= 0.0_dp) then
            output = 1.0_dp/(1.0_dp + exp(-value))
        else
            output = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

end module fortml_adaboost_classifier
