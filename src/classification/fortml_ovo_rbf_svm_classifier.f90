module fortml_ovo_rbf_svm_classifier
    !! Deterministic one-vs-one dense RBF-kernel SVM classification.
    !!
    !! A binary RBF SVM model is fitted for every pair of sorted classes.  A
    !! pair model predicts the second class probability; deployment combines
    !! these pair probabilities as pairwise votes and divides by the number
    !! of pair models.  This is a deliberately explicit, differentiable
    !! probability policy rather than an implicit claim to implement
    !! scikit-learn's opaque pairwise coupling solver.  Fit is transactional:
    !! a malformed request or failed pair solve leaves the previous model
    !! untouched.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_rbf_svm_classifier, only: rbf_svm_classifier_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    type, public :: ovo_rbf_svm_classifier_t
        private
        type(rbf_svm_classifier_t), allocatable :: models(:)
        integer, allocatable :: class_label(:)
        integer, allocatable :: pair_negative(:), pair_positive(:)
        integer :: n_classes = 0
        integer :: n_pairs = 0
        integer :: n_features = 0
        integer, allocatable :: parameter_offset(:)
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => ovo_rbf_svm_fit
        procedure, public :: decision_function => ovo_rbf_svm_decision
        procedure, public :: decision_function_device => ovo_rbf_svm_decision_device
        procedure, public :: decision_function_jvp => ovo_rbf_svm_decision_jvp
        procedure, public :: decision_function_parameter_jvp => &
            ovo_rbf_svm_decision_parameter_jvp
        procedure, public :: decision_function_vjp => ovo_rbf_svm_decision_vjp
        procedure, public :: decision_function_parameter_vjp => &
            ovo_rbf_svm_decision_parameter_vjp
        procedure, public :: predict_proba => ovo_rbf_svm_predict_proba
        procedure, public :: predict_proba_device => ovo_rbf_svm_predict_proba_device
        procedure, public :: predict_proba_jvp => ovo_rbf_svm_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            ovo_rbf_svm_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => ovo_rbf_svm_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            ovo_rbf_svm_predict_proba_parameter_vjp
        procedure, public :: predict => ovo_rbf_svm_predict
        procedure, public :: predict_device => ovo_rbf_svm_predict_device
        procedure, public :: device_supported => ovo_rbf_svm_device_supported
        procedure, public :: classes => ovo_rbf_svm_classes
        procedure, public :: pair_classes => ovo_rbf_svm_pair_classes
        procedure, public :: pair_count => ovo_rbf_svm_pair_count
        procedure, public :: class_count => ovo_rbf_svm_class_count
        procedure, public :: feature_count => ovo_rbf_svm_feature_count
        procedure, public :: parameter_count => ovo_rbf_svm_parameter_count
        procedure, public :: parameters => ovo_rbf_svm_parameters
        procedure, public :: set_parameters => ovo_rbf_svm_set_parameters
        procedure, public :: fitted => ovo_rbf_svm_fitted
    end type ovo_rbf_svm_classifier_t

    public :: ovo_rbf_svm_fit
    public :: ovo_rbf_svm_decision
    public :: ovo_rbf_svm_decision_device
    public :: ovo_rbf_svm_decision_jvp
    public :: ovo_rbf_svm_decision_parameter_jvp
    public :: ovo_rbf_svm_decision_vjp
    public :: ovo_rbf_svm_decision_parameter_vjp
    public :: ovo_rbf_svm_predict_proba
    public :: ovo_rbf_svm_predict_proba_jvp
    public :: ovo_rbf_svm_predict_proba_parameter_jvp
    public :: ovo_rbf_svm_predict_proba_vjp
    public :: ovo_rbf_svm_predict_proba_parameter_vjp
    public :: ovo_rbf_svm_predict

contains

    subroutine ovo_rbf_svm_fit(self, x, labels, status, c, gamma, &
            max_iterations, tolerance, sample_weight, class_weight)
        class(ovo_rbf_svm_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: c, gamma, tolerance
        integer, intent(in), optional :: max_iterations
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        type(ovo_rbf_svm_classifier_t) :: candidate
        integer, allocatable :: classes(:), pair_labels(:), indices(:)
        real(dp), allocatable :: effective_weight(:), class_factors(:)
        real(dp), allocatable :: pair_x(:, :), pair_weight(:)
        real(dp) :: requested_c, requested_gamma, requested_tolerance
        integer :: requested_iterations
        integer :: i, j, k, n_samples, n_features, n_classes, n_pair_samples
        integer :: pair_index, index
        character(256) :: failure_message

        requested_c = 1.0_dp
        if (present(c)) requested_c = c
        requested_gamma = 1.0_dp/real(max(1, size(x, 2)), dp)
        if (present(gamma)) requested_gamma = gamma
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        requested_iterations = 500
        if (present(max_iterations)) requested_iterations = max_iterations

        if (size(x, 1) < 2 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM fit: inputs must be finite")
            return
        end if
        if (.not. ieee_is_finite(requested_c) .or. requested_c <= 0.0_dp .or. &
            .not. ieee_is_finite(requested_gamma) .or. requested_gamma <= 0.0_dp .or. &
            .not. ieee_is_finite(requested_tolerance) .or. requested_tolerance <= 0.0_dp .or. &
            requested_iterations < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM fit: C, gamma, tolerance, and iterations must be valid")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM fit: at least two classes are required")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        allocate(effective_weight(n_samples), class_factors(n_classes))
        effective_weight = 1.0_dp
        class_factors = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVO RBF SVM fit: sample weights must be finite and nonnegative")
                return
            end if
            effective_weight = sample_weight
        end if
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVO RBF SVM fit: class weights must be finite and positive "// &
                    "in sorted class order")
                return
            end if
            class_factors = class_weight
            do i = 1, n_samples
                do j = 1, n_classes
                    if (labels(i) == classes(j)) then
                        effective_weight(i) = effective_weight(i)*class_factors(j)
                        exit
                    end if
                end do
            end do
        end if
        if (.not. ieee_is_finite(sum(effective_weight)) .or. &
            sum(effective_weight) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM fit: effective weights must have positive mass")
            return
        end if

        candidate%class_label = classes
        candidate%n_classes = n_classes
        candidate%n_pairs = n_classes*(n_classes - 1)/2
        candidate%n_features = n_features
        allocate(candidate%models(candidate%n_pairs), candidate%pair_negative(candidate%n_pairs), &
            candidate%pair_positive(candidate%n_pairs), candidate%parameter_offset(candidate%n_pairs + 1))
        candidate%parameter_offset(1) = 1
        pair_index = 0
        do i = 1, n_classes - 1
            do j = i + 1, n_classes
                pair_index = pair_index + 1
                candidate%pair_negative(pair_index) = i
                candidate%pair_positive(pair_index) = j
                n_pair_samples = count((labels == classes(i)) .or. &
                    (labels == classes(j)))
                if (n_pair_samples < 2) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "OVO RBF SVM fit: every class pair needs two samples")
                    return
                end if
                allocate(pair_x(n_pair_samples, n_features), pair_labels(n_pair_samples), &
                    pair_weight(n_pair_samples), indices(n_pair_samples))
                index = 0
                do k = 1, n_samples
                    if (labels(k) == classes(i) .or. labels(k) == classes(j)) then
                        index = index + 1
                        indices(index) = k
                    end if
                end do
                do k = 1, n_pair_samples
                    pair_x(k, :) = x(indices(k), :)
                    pair_labels(k) = 0
                    if (labels(indices(k)) == classes(j)) pair_labels(k) = 1
                    pair_weight(k) = effective_weight(indices(k))
                end do
                if (.not. ieee_is_finite(sum(pair_weight)) .or. &
                    sum(pair_weight) <= 0.0_dp .or. &
                    sum(pair_weight, mask=pair_labels == 0) <= 0.0_dp .or. &
                    sum(pair_weight, mask=pair_labels == 1) <= 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "OVO RBF SVM fit: every class pair needs positive weight mass")
                    return
                end if
                call candidate%models(pair_index)%fit(pair_x, pair_labels, status, &
                    c=requested_c, gamma=requested_gamma, &
                    max_iterations=requested_iterations, tolerance=requested_tolerance, &
                    sample_weight=pair_weight)
                deallocate(pair_x, pair_labels, pair_weight, indices)
                if (status%code /= FORTNUM_OK) then
                    write (failure_message, '(a,i0,a,a)') &
                        "OVO RBF SVM fit: pair estimator ", pair_index, " failed: ", &
                        trim(status%msg)
                    call status_set(status, FORTNUM_DOMAIN_ERROR, trim(failure_message))
                    return
                end if
            end do
        end do
        do pair_index = 1, candidate%n_pairs
            candidate%parameter_offset(pair_index + 1) = candidate%parameter_offset(pair_index) + &
                candidate%models(pair_index)%parameter_count()
        end do
        candidate%is_fitted = .true.
        call publish_candidate(self, candidate)
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_fit

    subroutine ovo_rbf_svm_decision(self, x, scores, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_score(:)
        integer :: pair_index

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(scores) /= [size(x, 1), self%n_pairs])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision: inputs must be finite")
            return
        end if
        allocate(pair_score(size(x, 1)))
        do pair_index = 1, self%n_pairs
            call self%models(pair_index)%decision_function(x, pair_score, status)
            if (status%code /= FORTNUM_OK) return
            scores(:, pair_index) = pair_score
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_decision

    subroutine ovo_rbf_svm_decision_device(self, device, x, scores, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        scores = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "OVO RBF SVM device decision: resident CUDA RBF kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM device decision: device kind is invalid")
        end select
    end subroutine ovo_rbf_svm_decision_device

    subroutine ovo_rbf_svm_decision_jvp(self, x, x_dot, scores, scores_dot, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: scores(:, :), scores_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_score(:), pair_dot(:), theta_dot(:)
        integer :: pair_index

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%n_features .or. any(shape(x_dot) /= shape(x)) .or. &
            any(shape(scores) /= [size(x, 1), self%n_pairs]) .or. &
            any(shape(scores_dot) /= shape(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision JVP: inputs and tangents must be finite")
            return
        end if
        allocate(pair_score(size(x, 1)), pair_dot(size(x, 1)))
        do pair_index = 1, self%n_pairs
            allocate(theta_dot(self%models(pair_index)%parameter_count()))
            theta_dot = 0.0_dp
            call self%models(pair_index)%decision_function_jvp(x, theta_dot, x_dot, &
                pair_score, pair_dot, status)
            deallocate(theta_dot)
            if (status%code /= FORTNUM_OK) return
            scores(:, pair_index) = pair_score
            scores_dot(:, pair_index) = pair_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_decision_jvp

    subroutine ovo_rbf_svm_decision_parameter_jvp(self, x, direction, scores, &
            scores_dot, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: scores(:, :), scores_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_score(:), pair_dot(:), x_zero(:, :)
        integer :: pair_index, first, last

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%n_features .or. size(direction) /= self%parameter_count() .or. &
            any(shape(scores) /= [size(x, 1), self%n_pairs]) .or. &
            any(shape(scores_dot) /= shape(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision parameter JVP: model, parameter, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision parameter JVP: inputs and tangents must be finite")
            return
        end if
        allocate(pair_score(size(x, 1)), pair_dot(size(x, 1)), &
            x_zero(size(x, 1), self%n_features))
        x_zero = 0.0_dp
        do pair_index = 1, self%n_pairs
            first = self%parameter_offset(pair_index)
            last = self%parameter_offset(pair_index + 1) - 1
            call self%models(pair_index)%decision_function_jvp(x, direction(first:last), &
                x_zero, pair_score, pair_dot, status)
            if (status%code /= FORTNUM_OK) return
            scores(:, pair_index) = pair_score
            scores_dot(:, pair_index) = pair_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_decision_parameter_jvp

    subroutine ovo_rbf_svm_decision_vjp(self, x, scores_bar, x_bar, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_theta_bar(:), pair_x_bar(:, :)
        integer :: pair_index

        x_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(shape(scores_bar) /= [size(x, 1), self%n_pairs]) .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(pair_x_bar(size(x, 1), self%n_features), &
            pair_theta_bar(self%models(1)%parameter_count()))
        do pair_index = 1, self%n_pairs
            if (pair_index > 1) then
                deallocate(pair_theta_bar)
                allocate(pair_theta_bar(self%models(pair_index)%parameter_count()))
            end if
            call self%models(pair_index)%decision_function_vjp(x, scores_bar(:, pair_index), &
                pair_theta_bar, pair_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + pair_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_decision_vjp

    subroutine ovo_rbf_svm_decision_parameter_vjp(self, x, scores_bar, parameter_bar, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_theta_bar(:), pair_x_bar(:, :)
        integer :: pair_index, first, last

        parameter_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(shape(scores_bar) /= [size(x, 1), self%n_pairs]) .or. &
            size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision parameter VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM decision parameter VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(pair_x_bar(size(x, 1), self%n_features))
        do pair_index = 1, self%n_pairs
            allocate(pair_theta_bar(self%models(pair_index)%parameter_count()))
            call self%models(pair_index)%decision_function_vjp(x, scores_bar(:, pair_index), &
                pair_theta_bar, pair_x_bar, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(pair_theta_bar)
                return
            end if
            first = self%parameter_offset(pair_index)
            last = self%parameter_offset(pair_index + 1) - 1
            parameter_bar(first:last) = pair_theta_bar
            deallocate(pair_theta_bar)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_decision_parameter_vjp

    subroutine ovo_rbf_svm_predict_proba(self, x, probabilities, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_probabilities(:, :)
        real(dp) :: positive
        integer :: pair_index, negative_class, positive_class

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM probability: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM probability: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM probability: inputs must be finite")
            return
        end if
        allocate(pair_probabilities(size(x, 1), 2))
        probabilities = 0.0_dp
        do pair_index = 1, self%n_pairs
            call self%models(pair_index)%predict_proba(x, pair_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            negative_class = self%pair_negative(pair_index)
            positive_class = self%pair_positive(pair_index)
            positive = 1.0_dp/real(self%n_pairs, dp)
            probabilities(:, negative_class) = probabilities(:, negative_class) + &
                pair_probabilities(:, 1)*positive
            probabilities(:, positive_class) = probabilities(:, positive_class) + &
                pair_probabilities(:, 2)*positive
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_predict_proba

    subroutine ovo_rbf_svm_predict_proba_device(self, device, x, probabilities, status)
        !! Predict probabilities through the explicit device contract.
        !!
        !! The pairwise RBF SVM reduction has no resident CUDA kernel yet.
        !! CPU dispatch is exact; CUDA returns a typed refusal rather than a
        !! hidden host fallback or an unaccounted transfer.
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "OVO RBF SVM device prediction: no resident CUDA RBF SVM kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM device prediction: device kind is invalid")
        end select
    end subroutine ovo_rbf_svm_predict_proba_device

    subroutine ovo_rbf_svm_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_probabilities(:, :), pair_dot(:, :)
        real(dp), allocatable :: theta_dot(:)
        real(dp) :: positive
        integer :: pair_index, negative_class, positive_class

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM probability JVP: tangent or output shape is invalid")
            return
        end if
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM probability JVP: model or input shape is invalid")
            return
        end if
        allocate(pair_probabilities(size(x, 1), 2), pair_dot(size(x, 1), 2))
        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        positive = 1.0_dp/real(self%n_pairs, dp)
        do pair_index = 1, self%n_pairs
            allocate(theta_dot(self%models(pair_index)%parameter_count()))
            theta_dot = 0.0_dp
            call self%models(pair_index)%predict_proba_jvp(x, theta_dot, x_dot, &
                pair_probabilities, pair_dot, status)
            deallocate(theta_dot)
            if (status%code /= FORTNUM_OK) return
            negative_class = self%pair_negative(pair_index)
            positive_class = self%pair_positive(pair_index)
            probabilities(:, negative_class) = probabilities(:, negative_class) + &
                positive*pair_probabilities(:, 1)
            probabilities(:, positive_class) = probabilities(:, positive_class) + &
                positive*pair_probabilities(:, 2)
            probabilities_dot(:, negative_class) = probabilities_dot(:, negative_class) + &
                positive*pair_dot(:, 1)
            probabilities_dot(:, positive_class) = probabilities_dot(:, positive_class) + &
                positive*pair_dot(:, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_predict_proba_jvp

    subroutine ovo_rbf_svm_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_probabilities(:, :), pair_dot(:, :), x_dot(:, :)
        real(dp), allocatable :: theta_slice(:)
        real(dp) :: positive
        integer :: pair_index, negative_class, positive_class, first, last

        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM parameter JVP: model, parameter, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM parameter JVP: inputs and tangents must be finite")
            return
        end if
        allocate(pair_probabilities(size(x, 1), 2), pair_dot(size(x, 1), 2), &
            x_dot(size(x, 1), self%n_features))
        x_dot = 0.0_dp
        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        positive = 1.0_dp/real(self%n_pairs, dp)
        do pair_index = 1, self%n_pairs
            first = self%parameter_offset(pair_index)
            last = self%parameter_offset(pair_index + 1) - 1
            theta_slice = theta_dot(first:last)
            call self%models(pair_index)%predict_proba_jvp(x, theta_slice, x_dot, &
                pair_probabilities, pair_dot, status)
            if (status%code /= FORTNUM_OK) return
            negative_class = self%pair_negative(pair_index)
            positive_class = self%pair_positive(pair_index)
            probabilities(:, negative_class) = probabilities(:, negative_class) + &
                positive*pair_probabilities(:, 1)
            probabilities(:, positive_class) = probabilities(:, positive_class) + &
                positive*pair_probabilities(:, 2)
            probabilities_dot(:, negative_class) = probabilities_dot(:, negative_class) + &
                positive*pair_dot(:, 1)
            probabilities_dot(:, positive_class) = probabilities_dot(:, positive_class) + &
                positive*pair_dot(:, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_predict_proba_parameter_jvp

    subroutine ovo_rbf_svm_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_probabilities_bar(:, :), pair_theta_bar(:)
        real(dp), allocatable :: pair_x_bar(:, :)
        real(dp) :: positive
        integer :: pair_index, negative_class, positive_class

        x_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM probability VJP: model or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM probability VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(pair_probabilities_bar(size(x, 1), 2), &
            pair_x_bar(size(x, 1), self%n_features))
        positive = 1.0_dp/real(self%n_pairs, dp)
        do pair_index = 1, self%n_pairs
            negative_class = self%pair_negative(pair_index)
            positive_class = self%pair_positive(pair_index)
            pair_probabilities_bar(:, 1) = positive* &
                probabilities_bar(:, negative_class)
            pair_probabilities_bar(:, 2) = positive* &
                probabilities_bar(:, positive_class)
            allocate(pair_theta_bar(self%models(pair_index)%parameter_count()))
            call self%models(pair_index)%predict_proba_vjp(x, pair_probabilities_bar, &
                pair_theta_bar, pair_x_bar, status)
            deallocate(pair_theta_bar)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + pair_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_predict_proba_vjp

    subroutine ovo_rbf_svm_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_probabilities_bar(:, :), pair_theta_bar(:)
        real(dp), allocatable :: pair_x_bar(:, :)
        real(dp) :: positive
        integer :: pair_index, negative_class, positive_class, first, last

        theta_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM parameter VJP: model or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM parameter VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(pair_probabilities_bar(size(x, 1), 2), &
            pair_x_bar(size(x, 1), self%n_features))
        positive = 1.0_dp/real(self%n_pairs, dp)
        do pair_index = 1, self%n_pairs
            negative_class = self%pair_negative(pair_index)
            positive_class = self%pair_positive(pair_index)
            pair_probabilities_bar(:, 1) = positive* &
                probabilities_bar(:, negative_class)
            pair_probabilities_bar(:, 2) = positive* &
                probabilities_bar(:, positive_class)
            allocate(pair_theta_bar(self%models(pair_index)%parameter_count()))
            call self%models(pair_index)%predict_proba_vjp(x, pair_probabilities_bar, &
                pair_theta_bar, pair_x_bar, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(pair_theta_bar)
                return
            end if
            first = self%parameter_offset(pair_index)
            last = self%parameter_offset(pair_index + 1) - 1
            theta_bar(first:last) = pair_theta_bar
            deallocate(pair_theta_bar)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_predict_proba_parameter_vjp

    subroutine ovo_rbf_svm_predict(self, x, labels, status)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, j, best

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            best = 1
            do j = 2, self%n_classes
                if (probabilities(i, j) > probabilities(i, best)) best = j
            end do
            labels(i) = self%class_label(best)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_predict

    subroutine ovo_rbf_svm_predict_device(self, device, x, labels, status)
        !! Class-label prediction through the explicit device contract.
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "OVO RBF SVM device prediction: no resident CUDA RBF SVM kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM device prediction: device kind is invalid")
        end select
    end subroutine ovo_rbf_svm_predict_device

    function ovo_rbf_svm_classes(self) result(labels)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function ovo_rbf_svm_classes

    function ovo_rbf_svm_pair_classes(self) result(pairs)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        integer, allocatable :: pairs(:, :)
        integer :: pair_index

        allocate(pairs(2, self%n_pairs))
        pairs = 0
        if (.not. self%is_fitted) return
        do pair_index = 1, self%n_pairs
            pairs(1, pair_index) = self%class_label(self%pair_negative(pair_index))
            pairs(2, pair_index) = self%class_label(self%pair_positive(pair_index))
        end do
    end function ovo_rbf_svm_pair_classes

    integer function ovo_rbf_svm_pair_count(self) result(count)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self

        count = self%n_pairs
    end function ovo_rbf_svm_pair_count

    integer function ovo_rbf_svm_class_count(self) result(count)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self

        count = self%n_classes
    end function ovo_rbf_svm_class_count

    integer function ovo_rbf_svm_feature_count(self) result(count)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self

        count = self%n_features
    end function ovo_rbf_svm_feature_count

    integer function ovo_rbf_svm_parameter_count(self) result(count)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self

        count = 0
        if (self%is_fitted .and. allocated(self%parameter_offset)) then
            count = self%parameter_offset(self%n_pairs + 1) - 1
        end if
    end function ovo_rbf_svm_parameter_count

    function ovo_rbf_svm_parameters(self) result(values)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:), model_values(:)
        integer :: pair_index, first, last

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        do pair_index = 1, self%n_pairs
            model_values = self%models(pair_index)%parameters()
            first = self%parameter_offset(pair_index)
            last = self%parameter_offset(pair_index + 1) - 1
            values(first:last) = model_values
        end do
    end function ovo_rbf_svm_parameters

    subroutine ovo_rbf_svm_set_parameters(self, values, status)
        class(ovo_rbf_svm_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        type(rbf_svm_classifier_t), allocatable :: backup(:)
        real(dp) :: gamma
        integer :: pair_index, first, last

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO RBF SVM set_parameters: model, shape, or values are invalid")
            return
        end if
        do pair_index = 1, self%n_pairs
            first = self%parameter_offset(pair_index)
            last = self%parameter_offset(pair_index + 1) - 1
            gamma = exp(values(last))
            if (.not. ieee_is_finite(gamma) .or. gamma <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVO RBF SVM set_parameters: log-gamma is invalid")
                return
            end if
        end do
        backup = self%models
        do pair_index = 1, self%n_pairs
            first = self%parameter_offset(pair_index)
            last = self%parameter_offset(pair_index + 1) - 1
            call self%models(pair_index)%set_parameters(values(first:last), status)
            if (status%code /= FORTNUM_OK) then
                self%models = backup
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_rbf_svm_set_parameters

    subroutine publish_candidate(self, candidate)
        class(ovo_rbf_svm_classifier_t), intent(inout) :: self
        type(ovo_rbf_svm_classifier_t), intent(inout) :: candidate

        if (allocated(self%models)) deallocate(self%models)
        if (allocated(self%class_label)) deallocate(self%class_label)
        if (allocated(self%pair_negative)) deallocate(self%pair_negative)
        if (allocated(self%pair_positive)) deallocate(self%pair_positive)
        if (allocated(self%parameter_offset)) deallocate(self%parameter_offset)
        call move_alloc(candidate%models, self%models)
        call move_alloc(candidate%class_label, self%class_label)
        call move_alloc(candidate%pair_negative, self%pair_negative)
        call move_alloc(candidate%pair_positive, self%pair_positive)
        call move_alloc(candidate%parameter_offset, self%parameter_offset)
        self%n_classes = candidate%n_classes
        self%n_pairs = candidate%n_pairs
        self%n_features = candidate%n_features
        self%is_fitted = candidate%is_fitted
    end subroutine publish_candidate

    logical function ovo_rbf_svm_fitted(self) result(is_fitted)
        class(ovo_rbf_svm_classifier_t), intent(in) :: self

        is_fitted = self%is_fitted
    end function ovo_rbf_svm_fitted

    logical function ovo_rbf_svm_device_supported(self, device_kind) result(supported)
        !! Report support without inferring a host fallback for accelerators.
        class(ovo_rbf_svm_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function ovo_rbf_svm_device_supported

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, n
        logical :: found

        allocate(work(size(labels)))
        n = 0
        do i = 1, size(labels)
            found = .false.
            do j = 1, n
                if (work(j) == labels(i)) then
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) then
                n = n + 1
                work(n) = labels(i)
            end if
        end do
        do i = 2, n
            j = i
            do while (j > 1)
                if (work(j) >= work(j - 1)) exit
                call swap(work(j), work(j - 1))
                j = j - 1
            end do
        end do
        allocate(classes(n))
        classes = work(:n)
    end subroutine sorted_unique_labels

    subroutine swap(left, right)
        integer, intent(inout) :: left, right
        integer :: temporary

        temporary = left
        left = right
        right = temporary
    end subroutine swap

end module fortml_ovo_rbf_svm_classifier
