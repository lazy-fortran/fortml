module fortml_ovo_logistic_classifier
    !! Deterministic one-vs-one logistic classification.
    !!
    !! A binary logistic model is fitted for every pair of sorted classes.  A
    !! pair model predicts the second class probability; deployment combines
    !! these pair probabilities as pairwise votes and divides by the number
    !! of pair models.  This is a deliberately explicit, differentiable
    !! probability policy rather than an implicit claim to implement
    !! scikit-learn's opaque pairwise coupling solver.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_logistic_regression, only: logistic_regression_t
    implicit none
    private

    type, public :: ovo_logistic_classifier_t
        private
        type(logistic_regression_t), allocatable :: models(:)
        integer, allocatable :: class_label(:)
        integer, allocatable :: pair_negative(:), pair_positive(:)
        integer :: n_classes = 0
        integer :: n_pairs = 0
        integer :: n_features = 0
        integer :: n_parameters_per_model = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => ovo_logistic_fit
        procedure, public :: decision_function => ovo_logistic_decision
        procedure, public :: predict_proba => ovo_logistic_predict_proba
        procedure, public :: predict_proba_jvp => ovo_logistic_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            ovo_logistic_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => ovo_logistic_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            ovo_logistic_predict_proba_parameter_vjp
        procedure, public :: predict => ovo_logistic_predict
        procedure, public :: classes => ovo_logistic_classes
        procedure, public :: pair_classes => ovo_logistic_pair_classes
        procedure, public :: pair_count => ovo_logistic_pair_count
        procedure, public :: class_count => ovo_logistic_class_count
        procedure, public :: feature_count => ovo_logistic_feature_count
        procedure, public :: parameter_count => ovo_logistic_parameter_count
        procedure, public :: parameters => ovo_logistic_parameters
        procedure, public :: set_parameters => ovo_logistic_set_parameters
        procedure, public :: fitted => ovo_logistic_fitted
    end type ovo_logistic_classifier_t

    public :: ovo_logistic_fit
    public :: ovo_logistic_decision
    public :: ovo_logistic_predict_proba
    public :: ovo_logistic_predict_proba_jvp
    public :: ovo_logistic_predict_proba_parameter_jvp
    public :: ovo_logistic_predict_proba_vjp
    public :: ovo_logistic_predict_proba_parameter_vjp
    public :: ovo_logistic_predict

contains

    subroutine ovo_logistic_fit(self, x, labels, status, l2, fit_intercept, &
            max_iterations, tolerance, sample_weight, class_weight)
        class(ovo_logistic_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        integer, allocatable :: classes(:), pair_labels(:), indices(:)
        real(dp), allocatable :: effective_weight(:), class_factors(:)
        real(dp), allocatable :: pair_x(:, :), pair_weight(:)
        integer :: i, j, k, n_samples, n_features, n_classes, n_pair_samples
        integer :: pair_index, index
        character(256) :: failure_message

        self%is_fitted = .false.
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
                size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic fit: inputs must be finite")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic fit: at least two classes are required")
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
                    "OVO logistic fit: sample weights must be finite and nonnegative")
                return
            end if
            effective_weight = sample_weight
        end if
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes .or. &
                    any(.not. ieee_is_finite(class_weight)) .or. &
                    any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVO logistic fit: class weights must be finite and positive "// &
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
                "OVO logistic fit: effective weights must have positive mass")
            return
        end if

        self%class_label = classes
        self%n_classes = n_classes
        self%n_pairs = n_classes*(n_classes - 1)/2
        self%n_features = n_features
        allocate(self%models(self%n_pairs), self%pair_negative(self%n_pairs), &
            self%pair_positive(self%n_pairs))
        pair_index = 0
        do i = 1, n_classes - 1
            do j = i + 1, n_classes
                pair_index = pair_index + 1
                self%pair_negative(pair_index) = i
                self%pair_positive(pair_index) = j
                n_pair_samples = count((labels == classes(i)) .or. &
                    (labels == classes(j)))
                if (n_pair_samples < 2) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "OVO logistic fit: every class pair needs two samples")
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
                        "OVO logistic fit: every class pair needs positive weight mass")
                    return
                end if
                call self%models(pair_index)%fit(pair_x, pair_labels, status, l2=l2, &
                    fit_intercept=fit_intercept, max_iterations=max_iterations, &
                    tolerance=tolerance, sample_weight=pair_weight)
                deallocate(pair_x, pair_labels, pair_weight, indices)
                if (status%code /= FORTNUM_OK) then
                    write (failure_message, '(a,i0,a,a)') &
                        "OVO logistic fit: pair estimator ", pair_index, " failed: ", &
                        trim(status%msg)
                    call status_set(status, FORTNUM_DOMAIN_ERROR, trim(failure_message))
                    return
                end if
            end do
        end do
        self%n_parameters_per_model = self%models(1)%parameter_count()
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_logistic_fit

    subroutine ovo_logistic_decision(self, x, scores, status)
        class(ovo_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_score(:)
        integer :: pair_index

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic decision: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
                any(shape(scores) /= [size(x, 1), self%n_pairs])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic decision: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic decision: inputs must be finite")
            return
        end if
        allocate(pair_score(size(x, 1)))
        do pair_index = 1, self%n_pairs
            call self%models(pair_index)%decision_function(x, pair_score, status)
            if (status%code /= FORTNUM_OK) return
            scores(:, pair_index) = pair_score
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_logistic_decision

    subroutine ovo_logistic_predict_proba(self, x, probabilities, status)
        class(ovo_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: pair_probabilities(:, :)
        real(dp) :: positive
        integer :: pair_index, negative_class, positive_class

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic probability: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
                any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic probability: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic probability: inputs must be finite")
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
    end subroutine ovo_logistic_predict_proba

    subroutine ovo_logistic_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(ovo_logistic_classifier_t), intent(in) :: self
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
                "OVO logistic probability JVP: tangent or output shape is invalid")
            return
        end if
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
                any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic probability JVP: model or input shape is invalid")
            return
        end if
        allocate(pair_probabilities(size(x, 1), 2), pair_dot(size(x, 1), 2), &
            theta_dot(self%n_parameters_per_model))
        theta_dot = 0.0_dp
        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        positive = 1.0_dp/real(self%n_pairs, dp)
        do pair_index = 1, self%n_pairs
            call self%models(pair_index)%predict_proba_jvp(x, theta_dot, x_dot, &
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
    end subroutine ovo_logistic_predict_proba_jvp

    subroutine ovo_logistic_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(ovo_logistic_classifier_t), intent(in) :: self
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
                "OVO logistic parameter JVP: model, parameter, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic parameter JVP: inputs and tangents must be finite")
            return
        end if
        allocate(pair_probabilities(size(x, 1), 2), pair_dot(size(x, 1), 2), &
            x_dot(size(x, 1), self%n_features), theta_slice(self%n_parameters_per_model))
        x_dot = 0.0_dp
        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        positive = 1.0_dp/real(self%n_pairs, dp)
        do pair_index = 1, self%n_pairs
            first = (pair_index - 1)*self%n_parameters_per_model + 1
            last = pair_index*self%n_parameters_per_model
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
    end subroutine ovo_logistic_predict_proba_parameter_jvp

    subroutine ovo_logistic_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(ovo_logistic_classifier_t), intent(in) :: self
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
                "OVO logistic probability VJP: model or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic probability VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(pair_probabilities_bar(size(x, 1), 2), &
            pair_theta_bar(self%n_parameters_per_model), &
            pair_x_bar(size(x, 1), self%n_features))
        positive = 1.0_dp/real(self%n_pairs, dp)
        do pair_index = 1, self%n_pairs
            negative_class = self%pair_negative(pair_index)
            positive_class = self%pair_positive(pair_index)
            pair_probabilities_bar(:, 1) = positive* &
                probabilities_bar(:, negative_class)
            pair_probabilities_bar(:, 2) = positive* &
                probabilities_bar(:, positive_class)
            call self%models(pair_index)%predict_proba_vjp(x, pair_probabilities_bar, &
                pair_theta_bar, pair_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + pair_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_logistic_predict_proba_vjp

    subroutine ovo_logistic_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(ovo_logistic_classifier_t), intent(in) :: self
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
                "OVO logistic parameter VJP: model or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic parameter VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(pair_probabilities_bar(size(x, 1), 2), &
            pair_theta_bar(self%n_parameters_per_model), &
            pair_x_bar(size(x, 1), self%n_features))
        positive = 1.0_dp/real(self%n_pairs, dp)
        do pair_index = 1, self%n_pairs
            negative_class = self%pair_negative(pair_index)
            positive_class = self%pair_positive(pair_index)
            pair_probabilities_bar(:, 1) = positive* &
                probabilities_bar(:, negative_class)
            pair_probabilities_bar(:, 2) = positive* &
                probabilities_bar(:, positive_class)
            call self%models(pair_index)%predict_proba_vjp(x, pair_probabilities_bar, &
                pair_theta_bar, pair_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            first = (pair_index - 1)*self%n_parameters_per_model + 1
            last = pair_index*self%n_parameters_per_model
            theta_bar(first:last) = pair_theta_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_logistic_predict_proba_parameter_vjp

    subroutine ovo_logistic_predict(self, x, labels, status)
        class(ovo_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, j, best

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic predict: output shape is invalid")
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
    end subroutine ovo_logistic_predict

    function ovo_logistic_classes(self) result(labels)
        class(ovo_logistic_classifier_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function ovo_logistic_classes

    function ovo_logistic_pair_classes(self) result(pairs)
        class(ovo_logistic_classifier_t), intent(in) :: self
        integer, allocatable :: pairs(:, :)
        integer :: pair_index

        allocate(pairs(2, self%n_pairs))
        pairs = 0
        if (.not. self%is_fitted) return
        do pair_index = 1, self%n_pairs
            pairs(1, pair_index) = self%class_label(self%pair_negative(pair_index))
            pairs(2, pair_index) = self%class_label(self%pair_positive(pair_index))
        end do
    end function ovo_logistic_pair_classes

    integer function ovo_logistic_pair_count(self) result(count)
        class(ovo_logistic_classifier_t), intent(in) :: self

        count = self%n_pairs
    end function ovo_logistic_pair_count

    integer function ovo_logistic_class_count(self) result(count)
        class(ovo_logistic_classifier_t), intent(in) :: self

        count = self%n_classes
    end function ovo_logistic_class_count

    integer function ovo_logistic_feature_count(self) result(count)
        class(ovo_logistic_classifier_t), intent(in) :: self

        count = self%n_features
    end function ovo_logistic_feature_count

    integer function ovo_logistic_parameter_count(self) result(count)
        class(ovo_logistic_classifier_t), intent(in) :: self

        count = self%n_pairs*self%n_parameters_per_model
    end function ovo_logistic_parameter_count

    function ovo_logistic_parameters(self) result(values)
        class(ovo_logistic_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:), model_values(:)
        integer :: pair_index, first, last

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        do pair_index = 1, self%n_pairs
            model_values = self%models(pair_index)%parameters()
            first = (pair_index - 1)*self%n_parameters_per_model + 1
            last = pair_index*self%n_parameters_per_model
            values(first:last) = model_values
        end do
    end function ovo_logistic_parameters

    subroutine ovo_logistic_set_parameters(self, values, status)
        class(ovo_logistic_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: pair_index, first, last

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVO logistic set_parameters: model, shape, or values are invalid")
            return
        end if
        do pair_index = 1, self%n_pairs
            first = (pair_index - 1)*self%n_parameters_per_model + 1
            last = pair_index*self%n_parameters_per_model
            call self%models(pair_index)%set_parameters(values(first:last), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovo_logistic_set_parameters

    logical function ovo_logistic_fitted(self) result(is_fitted)
        class(ovo_logistic_classifier_t), intent(in) :: self

        is_fitted = self%is_fitted
    end function ovo_logistic_fitted

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

end module fortml_ovo_logistic_classifier
