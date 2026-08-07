module fortml_ovr_logistic_classifier
    !! One-vs-rest multiclass wrapper around binary logistic regression.
    !!
    !! Each class owns an independent, regularized binary logistic model.  The
    !! positive class probabilities are normalized row-wise for deployment so
    !! that arbitrary integer labels share the same probability-simplex
    !! contract as the multinomial and GP classifiers.  Fit is a discrete
    !! operation.  Prediction products are smooth with respect to inputs and
    !! packed fitted parameters.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_logistic_regression, only: logistic_regression_t
    implicit none
    private

    type, public :: ovr_logistic_classifier_t
        private
        type(logistic_regression_t), allocatable :: models(:)
        integer, allocatable :: class_label(:)
        integer :: n_classes = 0
        integer :: n_features = 0
        integer :: n_parameters_per_model = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => ovr_logistic_fit
        procedure, public :: decision_function => ovr_logistic_decision
        procedure, public :: predict_proba => ovr_logistic_predict_proba
        procedure, public :: predict_proba_jvp => ovr_logistic_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            ovr_logistic_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => ovr_logistic_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            ovr_logistic_predict_proba_parameter_vjp
        procedure, public :: predict => ovr_logistic_predict
        procedure, public :: classes => ovr_logistic_classes
        procedure, public :: class_count => ovr_logistic_class_count
        procedure, public :: feature_count => ovr_logistic_feature_count
        procedure, public :: parameter_count => ovr_logistic_parameter_count
        procedure, public :: parameters => ovr_logistic_parameters
        procedure, public :: set_parameters => ovr_logistic_set_parameters
        procedure, public :: fitted => ovr_logistic_fitted
    end type ovr_logistic_classifier_t

    public :: ovr_logistic_fit
    public :: ovr_logistic_decision
    public :: ovr_logistic_predict_proba
    public :: ovr_logistic_predict_proba_jvp
    public :: ovr_logistic_predict_proba_parameter_jvp
    public :: ovr_logistic_predict_proba_vjp
    public :: ovr_logistic_predict_proba_parameter_vjp
    public :: ovr_logistic_predict

contains

    subroutine ovr_logistic_fit(self, x, labels, status, l2, fit_intercept, &
            max_iterations, tolerance, sample_weight, class_weight)
        class(ovr_logistic_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        integer, allocatable :: classes(:), binary_labels(:)
        real(dp), allocatable :: effective_weight(:), class_factors(:)
        integer :: i, j, n_samples, n_features, n_classes
        character(256) :: failure_message

        self%is_fitted = .false.
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic fit: inputs must be finite")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic fit: at least two classes are required")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        allocate(effective_weight(n_samples), binary_labels(n_samples))
        effective_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic fit: sample weights must be finite and nonnegative")
                return
            end if
            effective_weight = sample_weight
        end if
        allocate(class_factors(n_classes))
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic fit: class weights must be finite and positive "// &
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
                "OVR logistic fit: effective weights must have positive mass")
            return
        end if

        allocate(self%models(n_classes), self%class_label(n_classes))
        self%class_label = classes
        self%n_classes = n_classes
        self%n_features = n_features
        do j = 1, n_classes
            binary_labels = 0
            where (labels == classes(j)) binary_labels = 1
            call self%models(j)%fit(x, binary_labels, status, l2=l2, &
                fit_intercept=fit_intercept, max_iterations=max_iterations, &
                tolerance=tolerance, sample_weight=effective_weight)
            if (status%code /= FORTNUM_OK) then
                write (failure_message, '(a,i0,a,a)') &
                    "OVR logistic fit: binary estimator ", j, " failed: ", &
                    trim(status%msg)
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    trim(failure_message))
                return
            end if
        end do
        self%n_parameters_per_model = self%models(1)%parameter_count()
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_fit

    subroutine ovr_logistic_decision(self, x, scores, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_scores(:)
        integer :: j

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic decision: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(scores) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic decision: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic decision: inputs must be finite")
            return
        end if
        allocate(binary_scores(size(x, 1)))
        do j = 1, self%n_classes
            call self%models(j)%decision_function(x, binary_scores, status)
            if (status%code /= FORTNUM_OK) return
            scores(:, j) = binary_scores
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_decision

    subroutine ovr_logistic_predict_proba(self, x, probabilities, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :)
        real(dp) :: normalization
        integer :: i, j

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability: inputs must be finite")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(probabilities(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic probability: invalid normalization")
                return
            end if
            probabilities(i, :) = probabilities(i, :)/normalization
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba

    subroutine ovr_logistic_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), binary_dot(:, :)
        real(dp), allocatable :: raw(:, :), raw_dot(:, :)
        real(dp), allocatable :: theta_dot(:)
        real(dp) :: normalization, normalization_dot
        integer :: i, j

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability JVP: tangent or output shape is invalid")
            return
        end if
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability JVP: model or input shape is invalid")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2), binary_dot(size(x, 1), 2))
        allocate(raw(size(x, 1), self%n_classes), raw_dot(size(x, 1), self%n_classes))
        allocate(theta_dot(self%n_parameters_per_model))
        theta_dot = 0.0_dp
        do j = 1, self%n_classes
            call self%models(j)%predict_proba_jvp(x, theta_dot, x_dot, &
                binary_probabilities, binary_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
            raw_dot(:, j) = binary_dot(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(raw(i, :))
            normalization_dot = sum(raw_dot(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic probability JVP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/normalization
            probabilities_dot(i, :) = (raw_dot(i, :)*normalization - &
                raw(i, :)*normalization_dot)/(normalization*normalization)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba_jvp

    subroutine ovr_logistic_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), binary_dot(:, :)
        real(dp), allocatable :: raw(:, :), raw_dot(:, :), x_dot(:, :)
        real(dp), allocatable :: theta_slice(:)
        real(dp) :: normalization, normalization_dot
        integer :: i, j, first, last

        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic parameter JVP: model, parameter, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic parameter JVP: inputs and tangents must be finite")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2), binary_dot(size(x, 1), 2))
        allocate(raw(size(x, 1), self%n_classes), raw_dot(size(x, 1), self%n_classes))
        allocate(x_dot(size(x, 1), self%n_features), &
            theta_slice(self%n_parameters_per_model))
        x_dot = 0.0_dp
        do j = 1, self%n_classes
            first = (j - 1)*self%n_parameters_per_model + 1
            last = j*self%n_parameters_per_model
            theta_slice = theta_dot(first:last)
            call self%models(j)%predict_proba_jvp(x, theta_slice, x_dot, &
                binary_probabilities, binary_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
            raw_dot(:, j) = binary_dot(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(raw(i, :))
            normalization_dot = sum(raw_dot(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic parameter JVP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/normalization
            probabilities_dot(i, :) = (raw_dot(i, :)*normalization - &
                raw(i, :)*normalization_dot)/(normalization*normalization)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba_parameter_jvp

    subroutine ovr_logistic_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), raw(:, :), raw_bar(:)
        real(dp), allocatable :: binary_probabilities(:, :)
        real(dp), allocatable :: binary_probabilities_bar(:, :), binary_theta_bar(:)
        real(dp), allocatable :: binary_x_bar(:, :)
        real(dp) :: dot_product_bar
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability VJP: model or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic probability VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            raw(size(x, 1), self%n_classes), binary_probabilities(size(x, 1), 2))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
        end do
        do i = 1, size(x, 1)
            dot_product_bar = sum(raw(i, :))
            if (.not. ieee_is_finite(dot_product_bar) .or. &
                dot_product_bar <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic probability VJP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/dot_product_bar
        end do
        allocate(binary_probabilities_bar(size(x, 1), 2), &
            binary_x_bar(size(x, 1), self%n_features))
        allocate(binary_theta_bar(self%n_parameters_per_model), raw_bar(size(x, 1)))
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                dot_product_bar = sum(probabilities_bar(i, :)*probabilities(i, :))
                raw_bar(i) = (probabilities_bar(i, j) - dot_product_bar)/ &
                    sum(raw(i, :))
            end do
            binary_probabilities_bar(:, 1) = 0.0_dp
            binary_probabilities_bar(:, 2) = raw_bar
            call self%models(j)%predict_proba_vjp(x, binary_probabilities_bar, &
                binary_theta_bar, binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + binary_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba_vjp

    subroutine ovr_logistic_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), raw(:, :), raw_bar(:)
        real(dp), allocatable :: binary_probabilities(:, :), binary_probabilities_bar(:, :)
        real(dp), allocatable :: binary_theta_bar(:), binary_x_bar(:, :)
        real(dp) :: normalization, dot_product_bar
        integer :: i, j, first, last

        theta_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic parameter VJP: model, parameter, or cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic parameter VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            raw(size(x, 1), self%n_classes), binary_probabilities(size(x, 1), 2))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(raw(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "OVR logistic parameter VJP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/normalization
        end do
        allocate(raw_bar(size(x, 1)), &
            binary_probabilities_bar(size(x, 1), 2), &
            binary_theta_bar(self%n_parameters_per_model), &
            binary_x_bar(size(x, 1), self%n_features))
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                dot_product_bar = sum(probabilities_bar(i, :)*probabilities(i, :))
                raw_bar(i) = (probabilities_bar(i, j) - dot_product_bar)/ &
                    sum(raw(i, :))
            end do
            binary_probabilities_bar(:, 1) = 0.0_dp
            binary_probabilities_bar(:, 2) = raw_bar
            call self%models(j)%predict_proba_vjp(x, binary_probabilities_bar, &
                binary_theta_bar, binary_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            first = (j - 1)*self%n_parameters_per_model + 1
            last = j*self%n_parameters_per_model
            theta_bar(first:last) = binary_theta_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_predict_proba_parameter_vjp

    subroutine ovr_logistic_predict(self, x, labels, status)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, j, best

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic predict: output shape is invalid")
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
    end subroutine ovr_logistic_predict

    function ovr_logistic_classes(self) result(labels)
        class(ovr_logistic_classifier_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function ovr_logistic_classes

    integer function ovr_logistic_class_count(self) result(count)
        class(ovr_logistic_classifier_t), intent(in) :: self

        count = self%n_classes
    end function ovr_logistic_class_count

    integer function ovr_logistic_feature_count(self) result(count)
        class(ovr_logistic_classifier_t), intent(in) :: self

        count = self%n_features
    end function ovr_logistic_feature_count

    integer function ovr_logistic_parameter_count(self) result(count)
        class(ovr_logistic_classifier_t), intent(in) :: self

        count = self%n_classes*self%n_parameters_per_model
    end function ovr_logistic_parameter_count

    function ovr_logistic_parameters(self) result(values)
        class(ovr_logistic_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:), model_values(:)
        integer :: i, first, last

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        do i = 1, self%n_classes
            model_values = self%models(i)%parameters()
            first = (i - 1)*self%n_parameters_per_model + 1
            last = i*self%n_parameters_per_model
            values(first:last) = model_values
        end do
    end function ovr_logistic_parameters

    subroutine ovr_logistic_set_parameters(self, values, status)
        class(ovr_logistic_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "OVR logistic set_parameters: model, shape, or values are invalid")
            return
        end if
        do i = 1, self%n_classes
            first = (i - 1)*self%n_parameters_per_model + 1
            last = i*self%n_parameters_per_model
            call self%models(i)%set_parameters(values(first:last), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ovr_logistic_set_parameters

    logical function ovr_logistic_fitted(self) result(is_fitted)
        class(ovr_logistic_classifier_t), intent(in) :: self

        is_fitted = self%is_fitted
    end function ovr_logistic_fitted

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

end module fortml_ovr_logistic_classifier
