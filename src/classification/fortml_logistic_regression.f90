module fortml_logistic_regression
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: logistic_regression_t
        private
        real(dp), allocatable :: coefficient(:)
        real(dp) :: intercept = 0.0_dp
        integer :: class_label(2) = 0
        real(dp) :: l2 = 1.0_dp
        logical :: fit_intercept = .true.
    contains
        procedure, public :: fit => logistic_fit
        procedure, public :: decision_function => logistic_decision_function
        procedure, public :: decision_function_jvp => logistic_decision_jvp
        procedure, public :: decision_function_vjp => logistic_decision_vjp
        procedure, public :: predict_proba => logistic_predict_proba
        procedure, public :: predict_proba_jvp => logistic_predict_proba_jvp
        procedure, public :: predict_proba_vjp => logistic_predict_proba_vjp
        procedure, public :: predict => logistic_predict
        procedure, public :: coefficients => logistic_coefficients
        procedure, public :: intercept_value => logistic_intercept_value
        procedure, public :: classes => logistic_classes
        procedure, public :: feature_count => logistic_feature_count
        procedure, public :: parameter_count => logistic_parameter_count
        procedure, public :: parameters => logistic_parameters
        procedure, public :: set_parameters => logistic_set_parameters
        procedure, public :: fitted => logistic_fitted
    end type logistic_regression_t

contains

    subroutine logistic_fit(self, x, labels, status, l2, fit_intercept, &
            max_iterations, tolerance, sample_weight, class_weight)
        class(logistic_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:)
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: options
        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: theta(:), lower(:), upper(:), encoded(:)
        real(dp), allocatable :: weights(:)
        real(dp) :: class_factors(2)
        real(dp) :: penalty, requested_tolerance, weight_sum
        integer :: i, iterations, n_features, n_parameters
        integer :: negative_label, positive_label
        logical :: include_intercept

        if (size(x, 1) < 1 .or. size(x, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic fit: input dimensions must be positive")
            return
        end if
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic fit: label and sample counts differ")
            return
        end if
        allocate(weights(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic fit: inputs must be finite")
            return
        end if

        negative_label = minval(labels)
        positive_label = maxval(labels)
        if (negative_label == positive_label) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic fit: exactly two distinct classes are required")
            return
        end if
        do i = 1, size(labels)
            if (labels(i) /= negative_label .and. labels(i) /= positive_label) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic fit: exactly two distinct classes are required")
                return
            end if
        end do

        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= 2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic fit: class weights must match the two sorted classes")
                return
            end if
            if (any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic fit: class weights must be finite and positive")
                return
            end if
            class_factors = class_weight
        end if
        do i = 1, size(labels)
            if (labels(i) == negative_label) then
                weights(i) = weights(i)*class_factors(1)
            else
                weights(i) = weights(i)*class_factors(2)
            end if
        end do
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic fit: effective weights must have positive mass")
            return
        end if

        penalty = 1.0_dp
        if (present(l2)) penalty = l2
        if (.not. ieee_is_finite(penalty) .or. penalty < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic fit: L2 penalty must be finite and nonnegative")
            return
        end if
        include_intercept = .true.
        if (present(fit_intercept)) include_intercept = fit_intercept
        iterations = 200
        if (present(max_iterations)) iterations = max_iterations
        if (iterations < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic fit: iteration limit must be positive")
            return
        end if
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        if (.not. ieee_is_finite(requested_tolerance) .or. &
            requested_tolerance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic fit: tolerance must be finite and positive")
            return
        end if

        n_features = size(x, 2)
        n_parameters = n_features
        if (include_intercept) n_parameters = n_parameters + 1
        allocate(theta(n_parameters), lower(n_parameters), upper(n_parameters))
        allocate(encoded(size(labels)))
        theta = 0.0_dp
        lower = -huge(1.0_dp)
        upper = huge(1.0_dp)
        encoded = 0.0_dp
        where (labels == positive_label) encoded = 1.0_dp

        call objective%initialize(n_parameters, logistic_objective, status)
        if (status%code /= FORTNUM_OK) return
        options%max_iterations = iterations
        options%gradient_tolerance = requested_tolerance
        options%step_tolerance = min(1.0e-12_dp, requested_tolerance)
        options%objective_tolerance = min(1.0e-12_dp, requested_tolerance)
        call optimizer%minimize(objective, theta, lower, upper, options, result, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. result%state%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "logistic fit: optimizer reached its iteration limit")
            return
        end if
        if (any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "logistic fit: optimizer returned nonfinite parameters")
            return
        end if

        allocate(self%coefficient(n_features))
        self%coefficient = theta(:n_features)
        self%intercept = 0.0_dp
        if (include_intercept) self%intercept = theta(n_parameters)
        self%class_label = [negative_label, positive_label]
        self%l2 = penalty
        self%fit_intercept = include_intercept
        call status_set(status, FORTNUM_OK, "")

    contains

        subroutine logistic_objective(parameters, value, gradient, objective_status)
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: value
            real(dp), intent(out) :: gradient(:)
            type(fortnum_status_t), intent(out) :: objective_status
            real(dp) :: probability, residual, score
            integer :: i, j

            value = 0.0_dp
            gradient = 0.0_dp
            do i = 1, size(x, 1)
                score = 0.0_dp
                do j = 1, n_features
                    score = score + x(i, j)*parameters(j)
                end do
                if (include_intercept) score = score + parameters(n_parameters)
                probability = stable_sigmoid(score)
                residual = probability - encoded(i)
                if (encoded(i) > 0.5_dp) then
                    value = value + weights(i)*stable_softplus(-score)
                else
                    value = value + weights(i)*stable_softplus(score)
                end if
                do j = 1, n_features
                    gradient(j) = gradient(j) + weights(i)*residual*x(i, j)
                end do
                if (include_intercept) then
                    gradient(n_parameters) = gradient(n_parameters) + &
                        weights(i)*residual
                end if
            end do
            value = value/weight_sum
            gradient = gradient/weight_sum
            value = value + 0.5_dp*penalty*sum(parameters(:n_features)**2)
            gradient(:n_features) = gradient(:n_features) + &
                penalty*parameters(:n_features)
            if (.not. ieee_is_finite(value)) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "logistic objective: value is not finite")
                return
            end if
            if (any(.not. ieee_is_finite(gradient))) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "logistic objective: gradient is not finite")
                return
            end if
            call status_set(objective_status, FORTNUM_OK, "")
        end subroutine logistic_objective

    end subroutine logistic_fit

    subroutine logistic_decision_function(self, x, scores, status)
        class(logistic_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (.not. allocated(self%coefficient)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision_function: model is not fitted")
            return
        end if
        if (size(x, 2) /= size(self%coefficient)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision_function: feature count is invalid")
            return
        end if
        if (size(scores) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision_function: output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision_function: inputs must be finite")
            return
        end if

        scores = self%intercept
        do j = 1, size(self%coefficient)
            do i = 1, size(x, 1)
                scores(i) = scores(i) + x(i, j)*self%coefficient(j)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_decision_function

    subroutine logistic_decision_jvp(self, x, theta_dot, x_dot, scores, &
            scores_dot, status)
        class(logistic_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:), scores_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n_features

        if (.not. logistic_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision JVP: model is not fitted")
            return
        end if
        n_features = size(self%coefficient)
        if (size(x, 2) /= n_features .or. size(x_dot, 1) /= size(x, 1) .or. &
            size(x_dot, 2) /= size(x, 2) .or. size(scores) /= size(x, 1) .or. &
            size(scores_dot) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision JVP: parameter tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision JVP: tangents must be finite")
            return
        end if
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        scores_dot = 0.0_dp
        do j = 1, n_features
            do i = 1, size(x, 1)
                scores_dot(i) = scores_dot(i) + self%coefficient(j)*x_dot(i, j) + &
                    theta_dot(j)*x(i, j)
            end do
        end do
        if (self%fit_intercept) scores_dot = scores_dot + theta_dot(n_features + 1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_decision_jvp

    subroutine logistic_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(logistic_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n_features

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. logistic_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision VJP: model is not fitted")
            return
        end if
        n_features = size(self%coefficient)
        if (size(x, 2) /= n_features .or. size(x_bar, 1) /= size(x, 1) .or. &
            size(x_bar, 2) /= size(x, 2) .or. size(scores_bar) /= size(x, 1) .or. &
            size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic decision VJP: inputs and cotangents must be finite")
            return
        end if
        do j = 1, n_features
            do i = 1, size(x, 1)
                theta_bar(j) = theta_bar(j) + scores_bar(i)*x(i, j)
                x_bar(i, j) = scores_bar(i)*self%coefficient(j)
            end do
        end do
        if (self%fit_intercept) theta_bar(n_features + 1) = sum(scores_bar)
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_decision_vjp

    subroutine logistic_predict_proba(self, x, probabilities, status)
        class(logistic_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i

        if (size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic predict_proba: output shape must be (n_samples,2)")
            return
        end if
        allocate(scores(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(scores)
            probabilities(i, 2) = stable_sigmoid(scores(i))
            probabilities(i, 1) = 1.0_dp - probabilities(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_predict_proba

    subroutine logistic_predict_proba_jvp(self, x, theta_dot, x_dot, probabilities, &
            probabilities_dot, status)
        class(logistic_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:), scores_dot(:)
        real(dp) :: positive, positive_dot
        integer :: i

        if (.not. logistic_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic probability JVP: model is not fitted")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. size(probabilities, 2) /= 2 &
            .or. any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic probability JVP: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)), scores_dot(size(x, 1)))
        call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            positive = stable_sigmoid(scores(i))
            positive_dot = positive*(1.0_dp - positive)*scores_dot(i)
            probabilities(i, 2) = positive
            probabilities(i, 1) = 1.0_dp - positive
            probabilities_dot(i, 2) = positive_dot
            probabilities_dot(i, 1) = -positive_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_predict_proba_jvp

    subroutine logistic_predict_proba_vjp(self, x, probabilities_bar, theta_bar, &
            x_bar, status)
        class(logistic_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), scores_bar(:)
        integer :: i

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. logistic_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic probability VJP: model is not fitted")
            return
        end if
        if (size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic probability VJP: cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic probability VJP: cotangent must be finite")
            return
        end if
        allocate(probabilities(size(x, 1), 2), scores_bar(size(x, 1)))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            scores_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))* &
                probabilities(i, 1)*probabilities(i, 2)
        end do
        call self%decision_function_vjp(x, scores_bar, theta_bar, x_bar, status)
    end subroutine logistic_predict_proba_vjp

    subroutine logistic_predict(self, x, labels, status)
        class(logistic_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic predict: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(scores)
            if (scores(i) >= 0.0_dp) then
                labels(i) = self%class_label(2)
            else
                labels(i) = self%class_label(1)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_predict

    function logistic_coefficients(self) result(values)
        class(logistic_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%coefficient)) then
            values = self%coefficient
        else
            allocate(values(0))
        end if
    end function logistic_coefficients

    integer function logistic_parameter_count(self) result(count)
        class(logistic_regression_t), intent(in) :: self

        count = 0
        if (.not. allocated(self%coefficient)) return
        count = size(self%coefficient)
        if (self%fit_intercept) count = count + 1
    end function logistic_parameter_count

    function logistic_parameters(self) result(values)
        class(logistic_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: n_features

        if (.not. allocated(self%coefficient)) then
            allocate(values(0))
            return
        end if
        n_features = size(self%coefficient)
        allocate(values(self%parameter_count()))
        values(:n_features) = self%coefficient
        if (self%fit_intercept) values(n_features + 1) = self%intercept
    end function logistic_parameters

    subroutine logistic_set_parameters(self, values, status)
        class(logistic_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_features

        if (.not. logistic_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic set_parameters: model is not fitted")
            return
        end if
        if (size(values) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic set_parameters: parameter shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic set_parameters: parameters must be finite")
            return
        end if
        n_features = size(self%coefficient)
        self%coefficient = values(:n_features)
        if (self%fit_intercept) self%intercept = values(n_features + 1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_set_parameters

    real(dp) function logistic_intercept_value(self) result(value)
        class(logistic_regression_t), intent(in) :: self

        value = self%intercept
    end function logistic_intercept_value

    function logistic_classes(self) result(labels)
        class(logistic_regression_t), intent(in) :: self
        integer :: labels(2)

        labels = self%class_label
    end function logistic_classes

    integer function logistic_feature_count(self) result(count)
        class(logistic_regression_t), intent(in) :: self

        count = 0
        if (allocated(self%coefficient)) count = size(self%coefficient)
    end function logistic_feature_count

    logical function logistic_fitted(self) result(is_fitted)
        class(logistic_regression_t), intent(in) :: self

        is_fitted = allocated(self%coefficient)
    end function logistic_fitted

    pure real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        real(dp) :: exponential

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            exponential = exp(value)
            probability = exponential/(1.0_dp + exponential)
        end if
    end function stable_sigmoid

    pure real(dp) function stable_softplus(value) result(softplus)
        real(dp), intent(in) :: value

        softplus = max(value, 0.0_dp) + log(1.0_dp + exp(-abs(value)))
    end function stable_softplus

end module fortml_logistic_regression
