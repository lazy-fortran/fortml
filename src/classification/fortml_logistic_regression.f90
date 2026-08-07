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
        procedure, public :: predict_proba => logistic_predict_proba
        procedure, public :: predict => logistic_predict
        procedure, public :: coefficients => logistic_coefficients
        procedure, public :: intercept_value => logistic_intercept_value
        procedure, public :: classes => logistic_classes
        procedure, public :: feature_count => logistic_feature_count
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
