module fortml_ordinal_logistic_classifier
    !! Ordered cumulative-logit classifier with analytic prediction products.
    !!
    !! The model uses a single linear score and strictly increasing thresholds:
    !! ``P(Y <= k) = sigmoid(threshold(k) - score)``.  Thresholds are fitted
    !! through positive increments, while the public packed parameter vector
    !! contains the coefficients, optional intercept, and actual thresholds.
    !! This keeps the optimizer unconstrained without exposing a surprising
    !! parameterization to callers.  Fit is weighted and uses FortOpt
    !! L-BFGS-B.  Prediction JVP/VJP products are exact for inputs and packed
    !! parameters; the discrete fit operation is not differentiated.
    !!
    !! Device calls deliberately refuse CUDA until a resident ordinal kernel
    !! is linked.  A CUDA request never falls back to host execution.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    type, public :: ordinal_logistic_classifier_t
        private
        real(dp), allocatable :: coefficient(:)
        real(dp) :: intercept = 0.0_dp
        real(dp), allocatable :: threshold(:)
        integer, allocatable :: class_label(:)
        real(dp) :: l2 = 1.0_dp
        logical :: fit_intercept = .true.
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => ordinal_logistic_fit
        procedure, public :: decision_function => ordinal_logistic_decision
        procedure, public :: decision_function_device => &
            ordinal_logistic_decision_device
        procedure, public :: predict_proba => ordinal_logistic_predict_proba
        procedure, public :: predict_proba_device => &
            ordinal_logistic_predict_proba_device
        procedure, public :: predict_proba_jvp => ordinal_logistic_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            ordinal_logistic_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => ordinal_logistic_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            ordinal_logistic_predict_proba_parameter_vjp
        procedure, public :: predict => ordinal_logistic_predict
        procedure, public :: predict_device => ordinal_logistic_predict_device
        procedure, public :: classes => ordinal_logistic_classes
        procedure, public :: class_count => ordinal_logistic_class_count
        procedure, public :: feature_count => ordinal_logistic_feature_count
        procedure, public :: parameter_count => ordinal_logistic_parameter_count
        procedure, public :: parameters => ordinal_logistic_parameters
        procedure, public :: set_parameters => ordinal_logistic_set_parameters
        procedure, public :: thresholds => ordinal_logistic_thresholds
        procedure, public :: set_thresholds => ordinal_logistic_set_thresholds
        procedure, public :: regularization => ordinal_logistic_regularization
        procedure, public :: fitted => ordinal_logistic_fitted
        procedure, public :: device_supported => ordinal_logistic_device_supported
    end type ordinal_logistic_classifier_t

    public :: ordinal_logistic_fit
    public :: ordinal_logistic_decision
    public :: ordinal_logistic_predict_proba
    public :: ordinal_logistic_predict_proba_jvp
    public :: ordinal_logistic_predict_proba_parameter_jvp
    public :: ordinal_logistic_predict_proba_vjp
    public :: ordinal_logistic_predict_proba_parameter_vjp
    public :: ordinal_logistic_predict

contains

    subroutine ordinal_logistic_fit(self, x, labels, status, l2, fit_intercept, &
            max_iterations, tolerance, sample_weight, class_weight)
        class(ordinal_logistic_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: options
        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: theta(:), lower(:), upper(:), weights(:)
        real(dp), allocatable :: effective_class_weight(:)
        real(dp), allocatable :: initial_threshold(:)
        integer, allocatable :: classes(:), encoded(:)
        real(dp) :: penalty, requested_tolerance, weight_sum, cumulative
        integer :: n_samples, n_features, n_classes, n_thresholds
        integer :: n_parameters, iterations, i, j, position, class_index
        logical :: include_intercept

        self%is_fitted = .false.
        if (allocated(self%coefficient)) deallocate(self%coefficient)
        if (allocated(self%threshold)) deallocate(self%threshold)
        if (allocated(self%class_label)) deallocate(self%class_label)
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic fit: inputs must be finite")
            return
        end if
        call sorted_classes(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic fit: at least two ordered classes are required")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_thresholds = n_classes - 1
        allocate(weights(n_samples), encoded(n_samples), &
            effective_class_weight(n_classes))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal logistic fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        effective_class_weight = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal logistic fit: class weights must be finite and positive "// &
                    "in sorted class order")
                return
            end if
            effective_class_weight = class_weight
        end if
        do i = 1, n_samples
            class_index = 0
            do j = 1, n_classes
                if (labels(i) == classes(j)) then
                    class_index = j
                    exit
                end if
            end do
            encoded(i) = class_index
            weights(i) = weights(i)*effective_class_weight(class_index)
        end do
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic fit: effective weights must have positive mass")
            return
        end if
        do j = 1, n_classes
            if (sum(weights, mask=encoded == j) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal logistic fit: every ordered class needs positive weight")
                return
            end if
        end do
        penalty = 1.0_dp
        if (present(l2)) penalty = l2
        if (.not. ieee_is_finite(penalty) .or. penalty < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic fit: L2 penalty must be finite and nonnegative")
            return
        end if
        include_intercept = .true.
        if (present(fit_intercept)) include_intercept = fit_intercept
        iterations = 300
        if (present(max_iterations)) iterations = max_iterations
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        if (iterations < 1 .or. .not. ieee_is_finite(requested_tolerance) .or. &
            requested_tolerance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic fit: optimizer controls are invalid")
            return
        end if

        ! Empirical cumulative logits make a stable, ordered starting point.
        allocate(initial_threshold(n_thresholds))
        cumulative = 0.0_dp
        do j = 1, n_thresholds
            cumulative = cumulative + sum(weights, mask=encoded == j)/weight_sum
            cumulative = min(max(cumulative, 0.05_dp), 0.95_dp)
            initial_threshold(j) = log(cumulative/(1.0_dp - cumulative))
        end do
        n_parameters = n_features + n_thresholds
        if (include_intercept) n_parameters = n_parameters + 1
        allocate(theta(n_parameters), lower(n_parameters), upper(n_parameters))
        theta = 0.0_dp
        position = n_features + 1
        if (include_intercept) then
            theta(position) = 0.0_dp
            position = position + 1
        end if
        ! Internal threshold coordinates are raw first threshold and log-positive
        ! increments; public parameters are converted after optimization.
        theta(position) = initial_threshold(1)
        do j = 2, n_thresholds
            theta(position + j - 1) = log(max(initial_threshold(j) - &
                initial_threshold(j - 1), 1.0e-3_dp))
        end do
        lower = -huge(1.0_dp)
        upper = huge(1.0_dp)
        call objective%initialize(n_parameters, ordinal_objective, status)
        if (status%code /= FORTNUM_OK) return
        options%max_iterations = iterations
        options%gradient_tolerance = requested_tolerance
        options%step_tolerance = min(1.0e-12_dp, requested_tolerance)
        options%objective_tolerance = min(1.0e-12_dp, requested_tolerance)
        call optimizer%minimize(objective, theta, lower, upper, options, result, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. result%state%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal logistic fit: optimizer reached its iteration limit")
            return
        end if
        if (any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal logistic fit: optimizer returned nonfinite parameters")
            return
        end if
        allocate(self%coefficient(n_features), self%threshold(n_thresholds), &
            self%class_label(n_classes))
        self%coefficient = theta(:n_features)
        position = n_features + 1
        self%intercept = 0.0_dp
        if (include_intercept) then
            self%intercept = theta(position)
            position = position + 1
        end if
        call raw_to_threshold(theta(position:), self%threshold, status)
        if (status%code /= FORTNUM_OK) return
        self%class_label = classes
        self%l2 = penalty
        self%fit_intercept = include_intercept
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")

    contains

        subroutine ordinal_objective(parameters, value, gradient, objective_status)
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: value, gradient(:)
            type(fortnum_status_t), intent(out) :: objective_status
            real(dp), allocatable :: thresholds_local(:), q(:), qprime(:)
            real(dp), allocatable :: dp_threshold(:)
            real(dp) :: eta, probability, dp_eta, residual
            real(dp) :: raw_increment, chain
            real(dp), allocatable :: gradient_threshold(:)
            integer :: row, class_position, threshold_position, p

            value = 0.0_dp
            gradient = 0.0_dp
            allocate(thresholds_local(n_thresholds), gradient_threshold(n_thresholds), &
                q(n_thresholds), qprime(n_thresholds), dp_threshold(n_thresholds))
            call raw_to_threshold(parameters(n_features + 1 + merge(1, 0, include_intercept):), &
                thresholds_local, objective_status)
            if (objective_status%code /= FORTNUM_OK) return
            gradient_threshold = 0.0_dp
            do row = 1, n_samples
                eta = dot_product(x(row, :), parameters(:n_features))
                if (include_intercept) eta = eta + parameters(n_features + 1)
                do threshold_position = 1, n_thresholds
                    q(threshold_position) = stable_sigmoid(thresholds_local(threshold_position) - eta)
                    qprime(threshold_position) = q(threshold_position)* &
                        (1.0_dp - q(threshold_position))
                end do
                class_position = encoded(row)
                call class_probability_and_derivatives(class_position, q, qprime, &
                    probability, dp_eta, dp_threshold)
                probability = max(probability, tiny(1.0_dp))
                value = value - weights(row)*log(probability)/weight_sum
                residual = -weights(row)/weight_sum/probability
                gradient(:n_features) = gradient(:n_features) + &
                    residual*dp_eta*x(row, :)
                if (include_intercept) gradient(n_features + 1) = &
                    gradient(n_features + 1) + residual*dp_eta
                ! dp_threshold is the vector of derivatives of p wrt every tau.
                do p = 1, n_thresholds
                    gradient_threshold(p) = gradient_threshold(p) + residual*dp_threshold(p)
                end do
            end do
            gradient(:n_features) = gradient(:n_features) + penalty* &
                parameters(:n_features)
            p = n_features + merge(1, 0, include_intercept)
            call threshold_raw_gradient(parameters(p + 1:), gradient_threshold, &
                gradient(p + 1:))
            value = value + 0.5_dp*penalty*sum(parameters(:n_features)**2)
            call status_set(objective_status, FORTNUM_OK, "")
        end subroutine ordinal_objective

    end subroutine ordinal_logistic_fit

    subroutine ordinal_logistic_decision(self, x, scores, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic decision: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%feature_count() .or. size(scores) /= size(x, 1) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic decision: input or output shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            scores(i) = dot_product(x(i, :), self%coefficient)
            if (self%fit_intercept) scores(i) = scores(i) + self%intercept
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_logistic_decision

    subroutine ordinal_logistic_decision_device(self, device, x, scores, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal logistic device decision: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic device decision: device kind is invalid")
        end select
    end subroutine ordinal_logistic_decision_device

    subroutine ordinal_logistic_predict_proba(self, x, probabilities, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i
        if (.not. model_shapes_valid(self, x, probabilities, status, &
            "ordinal logistic probability")) return
        allocate(scores(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            call probabilities_from_score(scores(i), self%threshold, probabilities(i, :))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_logistic_predict_proba

    subroutine ordinal_logistic_predict_proba_device(self, device, x, probabilities, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal logistic device probability: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic device probability: device kind is invalid")
        end select
    end subroutine ordinal_logistic_predict_proba_device

    subroutine ordinal_logistic_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:), scores_dot(:)
        integer :: i
        if (.not. model_shapes_valid(self, x, probabilities, status, &
            "ordinal logistic probability JVP")) return
        if (any(shape(x_dot) /= shape(x)) .or. any(shape(probabilities_dot) /= &
            shape(probabilities)) .or. any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic probability JVP: tangent or output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)), scores_dot(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            scores_dot(i) = dot_product(x_dot(i, :), self%coefficient)
            call probabilities_from_score_jvp(scores(i), scores_dot(i), self%threshold, &
                probabilities(i, :), probabilities_dot(i, :))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_logistic_predict_proba_jvp

    subroutine ordinal_logistic_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:), scores_dot(:), threshold_dot(:)
        integer :: i, position
        if (.not. model_shapes_valid(self, x, probabilities, status, &
            "ordinal logistic parameter JVP")) return
        if (size(theta_dot) /= self%parameter_count() .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic parameter JVP: tangent or output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)), scores_dot(size(x, 1)), &
            threshold_dot(size(self%threshold)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        position = self%feature_count() + 1
        do i = 1, size(x, 1)
            scores_dot(i) = dot_product(x(i, :), theta_dot(:self%feature_count()))
            if (self%fit_intercept) then
                scores_dot(i) = scores_dot(i) + theta_dot(position)
            end if
        end do
        if (self%fit_intercept) position = position + 1
        threshold_dot = theta_dot(position:)
        do i = 1, size(x, 1)
            call probabilities_from_score_jvp(scores(i), scores_dot(i), self%threshold, &
                probabilities(i, :), probabilities_dot(i, :), threshold_dot)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_logistic_predict_proba_parameter_jvp

    subroutine ordinal_logistic_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:), eta_bar(:), threshold_bar(:)
        integer :: i
        x_bar = 0.0_dp
        if (.not. model_shapes_valid(self, x, probabilities_bar, status, &
            "ordinal logistic probability VJP")) return
        if (any(shape(x_bar) /= shape(x)) .or. any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic probability VJP: cotangent or output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)), eta_bar(size(x, 1)), threshold_bar(size(self%threshold)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        call probability_vjp(scores, self%threshold, probabilities_bar, eta_bar, threshold_bar)
        do i = 1, size(x, 1)
            x_bar(i, :) = eta_bar(i)*self%coefficient
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_logistic_predict_proba_vjp

    subroutine ordinal_logistic_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:), eta_bar(:), threshold_bar(:)
        integer :: i, position
        theta_bar = 0.0_dp
        if (.not. model_shapes_valid(self, x, probabilities_bar, status, &
            "ordinal logistic parameter VJP")) return
        if (size(theta_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic parameter VJP: cotangent or output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)), eta_bar(size(x, 1)), threshold_bar(size(self%threshold)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        call probability_vjp(scores, self%threshold, probabilities_bar, eta_bar, threshold_bar)
        do i = 1, size(x, 1)
            theta_bar(:self%feature_count()) = theta_bar(:self%feature_count()) + &
                eta_bar(i)*x(i, :)
        end do
        position = self%feature_count() + 1
        if (self%fit_intercept) then
            theta_bar(position) = sum(eta_bar)
            position = position + 1
        end if
        theta_bar(position:) = threshold_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_logistic_predict_proba_parameter_vjp

    subroutine ordinal_logistic_predict(self, x, labels, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, class_index
        if (.not. self%is_fitted .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic predict: model, input, or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%class_count()))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            class_index = maxloc(probabilities(i, :), dim=1)
            labels(i) = self%class_label(class_index)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_logistic_predict

    subroutine ordinal_logistic_predict_device(self, device, x, labels, status)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic device predict: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal logistic device predict: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic device predict: device kind is invalid")
        end select
    end subroutine ordinal_logistic_predict_device

    function ordinal_logistic_classes(self) result(labels)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        integer, allocatable :: labels(:)
        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function ordinal_logistic_classes

    integer function ordinal_logistic_class_count(self) result(count)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        if (allocated(self%class_label)) then
            count = size(self%class_label)
        else
            count = 0
        end if
    end function ordinal_logistic_class_count

    integer function ordinal_logistic_feature_count(self) result(count)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        if (allocated(self%coefficient)) then
            count = size(self%coefficient)
        else
            count = 0
        end if
    end function ordinal_logistic_feature_count

    integer function ordinal_logistic_parameter_count(self) result(count)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        count = self%feature_count() + size_or_zero(self%threshold)
        if (self%fit_intercept .and. self%is_fitted) count = count + 1
    end function ordinal_logistic_parameter_count

    function ordinal_logistic_parameters(self) result(values)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: position
        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        values(:self%feature_count()) = self%coefficient
        position = self%feature_count() + 1
        if (self%fit_intercept) then
            values(position) = self%intercept
            position = position + 1
        end if
        values(position:) = self%threshold
    end function ordinal_logistic_parameters

    subroutine ordinal_logistic_set_parameters(self, values, status)
        class(ordinal_logistic_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: position
        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic set_parameters: model or parameter shape is invalid")
            return
        end if
        self%coefficient = values(:self%feature_count())
        position = self%feature_count() + 1
        if (self%fit_intercept) then
            self%intercept = values(position)
            position = position + 1
        end if
        if (size(values(position:)) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic set_parameters: thresholds must be strictly increasing")
            return
        end if
        if (size(values(position:)) > 1) then
            if (any(values(position + 1:) <= values(position:position + size(values(position:)) - 2))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal logistic set_parameters: thresholds must be strictly increasing")
                return
            end if
        end if
        self%threshold = values(position:)
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_logistic_set_parameters

    function ordinal_logistic_thresholds(self) result(values)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        if (allocated(self%threshold)) then
            values = self%threshold
        else
            allocate(values(0))
        end if
    end function ordinal_logistic_thresholds

    subroutine ordinal_logistic_set_thresholds(self, values, status)
        class(ordinal_logistic_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. self%is_fitted .or. size(values) /= size(self%threshold) .or. &
            any(.not. ieee_is_finite(values)) .or. (size(values) > 1 .and. &
            any(values(2:) <= values(:size(values)-1)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic set_thresholds: thresholds must be strictly increasing")
            return
        end if
        self%threshold = values
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_logistic_set_thresholds

    real(dp) function ordinal_logistic_regularization(self) result(value)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        value = self%l2
    end function ordinal_logistic_regularization

    logical function ordinal_logistic_fitted(self) result(value)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        value = self%is_fitted
    end function ordinal_logistic_fitted

    logical function ordinal_logistic_device_supported(self, device_kind) result(value)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            value = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            value = .false.
        case default
            value = .false.
        end select
    end function ordinal_logistic_device_supported

    logical function model_shapes_valid(self, x, values, status, operation) result(valid)
        class(ordinal_logistic_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), values(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation
        valid = .false.
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)//": model is not fitted")
            return
        end if
        if (size(x, 2) /= self%feature_count() .or. &
            any(shape(values) /= [size(x, 1), self%class_count()]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)//": shapes or inputs are invalid")
            return
        end if
        valid = .true.
    end function model_shapes_valid

    subroutine probabilities_from_score(score, thresholds, probabilities)
        real(dp), intent(in) :: score, thresholds(:)
        real(dp), intent(out) :: probabilities(:)
        real(dp) :: previous, current
        integer :: j
        previous = 0.0_dp
        do j = 1, size(thresholds)
            current = stable_sigmoid(thresholds(j) - score)
            probabilities(j) = max(current - previous, tiny(1.0_dp))
            previous = current
        end do
        probabilities(size(thresholds) + 1) = max(1.0_dp - previous, tiny(1.0_dp))
        probabilities = probabilities/sum(probabilities)
    end subroutine probabilities_from_score

    subroutine probabilities_from_score_jvp(score, score_dot, thresholds, probabilities, &
            probabilities_dot, threshold_dot)
        real(dp), intent(in) :: score, score_dot, thresholds(:)
        real(dp), intent(out) :: probabilities(:), probabilities_dot(:)
        real(dp), intent(in), optional :: threshold_dot(:)
        real(dp) :: previous, current, previous_dot, current_dot, threshold_tangent
        integer :: j
        previous = 0.0_dp
        previous_dot = 0.0_dp
        do j = 1, size(thresholds)
            threshold_tangent = 0.0_dp
            if (present(threshold_dot)) threshold_tangent = threshold_dot(j)
            current = stable_sigmoid(thresholds(j) - score)
            current_dot = current*(1.0_dp-current)*(threshold_tangent-score_dot)
            probabilities(j) = max(current-previous, tiny(1.0_dp))
            probabilities_dot(j) = current_dot-previous_dot
            previous = current
            previous_dot = current_dot
        end do
        probabilities(size(thresholds)+1) = max(1.0_dp-previous, tiny(1.0_dp))
        probabilities_dot(size(thresholds)+1) = -previous_dot
        ! The max guards above are only active at machine underflow; normalize
        ! in the same way as prediction and propagate its exact tangent.
        call normalize_with_jvp(probabilities, probabilities_dot)
    end subroutine probabilities_from_score_jvp

    subroutine normalize_with_jvp(values, values_dot)
        real(dp), intent(inout) :: values(:), values_dot(:)
        real(dp) :: total, total_dot
        total = sum(values)
        total_dot = sum(values_dot)
        values_dot = (values_dot*total - values*total_dot)/(total*total)
        values = values/total
    end subroutine normalize_with_jvp

    subroutine probability_vjp(scores, thresholds, probability_bar, eta_bar, threshold_bar)
        real(dp), intent(in) :: scores(:), thresholds(:), probability_bar(:, :)
        real(dp), intent(out) :: eta_bar(:), threshold_bar(:)
        real(dp) :: q(size(thresholds)), qp(size(thresholds)), qbar
        real(dp) :: raw(size(thresholds)+1), raw_bar(size(thresholds)+1)
        real(dp) :: normalized(size(thresholds)+1), total, cotangent_mean
        integer :: i, j
        eta_bar = 0.0_dp
        threshold_bar = 0.0_dp
        do i = 1, size(scores)
            do j = 1, size(thresholds)
                q(j) = stable_sigmoid(thresholds(j)-scores(i))
                qp(j) = q(j)*(1.0_dp-q(j))
            end do
            raw(1) = q(1)
            do j = 2, size(thresholds)
                raw(j) = q(j)-q(j-1)
            end do
            raw(size(thresholds)+1) = 1.0_dp-q(size(thresholds))
            total = sum(raw)
            normalized = raw/total
            cotangent_mean = dot_product(probability_bar(i, :), normalized)
            raw_bar = (probability_bar(i, :)-cotangent_mean)/total
            do j = 1, size(thresholds)
                qbar = raw_bar(j)-raw_bar(j+1)
                threshold_bar(j) = threshold_bar(j)+qbar*qp(j)
                eta_bar(i) = eta_bar(i)-qbar*qp(j)
            end do
        end do
    end subroutine probability_vjp

    subroutine class_probability_and_derivatives(class_position, q, qprime, probability, &
            dp_eta, dp_threshold)
        integer, intent(in) :: class_position
        real(dp), intent(in) :: q(:), qprime(:)
        real(dp), intent(out) :: probability, dp_eta, dp_threshold(:)
        integer :: n_thresholds
        n_thresholds = size(q)
        dp_threshold = 0.0_dp
        if (class_position == 1) then
            probability = q(1)
            dp_eta = -qprime(1)
            dp_threshold(1) = qprime(1)
        else if (class_position == n_thresholds + 1) then
            probability = 1.0_dp-q(n_thresholds)
            dp_eta = qprime(n_thresholds)
            dp_threshold(n_thresholds) = -qprime(n_thresholds)
        else
            probability = q(class_position)-q(class_position-1)
            dp_eta = -qprime(class_position)+qprime(class_position-1)
            dp_threshold(class_position) = qprime(class_position)
            dp_threshold(class_position-1) = -qprime(class_position-1)
        end if
    end subroutine class_probability_and_derivatives

    subroutine raw_to_threshold(raw, thresholds, status)
        real(dp), intent(in) :: raw(:)
        real(dp), intent(out) :: thresholds(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: j
        if (size(raw) /= size(thresholds) .or. size(raw) < 1 .or. &
            any(.not. ieee_is_finite(raw))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic threshold coordinates are invalid")
            return
        end if
        thresholds(1) = raw(1)
        do j = 2, size(raw)
            if (raw(j) > log(huge(1.0_dp))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal logistic threshold increment overflow")
                return
            end if
            thresholds(j) = thresholds(j-1) + exp(raw(j))
        end do
        if (any(.not. ieee_is_finite(thresholds))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal logistic thresholds are nonfinite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine raw_to_threshold

    subroutine threshold_raw_gradient(raw, threshold_gradient, raw_gradient)
        real(dp), intent(in) :: raw(:), threshold_gradient(:)
        real(dp), intent(out) :: raw_gradient(:)
        real(dp) :: increment, cumulative
        integer :: j
        raw_gradient = 0.0_dp
        raw_gradient(1) = sum(threshold_gradient)
        do j = 2, size(raw)
            increment = exp(raw(j))
            cumulative = sum(threshold_gradient(j:))
            raw_gradient(j) = increment*cumulative
        end do
    end subroutine threshold_raw_gradient

    pure real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

    subroutine sorted_classes(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, n, temporary
        allocate(work, source=labels)
        do i = 2, size(work)
            temporary = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= temporary) exit
                work(j+1) = work(j)
                j = j - 1
            end do
            work(j+1) = temporary
        end do
        n = 1
        do i = 2, size(work)
            if (work(i) /= work(n)) then
                n = n + 1
                work(n) = work(i)
            end if
        end do
        allocate(classes(n))
        classes = work(:n)
    end subroutine sorted_classes

    integer function size_or_zero(values) result(value)
        real(dp), allocatable, intent(in) :: values(:)
        if (allocated(values)) then
            value = size(values)
        else
            value = 0
        end if
    end function size_or_zero

end module fortml_ordinal_logistic_classifier
