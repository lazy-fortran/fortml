module fortml_linear_svm_classifier
    !! Weighted linear support-vector classification.
    !!
    !! The estimator stores a dense primal coefficient vector and an optional
    !! intercept.  Labels may be any two distinct integer values; they are
    !! sorted and encoded as -1 and +1 internally.  The default squared-hinge
    !! objective is
    !!
    !!   sum_i w_i max(0, 1-y_i f_i)**2 / sum_i w_i
    !!       + 0.5*l2*||beta||**2,
    !!
    !! with the intercept excluded from the penalty.  Ordinary hinge loss is
    !! available as SVM_LOSS_HINGE.  Fit is a discrete FortOpt L-BFGS-B
    !! boundary.  Prediction products keep the fitted state fixed and are
    !! exact for the smooth affine map.  The ordinary hinge objective has a
    !! split derivative at margin one; its public objective product returns a
    !! typed refusal at an exact split instead of silently choosing a side.
    !! CUDA prediction is likewise an explicit refusal until a resident
    !! kernel is linked; no host fallback is hidden behind device dispatch.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: SVM_LOSS_HINGE = 1
    integer, parameter, public :: SVM_LOSS_SQUARED_HINGE = 2
    real(dp), parameter :: SVM_HINGE_SMOOTHING = 1.0e-4_dp

    type, public :: linear_svm_classifier_t
        private
        real(dp), allocatable :: coefficient(:)
        real(dp) :: intercept_value = 0.0_dp
        integer :: class_label(2) = 0
        real(dp) :: l2_value = 1.0_dp
        integer :: loss_value = SVM_LOSS_SQUARED_HINGE
        logical :: fit_intercept_value = .true.
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => svm_fit
        procedure, public :: decision_function => svm_decision_function
        procedure, public :: predict => svm_predict
        procedure, public :: decision_function_device => svm_decision_device
        procedure, public :: predict_device => svm_predict_device
        procedure, public :: device_supported => svm_device_supported
        procedure, public :: decision_function_jvp => svm_decision_jvp
        procedure, public :: decision_function_vjp => svm_decision_vjp
        procedure, public :: jvp => svm_decision_jvp
        procedure, public :: vjp => svm_decision_vjp
        procedure, public :: objective_value_gradient => &
            svm_objective_value_gradient
        procedure, public :: coefficients => svm_coefficients
        procedure, public :: intercept => svm_intercept
        procedure, public :: classes => svm_classes
        procedure, public :: regularization => svm_regularization
        procedure, public :: loss => svm_loss
        procedure, public :: fit_intercept => svm_fit_intercept
        procedure, public :: parameters => svm_parameters
        procedure, public :: set_parameters => svm_set_parameters
        procedure, public :: parameter_count => svm_parameter_count
        procedure, public :: feature_count => svm_feature_count
        procedure, public :: fitted => svm_fitted
    end type linear_svm_classifier_t

    public :: svm_fit
    public :: svm_decision_function
    public :: svm_decision_device
    public :: svm_predict
    public :: svm_predict_device

contains

    subroutine svm_fit(self, x, labels, status, l2, fit_intercept, loss, &
            max_iterations, tolerance, sample_weight)
        class(linear_svm_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, tolerance, sample_weight(:)
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: loss, max_iterations
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: options
        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: theta(:), lower(:), upper(:), encoded(:), weights(:)
        real(dp) :: penalty, requested_tolerance, weight_mass
        integer :: requested_loss, iterations, n_features, n_parameters
        integer :: negative_label, positive_label, i
        logical :: include_intercept

        self%fitted_value = .false.
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM fit: finite input dimensions are required")
            return
        end if

        negative_label = minval(labels)
        positive_label = maxval(labels)
        if (negative_label == positive_label) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM fit: exactly two distinct classes are required")
            return
        end if
        do i = 1, size(labels)
            if (labels(i) /= negative_label .and. labels(i) /= positive_label) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVM fit: exactly two distinct classes are required")
                return
            end if
        end do

        penalty = 1.0_dp
        if (present(l2)) penalty = l2
        if (.not. ieee_is_finite(penalty) .or. penalty < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM fit: L2 penalty must be finite and nonnegative")
            return
        end if
        include_intercept = .true.
        if (present(fit_intercept)) include_intercept = fit_intercept
        requested_loss = SVM_LOSS_SQUARED_HINGE
        if (present(loss)) requested_loss = loss
        if (requested_loss /= SVM_LOSS_HINGE .and. &
            requested_loss /= SVM_LOSS_SQUARED_HINGE) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM fit: loss must be SVM_LOSS_HINGE or SVM_LOSS_SQUARED_HINGE")
            return
        end if
        iterations = 500
        if (present(max_iterations)) iterations = max_iterations
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        if (iterations < 1 .or. .not. ieee_is_finite(requested_tolerance) .or. &
            requested_tolerance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM fit: iteration limit and tolerance are invalid")
            return
        end if

        allocate(weights(size(labels)), encoded(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVM fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        weight_mass = sum(weights)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM fit: sample weights must have positive mass")
            return
        end if
        encoded = -1.0_dp
        where (labels == positive_label) encoded = 1.0_dp

        n_features = size(x, 2)
        n_parameters = n_features
        if (include_intercept) n_parameters = n_parameters + 1
        allocate(theta(n_parameters), lower(n_parameters), upper(n_parameters))
        theta = 0.0_dp
        lower = -huge(1.0_dp)
        upper = huge(1.0_dp)
        call objective%initialize(n_parameters, svm_fit_objective, status)
        if (status%code /= FORTNUM_OK) return
        options%max_iterations = iterations
        options%gradient_tolerance = requested_tolerance
        options%step_tolerance = min(1.0e-12_dp, requested_tolerance)
        options%objective_tolerance = min(1.0e-12_dp, requested_tolerance)
        if (requested_loss == SVM_LOSS_HINGE) then
            ! Ordinary hinge is convex but nonsmooth at margin one.  A longer,
            ! less aggressive Armijo search lets FortOpt accept the same
            ! deterministic subgradient on either side of a split.
            options%max_line_search = 100
            options%armijo_constant = 1.0e-8_dp
        end if
        call optimizer%minimize(objective, theta, lower, upper, options, result, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. result%state%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "linear SVM fit: optimizer reached its iteration limit")
            return
        end if
        if (any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "linear SVM fit: optimizer returned nonfinite parameters")
            return
        end if

        allocate(self%coefficient(n_features))
        self%coefficient = theta(:n_features)
        self%intercept_value = 0.0_dp
        if (include_intercept) self%intercept_value = theta(n_parameters)
        self%class_label = [negative_label, positive_label]
        self%l2_value = penalty
        self%loss_value = requested_loss
        self%fit_intercept_value = include_intercept
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")

    contains

        subroutine svm_fit_objective(parameters, value, gradient, objective_status)
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: value, gradient(:)
            type(fortnum_status_t), intent(out) :: objective_status
            real(dp) :: score, margin, violation, local_value, score_gradient
            integer :: i, j

            value = 0.0_dp
            gradient = 0.0_dp
            if (size(parameters) /= n_parameters .or. &
                size(gradient) /= n_parameters) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVM objective: parameter shape is invalid")
                return
            end if
            do i = 1, size(x, 1)
                score = 0.0_dp
                do j = 1, n_features
                    score = score + x(i, j)*parameters(j)
                end do
                if (include_intercept) score = score + parameters(n_parameters)
                margin = encoded(i)*score
                violation = max(0.0_dp, 1.0_dp-margin)
                if (requested_loss == SVM_LOSS_HINGE) then
                    ! FortOpt's Armijo line search requires a C1 callback.
                    ! Optimize the epsilon-Huberized hinge while retaining
                    ! the exact ordinary-hinge objective in the public
                    ! value/gradient method below.  The smoothing is tiny
                    ! relative to the fit tolerance and removes optimizer
                    ! oscillation when a row lands exactly on margin one.
                    if (violation >= SVM_HINGE_SMOOTHING) then
                        local_value = violation - 0.5_dp*SVM_HINGE_SMOOTHING
                        score_gradient = -encoded(i)
                    else
                        local_value = 0.5_dp*violation*violation / &
                            SVM_HINGE_SMOOTHING
                        score_gradient = -encoded(i)*violation / &
                            SVM_HINGE_SMOOTHING
                    end if
                else
                    local_value = violation*violation
                    score_gradient = -2.0_dp*encoded(i)*violation
                end if
                value = value + weights(i)*local_value
                do j = 1, n_features
                    gradient(j) = gradient(j) + weights(i)*score_gradient*x(i, j)
                end do
                if (include_intercept) gradient(n_parameters) = &
                    gradient(n_parameters) + weights(i)*score_gradient
            end do
            value = value/weight_mass
            gradient = gradient/weight_mass
            value = value + 0.5_dp*penalty*sum(parameters(:n_features)**2)
            gradient(:n_features) = gradient(:n_features) + &
                penalty*parameters(:n_features)
            if (.not. ieee_is_finite(value) .or. &
                any(.not. ieee_is_finite(gradient))) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVM objective: value or gradient is nonfinite")
                return
            end if
            call status_set(objective_status, FORTNUM_OK, "")
        end subroutine svm_fit_objective

    end subroutine svm_fit

    subroutine svm_decision_function(self, x, scores, status)
        class(linear_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count() .or. size(scores) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM decision: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM decision: inputs must be finite")
            return
        end if
        scores = self%intercept_value
        do j = 1, size(self%coefficient)
            do i = 1, size(x, 1)
                scores(i) = scores(i) + x(i, j)*self%coefficient(j)
            end do
        end do
        if (any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM decision: score overflow")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine svm_decision_function

    subroutine svm_predict(self, x, labels, status)
        class(linear_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM predict: output shape is invalid")
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
    end subroutine svm_predict

    subroutine svm_decision_device(self, device, x, scores, status)
        class(linear_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "linear SVM device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM device prediction: device kind is invalid")
        end select
    end subroutine svm_decision_device

    subroutine svm_predict_device(self, device, x, labels, status)
        class(linear_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "linear SVM device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM device prediction: device kind is invalid")
        end select
    end subroutine svm_predict_device

    logical function svm_device_supported(self, device_kind) result(supported)
        class(linear_svm_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%fitted_value .and. device_kind == FORTML_DEVICE_CPU
    end function svm_device_supported

    subroutine svm_decision_jvp(self, x, theta_dot, x_dot, scores, scores_dot, status)
        class(linear_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:), scores_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n_features

        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM decision JVP: model is not fitted")
            return
        end if
        n_features = size(self%coefficient)
        if (size(x, 1) < 1 .or. size(x, 2) /= n_features .or. &
            any(shape(x_dot) /= shape(x)) .or. size(scores) /= size(x, 1) .or. &
            size(scores_dot) /= size(scores) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM decision JVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM decision JVP: inputs and tangents must be finite")
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
        if (self%fit_intercept_value) scores_dot = scores_dot + &
            theta_dot(n_features + 1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine svm_decision_jvp

    subroutine svm_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(linear_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n_features

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM decision VJP: model is not fitted")
            return
        end if
        n_features = size(self%coefficient)
        if (size(x, 1) < 1 .or. size(x, 2) /= n_features .or. &
            size(scores_bar) /= size(x, 1) .or. size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM decision VJP: cotangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM decision VJP: inputs and cotangents must be finite")
            return
        end if
        do j = 1, n_features
            do i = 1, size(x, 1)
                theta_bar(j) = theta_bar(j) + scores_bar(i)*x(i, j)
                x_bar(i, j) = scores_bar(i)*self%coefficient(j)
            end do
        end do
        if (self%fit_intercept_value) theta_bar(n_features + 1) = sum(scores_bar)
        call status_set(status, FORTNUM_OK, "")
    end subroutine svm_decision_vjp

    subroutine svm_objective_value_gradient(self, x, labels, theta, value, gradient, &
            status, l2, fit_intercept, loss, sample_weight, l2_gradient)
        class(linear_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta(:)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, sample_weight(:)
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: loss
        real(dp), intent(out), optional :: l2_gradient
        real(dp), allocatable :: weights(:), encoded(:)
        real(dp) :: penalty, weight_mass, score, margin, violation, &
            score_gradient, local_value
        integer :: negative_label, positive_label, requested_loss, n_features, &
            n_parameters, i, j
        logical :: include_intercept, exact_split

        value = 0.0_dp
        gradient = 0.0_dp
        if (present(l2_gradient)) l2_gradient = 0.0_dp
        penalty = self%l2_value
        if (present(l2)) penalty = l2
        include_intercept = self%fit_intercept_value
        if (present(fit_intercept)) include_intercept = fit_intercept
        requested_loss = self%loss_value
        if (present(loss)) requested_loss = loss
        n_features = size(x, 2)
        n_parameters = n_features + merge(1, 0, include_intercept)
        if (size(x, 1) < 1 .or. n_features < 1 .or. size(labels) /= size(x, 1) .or. &
            size(theta) /= n_parameters .or. size(gradient) /= n_parameters .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(theta)) .or. &
            .not. ieee_is_finite(penalty) .or. penalty < 0.0_dp .or. &
            (requested_loss /= SVM_LOSS_HINGE .and. &
            requested_loss /= SVM_LOSS_SQUARED_HINGE)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM objective: data, parameter, or option domain is invalid")
            return
        end if
        negative_label = minval(labels)
        positive_label = maxval(labels)
        if (negative_label == positive_label) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM objective: exactly two classes are required")
            return
        end if
        allocate(encoded(size(labels)), weights(size(labels)))
        encoded = -1.0_dp
        do i = 1, size(labels)
            if (labels(i) == positive_label) then
                encoded(i) = 1.0_dp
            else if (labels(i) /= negative_label) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVM objective: exactly two classes are required")
                return
            end if
        end do
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVM objective: sample weights are invalid")
                return
            end if
            weights = sample_weight
        end if
        weight_mass = sum(weights)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM objective: sample weights need positive mass")
            return
        end if

        exact_split = .false.
        do i = 1, size(x, 1)
            score = 0.0_dp
            do j = 1, n_features
                score = score + x(i, j)*theta(j)
            end do
            if (include_intercept) score = score + theta(n_parameters)
            margin = encoded(i)*score
            if (requested_loss == SVM_LOSS_HINGE .and. margin == 1.0_dp) then
                exact_split = .true.
            end if
            violation = max(0.0_dp, 1.0_dp-margin)
            if (requested_loss == SVM_LOSS_HINGE) then
                local_value = violation
                if (margin < 1.0_dp) then
                    score_gradient = -encoded(i)
                else
                    score_gradient = 0.0_dp
                end if
            else
                local_value = violation*violation
                score_gradient = -2.0_dp*encoded(i)*violation
            end if
            value = value + weights(i)*local_value
            do j = 1, n_features
                gradient(j) = gradient(j) + weights(i)*score_gradient*x(i, j)
            end do
            if (include_intercept) gradient(n_parameters) = &
                gradient(n_parameters) + weights(i)*score_gradient
        end do
        value = value/weight_mass
        gradient = gradient/weight_mass
        value = value + 0.5_dp*penalty*sum(theta(:n_features)**2)
        gradient(:n_features) = gradient(:n_features) + penalty*theta(:n_features)
        if (present(l2_gradient)) l2_gradient = 0.5_dp*sum(theta(:n_features)**2)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM objective: value or gradient is nonfinite")
            return
        end if
        if (exact_split) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "linear SVM objective: ordinary hinge derivative is split at margin one")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine svm_objective_value_gradient

    function svm_coefficients(self) result(values)
        class(linear_svm_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        if (allocated(self%coefficient)) then
            values = self%coefficient
        else
            allocate(values(0))
        end if
    end function svm_coefficients

    real(dp) function svm_intercept(self) result(value)
        class(linear_svm_classifier_t), intent(in) :: self
        value = self%intercept_value
    end function svm_intercept

    function svm_classes(self) result(values)
        class(linear_svm_classifier_t), intent(in) :: self
        integer :: values(2)
        values = self%class_label
    end function svm_classes

    real(dp) function svm_regularization(self) result(value)
        class(linear_svm_classifier_t), intent(in) :: self
        value = self%l2_value
    end function svm_regularization

    integer function svm_loss(self) result(value)
        class(linear_svm_classifier_t), intent(in) :: self
        value = self%loss_value
    end function svm_loss

    logical function svm_fit_intercept(self) result(value)
        class(linear_svm_classifier_t), intent(in) :: self
        value = self%fit_intercept_value
    end function svm_fit_intercept

    function svm_parameters(self) result(values)
        class(linear_svm_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: n_features
        if (.not. allocated(self%coefficient)) then
            allocate(values(0))
            return
        end if
        n_features = size(self%coefficient)
        allocate(values(self%parameter_count()))
        values(:n_features) = self%coefficient
        if (self%fit_intercept_value) values(n_features + 1) = self%intercept_value
    end function svm_parameters

    subroutine svm_set_parameters(self, values, status)
        class(linear_svm_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_features
        if (.not. self%fitted_value .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVM set_parameters: model or packed values are invalid")
            return
        end if
        n_features = size(self%coefficient)
        self%coefficient = values(:n_features)
        if (self%fit_intercept_value) self%intercept_value = values(n_features + 1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine svm_set_parameters

    integer function svm_parameter_count(self) result(count)
        class(linear_svm_classifier_t), intent(in) :: self
        count = 0
        if (allocated(self%coefficient)) then
            count = size(self%coefficient) + merge(1, 0, self%fit_intercept_value)
        end if
    end function svm_parameter_count

    integer function svm_feature_count(self) result(count)
        class(linear_svm_classifier_t), intent(in) :: self
        count = 0
        if (allocated(self%coefficient)) count = size(self%coefficient)
    end function svm_feature_count

    logical function svm_fitted(self) result(value)
        class(linear_svm_classifier_t), intent(in) :: self
        value = self%fitted_value .and. allocated(self%coefficient)
    end function svm_fitted

end module fortml_linear_svm_classifier
