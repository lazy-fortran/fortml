module fortml_linear_svr
    !! Weighted dense linear epsilon-insensitive regression.
    !!
    !! The fitted state is a primal affine predictor with feature-only L2
    !! regularization.  The default squared epsilon-insensitive loss is
    !!
    !!   sum_i w_i max(0, abs(y_i-f_i)-epsilon)**2 / sum_i w_i
    !!       + 0.5*l2*||beta||**2.
    !!
    !! The ordinary epsilon-insensitive loss is available as
    !! SVR_LOSS_EPSILON.  Its exact public objective product refuses at the
    !! two residual kinks rather than inventing a derivative.  Fit uses a tiny
    !! C1 continuation around those kinks so FortOpt's Armijo line search has
    !! a deterministic callback.  Prediction products hold the fitted affine
    !! state fixed.  CUDA dispatch is an explicit refusal until a resident
    !! SVR kernel is linked; no host fallback is hidden.
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

    integer, parameter, public :: SVR_LOSS_EPSILON = 1
    integer, parameter, public :: SVR_LOSS_SQUARED_EPSILON = 2
    real(dp), parameter :: SVR_EPSILON_SMOOTHING = 1.0e-2_dp

    type, public :: linear_svr_regression_t
        private
        real(dp), allocatable :: coefficient(:)
        real(dp) :: intercept_value = 0.0_dp
        real(dp) :: l2_value = 1.0_dp
        real(dp) :: epsilon_value = 0.1_dp
        integer :: loss_value = SVR_LOSS_SQUARED_EPSILON
        logical :: fit_intercept_value = .true.
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => svr_fit
        procedure, public :: predict => svr_predict
        procedure, public :: decision_function => svr_predict
        procedure, public :: predict_device => svr_predict_device
        procedure, public :: decision_function_device => svr_predict_device
        procedure, public :: device_supported => svr_device_supported
        procedure, public :: predict_jvp => svr_predict_jvp
        procedure, public :: predict_vjp => svr_predict_vjp
        procedure, public :: jvp => svr_predict_jvp
        procedure, public :: vjp => svr_predict_vjp
        procedure, public :: objective_value_gradient => &
            svr_objective_value_gradient
        procedure, public :: coefficients => svr_coefficients
        procedure, public :: intercept => svr_intercept
        procedure, public :: regularization => svr_regularization
        procedure, public :: epsilon => svr_epsilon
        procedure, public :: loss => svr_loss
        procedure, public :: fit_intercept => svr_fit_intercept
        procedure, public :: parameters => svr_parameters
        procedure, public :: set_parameters => svr_set_parameters
        procedure, public :: parameter_count => svr_parameter_count
        procedure, public :: feature_count => svr_feature_count
        procedure, public :: fitted => svr_fitted
    end type linear_svr_regression_t

    public :: svr_fit
    public :: svr_predict
    public :: svr_predict_device

contains

    subroutine svr_fit(self, x, targets, status, l2, epsilon, fit_intercept, &
            loss, max_iterations, tolerance, sample_weight)
        class(linear_svr_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), targets(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, epsilon, tolerance, sample_weight(:)
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: loss, max_iterations
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: options
        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: theta(:), lower(:), upper(:), weights(:)
        real(dp) :: penalty, requested_epsilon, requested_tolerance, weight_mass
        integer :: requested_loss, iterations, n_features, n_parameters
        logical :: include_intercept

        self%fitted_value = .false.
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(targets) /= size(x, 1) .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR fit: finite inputs and matching dimensions are required")
            return
        end if
        penalty = 1.0_dp
        if (present(l2)) penalty = l2
        requested_epsilon = 0.1_dp
        if (present(epsilon)) requested_epsilon = epsilon
        if (.not. ieee_is_finite(penalty) .or. penalty < 0.0_dp .or. &
            .not. ieee_is_finite(requested_epsilon) .or. requested_epsilon < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR fit: L2 must be nonnegative and epsilon must be finite")
            return
        end if
        include_intercept = .true.
        if (present(fit_intercept)) include_intercept = fit_intercept
        requested_loss = SVR_LOSS_SQUARED_EPSILON
        if (present(loss)) requested_loss = loss
        if (requested_loss /= SVR_LOSS_EPSILON .and. &
            requested_loss /= SVR_LOSS_SQUARED_EPSILON) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR fit: unsupported epsilon-insensitive loss")
            return
        end if
        iterations = 500
        if (present(max_iterations)) iterations = max_iterations
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        if (iterations < 1 .or. .not. ieee_is_finite(requested_tolerance) .or. &
            requested_tolerance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR fit: iteration limit and tolerance are invalid")
            return
        end if
        allocate(weights(size(targets)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(targets) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVR fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        weight_mass = sum(weights)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR fit: sample weights must have positive mass")
            return
        end if

        n_features = size(x, 2)
        n_parameters = n_features + merge(1, 0, include_intercept)
        allocate(theta(n_parameters), lower(n_parameters), upper(n_parameters))
        theta = 0.0_dp
        lower = -huge(1.0_dp)
        upper = huge(1.0_dp)
        call objective%initialize(n_parameters, svr_fit_objective, status)
        if (status%code /= FORTNUM_OK) return
        options%max_iterations = iterations
        options%gradient_tolerance = requested_tolerance
        options%step_tolerance = min(1.0e-12_dp, requested_tolerance)
        options%objective_tolerance = min(1.0e-12_dp, requested_tolerance)
        if (requested_loss == SVR_LOSS_EPSILON) then
            options%max_line_search = 100
            options%armijo_constant = 1.0e-8_dp
        end if
        call optimizer%minimize(objective, theta, lower, upper, options, result, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. result%state%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "linear SVR fit: optimizer reached its iteration limit")
            return
        end if
        if (any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "linear SVR fit: optimizer returned nonfinite parameters")
            return
        end if

        allocate(self%coefficient(n_features))
        self%coefficient = theta(:n_features)
        self%intercept_value = 0.0_dp
        if (include_intercept) self%intercept_value = theta(n_parameters)
        self%l2_value = penalty
        self%epsilon_value = requested_epsilon
        self%loss_value = requested_loss
        self%fit_intercept_value = include_intercept
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")

    contains

        subroutine svr_fit_objective(parameters, value, gradient, objective_status)
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: value, gradient(:)
            type(fortnum_status_t), intent(out) :: objective_status
            real(dp) :: prediction, residual, absolute_residual, excess
            real(dp) :: local_value, residual_gradient
            integer :: i, j

            value = 0.0_dp
            gradient = 0.0_dp
            if (size(parameters) /= n_parameters .or. size(gradient) /= n_parameters) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVR objective: parameter shape is invalid")
                return
            end if
            do i = 1, size(x, 1)
                prediction = 0.0_dp
                do j = 1, n_features
                    prediction = prediction + x(i, j)*parameters(j)
                end do
                if (include_intercept) prediction = prediction + parameters(n_parameters)
                residual = prediction - targets(i)
                absolute_residual = abs(residual)
                excess = absolute_residual - requested_epsilon
                if (requested_loss == SVR_LOSS_EPSILON) then
                    if (excess >= SVR_EPSILON_SMOOTHING) then
                        local_value = excess - 0.5_dp*SVR_EPSILON_SMOOTHING
                        residual_gradient = sign(1.0_dp, residual)
                    else if (excess > 0.0_dp) then
                        local_value = 0.5_dp*excess*excess / SVR_EPSILON_SMOOTHING
                        residual_gradient = excess / SVR_EPSILON_SMOOTHING * &
                            sign(1.0_dp, residual)
                    else
                        local_value = 0.0_dp
                        residual_gradient = 0.0_dp
                    end if
                else
                    local_value = max(0.0_dp, excess)**2
                    residual_gradient = 2.0_dp*max(0.0_dp, excess)* &
                        sign(1.0_dp, residual)
                end if
                value = value + weights(i)*local_value
                do j = 1, n_features
                    gradient(j) = gradient(j) + weights(i)*residual_gradient*x(i, j)
                end do
                if (include_intercept) gradient(n_parameters) = &
                    gradient(n_parameters) + weights(i)*residual_gradient
            end do
            value = value/weight_mass
            gradient = gradient/weight_mass
            value = value + 0.5_dp*penalty*sum(parameters(:n_features)**2)
            gradient(:n_features) = gradient(:n_features) + penalty*parameters(:n_features)
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVR objective: value or gradient is nonfinite")
                return
            end if
            call status_set(objective_status, FORTNUM_OK, "")
        end subroutine svr_fit_objective

    end subroutine svr_fit

    subroutine svr_predict(self, x, targets, status)
        class(linear_svr_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: targets(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count() .or. size(targets) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR predict: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR predict: inputs must be finite")
            return
        end if
        targets = self%intercept_value
        do j = 1, size(self%coefficient)
            do i = 1, size(x, 1)
                targets(i) = targets(i) + x(i, j)*self%coefficient(j)
            end do
        end do
        if (any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR predict: prediction overflow")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine svr_predict

    subroutine svr_predict_device(self, device, x, targets, status)
        class(linear_svr_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: targets(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, targets, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "linear SVR device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR device prediction: device kind is invalid")
        end select
    end subroutine svr_predict_device

    logical function svr_device_supported(self, device_kind) result(supported)
        class(linear_svr_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind
        supported = self%fitted_value .and. device_kind == FORTML_DEVICE_CPU
    end function svr_device_supported

    subroutine svr_predict_jvp(self, x, theta_dot, x_dot, targets, targets_dot, status)
        class(linear_svr_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: targets(:), targets_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n_features

        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR JVP: model is not fitted")
            return
        end if
        n_features = size(self%coefficient)
        if (size(x, 1) < 1 .or. size(x, 2) /= n_features .or. &
            any(shape(x_dot) /= shape(x)) .or. size(targets) /= size(x, 1) .or. &
            size(targets_dot) /= size(targets) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR JVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR JVP: inputs and tangents must be finite")
            return
        end if
        call self%predict(x, targets, status)
        if (status%code /= FORTNUM_OK) return
        targets_dot = 0.0_dp
        do j = 1, n_features
            do i = 1, size(x, 1)
                targets_dot(i) = targets_dot(i) + self%coefficient(j)*x_dot(i, j) + &
                    theta_dot(j)*x(i, j)
            end do
        end do
        if (self%fit_intercept_value) targets_dot = targets_dot + &
            theta_dot(n_features + 1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine svr_predict_jvp

    subroutine svr_predict_vjp(self, x, targets_bar, theta_bar, x_bar, status)
        class(linear_svr_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), targets_bar(:)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n_features

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR VJP: model is not fitted")
            return
        end if
        n_features = size(self%coefficient)
        if (size(x, 1) < 1 .or. size(x, 2) /= n_features .or. &
            size(targets_bar) /= size(x, 1) .or. size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR VJP: cotangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR VJP: inputs and cotangents must be finite")
            return
        end if
        do j = 1, n_features
            do i = 1, size(x, 1)
                theta_bar(j) = theta_bar(j) + targets_bar(i)*x(i, j)
                x_bar(i, j) = targets_bar(i)*self%coefficient(j)
            end do
        end do
        if (self%fit_intercept_value) theta_bar(n_features + 1) = sum(targets_bar)
        call status_set(status, FORTNUM_OK, "")
    end subroutine svr_predict_vjp

    subroutine svr_objective_value_gradient(self, x, targets, theta, value, gradient, &
            status, l2, epsilon, fit_intercept, loss, sample_weight, l2_gradient, &
            epsilon_gradient)
        class(linear_svr_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), targets(:), theta(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, epsilon, sample_weight(:)
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: loss
        real(dp), intent(out), optional :: l2_gradient, epsilon_gradient
        real(dp), allocatable :: weights(:)
        real(dp) :: penalty, requested_epsilon, weight_mass, prediction, residual
        real(dp) :: absolute_residual, excess, local_value, residual_gradient
        integer :: n_features, n_parameters, i, j, loss_code
        logical :: include_intercept, exact_split

        value = 0.0_dp
        gradient = 0.0_dp
        if (present(l2_gradient)) l2_gradient = 0.0_dp
        if (present(epsilon_gradient)) epsilon_gradient = 0.0_dp
        penalty = self%l2_value
        if (present(l2)) penalty = l2
        requested_epsilon = self%epsilon_value
        if (present(epsilon)) requested_epsilon = epsilon
        include_intercept = self%fit_intercept_value
        if (present(fit_intercept)) include_intercept = fit_intercept
        loss_code = self%loss_value
        if (present(loss)) loss_code = loss
        n_features = size(x, 2)
        n_parameters = n_features + merge(1, 0, include_intercept)
        if (size(x, 1) < 1 .or. n_features < 1 .or. size(targets) /= size(x, 1) .or. &
            size(theta) /= n_parameters .or. size(gradient) /= n_parameters .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets)) .or. &
            any(.not. ieee_is_finite(theta)) .or. .not. ieee_is_finite(penalty) .or. &
            penalty < 0.0_dp .or. .not. ieee_is_finite(requested_epsilon) .or. &
            requested_epsilon < 0.0_dp .or. (loss_code /= SVR_LOSS_EPSILON .and. &
            loss_code /= SVR_LOSS_SQUARED_EPSILON)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR objective: data, parameter, or option domain is invalid")
            return
        end if
        allocate(weights(size(targets)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(targets) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SVR objective: sample weights are invalid")
                return
            end if
            weights = sample_weight
        end if
        weight_mass = sum(weights)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR objective: sample weights need positive mass")
            return
        end if

        exact_split = .false.
        do i = 1, size(x, 1)
            prediction = 0.0_dp
            do j = 1, n_features
                prediction = prediction + x(i, j)*theta(j)
            end do
            if (include_intercept) prediction = prediction + theta(n_parameters)
            residual = prediction - targets(i)
            absolute_residual = abs(residual)
            excess = absolute_residual - requested_epsilon
            if (loss_code == SVR_LOSS_EPSILON) then
                if (excess == 0.0_dp) exact_split = .true.
                local_value = max(0.0_dp, excess)
                if (excess > 0.0_dp) then
                    residual_gradient = sign(1.0_dp, residual)
                else
                    residual_gradient = 0.0_dp
                end if
            else
                local_value = max(0.0_dp, excess)**2
                residual_gradient = 2.0_dp*max(0.0_dp, excess)* &
                    sign(1.0_dp, residual)
            end if
            value = value + weights(i)*local_value
            do j = 1, n_features
                gradient(j) = gradient(j) + weights(i)*residual_gradient*x(i, j)
            end do
            if (include_intercept) gradient(n_parameters) = &
                gradient(n_parameters) + weights(i)*residual_gradient
            if (present(epsilon_gradient)) then
                if (loss_code == SVR_LOSS_EPSILON) then
                    if (excess > 0.0_dp) epsilon_gradient = epsilon_gradient - weights(i)
                else
                    epsilon_gradient = epsilon_gradient - &
                        2.0_dp*weights(i)*max(0.0_dp, excess)
                end if
            end if
        end do
        value = value/weight_mass
        gradient = gradient/weight_mass
        if (present(epsilon_gradient)) epsilon_gradient = epsilon_gradient/weight_mass
        value = value + 0.5_dp*penalty*sum(theta(:n_features)**2)
        gradient(:n_features) = gradient(:n_features) + penalty*theta(:n_features)
        if (present(l2_gradient)) l2_gradient = 0.5_dp*sum(theta(:n_features)**2)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR objective: value or gradient is nonfinite")
            return
        end if
        if (exact_split) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "linear SVR objective: epsilon-insensitive derivative is split at a kink")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine svr_objective_value_gradient

    function svr_coefficients(self) result(values)
        class(linear_svr_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        if (allocated(self%coefficient)) then
            values = self%coefficient
        else
            allocate(values(0))
        end if
    end function svr_coefficients

    real(dp) function svr_intercept(self) result(value)
        class(linear_svr_regression_t), intent(in) :: self
        value = self%intercept_value
    end function svr_intercept

    real(dp) function svr_regularization(self) result(value)
        class(linear_svr_regression_t), intent(in) :: self
        value = self%l2_value
    end function svr_regularization

    real(dp) function svr_epsilon(self) result(value)
        class(linear_svr_regression_t), intent(in) :: self
        value = self%epsilon_value
    end function svr_epsilon

    integer function svr_loss(self) result(value)
        class(linear_svr_regression_t), intent(in) :: self
        value = self%loss_value
    end function svr_loss

    logical function svr_fit_intercept(self) result(value)
        class(linear_svr_regression_t), intent(in) :: self
        value = self%fit_intercept_value
    end function svr_fit_intercept

    function svr_parameters(self) result(values)
        class(linear_svr_regression_t), intent(in) :: self
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
    end function svr_parameters

    subroutine svr_set_parameters(self, values, status)
        class(linear_svr_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_features
        if (.not. self%fitted_value .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SVR set_parameters: model or packed values are invalid")
            return
        end if
        n_features = size(self%coefficient)
        self%coefficient = values(:n_features)
        if (self%fit_intercept_value) self%intercept_value = values(n_features + 1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine svr_set_parameters

    integer function svr_parameter_count(self) result(count)
        class(linear_svr_regression_t), intent(in) :: self
        count = 0
        if (allocated(self%coefficient)) then
            count = size(self%coefficient) + merge(1, 0, self%fit_intercept_value)
        end if
    end function svr_parameter_count

    integer function svr_feature_count(self) result(count)
        class(linear_svr_regression_t), intent(in) :: self
        count = 0
        if (allocated(self%coefficient)) count = size(self%coefficient)
    end function svr_feature_count

    logical function svr_fitted(self) result(value)
        class(linear_svr_regression_t), intent(in) :: self
        value = self%fitted_value .and. allocated(self%coefficient)
    end function svr_fitted

end module fortml_linear_svr
