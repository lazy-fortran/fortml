module fortml_glm_regression
    !! Weighted generalized linear regression with stable log-link products.
    !!
    !! The estimator currently provides the canonical positive-response
    !! Poisson and Gamma families.  Both use a log link, so predictions are
    !! positive and their parameter/input JVP and VJP products are analytic.
    !! Fit minimizes the weighted negative log likelihood with an optional
    !! L2 penalty (the intercept is never penalized) through FortOpt
    !! L-BFGS-B.  The fit itself is a discrete optimizer boundary; products
    !! hold the fitted coefficients fixed.
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

    integer, parameter, public :: GLM_FAMILY_POISSON = 1
    integer, parameter, public :: GLM_FAMILY_GAMMA = 2
    integer, parameter, public :: GLM_LINK_LOG = 1

    real(dp), parameter :: GLM_DEFAULT_BOUND = 30.0_dp

    type, private :: glm_objective_context_t
        real(dp), allocatable :: x(:, :), target(:), weight(:)
        real(dp) :: weight_mass = 0.0_dp
        real(dp) :: alpha = 0.0_dp
        real(dp) :: dispersion = 1.0_dp
        integer :: family = GLM_FAMILY_POISSON
        logical :: fit_intercept = .true.
    end type glm_objective_context_t

    !> Weighted positive-response generalized linear regression.
    type, public :: glm_regression_t
        private
        real(dp), allocatable :: coefficient(:, :)
        integer :: feature_count_value = 0
        integer :: output_count_value = 0
        integer :: family_value = GLM_FAMILY_POISSON
        integer :: link_value = GLM_LINK_LOG
        real(dp) :: alpha_value = 0.0_dp
        real(dp) :: dispersion_value = 1.0_dp
        logical :: fit_intercept_value = .true.
        integer :: max_iterations_value = 500
        real(dp) :: tolerance_value = 1.0e-8_dp
        real(dp) :: lower_bound_value = -GLM_DEFAULT_BOUND
        real(dp) :: upper_bound_value = GLM_DEFAULT_BOUND
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit_matrix => glm_fit_matrix
        procedure, public :: fit_vector => glm_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: predict_matrix => glm_predict_matrix
        procedure, public :: predict_vector => glm_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_device => glm_predict_device
        procedure, public :: device_supported => glm_device_supported
        procedure, public :: predict_jvp => glm_predict_jvp
        procedure, public :: predict_vjp => glm_predict_vjp
        procedure, public :: jvp => glm_predict_jvp
        procedure, public :: vjp => glm_predict_vjp
        procedure, public :: objective_value_gradient => glm_objective_value_gradient
        procedure, public :: coefficients => glm_coefficients
        procedure, public :: parameters => glm_parameters
        procedure, public :: set_parameters => glm_set_parameters
        procedure, public :: parameter_count => glm_parameter_count
        procedure, public :: feature_count => glm_feature_count
        procedure, public :: output_count => glm_output_count
        procedure, public :: family => glm_family
        procedure, public :: link => glm_link
        procedure, public :: regularization => glm_regularization
        procedure, public :: dispersion => glm_dispersion
        procedure, public :: fit_intercept => glm_fit_intercept
        procedure, public :: max_iterations => glm_max_iterations
        procedure, public :: tolerance => glm_tolerance
        procedure, public :: lower_bound => glm_lower_bound
        procedure, public :: upper_bound => glm_upper_bound
        procedure, public :: fitted => glm_fitted
    end type glm_regression_t

contains

    subroutine glm_fit_matrix(self, x, y, status, family, alpha, fit_intercept, &
            sample_weight, max_iterations, tolerance, dispersion, lower_bound, &
            upper_bound)
        class(glm_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: family, max_iterations
        real(dp), intent(in), optional :: alpha, sample_weight(:), tolerance, &
            dispersion, lower_bound, upper_bound
        logical, intent(in), optional :: fit_intercept

        type(glm_objective_context_t), target :: context
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: options
        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: theta(:), lower(:), upper(:), weights(:)
        real(dp) :: penalty, requested_tolerance, requested_dispersion
        real(dp) :: requested_lower, requested_upper, mass
        integer :: n_samples, n_features, n_outputs, n_parameters
        integer :: iterations, k
        logical :: include_intercept

        self%fitted_value = .false.
        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_outputs = size(y, 2)
        if (n_samples < 1 .or. n_features < 1 .or. n_outputs < 1 .or. &
                size(y, 1) /= n_samples .or. any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm fit: inputs and targets must be finite and dimensions must match")
            return
        end if

        context%family = GLM_FAMILY_POISSON
        if (present(family)) context%family = family
        if (context%family /= GLM_FAMILY_POISSON .and. &
                context%family /= GLM_FAMILY_GAMMA) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm fit: family must be GLM_FAMILY_POISSON or GLM_FAMILY_GAMMA")
            return
        end if
        if (context%family == GLM_FAMILY_POISSON) then
            if (any(y < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "glm fit: Poisson targets must be nonnegative")
                return
            end if
        else
            if (any(y <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "glm fit: Gamma targets must be strictly positive")
                return
            end if
        end if

        penalty = 0.0_dp
        if (present(alpha)) penalty = alpha
        requested_dispersion = 1.0_dp
        if (present(dispersion)) requested_dispersion = dispersion
        if (.not. ieee_is_finite(penalty) .or. penalty < 0.0_dp .or. &
                .not. ieee_is_finite(requested_dispersion) .or. &
                requested_dispersion <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm fit: alpha must be nonnegative and dispersion positive")
            return
        end if

        include_intercept = .true.
        if (present(fit_intercept)) include_intercept = fit_intercept
        iterations = 500
        if (present(max_iterations)) iterations = max_iterations
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        if (iterations < 1 .or. .not. ieee_is_finite(requested_tolerance) .or. &
                requested_tolerance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm fit: max_iterations must be positive and tolerance finite")
            return
        end if
        requested_lower = -GLM_DEFAULT_BOUND
        if (present(lower_bound)) requested_lower = lower_bound
        requested_upper = GLM_DEFAULT_BOUND
        if (present(upper_bound)) requested_upper = upper_bound
        if (.not. ieee_is_finite(requested_lower) .or. &
                .not. ieee_is_finite(requested_upper) .or. &
                requested_lower >= requested_upper) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm fit: finite lower_bound must be less than upper_bound")
            return
        end if

        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                    any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "glm fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        mass = sum(weights)
        if (.not. ieee_is_finite(mass) .or. mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm fit: sample weights must have positive mass")
            return
        end if

        n_parameters = n_features
        if (include_intercept) n_parameters = n_parameters + 1
        allocate(theta(n_parameters), lower(n_parameters), upper(n_parameters))
        theta = 0.0_dp
        lower = requested_lower
        upper = requested_upper
        allocate(self%coefficient(n_parameters, n_outputs))

        context%x = x
        context%weight = weights
        context%weight_mass = mass
        context%alpha = penalty
        context%dispersion = requested_dispersion
        context%fit_intercept = include_intercept
        do k = 1, n_outputs
            context%target = y(:, k)
            call objective%initialize_context(n_parameters, context, &
                glm_objective_callback, status)
            if (status%code /= FORTNUM_OK) return
            options%max_iterations = iterations
            options%gradient_tolerance = requested_tolerance
            ! The objective is evaluated in double precision and the bounded
            ! line search can otherwise oscillate below machine-scale changes
            ! in the deviance.  A small absolute step floor lets FortOpt
            ! report the numerically converged state instead of a spurious
            ! Armijo refusal; the requested gradient tolerance is retained.
            options%step_tolerance = max(1.0e-6_dp, requested_tolerance)
            options%objective_tolerance = max(1.0e-12_dp, requested_tolerance*1.0e-2_dp)
            options%armijo_constant = 1.0e-8_dp
            theta = 0.0_dp
            call optimizer%minimize(objective, theta, lower, upper, options, &
                result, status)
            if (status%code /= FORTNUM_OK) return
            if (.not. result%state%converged) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "glm fit: L-BFGS-B reached its iteration limit")
                return
            end if
            if (any(.not. ieee_is_finite(theta))) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "glm fit: optimizer returned nonfinite coefficients")
                return
            end if
            self%coefficient(:, k) = theta
        end do

        self%feature_count_value = n_features
        self%output_count_value = n_outputs
        self%family_value = context%family
        self%link_value = GLM_LINK_LOG
        self%alpha_value = penalty
        self%dispersion_value = requested_dispersion
        self%fit_intercept_value = include_intercept
        self%max_iterations_value = iterations
        self%tolerance_value = requested_tolerance
        self%lower_bound_value = requested_lower
        self%upper_bound_value = requested_upper
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine glm_fit_matrix

    subroutine glm_fit_vector(self, x, y, status, family, alpha, fit_intercept, &
            sample_weight, max_iterations, tolerance, dispersion, lower_bound, &
            upper_bound)
        class(glm_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: family, max_iterations
        real(dp), intent(in), optional :: alpha, sample_weight(:), tolerance, &
            dispersion, lower_bound, upper_bound
        logical, intent(in), optional :: fit_intercept
        real(dp), allocatable :: target(:, :)

        allocate(target(size(y), 1))
        target(:, 1) = y
        call glm_fit_matrix(self, x, target, status, family, alpha, fit_intercept, &
            sample_weight, max_iterations, tolerance, dispersion, lower_bound, &
            upper_bound)
    end subroutine glm_fit_vector

    subroutine glm_predict_matrix(self, x, y, status)
        class(glm_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), eta(:, :)
        integer :: i, k

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
                size(x, 2) /= self%feature_count_value .or. &
                any(shape(y) /= [size(x, 1), self%output_count_value]) .or. &
                any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm predict: model, inputs, or output shape is invalid")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(eta(size(x, 1), self%output_count_value))
        call make_design(x, self%fit_intercept_value, design)
        eta = matmul(design, self%coefficient)
        do k = 1, size(eta, 2)
            do i = 1, size(eta, 1)
                if (.not. ieee_is_finite(eta(i, k)) .or. &
                        eta(i, k) > log(huge(1.0_dp))-2.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "glm predict: linear predictor is outside stable log-link domain")
                    return
                end if
                y(i, k) = exp(eta(i, k))
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine glm_predict_matrix

    subroutine glm_predict_vector(self, x, y, status)
        class(glm_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm predict: output shape is invalid")
            return
        end if
        allocate(values(size(y), 1))
        call glm_predict_matrix(self, x, values, status)
        if (status%code == FORTNUM_OK) y = values(:, 1)
    end subroutine glm_predict_vector

    subroutine glm_predict_device(self, device, x, y, status)
        class(glm_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_matrix(x, y, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "glm device prediction: no resident CUDA log-link kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm device prediction: device kind is invalid")
        end select
    end subroutine glm_predict_device

    subroutine glm_predict_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(glm_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), design_dot(:, :), eta(:, :), &
            eta_dot(:, :), coefficient_dot(:, :)

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
                size(x, 2) /= self%feature_count_value .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                any(shape(y) /= [size(x, 1), self%output_count_value]) .or. &
                any(shape(y_dot) /= shape(y)) .or. &
                size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm JVP: inputs and tangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(design_dot, mold=design)
        allocate(eta(size(x, 1), self%output_count_value))
        allocate(eta_dot, mold=eta)
        allocate(coefficient_dot, mold=self%coefficient)
        call make_design(x, self%fit_intercept_value, design)
        call make_tangent_design(x_dot, self%fit_intercept_value, design_dot)
        coefficient_dot = reshape(theta_dot, shape(coefficient_dot))
        eta = matmul(design, self%coefficient)
        eta_dot = matmul(design_dot, self%coefficient) + &
            matmul(design, coefficient_dot)
        if (any(eta > log(huge(1.0_dp))-2.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm JVP: linear predictor is outside stable log-link domain")
            return
        end if
        y = exp(eta)
        y_dot = y*eta_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine glm_predict_jvp

    subroutine glm_predict_vjp(self, x, u, theta_bar, x_bar, status)
        class(glm_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), eta(:, :), prediction(:, :), &
            cotangent_eta(:, :), coefficient_bar(:, :)
        integer :: j

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
                size(x, 2) /= self%feature_count_value .or. &
                any(shape(u) /= [size(x, 1), self%output_count_value]) .or. &
                size(theta_bar) /= self%parameter_count() .or. &
                any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm VJP: model, cotangent, or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(u))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(eta(size(x, 1), self%output_count_value))
        allocate(prediction, mold=eta)
        allocate(cotangent_eta, mold=eta)
        allocate(coefficient_bar, mold=self%coefficient)
        call make_design(x, self%fit_intercept_value, design)
        eta = matmul(design, self%coefficient)
        if (any(eta > log(huge(1.0_dp))-2.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm VJP: linear predictor is outside stable log-link domain")
            return
        end if
        prediction = exp(eta)
        cotangent_eta = u*prediction
        coefficient_bar = matmul(transpose(design), cotangent_eta)
        theta_bar = reshape(coefficient_bar, [size(theta_bar)])
        x_bar = 0.0_dp
        do j = 1, self%feature_count_value
            x_bar(:, j) = matmul(cotangent_eta, self%coefficient(j+1, :))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine glm_predict_vjp

    subroutine glm_objective_value_gradient(self, x, y, theta, value, gradient, &
            status, family, alpha, fit_intercept, sample_weight, dispersion, &
            alpha_gradient, dispersion_gradient)
        class(glm_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:), theta(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: family
        real(dp), intent(in), optional :: alpha, sample_weight(:), dispersion
        logical, intent(in), optional :: fit_intercept
        real(dp), intent(out), optional :: alpha_gradient, dispersion_gradient
        type(glm_objective_context_t) :: context
        integer :: family_code
        real(dp) :: penalty, requested_dispersion
        logical :: include_intercept

        value = 0.0_dp
        gradient = 0.0_dp
        if (present(alpha_gradient)) alpha_gradient = 0.0_dp
        if (present(dispersion_gradient)) dispersion_gradient = 0.0_dp
        family_code = self%family_value
        if (present(family)) family_code = family
        penalty = self%alpha_value
        if (present(alpha)) penalty = alpha
        requested_dispersion = self%dispersion_value
        if (present(dispersion)) requested_dispersion = dispersion
        include_intercept = self%fit_intercept_value
        if (present(fit_intercept)) include_intercept = fit_intercept
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(y) /= size(x, 1) .or. &
                size(theta) /= size(gradient) .or. &
                any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y)) .or. &
                any(.not. ieee_is_finite(theta)) .or. &
                (include_intercept .and. size(theta) /= size(x, 2)+1) .or. &
                ((.not. include_intercept) .and. size(theta) /= size(x, 2))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm objective: model, data, or parameter shape is invalid")
            return
        end if
        if (family_code /= GLM_FAMILY_POISSON .and. family_code /= GLM_FAMILY_GAMMA) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm objective: unsupported family")
            return
        end if
        if ((family_code == GLM_FAMILY_POISSON .and. any(y < 0.0_dp)) .or. &
                (family_code == GLM_FAMILY_GAMMA .and. any(y <= 0.0_dp)) .or. &
                .not. ieee_is_finite(penalty) .or. penalty < 0.0_dp .or. &
                .not. ieee_is_finite(requested_dispersion) .or. requested_dispersion <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm objective: response or hyperparameter domain is invalid")
            return
        end if
        allocate(context%x, source=x)
        allocate(context%target, source=y)
        allocate(context%weight(size(y)))
        context%weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(y) .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "glm objective: sample weights are invalid")
                return
            end if
            context%weight = sample_weight
        end if
        context%weight_mass = sum(context%weight)
        if (context%weight_mass <= 0.0_dp .or. .not. ieee_is_finite(context%weight_mass)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm objective: sample weights must have positive mass")
            return
        end if
        context%family = family_code
        context%alpha = penalty
        context%dispersion = requested_dispersion
        context%fit_intercept = include_intercept
        call glm_objective_callback(context, theta, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        if (present(alpha_gradient)) then
            if (include_intercept) then
                alpha_gradient = 0.5_dp*sum(theta(2:)**2)
            else
                alpha_gradient = 0.5_dp*sum(theta**2)
            end if
        end if
        if (present(dispersion_gradient) .and. family_code == GLM_FAMILY_GAMMA) then
            if (include_intercept) then
                dispersion_gradient = -(value - 0.5_dp*penalty*sum(theta(2:)**2))/ &
                    requested_dispersion
            else
                dispersion_gradient = -(value - 0.5_dp*penalty*sum(theta**2))/ &
                    requested_dispersion
            end if
        end if
    end subroutine glm_objective_value_gradient

    subroutine glm_objective_callback(raw_context, theta, value, gradient, status)
        class(*), intent(inout) :: raw_context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        select type (context => raw_context)
        type is (glm_objective_context_t)
            call glm_objective_value_gradient_context(context, theta, value, &
                gradient, status)
        class default
            value = 0.0_dp
            gradient = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm objective: invalid callback context")
        end select
    end subroutine glm_objective_callback

    subroutine glm_objective_value_gradient_context(context, theta, value, &
            gradient, status)
        type(glm_objective_context_t), intent(in) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: eta, mu, response, residual, local_value
        integer :: i, j, n_features, n_parameters

        value = 0.0_dp
        gradient = 0.0_dp
        n_features = size(context%x, 2)
        n_parameters = n_features
        if (context%fit_intercept) n_parameters = n_parameters + 1
        if (size(theta) /= n_parameters .or. size(gradient) /= n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm objective callback: parameter shape is invalid")
            return
        end if
        do i = 1, size(context%x, 1)
            eta = 0.0_dp
            if (context%fit_intercept) eta = theta(1)
            do j = 1, n_features
                if (context%fit_intercept) then
                    eta = eta + context%x(i, j)*theta(j+1)
                else
                    eta = eta + context%x(i, j)*theta(j)
                end if
            end do
            if (.not. ieee_is_finite(eta) .or. eta > log(huge(1.0_dp))-2.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "glm objective callback: linear predictor is outside stable domain")
                return
            end if
            mu = exp(eta)
            response = context%target(i)
            if (context%family == GLM_FAMILY_POISSON) then
                local_value = mu - response*eta
                residual = mu - response
            else
                local_value = mu
                if (response > 0.0_dp) then
                    local_value = response/mu + eta
                    residual = 1.0_dp - response/mu
                else
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "glm objective callback: Gamma targets must be positive")
                    return
                end if
            end if
            local_value = local_value/context%dispersion
            residual = residual/context%dispersion
            value = value + context%weight(i)*local_value
            do j = 1, n_features
                if (context%fit_intercept) then
                    gradient(j+1) = gradient(j+1) + context%weight(i)*residual*context%x(i, j)
                else
                    gradient(j) = gradient(j) + context%weight(i)*residual*context%x(i, j)
                end if
            end do
            if (context%fit_intercept) gradient(1) = &
                gradient(1) + context%weight(i)*residual
        end do
        value = value/context%weight_mass
        gradient = gradient/context%weight_mass
        if (context%fit_intercept) then
            value = value + 0.5_dp*context%alpha*sum(theta(2:n_parameters)**2)
            gradient(2:n_parameters) = gradient(2:n_parameters) + &
                context%alpha*theta(2:n_parameters)
        else
            value = value + 0.5_dp*context%alpha*sum(theta**2)
            gradient = gradient + context%alpha*theta
        end if
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm objective callback: value or gradient is nonfinite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine glm_objective_value_gradient_context

    function glm_coefficients(self) result(values)
        class(glm_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        if (allocated(self%coefficient)) then
            allocate(values, mold=self%coefficient)
            values = self%coefficient
        else
            allocate(values(0, 0))
        end if
    end function glm_coefficients

    function glm_parameters(self) result(values)
        class(glm_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        allocate(values(self%parameter_count()))
        if (self%parameter_count() > 0) values = reshape(self%coefficient, &
            [self%parameter_count()])
    end function glm_parameters

    subroutine glm_set_parameters(self, values, status)
        class(glm_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. self%fitted_value .or. size(values) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(values)) .or. any(values < self%lower_bound_value) .or. &
                any(values > self%upper_bound_value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "glm set_parameters: model or packed values are invalid")
            return
        end if
        self%coefficient = reshape(values, shape(self%coefficient))
        call status_set(status, FORTNUM_OK, "")
    end subroutine glm_set_parameters

    integer function glm_parameter_count(self) result(count)
        class(glm_regression_t), intent(in) :: self
        if (allocated(self%coefficient)) then
            count = size(self%coefficient)
        else
            count = 0
        end if
    end function glm_parameter_count

    integer function glm_feature_count(self) result(count)
        class(glm_regression_t), intent(in) :: self
        count = self%feature_count_value
    end function glm_feature_count

    integer function glm_output_count(self) result(count)
        class(glm_regression_t), intent(in) :: self
        count = self%output_count_value
    end function glm_output_count

    integer function glm_family(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%family_value
    end function glm_family

    integer function glm_link(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%link_value
    end function glm_link

    real(dp) function glm_regularization(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%alpha_value
    end function glm_regularization

    real(dp) function glm_dispersion(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%dispersion_value
    end function glm_dispersion

    logical function glm_fit_intercept(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%fit_intercept_value
    end function glm_fit_intercept

    integer function glm_max_iterations(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%max_iterations_value
    end function glm_max_iterations

    real(dp) function glm_tolerance(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%tolerance_value
    end function glm_tolerance

    real(dp) function glm_lower_bound(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%lower_bound_value
    end function glm_lower_bound

    real(dp) function glm_upper_bound(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%upper_bound_value
    end function glm_upper_bound

    logical function glm_fitted(self) result(value)
        class(glm_regression_t), intent(in) :: self
        value = self%fitted_value .and. allocated(self%coefficient)
    end function glm_fitted

    logical function glm_device_supported(self, device_kind) result(supported)
        class(glm_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind
        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%fitted()
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function glm_device_supported

    subroutine make_design(x, intercept, design)
        real(dp), intent(in) :: x(:, :)
        logical, intent(in) :: intercept
        real(dp), intent(out) :: design(:, :)
        design = 0.0_dp
        if (intercept) then
            design(:, 1) = 1.0_dp
            design(:, 2:) = x
        else
            design = x
        end if
    end subroutine make_design

    subroutine make_tangent_design(x_dot, intercept, design_dot)
        real(dp), intent(in) :: x_dot(:, :)
        logical, intent(in) :: intercept
        real(dp), intent(out) :: design_dot(:, :)
        design_dot = 0.0_dp
        if (intercept) then
            design_dot(:, 2:) = x_dot
        else
            design_dot = x_dot
        end if
    end subroutine make_tangent_design

end module fortml_glm_regression
