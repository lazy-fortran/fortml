module fortml_basis_pipeline_training
    !! Differentiable least-squares objective for a basis feature pipeline.
    !!
    !! The packed variable is `[pipeline parameters, linear coefficients]`.
    !! The objective keeps the pipeline differentiable instead of fitting the
    !! linear block once and silently freezing the basis hyperparameters.  Its
    !! value, gradient, JVP, VJP, and HVP are analytic and use the pipeline's
    !! own chained products.  The adapter is CPU-only until a resident basis
    !! transform and linear solve are available; CUDA requests return a typed
    !! refusal rather than copying through the host.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_pipeline, only: basis_pipeline_t
    use fortml_linear_regression, only: linear_predict_jvp, linear_predict_vjp
    use fortopt_objective, only: objective_t
    implicit none
    private

    type, public :: basis_pipeline_training_objective_t
        !! MSE plus optional ridge penalty over a composable basis pipeline.
        private
        type(basis_pipeline_t), pointer :: pipeline => null()
        real(dp), allocatable :: features(:, :), targets(:, :)
        real(dp), allocatable :: coef(:, :)
        real(dp) :: ridge = 0.0_dp
        logical :: fit_intercept = .true.
        integer :: device_kind = FORTML_DEVICE_CPU
    contains
        procedure, public :: initialize => basis_training_initialize
        procedure, public :: initialized => basis_training_initialized
        procedure, public :: device_supported => basis_training_device_supported
        procedure, public :: parameter_count => basis_training_parameter_count
        procedure, public :: parameters => basis_training_parameters
        procedure, public :: set_parameters => basis_training_set_parameters
        procedure, public :: value_gradient => basis_training_value_gradient
        procedure, public :: jvp => basis_training_jvp
        procedure, public :: vjp => basis_training_vjp
        procedure, public :: hvp => basis_training_hvp
        procedure, public :: fortopt => basis_training_fortopt
    end type basis_pipeline_training_objective_t

contains

    subroutine basis_training_initialize(self, pipeline, x, target, status, &
            ridge, fit_intercept, device_kind)
        class(basis_pipeline_training_objective_t), intent(out) :: self
        type(basis_pipeline_t), target, intent(inout) :: pipeline
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: ridge
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: device_kind
        integer :: requested_device

        self%device_kind = FORTML_DEVICE_CPU
        if (present(device_kind)) self%device_kind = device_kind
        requested_device = self%device_kind
        if (requested_device == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "basis training objective: CUDA resident products are not implemented")
            return
        end if
        if (requested_device /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective: device kind is invalid")
            return
        end if
        self%ridge = 0.0_dp
        if (present(ridge)) self%ridge = ridge
        self%fit_intercept = .true.
        if (present(fit_intercept)) self%fit_intercept = fit_intercept
        if (.not. pipeline%valid() .or. size(x, 1) < 1 .or. &
            size(x, 2) /= pipeline%input_count() .or. &
            size(target, 1) /= size(x, 1) .or. size(target, 2) < 1 .or. &
            self%ridge < 0.0_dp .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(target))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective: pipeline or data is invalid")
            return
        end if
        call pipeline%fit(x, status)
        if (status%code /= FORTNUM_OK) return
        self%pipeline => pipeline
        allocate(self%features, source=x)
        allocate(self%targets, source=target)
        allocate(self%coef(pipeline%feature_count() + 1, size(target, 2)))
        self%coef = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_training_initialize

    logical function basis_training_initialized(self) result(yes)
        class(basis_pipeline_training_objective_t), intent(in) :: self

        yes = associated(self%pipeline) .and. allocated(self%features) .and. &
            allocated(self%targets) .and. allocated(self%coef)
        if (.not. yes) return
        yes = self%pipeline%valid() .and. self%pipeline%is_fitted() .and. &
            size(self%coef, 1) == self%pipeline%feature_count() + 1 .and. &
            size(self%coef, 2) == size(self%targets, 2)
    end function basis_training_initialized

    logical function basis_training_device_supported(self, device_kind) result(yes)
        class(basis_pipeline_training_objective_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = requested == FORTML_DEVICE_CPU
    end function basis_training_device_supported

    integer function basis_training_parameter_count(self) result(count)
        class(basis_pipeline_training_objective_t), intent(in) :: self

        count = 0
        if (.not. associated(self%pipeline)) return
        count = self%pipeline%parameter_count()
        if (allocated(self%coef)) count = count + size(self%coef)
    end function basis_training_parameter_count

    function basis_training_parameters(self) result(parameters)
        class(basis_pipeline_training_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:), pipeline_parameters(:)
        integer :: n_pipeline

        allocate(parameters(self%parameter_count()))
        parameters = 0.0_dp
        if (.not. self%initialized()) return
        n_pipeline = self%pipeline%parameter_count()
        if (n_pipeline > 0) then
            pipeline_parameters = self%pipeline%parameters()
            parameters(:n_pipeline) = pipeline_parameters
        end if
        parameters(n_pipeline + 1:) = reshape(self%coef, [size(self%coef)])
    end function basis_training_parameters

    subroutine basis_training_set_parameters(self, parameters, status)
        class(basis_pipeline_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_pipeline

        if (.not. self%initialized() .or. size(parameters) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective: parameter vector is invalid")
            return
        end if
        n_pipeline = self%pipeline%parameter_count()
        if (n_pipeline > 0) then
            call self%pipeline%set_parameters(parameters(:n_pipeline), status)
            if (status%code /= FORTNUM_OK) return
        end if
        self%coef = reshape(parameters(n_pipeline + 1:), shape(self%coef))
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_training_set_parameters

    subroutine basis_training_value_gradient(self, parameters, value, gradient, status)
        class(basis_pipeline_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: phi(:, :), prediction(:, :), prediction_dot(:, :)
        real(dp), allocatable :: residual(:, :), feature_bar(:, :)
        real(dp), allocatable :: pipeline_bar(:), coefficient_bar(:, :), x_bar(:, :)
        real(dp) :: scale, penalty
        integer :: n_pipeline

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. valid_objective_inputs(self, parameters, gradient, status)) return
        call self%set_parameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        n_pipeline = self%pipeline%parameter_count()
        scale = 1.0_dp / real(size(self%features, 1)*size(self%targets, 2), dp)
        allocate(phi(size(self%features, 1), self%pipeline%feature_count()))
        allocate(prediction(size(self%features, 1), size(self%targets, 2)))
        allocate(prediction_dot, mold=prediction)
        allocate(residual, mold=prediction)
        allocate(coefficient_bar, mold=self%coef)
        allocate(feature_bar, mold=phi)
        allocate(x_bar, mold=self%features)
        allocate(pipeline_bar(n_pipeline))
        call self%pipeline%transform(self%features, phi, status)
        if (status%code /= FORTNUM_OK) return
        call linear_predict_jvp(self%coef, phi, 0.0_dp*self%coef, &
            0.0_dp*phi, prediction, prediction_dot, self%fit_intercept)
        residual = prediction - self%targets
        value = 0.5_dp*scale*sum(residual*residual)
        call linear_predict_vjp(self%coef, phi, scale*residual, coefficient_bar, &
            feature_bar, self%fit_intercept)
        call self%pipeline%vjp(self%features, feature_bar, pipeline_bar, &
            x_bar, status)
        if (status%code /= FORTNUM_OK) return
        gradient(:n_pipeline) = pipeline_bar
        gradient(n_pipeline + 1:) = reshape(coefficient_bar, [size(coefficient_bar)])
        penalty = ridge_penalty(self%coef, self%ridge, self%fit_intercept)
        value = value + penalty
        call add_ridge_gradient(self%coef, self%ridge, self%fit_intercept, &
            gradient(n_pipeline + 1:))
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_training_value_gradient

    subroutine basis_training_jvp(self, parameters, direction, value, tangent, status)
        class(basis_pipeline_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective JVP: direction is invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        tangent = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_training_jvp

    subroutine basis_training_vjp(self, parameters, output_bar, gradient, status)
        class(basis_pipeline_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective VJP: output cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_training_vjp

    subroutine basis_training_hvp(self, parameters, direction, product, status)
        class(basis_pipeline_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: phi(:, :), phi_dot(:, :), prediction(:, :)
        real(dp), allocatable :: prediction_dot(:, :), residual(:, :), u_dot(:, :)
        real(dp), allocatable :: feature_bar(:, :), feature_bar_dot(:, :)
        real(dp), allocatable :: pipeline_hvp(:), pipeline_vjp(:), zero_x(:, :), &
            x_hvp(:, :)
        real(dp), allocatable :: coefficient_bar_dot(:, :), coefficient_hvp(:, :)
        real(dp), allocatable :: coefficient_dot(:, :)
        real(dp) :: scale
        integer :: n_pipeline, n_features

        product = 0.0_dp
        if (.not. valid_objective_inputs(self, parameters, product, status)) return
        if (size(direction) /= size(parameters) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective HVP: direction is invalid")
            return
        end if
        call self%set_parameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        n_pipeline = self%pipeline%parameter_count()
        n_features = self%pipeline%feature_count()
        scale = 1.0_dp / real(size(self%features, 1)*size(self%targets, 2), dp)
        allocate(phi(size(self%features, 1), n_features))
        allocate(phi_dot, mold=phi)
        allocate(prediction(size(self%features, 1), size(self%targets, 2)))
        allocate(prediction_dot, mold=prediction)
        allocate(residual, mold=prediction)
        allocate(u_dot, mold=prediction)
        allocate(zero_x, mold=self%features)
        allocate(x_hvp, mold=self%features)
        zero_x = 0.0_dp
        allocate(coefficient_dot, mold=self%coef)
        coefficient_dot = reshape(direction(n_pipeline + 1:), shape(self%coef))
        call self%pipeline%jvp(self%features, direction(:n_pipeline), zero_x, phi, &
            phi_dot, status)
        if (status%code /= FORTNUM_OK) return
        call linear_predict_jvp(self%coef, phi, coefficient_dot, phi_dot, prediction, &
            prediction_dot, self%fit_intercept)
        residual = scale*(prediction - self%targets)
        u_dot = scale*prediction_dot
        allocate(feature_bar, mold=phi)
        allocate(feature_bar_dot, mold=phi)
        allocate(coefficient_bar_dot, mold=self%coef)
        call linear_predict_vjp(self%coef, phi, residual, coefficient_bar_dot, &
            feature_bar, self%fit_intercept)
        call linear_predict_vjp(self%coef, phi, u_dot, coefficient_bar_dot, &
            feature_bar_dot, self%fit_intercept)
        feature_bar_dot = feature_bar_dot + matmul(residual, &
            transpose(coefficient_dot(2:, :)))
        allocate(pipeline_hvp(n_pipeline), pipeline_vjp(n_pipeline))
        call self%pipeline%hvp(self%features, feature_bar, direction(:n_pipeline), &
            zero_x, pipeline_hvp, x_hvp, status)
        if (status%code /= FORTNUM_OK) return
        call self%pipeline%vjp(self%features, feature_bar_dot, pipeline_vjp, &
            x_hvp, status)
        if (status%code /= FORTNUM_OK) return
        allocate(coefficient_hvp, mold=self%coef)
        coefficient_hvp = coefficient_bar_dot + 0.0_dp
        coefficient_hvp(2:, :) = coefficient_hvp(2:, :) + &
            matmul(transpose(phi_dot), residual)
        call add_ridge_hvp(self%coef, coefficient_dot, self%ridge, &
            self%fit_intercept, coefficient_hvp)
        product(:n_pipeline) = pipeline_hvp + pipeline_vjp
        product(n_pipeline + 1:) = reshape(coefficient_hvp, [size(coefficient_hvp)])
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_training_hvp

    subroutine basis_training_fortopt(self, objective, status)
        class(basis_pipeline_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            basis_training_context_callback, status)
    end subroutine basis_training_fortopt

    subroutine basis_training_context_callback(context, parameters, value, gradient, &
            status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (basis_pipeline_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective: context has the wrong type")
        end select
    end subroutine basis_training_context_callback

    logical function valid_objective_inputs(self, parameters, gradient, status) &
            result(valid)
        class(basis_pipeline_training_objective_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:), gradient(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective: adapter is not initialized")
            return
        end if
        if (size(parameters) /= self%parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis training objective: parameter or gradient shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_objective_inputs

    real(dp) function ridge_penalty(coef, ridge, fit_intercept) result(value)
        real(dp), intent(in) :: coef(:, :), ridge
        logical, intent(in) :: fit_intercept

        value = 0.5_dp*ridge*sum(coef(1 + merge(1, 0, fit_intercept):, :)**2)
    end function ridge_penalty

    subroutine add_ridge_gradient(coef, ridge, fit_intercept, gradient)
        real(dp), intent(in) :: coef(:, :), ridge
        logical, intent(in) :: fit_intercept
        real(dp), intent(inout) :: gradient(:)
        integer :: offset

        offset = 1 + merge(1, 0, fit_intercept)
        gradient(offset:) = gradient(offset:) + ridge*reshape(coef(offset:, :), &
            [size(coef(offset:, :))])
    end subroutine add_ridge_gradient

    subroutine add_ridge_hvp(coef, coefficient_dot, ridge, fit_intercept, product)
        real(dp), intent(in) :: coef(:, :), coefficient_dot(:, :), ridge
        logical, intent(in) :: fit_intercept
        real(dp), intent(inout) :: product(:, :)
        integer :: offset

        offset = 1 + merge(1, 0, fit_intercept)
        product(offset:, :) = product(offset:, :) + ridge*coefficient_dot(offset:, :)
    end subroutine add_ridge_hvp

end module fortml_basis_pipeline_training
