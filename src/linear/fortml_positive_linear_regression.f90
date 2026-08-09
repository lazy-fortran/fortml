module fortml_positive_linear_regression
    !! Weighted nonnegative multi-output least-squares regression.
    !!
    !! Feature coefficients are constrained to be nonnegative.  The intercept
    !! is unconstrained by default and may be constrained with
    !! ``nonnegative_intercept=.true.``.  Fits use a deterministic projected
    !! gradient solve with a conservative Lipschitz step; prediction products
    !! hold the fitted state fixed and do not differentiate fit-time active-set
    !! decisions.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU
    implicit none
    private

    type, public :: positive_linear_regression_t
        private
        real(dp), allocatable :: coefficient(:, :)
        integer :: feature_count_value = 0
        integer :: output_count_value = 0
        integer :: max_iterations_value = 10000
        real(dp) :: tolerance_value = 1.0e-10_dp
        logical :: fit_intercept_value = .true.
        logical :: nonnegative_intercept_value = .false.
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit_matrix => positive_fit_matrix
        procedure, public :: fit_vector => positive_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: predict_matrix => positive_predict_matrix
        procedure, public :: predict_vector => positive_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_device => positive_predict_device
        procedure, public :: device_supported => positive_device_supported
        procedure, public :: predict_jvp => positive_predict_jvp
        procedure, public :: predict_vjp => positive_predict_vjp
        procedure, public :: jvp => positive_predict_jvp
        procedure, public :: vjp => positive_predict_vjp
        procedure, public :: coefficients => positive_coefficients
        procedure, public :: parameters => positive_parameters
        procedure, public :: set_parameters => positive_set_parameters
        procedure, public :: parameter_count => positive_parameter_count
        procedure, public :: feature_count => positive_feature_count
        procedure, public :: output_count => positive_output_count
        procedure, public :: fit_intercept => positive_fit_intercept
        procedure, public :: nonnegative_intercept => positive_nonnegative_intercept
        procedure, public :: max_iterations => positive_max_iterations
        procedure, public :: tolerance => positive_tolerance
        procedure, public :: fitted => positive_fitted
    end type positive_linear_regression_t

contains

    subroutine positive_fit_matrix(self, x, y, status, fit_intercept, &
            nonnegative_intercept, sample_weight, max_iterations, tolerance)
        class(positive_linear_regression_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fit_intercept, nonnegative_intercept
        real(dp), intent(in), optional :: sample_weight(:), tolerance
        integer, intent(in), optional :: max_iterations
        real(dp), allocatable :: weights(:), design(:, :), beta(:, :), gradient(:, :)
        real(dp), allocatable :: residual(:), candidate(:, :)
        real(dp) :: mass, step, objective, next_objective, scale, delta
        real(dp) :: curvature_total
        real(dp) :: tol
        integer :: n_samples, n_features, n_outputs, n_parameters
        integer :: iterations, iteration, j, k
        logical :: include_intercept, constrain_intercept, converged
        type(positive_linear_regression_t) :: candidate_state

        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_outputs = size(y, 2)
        include_intercept = .true.
        constrain_intercept = .false.
        iterations = 10000
        tol = 1.0e-10_dp
        if (present(fit_intercept)) include_intercept = fit_intercept
        if (present(nonnegative_intercept)) constrain_intercept = nonnegative_intercept
        if (present(max_iterations)) iterations = max_iterations
        if (present(tolerance)) tol = tolerance

        if (n_samples < 1 .or. n_features < 1 .or. n_outputs < 1 .or. &
            size(y, 1) /= n_samples .or. iterations < 1 .or. &
            .not. ieee_is_finite(tol) .or. tol <= 0.0_dp .or. &
            (constrain_intercept .and. .not. include_intercept)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear fit: invalid dimensions or solver options")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear fit: inputs and targets must be finite")
            return
        end if

        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "positive linear fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        mass = sum(weights)
        if (.not. ieee_is_finite(mass) .or. mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear fit: sample weights must have positive mass")
            return
        end if

        n_parameters = n_features
        if (include_intercept) n_parameters = n_parameters + 1
        allocate(design(n_samples, n_parameters))
        call make_design(x, include_intercept, design)
        allocate(beta(n_parameters, n_outputs), candidate(n_parameters, n_outputs))
        allocate(gradient(n_parameters, n_outputs), residual(n_samples))
        beta = 0.0_dp
        if (include_intercept .and. .not. constrain_intercept) then
            do k = 1, n_outputs
                beta(1, k) = sum(weights*y(:, k))/mass
            end do
        end if

        ! The Frobenius bound is conservative for D^T diag(w) D / mass,
        ! making the fixed step globally nonexpansive for every output.
        step = 0.0_dp
        do j = 1, n_parameters
            step = max(step, sum(weights*design(:, j)*design(:, j))/mass)
        end do
        curvature_total = sum(weights*design(:, 1)*design(:, 1))
        do j = 2, n_parameters
            curvature_total = curvature_total + &
                sum(weights*design(:, j)*design(:, j))
        end do
        if (curvature_total <= tiny(1.0_dp)) then
            candidate = beta
            converged = .true.
        else
            step = mass/curvature_total
            converged = .false.
            do iteration = 1, iterations
                do k = 1, n_outputs
                    residual = matmul(design, beta(:, k))-y(:, k)
                    gradient(:, k) = matmul(transpose(design), weights*residual)/mass
                end do
                candidate = beta-step*gradient
                call project_coefficients(candidate, include_intercept, &
                    constrain_intercept)
                delta = maxval(abs(candidate-beta))
                scale = 1.0_dp + maxval(abs(candidate))
                beta = candidate
                if (delta <= tol*scale) then
                    converged = .true.
                    exit
                end if
            end do
        end if
        if (.not. converged .or. any(.not. ieee_is_finite(beta))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "positive linear fit: projected gradient did not converge")
            return
        end if
        ! Validate the resulting objective before committing state.
        objective = 0.0_dp
        do k = 1, n_outputs
            residual = matmul(design, beta(:, k))-y(:, k)
            objective = objective + 0.5_dp*sum(weights*residual*residual)/mass
        end do
        next_objective = objective
        if (.not. ieee_is_finite(next_objective)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "positive linear fit: objective is nonfinite")
            return
        end if

        candidate_state%coefficient = beta
        candidate_state%feature_count_value = n_features
        candidate_state%output_count_value = n_outputs
        candidate_state%max_iterations_value = iterations
        candidate_state%tolerance_value = tol
        candidate_state%fit_intercept_value = include_intercept
        candidate_state%nonnegative_intercept_value = constrain_intercept
        candidate_state%fitted_value = .true.
        call move_alloc(candidate_state%coefficient, self%coefficient)
        self%feature_count_value = candidate_state%feature_count_value
        self%output_count_value = candidate_state%output_count_value
        self%max_iterations_value = candidate_state%max_iterations_value
        self%tolerance_value = candidate_state%tolerance_value
        self%fit_intercept_value = candidate_state%fit_intercept_value
        self%nonnegative_intercept_value = candidate_state%nonnegative_intercept_value
        self%fitted_value = candidate_state%fitted_value
        call status_set(status, FORTNUM_OK, "")
    end subroutine positive_fit_matrix

    subroutine positive_fit_vector(self, x, y, status, fit_intercept, &
            nonnegative_intercept, sample_weight, max_iterations, tolerance)
        class(positive_linear_regression_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fit_intercept, nonnegative_intercept
        real(dp), intent(in), optional :: sample_weight(:), tolerance
        integer, intent(in), optional :: max_iterations
        real(dp), allocatable :: target(:, :)

        allocate(target(size(y), 1))
        target(:, 1) = y
        call positive_fit_matrix(self, x, target, status, fit_intercept, &
            nonnegative_intercept, sample_weight, max_iterations, tolerance)
    end subroutine positive_fit_vector

    subroutine positive_predict_matrix(self, x, y, status)
        class(positive_linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :)

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear predict: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear predict: inputs must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        call make_design(x, self%fit_intercept_value, design)
        y = matmul(design, self%coefficient)
        call status_set(status, FORTNUM_OK, "")
    end subroutine positive_predict_matrix

    subroutine positive_predict_vector(self, x, y, status)
        class(positive_linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear predict: output shape is invalid")
            return
        end if
        allocate(values(size(y), 1))
        call positive_predict_matrix(self, x, values, status)
        if (status_ok(status)) y = values(:, 1)
    end subroutine positive_predict_vector

    subroutine positive_predict_device(self, device, x, y, status)
        class(positive_linear_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (device%kind == FORTML_DEVICE_CPU) then
            call self%predict_matrix(x, y, status)
        else
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "positive linear prediction: CUDA kernel is not resident")
        end if
    end subroutine positive_predict_device

    logical function positive_device_supported(self, device_kind) result(supported)
        class(positive_linear_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function positive_device_supported

    subroutine positive_predict_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(positive_linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), design_dot(:, :), coefficient_dot(:, :)

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value]) .or. &
            any(shape(y_dot) /= shape(y)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(theta_dot)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear JVP: inputs and tangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(design_dot, mold=design)
        allocate(coefficient_dot, mold=self%coefficient)
        call make_design(x, self%fit_intercept_value, design)
        call make_tangent_design(x_dot, self%fit_intercept_value, design_dot)
        coefficient_dot = reshape(theta_dot, shape(coefficient_dot))
        y = matmul(design, self%coefficient)
        y_dot = matmul(design_dot, self%coefficient) + &
            matmul(design, coefficient_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine positive_predict_jvp

    subroutine positive_predict_vjp(self, x, u, theta_bar, x_bar, status)
        class(positive_linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), coefficient_bar(:, :)
        integer :: j, offset

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(u) /= [size(x, 1), self%output_count_value]) .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(u))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(coefficient_bar, mold=self%coefficient)
        call make_design(x, self%fit_intercept_value, design)
        coefficient_bar = matmul(transpose(design), u)
        theta_bar = reshape(coefficient_bar, [size(theta_bar)])
        x_bar = 0.0_dp
        offset = 0
        if (self%fit_intercept_value) offset = 1
        do j = 1, self%feature_count_value
            x_bar(:, j) = matmul(u, self%coefficient(offset+j, :))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine positive_predict_vjp

    function positive_coefficients(self) result(values)
        class(positive_linear_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%coefficient)) then
            allocate(values, mold=self%coefficient)
            values = self%coefficient
        else
            allocate(values(0, 0))
        end if
    end function positive_coefficients

    function positive_parameters(self) result(values)
        class(positive_linear_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        allocate(values(self%parameter_count()))
        if (self%parameter_count() > 0) values = reshape(self%coefficient, [size(values)])
    end function positive_parameters

    subroutine positive_set_parameters(self, values, status)
        class(positive_linear_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted_value .or. size(values) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear set_parameters: model or packed shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear set_parameters: packed values must be finite")
            return
        end if
        if (any(values(positive_constraint_start(self):) < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "positive linear set_parameters: constrained coefficients must be nonnegative")
            return
        end if
        self%coefficient = reshape(values, shape(self%coefficient))
        call status_set(status, FORTNUM_OK, "")
    end subroutine positive_set_parameters

    integer function positive_constraint_start(self) result(first)
        class(positive_linear_regression_t), intent(in) :: self

        first = 1
        if (self%fit_intercept_value .and. .not. self%nonnegative_intercept_value) first = 2
    end function positive_constraint_start

    integer function positive_parameter_count(self) result(count)
        class(positive_linear_regression_t), intent(in) :: self

        if (allocated(self%coefficient)) then
            count = size(self%coefficient)
        else
            count = 0
        end if
    end function positive_parameter_count

    integer function positive_feature_count(self) result(count)
        class(positive_linear_regression_t), intent(in) :: self

        count = self%feature_count_value
    end function positive_feature_count

    integer function positive_output_count(self) result(count)
        class(positive_linear_regression_t), intent(in) :: self

        count = self%output_count_value
    end function positive_output_count

    logical function positive_fit_intercept(self) result(value)
        class(positive_linear_regression_t), intent(in) :: self

        value = self%fit_intercept_value
    end function positive_fit_intercept

    logical function positive_nonnegative_intercept(self) result(value)
        class(positive_linear_regression_t), intent(in) :: self

        value = self%nonnegative_intercept_value
    end function positive_nonnegative_intercept

    integer function positive_max_iterations(self) result(value)
        class(positive_linear_regression_t), intent(in) :: self

        value = self%max_iterations_value
    end function positive_max_iterations

    real(dp) function positive_tolerance(self) result(value)
        class(positive_linear_regression_t), intent(in) :: self

        value = self%tolerance_value
    end function positive_tolerance

    logical function positive_fitted(self) result(value)
        class(positive_linear_regression_t), intent(in) :: self

        value = self%fitted_value .and. allocated(self%coefficient)
    end function positive_fitted

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

    subroutine project_coefficients(values, include_intercept, constrain_intercept)
        real(dp), intent(inout) :: values(:, :)
        logical, intent(in) :: include_intercept, constrain_intercept
        integer :: first

        first = 1
        if (include_intercept .and. .not. constrain_intercept) first = 2
        values(first:, :) = max(values(first:, :), 0.0_dp)
    end subroutine project_coefficients

end module fortml_positive_linear_regression
