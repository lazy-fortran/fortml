module fortml_elastic_net_regression
    !! Weighted elastic-net regression with a deterministic coordinate solver.
    !!
    !! The fitted objective is
    !!
    !!   0.5 * sum_i w_i (y_i - b - x_i beta)^2 / sum_i w_i
    !!       + alpha * [ l1_ratio * ||beta||_1
    !!                   + 0.5 * (1-l1_ratio) * ||beta||_2^2 ].
    !!
    !! The intercept is never regularized.  The coordinate-descent fit is a
    !! deliberate nonsmooth boundary: products below differentiate a fixed
    !! fitted prediction state, while fit-time active-set and convergence
    !! decisions are not differentiated.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    type, public :: elastic_net_regression_t
        private
        real(dp), allocatable :: coefficient(:, :)
        integer :: feature_count_value = 0
        integer :: output_count_value = 0
        real(dp) :: alpha_value = 1.0_dp
        real(dp) :: l1_ratio_value = 0.5_dp
        logical :: fit_intercept_value = .true.
        integer :: max_iterations_value = 1000
        real(dp) :: tolerance_value = 1.0e-8_dp
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit_matrix => elastic_net_fit_matrix
        procedure, public :: fit_vector => elastic_net_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: predict_matrix => elastic_net_predict_matrix
        procedure, public :: predict_vector => elastic_net_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_device => elastic_net_predict_device
        procedure, public :: device_supported => elastic_net_device_supported
        procedure, public :: predict_jvp => elastic_net_predict_jvp
        procedure, public :: predict_vjp => elastic_net_predict_vjp
        procedure, public :: jvp => elastic_net_predict_jvp
        procedure, public :: vjp => elastic_net_predict_vjp
        procedure, public :: coefficients => elastic_net_coefficients
        procedure, public :: parameters => elastic_net_parameters
        procedure, public :: set_parameters => elastic_net_set_parameters
        procedure, public :: parameter_count => elastic_net_parameter_count
        procedure, public :: feature_count => elastic_net_feature_count
        procedure, public :: output_count => elastic_net_output_count
        procedure, public :: regularization => elastic_net_regularization
        procedure, public :: l1_ratio => elastic_net_l1_ratio
        procedure, public :: fit_intercept => elastic_net_fit_intercept
        procedure, public :: max_iterations => elastic_net_max_iterations
        procedure, public :: tolerance => elastic_net_tolerance
        procedure, public :: fitted => elastic_net_fitted
    end type elastic_net_regression_t

contains

    subroutine elastic_net_fit_matrix(self, x, y, status, alpha, l1_ratio, &
            fit_intercept, sample_weight, max_iterations, tolerance)
        class(elastic_net_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha, l1_ratio, sample_weight(:), tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), allocatable :: weights(:), beta(:, :), residual(:), prediction(:)
        real(dp) :: penalty, ratio, tol, mass, intercept, old_intercept
        real(dp) :: rho, z, beta_old, beta_new, delta, scale
        integer :: n_samples, n_features, n_outputs, iterations, i, j, k, iteration
        logical :: include_intercept, converged

        self%fitted_value = .false.
        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_outputs = size(y, 2)
        penalty = 1.0_dp
        ratio = 0.5_dp
        include_intercept = .true.
        iterations = 1000
        tol = 1.0e-8_dp
        if (present(alpha)) penalty = alpha
        if (present(l1_ratio)) ratio = l1_ratio
        if (present(fit_intercept)) include_intercept = fit_intercept
        if (present(max_iterations)) iterations = max_iterations
        if (present(tolerance)) tol = tolerance

        if (n_samples < 1 .or. n_features < 1 .or. n_outputs < 1 .or. &
            size(y, 1) /= n_samples .or. .not. ieee_is_finite(penalty) .or. &
            penalty < 0.0_dp .or. .not. ieee_is_finite(ratio) .or. ratio < 0.0_dp .or. &
            ratio > 1.0_dp .or. iterations < 1 .or. .not. ieee_is_finite(tol) .or. &
            tol <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net fit: invalid dimensions or solver options")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net fit: inputs and targets must be finite")
            return
        end if

        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "elastic-net fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        mass = sum(weights)
        if (.not. ieee_is_finite(mass) .or. mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net fit: sample weights must have positive mass")
            return
        end if

        allocate(beta(n_features, n_outputs), prediction(n_samples), residual(n_samples))
        allocate(self%coefficient(n_features+1, n_outputs), source=0.0_dp)
        beta = 0.0_dp
        converged = .false.
        do k = 1, n_outputs
            beta(:, k) = 0.0_dp
            if (include_intercept) then
                intercept = sum(weights*y(:, k))/mass
            else
                intercept = 0.0_dp
            end if
            do iteration = 1, iterations
                old_intercept = intercept
                prediction = matmul(x, beta(:, k))
                if (include_intercept) then
                    intercept = sum(weights*(y(:, k)-prediction))/mass
                else
                    intercept = 0.0_dp
                end if
                residual = y(:, k)-intercept-prediction
                delta = abs(intercept-old_intercept)
                do j = 1, n_features
                    beta_old = beta(j, k)
                    residual = residual + x(:, j)*beta_old
                    rho = sum(weights*x(:, j)*residual)/mass
                    z = sum(weights*x(:, j)*x(:, j))/mass
                    if (z <= tiny(1.0_dp) .and. penalty*(1.0_dp-ratio) <= 0.0_dp) then
                        beta_new = 0.0_dp
                    else
                        beta_new = soft_threshold(rho, penalty*ratio)/ &
                            (z + penalty*(1.0_dp-ratio))
                    end if
                    beta(j, k) = beta_new
                    residual = residual-x(:, j)*beta_new
                    delta = max(delta, abs(beta_new-beta_old))
                end do
                scale = 1.0_dp + max(abs(intercept), maxval(abs(beta(:, k))))
                if (delta <= tol*scale) then
                    converged = .true.
                    exit
                end if
            end do
            if (.not. converged) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "elastic-net fit: coordinate descent did not converge")
                return
            end if
            self%coefficient(1, k) = intercept
            self%coefficient(2:, k) = beta(:, k)
            converged = .false.
        end do
        if (.not. include_intercept) self%coefficient(1, :) = 0.0_dp
        self%feature_count_value = n_features
        self%output_count_value = n_outputs
        self%alpha_value = penalty
        self%l1_ratio_value = ratio
        self%fit_intercept_value = include_intercept
        self%max_iterations_value = iterations
        self%tolerance_value = tol
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine elastic_net_fit_matrix

    subroutine elastic_net_fit_vector(self, x, y, status, alpha, l1_ratio, &
            fit_intercept, sample_weight, max_iterations, tolerance)
        class(elastic_net_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha, l1_ratio, sample_weight(:), tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), allocatable :: target(:, :)

        allocate(target(size(y), 1))
        target(:, 1) = y
        call elastic_net_fit_matrix(self, x, target, status, alpha, l1_ratio, &
            fit_intercept, sample_weight, max_iterations, tolerance)
    end subroutine elastic_net_fit_vector

    subroutine elastic_net_predict_matrix(self, x, y, status)
        class(elastic_net_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :)

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net predict: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net predict: inputs must be finite")
            return
        end if
        allocate(design(size(x, 1), self%feature_count_value+1))
        call make_design(x, self%fit_intercept_value, design)
        y = matmul(design, self%coefficient)
        call status_set(status, FORTNUM_OK, "")
    end subroutine elastic_net_predict_matrix

    subroutine elastic_net_predict_vector(self, x, y, status)
        class(elastic_net_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net predict: output shape is invalid")
            return
        end if
        allocate(values(size(y), 1))
        call elastic_net_predict_matrix(self, x, values, status)
        if (status%code == FORTNUM_OK) y = values(:, 1)
    end subroutine elastic_net_predict_vector

    subroutine elastic_net_predict_device(self, device, x, y, status)
        !! Predict through the explicit device contract.
        !!
        !! A resident elastic-net prediction kernel is not linked yet.  CPU
        !! dispatch is therefore the only successful path; a selected CUDA
        !! context returns a typed refusal instead of silently copying to the
        !! host.  This method is intentionally matrix-shaped, matching the
        !! complete multi-output prediction primitive.
        class(elastic_net_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_matrix(x, y, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "elastic-net device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net device prediction: device kind is invalid")
        end select
    end subroutine elastic_net_predict_device

    subroutine elastic_net_predict_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(elastic_net_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), design_dot(:, :), coefficient_dot(:, :)

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. any(shape(x_dot) /= shape(x)) .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value]) .or. &
            any(shape(y_dot) /= shape(y)) .or. size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net JVP: inputs and tangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(design_dot(size(x, 1), size(self%coefficient, 1)))
        allocate(coefficient_dot(size(self%coefficient, 1), &
            size(self%coefficient, 2)))
        call make_design(x, self%fit_intercept_value, design)
        call make_tangent_design(x_dot, self%fit_intercept_value, design_dot)
        coefficient_dot = reshape(theta_dot, shape(coefficient_dot))
        y = matmul(design, self%coefficient)
        y_dot = matmul(design_dot, self%coefficient) + matmul(design, coefficient_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine elastic_net_predict_jvp

    subroutine elastic_net_predict_vjp(self, x, u, theta_bar, x_bar, status)
        class(elastic_net_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), coefficient_bar(:, :)
        integer :: j

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(u) /= [size(x, 1), self%output_count_value]) .or. &
            size(theta_bar) /= self%parameter_count() .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(u))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(coefficient_bar(size(self%coefficient, 1), &
            size(self%coefficient, 2)))
        call make_design(x, self%fit_intercept_value, design)
        coefficient_bar = matmul(transpose(design), u)
        theta_bar = reshape(coefficient_bar, [size(theta_bar)])
        x_bar = 0.0_dp
        do j = 1, self%feature_count_value
            x_bar(:, j) = matmul(u, self%coefficient(j+1, :))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine elastic_net_predict_vjp

    function elastic_net_coefficients(self) result(values)
        class(elastic_net_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%coefficient)) then
            allocate(values, mold=self%coefficient)
            values = self%coefficient
        else
            allocate(values(0, 0))
        end if
    end function elastic_net_coefficients

    function elastic_net_parameters(self) result(values)
        class(elastic_net_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        allocate(values(self%parameter_count()))
        if (self%parameter_count() > 0) values = reshape(self%coefficient, [self%parameter_count()])
    end function elastic_net_parameters

    subroutine elastic_net_set_parameters(self, values, status)
        class(elastic_net_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted_value .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "elastic-net set_parameters: model or packed values are invalid")
            return
        end if
        self%coefficient = reshape(values, shape(self%coefficient))
        call status_set(status, FORTNUM_OK, "")
    end subroutine elastic_net_set_parameters

    integer function elastic_net_parameter_count(self) result(count)
        class(elastic_net_regression_t), intent(in) :: self

        if (allocated(self%coefficient)) then
            count = size(self%coefficient)
        else
            count = 0
        end if
    end function elastic_net_parameter_count

    integer function elastic_net_feature_count(self) result(count)
        class(elastic_net_regression_t), intent(in) :: self
        count = self%feature_count_value
    end function elastic_net_feature_count

    integer function elastic_net_output_count(self) result(count)
        class(elastic_net_regression_t), intent(in) :: self
        count = self%output_count_value
    end function elastic_net_output_count

    real(dp) function elastic_net_regularization(self) result(value)
        class(elastic_net_regression_t), intent(in) :: self
        value = self%alpha_value
    end function elastic_net_regularization

    real(dp) function elastic_net_l1_ratio(self) result(value)
        class(elastic_net_regression_t), intent(in) :: self
        value = self%l1_ratio_value
    end function elastic_net_l1_ratio

    logical function elastic_net_fit_intercept(self) result(value)
        class(elastic_net_regression_t), intent(in) :: self
        value = self%fit_intercept_value
    end function elastic_net_fit_intercept

    integer function elastic_net_max_iterations(self) result(value)
        class(elastic_net_regression_t), intent(in) :: self
        value = self%max_iterations_value
    end function elastic_net_max_iterations

    real(dp) function elastic_net_tolerance(self) result(value)
        class(elastic_net_regression_t), intent(in) :: self
        value = self%tolerance_value
    end function elastic_net_tolerance

    logical function elastic_net_fitted(self) result(value)
        class(elastic_net_regression_t), intent(in) :: self
        value = self%fitted_value .and. allocated(self%coefficient)
    end function elastic_net_fitted

    logical function elastic_net_device_supported(self, device_kind) result(supported)
        !! Report support without inferring a host fallback for accelerators.
        class(elastic_net_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%fitted()
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function elastic_net_device_supported

    pure real(dp) function soft_threshold(value, threshold) result(output)
        real(dp), intent(in) :: value, threshold

        if (value > threshold) then
            output = value-threshold
        else if (value < -threshold) then
            output = value+threshold
        else
            output = 0.0_dp
        end if
    end function soft_threshold

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

end module fortml_elastic_net_regression
