module fortml_bayesian_ridge
    !! Weighted Bayesian ridge regression with deterministic posterior state.
    !!
    !! This is the fixed-hyperparameter, dense conjugate-Gaussian slice of
    !! scikit-learn's BayesianRidge estimator.  `alpha` is the observation
    !! precision and `lambda` is the isotropic coefficient prior precision.
    !! Both are explicit fitted metadata; evidence maximisation/ARD and
    !! derivative-through-fit are intentionally separate contracts.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU
    implicit none
    private

    type, public :: bayesian_ridge_regression_t
        private
        real(dp), allocatable :: coefficient(:, :)
        real(dp), allocatable :: posterior_precision_value(:, :)
        real(dp), allocatable :: posterior_covariance_value(:, :)
        integer :: feature_count_value = 0
        integer :: output_count_value = 0
        real(dp) :: alpha_value = 1.0_dp
        real(dp) :: lambda_value = 1.0_dp
        real(dp) :: log_evidence_value = 0.0_dp
        logical :: fit_intercept_value = .true.
        logical :: fitted_value = .false.
    contains
        procedure, public :: initialize => bayesian_ridge_initialize
        procedure, public :: fit_matrix => bayesian_ridge_fit_matrix
        procedure, public :: fit_vector => bayesian_ridge_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: predict_matrix => bayesian_ridge_predict_matrix
        procedure, public :: predict_vector => bayesian_ridge_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_device => bayesian_ridge_predict_device
        procedure, public :: device_supported => bayesian_ridge_device_supported
        procedure, public :: predict_jvp => bayesian_ridge_predict_jvp
        procedure, public :: predict_vjp => bayesian_ridge_predict_vjp
        procedure, public :: jvp => bayesian_ridge_predict_jvp
        procedure, public :: vjp => bayesian_ridge_predict_vjp
        procedure, public :: parameters => bayesian_ridge_parameters
        procedure, public :: set_parameters => bayesian_ridge_set_parameters
        procedure, public :: posterior_mean => bayesian_ridge_posterior_mean
        procedure, public :: posterior_precision => bayesian_ridge_posterior_precision
        procedure, public :: posterior_covariance => bayesian_ridge_posterior_covariance
        procedure, public :: alpha => bayesian_ridge_alpha
        procedure, public :: lambda => bayesian_ridge_lambda
        procedure, public :: log_evidence => bayesian_ridge_log_evidence
        procedure, public :: feature_count => bayesian_ridge_feature_count
        procedure, public :: output_count => bayesian_ridge_output_count
        procedure, public :: parameter_count => bayesian_ridge_parameter_count
        procedure, public :: fit_intercept => bayesian_ridge_fit_intercept
        procedure, public :: fitted => bayesian_ridge_fitted
    end type bayesian_ridge_regression_t

contains

    subroutine bayesian_ridge_initialize(self, n_features, n_outputs, status, &
            fit_intercept)
        class(bayesian_ridge_regression_t), intent(out) :: self
        integer, intent(in) :: n_features, n_outputs
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fit_intercept
        logical :: intercept
        integer :: n_parameters

        intercept = .true.
        if (present(fit_intercept)) intercept = fit_intercept
        if (n_features < 1 .or. n_outputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Bayesian ridge initialize: dimensions must be positive")
            return
        end if
        n_parameters = n_features + merge(1, 0, intercept)
        allocate(self%coefficient(n_parameters, n_outputs))
        allocate(self%posterior_precision_value(n_parameters, n_parameters))
        allocate(self%posterior_covariance_value(n_parameters, n_parameters))
        self%coefficient = 0.0_dp
        self%posterior_precision_value = 0.0_dp
        self%posterior_covariance_value = 0.0_dp
        self%feature_count_value = n_features
        self%output_count_value = n_outputs
        self%fit_intercept_value = intercept
        self%alpha_value = 1.0_dp
        self%lambda_value = 1.0_dp
        self%log_evidence_value = 0.0_dp
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine bayesian_ridge_initialize

    subroutine bayesian_ridge_fit_matrix(self, x, y, status, alpha, lambda, &
            fit_intercept, sample_weight)
        class(bayesian_ridge_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha, lambda, sample_weight(:)
        logical, intent(in), optional :: fit_intercept
        real(dp), allocatable :: design(:, :), precision(:, :), rhs(:, :), &
            solution(:, :), identity(:, :), fitted_row(:)
        real(dp) :: requested_alpha, requested_lambda, weight_sum, logdet
        real(dp) :: residual, quad, logdet_likelihood, logdet_prior
        integer :: n_samples, n_features, n_outputs, n_parameters, i, j, k
        logical :: intercept

        requested_alpha = 1.0_dp
        if (present(alpha)) requested_alpha = alpha
        requested_lambda = 1.0_dp
        if (present(lambda)) requested_lambda = lambda
        intercept = .true.
        if (present(fit_intercept)) intercept = fit_intercept
        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_outputs = size(y, 2)
        if (n_samples < 1 .or. n_features < 1 .or. n_outputs < 1 .or. &
                size(y, 1) /= n_samples .or. requested_alpha <= 0.0_dp .or. &
                requested_lambda <= 0.0_dp .or. any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(y)) .or. &
                .not. ieee_is_finite(requested_alpha) .or. &
                .not. ieee_is_finite(requested_lambda)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Bayesian ridge fit: invalid dimensions, hyperparameters, or data")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                    any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Bayesian ridge fit: sample weights must be finite and nonnegative")
                return
            end if
            weight_sum = sum(sample_weight)
        else
            weight_sum = real(n_samples, dp)
        end if
        if (weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Bayesian ridge fit: at least one sample weight must be positive")
            return
        end if

        call self%initialize(n_features, n_outputs, status, intercept)
        if (.not. status_ok(status)) return
        n_parameters = size(self%coefficient, 1)
        allocate(design(n_samples, n_parameters), precision(n_parameters, n_parameters))
        allocate(rhs(n_parameters, n_outputs), solution(n_parameters, n_outputs))
        allocate(identity(n_parameters, n_parameters), fitted_row(n_outputs))
        call make_design(x, intercept, design)
        precision = 0.0_dp
        rhs = 0.0_dp
        do i = 1, n_samples
            if (present(sample_weight)) then
                if (sample_weight(i) == 0.0_dp) cycle
                do j = 1, n_parameters
                    do k = 1, n_parameters
                        precision(j, k) = precision(j, k) + requested_alpha * &
                            sample_weight(i) * design(i, j) * design(i, k)
                    end do
                    rhs(j, :) = rhs(j, :) + requested_alpha * sample_weight(i) * &
                        design(i, j) * y(i, :)
                end do
            else
                do j = 1, n_parameters
                    do k = 1, n_parameters
                        precision(j, k) = precision(j, k) + requested_alpha * &
                            design(i, j) * design(i, k)
                    end do
                    rhs(j, :) = rhs(j, :) + requested_alpha * design(i, j) * y(i, :)
                end do
            end if
        end do
        do j = 1, n_parameters
            precision(j, j) = precision(j, j) + requested_lambda
        end do
        call solve_spd(precision, rhs, solution, logdet, status)
        if (.not. status_ok(status)) then
            self%fitted_value = .false.
            return
        end if
        identity = 0.0_dp
        do j = 1, n_parameters
            identity(j, j) = 1.0_dp
        end do
        call solve_spd(precision, identity, self%posterior_covariance_value, logdet, status)
        if (.not. status_ok(status)) then
            self%fitted_value = .false.
            return
        end if
        self%coefficient = solution
        self%posterior_precision_value = precision
        self%alpha_value = requested_alpha
        self%lambda_value = requested_lambda
        residual = 0.0_dp
        do i = 1, n_samples
            fitted_row = matmul(design(i, :), solution)
            if (present(sample_weight)) then
                residual = residual + sample_weight(i) * sum((y(i, :) - fitted_row)**2)
            else
                residual = residual + sum((y(i, :) - fitted_row)**2)
            end if
        end do
        quad = requested_alpha * residual + requested_lambda * sum(solution**2)
        logdet_prior = real(n_parameters, dp) * log(requested_lambda)
        logdet_likelihood = 0.0_dp
        do i = 1, n_samples
            if (present(sample_weight)) then
                if (sample_weight(i) > 0.0_dp) logdet_likelihood = logdet_likelihood + &
                    log(requested_alpha * sample_weight(i))
            else
                logdet_likelihood = logdet_likelihood + log(requested_alpha)
            end if
        end do
        self%log_evidence_value = 0.5_dp * real(n_outputs, dp) * &
            (logdet_prior - logdet + logdet_likelihood - quad - &
             real(count_positive_weights(n_samples, sample_weight), dp) * &
             log(2.0_dp * acos(-1.0_dp)))
        call status_set(status, FORTNUM_OK, "")
    end subroutine bayesian_ridge_fit_matrix

    subroutine bayesian_ridge_fit_vector(self, x, y, status, alpha, lambda, &
            fit_intercept, sample_weight)
        class(bayesian_ridge_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha, lambda, sample_weight(:)
        logical, intent(in), optional :: fit_intercept
        real(dp), allocatable :: target(:, :)
        allocate(target(size(y), 1))
        target(:, 1) = y
        call self%fit_matrix(x, target, status, alpha, lambda, fit_intercept, sample_weight)
    end subroutine bayesian_ridge_fit_vector

    subroutine bayesian_ridge_predict_matrix(self, x, y, status)
        class(bayesian_ridge_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :)
        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
                size(x, 2) /= self%feature_count_value .or. &
                any(shape(y) /= [size(x, 1), self%output_count_value]) .or. &
                any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Bayesian ridge predict: model, input, or output shape is invalid")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        call make_design(x, self%fit_intercept_value, design)
        y = matmul(design, self%coefficient)
        call status_set(status, FORTNUM_OK, "")
    end subroutine bayesian_ridge_predict_matrix

    subroutine bayesian_ridge_predict_vector(self, x, y, status)
        class(bayesian_ridge_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:, :)
        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Bayesian ridge predict: output shape is invalid")
            return
        end if
        allocate(values(size(y), 1))
        call self%predict_matrix(x, values, status)
        if (status_ok(status)) y = values(:, 1)
    end subroutine bayesian_ridge_predict_vector

    subroutine bayesian_ridge_predict_device(self, device, x, y, status)
        class(bayesian_ridge_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (device%kind == FORTML_DEVICE_CPU) then
            call self%predict_matrix(x, y, status)
        else
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "Bayesian ridge prediction: CUDA kernel is not resident")
        end if
    end subroutine bayesian_ridge_predict_device

    logical function bayesian_ridge_device_supported(self, device_kind) result(supported)
        class(bayesian_ridge_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind
        supported = device_kind == FORTML_DEVICE_CPU
    end function bayesian_ridge_device_supported

    subroutine bayesian_ridge_predict_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(bayesian_ridge_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), design_dot(:, :), coefficient_dot(:, :)
        if (.not. self%fitted_value .or. size(x, 2) /= self%feature_count_value .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                any(shape(y) /= [size(x, 1), self%output_count_value]) .or. &
                any(shape(y_dot) /= shape(y)) .or. size(theta_dot) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Bayesian ridge JVP: model, tangent, or output shape is invalid")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)), design_dot(size(x, 1), &
            size(self%coefficient, 1)), coefficient_dot(size(self%coefficient, 1), &
            size(self%coefficient, 2)))
        call make_design(x, self%fit_intercept_value, design)
        call make_tangent_design(x_dot, self%fit_intercept_value, design_dot)
        coefficient_dot = reshape(theta_dot, shape(coefficient_dot))
        y = matmul(design, self%coefficient)
        y_dot = matmul(design_dot, self%coefficient) + matmul(design, coefficient_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine bayesian_ridge_predict_jvp

    subroutine bayesian_ridge_predict_vjp(self, x, u, theta_bar, x_bar, status)
        class(bayesian_ridge_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), coefficient_bar(:, :)
        integer :: j, offset
        if (.not. self%fitted_value .or. size(x, 2) /= self%feature_count_value .or. &
                any(shape(u) /= [size(x, 1), self%output_count_value]) .or. &
                size(theta_bar) /= self%parameter_count() .or. any(shape(x_bar) /= shape(x)) .or. &
                any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(u))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Bayesian ridge VJP: model, cotangent, or output shape is invalid")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)), &
            coefficient_bar(size(self%coefficient, 1), size(self%coefficient, 2)))
        call make_design(x, self%fit_intercept_value, design)
        coefficient_bar = matmul(transpose(design), u)
        theta_bar = reshape(coefficient_bar, [size(theta_bar)])
        x_bar = 0.0_dp
        offset = merge(1, 0, self%fit_intercept_value)
        do j = 1, self%feature_count_value
            x_bar(:, j) = matmul(u, self%coefficient(offset + j, :))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bayesian_ridge_predict_vjp

    function bayesian_ridge_parameters(self) result(values)
        class(bayesian_ridge_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        if (allocated(self%coefficient)) then
            allocate(values(size(self%coefficient)))
            values = reshape(self%coefficient, [size(values)])
        else
            allocate(values(0))
        end if
    end function bayesian_ridge_parameters

    subroutine bayesian_ridge_set_parameters(self, values, status)
        class(bayesian_ridge_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. self%fitted_value .or. size(values) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Bayesian ridge set_parameters: model or packed values are invalid")
            return
        end if
        self%coefficient = reshape(values, shape(self%coefficient))
        call status_set(status, FORTNUM_OK, "")
    end subroutine bayesian_ridge_set_parameters

    function bayesian_ridge_posterior_mean(self) result(values)
        class(bayesian_ridge_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        if (allocated(self%coefficient)) then
            allocate(values, mold=self%coefficient); values = self%coefficient
        else
            allocate(values(0, 0))
        end if
    end function bayesian_ridge_posterior_mean

    function bayesian_ridge_posterior_precision(self) result(values)
        class(bayesian_ridge_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        if (allocated(self%posterior_precision_value)) then
            allocate(values, mold=self%posterior_precision_value); values = self%posterior_precision_value
        else
            allocate(values(0, 0))
        end if
    end function bayesian_ridge_posterior_precision

    function bayesian_ridge_posterior_covariance(self) result(values)
        class(bayesian_ridge_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        if (allocated(self%posterior_covariance_value)) then
            allocate(values, mold=self%posterior_covariance_value); values = self%posterior_covariance_value
        else
            allocate(values(0, 0))
        end if
    end function bayesian_ridge_posterior_covariance

    real(dp) function bayesian_ridge_alpha(self) result(value)
        class(bayesian_ridge_regression_t), intent(in) :: self; value = self%alpha_value
    end function bayesian_ridge_alpha
    real(dp) function bayesian_ridge_lambda(self) result(value)
        class(bayesian_ridge_regression_t), intent(in) :: self; value = self%lambda_value
    end function bayesian_ridge_lambda
    real(dp) function bayesian_ridge_log_evidence(self) result(value)
        class(bayesian_ridge_regression_t), intent(in) :: self; value = self%log_evidence_value
    end function bayesian_ridge_log_evidence
    integer function bayesian_ridge_feature_count(self) result(value)
        class(bayesian_ridge_regression_t), intent(in) :: self; value = self%feature_count_value
    end function bayesian_ridge_feature_count
    integer function bayesian_ridge_output_count(self) result(value)
        class(bayesian_ridge_regression_t), intent(in) :: self; value = self%output_count_value
    end function bayesian_ridge_output_count
    integer function bayesian_ridge_parameter_count(self) result(value)
        class(bayesian_ridge_regression_t), intent(in) :: self
        if (allocated(self%coefficient)) then; value = size(self%coefficient); else; value = 0; end if
    end function bayesian_ridge_parameter_count
    logical function bayesian_ridge_fit_intercept(self) result(value)
        class(bayesian_ridge_regression_t), intent(in) :: self; value = self%fit_intercept_value
    end function bayesian_ridge_fit_intercept
    logical function bayesian_ridge_fitted(self) result(value)
        class(bayesian_ridge_regression_t), intent(in) :: self; value = self%fitted_value
    end function bayesian_ridge_fitted

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

    integer function count_positive_weights(n, weights) result(npositive)
        integer, intent(in) :: n
        real(dp), intent(in), optional :: weights(:)
        if (present(weights)) then
            npositive = count(weights > 0.0_dp)
        else
            npositive = n
        end if
    end function count_positive_weights

    subroutine solve_spd(matrix, rhs, solution, logdet, status)
        real(dp), intent(in) :: matrix(:, :), rhs(:, :)
        real(dp), intent(out) :: solution(:, :)
        real(dp), intent(out) :: logdet
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: lower(:, :)
        real(dp) :: value
        integer :: n, nrhs, i, j, k
        n = size(matrix, 1); nrhs = size(rhs, 2)
        if (size(matrix, 2) /= n .or. size(rhs, 1) /= n .or. &
                any(shape(solution) /= shape(rhs))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Bayesian ridge solve: shape mismatch")
            return
        end if
        allocate(lower(n, n)); lower = 0.0_dp
        do i = 1, n
            do j = 1, i
                value = matrix(i, j)
                do k = 1, j - 1
                    value = value - lower(i, k) * lower(j, k)
                end do
                if (i == j) then
                    if (value <= 0.0_dp .or. .not. ieee_is_finite(value)) then
                        call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                            "Bayesian ridge solve: precision is not positive definite")
                        return
                    end if
                    lower(i, j) = sqrt(value)
                else
                    lower(i, j) = value / lower(j, j)
                end if
            end do
        end do
        solution = rhs
        do k = 1, nrhs
            do i = 1, n
                solution(i, k) = (solution(i, k) - &
                    sum(lower(i, 1:i-1) * solution(1:i-1, k))) / lower(i, i)
            end do
            do i = n, 1, -1
                if (i < n) then
                    solution(i, k) = (solution(i, k) - &
                        sum(lower(i+1:n, i) * solution(i+1:n, k))) / lower(i, i)
                else
                    solution(i, k) = solution(i, k) / lower(i, i)
                end if
            end do
        end do
        logdet = 0.0_dp
        do i = 1, n; logdet = logdet + 2.0_dp * log(lower(i, i)); end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine solve_spd

end module fortml_bayesian_ridge
