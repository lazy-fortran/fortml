module fortml_derivative_gaussian_process
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortml_kernels, only: kernel_t, clone_kernel_into, KERNEL_RBF, KERNEL_MATERN12, &
        KERNEL_MATERN32, KERNEL_MATERN52, KERNEL_LINEAR, KERNEL_CONSTANT, &
        KERNEL_WHITE_NOISE, KERNEL_SUM, KERNEL_PRODUCT
    implicit none
    private

    real(dp), parameter :: LOG_TWO_PI = 1.837877066409345483560659472811_dp

    type, public :: gp_derivative_regression_t
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: y_train(:, :)
        real(dp), allocatable :: alpha(:, :)
        integer, allocatable :: components(:)
        real(dp) :: noise_variance = 1.0e-6_dp
        real(dp) :: jitter = 1.0e-10_dp
        integer :: n_observations = 0
        integer :: n_features = 0
        integer :: n_outputs = 0
    contains
        procedure, public :: fit => gp_derivative_fit
        procedure, public :: predict => gp_derivative_predict
        procedure, public :: observation_count => gp_derivative_observation_count
        procedure, public :: parameter_count => gp_derivative_parameter_count
        procedure, public :: parameters => gp_derivative_parameters
        procedure, public :: set_parameters => gp_derivative_set_parameters
        procedure, public :: log_marginal_likelihood => &
            gp_derivative_log_marginal_likelihood
        procedure, public :: log_marginal_likelihood_jvp => &
            gp_derivative_log_marginal_likelihood_jvp
        procedure, public :: hyperparameter_gradient => &
            gp_derivative_hyperparameter_gradient
        procedure, public :: hyperparameter_hvp => gp_derivative_hyperparameter_hvp
    end type gp_derivative_regression_t

    public :: gp_derivative_fit
    public :: gp_derivative_predict
    public :: gp_derivative_log_marginal_likelihood
    public :: gp_derivative_hyperparameter_gradient
    public :: gp_derivative_hyperparameter_hvp

contains

    subroutine gp_derivative_fit( &
            self, x, components, y, kernel, noise_variance, status, jitter)
        class(gp_derivative_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        integer, intent(in) :: components(:)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: jitter
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(components) /= size(x, 1) .or. size(y, 1) /= size(x, 1) .or. &
            size(y, 2) < 1 .or. noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP fit: invalid data dimensions or noise variance")
            return
        end if
        if (any(components < 0) .or. &
            any(components > size(x, 2))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP fit: observation component is invalid")
            return
        end if
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP fit: kernel input dimension is invalid")
            return
        end if

        call clone_kernel_into(kernel, self%kernel)
        allocate(self%x_train, source=x)
        allocate(self%y_train, source=y)
        allocate(self%components, source=components)
        self%n_observations = size(x, 1)
        self%n_features = size(x, 2)
        self%n_outputs = size(y, 2)
        self%noise_variance = noise_variance
        self%jitter = 1.0e-10_dp
        if (present(jitter)) self%jitter = jitter
        if (self%jitter < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP fit: jitter must be nonnegative")
            return
        end if

        call derivative_gp_refactor(self, status)
    end subroutine gp_derivative_fit

    subroutine gp_derivative_predict( &
            self, x, components, mean, variance, status)
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: mean(:, :), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :)
        real(dp) :: prior
        integer :: i, j

        if (.not. derivative_gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP prediction: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(components) /= size(x, 1) .or. &
            any(components < 0) .or. any(components > self%n_features) .or. &
            any(shape(mean) /= [size(x, 1), self%n_outputs]) .or. &
            size(variance) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP prediction: output or component shape is invalid")
            return
        end if

        allocate(cross(self%n_observations, size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                call derivative_covariance( &
                    self%kernel, self%x_train(i, :), self%components(i), &
                    x(j, :), components(j), cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        mean = matmul(transpose(cross), self%alpha)
        allocate(work, source=cross)
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(x, 1)
            call derivative_covariance( &
                self%kernel, x(j, :), components(j), x(j, :), components(j), &
                prior, status)
            if (status%code /= FORTNUM_OK) return
            variance(j) = prior - dot_product(cross(:, j), work(:, j))
            if (variance(j) < 0.0_dp) then
                if (variance(j) < -1.0e-9_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "derivative GP prediction: posterior variance is not positive")
                    return
                end if
                variance(j) = 0.0_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_predict

    integer function gp_derivative_parameter_count(self) result(count)
        class(gp_derivative_regression_t), intent(in) :: self

        count = self%kernel%parameter_count() + 1
    end function gp_derivative_parameter_count

    function gp_derivative_parameters(self) result(parameters)
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: kernel_count

        kernel_count = self%kernel%parameter_count()
        allocate(parameters(kernel_count + 1))
        if (kernel_count > 0) parameters(:kernel_count) = self%kernel%parameters()
        if (.not. ieee_is_finite(self%noise_variance)) then
            parameters(kernel_count + 1) = -huge(1.0_dp)
        else if (self%noise_variance > 0.0_dp) then
            parameters(kernel_count + 1) = log(self%noise_variance)
        else
            parameters(kernel_count + 1) = -huge(1.0_dp)
        end if
    end function gp_derivative_parameters

    subroutine gp_derivative_set_parameters(self, parameters, status)
        class(gp_derivative_regression_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: kernel_count
        real(dp) :: noise_variance

        if (.not. derivative_gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP set_parameters: model is not fitted")
            return
        end if
        if (size(parameters) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP set_parameters: parameter shape or value is invalid")
            return
        end if
        kernel_count = self%kernel%parameter_count()
        noise_variance = exp(parameters(kernel_count + 1))
        if (.not. ieee_is_finite(noise_variance) .or. noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP set_parameters: noise parameter is invalid")
            return
        end if
        call self%kernel%set_parameters(parameters(:kernel_count), status)
        if (status%code /= FORTNUM_OK) return
        self%noise_variance = noise_variance
        call derivative_gp_refactor(self, status)
    end subroutine gp_derivative_set_parameters

    subroutine gp_derivative_log_marginal_likelihood(self, value, status)
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: logdet

        value = 0.0_dp
        if (.not. derivative_gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP log marginal likelihood: model is not fitted")
            return
        end if
        call self%factorization%log_determinant(logdet, status)
        if (status%code /= FORTNUM_OK) return
        value = -0.5_dp*sum(self%y_train*self%alpha) - &
            0.5_dp*real(self%n_outputs, dp)*logdet - &
            0.5_dp*real(self%n_observations*self%n_outputs, dp)*LOG_TWO_PI
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_log_marginal_likelihood

    subroutine gp_derivative_log_marginal_likelihood_jvp(self, direction, value_dot, status)
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value_dot = 0.0_dp
        if (size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP likelihood_jvp: direction shape is invalid")
            return
        end if
        allocate(gradient(size(direction)))
        call self%hyperparameter_gradient(gradient, status)
        if (status%code /= FORTNUM_OK) return
        value_dot = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_log_marginal_likelihood_jvp

    subroutine gp_derivative_hyperparameter_gradient(self, gradient, status)
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: identity(:, :), inverse(:, :), matrix_bar(:, :)
        real(dp), allocatable :: covariance_dot(:, :)
        integer :: i, j, kernel_count

        gradient = 0.0_dp
        if (.not. derivative_gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP hyperparameter_gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP hyperparameter_gradient: output shape is invalid")
            return
        end if
        allocate(identity(self%n_observations, self%n_observations))
        allocate(inverse, mold=identity)
        allocate(matrix_bar, mold=identity)
        allocate(covariance_dot, mold=identity)
        identity = 0.0_dp
        do i = 1, self%n_observations
            identity(i, i) = 1.0_dp
        end do
        inverse = identity
        call self%factorization%solve(inverse, status)
        if (status%code /= FORTNUM_OK) return
        matrix_bar = 0.5_dp*(matmul(self%alpha, transpose(self%alpha)) - &
            real(self%n_outputs, dp)*inverse)
        kernel_count = self%kernel%parameter_count()
        do j = 1, kernel_count
            call derivative_covariance_parameter_matrix(self, j, covariance_dot, status)
            if (status%code /= FORTNUM_OK) return
            gradient(j) = sum(matrix_bar*covariance_dot)
        end do
        covariance_dot = 0.0_dp
        do i = 1, self%n_observations
            covariance_dot(i, i) = self%noise_variance
        end do
        gradient(kernel_count + 1) = sum(matrix_bar*covariance_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_hyperparameter_gradient

    subroutine gp_derivative_hyperparameter_hvp(self, direction, parameter_hvp, status)
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: parameters(:), plus(:), minus(:)
        real(dp), allocatable :: gradient_plus(:), gradient_minus(:)
        real(dp) :: scale, step

        parameter_hvp = 0.0_dp
        if (.not. derivative_gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP hyperparameter_hvp: model is not fitted")
            return
        end if
        if (size(direction) /= self%parameter_count() .or. &
            size(parameter_hvp) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP hyperparameter_hvp: parameter shape is invalid")
            return
        end if
        parameters = self%parameters()
        scale = max(1.0_dp, maxval(abs(direction)))
        step = 3.0e-5_dp/scale
        plus = parameters + step*direction
        minus = parameters - step*direction
        allocate(gradient_plus(size(parameters)), gradient_minus(size(parameters)))
        call derivative_gp_gradient_at(self, plus, gradient_plus, status)
        if (status%code /= FORTNUM_OK) return
        call derivative_gp_gradient_at(self, minus, gradient_minus, status)
        if (status%code /= FORTNUM_OK) return
        parameter_hvp = (gradient_plus - gradient_minus)/(2.0_dp*step)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_hyperparameter_hvp

    integer function gp_derivative_observation_count(self) result(count)
        class(gp_derivative_regression_t), intent(in) :: self

        count = self%n_observations
    end function gp_derivative_observation_count

    subroutine derivative_gp_refactor(self, status)
        class(gp_derivative_regression_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance(:, :)
        integer :: i

        allocate(covariance(self%n_observations, self%n_observations))
        call derivative_covariance_matrix(self%kernel, self%x_train, &
            self%components, covariance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%n_observations
            covariance(i, i) = covariance(i, i) + self%noise_variance + self%jitter
        end do
        call self%factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        if (allocated(self%alpha)) deallocate(self%alpha)
        allocate(self%alpha, source=self%y_train)
        call self%factorization%solve(self%alpha, status)
    end subroutine derivative_gp_refactor

    subroutine derivative_covariance_matrix(kernel, x, components, covariance, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (size(components) /= size(x, 1) .or. &
            any(shape(covariance) /= [size(x, 1), size(x, 1)])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP covariance: shape is invalid")
            return
        end if
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                call derivative_covariance( &
                    kernel, x(i, :), components(i), x(j, :), components(j), &
                    covariance(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine derivative_covariance_matrix

    subroutine derivative_covariance_parameter_matrix(self, parameter, matrix, status)
        class(gp_derivative_regression_t), intent(in) :: self
        integer, intent(in) :: parameter
        real(dp), intent(out) :: matrix(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: covariance, covariance_dot
        integer :: i, j, kernel_count

        if (any(shape(matrix) /= [self%n_observations, self%n_observations])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP covariance parameter: output shape is invalid")
            return
        end if
        kernel_count = self%kernel%parameter_count()
        if (parameter < 1 .or. parameter > kernel_count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP covariance parameter: parameter index is invalid")
            return
        end if
        do j = 1, self%n_observations
            do i = 1, self%n_observations
                call derivative_covariance_parameter(self%kernel, self%x_train(i, :), &
                    self%components(i), self%x_train(j, :), self%components(j), &
                    parameter, covariance, covariance_dot, status)
                if (status%code /= FORTNUM_OK) return
                matrix(i, j) = covariance_dot
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine derivative_covariance_parameter_matrix

    subroutine derivative_gp_gradient_at(self, parameters, gradient, status)
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        type(gp_derivative_regression_t) :: probe
        integer :: kernel_count

        if (size(parameters) /= self%parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP gradient probe: parameter shape is invalid")
            return
        end if
        kernel_count = self%kernel%parameter_count()
        call clone_kernel_into(self%kernel, probe%kernel)
        allocate(probe%x_train, source=self%x_train)
        allocate(probe%y_train, source=self%y_train)
        allocate(probe%components, source=self%components)
        probe%n_observations = self%n_observations
        probe%n_features = self%n_features
        probe%n_outputs = self%n_outputs
        probe%jitter = self%jitter
        probe%noise_variance = exp(parameters(kernel_count + 1))
        if (.not. ieee_is_finite(probe%noise_variance) .or. &
            probe%noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP gradient probe: noise parameter is invalid")
            call release_kernel(probe%kernel)
            return
        end if
        call probe%kernel%set_parameters(parameters(:kernel_count), status)
        if (status%code /= FORTNUM_OK) then
            call release_kernel(probe%kernel)
            return
        end if
        call derivative_gp_refactor(probe, status)
        if (status%code /= FORTNUM_OK) then
            call release_kernel(probe%kernel)
            return
        end if
        call probe%hyperparameter_gradient(gradient, status)
        call release_kernel(probe%kernel)
    end subroutine derivative_gp_gradient_at

    recursive subroutine release_kernel(kernel)
        type(kernel_t), intent(inout) :: kernel

        if (associated(kernel%left)) then
            call release_kernel(kernel%left)
            deallocate(kernel%left)
        end if
        if (associated(kernel%right)) then
            call release_kernel(kernel%right)
            deallocate(kernel%right)
        end if
        if (allocated(kernel%formula)) deallocate(kernel%formula)
        if (allocated(kernel%log_parameters)) deallocate(kernel%log_parameters)
        nullify(kernel%left, kernel%right)
    end subroutine release_kernel

    subroutine derivative_covariance_parameter(kernel, x1, component1, x2, &
            component2, parameter, covariance, covariance_dot, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:)
        integer, intent(in) :: component1, component2, parameter
        real(dp), intent(out) :: covariance, covariance_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:)
        real(dp), allocatable :: mixed_hessian(:, :), gradient_x1_dot(:)
        real(dp), allocatable :: gradient_x2_dot(:), mixed_hessian_dot(:, :)
        real(dp) :: value, value_dot

        if ((component1 > 0 .or. component2 > 0) .and. &
            kernel_contains_white_noise(kernel)) then
            covariance = 0.0_dp
            covariance_dot = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel parameter JVP: white-noise derivative observations are undefined")
            return
        end if

        allocate(gradient_x1(size(x1)), gradient_x2(size(x2)))
        allocate(mixed_hessian(size(x1), size(x2)))
        allocate(gradient_x1_dot(size(x1)), gradient_x2_dot(size(x2)))
        allocate(mixed_hessian_dot(size(x1), size(x2)))
        call kernel_input_parameter_jvp(kernel, x1, x2, parameter, value, &
            gradient_x1, gradient_x2, mixed_hessian, value_dot, gradient_x1_dot, &
            gradient_x2_dot, mixed_hessian_dot, status)
        if (status%code /= FORTNUM_OK) then
            covariance = 0.0_dp
            covariance_dot = 0.0_dp
            return
        end if
        if (component1 == 0 .and. component2 == 0) then
            covariance = value
            covariance_dot = value_dot
        else if (component1 > 0 .and. component2 == 0) then
            covariance = gradient_x1(component1)
            covariance_dot = gradient_x1_dot(component1)
        else if (component1 == 0 .and. component2 > 0) then
            covariance = gradient_x2(component2)
            covariance_dot = gradient_x2_dot(component2)
        else if (component1 > 0 .and. component2 > 0) then
            covariance = mixed_hessian(component1, component2)
            covariance_dot = mixed_hessian_dot(component1, component2)
        else
            covariance = 0.0_dp
            covariance_dot = 0.0_dp
        end if
    end subroutine derivative_covariance_parameter

    recursive subroutine kernel_input_parameter_jvp(kernel, x1, x2, parameter, &
            value, gradient_x1, gradient_x2, mixed_hessian, value_dot, &
            gradient_x1_dot, gradient_x2_dot, mixed_hessian_dot, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:)
        integer, intent(in) :: parameter
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :), value_dot
        real(dp), intent(out) :: gradient_x1_dot(:), gradient_x2_dot(:)
        real(dp), intent(out) :: mixed_hessian_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: left_count
        real(dp), allocatable :: left_gradient_x1(:), right_gradient_x1(:)
        real(dp), allocatable :: left_gradient_x2(:), right_gradient_x2(:)
        real(dp), allocatable :: left_hessian(:, :), right_hessian(:, :)
        real(dp), allocatable :: left_gradient_x1_dot(:), right_gradient_x1_dot(:)
        real(dp), allocatable :: left_gradient_x2_dot(:), right_gradient_x2_dot(:)
        real(dp), allocatable :: left_hessian_dot(:, :), right_hessian_dot(:, :)
        real(dp) :: left_value, right_value, left_value_dot, right_value_dot

        value = 0.0_dp
        value_dot = 0.0_dp
        gradient_x1 = 0.0_dp
        gradient_x2 = 0.0_dp
        mixed_hessian = 0.0_dp
        gradient_x1_dot = 0.0_dp
        gradient_x2_dot = 0.0_dp
        mixed_hessian_dot = 0.0_dp
        if (parameter < 1 .or. parameter > kernel%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel input parameter JVP: parameter index is invalid")
            return
        end if

        select case (kernel%kind)
        case (KERNEL_SUM)
            left_count = kernel%left%parameter_count()
            if (parameter <= left_count) then
                allocate(right_gradient_x1(size(x1)), right_gradient_x2(size(x2)))
                allocate(right_hessian(size(x1), size(x2)))
                call kernel_input_parameter_jvp(kernel%left, x1, x2, parameter, &
                    value, gradient_x1, gradient_x2, mixed_hessian, value_dot, &
                    gradient_x1_dot, gradient_x2_dot, mixed_hessian_dot, status)
                if (status%code /= FORTNUM_OK) return
                call kernel_input_data(kernel%right, x1, x2, right_value, &
                    right_gradient_x1, right_gradient_x2, right_hessian, status)
                if (status%code /= FORTNUM_OK) return
                value = value + right_value
                gradient_x1 = gradient_x1 + right_gradient_x1
                gradient_x2 = gradient_x2 + right_gradient_x2
                mixed_hessian = mixed_hessian + right_hessian
            else
                allocate(left_gradient_x1(size(x1)), left_gradient_x2(size(x2)))
                allocate(left_hessian(size(x1), size(x2)))
                call kernel_input_data(kernel%left, x1, x2, left_value, &
                    left_gradient_x1, left_gradient_x2, left_hessian, status)
                if (status%code /= FORTNUM_OK) return
                call kernel_input_parameter_jvp(kernel%right, x1, x2, &
                    parameter - left_count, value, gradient_x1, gradient_x2, &
                    mixed_hessian, value_dot, gradient_x1_dot, gradient_x2_dot, &
                    mixed_hessian_dot, status)
                if (status%code /= FORTNUM_OK) return
                value = left_value + value
                gradient_x1 = left_gradient_x1 + gradient_x1
                gradient_x2 = left_gradient_x2 + gradient_x2
                mixed_hessian = left_hessian + mixed_hessian
            end if
            call status_set(status, FORTNUM_OK, "")
        case (KERNEL_PRODUCT)
            left_count = kernel%left%parameter_count()
            allocate(left_gradient_x1(size(x1)), right_gradient_x1(size(x1)))
            allocate(left_gradient_x2(size(x2)), right_gradient_x2(size(x2)))
            allocate(left_hessian(size(x1), size(x2)), right_hessian(size(x1), size(x2)))
            allocate(left_gradient_x1_dot(size(x1)), right_gradient_x1_dot(size(x1)))
            allocate(left_gradient_x2_dot(size(x2)), right_gradient_x2_dot(size(x2)))
            allocate(left_hessian_dot(size(x1), size(x2)))
            allocate(right_hessian_dot(size(x1), size(x2)))
            if (parameter <= left_count) then
                call kernel_input_parameter_jvp(kernel%left, x1, x2, parameter, &
                    left_value, left_gradient_x1, left_gradient_x2, left_hessian, &
                    left_value_dot, left_gradient_x1_dot, left_gradient_x2_dot, &
                    left_hessian_dot, status)
                if (status%code /= FORTNUM_OK) return
                call kernel_input_data(kernel%right, x1, x2, right_value, &
                    right_gradient_x1, right_gradient_x2, right_hessian, status)
                if (status%code /= FORTNUM_OK) return
                right_value_dot = 0.0_dp
                right_gradient_x1_dot = 0.0_dp
                right_gradient_x2_dot = 0.0_dp
                right_hessian_dot = 0.0_dp
            else
                call kernel_input_data(kernel%left, x1, x2, left_value, &
                    left_gradient_x1, left_gradient_x2, left_hessian, status)
                if (status%code /= FORTNUM_OK) return
                call kernel_input_parameter_jvp(kernel%right, x1, x2, &
                    parameter - left_count, right_value, right_gradient_x1, &
                    right_gradient_x2, right_hessian, right_value_dot, &
                    right_gradient_x1_dot, right_gradient_x2_dot, right_hessian_dot, &
                    status)
                if (status%code /= FORTNUM_OK) return
                left_value_dot = 0.0_dp
                left_gradient_x1_dot = 0.0_dp
                left_gradient_x2_dot = 0.0_dp
                left_hessian_dot = 0.0_dp
            end if
            value = left_value*right_value
            value_dot = left_value_dot*right_value + left_value*right_value_dot
            gradient_x1 = left_gradient_x1*right_value + right_gradient_x1*left_value
            gradient_x2 = left_gradient_x2*right_value + right_gradient_x2*left_value
            mixed_hessian = left_hessian*right_value + right_hessian*left_value + &
                spread(left_gradient_x1, dim=2, ncopies=size(x2))* &
                spread(right_gradient_x2, dim=1, ncopies=size(x1)) + &
                spread(right_gradient_x1, dim=2, ncopies=size(x2))* &
                spread(left_gradient_x2, dim=1, ncopies=size(x1))
            gradient_x1_dot = left_gradient_x1_dot*right_value + &
                right_gradient_x1_dot*left_value + left_gradient_x1*right_value_dot + &
                right_gradient_x1*left_value_dot
            gradient_x2_dot = left_gradient_x2_dot*right_value + &
                right_gradient_x2_dot*left_value + left_gradient_x2*right_value_dot + &
                right_gradient_x2*left_value_dot
            mixed_hessian_dot = left_hessian_dot*right_value + &
                right_hessian_dot*left_value + left_hessian*right_value_dot + &
                right_hessian*left_value_dot + &
                spread(left_gradient_x1_dot, dim=2, ncopies=size(x2))* &
                spread(right_gradient_x2, dim=1, ncopies=size(x1)) + &
                spread(left_gradient_x1, dim=2, ncopies=size(x2))* &
                spread(right_gradient_x2_dot, dim=1, ncopies=size(x1)) + &
                spread(right_gradient_x1_dot, dim=2, ncopies=size(x2))* &
                spread(left_gradient_x2, dim=1, ncopies=size(x1)) + &
                spread(right_gradient_x1, dim=2, ncopies=size(x2))* &
                spread(left_gradient_x2_dot, dim=1, ncopies=size(x1))
            call status_set(status, FORTNUM_OK, "")
        case default
            call leaf_input_parameter_jvp(kernel, x1, x2, parameter, value, &
                gradient_x1, gradient_x2, mixed_hessian, value_dot, gradient_x1_dot, &
                gradient_x2_dot, mixed_hessian_dot, status)
        end select
    end subroutine kernel_input_parameter_jvp

    recursive subroutine kernel_input_data(kernel, x1, x2, value, gradient_x1, &
            gradient_x2, mixed_hessian, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: left_gradient_x1(:), right_gradient_x1(:)
        real(dp), allocatable :: left_gradient_x2(:), right_gradient_x2(:)
        real(dp), allocatable :: left_hessian(:, :), right_hessian(:, :)
        real(dp) :: left_value, right_value

        if (kernel%kind == KERNEL_SUM .or. kernel%kind == KERNEL_PRODUCT) then
            allocate(left_gradient_x1(size(x1)), right_gradient_x1(size(x1)))
            allocate(left_gradient_x2(size(x2)), right_gradient_x2(size(x2)))
            allocate(left_hessian(size(x1), size(x2)), right_hessian(size(x1), size(x2)))
            call kernel_input_data(kernel%left, x1, x2, left_value, left_gradient_x1, &
                left_gradient_x2, left_hessian, status)
            if (status%code /= FORTNUM_OK) return
            call kernel_input_data(kernel%right, x1, x2, right_value, right_gradient_x1, &
                right_gradient_x2, right_hessian, status)
            if (status%code /= FORTNUM_OK) return
            if (kernel%kind == KERNEL_SUM) then
                value = left_value + right_value
                gradient_x1 = left_gradient_x1 + right_gradient_x1
                gradient_x2 = left_gradient_x2 + right_gradient_x2
                mixed_hessian = left_hessian + right_hessian
            else
                value = left_value*right_value
                gradient_x1 = left_gradient_x1*right_value + right_gradient_x1*left_value
                gradient_x2 = left_gradient_x2*right_value + right_gradient_x2*left_value
                mixed_hessian = left_hessian*right_value + right_hessian*left_value + &
                    spread(left_gradient_x1, dim=2, ncopies=size(x2))* &
                    spread(right_gradient_x2, dim=1, ncopies=size(x1)) + &
                    spread(right_gradient_x1, dim=2, ncopies=size(x2))* &
                    spread(left_gradient_x2, dim=1, ncopies=size(x1))
            end if
            call status_set(status, FORTNUM_OK, "")
        else if (kernel%kind == KERNEL_WHITE_NOISE) then
            value = kernel%value(x1, x2)
            gradient_x1 = 0.0_dp
            gradient_x2 = 0.0_dp
            mixed_hessian = 0.0_dp
            call status_set(status, FORTNUM_OK, "")
        else
            call kernel%input_derivatives(x1, x2, value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
        end if
    end subroutine kernel_input_data

    subroutine leaf_input_parameter_jvp(kernel, x1, x2, parameter, value, &
            gradient_x1, gradient_x2, mixed_hessian, value_dot, gradient_x1_dot, &
            gradient_x2_dot, mixed_hessian_dot, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:)
        integer, intent(in) :: parameter
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :), value_dot
        real(dp), intent(out) :: gradient_x1_dot(:), gradient_x2_dot(:)
        real(dp), intent(out) :: mixed_hessian_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: variance, lengthscale, distance, squared_distance, z
        real(dp) :: f, f_first, f_second, f_dot, f_first_dot, f_second_dot
        real(dp) :: radial_scale, radial_scale_dot, radial_coefficient
        real(dp) :: radial_coefficient_dot, difference
        real(dp) :: exponential, a
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        gradient_x1 = 0.0_dp
        gradient_x2 = 0.0_dp
        mixed_hessian = 0.0_dp
        gradient_x1_dot = 0.0_dp
        gradient_x2_dot = 0.0_dp
        mixed_hessian_dot = 0.0_dp
        if (parameter < 1 .or. parameter > kernel%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel leaf parameter JVP: parameter index is invalid")
            return
        end if

        select case (kernel%kind)
        case (KERNEL_WHITE_NOISE)
            value = kernel%value(x1, x2)
            if (parameter == 1) value_dot = value
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_LINEAR)
            variance = exp(kernel%log_parameters(1))
            value = variance*dot_product(x1, x2)
            gradient_x1 = variance*x2
            gradient_x2 = variance*x1
            do i = 1, size(x1)
                mixed_hessian(i, i) = variance
            end do
            if (parameter == 1) then
                value_dot = value
                gradient_x1_dot = gradient_x1
                gradient_x2_dot = gradient_x2
                mixed_hessian_dot = mixed_hessian
            end if
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_CONSTANT)
            variance = exp(kernel%log_parameters(1))
            value = variance
            if (parameter == 1) value_dot = value
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_RBF, KERNEL_MATERN12, KERNEL_MATERN32, KERNEL_MATERN52)
            variance = exp(kernel%log_parameters(1))
            lengthscale = exp(kernel%log_parameters(2))
            squared_distance = sum((x1 - x2)**2)
            distance = sqrt(squared_distance)
            if (kernel%kind == KERNEL_MATERN12 .and. distance == 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel leaf parameter JVP: Matern 1/2 coincident derivative is undefined")
                return
            end if
            z = distance/lengthscale
            if (kernel%kind == KERNEL_RBF) then
                exponential = exp(-0.5_dp*z*z)
                f = variance*exponential
                f_first = -f*distance/(lengthscale*lengthscale)
                f_second = f*(z*z - 1.0_dp)/(lengthscale*lengthscale)
                if (parameter == 1) then
                    f_dot = f
                    f_first_dot = f_first
                    f_second_dot = f_second
                else
                    f_dot = f*z*z
                    f_first_dot = f_first*(z*z - 2.0_dp)
                    f_second_dot = f*(z**4 - 5.0_dp*z*z + 2.0_dp) / &
                        (lengthscale*lengthscale)
                end if
            else if (kernel%kind == KERNEL_MATERN12) then
                exponential = exp(-z)
                f = variance*exponential
                f_first = -f/lengthscale
                f_second = f/(lengthscale*lengthscale)
                if (parameter == 1) then
                    f_dot = f
                    f_first_dot = f_first
                    f_second_dot = f_second
                else
                    f_dot = f*z
                    f_first_dot = f_first*(z - 1.0_dp)
                    f_second_dot = f_second*(z - 2.0_dp)
                end if
            else if (kernel%kind == KERNEL_MATERN32) then
                a = sqrt(3.0_dp)
                exponential = exp(-a*z)
                f = variance*(1.0_dp + a*z)*exponential
                f_first = -3.0_dp*variance*z*exponential/lengthscale
                f_second = 3.0_dp*variance*exponential*(a*z - 1.0_dp) / &
                    (lengthscale*lengthscale)
                if (parameter == 1) then
                    f_dot = f
                    f_first_dot = f_first
                    f_second_dot = f_second
                else
                    f_dot = 3.0_dp*variance*z*z*exponential
                    f_first_dot = f_first*(a*z - 2.0_dp)
                    f_second_dot = 3.0_dp*variance*exponential*( &
                        3.0_dp*z*z - 4.0_dp*a*z + 2.0_dp) / &
                        (lengthscale*lengthscale)
                end if
            else
                a = sqrt(5.0_dp)
                exponential = exp(-a*z)
                f = variance*(1.0_dp + a*z + 5.0_dp*z*z/3.0_dp)*exponential
                f_first = -(5.0_dp/3.0_dp)*variance*z*(1.0_dp + a*z)* &
                    exponential/lengthscale
                f_second = (5.0_dp/3.0_dp)*variance*exponential*( &
                    5.0_dp*z*z - a*z - 1.0_dp)/(lengthscale*lengthscale)
                if (parameter == 1) then
                    f_dot = f
                    f_first_dot = f_first
                    f_second_dot = f_second
                else
                    f_dot = (5.0_dp/3.0_dp)*variance*z*z*(1.0_dp + a*z)*exponential
                    f_first_dot = (5.0_dp/3.0_dp)*variance*exponential/lengthscale*( &
                        2.0_dp*z + 2.0_dp*a*z*z - 5.0_dp*z**3)
                    f_second_dot = (5.0_dp/3.0_dp)*variance*exponential/ &
                        (lengthscale*lengthscale)*( &
                        5.0_dp*a*z**3 - 25.0_dp*z*z + 2.0_dp*a*z + 2.0_dp)
                end if
            end if
            value = f
            value_dot = f_dot
            if (distance == 0.0_dp) then
                do i = 1, size(x1)
                    mixed_hessian(i, i) = -f_second
                    mixed_hessian_dot(i, i) = -f_second_dot
                end do
            else
                radial_scale = f_first/distance
                radial_scale_dot = f_first_dot/distance
                radial_coefficient = (f_second - radial_scale)/squared_distance
                radial_coefficient_dot = (f_second_dot - radial_scale_dot)/squared_distance
                do i = 1, size(x1)
                    difference = x1(i) - x2(i)
                    gradient_x1(i) = radial_scale*difference
                    gradient_x2(i) = -gradient_x1(i)
                    gradient_x1_dot(i) = radial_scale_dot*difference
                    gradient_x2_dot(i) = -gradient_x1_dot(i)
                    do j = 1, size(x2)
                        mixed_hessian(i, j) = -(radial_scale*merge(1.0_dp, 0.0_dp, i == j) + &
                            radial_coefficient*(x1(i) - x2(i))*(x1(j) - x2(j)))
                        mixed_hessian_dot(i, j) = -(radial_scale_dot* &
                            merge(1.0_dp, 0.0_dp, i == j) + radial_coefficient_dot* &
                            (x1(i) - x2(i))*(x1(j) - x2(j)))
                    end do
                end do
            end if
            call status_set(status, FORTNUM_OK, "")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel leaf parameter JVP: input derivatives are unsupported")
        end select
    end subroutine leaf_input_parameter_jvp

    subroutine derivative_covariance( &
            kernel, x1, component1, x2, component2, covariance, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:)
        integer, intent(in) :: component1, component2
        real(dp), intent(out) :: covariance
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:)
        real(dp), allocatable :: mixed_hessian(:, :)
        real(dp) :: value

        if (component1 == 0 .and. component2 == 0) then
            covariance = kernel%value(x1, x2)
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (kernel_contains_white_noise(kernel)) then
            covariance = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "white-noise derivative observations are undefined")
            return
        end if
        allocate(gradient_x1(size(x1)), gradient_x2(size(x2)))
        allocate(mixed_hessian(size(x1), size(x2)))
        call kernel%input_derivatives( &
            x1, x2, value, gradient_x1, gradient_x2, mixed_hessian, status)
        if (status%code /= FORTNUM_OK) then
            covariance = 0.0_dp
            return
        end if
        if (component1 == 0 .and. component2 == 0) then
            covariance = value
        else if (component1 > 0 .and. component2 == 0) then
            covariance = gradient_x1(component1)
        else if (component1 == 0 .and. component2 > 0) then
            covariance = gradient_x2(component2)
        else
            covariance = mixed_hessian(component1, component2)
        end if
    end subroutine derivative_covariance

    recursive logical function kernel_contains_white_noise(kernel) result(found)
        type(kernel_t), intent(in) :: kernel

        found = kernel%kind == KERNEL_WHITE_NOISE
        if (found) return
        if (associated(kernel%left)) found = kernel_contains_white_noise(kernel%left)
        if (found) return
        if (associated(kernel%right)) found = kernel_contains_white_noise(kernel%right)
    end function kernel_contains_white_noise

    logical function derivative_gp_fitted(self) result(fitted)
        class(gp_derivative_regression_t), intent(in) :: self

        fitted = allocated(self%x_train) .and. allocated(self%y_train) .and. &
            allocated(self%alpha) .and. allocated(self%components)
        if (.not. fitted) return
        fitted = self%n_observations > 0 .and. self%n_features > 0 .and. &
            self%n_outputs > 0 .and. size(self%components) == self%n_observations
    end function derivative_gp_fitted

end module fortml_derivative_gaussian_process
