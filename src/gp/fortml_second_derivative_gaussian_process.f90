module fortml_second_derivative_gaussian_process
    !! Scalar smooth-kernel GP with value, first-, and second-derivative observations.
    !!
    !! `gp_derivative_regression_t` deliberately stops at first derivatives in
    !! its component encoding.  This bounded companion uses an explicit order
    !! vector (`0`, `1`, or `2`) for a one-dimensional RBF or Matérn-5/2
    !! kernel.  Covariance blocks are generated from exact derivatives through
    !! order four, so mixed value/gradient/Hessian observations and predictions
    !! share one Cholesky state.  Query input JVP/VJP products use the fifth
    !! derivative.  The Matérn-5/2 fifth derivative is discontinuous at
    !! coincident inputs, so that boundary is a typed refusal.  RBF and
    !! Matérn-5/2 likelihood gradients and HVPs are analytic; non-RBF/Matérn
    !! leaves remain explicit boundaries rather than hidden finite-difference
    !! fallbacks.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, KERNEL_RBF, KERNEL_MATERN52
    implicit none
    private

    real(dp), parameter :: MIN_VARIANCE = -1.0e-9_dp
    real(dp), parameter :: LOG_TWO_PI = 1.837877066409345483560659472811_dp

    type, public :: second_derivative_gp_t
        private
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: x_train(:), y_train(:), alpha(:)
        integer, allocatable :: orders(:)
        real(dp) :: noise_variance = 1.0e-6_dp
        real(dp) :: jitter = 1.0e-10_dp
        real(dp) :: kernel_variance = 1.0_dp
        real(dp) :: lengthscale = 1.0_dp
        integer :: n_observations = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => second_derivative_fit
        procedure, public :: predict => second_derivative_predict
        procedure, public :: joint_covariance => second_derivative_joint_covariance
        procedure, public :: predict_input_jvp => second_derivative_predict_input_jvp
        procedure, public :: predict_input_vjp => second_derivative_predict_input_vjp
        procedure, public :: predict_input_jvp_device => &
            second_derivative_predict_input_jvp_device
        procedure, public :: predict_input_vjp_device => &
            second_derivative_predict_input_vjp_device
        procedure, public :: predict_device => second_derivative_predict_device
        procedure, public :: joint_covariance_device => &
            second_derivative_joint_covariance_device
        procedure, public :: device_supported => second_derivative_device_supported
        procedure, public :: observation_count => second_derivative_observation_count
        procedure, public :: parameters => second_derivative_parameters
        procedure, public :: set_parameters => second_derivative_set_parameters
        procedure, public :: log_marginal_likelihood => &
            second_derivative_log_marginal_likelihood
        procedure, public :: log_marginal_likelihood_jvp => &
            second_derivative_log_marginal_likelihood_jvp
        procedure, public :: log_marginal_likelihood_vjp => &
            second_derivative_log_marginal_likelihood_vjp
        procedure, public :: hyperparameter_gradient => &
            second_derivative_hyperparameter_gradient
        procedure, public :: hyperparameter_vjp => second_derivative_hyperparameter_vjp
        procedure, public :: hyperparameter_hvp => second_derivative_hyperparameter_hvp
        procedure, public :: parameter_count => second_derivative_parameter_count
        procedure, public :: fitted => second_derivative_fitted
    end type second_derivative_gp_t

contains

    subroutine second_derivative_fit(self, x, orders, y, kernel, noise_variance, status, jitter)
        class(second_derivative_gp_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        integer, intent(in) :: orders(:)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: jitter
        real(dp), allocatable :: covariance(:, :), parameters(:)
        integer :: i, j, n

        n = size(x, 1)
        if (n < 1 .or. size(x, 2) /= 1 .or. size(y) /= n .or. &
            size(orders) /= n .or. noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP fit: input or noise shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y)) .or. &
            any(orders < 0)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP fit: values or derivative orders are invalid")
            return
        end if
        if ((kernel%kind /= KERNEL_RBF .and. kernel%kind /= KERNEL_MATERN52) .or. &
            kernel%input_dim /= 1 .or. kernel%parameter_count() /= 2) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP fit: only one-dimensional RBF or Matern-5/2 is generated")
            return
        end if
        if (kernel%kind == KERNEL_RBF) then
            if (any(orders > 3)) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "second-derivative GP fit: RBF order is limited to three")
                return
            end if
        else
            if (any(orders > 2)) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "second-derivative GP fit: Matern-5/2 order is limited to two")
                return
            end if
        end if
        if (present(jitter)) then
            if (jitter < 0.0_dp .or. .not. ieee_is_finite(jitter)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "second-derivative GP fit: jitter is invalid")
                return
            end if
            self%jitter = jitter
        end if
        parameters = kernel%parameters()
        self%kernel = kernel
        self%kernel_variance = exp(parameters(1))
        self%lengthscale = exp(parameters(2))
        self%noise_variance = noise_variance
        self%n_observations = n
        allocate(self%x_train(n), self%y_train(n), self%alpha(n), self%orders(n))
        self%x_train = x(:, 1)
        self%y_train = y
        self%orders = orders
        allocate(covariance(n, n))
        do j = 1, n
            do i = 1, n
                covariance(i, j) = second_derivative_covariance(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), orders(i), &
                    self%x_train(j), orders(j))
            end do
            covariance(j, j) = covariance(j, j) + noise_variance + self%jitter
        end do
        call factorize_state(self, covariance, status)
        if (status%code /= FORTNUM_OK) then
            self%is_fitted = .false.
            return
        end if
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_fit

    subroutine second_derivative_predict(self, x, orders, mean, variance, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: orders(:)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :)
        real(dp) :: prior
        integer :: i, j, n

        if (.not. valid_query(self, x, orders, mean, variance, status)) return
        n = size(x, 1)
        allocate(cross(self%n_observations, n), work(self%n_observations, n))
        do j = 1, n
            do i = 1, self%n_observations
                cross(i, j) = second_derivative_covariance(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), &
                    self%orders(i), x(j, 1), orders(j))
            end do
        end do
        mean = matmul(transpose(cross), self%alpha)
        work = cross
        call solve_factor_matrix(self, work, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, n
            prior = second_derivative_covariance(self%kernel%kind, self%kernel_variance, &
                self%lengthscale, x(j, 1), orders(j), x(j, 1), orders(j))
            variance(j) = prior - dot_product(cross(:, j), work(:, j))
            if (variance(j) < MIN_VARIANCE) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "second-derivative GP prediction: posterior variance is not positive")
                return
            end if
            if (variance(j) < 0.0_dp) variance(j) = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_predict

    subroutine second_derivative_joint_covariance(self, x, orders, covariance, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: orders(:)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior(:, :), work(:, :)
        integer :: i, j, n

        if (.not. valid_joint_query(self, x, orders, covariance, status)) return
        n = size(x, 1)
        allocate(cross(self%n_observations, n), prior(n, n), work(self%n_observations, n))
        do j = 1, n
            do i = 1, self%n_observations
                cross(i, j) = second_derivative_covariance(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), &
                    self%orders(i), x(j, 1), orders(j))
            end do
            do i = 1, n
                prior(i, j) = second_derivative_covariance(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, x(i, 1), orders(i), &
                    x(j, 1), orders(j))
            end do
        end do
        work = cross
        call solve_factor_matrix(self, work, status)
        if (status%code /= FORTNUM_OK) return
        covariance = prior - matmul(transpose(cross), work)
        covariance = 0.5_dp*(covariance + transpose(covariance))
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_joint_covariance

    subroutine second_derivative_predict_input_jvp(self, x, orders, direction, mean, &
            mean_dot, variance, variance_dot, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        integer, intent(in) :: orders(:)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), work(:, :), work_dot(:, :)
        real(dp) :: prior
        integer :: i, j

        if (.not. valid_query(self, x, orders, mean, variance, status)) return
        if (size(direction) /= size(x, 1) .or. any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP input JVP: direction shape is invalid")
            return
        end if
        if (.not. input_jvp_supported(self, x, orders)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP input JVP: Matern-5/2 fifth derivative is undefined at coincidence")
            return
        end if
        allocate(cross(self%n_observations, size(x, 1)), cross_dot(self%n_observations, size(x, 1)), &
            work(self%n_observations, size(x, 1)), work_dot(self%n_observations, size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                cross(i, j) = second_derivative_covariance(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), &
                    self%orders(i), x(j, 1), orders(j))
                cross_dot(i, j) = second_derivative_covariance_input_dot(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), self%orders(i), &
                    x(j, 1), orders(j), direction(j))
            end do
        end do
        mean = matmul(transpose(cross), self%alpha)
        mean_dot = matmul(transpose(cross_dot), self%alpha)
        work = cross
        work_dot = cross_dot
        call solve_factor_matrix(self, work, status)
        if (status%code /= FORTNUM_OK) return
        call solve_factor_matrix(self, work_dot, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(x, 1)
            prior = second_derivative_covariance(self%kernel%kind, self%kernel_variance, &
                self%lengthscale, x(j, 1), orders(j), x(j, 1), orders(j))
            variance(j) = prior - dot_product(cross(:, j), work(:, j))
            variance_dot(j) = -dot_product(cross_dot(:, j), work(:, j)) - &
                dot_product(cross(:, j), work_dot(:, j))
            if (variance(j) < MIN_VARIANCE) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "second-derivative GP input JVP: posterior variance is not positive")
                return
            end if
            if (variance(j) < 0.0_dp) variance(j) = 0.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_predict_input_jvp

    subroutine second_derivative_predict_input_vjp(self, x, orders, mean_bar, variance_bar, &
            x_bar, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        integer, intent(in) :: orders(:)
        real(dp), intent(out) :: x_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :)
        real(dp) :: cross_bar, derivative
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. valid_query(self, x, orders, mean_bar, variance_bar, status)) return
        if (size(x_bar) /= size(x, 1) .or. any(.not. ieee_is_finite(mean_bar)) .or. &
            any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP input VJP: cotangent shape is invalid")
            return
        end if
        if (.not. input_jvp_supported(self, x, orders)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP input VJP: Matern-5/2 fifth derivative is undefined at coincidence")
            return
        end if
        allocate(cross(self%n_observations, size(x, 1)), work(self%n_observations, size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                cross(i, j) = second_derivative_covariance(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), &
                    self%orders(i), x(j, 1), orders(j))
            end do
        end do
        work = cross
        call solve_factor_matrix(self, work, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                cross_bar = self%alpha(i)*mean_bar(j) - &
                    2.0_dp*work(i, j)*variance_bar(j)
                derivative = second_derivative_covariance_input_dot(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), self%orders(i), &
                    x(j, 1), orders(j), 1.0_dp)
                x_bar(j) = x_bar(j) + cross_bar*derivative
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_predict_input_vjp

    subroutine second_derivative_predict_input_jvp_device(self, device, x, orders, direction, &
            mean, mean_dot, variance, variance_dot, status)
        class(second_derivative_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), direction(:)
        integer, intent(in) :: orders(:)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP input JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_input_jvp(x, orders, direction, mean, mean_dot, variance, &
                variance_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP input JVP device: resident RBF/Matern-5/2 order-two kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP input JVP device: device kind is invalid")
        end select
    end subroutine second_derivative_predict_input_jvp_device

    subroutine second_derivative_predict_input_vjp_device(self, device, x, orders, mean_bar, &
            variance_bar, x_bar, status)
        class(second_derivative_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        integer, intent(in) :: orders(:)
        real(dp), intent(out) :: x_bar(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP input VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_input_vjp(x, orders, mean_bar, variance_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP input VJP device: resident RBF/Matern-5/2 order-two kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP input VJP device: device kind is invalid")
        end select
    end subroutine second_derivative_predict_input_vjp_device

    subroutine second_derivative_predict_device(self, device, x, orders, mean, variance, status)
        class(second_derivative_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: orders(:)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, orders, mean, variance, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP device prediction: resident RBF/Matern-5/2 order-two kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP device prediction: device kind is invalid")
        end select
    end subroutine second_derivative_predict_device

    subroutine second_derivative_joint_covariance_device(self, device, x, orders, covariance, status)
        class(second_derivative_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: orders(:)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP covariance device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%joint_covariance(x, orders, covariance, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP covariance device: resident RBF/Matern-5/2 order-two kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP covariance device: device kind is invalid")
        end select
    end subroutine second_derivative_joint_covariance_device

    logical function second_derivative_device_supported(self, device_kind) result(value)
        class(second_derivative_gp_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            value = self%is_fitted
        case default
            value = .false.
        end select
    end function second_derivative_device_supported

    integer function second_derivative_observation_count(self) result(value)
        class(second_derivative_gp_t), intent(in) :: self

        value = self%n_observations
    end function second_derivative_observation_count

    integer function second_derivative_parameter_count(self) result(value)
        class(second_derivative_gp_t), intent(in) :: self

        value = 0
        if (self%is_fitted) value = 3
    end function second_derivative_parameter_count

    function second_derivative_parameters(self) result(parameters)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(self%parameter_count()))
        if (.not. self%is_fitted) return
        parameters = [log(self%kernel_variance), log(self%lengthscale), log(self%noise_variance)]
    end function second_derivative_parameters

    subroutine second_derivative_set_parameters(self, parameters, status)
        class(second_derivative_gp_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: old_parameters(:), covariance(:, :)
        real(dp) :: old_noise, noise_variance
        integer :: old_code
        character(120) :: old_message

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP set_parameters: model is not fitted")
            return
        end if
        if (size(parameters) /= 3 .or. any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP set_parameters: parameter shape or value is invalid")
            return
        end if
        noise_variance = exp(parameters(3))
        if (.not. ieee_is_finite(noise_variance) .or. noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP set_parameters: noise parameter is invalid")
            return
        end if

        old_parameters = self%kernel%parameters()
        old_noise = self%noise_variance
        call self%kernel%set_parameters(parameters(:2), status)
        if (status%code /= FORTNUM_OK) return
        self%kernel_variance = exp(parameters(1))
        self%lengthscale = exp(parameters(2))
        self%noise_variance = noise_variance
        allocate(covariance(self%n_observations, self%n_observations))
        call second_derivative_build_covariance(self, covariance, status)
        if (status%code == FORTNUM_OK) call factorize_state(self, covariance, status)
        if (status%code == FORTNUM_OK) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        ! Refactor the previous state before returning the original failure.
        old_code = status%code
        old_message = status%msg
        call self%kernel%set_parameters(old_parameters, status)
        self%kernel_variance = exp(old_parameters(1))
        self%lengthscale = exp(old_parameters(2))
        self%noise_variance = old_noise
        call second_derivative_build_covariance(self, covariance, status)
        if (status%code == FORTNUM_OK) call factorize_state(self, covariance, status)
        call status_set(status, old_code, old_message)
    end subroutine second_derivative_set_parameters

    subroutine second_derivative_log_marginal_likelihood(self, value, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: logdet

        value = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP log marginal likelihood: model is not fitted")
            return
        end if
        call self%factorization%log_determinant(logdet, status)
        if (status%code /= FORTNUM_OK) return
        value = -0.5_dp*dot_product(self%y_train, self%alpha) - 0.5_dp*logdet - &
            0.5_dp*real(self%n_observations, dp)*LOG_TWO_PI
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_log_marginal_likelihood

    subroutine second_derivative_log_marginal_likelihood_jvp(self, direction, value_dot, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value_dot = 0.0_dp
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP likelihood JVP: direction shape is invalid")
            return
        end if
        allocate(gradient(size(direction)))
        call self%hyperparameter_gradient(gradient, status)
        if (status%code /= FORTNUM_OK) return
        value_dot = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_log_marginal_likelihood_jvp

    subroutine second_derivative_log_marginal_likelihood_vjp(self, value_bar, parameter_bar, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        parameter_bar = 0.0_dp
        if (size(parameter_bar) /= self%parameter_count() .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP likelihood VJP: cotangent or shape is invalid")
            return
        end if
        allocate(gradient(size(parameter_bar)))
        call self%hyperparameter_gradient(gradient, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = value_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_log_marginal_likelihood_vjp

    subroutine second_derivative_hyperparameter_vjp(self, value_bar, parameter_bar, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        call second_derivative_log_marginal_likelihood_vjp(self, value_bar, parameter_bar, status)
    end subroutine second_derivative_hyperparameter_vjp

    subroutine second_derivative_hyperparameter_gradient(self, gradient, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: identity(:, :), inverse(:, :), matrix_bar(:, :), matrix(:, :)
        integer :: i, j, n

        gradient = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP hyperparameter_gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP hyperparameter_gradient: output shape is invalid")
            return
        end if
        if (self%kernel%kind /= KERNEL_RBF .and. self%kernel%kind /= KERNEL_MATERN52) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP hyperparameter_gradient: kernel parameter products are not generated")
            return
        end if
        n = self%n_observations
        allocate(identity(n, n), inverse(n, n), matrix_bar(n, n), matrix(n, n))
        identity = 0.0_dp
        do i = 1, n
            identity(i, i) = 1.0_dp
        end do
        inverse = identity
        call self%factorization%solve(inverse, status)
        if (status%code /= FORTNUM_OK) return
        matrix_bar = 0.5_dp*(outer_product(self%alpha, self%alpha) - inverse)
        do j = 1, 2
            call second_derivative_covariance_parameter_matrix(self, j, matrix, status)
            if (status%code /= FORTNUM_OK) return
            gradient(j) = sum(matrix_bar*matrix)
        end do
        gradient(3) = 0.0_dp
        do i = 1, n
            gradient(3) = gradient(3) + self%noise_variance*matrix_bar(i, i)
        end do
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "second-derivative GP hyperparameter_gradient: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_hyperparameter_gradient

    subroutine second_derivative_hyperparameter_hvp(self, direction, parameter_hvp, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance(:, :), covariance_dot(:, :), alpha_dot(:)
        real(dp), allocatable :: identity(:, :), inverse(:, :), inverse_dot(:, :)
        real(dp), allocatable :: matrix_bar(:, :), matrix_bar_dot(:, :)
        real(dp), allocatable :: parameter_matrix(:, :), parameter_matrix_dot(:, :)
        real(dp) :: noise_dot, trace_bar, trace_bar_dot
        integer :: i, j, k, n

        parameter_hvp = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP hyperparameter_hvp: model is not fitted")
            return
        end if
        if (size(direction) /= self%parameter_count() .or. size(parameter_hvp) /= &
            self%parameter_count() .or. any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP hyperparameter_hvp: parameter shape is invalid")
            return
        end if
        if (self%kernel%kind /= KERNEL_RBF .and. self%kernel%kind /= KERNEL_MATERN52) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP hyperparameter_hvp: kernel parameter products are not generated")
            return
        end if
        n = self%n_observations
        allocate(covariance(n, n), covariance_dot(n, n), alpha_dot(n))
        allocate(identity(n, n), inverse(n, n), inverse_dot(n, n))
        allocate(matrix_bar(n, n), matrix_bar_dot(n, n))
        allocate(parameter_matrix(n, n), parameter_matrix_dot(n, n))
        call second_derivative_build_covariance(self, covariance, status)
        if (status%code /= FORTNUM_OK) return
        covariance_dot = 0.0_dp
        do j = 1, n
            do i = 1, n
                covariance_dot(i, j) = direction(1)* &
                    second_derivative_covariance_parameter(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), self%orders(i), &
                    self%x_train(j), self%orders(j), 1) + direction(2)* &
                    second_derivative_covariance_parameter(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), self%orders(i), &
                    self%x_train(j), self%orders(j), 2)
            end do
        end do
        noise_dot = self%noise_variance*direction(3)
        do i = 1, n
            covariance_dot(i, i) = covariance_dot(i, i) + noise_dot
        end do
        alpha_dot = -matmul(covariance_dot, self%alpha)
        call self%factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return
        identity = 0.0_dp
        do i = 1, n
            identity(i, i) = 1.0_dp
        end do
        inverse = identity
        call self%factorization%solve(inverse, status)
        if (status%code /= FORTNUM_OK) return
        inverse_dot = -matmul(inverse, matmul(covariance_dot, inverse))
        matrix_bar = 0.5_dp*(outer_product(self%alpha, self%alpha) - inverse)
        matrix_bar_dot = 0.5_dp*(outer_product(alpha_dot, self%alpha) + &
            outer_product(self%alpha, alpha_dot) - inverse_dot)
        do j = 1, 2
            call second_derivative_covariance_parameter_matrix(self, j, parameter_matrix, status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, n
                do k = 1, n
                    parameter_matrix_dot(i, k) = &
                        second_derivative_covariance_parameter_dot(self%kernel%kind, &
                        self%kernel_variance, self%lengthscale, self%x_train(i), self%orders(i), &
                        self%x_train(k), self%orders(k), j, direction(:2))
                end do
            end do
            parameter_hvp(j) = sum(matrix_bar_dot*parameter_matrix) + &
                sum(matrix_bar*parameter_matrix_dot)
        end do
        trace_bar = 0.0_dp
        trace_bar_dot = 0.0_dp
        do i = 1, n
            trace_bar = trace_bar + matrix_bar(i, i)
            trace_bar_dot = trace_bar_dot + matrix_bar_dot(i, i)
        end do
        parameter_hvp(3) = self%noise_variance*(direction(3)*trace_bar + trace_bar_dot)
        if (any(.not. ieee_is_finite(parameter_hvp))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "second-derivative GP hyperparameter_hvp: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_hyperparameter_hvp

    logical function second_derivative_fitted(self) result(value)
        class(second_derivative_gp_t), intent(in) :: self

        value = self%is_fitted
    end function second_derivative_fitted

    logical function valid_query(self, x, orders, mean, variance, status) result(value)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean(:), variance(:)
        integer, intent(in) :: orders(:)
        type(fortnum_status_t), intent(out) :: status

        value = .false.
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP query: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= 1 .or. size(orders) /= size(x, 1) .or. &
            size(mean) /= size(x, 1) .or. size(variance) /= size(x, 1) .or. &
            any(orders < 0) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP query: shape or order is invalid")
            return
        end if
        if (.not. query_orders_supported(self, orders)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP query: requested derivative order is not generated")
            return
        end if
        value = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_query

    logical function valid_joint_query(self, x, orders, covariance, status) result(value)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), covariance(:, :)
        integer, intent(in) :: orders(:)
        type(fortnum_status_t), intent(out) :: status

        value = .false.
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP covariance: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= 1 .or. size(orders) /= size(x, 1) .or. &
            any(orders < 0) .or. any(shape(covariance) /= &
            [size(x, 1), size(x, 1)]) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP covariance: shape or order is invalid")
            return
        end if
        if (.not. query_orders_supported(self, orders)) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP covariance: requested derivative order is not generated")
            return
        end if
        value = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_joint_query

    logical function query_orders_supported(self, orders) result(value)
        class(second_derivative_gp_t), intent(in) :: self
        integer, intent(in) :: orders(:)

        if (self%kernel%kind == KERNEL_RBF) then
            value = .not. any(orders > 3)
        else
            value = .not. any(orders > 2)
        end if
    end function query_orders_supported

    subroutine factorize_state(self, covariance, status)
        class(second_derivative_gp_t), intent(inout) :: self
        real(dp), intent(in) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        call self%factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        self%alpha = self%y_train
        call self%factorization%solve(self%alpha, status)
    end subroutine factorize_state

    subroutine second_derivative_build_covariance(self, covariance, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (any(shape(covariance) /= [self%n_observations, self%n_observations])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP covariance rebuild: output shape is invalid")
            return
        end if
        do j = 1, self%n_observations
            do i = 1, self%n_observations
                covariance(i, j) = second_derivative_covariance(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), self%orders(i), &
                    self%x_train(j), self%orders(j))
            end do
            covariance(j, j) = covariance(j, j) + self%noise_variance + self%jitter
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_build_covariance

    pure function outer_product(left, right) result(value)
        real(dp), intent(in) :: left(:), right(:)
        real(dp) :: value(size(left), size(right))

        value = spread(left, 2, size(right))*spread(right, 1, size(left))
    end function outer_product

    subroutine second_derivative_covariance_parameter_matrix(self, parameter, matrix, status)
        class(second_derivative_gp_t), intent(in) :: self
        integer, intent(in) :: parameter
        real(dp), intent(out) :: matrix(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (any(shape(matrix) /= [self%n_observations, self%n_observations]) .or. &
            parameter < 1 .or. parameter > 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP covariance parameter: shape or index is invalid")
            return
        end if
        do j = 1, self%n_observations
            do i = 1, self%n_observations
                matrix(i, j) = second_derivative_covariance_parameter(self%kernel%kind, &
                    self%kernel_variance, self%lengthscale, self%x_train(i), self%orders(i), &
                    self%x_train(j), self%orders(j), parameter)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_covariance_parameter_matrix

    subroutine solve_factor_matrix(self, matrix, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(inout) :: matrix(:, :)
        type(fortnum_status_t), intent(out) :: status

        call self%factorization%solve(matrix, status)
    end subroutine solve_factor_matrix

    pure real(dp) function second_derivative_covariance(kind, variance, lengthscale, x1, &
            order1, x2, order2) result(value)
        integer, intent(in) :: kind, order1, order2
        real(dp), intent(in) :: variance, lengthscale, x1, x2

        select case (kind)
        case (KERNEL_RBF)
            value = rbf_derivative_covariance(variance, lengthscale, x1, order1, x2, order2)
        case (KERNEL_MATERN52)
            value = matern52_derivative_covariance(variance, lengthscale, x1, order1, x2, order2)
        case default
            value = 0.0_dp
        end select
    end function second_derivative_covariance

    pure real(dp) function second_derivative_covariance_input_dot(kind, variance, lengthscale, &
            x1, order1, x2, order2, direction) result(value)
        integer, intent(in) :: kind, order1, order2
        real(dp), intent(in) :: variance, lengthscale, x1, x2, direction

        select case (kind)
        case (KERNEL_RBF)
            value = rbf_derivative_covariance_input_dot(variance, lengthscale, x1, order1, &
                x2, order2, direction)
        case (KERNEL_MATERN52)
            value = direction*matern52_derivative_covariance(variance, lengthscale, x1, order1, &
                x2, order2 + 1)
        case default
            value = 0.0_dp
        end select
    end function second_derivative_covariance_input_dot

    pure real(dp) function second_derivative_covariance_parameter(kind, variance, lengthscale, &
            x1, order1, x2, order2, parameter) result(value)
        integer, intent(in) :: kind, order1, order2, parameter
        real(dp), intent(in) :: variance, lengthscale, x1, x2

        value = 0.0_dp
        if (kind == KERNEL_RBF) then
            if (parameter == 1) then
                value = rbf_derivative_covariance(variance, lengthscale, x1, order1, x2, order2)
            else if (parameter == 2) then
                value = (-1.0_dp)**order2*rbf_log_lengthscale_derivative(variance, lengthscale, &
                    x1 - x2, order1 + order2)
            end if
        else if (kind == KERNEL_MATERN52) then
            value = matern52_parameter_derivative(variance, lengthscale, x1, order1, x2, &
                order2, parameter)
        end if
    end function second_derivative_covariance_parameter

    pure real(dp) function second_derivative_covariance_parameter_dot(kind, variance, lengthscale, &
            x1, order1, x2, order2, parameter, direction) result(value)
        integer, intent(in) :: kind, order1, order2, parameter
        real(dp), intent(in) :: variance, lengthscale, x1, x2, direction(:)
        real(dp) :: covariance, length_dot

        value = 0.0_dp
        if (size(direction) /= 2) return
        if (kind == KERNEL_RBF) then
            covariance = rbf_derivative_covariance(variance, lengthscale, x1, order1, x2, order2)
            length_dot = (-1.0_dp)**order2*rbf_log_lengthscale_derivative(variance, lengthscale, &
                x1 - x2, order1 + order2)
            if (parameter == 1) then
                value = direction(1)*covariance + direction(2)*length_dot
            else if (parameter == 2) then
                value = direction(1)*length_dot + direction(2)*(-1.0_dp)**order2* &
                    rbf_log_lengthscale_second_derivative(variance, lengthscale, x1 - x2, &
                    order1 + order2)
            end if
        else if (kind == KERNEL_MATERN52) then
            value = matern52_parameter_derivative_dot(variance, lengthscale, x1, order1, x2, &
                order2, parameter, direction)
        end if
    end function second_derivative_covariance_parameter_dot

    logical function input_jvp_supported(self, x, orders) result(supported)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: orders(:)
        integer :: i, j
        real(dp) :: tolerance

        supported = .true.
        if (self%kernel%kind /= KERNEL_MATERN52) return
        tolerance = 1.0e-12_dp*max(1.0_dp, self%lengthscale)
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                if (self%orders(i) + orders(j) + 1 == 5 .and. &
                    abs(self%x_train(i) - x(j, 1)) <= tolerance) then
                    supported = .false.
                    return
                end if
            end do
        end do
    end function input_jvp_supported

    pure real(dp) function matern52_derivative_covariance(variance, lengthscale, x1, order1, &
            x2, order2) result(value)
        real(dp), intent(in) :: variance, lengthscale, x1, x2
        integer, intent(in) :: order1, order2
        real(dp), parameter :: root_five = 2.2360679774997896964_dp
        real(dp) :: tau, radius, base, radial_derivative
        integer :: total_order, sign_tau, sign_factor

        tau = x1 - x2
        radius = abs(tau)/lengthscale
        base = variance*exp(-root_five*radius)
        total_order = order1 + order2
        select case (total_order)
        case (0)
            radial_derivative = 1.0_dp + root_five*radius + (5.0_dp/3.0_dp)*radius*radius
        case (1)
            radial_derivative = -(5.0_dp/3.0_dp)*radius*(1.0_dp + root_five*radius)
        case (2)
            radial_derivative = (5.0_dp/3.0_dp)*(5.0_dp*radius*radius - &
                root_five*radius - 1.0_dp)
        case (3)
            radial_derivative = (25.0_dp/3.0_dp)*radius*(3.0_dp - root_five*radius)
        case (4)
            radial_derivative = (25.0_dp/3.0_dp)*(3.0_dp - 5.0_dp*root_five*radius + &
                5.0_dp*radius*radius)
        case (5)
            radial_derivative = (root_five**5/3.0_dp)*(-8.0_dp + 7.0_dp*root_five*radius - &
                5.0_dp*radius*radius)
        case (6)
            radial_derivative = (root_five**5/3.0_dp)*(15.0_dp*root_five - 45.0_dp*radius + &
                5.0_dp*root_five*radius*radius)
        case default
            radial_derivative = 0.0_dp
        end select
        sign_tau = 1
        if (tau < 0.0_dp) sign_tau = -1
        sign_factor = 1
        if (mod(total_order, 2) == 1) sign_factor = sign_tau
        value = base*radial_derivative/lengthscale**total_order
        value = sign_factor*value
        if (mod(order2, 2) == 1) value = -value
    end function matern52_derivative_covariance

    pure real(dp) function matern52_parameter_derivative(variance, lengthscale, x1, order1, &
            x2, order2, parameter) result(value)
        real(dp), intent(in) :: variance, lengthscale, x1, x2
        integer, intent(in) :: order1, order2, parameter
        real(dp) :: covariance, next_covariance, tau
        integer :: total_order

        value = 0.0_dp
        total_order = order1 + order2
        covariance = matern52_derivative_covariance(variance, lengthscale, x1, order1, x2, order2)
        if (parameter == 1) then
            value = covariance
        else if (parameter == 2) then
            tau = x1 - x2
            value = -real(total_order, dp)*covariance
            if (tau /= 0.0_dp) then
                next_covariance = matern52_derivative_covariance(variance, lengthscale, x1, &
                    order1, x2, order2 + 1)
                value = value + tau*next_covariance
            end if
        end if
    end function matern52_parameter_derivative

    pure real(dp) function matern52_parameter_derivative_dot(variance, lengthscale, x1, order1, &
            x2, order2, parameter, direction) result(value)
        real(dp), intent(in) :: variance, lengthscale, x1, x2, direction(:)
        integer, intent(in) :: order1, order2, parameter
        real(dp) :: covariance, length_covariance, next_covariance, next_next_covariance
        real(dp) :: tau
        integer :: total_order

        value = 0.0_dp
        if (size(direction) /= 2) return
        total_order = order1 + order2
        covariance = matern52_derivative_covariance(variance, lengthscale, x1, order1, x2, order2)
        length_covariance = matern52_parameter_derivative(variance, lengthscale, x1, order1, &
            x2, order2, 2)
        if (parameter == 1) then
            value = direction(1)*covariance + direction(2)*length_covariance
        else if (parameter == 2) then
            value = direction(1)*length_covariance
            tau = x1 - x2
            value = value + direction(2)*(-real(total_order, dp)*length_covariance)
            if (tau /= 0.0_dp) then
                next_covariance = matern52_derivative_covariance(variance, lengthscale, x1, &
                    order1, x2, order2 + 1)
                next_next_covariance = matern52_derivative_covariance(variance, lengthscale, &
                    x1, order1, x2, order2 + 2)
                ! h_n = -n*k_n + tau*k_(n+1), so its lengthscale tangent is
                ! -n*h_n + tau*(-(n+1)*k_(n+1) + tau*k_(n+2)).
                value = value + direction(2)*tau*( -real(total_order + 1, dp)*next_covariance + &
                    tau*next_next_covariance )
            end if
        end if
    end function matern52_parameter_derivative_dot

    pure real(dp) function rbf_derivative_covariance(variance, lengthscale, x1, order1, &
            x2, order2) result(value)
        real(dp), intent(in) :: variance, lengthscale, x1, x2
        integer, intent(in) :: order1, order2
        real(dp) :: d, base

        d = x1 - x2
        base = variance*exp(-0.5_dp*d*d/(lengthscale*lengthscale))
        value = (-1.0_dp)**order2 * rbf_distance_derivative(base, d, lengthscale, order1 + order2)
    end function rbf_derivative_covariance

    pure real(dp) function rbf_derivative_covariance_input_dot(variance, lengthscale, x1, &
            order1, x2, order2, direction) result(value)
        real(dp), intent(in) :: variance, lengthscale, x1, x2, direction
        integer, intent(in) :: order1, order2
        real(dp) :: d, base

        d = x1 - x2
        base = variance*exp(-0.5_dp*d*d/(lengthscale*lengthscale))
        value = direction*(-1.0_dp)**(order2 + 1)* &
            rbf_distance_derivative(base, d, lengthscale, order1 + order2 + 1)
    end function rbf_derivative_covariance_input_dot

    pure real(dp) function rbf_distance_derivative(base, d, lengthscale, order) result(value)
        real(dp), intent(in) :: base, d, lengthscale
        integer, intent(in) :: order
        real(dp) :: inv2, inv4, inv6, inv8, inv10, inv12, inv14

        inv2 = 1.0_dp/(lengthscale*lengthscale)
        inv4 = inv2*inv2
        inv6 = inv4*inv2
        inv8 = inv4*inv4
        inv10 = inv8*inv2
        inv12 = inv10*inv2
        inv14 = inv12*inv2
        select case (order)
        case (0)
            value = base
        case (1)
            value = -d*inv2*base
        case (2)
            value = (d*d*inv4 - inv2)*base
        case (3)
            value = (3.0_dp*d*inv4 - d*d*d*inv6)*base
        case (4)
            value = (d**4*inv8 - 6.0_dp*d*d*inv6 + 3.0_dp*inv4)*base
        case (5)
            value = (-d**5*inv10 + 10.0_dp*d**3*inv8 - 15.0_dp*d*inv6)*base
        case (6)
            value = (d**6*inv12 - 15.0_dp*d**4*inv10 + 45.0_dp*d*d*inv8 - &
                15.0_dp*inv6)*base
        case (7)
            value = (-d**7*inv14 + 21.0_dp*d**5*inv12 - 105.0_dp*d**3*inv10 + &
                105.0_dp*d*inv8)*base
        case default
            value = 0.0_dp
        end select
    end function rbf_distance_derivative

    pure real(dp) function rbf_log_lengthscale_derivative(variance, lengthscale, d, order) result(value)
        real(dp), intent(in) :: variance, lengthscale, d
        integer, intent(in) :: order
        real(dp) :: base, first, second, coefficient

        base = variance*exp(-0.5_dp*d*d/(lengthscale*lengthscale))
        first = rbf_distance_derivative(base, d, lengthscale, order)
        second = 0.0_dp
        if (order >= 1) second = 2.0_dp*real(order, dp)*d/(lengthscale*lengthscale)* &
            rbf_distance_derivative(base, d, lengthscale, order - 1)
        if (order >= 2) second = second + real(order*(order - 1), dp)/ &
            (lengthscale*lengthscale)*rbf_distance_derivative(base, d, lengthscale, order - 2)
        coefficient = d*d/(lengthscale*lengthscale)
        value = coefficient*first + second
    end function rbf_log_lengthscale_derivative

    pure real(dp) function rbf_log_lengthscale_second_derivative(variance, lengthscale, d, order) &
            result(value)
        real(dp), intent(in) :: variance, lengthscale, d
        integer, intent(in) :: order
        real(dp) :: base, first, first_dot, second, second_dot, coefficient

        base = variance*exp(-0.5_dp*d*d/(lengthscale*lengthscale))
        first = rbf_distance_derivative(base, d, lengthscale, order)
        first_dot = rbf_log_lengthscale_derivative(variance, lengthscale, d, order)
        coefficient = d*d/(lengthscale*lengthscale)
        value = -2.0_dp*coefficient*first + coefficient*first_dot
        if (order >= 1) then
            second = 2.0_dp*real(order, dp)*d/(lengthscale*lengthscale)* &
                rbf_distance_derivative(base, d, lengthscale, order - 1)
            second_dot = -2.0_dp*second + 2.0_dp*real(order, dp)*d/(lengthscale*lengthscale)* &
                rbf_log_lengthscale_derivative(variance, lengthscale, d, order - 1)
            value = value + second_dot
        end if
        if (order >= 2) then
            second = real(order*(order - 1), dp)/(lengthscale*lengthscale)* &
                rbf_distance_derivative(base, d, lengthscale, order - 2)
            second_dot = -2.0_dp*second + real(order*(order - 1), dp)/ &
                (lengthscale*lengthscale)*rbf_log_lengthscale_derivative(variance, lengthscale, d, order - 2)
            value = value + second_dot
        end if
    end function rbf_log_lengthscale_second_derivative

end module fortml_second_derivative_gaussian_process
