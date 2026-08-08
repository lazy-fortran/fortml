module fortml_second_derivative_gaussian_process
    !! Scalar RBF GP with value, first-, and second-derivative observations.
    !!
    !! `gp_derivative_regression_t` deliberately stops at first derivatives in
    !! its component encoding.  This bounded companion uses an explicit order
    !! vector (`0`, `1`, or `2`) for a one-dimensional RBF kernel.  Covariance
    !! blocks are generated from the exact derivatives through order four, so
    !! mixed value/gradient/Hessian observations and predictions share one
    !! Cholesky state.  Query input JVP/VJP products use the fifth RBF
    !! derivative; hyperparameter products and non-RBF leaves are explicit
    !! boundaries rather than finite-difference fallbacks.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, KERNEL_RBF
    implicit none
    private

    real(dp), parameter :: MIN_VARIANCE = -1.0e-9_dp

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
        procedure, public :: predict_device => second_derivative_predict_device
        procedure, public :: joint_covariance_device => &
            second_derivative_joint_covariance_device
        procedure, public :: device_supported => second_derivative_device_supported
        procedure, public :: observation_count => second_derivative_observation_count
        procedure, public :: parameters => second_derivative_parameters
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
                any(orders < 0) .or. any(orders > 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP fit: values or derivative orders are invalid")
            return
        end if
        if (kernel%kind /= KERNEL_RBF .or. kernel%input_dim /= 1 .or. &
                kernel%parameter_count() /= 2) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "second-derivative GP fit: only one-dimensional RBF is generated")
            return
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
                covariance(i, j) = rbf_derivative_covariance(self%kernel_variance, &
                    self%lengthscale, self%x_train(i), orders(i), self%x_train(j), orders(j))
            end do
            covariance(j, j) = covariance(j, j) + noise_variance + self%jitter
        end do
        call solve_factor(covariance, self%alpha, self%y_train, status)
        if (status%code /= FORTNUM_OK) then
            self%is_fitted = .false.
            return
        end if
        ! solve_factor stores only alpha; refactor once for prediction solves.
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
                cross(i, j) = rbf_derivative_covariance(self%kernel_variance, &
                    self%lengthscale, self%x_train(i), self%orders(i), x(j, 1), orders(j))
            end do
        end do
        mean = matmul(transpose(cross), self%alpha)
        work = cross
        call solve_factor_matrix(self, work, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, n
            prior = rbf_derivative_covariance(self%kernel_variance, self%lengthscale, &
                x(j, 1), orders(j), x(j, 1), orders(j))
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
                cross(i, j) = rbf_derivative_covariance(self%kernel_variance, &
                    self%lengthscale, self%x_train(i), self%orders(i), x(j, 1), orders(j))
            end do
            do i = 1, n
                prior(i, j) = rbf_derivative_covariance(self%kernel_variance, self%lengthscale, &
                    x(i, 1), orders(i), x(j, 1), orders(j))
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
        allocate(cross(self%n_observations, size(x, 1)), cross_dot(self%n_observations, size(x, 1)), &
            work(self%n_observations, size(x, 1)), work_dot(self%n_observations, size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                cross(i, j) = rbf_derivative_covariance(self%kernel_variance, self%lengthscale, &
                    self%x_train(i), self%orders(i), x(j, 1), orders(j))
                cross_dot(i, j) = rbf_derivative_covariance_input_dot(self%kernel_variance, &
                    self%lengthscale, self%x_train(i), self%orders(i), x(j, 1), orders(j), direction(j))
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
            prior = rbf_derivative_covariance(self%kernel_variance, self%lengthscale, &
                x(j, 1), orders(j), x(j, 1), orders(j))
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
        allocate(cross(self%n_observations, size(x, 1)), work(self%n_observations, size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                cross(i, j) = rbf_derivative_covariance(self%kernel_variance, self%lengthscale, &
                    self%x_train(i), self%orders(i), x(j, 1), orders(j))
            end do
        end do
        work = cross
        call solve_factor_matrix(self, work, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                cross_bar = self%alpha(i)*mean_bar(j) - &
                    2.0_dp*work(i, j)*variance_bar(j)
                derivative = rbf_derivative_covariance_input_dot(self%kernel_variance, &
                    self%lengthscale, self%x_train(i), self%orders(i), x(j, 1), orders(j), 1.0_dp)
                x_bar(j) = x_bar(j) + cross_bar*derivative
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine second_derivative_predict_input_vjp

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
                "second-derivative GP device prediction: resident RBF order-two kernel is not linked")
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
                "second-derivative GP covariance device: resident RBF order-two kernel is not linked")
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
                any(orders < 0) .or. any(orders > 2) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP query: shape or order is invalid")
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
                any(orders < 0) .or. any(orders > 2) .or. any(shape(covariance) /= &
                [size(x, 1), size(x, 1)]) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "second-derivative GP covariance: shape or order is invalid")
            return
        end if
        value = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_joint_query

    subroutine solve_factor(matrix, solution, rhs, status)
        real(dp), intent(in) :: matrix(:, :), rhs(:)
        real(dp), intent(out) :: solution(:)
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factor

        call factor%factorize(matrix, status)
        if (status%code /= FORTNUM_OK) return
        solution = rhs
        call factor%solve(solution, status)
    end subroutine solve_factor

    subroutine factorize_state(self, covariance, status)
        class(second_derivative_gp_t), intent(inout) :: self
        real(dp), intent(in) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        ! Re-factorization is intentionally local: solve_factor above is the
        ! independent fit solve, while prediction owns this persistent factor.
        call self%factorization%factorize(covariance, status)
    end subroutine factorize_state

    subroutine solve_factor_matrix(self, matrix, status)
        class(second_derivative_gp_t), intent(in) :: self
        real(dp), intent(inout) :: matrix(:, :)
        type(fortnum_status_t), intent(out) :: status

        call self%factorization%solve(matrix, status)
    end subroutine solve_factor_matrix

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
        real(dp) :: inv2, inv4, inv6, inv8, inv10, inv12

        inv2 = 1.0_dp/(lengthscale*lengthscale)
        inv4 = inv2*inv2
        inv6 = inv4*inv2
        inv8 = inv4*inv4
        inv10 = inv8*inv2
        inv12 = inv10*inv2
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
        case default
            value = 0.0_dp
        end select
    end function rbf_distance_derivative

end module fortml_second_derivative_gaussian_process
