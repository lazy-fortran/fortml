module fortml_derivative_gaussian_process
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, clone_kernel_into, KERNEL_RBF, KERNEL_MATERN12, &
        KERNEL_MATERN32, KERNEL_MATERN52, KERNEL_LINEAR, KERNEL_CONSTANT, &
        KERNEL_WHITE_NOISE, KERNEL_SUM, KERNEL_PRODUCT, KERNEL_USER, &
        KERNEL_PERIODIC, KERNEL_RATIONAL_QUADRATIC, KERNEL_COSINE, KERNEL_POLYNOMIAL
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
        procedure, public :: predict_device => gp_derivative_predict_device
        procedure, public :: joint_covariance => gp_derivative_joint_covariance
        procedure, public :: joint_covariance_device => gp_derivative_joint_covariance_device
        procedure, public :: joint_covariance_jvp => gp_derivative_joint_covariance_jvp
        procedure, public :: joint_covariance_vjp => gp_derivative_joint_covariance_vjp
        procedure, public :: device_supported => gp_derivative_device_supported
        procedure, public :: predict_jvp => gp_derivative_predict_jvp
        procedure, public :: predict_vjp => gp_derivative_predict_vjp
        procedure, public :: predict_input_jvp => gp_derivative_predict_input_jvp
        procedure, public :: predict_input_vjp => gp_derivative_predict_input_vjp
        procedure, public :: observation_count => gp_derivative_observation_count
        procedure, public :: parameter_count => gp_derivative_parameter_count
        procedure, public :: parameters => gp_derivative_parameters
        procedure, public :: set_parameters => gp_derivative_set_parameters
        procedure, public :: log_marginal_likelihood => &
            gp_derivative_log_marginal_likelihood
        procedure, public :: log_marginal_likelihood_jvp => &
            gp_derivative_log_marginal_likelihood_jvp
        procedure, public :: log_marginal_likelihood_vjp => &
            gp_derivative_log_marginal_likelihood_vjp
        procedure, public :: hyperparameter_gradient => &
            gp_derivative_hyperparameter_gradient
        procedure, public :: hyperparameter_vjp => &
            gp_derivative_hyperparameter_vjp
        procedure, public :: hyperparameter_hvp => gp_derivative_hyperparameter_hvp
    end type gp_derivative_regression_t

    public :: gp_derivative_fit
    public :: gp_derivative_predict
    public :: gp_derivative_predict_device
    public :: gp_derivative_joint_covariance
    public :: gp_derivative_joint_covariance_device
    public :: gp_derivative_joint_covariance_jvp
    public :: gp_derivative_joint_covariance_vjp
    public :: gp_derivative_device_supported
    public :: gp_derivative_predict_jvp
    public :: gp_derivative_predict_vjp
    public :: gp_derivative_predict_input_jvp
    public :: gp_derivative_predict_input_vjp
    public :: gp_derivative_log_marginal_likelihood
    public :: gp_derivative_log_marginal_likelihood_vjp
    public :: gp_derivative_hyperparameter_gradient
    public :: gp_derivative_hyperparameter_vjp
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

    subroutine gp_derivative_predict_device(self, device, x, components, mean, &
            variance, status)
        !! Predict mixed value/first-derivative observations through the
        !! explicit device boundary.  The CPU path is the reference
        !! implementation; CUDA is refused until a resident covariance,
        !! factorization, and derivative-query kernel is linked.
        class(gp_derivative_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: mean(:, :), variance(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, components, mean, variance, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "derivative GP device prediction: no resident CUDA covariance "// &
                "or derivative-observation kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP device prediction: device kind is invalid")
        end select
    end subroutine gp_derivative_predict_device

    subroutine gp_derivative_joint_covariance(self, x, components, covariance, status)
        !! Dense latent posterior covariance for an arbitrary mixed query set.
        !! Rows in `x` and `components` are paired exactly as in `predict`; the
        !! returned matrix is ordered by that query list and excludes
        !! observation noise.  CPU is the reference implementation.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior(:, :), work(:, :)
        integer :: i, j

        if (.not. derivative_gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP joint covariance: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            size(components) /= size(x, 1) .or. any(components < 0) .or. &
            any(components > self%n_features) .or. any(.not. ieee_is_finite(x)) .or. &
            any(shape(covariance) /= [size(x, 1), size(x, 1)])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP joint covariance: input or output shape is invalid")
            return
        end if
        allocate(cross(self%n_observations, size(x, 1)))
        allocate(prior(size(x, 1), size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                call derivative_covariance(self%kernel, self%x_train(i, :), self%components(i), &
                    x(j, :), components(j), cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
            do i = 1, size(x, 1)
                call derivative_covariance(self%kernel, x(i, :), components(i), x(j, :), &
                    components(j), prior(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        allocate(work, source=cross)
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        covariance = prior - matmul(transpose(cross), work)
        covariance = 0.5_dp*(covariance + transpose(covariance))
        do i = 1, size(x, 1)
            if (covariance(i, i) < 0.0_dp) then
                if (covariance(i, i) < -1.0e-9_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "derivative GP joint covariance: posterior variance is not positive")
                    return
                end if
                covariance(i, i) = 0.0_dp
            end if
        end do
        if (any(.not. ieee_is_finite(covariance))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "derivative GP joint covariance: nonfinite result")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_joint_covariance

    subroutine gp_derivative_joint_covariance_device(self, device, x, components, covariance, status)
        !! Explicit backend boundary for joint mixed-query covariance.  CUDA
        !! remains refused until the resident derivative covariance graph is
        !! linked; no hidden host fallback is permitted.
        class(gp_derivative_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP joint covariance device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%joint_covariance(x, components, covariance, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "derivative GP joint covariance device: no resident CUDA covariance graph is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP joint covariance device: device kind is invalid")
        end select
    end subroutine gp_derivative_joint_covariance_device

    subroutine gp_derivative_joint_covariance_jvp(self, x, components, direction, &
            covariance, covariance_dot, status)
        !! Directional parameter product of the dense latent posterior
        !! covariance.  This is the exact derivative of
        !! ``P - C^T K^{-1} C`` in packed kernel-log/noise-log coordinates;
        !! the factorization and all covariance blocks remain on the CPU
        !! reference path.  CUDA has no implicit host fallback.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: covariance(:, :), covariance_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), prior(:, :)
        real(dp), allocatable :: prior_dot(:, :), train_dot(:, :)
        real(dp), allocatable :: work(:, :), work_dot(:, :)
        real(dp) :: noise_dot, value, value_dot
        integer :: i, j, kernel_count, n_query

        n_query = size(x, 1)
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. n_query < 1 .or. &
            size(x, 2) /= self%n_features .or. size(components) /= n_query .or. &
            any(components < 0) .or. any(components > self%n_features) .or. &
            any(shape(covariance) /= [n_query, n_query]) .or. &
            any(shape(covariance_dot) /= [n_query, n_query])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP joint covariance JVP: input or output shape is invalid")
            return
        end if
        call self%joint_covariance(x, components, covariance, status)
        if (status%code /= FORTNUM_OK) return

        kernel_count = self%kernel%parameter_count()
        allocate(cross(self%n_observations, n_query))
        allocate(cross_dot, mold=cross)
        allocate(prior(n_query, n_query), prior_dot(n_query, n_query))
        allocate(train_dot(self%n_observations, self%n_observations))
        allocate(work, mold=cross)
        allocate(work_dot, mold=cross)
        do j = 1, n_query
            do i = 1, self%n_observations
                call derivative_covariance(self%kernel, self%x_train(i, :), &
                    self%components(i), x(j, :), components(j), cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
                call derivative_covariance_direction(self%kernel, self%x_train(i, :), &
                    self%components(i), x(j, :), components(j), direction(:kernel_count), &
                    value, value_dot, status)
                if (status%code /= FORTNUM_OK) return
                cross_dot(i, j) = value_dot
            end do
            do i = 1, n_query
                call derivative_covariance(self%kernel, x(i, :), components(i), &
                    x(j, :), components(j), prior(i, j), status)
                if (status%code /= FORTNUM_OK) return
                call derivative_covariance_direction(self%kernel, x(i, :), components(i), &
                    x(j, :), components(j), direction(:kernel_count), value, &
                    prior_dot(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        do j = 1, self%n_observations
            do i = 1, self%n_observations
                call derivative_covariance_direction(self%kernel, self%x_train(i, :), &
                    self%components(i), self%x_train(j, :), self%components(j), &
                    direction(:kernel_count), value, train_dot(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        noise_dot = self%noise_variance*direction(kernel_count + 1)
        do i = 1, self%n_observations
            train_dot(i, i) = train_dot(i, i) + noise_dot
        end do

        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        work_dot = cross_dot - matmul(train_dot, work)
        call self%factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        covariance_dot = prior_dot - matmul(transpose(cross_dot), work) - &
            matmul(transpose(cross), work_dot)
        covariance_dot = 0.5_dp*(covariance_dot + transpose(covariance_dot))
        if (any(.not. ieee_is_finite(covariance_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "derivative GP joint covariance JVP: nonfinite result")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_joint_covariance_jvp

    subroutine gp_derivative_joint_covariance_vjp(self, x, components, covariance_bar, &
            parameter_bar, status)
        !! Reverse product of the dense latent posterior covariance with
        !! respect to packed kernel-log/noise-log parameters.  The symmetric
        !! cotangent is propagated through the exact dense solve; unsupported
        !! derivative-observation kernels return their ordinary typed refusal.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), covariance_bar(:, :)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance(:, :), cross(:, :), work(:, :)
        real(dp), allocatable :: cotangent(:, :), cross_bar(:, :), train_bar(:, :)
        real(dp) :: value, value_dot
        integer :: i, j, p, kernel_count, n_query

        parameter_bar = 0.0_dp
        n_query = size(x, 1)
        if (size(parameter_bar) /= self%parameter_count() .or. n_query < 1 .or. &
            size(x, 2) /= self%n_features .or. size(components) /= n_query .or. &
            any(components < 0) .or. any(components > self%n_features) .or. &
            any(shape(covariance_bar) /= [n_query, n_query]) .or. &
            any(.not. ieee_is_finite(covariance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP joint covariance VJP: input or output shape is invalid")
            return
        end if
        allocate(covariance(n_query, n_query))
        call self%joint_covariance(x, components, covariance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(cross(self%n_observations, n_query), work(self%n_observations, n_query))
        do j = 1, n_query
            do i = 1, self%n_observations
                call derivative_covariance(self%kernel, self%x_train(i, :), &
                    self%components(i), x(j, :), components(j), cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        cotangent = 0.5_dp*(covariance_bar + transpose(covariance_bar))
        cross_bar = -2.0_dp*matmul(work, cotangent)
        train_bar = matmul(work, matmul(cotangent, transpose(work)))
        kernel_count = self%kernel%parameter_count()
        do p = 1, kernel_count
            do j = 1, self%n_observations
                do i = 1, self%n_observations
                    call derivative_covariance_parameter(self%kernel, self%x_train(i, :), &
                        self%components(i), self%x_train(j, :), self%components(j), p, &
                        value, value_dot, status)
                    if (status%code /= FORTNUM_OK) return
                    parameter_bar(p) = parameter_bar(p) + train_bar(i, j)*value_dot
                end do
            end do
            do j = 1, n_query
                do i = 1, self%n_observations
                    call derivative_covariance_parameter(self%kernel, self%x_train(i, :), &
                        self%components(i), x(j, :), components(j), p, value, value_dot, status)
                    if (status%code /= FORTNUM_OK) return
                    parameter_bar(p) = parameter_bar(p) + cross_bar(i, j)*value_dot
                end do
            end do
            do j = 1, n_query
                do i = 1, n_query
                    call derivative_covariance_parameter(self%kernel, x(i, :), components(i), &
                        x(j, :), components(j), p, value, value_dot, status)
                    if (status%code /= FORTNUM_OK) return
                    parameter_bar(p) = parameter_bar(p) + cotangent(i, j)*value_dot
                end do
            end do
        end do
        parameter_bar(kernel_count + 1) = self%noise_variance*sum(diagonal(train_bar))
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "derivative GP joint covariance VJP: nonfinite result")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_joint_covariance_vjp

    logical function gp_derivative_device_supported(self, device_kind) result(supported)
        !! Report capability without implying an implicit host fallback.
        class(gp_derivative_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = derivative_gp_fitted(self)
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function gp_derivative_device_supported

    subroutine gp_derivative_predict_jvp(self, x, components, direction, mean, &
            mean_dot, variance, variance_dot, status)
        !! Parameter JVP of derivative-observation GP prediction.
        !! Query inputs and derivative components are held fixed; `direction`
        !! follows the packed kernel-log/noise-log parameter order.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), prior(:), prior_dot(:)
        real(dp), allocatable :: train_dot(:, :), alpha_dot(:, :), work(:, :), work_dot(:, :)
        real(dp) :: noise_dot, covariance, covariance_dot
        integer :: i, j, p, kernel_count

        call check_derivative_prediction_shapes(self, x, components, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        if (any(shape(mean_dot) /= shape(mean)) .or. size(variance_dot) /= size(variance) .or. &
            size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP predict_jvp: tangent shape or value is invalid")
            return
        end if

        kernel_count = self%kernel%parameter_count()
        allocate(cross(self%n_observations, size(x, 1)))
        allocate(cross_dot(self%n_observations, size(x, 1)))
        allocate(prior(size(x, 1)))
        allocate(prior_dot(size(x, 1)))
        allocate(train_dot(self%n_observations, self%n_observations))
        allocate(alpha_dot(size(self%alpha, 1), size(self%alpha, 2)))
        allocate(work(self%n_observations, size(x, 1)))
        allocate(work_dot(self%n_observations, size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                call derivative_covariance_direction(self%kernel, self%x_train(i, :), &
                    self%components(i), x(j, :), components(j), direction(:kernel_count), &
                    cross(i, j), cross_dot(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
            call derivative_covariance_direction(self%kernel, x(j, :), components(j), &
                x(j, :), components(j), direction(:kernel_count), prior(j), prior_dot(j), &
                status)
            if (status%code /= FORTNUM_OK) return
        end do
        do j = 1, self%n_observations
            do i = 1, self%n_observations
                call derivative_covariance_direction(self%kernel, self%x_train(i, :), &
                    self%components(i), self%x_train(j, :), self%components(j), &
                    direction(:kernel_count), covariance, train_dot(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        noise_dot = self%noise_variance*direction(kernel_count + 1)
        do i = 1, self%n_observations
            train_dot(i, i) = train_dot(i, i) + noise_dot
        end do
        alpha_dot = -matmul(train_dot, self%alpha)
        call self%factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return
        mean = matmul(transpose(cross), self%alpha)
        mean_dot = matmul(transpose(cross_dot), self%alpha) + &
            matmul(transpose(cross), alpha_dot)
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        work_dot = cross_dot - matmul(train_dot, work)
        call self%factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        variance = prior - sum(cross*work, dim=1)
        variance_dot = prior_dot - sum(cross_dot*work + cross*work_dot, dim=1)
        call clamp_derivative_prediction_variance(variance, status)
    end subroutine gp_derivative_predict_jvp

    subroutine gp_derivative_predict_vjp(self, x, components, mean_bar, variance_bar, &
            parameter_bar, status)
        !! Reverse product of derivative-observation GP prediction with respect
        !! to packed kernel-log/noise-log parameters. Query inputs are fixed.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:, :), variance_bar(:)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), cross_bar(:, :)
        real(dp), allocatable :: train_bar(:, :), alpha_bar(:, :), lambda(:, :)
        real(dp) :: covariance, covariance_dot, local_value
        integer :: i, j, p, kernel_count

        call check_derivative_prediction_shapes(self, x, components, mean_bar, variance_bar, status)
        if (status%code /= FORTNUM_OK) return
        if (size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP predict_vjp: parameter output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(mean_bar)) .or. &
            any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP predict_vjp: cotangents must be finite")
            return
        end if

        kernel_count = self%kernel%parameter_count()
        allocate(cross(self%n_observations, size(x, 1)))
        allocate(work, mold=cross)
        allocate(cross_bar, mold=cross)
        allocate(train_bar(self%n_observations, self%n_observations))
        allocate(alpha_bar, mold=self%alpha)
        allocate(lambda, mold=self%alpha)
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                call derivative_covariance(self%kernel, self%x_train(i, :), &
                    self%components(i), x(j, :), components(j), cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        cross_bar = matmul(self%alpha, transpose(mean_bar)) - &
            2.0_dp*work*spread(variance_bar, dim=1, ncopies=self%n_observations)
        alpha_bar = matmul(cross, mean_bar)
        lambda = alpha_bar
        call self%factorization%solve(lambda, status)
        if (status%code /= FORTNUM_OK) return
        train_bar = matmul(work*spread(variance_bar, dim=1, &
            ncopies=self%n_observations), transpose(work)) - 0.5_dp*( &
            matmul(lambda, transpose(self%alpha)) + matmul(self%alpha, transpose(lambda)))

        parameter_bar = 0.0_dp
        do p = 1, kernel_count
            do j = 1, self%n_observations
                do i = 1, self%n_observations
                    call derivative_covariance_parameter(self%kernel, self%x_train(i, :), &
                        self%components(i), self%x_train(j, :), self%components(j), p, &
                        covariance, covariance_dot, status)
                    if (status%code /= FORTNUM_OK) return
                    parameter_bar(p) = parameter_bar(p) + train_bar(i, j)*covariance_dot
                end do
            end do
            do j = 1, size(x, 1)
                do i = 1, self%n_observations
                    call derivative_covariance_parameter(self%kernel, self%x_train(i, :), &
                        self%components(i), x(j, :), components(j), p, local_value, &
                        covariance_dot, status)
                    if (status%code /= FORTNUM_OK) return
                    parameter_bar(p) = parameter_bar(p) + cross_bar(i, j)*covariance_dot
                end do
                call derivative_covariance_parameter(self%kernel, x(j, :), components(j), &
                    x(j, :), components(j), p, local_value, covariance_dot, status)
                if (status%code /= FORTNUM_OK) return
                parameter_bar(p) = parameter_bar(p) + variance_bar(j)*covariance_dot
            end do
        end do
        parameter_bar(kernel_count + 1) = self%noise_variance*sum(diagonal(train_bar))
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_predict_vjp

    subroutine gp_derivative_predict_input_jvp(self, x, components, direction, mean, &
            mean_dot, variance, variance_dot, status)
        !! Query-input JVP of a derivative-observation GP prediction.
        !!
        !! Smooth built-in leaves and sum/product trees propagate their exact
        !! third-input covariance products. Parameters, training inputs, and
        !! derivative components are held fixed; unsupported user formulas
        !! return a typed capability refusal.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:, :)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), work(:, :)
        real(dp), allocatable :: prior_dot(:), zero_direction(:)
        integer :: i, j

        call check_derivative_prediction_shapes(self, x, components, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        if (size(x, 1) < 1 .or. any(shape(direction) /= shape(x)) .or. &
            any(shape(mean_dot) /= shape(mean)) .or. size(variance_dot) /= size(variance) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP predict_input_jvp: direction or output shape is invalid")
            return
        end if

        call self%predict(x, components, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(cross(self%n_observations, size(x, 1)))
        allocate(cross_dot(self%n_observations, size(x, 1)))
        allocate(work(self%n_observations, size(x, 1)))
        allocate(prior_dot(size(x, 1)))
        allocate(zero_direction(self%n_features))
        zero_direction = 0.0_dp
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                call derivative_covariance_query_direction(self%kernel, &
                    self%x_train(i, :), self%components(i), x(j, :), components(j), &
                    zero_direction, direction(j, :), cross(i, j), cross_dot(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
            call derivative_covariance_query_direction(self%kernel, x(j, :), components(j), &
                x(j, :), components(j), direction(j, :), direction(j, :), &
                variance(j), prior_dot(j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        mean_dot = matmul(transpose(cross_dot), self%alpha)
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        variance_dot = prior_dot - 2.0_dp*sum(cross_dot*work, dim=1)
        call clamp_derivative_prediction_variance(variance, status)
        if (status%code /= FORTNUM_OK) return
        if (any(.not. ieee_is_finite(mean_dot)) .or. any(.not. ieee_is_finite(variance_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "derivative GP predict_input_jvp: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_predict_input_jvp

    subroutine gp_derivative_predict_input_vjp(self, x, components, mean_bar, variance_bar, &
            x_bar, status)
        !! Query-input VJP of a derivative-observation GP prediction.
        !!
        !! This is the adjoint of `predict_input_jvp` for the same deterministic
        !! central covariance difference.  Query inputs are the only returned
        !! cotangent; model parameters and training inputs are held fixed.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:, :), variance_bar(:)
        integer, intent(in) :: components(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), cross_bar(:, :)
        real(dp), allocatable :: zero_direction(:), basis(:)
        real(dp) :: covariance, covariance_dot, prior, prior_dot
        integer :: i, j, k

        call check_derivative_prediction_shapes(self, x, components, mean_bar, variance_bar, status)
        if (status%code /= FORTNUM_OK) return
        if (size(x, 1) < 1 .or. any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(mean_bar)) .or. &
            any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP predict_input_vjp: cotangent or output shape is invalid")
            return
        end if

        allocate(cross(self%n_observations, size(x, 1)))
        allocate(work(self%n_observations, size(x, 1)))
        allocate(cross_bar(self%n_observations, size(x, 1)))
        allocate(zero_direction(self%n_features), basis(self%n_features))
        zero_direction = 0.0_dp
        do j = 1, size(x, 1)
            do i = 1, self%n_observations
                call derivative_covariance(self%kernel, self%x_train(i, :), &
                    self%components(i), x(j, :), components(j), cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        cross_bar = matmul(self%alpha, transpose(mean_bar)) - 2.0_dp*work* &
            spread(variance_bar, dim=1, ncopies=self%n_observations)
        x_bar = 0.0_dp
        do j = 1, size(x, 1)
            do k = 1, self%n_features
                basis = 0.0_dp
                basis(k) = 1.0_dp
                do i = 1, self%n_observations
                    call derivative_covariance_query_direction(self%kernel, &
                        self%x_train(i, :), self%components(i), x(j, :), components(j), &
                        zero_direction, basis, covariance, covariance_dot, status)
                    if (status%code /= FORTNUM_OK) return
                    x_bar(j, k) = x_bar(j, k) + cross_bar(i, j)*covariance_dot
                end do
                call derivative_covariance_query_direction(self%kernel, x(j, :), components(j), &
                    x(j, :), components(j), basis, basis, prior, prior_dot, status)
                if (status%code /= FORTNUM_OK) return
                x_bar(j, k) = x_bar(j, k) + variance_bar(j)*prior_dot
            end do
        end do
        if (any(.not. ieee_is_finite(x_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "derivative GP predict_input_vjp: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_predict_input_vjp

    subroutine check_derivative_prediction_shapes(self, x, components, mean, variance, status)
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean(:, :), variance(:)
        integer, intent(in) :: components(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. derivative_gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP prediction: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(components) /= size(x, 1) .or. &
            any(components < 0) .or. any(components > self%n_features) .or. &
            any(.not. ieee_is_finite(x)) .or. &
            any(shape(mean) /= [size(x, 1), self%n_outputs]) .or. &
            size(variance) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP prediction: input or output shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_derivative_prediction_shapes

    subroutine clamp_derivative_prediction_variance(variance, status)
        real(dp), intent(inout) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        do i = 1, size(variance)
            if (variance(i) < 0.0_dp) then
                if (variance(i) < -1.0e-9_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "derivative GP prediction: posterior variance is not positive")
                    return
                end if
                variance(i) = 0.0_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine clamp_derivative_prediction_variance

    subroutine derivative_covariance_query_direction(kernel, x1, component1, x2, component2, &
            direction1, direction2, covariance, covariance_dot, status)
        !! Exact directional query derivative of one mixed value/derivative
        !! covariance.  Smooth built-in leaves use their analytic third input
        !! derivative; sum/product trees are propagated by the product rule.
        !! User formulas and nonsmooth leaves return a typed refusal rather than
        !! silently falling back to finite differences.
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:), direction1(:), direction2(:)
        integer, intent(in) :: component1, component2
        real(dp), intent(out) :: covariance, covariance_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        real(dp), allocatable :: gradient_x1_dot(:), gradient_x2_dot(:), mixed_hessian_dot(:, :)
        real(dp) :: value, value_dot

        covariance = 0.0_dp
        covariance_dot = 0.0_dp
        if (size(x1) /= size(x2) .or. size(direction1) /= size(x1) .or. &
            size(direction2) /= size(x2) .or. any(.not. ieee_is_finite(x1)) .or. &
            any(.not. ieee_is_finite(x2)) .or. any(.not. ieee_is_finite(direction1)) .or. &
            any(.not. ieee_is_finite(direction2)) .or. component1 < 0 .or. &
            component2 < 0 .or. component1 > size(x1) .or. component2 > size(x2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP query covariance: direction or component is invalid")
            return
        end if
        if (kernel_contains_white_noise(kernel) .and. &
            (component1 > 0 .or. component2 > 0)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP query covariance: white-noise derivative is undefined")
            return
        end if
        allocate(gradient_x1(size(x1)), gradient_x2(size(x2)))
        allocate(mixed_hessian(size(x1), size(x2)))
        allocate(gradient_x1_dot(size(x1)), gradient_x2_dot(size(x2)))
        allocate(mixed_hessian_dot(size(x1), size(x2)))
        call kernel_input_third_direction(kernel, x1, x2, direction1, direction2, &
            value, gradient_x1, gradient_x2, mixed_hessian, value_dot, gradient_x1_dot, &
            gradient_x2_dot, mixed_hessian_dot, status)
        if (status%code /= FORTNUM_OK) return
        if (component1 == 0 .and. component2 == 0) then
            covariance = value
            covariance_dot = value_dot
        else if (component1 > 0 .and. component2 == 0) then
            covariance = gradient_x1(component1)
            covariance_dot = gradient_x1_dot(component1)
        else if (component1 == 0 .and. component2 > 0) then
            covariance = gradient_x2(component2)
            covariance_dot = gradient_x2_dot(component2)
        else
            covariance = mixed_hessian(component1, component2)
            covariance_dot = mixed_hessian_dot(component1, component2)
        end if
        if (.not. ieee_is_finite(covariance) .or. .not. ieee_is_finite(covariance_dot)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "derivative GP query covariance: nonfinite exact product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine derivative_covariance_query_direction

    subroutine derivative_covariance_direction(kernel, x1, component1, x2, component2, &
            direction, covariance, covariance_dot, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:), direction(:)
        integer, intent(in) :: component1, component2
        real(dp), intent(out) :: covariance, covariance_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: local_value, local_dot
        integer :: parameter

        if (size(direction) /= kernel%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            covariance = 0.0_dp
            covariance_dot = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP covariance JVP: parameter direction is invalid")
            return
        end if
        call derivative_covariance(kernel, x1, component1, x2, component2, covariance, status)
        if (status%code /= FORTNUM_OK) then
            covariance_dot = 0.0_dp
            return
        end if
        covariance_dot = 0.0_dp
        do parameter = 1, kernel%parameter_count()
            call derivative_covariance_parameter(kernel, x1, component1, x2, component2, &
                parameter, local_value, local_dot, status)
            if (status%code /= FORTNUM_OK) then
                covariance_dot = 0.0_dp
                return
            end if
            covariance_dot = covariance_dot + direction(parameter)*local_dot
        end do
    end subroutine derivative_covariance_direction

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
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
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

    subroutine gp_derivative_log_marginal_likelihood_vjp(self, value_bar, parameter_bar, status)
        !! Reverse product of the scalar derivative-GP log marginal likelihood.
        !! The packed coordinates are kernel log parameters followed by the
        !! log observation-noise variance.  A scalar objective cotangent is
        !! accepted explicitly so callers can compose this likelihood with a
        !! larger scalar objective without special-casing a gradient.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        parameter_bar = 0.0_dp
        if (.not. derivative_gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP likelihood_vjp: model is not fitted")
            return
        end if
        if (size(parameter_bar) /= self%parameter_count() .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP likelihood_vjp: cotangent or output shape is invalid")
            return
        end if
        allocate(gradient(size(parameter_bar)))
        call self%hyperparameter_gradient(gradient, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar = value_bar*gradient
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "derivative GP likelihood_vjp: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_derivative_log_marginal_likelihood_vjp

    subroutine gp_derivative_hyperparameter_vjp(self, value_bar, parameter_bar, status)
        !! Alias for the scalar likelihood VJP in the model's public
        !! hyperparameter-product vocabulary.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        call self%log_marginal_likelihood_vjp(value_bar, parameter_bar, status)
    end subroutine gp_derivative_hyperparameter_vjp

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

        if (all(self%components == 0)) then
            call derivative_gp_value_only_hvp(self, direction, parameter_hvp, status)
            return
        end if
        call derivative_gp_mixed_hvp(self, direction, parameter_hvp, status)
    end subroutine gp_derivative_hyperparameter_hvp

    subroutine derivative_gp_mixed_hvp(self, direction, parameter_hvp, status)
        !! Analytic Hessian-vector product for mixed value/first-derivative
        !! observations.  The covariance directional derivative, parameter
        !! derivative, and parameter/direction mixed derivative are assembled
        !! from the supported smooth leaf rules below.  A leaf without the
        !! required second input/parameter product returns
        !! `FORTNUM_NOT_IMPLEMENTED`; no finite-difference fallback is hidden
        !! behind this public HVP entry point.
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance(:, :), covariance_dot(:, :)
        real(dp), allocatable :: alpha_dot(:, :), identity(:, :), inverse(:, :)
        real(dp), allocatable :: inverse_dot(:, :), matrix_bar(:, :)
        real(dp), allocatable :: matrix_bar_dot(:, :), parameter_matrix(:, :)
        real(dp), allocatable :: parameter_matrix_dot(:, :)
        real(dp) :: covariance_value, covariance_dot_value
        real(dp) :: covariance_parameter, covariance_parameter_dot
        real(dp) :: noise_dot, trace_matrix_bar, trace_matrix_bar_dot
        integer :: i, j, p, kernel_count

        parameter_hvp = 0.0_dp
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP mixed HVP: direction shape or value is invalid")
            return
        end if
        kernel_count = self%kernel%parameter_count()
        allocate(covariance(self%n_observations, self%n_observations))
        allocate(covariance_dot, mold=covariance)
        allocate(alpha_dot, mold=self%alpha)
        allocate(identity, mold=covariance)
        allocate(inverse, mold=covariance)
        allocate(inverse_dot, mold=covariance)
        allocate(matrix_bar, mold=covariance)
        allocate(matrix_bar_dot, mold=covariance)
        allocate(parameter_matrix, mold=covariance)
        allocate(parameter_matrix_dot, mold=covariance)

        do j = 1, self%n_observations
            do i = 1, self%n_observations
                call derivative_covariance_direction(self%kernel, self%x_train(i, :), &
                    self%components(i), self%x_train(j, :), self%components(j), &
                    direction(:kernel_count), covariance(i, j), covariance_dot(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        noise_dot = self%noise_variance*direction(kernel_count + 1)
        do i = 1, self%n_observations
            covariance_dot(i, i) = covariance_dot(i, i) + noise_dot
        end do

        alpha_dot = -matmul(covariance_dot, self%alpha)
        call self%factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return
        identity = 0.0_dp
        do i = 1, self%n_observations
            identity(i, i) = 1.0_dp
        end do
        inverse = identity
        call self%factorization%solve(inverse, status)
        if (status%code /= FORTNUM_OK) return
        inverse_dot = -matmul(inverse, matmul(covariance_dot, inverse))
        matrix_bar = 0.5_dp*(matmul(self%alpha, transpose(self%alpha)) - &
            real(self%n_outputs, dp)*inverse)
        matrix_bar_dot = 0.5_dp*(matmul(alpha_dot, transpose(self%alpha)) + &
            matmul(self%alpha, transpose(alpha_dot)) - &
            real(self%n_outputs, dp)*inverse_dot)

        parameter_hvp = 0.0_dp
        do p = 1, kernel_count
            do j = 1, self%n_observations
                do i = 1, self%n_observations
                    call derivative_covariance_parameter_hvp(self%kernel, &
                        self%x_train(i, :), self%components(i), self%x_train(j, :), &
                        self%components(j), p, direction(:kernel_count), covariance_value, &
                        covariance_dot_value, covariance_parameter, covariance_parameter_dot, &
                        status)
                    if (status%code /= FORTNUM_OK) return
                    parameter_matrix(i, j) = covariance_parameter
                    parameter_matrix_dot(i, j) = covariance_parameter_dot
                end do
            end do
            parameter_hvp(p) = sum(matrix_bar_dot*parameter_matrix) + &
                sum(matrix_bar*parameter_matrix_dot)
        end do
        trace_matrix_bar = sum(diagonal(matrix_bar))
        trace_matrix_bar_dot = sum(diagonal(matrix_bar_dot))
        parameter_hvp(kernel_count + 1) = self%noise_variance*( &
            direction(kernel_count + 1)*trace_matrix_bar + trace_matrix_bar_dot)
        if (any(.not. ieee_is_finite(parameter_hvp))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "derivative GP mixed HVP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine derivative_gp_mixed_hvp

    subroutine derivative_gp_value_only_hvp(self, direction, parameter_hvp, status)
        class(gp_derivative_regression_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance(:, :), covariance_dot(:, :)
        real(dp), allocatable :: alpha_dot(:, :), identity(:, :), inverse(:, :)
        real(dp), allocatable :: inverse_dot(:, :), matrix_bar(:, :)
        real(dp), allocatable :: matrix_bar_dot(:, :), local_bar(:), local_bar_dot(:)
        real(dp) :: noise_dot, trace_matrix_bar, trace_matrix_bar_dot
        integer :: i, kernel_count

        kernel_count = self%kernel%parameter_count()
        allocate(covariance(self%n_observations, self%n_observations))
        allocate(covariance_dot, mold=covariance)
        allocate(alpha_dot, mold=self%alpha)
        allocate(identity, mold=covariance)
        allocate(inverse, mold=covariance)
        allocate(inverse_dot, mold=covariance)
        allocate(matrix_bar, mold=covariance)
        allocate(matrix_bar_dot, mold=covariance)
        allocate(local_bar(kernel_count), local_bar_dot(kernel_count))

        call self%kernel%matrix_jvp(self%x_train, self%x_train, &
            direction(:kernel_count), covariance, covariance_dot, status)
        if (status%code /= FORTNUM_OK) return
        noise_dot = self%noise_variance*direction(kernel_count + 1)
        do i = 1, self%n_observations
            covariance_dot(i, i) = covariance_dot(i, i) + noise_dot
        end do

        alpha_dot = -matmul(covariance_dot, self%alpha)
        call self%factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return

        identity = 0.0_dp
        do i = 1, self%n_observations
            identity(i, i) = 1.0_dp
        end do
        inverse = identity
        call self%factorization%solve(inverse, status)
        if (status%code /= FORTNUM_OK) return
        inverse_dot = -matmul(inverse, matmul(covariance_dot, inverse))

        matrix_bar = 0.5_dp*(matmul(self%alpha, transpose(self%alpha)) - &
            real(self%n_outputs, dp)*inverse)
        matrix_bar_dot = 0.5_dp*( &
            matmul(alpha_dot, transpose(self%alpha)) + &
            matmul(self%alpha, transpose(alpha_dot)) - &
            real(self%n_outputs, dp)*inverse_dot)

        parameter_hvp = 0.0_dp
        call self%kernel%parameter_hvp( &
            self%x_train, self%x_train, matrix_bar, direction(:kernel_count), &
            local_bar, local_bar_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%parameter_vjp( &
            self%x_train, self%x_train, matrix_bar_dot, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_hvp(:kernel_count) = local_bar + local_bar_dot

        trace_matrix_bar = 0.0_dp
        trace_matrix_bar_dot = 0.0_dp
        do i = 1, self%n_observations
            trace_matrix_bar = trace_matrix_bar + matrix_bar(i, i)
            trace_matrix_bar_dot = trace_matrix_bar_dot + matrix_bar_dot(i, i)
        end do
        parameter_hvp(kernel_count + 1) = self%noise_variance*( &
            direction(kernel_count + 1)*trace_matrix_bar + trace_matrix_bar_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine derivative_gp_value_only_hvp

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

        if (component1 == 0 .and. component2 == 0) then
            call kernel_value_parameter_jvp(kernel, x1, x2, parameter, covariance, &
                covariance_dot, status)
            return
        end if

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

    recursive subroutine derivative_covariance_parameter_hvp(kernel, x1, component1, x2, &
            component2, parameter, direction, covariance, covariance_dot, &
            covariance_parameter, covariance_parameter_dot, status)
        !! Return one derivative-observation covariance block and its
        !! parameter/directional products.  The HVP needs
        !! ``d C_p / d direction`` in addition to ``C_p``.  RBF, linear, and
        !! constant leaves have closed forms; sums/products use exact product
        !! rules.  Other leaves deliberately refuse until their fourth-order
        !! input/parameter products are generated and independently checked.
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:), direction(:)
        integer, intent(in) :: component1, component2, parameter
        real(dp), intent(out) :: covariance, covariance_dot
        real(dp), intent(out) :: covariance_parameter, covariance_parameter_dot
        type(fortnum_status_t), intent(out) :: status
        integer :: left_count
        real(dp) :: left_value, right_value, left_dot, right_dot
        real(dp) :: left_parameter, right_parameter
        real(dp) :: left_parameter_dot, right_parameter_dot
        real(dp) :: variance, lengthscale, q, r2, f, t
        real(dp) :: c_length, c_length_length, a, a_length, a_length_length
        real(dp) :: difference

        covariance = 0.0_dp
        covariance_dot = 0.0_dp
        covariance_parameter = 0.0_dp
        covariance_parameter_dot = 0.0_dp
        if (size(x1) /= size(x2) .or. size(direction) /= kernel%parameter_count() .or. &
            any(.not. ieee_is_finite(x1)) .or. any(.not. ieee_is_finite(x2)) .or. &
            any(.not. ieee_is_finite(direction)) .or. component1 < 0 .or. component2 < 0 .or. &
            component1 > size(x1) .or. component2 > size(x2) .or. parameter < 1 .or. &
            parameter > kernel%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP mixed HVP: input, component, or parameter is invalid")
            return
        end if
        if (kernel_contains_white_noise(kernel) .and. &
            (component1 > 0 .or. component2 > 0)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP mixed HVP: white-noise derivative is undefined")
            return
        end if

        select case (kernel%kind)
        case (KERNEL_SUM)
            left_count = kernel%left%parameter_count()
            if (parameter <= left_count) then
                call derivative_covariance_parameter_hvp(kernel%left, x1, component1, x2, &
                    component2, parameter, direction(:left_count), covariance, covariance_dot, &
                    covariance_parameter, covariance_parameter_dot, status)
                if (status%code /= FORTNUM_OK) return
                call derivative_covariance_direction(kernel%right, x1, component1, x2, &
                    component2, direction(left_count + 1:), right_value, right_dot, status)
                if (status%code /= FORTNUM_OK) return
                covariance = covariance + right_value
                covariance_dot = covariance_dot + right_dot
            else
                call derivative_covariance_direction(kernel%left, x1, component1, x2, &
                    component2, direction(:left_count), left_value, left_dot, status)
                if (status%code /= FORTNUM_OK) return
                call derivative_covariance_parameter_hvp(kernel%right, x1, component1, x2, &
                    component2, parameter - left_count, direction(left_count + 1:), &
                    covariance, covariance_dot, covariance_parameter, &
                    covariance_parameter_dot, status)
                if (status%code /= FORTNUM_OK) return
                covariance = covariance + left_value
                covariance_dot = covariance_dot + left_dot
            end if
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_PRODUCT)
            left_count = kernel%left%parameter_count()
            if (parameter <= left_count) then
                call derivative_covariance_parameter_hvp(kernel%left, x1, component1, x2, &
                    component2, parameter, direction(:left_count), left_value, left_dot, &
                    left_parameter, left_parameter_dot, status)
                if (status%code /= FORTNUM_OK) return
                call derivative_covariance_direction(kernel%right, x1, component1, x2, &
                    component2, direction(left_count + 1:), right_value, right_dot, status)
                if (status%code /= FORTNUM_OK) return
                covariance = left_value*right_value
                covariance_dot = left_dot*right_value + left_value*right_dot
                covariance_parameter = left_parameter*right_value
                covariance_parameter_dot = left_parameter_dot*right_value + &
                    left_parameter*right_dot
            else
                call derivative_covariance_direction(kernel%left, x1, component1, x2, &
                    component2, direction(:left_count), left_value, left_dot, status)
                if (status%code /= FORTNUM_OK) return
                call derivative_covariance_parameter_hvp(kernel%right, x1, component1, x2, &
                    component2, parameter - left_count, direction(left_count + 1:), &
                    right_value, right_dot, right_parameter, right_parameter_dot, status)
                if (status%code /= FORTNUM_OK) return
                covariance = left_value*right_value
                covariance_dot = left_dot*right_value + left_value*right_dot
                covariance_parameter = left_value*right_parameter
                covariance_parameter_dot = left_dot*right_parameter + &
                    left_value*right_parameter_dot
            end if
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_RBF)
            if (kernel%parameter_count() /= 2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "derivative GP mixed HVP: RBF parameter layout is invalid")
                return
            end if
            variance = exp(kernel%log_parameters(1))
            lengthscale = exp(kernel%log_parameters(2))
            q = 1.0_dp/(lengthscale*lengthscale)
            r2 = sum((x1 - x2)**2)
            f = variance*exp(-0.5_dp*q*r2)
            t = q*r2
            if (component1 == 0 .and. component2 == 0) then
                covariance = f
                c_length = f*t
                c_length_length = f*(t*t - 2.0_dp*t)
            else if (component1 > 0 .and. component2 == 0) then
                difference = x1(component1) - x2(component1)
                covariance = -f*q*difference
                c_length = covariance*(t - 2.0_dp)
                c_length_length = covariance*(t*t - 6.0_dp*t + 4.0_dp)
            else if (component1 == 0 .and. component2 > 0) then
                difference = x1(component2) - x2(component2)
                covariance = f*q*difference
                c_length = covariance*(t - 2.0_dp)
                c_length_length = covariance*(t*t - 6.0_dp*t + 4.0_dp)
            else
                a = q*merge(1.0_dp, 0.0_dp, component1 == component2) - &
                    q*q*(x1(component1) - x2(component1))* &
                    (x1(component2) - x2(component2))
                a_length = -2.0_dp*q*merge(1.0_dp, 0.0_dp, component1 == component2) + &
                    4.0_dp*q*q*(x1(component1) - x2(component1))* &
                    (x1(component2) - x2(component2))
                a_length_length = 4.0_dp*q*merge(1.0_dp, 0.0_dp, component1 == component2) - &
                    16.0_dp*q*q*(x1(component1) - x2(component1))* &
                    (x1(component2) - x2(component2))
                covariance = f*a
                c_length = f*(t*a + a_length)
                c_length_length = f*((t*t - 2.0_dp*t)*a + 2.0_dp*t*a_length + &
                    a_length_length)
            end if
            covariance_dot = direction(1)*covariance + direction(2)*c_length
            if (parameter == 1) then
                covariance_parameter = covariance
                covariance_parameter_dot = covariance_dot
            else
                covariance_parameter = c_length
                covariance_parameter_dot = direction(1)*c_length + &
                    direction(2)*c_length_length
            end if
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_LINEAR)
            if (kernel%parameter_count() /= 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "derivative GP mixed HVP: linear parameter layout is invalid")
                return
            end if
            variance = exp(kernel%log_parameters(1))
            if (component1 == 0 .and. component2 == 0) then
                covariance = variance*dot_product(x1, x2)
            else if (component1 > 0 .and. component2 == 0) then
                covariance = variance*x2(component1)
            else if (component1 == 0 .and. component2 > 0) then
                covariance = variance*x1(component2)
            else
                covariance = variance*merge(1.0_dp, 0.0_dp, component1 == component2)
            end if
            covariance_dot = direction(1)*covariance
            covariance_parameter = covariance
            covariance_parameter_dot = covariance_dot
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_CONSTANT)
            if (kernel%parameter_count() /= 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "derivative GP mixed HVP: constant parameter layout is invalid")
                return
            end if
            if (component1 == 0 .and. component2 == 0) then
                covariance = exp(kernel%log_parameters(1))
            else
                covariance = 0.0_dp
            end if
            covariance_dot = direction(1)*covariance
            covariance_parameter = covariance
            covariance_parameter_dot = covariance_dot
            call status_set(status, FORTNUM_OK, "")
            return
        case default
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "derivative GP mixed HVP: kernel leaf lacks analytic second products")
            return
        end select
    end subroutine derivative_covariance_parameter_hvp

    recursive subroutine kernel_value_parameter_jvp(kernel, x1, x2, parameter, &
            value, value_dot, status)
        !! Value-only parameter products do not need input derivatives.  Keep
        !! this path separate so a user formula containing `push_distance` can
        !! still train on function values at coincident points; derivative
        !! observations continue to refuse the singular input derivative.
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:)
        integer, intent(in) :: parameter
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:)
        real(dp), allocatable :: mixed_hessian(:, :), gradient_x1_dot(:)
        real(dp), allocatable :: gradient_x2_dot(:), mixed_hessian_dot(:, :)
        real(dp) :: left_value, right_value, left_dot, right_dot
        integer :: left_count

        value = 0.0_dp
        value_dot = 0.0_dp
        if (parameter < 1 .or. parameter > kernel%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel value parameter JVP: parameter index is invalid")
            return
        end if
        select case (kernel%kind)
        case (KERNEL_SUM)
            left_count = kernel%left%parameter_count()
            if (parameter <= left_count) then
                call kernel_value_parameter_jvp(kernel%left, x1, x2, parameter, &
                    left_value, left_dot, status)
                if (status%code /= FORTNUM_OK) return
                right_value = kernel%right%value(x1, x2)
                right_dot = 0.0_dp
            else
                left_value = kernel%left%value(x1, x2)
                left_dot = 0.0_dp
                call kernel_value_parameter_jvp(kernel%right, x1, x2, &
                    parameter - left_count, right_value, right_dot, status)
                if (status%code /= FORTNUM_OK) return
            end if
            value = left_value + right_value
            value_dot = left_dot + right_dot
        case (KERNEL_PRODUCT)
            left_count = kernel%left%parameter_count()
            if (parameter <= left_count) then
                call kernel_value_parameter_jvp(kernel%left, x1, x2, parameter, &
                    left_value, left_dot, status)
                if (status%code /= FORTNUM_OK) return
                right_value = kernel%right%value(x1, x2)
                right_dot = 0.0_dp
            else
                left_value = kernel%left%value(x1, x2)
                left_dot = 0.0_dp
                call kernel_value_parameter_jvp(kernel%right, x1, x2, &
                    parameter - left_count, right_value, right_dot, status)
                if (status%code /= FORTNUM_OK) return
            end if
            value = left_value*right_value
            value_dot = left_dot*right_value + left_value*right_dot
        case (KERNEL_USER)
            value = kernel%value(x1, x2)
            value_dot = value
        case default
            allocate(gradient_x1(size(x1)), gradient_x2(size(x2)))
            allocate(mixed_hessian(size(x1), size(x2)))
            allocate(gradient_x1_dot(size(x1)), gradient_x2_dot(size(x2)))
            allocate(mixed_hessian_dot(size(x1), size(x2)))
            call kernel_input_parameter_jvp(kernel, x1, x2, parameter, value, &
                gradient_x1, gradient_x2, mixed_hessian, value_dot, gradient_x1_dot, &
                gradient_x2_dot, mixed_hessian_dot, status)
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_value_parameter_jvp

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
        real(dp) :: difference_vector(size(x1))
        real(dp) :: exponential, a
        real(dp) :: alpha, period, inverse_length_squared, denominator, t
        real(dp) :: frequency, argument, t1, t2, t1_dot, t2_dot, b, b_dot
        real(dp) :: p, p2, p_dot, p2_dot, curvature, curvature_dot, logf_dot
        real(dp) :: sine_value, cosine_value, cosine_numerator
        real(dp) :: scale, offset, degree, base, inner_product, base_dot
        real(dp) :: variance_log_dot, scale_log_dot, offset_log_dot, degree_log_dot
        real(dp) :: coefficient, coefficient_dot, curvature_dot_input, log_direction
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
        case (KERNEL_POLYNOMIAL)
            !! k(x1,x2) = variance*(offset + scale*dot(x1,x2))**degree.
            !! Keep the input derivatives and their parameter tangents in
            !! closed form so derivative-observation GP likelihoods can use
            !! all four logarithmic polynomial parameters.
            variance = exp(kernel%log_parameters(1))
            scale = exp(kernel%log_parameters(2))
            offset = exp(kernel%log_parameters(3))
            degree = exp(kernel%log_parameters(4))
            inner_product = dot_product(x1, x2)
            base = offset + scale*inner_product
            if (base <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel polynomial parameter JVP: base must be positive")
                return
            end if
            variance_log_dot = 0.0_dp
            scale_log_dot = 0.0_dp
            offset_log_dot = 0.0_dp
            degree_log_dot = 0.0_dp
            select case (parameter)
            case (1)
                variance_log_dot = 1.0_dp
            case (2)
                scale_log_dot = 1.0_dp
            case (3)
                offset_log_dot = 1.0_dp
            case (4)
                degree_log_dot = 1.0_dp
            case default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel polynomial parameter JVP: parameter index is invalid")
                return
            end select
            base_dot = scale*inner_product*scale_log_dot + offset*offset_log_dot
            value = variance*base**degree
            log_direction = variance_log_dot + degree*base_dot/base + &
                degree*degree_log_dot*log(base)
            value_dot = value*log_direction
            coefficient = variance*degree*scale*base**(degree - 1.0_dp)
            coefficient_dot = coefficient*(variance_log_dot + degree_log_dot + &
                scale_log_dot + (degree - 1.0_dp)*base_dot/base + &
                degree*degree_log_dot*log(base))
            curvature = variance*degree*(degree - 1.0_dp)*scale*scale* &
                base**(degree - 2.0_dp)
            if (parameter == 4) then
                !! d/dd [d(d-1)b**(d-2)] times dd/d(log d)=d.
                curvature_dot_input = variance*scale*scale*base**(degree - 2.0_dp)*degree*( &
                    2.0_dp*degree - 1.0_dp + degree*(degree - 1.0_dp)*log(base))
            else
                curvature_dot_input = curvature*(variance_log_dot + degree_log_dot + &
                    2.0_dp*scale_log_dot + (degree - 2.0_dp)*base_dot/base + &
                    degree*degree_log_dot*log(base))
            end if
            gradient_x1 = coefficient*x2
            gradient_x2 = coefficient*x1
            gradient_x1_dot = coefficient_dot*x2
            gradient_x2_dot = coefficient_dot*x1
            do i = 1, size(x1)
                do j = 1, size(x2)
                    mixed_hessian(i, j) = coefficient*merge(1.0_dp, 0.0_dp, i == j) + &
                        curvature*x2(i)*x1(j)
                    mixed_hessian_dot(i, j) = coefficient_dot*merge(1.0_dp, 0.0_dp, i == j) + &
                        curvature_dot_input*x2(i)*x1(j)
                end do
            end do
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_USER)
            call kernel%input_derivatives(x1, x2, value, gradient_x1, gradient_x2, &
                mixed_hessian, status)
            if (status%code /= FORTNUM_OK) return
            if (parameter == 1) then
                value_dot = value
                gradient_x1_dot = gradient_x1
                gradient_x2_dot = gradient_x2
                mixed_hessian_dot = mixed_hessian
            end if
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_PERIODIC, KERNEL_RATIONAL_QUADRATIC)
            !! Both leaves are functions F(s), s=||x1-x2||**2.  Work in
            !! s-space so the value, gradient, and mixed Hessian parameter
            !! products share the same stable expressions at coincidence.
            difference_vector = x1 - x2
            squared_distance = sum(difference_vector*difference_vector)
            variance = exp(kernel%log_parameters(1))
            lengthscale = exp(kernel%log_parameters(2))
            if (kernel%kind == KERNEL_PERIODIC) then
                period = exp(kernel%log_parameters(3))
                frequency = acos(-1.0_dp)/period
                distance = sqrt(squared_distance)
                argument = frequency*distance
                if (distance <= 1.0e-8_dp) then
                    t1 = frequency*frequency
                    t2 = -2.0_dp*frequency**4/3.0_dp
                    t = frequency*frequency*squared_distance
                else
                    t1 = frequency*sin(2.0_dp*argument)/(2.0_dp*distance)
                    t2 = frequency*(2.0_dp*frequency*distance*cos(2.0_dp*argument) - &
                        sin(2.0_dp*argument))/(4.0_dp*distance**3)
                    t = sin(argument)**2
                end if
                b = 2.0_dp/(lengthscale*lengthscale)
                value = variance*exp(-b*t)
                p = -b*t1*value
                p2 = (b*b*t1*t1 - b*t2)*value
                if (parameter == 1) then
                    logf_dot = 1.0_dp
                    b_dot = 0.0_dp
                    t1_dot = 0.0_dp
                    t2_dot = 0.0_dp
                else if (parameter == 2) then
                    logf_dot = 2.0_dp*b*t
                    b_dot = -2.0_dp*b
                    t1_dot = 0.0_dp
                    t2_dot = 0.0_dp
                else
                    logf_dot = b*frequency*distance*sin(2.0_dp*argument)
                    b_dot = 0.0_dp
                    if (distance <= 1.0e-8_dp) then
                        t1_dot = -2.0_dp*t1
                        t2_dot = -4.0_dp*t2
                    else
                        t1_dot = -t1 - frequency*frequency*cos(2.0_dp*argument)
                        t2_dot = -t2 + frequency**3*sin(2.0_dp*argument)/distance
                    end if
                end if
                value_dot = value*logf_dot
                p_dot = -(b_dot*t1 + b*t1_dot)*value - b*t1*value_dot
                curvature = b*b*t1*t1 - b*t2
                curvature_dot = 2.0_dp*b*b_dot*t1*t1 + 2.0_dp*b*b*t1*t1_dot - &
                    b_dot*t2 - b*t2_dot
                p2_dot = curvature_dot*value + curvature*value_dot
            else
                alpha = exp(kernel%log_parameters(3))
                denominator = 1.0_dp + 0.5_dp*squared_distance/(alpha*lengthscale*lengthscale)
                t = 0.5_dp*squared_distance/(alpha*lengthscale*lengthscale)
                value = variance*denominator**(-alpha)
                p = -0.5_dp*value/(lengthscale*lengthscale*denominator)
                p2 = value*(alpha + 1.0_dp)/(4.0_dp*alpha*lengthscale**4*denominator**2)
                if (parameter == 1) then
                    logf_dot = 1.0_dp
                    p_dot = p
                    p2_dot = p2
                else if (parameter == 2) then
                    logf_dot = 2.0_dp*alpha*t/denominator
                    p_dot = p*(-2.0_dp + 2.0_dp*(alpha + 1.0_dp)*t/denominator)
                    p2_dot = p2*(-4.0_dp + 2.0_dp*(alpha + 2.0_dp)*t/denominator)
                else
                    logf_dot = alpha*(t/denominator - log(denominator))
                    p_dot = p*(-alpha*log(denominator) + (alpha + 1.0_dp)*t/denominator)
                    p2_dot = p2*(-1.0_dp/(alpha + 1.0_dp) - alpha*log(denominator) + &
                        (alpha + 2.0_dp)*t/denominator)
                end if
                value_dot = value*logf_dot
            end if
            gradient_x1 = 2.0_dp*p*difference_vector
            gradient_x2 = -gradient_x1
            gradient_x1_dot = 2.0_dp*p_dot*difference_vector
            gradient_x2_dot = -gradient_x1_dot
            do i = 1, size(x1)
                do j = 1, size(x2)
                    mixed_hessian(i, j) = -2.0_dp*p*merge(1.0_dp, 0.0_dp, i == j) - &
                        4.0_dp*p2*difference_vector(i)*difference_vector(j)
                    mixed_hessian_dot(i, j) = -2.0_dp*p_dot*merge(1.0_dp, 0.0_dp, i == j) - &
                        4.0_dp*p2_dot*difference_vector(i)*difference_vector(j)
                end do
            end do
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_COSINE)
            !! The cosine leaf is radial, F(s)=variance*cos(sqrt(s)/lengthscale).
            !! Work in s=||x1-x2||**2 so value, input gradients, and mixed
            !! Hessians share one stable parameter tangent.  This closes the
            !! derivative-observation parameter product for cosine kernels;
            !! unsupported leaves continue to return a typed refusal below.
            difference_vector = x1 - x2
            squared_distance = sum(difference_vector*difference_vector)
            variance = exp(kernel%log_parameters(1))
            lengthscale = exp(kernel%log_parameters(2))
            distance = sqrt(squared_distance)
            if (distance <= 1.0e-8_dp) then
                value = variance
                p = -variance/(2.0_dp*lengthscale*lengthscale)
                p2 = variance/(12.0_dp*lengthscale**4)
                if (parameter == 1) then
                    value_dot = value
                    p_dot = p
                    p2_dot = p2
                else if (parameter == 2) then
                    value_dot = 0.0_dp
                    p_dot = -2.0_dp*p
                    p2_dot = -4.0_dp*p2
                else
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel leaf parameter JVP: cosine parameter index is invalid")
                    return
                end if
            else
                z = distance/lengthscale
                sine_value = sin(z)
                cosine_value = cos(z)
                value = variance*cosine_value
                p = -variance*sine_value/(2.0_dp*lengthscale*distance)
                cosine_numerator = sine_value - z*cosine_value
                p2 = variance*cosine_numerator/(4.0_dp*lengthscale**4*z**3)
                if (parameter == 1) then
                    value_dot = value
                    p_dot = p
                    p2_dot = p2
                else if (parameter == 2) then
                    value_dot = variance*z*sine_value
                    p_dot = p*(-1.0_dp - z*cosine_value/sine_value)
                    if (abs(sine_value) <= 1.0e-10_dp) then
                        !! Avoid a removable 0/0 in the logarithmic tangent at
                        !! cosine extrema; evaluate the regular expression.
                        p_dot = variance*(sine_value + z*cosine_value)/ &
                            (2.0_dp*lengthscale*lengthscale*z)
                    end if
                    !! Differentiate n(z)/z**3 directly instead of forming a
                    !! ratio through n(z); n can vanish away from the origin.
                    p2_dot = variance/(4.0_dp*lengthscale**4)* &
                        (-4.0_dp*cosine_numerator/z**3 - sine_value/z + &
                        3.0_dp*cosine_numerator/z**4)
                else
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel leaf parameter JVP: cosine parameter index is invalid")
                    return
                end if
            end if
            gradient_x1 = 2.0_dp*p*difference_vector
            gradient_x2 = -gradient_x1
            gradient_x1_dot = 2.0_dp*p_dot*difference_vector
            gradient_x2_dot = -gradient_x1_dot
            do i = 1, size(x1)
                do j = 1, size(x2)
                    mixed_hessian(i, j) = -2.0_dp*p*merge(1.0_dp, 0.0_dp, i == j) - &
                        4.0_dp*p2*difference_vector(i)*difference_vector(j)
                    mixed_hessian_dot(i, j) = -2.0_dp*p_dot*merge(1.0_dp, 0.0_dp, i == j) - &
                        4.0_dp*p2_dot*difference_vector(i)*difference_vector(j)
                end do
            end do
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

    recursive subroutine kernel_input_third_direction(kernel, x1, x2, direction1, &
            direction2, value, gradient_x1, gradient_x2, mixed_hessian, value_dot, &
            gradient_x1_dot, gradient_x2_dot, mixed_hessian_dot, status)
        !! Value, first input derivatives, mixed Hessian, and its exact
        !! directional derivative.  The latter is the third input derivative
        !! needed by derivative-observation query products.
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:), direction1(:), direction2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :), value_dot
        real(dp), intent(out) :: gradient_x1_dot(:), gradient_x2_dot(:)
        real(dp), intent(out) :: mixed_hessian_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: left_g1(:), right_g1(:), left_g2(:), right_g2(:)
        real(dp), allocatable :: left_h(:, :), right_h(:, :), left_g1d(:), right_g1d(:)
        real(dp), allocatable :: left_g2d(:), right_g2d(:), left_hd(:, :), right_hd(:, :)
        real(dp) :: left_value, right_value, left_value_dot, right_value_dot
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        gradient_x1 = 0.0_dp
        gradient_x2 = 0.0_dp
        mixed_hessian = 0.0_dp
        gradient_x1_dot = 0.0_dp
        gradient_x2_dot = 0.0_dp
        mixed_hessian_dot = 0.0_dp
        if (size(x1) /= size(x2) .or. size(direction1) /= size(x1) .or. &
            size(direction2) /= size(x2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel third input derivative: dimension mismatch")
            return
        end if
        if (kernel%kind == KERNEL_SUM .or. kernel%kind == KERNEL_PRODUCT) then
            allocate(left_g1(size(x1)), right_g1(size(x1)), left_g2(size(x2)), right_g2(size(x2)))
            allocate(left_h(size(x1), size(x2)), right_h(size(x1), size(x2)))
            allocate(left_g1d(size(x1)), right_g1d(size(x1)), left_g2d(size(x2)), right_g2d(size(x2)))
            allocate(left_hd(size(x1), size(x2)), right_hd(size(x1), size(x2)))
            call kernel_input_third_direction(kernel%left, x1, x2, direction1, direction2, &
                left_value, left_g1, left_g2, left_h, left_value_dot, left_g1d, left_g2d, &
                left_hd, status)
            if (status%code /= FORTNUM_OK) return
            call kernel_input_third_direction(kernel%right, x1, x2, direction1, direction2, &
                right_value, right_g1, right_g2, right_h, right_value_dot, right_g1d, right_g2d, &
                right_hd, status)
            if (status%code /= FORTNUM_OK) return
            if (kernel%kind == KERNEL_SUM) then
                value = left_value + right_value
                value_dot = left_value_dot + right_value_dot
                gradient_x1 = left_g1 + right_g1
                gradient_x2 = left_g2 + right_g2
                mixed_hessian = left_h + right_h
                gradient_x1_dot = left_g1d + right_g1d
                gradient_x2_dot = left_g2d + right_g2d
                mixed_hessian_dot = left_hd + right_hd
            else
                value = left_value*right_value
                value_dot = left_value_dot*right_value + left_value*right_value_dot
                gradient_x1 = left_g1*right_value + right_g1*left_value
                gradient_x2 = left_g2*right_value + right_g2*left_value
                mixed_hessian = left_h*right_value + right_h*left_value + &
                    spread(left_g1, dim=2, ncopies=size(x2))*spread(right_g2, dim=1, ncopies=size(x1)) + &
                    spread(right_g1, dim=2, ncopies=size(x2))*spread(left_g2, dim=1, ncopies=size(x1))
                gradient_x1_dot = left_g1d*right_value + right_g1d*left_value + &
                    left_g1*right_value_dot + right_g1*left_value_dot
                gradient_x2_dot = left_g2d*right_value + right_g2d*left_value + &
                    left_g2*right_value_dot + right_g2*left_value_dot
                mixed_hessian_dot = left_hd*right_value + right_hd*left_value + &
                    left_h*right_value_dot + right_h*left_value_dot + &
                    spread(left_g1d, dim=2, ncopies=size(x2))*spread(right_g2, dim=1, ncopies=size(x1)) + &
                    spread(left_g1, dim=2, ncopies=size(x2))*spread(right_g2d, dim=1, ncopies=size(x1)) + &
                    spread(right_g1d, dim=2, ncopies=size(x2))*spread(left_g2, dim=1, ncopies=size(x1)) + &
                    spread(right_g1, dim=2, ncopies=size(x2))*spread(left_g2d, dim=1, ncopies=size(x1))
            end if
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call smooth_leaf_input_third_direction(kernel, x1, x2, direction1, direction2, &
            value, gradient_x1, gradient_x2, mixed_hessian, value_dot, gradient_x1_dot, &
            gradient_x2_dot, mixed_hessian_dot, status)
    end subroutine kernel_input_third_direction

    subroutine smooth_leaf_input_third_direction(kernel, x1, x2, direction1, direction2, &
            value, gradient_x1, gradient_x2, mixed_hessian, value_dot, gradient_x1_dot, &
            gradient_x2_dot, mixed_hessian_dot, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), x2(:), direction1(:), direction2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :), value_dot
        real(dp), intent(out) :: gradient_x1_dot(:), gradient_x2_dot(:)
        real(dp), intent(out) :: mixed_hessian_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: difference(size(x1)), delta(size(x1)), squared_distance, distance
        real(dp) :: p, p2, p3, radial_first, radial_second, radial_third
        real(dp) :: radial_scale, radial_coefficient, dot_difference
        real(dp) :: variance, lengthscale, alpha, period, inverse_length_squared
        real(dp) :: denominator, t, argument, sine_value, cosine_value, pi_over_period
        real(dp) :: cosine_numerator
        real(dp) :: t1, t2, t3, b, f, rscale, rsecond
        real(dp) :: a, exponential, z
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        gradient_x1 = 0.0_dp
        gradient_x2 = 0.0_dp
        mixed_hessian = 0.0_dp
        gradient_x1_dot = 0.0_dp
        gradient_x2_dot = 0.0_dp
        mixed_hessian_dot = 0.0_dp
        if (kernel%kind == KERNEL_USER) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "derivative GP query products: user kernels have no third-input contract")
            return
        end if
        if (kernel%kind == KERNEL_WHITE_NOISE) then
            value = kernel%value(x1, x2)
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (kernel%kind == KERNEL_CONSTANT) then
            value = exp(kernel%log_parameters(1))
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (kernel%kind == KERNEL_LINEAR) then
            variance = exp(kernel%log_parameters(1))
            value = variance*dot_product(x1, x2)
            value_dot = variance*(dot_product(direction1, x2) + dot_product(x1, direction2))
            gradient_x1 = variance*x2
            gradient_x2 = variance*x1
            gradient_x1_dot = variance*direction2
            gradient_x2_dot = variance*direction1
            do i = 1, size(x1)
                mixed_hessian(i, i) = variance
            end do
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        if (kernel%kind == KERNEL_POLYNOMIAL) then
            !! Polynomial covariance depends on the inner product rather
            !! than the Euclidean distance.  The closed-form directional
            !! product below supplies the third input derivative required by
            !! derivative-observation query JVP/VJP products.
            variance = exp(kernel%log_parameters(1))
            lengthscale = exp(kernel%log_parameters(2))
            alpha = exp(kernel%log_parameters(3))
            period = exp(kernel%log_parameters(4))
            dot_difference = dot_product(x1, x2)
            dot_difference = alpha + lengthscale*dot_difference
            if (dot_difference <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "derivative GP polynomial query products: base must be positive")
                return
            end if
            dot_difference = dot_product(direction1, x2) + dot_product(x1, direction2)
            value = variance*(alpha + lengthscale*dot_product(x1, x2))**period
            p = variance*period*lengthscale*(alpha + lengthscale*dot_product(x1, x2))** &
                (period - 1.0_dp)
            p2 = variance*period*(period - 1.0_dp)*lengthscale*lengthscale* &
                (alpha + lengthscale*dot_product(x1, x2))**(period - 2.0_dp)
            value_dot = p*dot_difference
            gradient_x1 = p*x2
            gradient_x2 = p*x1
            gradient_x1_dot = p2*dot_difference*x2 + p*direction2
            gradient_x2_dot = p2*dot_difference*x1 + p*direction1
            do i = 1, size(x1)
                do j = 1, size(x2)
                    mixed_hessian(i, j) = p*merge(1.0_dp, 0.0_dp, i == j) + p2*x2(i)*x1(j)
                    mixed_hessian_dot(i, j) = p2*dot_difference*merge(1.0_dp, 0.0_dp, i == j) + &
                        p2*direction2(i)*x1(j) + p2*x2(i)*direction1(j) + &
                        variance*period*(period - 1.0_dp)*(period - 2.0_dp)*lengthscale**3* &
                        (alpha + lengthscale*dot_product(x1, x2))**(period - 3.0_dp)* &
                        dot_difference*x2(i)*x1(j)
                end do
            end do
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        difference = x1 - x2
        delta = direction1 - direction2
        squared_distance = sum(difference*difference)
        distance = sqrt(squared_distance)
        if (kernel%kind == KERNEL_RBF .or. kernel%kind == KERNEL_PERIODIC .or. &
            kernel%kind == KERNEL_RATIONAL_QUADRATIC .or. kernel%kind == KERNEL_COSINE) then
            variance = exp(kernel%log_parameters(1))
            lengthscale = exp(kernel%log_parameters(2))
            if (kernel%kind == KERNEL_RBF) then
                inverse_length_squared = 1.0_dp/(lengthscale*lengthscale)
                value = variance*exp(-0.5_dp*squared_distance*inverse_length_squared)
                p = -0.5_dp*inverse_length_squared*value
                p2 = 0.25_dp*inverse_length_squared**2*value
                p3 = -0.125_dp*inverse_length_squared**3*value
            else if (kernel%kind == KERNEL_COSINE) then
                !! Cosine radial leaf: F(r)=variance*cos(r/lengthscale).
                !! These s-space derivatives remain finite at coincidence and
                !! provide the third input derivative needed by query JVP/VJP.
                if (distance <= 1.0e-8_dp) then
                    value = variance
                    p = -variance/(2.0_dp*lengthscale*lengthscale)
                    p2 = variance/(12.0_dp*lengthscale**4)
                    p3 = -variance/(120.0_dp*lengthscale**6)
                else
                    z = distance/lengthscale
                    sine_value = sin(z)
                    cosine_value = cos(z)
                    value = variance*cosine_value
                    p = -variance*sine_value/(2.0_dp*lengthscale*distance)
                    cosine_numerator = sine_value - z*cosine_value
                    p2 = variance*cosine_numerator/(4.0_dp*lengthscale**4*z**3)
                    p3 = variance*((z*z - 3.0_dp)*sine_value + 3.0_dp*z*cosine_value)/ &
                        (8.0_dp*lengthscale**6*z**5)
                end if
            else if (kernel%kind == KERNEL_RATIONAL_QUADRATIC) then
                alpha = exp(kernel%log_parameters(3))
                denominator = 1.0_dp + 0.5_dp*squared_distance/(alpha*lengthscale*lengthscale)
                value = variance*denominator**(-alpha)
                p = -0.5_dp*value/(lengthscale*lengthscale*denominator)
                p2 = value*(alpha + 1.0_dp)/(4.0_dp*alpha*lengthscale**4*denominator**2)
                p3 = -value*(alpha + 1.0_dp)*(alpha + 2.0_dp)/ &
                    (8.0_dp*alpha*alpha*lengthscale**6*denominator**3)
            else
                period = exp(kernel%log_parameters(3))
                pi_over_period = acos(-1.0_dp)/period
                argument = pi_over_period*distance
                b = 2.0_dp/(lengthscale*lengthscale)
                if (distance <= 1.0e-8_dp) then
                    t1 = pi_over_period*pi_over_period
                    t2 = -2.0_dp*pi_over_period**4/3.0_dp
                    t3 = 4.0_dp*pi_over_period**6/15.0_dp
                else
                    sine_value = sin(argument)
                    cosine_value = cos(argument)
                    t1 = pi_over_period*sin(2.0_dp*argument)/(2.0_dp*distance)
                    t2 = pi_over_period*(2.0_dp*pi_over_period*distance* &
                        cos(2.0_dp*argument) - sin(2.0_dp*argument))/(4.0_dp*distance**3)
                    t3 = pi_over_period*( &
                        3.0_dp*sin(2.0_dp*argument) - 6.0_dp*pi_over_period*distance* &
                        cos(2.0_dp*argument) - 4.0_dp*pi_over_period**2*distance**2* &
                        sin(2.0_dp*argument))/(8.0_dp*distance**5)
                end if
                t = sin(argument)**2
                value = variance*exp(-b*t)
                p = -b*t1*value
                p2 = (b*b*t1*t1 - b*t2)*value
                p3 = (-b**3*t1**3 + 3.0_dp*b*b*t1*t2 - b*t3)*value
            end if
            gradient_x1 = 2.0_dp*p*difference
            gradient_x2 = -gradient_x1
            dot_difference = dot_product(difference, delta)
            value_dot = dot_product(gradient_x1, delta)
            gradient_x1_dot = 2.0_dp*p*delta + 4.0_dp*p2*dot_difference*difference
            gradient_x2_dot = -gradient_x1_dot
            do i = 1, size(x1)
                do j = 1, size(x2)
                    mixed_hessian(i, j) = -2.0_dp*p*merge(1.0_dp, 0.0_dp, i == j) - &
                        4.0_dp*p2*difference(i)*difference(j)
                    mixed_hessian_dot(i, j) = -4.0_dp*p2*dot_difference* &
                        merge(1.0_dp, 0.0_dp, i == j) - 8.0_dp*p3*dot_difference* &
                        difference(i)*difference(j) - 4.0_dp*p2*(delta(i)*difference(j) + &
                        difference(i)*delta(j))
                end do
            end do
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        if (kernel%kind /= KERNEL_MATERN12 .and. kernel%kind /= KERNEL_MATERN32 .and. &
            kernel%kind /= KERNEL_MATERN52) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "derivative GP query products: leaf has no third-input contract")
            return
        end if
        variance = exp(kernel%log_parameters(1))
        lengthscale = exp(kernel%log_parameters(2))
        if (distance == 0.0_dp .and. kernel%kind == KERNEL_MATERN12) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "derivative GP query products: Matern 1/2 is singular at coincidence")
            return
        end if
        if (kernel%kind == KERNEL_MATERN12) then
            exponential = exp(-distance/lengthscale)
            value = variance*exponential
            radial_first = -value/lengthscale
            radial_second = value/(lengthscale*lengthscale)
            radial_third = -value/(lengthscale**3)
        else if (kernel%kind == KERNEL_MATERN32) then
            a = sqrt(3.0_dp)
            z = distance/lengthscale
            exponential = exp(-a*z)
            value = variance*(1.0_dp + a*z)*exponential
            radial_first = -3.0_dp*variance*z*exponential/lengthscale
            radial_second = 3.0_dp*variance*exponential*(a*z - 1.0_dp)/(lengthscale**2)
            radial_third = 3.0_dp*a*variance*exponential*(2.0_dp - a*z)/(lengthscale**3)
            if (distance == 0.0_dp .and. dot_product(delta, delta) > 0.0_dp) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "derivative GP query products: Matern 3/2 third derivative at coincidence")
                return
            end if
        else
            a = sqrt(5.0_dp)
            z = distance/lengthscale
            exponential = exp(-a*z)
            value = variance*(1.0_dp + a*z + 5.0_dp*z*z/3.0_dp)*exponential
            radial_first = -(5.0_dp/3.0_dp)*variance*z*(1.0_dp + a*z)*exponential/lengthscale
            radial_second = (5.0_dp/3.0_dp)*variance*exponential*(5.0_dp*z*z - a*z - 1.0_dp)/ &
                (lengthscale**2)
            radial_third = (25.0_dp/3.0_dp)*variance*z*(3.0_dp - a*z)*exponential/ &
                (lengthscale**3)
        end if
        if (distance == 0.0_dp) then
            if (kernel%kind == KERNEL_MATERN32) then
                radial_scale = -3.0_dp*variance/(lengthscale*lengthscale)
            else if (kernel%kind == KERNEL_MATERN52) then
                radial_scale = -5.0_dp*variance/(3.0_dp*lengthscale*lengthscale)
            else
                radial_scale = 0.0_dp
            end if
            gradient_x1 = 0.0_dp
            gradient_x2 = 0.0_dp
            do i = 1, size(x1)
                mixed_hessian(i, i) = -radial_scale
            end do
            value_dot = 0.0_dp
            gradient_x1_dot = 0.0_dp
            gradient_x2_dot = 0.0_dp
            mixed_hessian_dot = 0.0_dp
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        radial_scale = radial_first/distance
        radial_coefficient = (radial_second - radial_scale)/(distance*distance)
        radial_third = radial_third/(distance**3) - 3.0_dp*radial_coefficient/(distance*distance)
        value_dot = radial_scale*dot_product(difference, delta)
        gradient_x1 = radial_scale*difference
        gradient_x2 = -gradient_x1
        gradient_x1_dot = radial_scale*delta + radial_coefficient* &
            dot_product(difference, delta)*difference
        gradient_x2_dot = -gradient_x1_dot
        do i = 1, size(x1)
            do j = 1, size(x2)
                mixed_hessian(i, j) = -(radial_scale*merge(1.0_dp, 0.0_dp, i == j) + &
                    radial_coefficient*difference(i)*difference(j))
                mixed_hessian_dot(i, j) = -(radial_coefficient*dot_product(difference, delta)* &
                    merge(1.0_dp, 0.0_dp, i == j) + radial_third*dot_product(difference, delta)* &
                    difference(i)*difference(j) + radial_coefficient*(delta(i)*difference(j) + &
                    difference(i)*delta(j)))
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine smooth_leaf_input_third_direction

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

    function diagonal(matrix) result(values)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), allocatable :: values(:)
        integer :: i

        allocate(values(min(size(matrix, 1), size(matrix, 2))))
        do i = 1, size(values)
            values(i) = matrix(i, i)
        end do
    end function diagonal

end module fortml_derivative_gaussian_process
