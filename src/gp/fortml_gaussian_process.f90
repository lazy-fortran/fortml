module fortml_gaussian_process
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t
    implicit none
    private

    real(dp), parameter :: LOG_TWO_PI = 1.837877066409345483560659472811_dp

    type, public :: gp_regression_t
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: y_train(:, :)
        real(dp), allocatable :: alpha(:, :)
        real(dp) :: log_noise_variance = log(1.0e-6_dp)
        real(dp) :: jitter = 1.0e-10_dp
        integer :: n_samples = 0
        integer :: n_features = 0
        integer :: n_outputs = 0
    contains
        procedure, public :: fit => gp_fit
        procedure, public :: parameter_count => gp_parameter_count
        procedure, public :: parameters => gp_parameters
        procedure, public :: set_parameters => gp_set_parameters
        procedure, public :: predict => gp_predict
        procedure, public :: predict_jvp => gp_predict_jvp
        procedure, public :: predict_vjp => gp_predict_vjp
        procedure, public :: log_marginal_likelihood => gp_log_marginal_likelihood
        procedure, public :: hyperparameter_gradient => gp_hyperparameter_gradient
        procedure, public :: log_marginal_likelihood_jvp => gp_lml_jvp
    end type gp_regression_t

    public :: gp_fit
    public :: gp_predict
    public :: gp_predict_jvp
    public :: gp_predict_vjp
    public :: gp_log_marginal_likelihood
    public :: gp_hyperparameter_gradient

contains

    subroutine gp_fit(self, x, y, kernel, noise_variance, status, jitter)
        class(gp_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: jitter

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(y, 1) /= size(x, 1) .or. &
            size(y, 2) < 1 .or. noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP fit: invalid data dimensions or noise variance")
            return
        end if
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP fit: kernel input dimension or parameters are invalid")
            return
        end if

        self%kernel = kernel
        allocate(self%x_train(size(x, 1), size(x, 2)))
        allocate(self%y_train(size(y, 1), size(y, 2)))
        self%x_train = x
        self%y_train = y
        self%n_samples = size(x, 1)
        self%n_features = size(x, 2)
        self%n_outputs = size(y, 2)
        self%log_noise_variance = log(noise_variance)
        self%jitter = 1.0e-10_dp
        if (present(jitter)) self%jitter = jitter
        if (self%jitter < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP fit: jitter must be nonnegative")
            return
        end if
        call gp_refactor(self, status)
    end subroutine gp_fit

    integer function gp_parameter_count(self) result(count)
        class(gp_regression_t), intent(in) :: self

        count = self%kernel%parameter_count() + 1
    end function gp_parameter_count

    function gp_parameters(self) result(parameters)
        class(gp_regression_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: kernel_count

        kernel_count = self%kernel%parameter_count()
        allocate(parameters(kernel_count + 1))
        if (kernel_count > 0) parameters(:kernel_count) = self%kernel%parameters()
        parameters(kernel_count + 1) = self%log_noise_variance
    end function gp_parameters

    subroutine gp_set_parameters(self, parameters, status)
        class(gp_regression_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: kernel_count

        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP set_parameters: model is not fitted")
            return
        end if
        if (size(parameters) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP set_parameters: parameter shape is invalid")
            return
        end if
        kernel_count = self%kernel%parameter_count()
        call self%kernel%set_parameters(parameters(:kernel_count), status)
        if (status%code /= FORTNUM_OK) return
        self%log_noise_variance = parameters(kernel_count + 1)
        call gp_refactor(self, status)
    end subroutine gp_set_parameters

    subroutine gp_predict(self, x, mean, variance, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:, :), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior(:, :), work(:, :)
        integer :: i

        call check_prediction_shapes(self, x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(prior(size(x, 1), size(x, 1)))
        allocate(work, mold=cross)
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix(x, x, prior, status)
        if (status%code /= FORTNUM_OK) return
        mean = matmul(transpose(cross), self%alpha)
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        variance = diagonal(prior) - sum(cross*work, dim=1)
        do i = 1, size(variance)
            if (variance(i) < 0.0_dp) then
                if (variance(i) < -1.0e-9_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "GP predict: posterior variance is not positive")
                    return
                end if
                variance(i) = 0.0_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict

    subroutine gp_predict_jvp(self, x, direction, mean, mean_dot, variance, &
            variance_dot, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), prior(:, :), prior_dot(:, :)
        real(dp), allocatable :: train_dot(:, :), train_matrix_dot(:, :)
        real(dp), allocatable :: alpha_dot(:, :), work(:, :)
        real(dp), allocatable :: work_dot(:, :)
        real(dp) :: noise_dot
        integer :: i, kernel_count

        call check_prediction_shapes(self, x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        if (any(shape(mean_dot) /= shape(mean)) .or. size(variance_dot) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_jvp: tangent output shape is invalid")
            return
        end if
        if (size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_jvp: parameter tangent shape is invalid")
            return
        end if

        kernel_count = self%kernel%parameter_count()
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(cross_dot, mold=cross)
        allocate(prior(size(x, 1), size(x, 1)))
        allocate(prior_dot, mold=prior)
        allocate(train_dot(self%n_samples, self%n_samples))
        allocate(train_matrix_dot, mold=train_dot)
        allocate(alpha_dot, mold=self%alpha)
        allocate(work, mold=cross)
        allocate(work_dot, mold=cross)
        call self%kernel%matrix_jvp(self%x_train, x, direction(:kernel_count), &
            cross, cross_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix_jvp(self%x_train, self%x_train, &
            direction(:kernel_count), train_dot, train_matrix_dot, status)
        if (status%code /= FORTNUM_OK) return
        noise_dot = exp(self%log_noise_variance)*direction(kernel_count + 1)
        do i = 1, self%n_samples
            train_matrix_dot(i, i) = train_matrix_dot(i, i) + noise_dot
        end do
        alpha_dot = -matmul(train_matrix_dot, self%alpha)
        call self%factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return
        mean = matmul(transpose(cross), self%alpha)
        mean_dot = matmul(transpose(cross_dot), self%alpha) + &
            matmul(transpose(cross), alpha_dot)
        call self%kernel%matrix_jvp(x, x, direction(:kernel_count), prior, prior_dot, status)
        if (status%code /= FORTNUM_OK) return
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        work_dot = cross_dot - matmul(train_matrix_dot, work)
        call self%factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        variance = diagonal(prior) - sum(cross*work, dim=1)
        variance_dot = diagonal(prior_dot) - sum(cross_dot*work + &
            cross*work_dot, dim=1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict_jvp

    subroutine gp_predict_vjp(self, x, mean_bar, variance_bar, parameter_bar, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:, :), variance_bar(:)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), cross_bar(:, :)
        real(dp), allocatable :: train_bar(:, :), prior_bar(:, :), alpha_bar(:, :)
        real(dp), allocatable :: lambda(:, :), local_bar(:)
        integer :: i, kernel_count

        call check_prediction_shapes(self, x, mean_bar, variance_bar, status)
        if (status%code /= FORTNUM_OK) return
        if (size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP predict_vjp: parameter output shape is invalid")
            return
        end if

        kernel_count = self%kernel%parameter_count()
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(work, mold=cross)
        allocate(cross_bar, mold=cross)
        allocate(train_bar(self%n_samples, self%n_samples))
        allocate(prior_bar(size(x, 1), size(x, 1)))
        allocate(alpha_bar, mold=self%alpha)
        allocate(lambda, mold=self%alpha)
        allocate(local_bar(kernel_count))
        call self%kernel%matrix(self%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return

        cross_bar = matmul(self%alpha, transpose(mean_bar))
        alpha_bar = matmul(cross, mean_bar)
        prior_bar = 0.0_dp
        do i = 1, size(x, 1)
            prior_bar(i, i) = variance_bar(i)
        end do
        cross_bar = cross_bar - 2.0_dp*work*spread(variance_bar, dim=1, &
            ncopies=self%n_samples)
        train_bar = matmul(work*spread(variance_bar, dim=1, ncopies=self%n_samples), &
            transpose(work))
        lambda = alpha_bar
        call self%factorization%solve(lambda, status)
        if (status%code /= FORTNUM_OK) return
        train_bar = train_bar - 0.5_dp*(matmul(lambda, transpose(self%alpha)) + &
            matmul(self%alpha, transpose(lambda)))
        parameter_bar = 0.0_dp
        call self%kernel%parameter_vjp(self%x_train, self%x_train, train_bar, &
            local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kernel_count) = parameter_bar(:kernel_count) + local_bar
        call self%kernel%parameter_vjp(self%x_train, x, cross_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kernel_count) = parameter_bar(:kernel_count) + local_bar
        call self%kernel%parameter_vjp(x, x, prior_bar, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kernel_count) = parameter_bar(:kernel_count) + local_bar
        parameter_bar(kernel_count + 1) = exp(self%log_noise_variance)* &
            sum(diagonal(train_bar))
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_predict_vjp

    subroutine gp_log_marginal_likelihood(self, value, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: logdet

        value = 0.0_dp
        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP log marginal likelihood: model is not fitted")
            return
        end if
        call self%factorization%log_determinant(logdet, status)
        if (status%code /= FORTNUM_OK) return
        value = -0.5_dp*sum(self%y_train*self%alpha) - &
            0.5_dp*real(self%n_outputs, dp)*logdet - &
            0.5_dp*real(self%n_samples*self%n_outputs, dp)*LOG_TWO_PI
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_log_marginal_likelihood

    subroutine gp_hyperparameter_gradient(self, gradient, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: identity(:, :), inverse(:, :), matrix_bar(:, :)
        integer :: i, kernel_count

        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP hyperparameter_gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP hyperparameter_gradient: output shape is invalid")
            return
        end if
        allocate(identity(self%n_samples, self%n_samples))
        allocate(inverse, mold=identity)
        allocate(matrix_bar, mold=identity)
        identity = 0.0_dp
        do i = 1, self%n_samples
            identity(i, i) = 1.0_dp
        end do
        inverse = identity
        call self%factorization%solve(inverse, status)
        if (status%code /= FORTNUM_OK) return
        matrix_bar = 0.5_dp*(matmul(self%alpha, transpose(self%alpha)) - &
            real(self%n_outputs, dp)*inverse)
        kernel_count = self%kernel%parameter_count()
        call self%kernel%parameter_vjp(self%x_train, self%x_train, matrix_bar, &
            gradient(:kernel_count), status)
        if (status%code /= FORTNUM_OK) return
        gradient(kernel_count + 1) = exp(self%log_noise_variance)* &
            sum(diagonal(matrix_bar))
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_hyperparameter_gradient

    subroutine gp_lml_jvp(self, direction, value_dot, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value_dot = 0.0_dp
        if (size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP log marginal likelihood_jvp: direction shape is invalid")
            return
        end if
        allocate(gradient(size(direction)))
        call self%hyperparameter_gradient(gradient, status)
        if (status%code /= FORTNUM_OK) return
        value_dot = dot_product(gradient, direction)
    end subroutine gp_lml_jvp

    subroutine gp_refactor(self, status)
        class(gp_regression_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance(:, :)
        real(dp) :: noise_variance
        integer :: i

        if (.not. allocated(self%x_train) .or. .not. allocated(self%y_train)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP refactor: training data is not initialized")
            return
        end if
        allocate(covariance(self%n_samples, self%n_samples))
        call self%kernel%matrix(self%x_train, self%x_train, covariance, status)
        if (status%code /= FORTNUM_OK) return
        noise_variance = exp(self%log_noise_variance)
        do i = 1, self%n_samples
            covariance(i, i) = covariance(i, i) + noise_variance + self%jitter
        end do
        call self%factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        if (allocated(self%alpha)) deallocate(self%alpha)
        allocate(self%alpha, source=self%y_train)
        call self%factorization%solve(self%alpha, status)
    end subroutine gp_refactor

    subroutine check_prediction_shapes(self, x, mean, variance, status)
        class(gp_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean(:, :), variance(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP prediction: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP prediction: input dimension is invalid")
            return
        end if
        if (any(shape(mean) /= [size(x, 1), self%n_outputs]) .or. &
            size(variance) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP prediction: output shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_prediction_shapes

    logical function gp_fitted(self) result(fitted)
        class(gp_regression_t), intent(in) :: self

        fitted = allocated(self%x_train)
        if (.not. fitted) return
        fitted = allocated(self%y_train)
        if (.not. fitted) return
        fitted = allocated(self%alpha)
        if (.not. fitted) return
        fitted = self%n_samples > 0 .and. self%n_features > 0 .and. &
            self%n_outputs > 0
    end function gp_fitted

    function diagonal(matrix) result(values)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), allocatable :: values(:)
        integer :: i

        allocate(values(min(size(matrix, 1), size(matrix, 2))))
        do i = 1, size(values)
            values(i) = matrix(i, i)
        end do
    end function diagonal

end module fortml_gaussian_process
