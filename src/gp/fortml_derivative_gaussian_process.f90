module fortml_derivative_gaussian_process
    use fortnum_kinds, only: dp
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortml_kernels, only: kernel_t, KERNEL_WHITE_NOISE
    implicit none
    private

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
    end type gp_derivative_regression_t

    public :: gp_derivative_fit
    public :: gp_derivative_predict

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
        real(dp), allocatable :: covariance(:, :)
        integer :: i, j

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

        self%kernel = kernel
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

        allocate(covariance(self%n_observations, self%n_observations))
        do j = 1, self%n_observations
            do i = 1, self%n_observations
                call derivative_covariance( &
                    self%kernel, self%x_train(i, :), self%components(i), &
                    self%x_train(j, :), self%components(j), covariance(i, j), &
                    status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        do i = 1, self%n_observations
            covariance(i, i) = covariance(i, i) + self%noise_variance + self%jitter
        end do
        call self%factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(self%alpha, source=self%y_train)
        call self%factorization%solve(self%alpha, status)
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

    integer function gp_derivative_observation_count(self) result(count)
        class(gp_derivative_regression_t), intent(in) :: self

        count = self%n_observations
    end function gp_derivative_observation_count

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
        if (kernel%kind == KERNEL_WHITE_NOISE) then
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

    logical function derivative_gp_fitted(self) result(fitted)
        class(gp_derivative_regression_t), intent(in) :: self

        fitted = allocated(self%x_train) .and. allocated(self%y_train) .and. &
            allocated(self%alpha) .and. allocated(self%components)
        if (.not. fitted) return
        fitted = self%n_observations > 0 .and. self%n_features > 0 .and. &
            self%n_outputs > 0 .and. size(self%components) == self%n_observations
    end function derivative_gp_fitted

end module fortml_derivative_gaussian_process
