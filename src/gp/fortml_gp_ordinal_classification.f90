module fortml_gp_ordinal_classification
    !! Ordered Gaussian-process classification through a latent Gaussian GP.
    !!
    !! This bounded ordinal contract uses one zero-mean GP for the ordered
    !! class score.  Integer classes are mapped to ranks, the GP is fit to
    !! those ranks with a Gaussian observation model, and predictive class
    !! probabilities are adjacent normal-CDF differences at fixed mid-rank
    !! cut points.  This gives a stable ordered baseline with the same kernel,
    !! parameter, and input-product contracts as GP regression.  It is
    !! intentionally explicit about being a latent-Gaussian surrogate rather
    !! than a Laplace approximation to a cumulative likelihood.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t
    use fortml_gaussian_process, only: gp_regression_t
    implicit none
    private

    real(dp), parameter :: SQRT_TWO = 1.4142135623730950488016887242097_dp
    real(dp), parameter :: SQRT_TWO_PI = 2.506628274631000502415765284811_dp
    real(dp), parameter :: MIN_SCALE = 1.0e-12_dp

    type, public :: gp_ordinal_classification_options_t
        !! Controls the latent Gaussian fit and predictive uncertainty.
        real(dp) :: noise_variance = 0.05_dp
        real(dp) :: jitter = 1.0e-8_dp
    end type gp_ordinal_classification_options_t

    type, public :: gp_ordinal_classification_state_t
        integer :: class_count = 0
        integer :: iterations = 1
        logical :: converged = .false.
        real(dp) :: noise_variance = 0.0_dp
    end type gp_ordinal_classification_state_t

    type, public :: gp_ordinal_classification_t
        private
        type(gp_regression_t) :: latent
        integer, allocatable :: class_label(:)
        real(dp), allocatable :: cut_points(:)
        integer :: n_classes = 0
        integer :: n_features = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => gp_ordinal_fit
        procedure, public :: predict_latent => gp_ordinal_predict_latent
        procedure, public :: predict_proba => gp_ordinal_predict_proba
        procedure, public :: predict => gp_ordinal_predict
        procedure, public :: predict_latent_parameter_jvp => &
            gp_ordinal_predict_latent_parameter_jvp
        procedure, public :: predict_latent_parameter_vjp => &
            gp_ordinal_predict_latent_parameter_vjp
        procedure, public :: predict_proba_parameter_jvp => &
            gp_ordinal_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            gp_ordinal_predict_proba_parameter_vjp
        procedure, public :: predict_latent_input_jvp => &
            gp_ordinal_predict_latent_input_jvp
        procedure, public :: predict_latent_input_vjp => &
            gp_ordinal_predict_latent_input_vjp
        procedure, public :: predict_proba_input_jvp => &
            gp_ordinal_predict_proba_input_jvp
        procedure, public :: predict_proba_input_vjp => &
            gp_ordinal_predict_proba_input_vjp
        procedure, public :: predict_proba_device => &
            gp_ordinal_predict_proba_device
        procedure, public :: predict_proba_parameter_vjp_device => &
            gp_ordinal_predict_proba_parameter_vjp_device
        procedure, public :: predict_proba_input_vjp_device => &
            gp_ordinal_predict_proba_input_vjp_device
        procedure, public :: classes => gp_ordinal_classes
        procedure, public :: thresholds => gp_ordinal_thresholds
        procedure, public :: class_count => gp_ordinal_class_count
        procedure, public :: feature_count => gp_ordinal_feature_count
        procedure, public :: parameter_count => gp_ordinal_parameter_count
        procedure, public :: hyperparameter_count => gp_ordinal_parameter_count
        procedure, public :: parameters => gp_ordinal_parameters
        procedure, public :: hyperparameters => gp_ordinal_parameters
        procedure, public :: set_parameters => gp_ordinal_set_parameters
        procedure, public :: set_hyperparameters => gp_ordinal_set_parameters
        procedure, public :: log_marginal_likelihood => &
            gp_ordinal_log_marginal_likelihood
        procedure, public :: log_marginal_likelihood_jvp => &
            gp_ordinal_log_marginal_likelihood_jvp
        procedure, public :: hyperparameter_gradient => &
            gp_ordinal_hyperparameter_gradient
        procedure, public :: hyperparameter_hvp => gp_ordinal_hyperparameter_hvp
        procedure, public :: hyperparameter_gradient_device => &
            gp_ordinal_hyperparameter_gradient_device
        procedure, public :: hyperparameter_hvp_device => &
            gp_ordinal_hyperparameter_hvp_device
        procedure, public :: fitted => gp_ordinal_fitted
        procedure, public :: device_supported => gp_ordinal_device_supported
    end type gp_ordinal_classification_t

contains

    subroutine gp_ordinal_fit(self, x, labels, kernel, status, options, state)
        class(gp_ordinal_classification_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), intent(in) :: kernel
        type(fortnum_status_t), intent(out) :: status
        type(gp_ordinal_classification_options_t), intent(in), optional :: options
        type(gp_ordinal_classification_state_t), intent(out), optional :: state
        type(gp_ordinal_classification_options_t) :: requested
        type(gp_ordinal_classification_state_t) :: result
        real(dp), allocatable :: targets(:, :)
        integer, allocatable :: unique_labels(:)
        integer :: i, j
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(gp_ordinal_classification_options_t) :: gp_ordinal_classification_options_t_default
        type(gp_ordinal_classification_state_t) :: gp_ordinal_classification_state_t_default

        result = gp_ordinal_classification_state_t_default
        if (present(state)) state = result
        requested = gp_ordinal_classification_options_t_default
        if (present(options)) requested = options
        if (.not. valid_options(requested)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: options are invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: inputs must be finite")
            return
        end if
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: kernel dimension is invalid")
            return
        end if
        call sorted_unique(labels, unique_labels)
        if (size(unique_labels) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: at least two ordered classes are required")
            return
        end if

        self%n_classes = size(unique_labels)
        self%n_features = size(x, 2)
        allocate(self%class_label(self%n_classes), self%cut_points(self%n_classes - 1))
        self%class_label = unique_labels
        do j = 1, self%n_classes - 1
            self%cut_points(j) = real(j, dp) + 0.5_dp
        end do
        allocate(targets(size(x, 1), 1))
        do i = 1, size(x, 1)
            j = index_of(unique_labels, labels(i))
            if (j < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP fit: class label mapping failed")
                return
            end if
            targets(i, 1) = real(j, dp)
        end do
        call self%latent%fit(x, targets, kernel, requested%noise_variance, status, &
            jitter=requested%jitter)
        if (status%code /= FORTNUM_OK) then
            self%is_fitted = .false.
            return
        end if
        self%is_fitted = .true.
        result%class_count = self%n_classes
        result%iterations = 1
        result%converged = .true.
        result%noise_variance = requested%noise_variance
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_fit

    subroutine gp_ordinal_predict_latent(self, x, mean, variance, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_mean(:, :)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP prediction: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(mean) /= size(x, 1) .or. &
                size(variance) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP prediction: output shape is invalid")
            return
        end if
        allocate(latent_mean(size(x, 1), 1))
        call self%latent%predict(x, latent_mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        mean = latent_mean(:, 1)
    end subroutine gp_ordinal_predict_latent

    subroutine gp_ordinal_predict_proba(self, x, probabilities, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability prediction: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability prediction: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), variance(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probabilities(mean, variance, self%cut_points, probabilities, status)
    end subroutine gp_ordinal_predict_proba

    subroutine gp_ordinal_predict(self, x, labels, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. self%is_fitted .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP label prediction: model or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict

    subroutine gp_ordinal_predict_latent_parameter_jvp(self, x, direction, mean, &
            mean_dot, variance, variance_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_mean(:, :), latent_mean_dot(:, :)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent parameter JVP: model is not fitted")
            return
        end if
        if (size(mean) /= size(x, 1) .or. size(mean_dot) /= size(x, 1) .or. &
                size(variance) /= size(x, 1) .or. size(variance_dot) /= size(x, 1) .or. &
                size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent parameter JVP: shape is invalid")
            return
        end if
        allocate(latent_mean(size(x, 1), 1), latent_mean_dot(size(x, 1), 1))
        call self%latent%predict_jvp(x, direction, latent_mean, latent_mean_dot, &
            variance, variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        mean = latent_mean(:, 1)
        mean_dot = latent_mean_dot(:, 1)
    end subroutine gp_ordinal_predict_latent_parameter_jvp

    subroutine gp_ordinal_predict_latent_parameter_vjp(self, x, mean_bar, variance_bar, &
            parameter_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean_bar_matrix(:, :)

        parameter_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent parameter VJP: model is not fitted")
            return
        end if
        if (size(mean_bar) /= size(x, 1) .or. size(variance_bar) /= size(x, 1) .or. &
                size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent parameter VJP: shape is invalid")
            return
        end if
        allocate(mean_bar_matrix(size(x, 1), 1))
        mean_bar_matrix(:, 1) = mean_bar
        call self%latent%predict_vjp(x, mean_bar_matrix, variance_bar, parameter_bar, status)
    end subroutine gp_ordinal_predict_latent_parameter_vjp

    subroutine gp_ordinal_predict_proba_parameter_jvp(self, x, direction, probabilities, &
            probabilities_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)

        if (any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
                any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability parameter JVP: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_parameter_jvp(x, direction, mean, mean_dot, variance, &
            variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probabilities_jvp(mean, mean_dot, variance, variance_dot, &
            self%cut_points, probabilities, probabilities_dot, status)
    end subroutine gp_ordinal_predict_proba_parameter_jvp

    subroutine gp_ordinal_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            parameter_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)

        parameter_bar = 0.0_dp
        if (any(shape(probabilities_bar) /= [size(x, 1), self%n_classes]) .or. &
                size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability parameter VJP: shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), variance(size(x, 1)), mean_bar(size(x, 1)), &
            variance_bar(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probability_vjp(mean, variance, self%cut_points, probabilities_bar, &
            mean_bar, variance_bar, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_latent_parameter_vjp(x, mean_bar, variance_bar, parameter_bar, status)
    end subroutine gp_ordinal_predict_proba_parameter_vjp

    subroutine gp_ordinal_predict_latent_input_jvp(self, x, x_dot, mean, mean_dot, &
            variance, variance_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_mean(:, :), cross(:, :), cross_dot(:, :), work(:, :)
        real(dp), allocatable :: work_dot(:, :)
        real(dp) :: value, grad_x1(self%n_features), grad_x2(self%n_features)
        real(dp), allocatable :: hessian(:, :)
        real(dp) :: prior_dot
        integer :: i, j

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input JVP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. any(shape(x_dot) /= shape(x)) .or. &
                size(mean) /= size(x, 1) .or. size(mean_dot) /= size(x, 1) .or. &
                size(variance) /= size(x, 1) .or. size(variance_dot) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input JVP: shape is invalid")
            return
        end if
        allocate(latent_mean(size(x, 1), 1), cross(self%latent%n_samples, size(x, 1)), &
            cross_dot(self%latent%n_samples, size(x, 1)), work(self%latent%n_samples, size(x, 1)), &
            work_dot(self%latent%n_samples, size(x, 1)), hessian(self%n_features, self%n_features))
        call self%latent%predict(x, latent_mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call self%latent%kernel%matrix(self%latent%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        work = cross
        call self%latent%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        cross_dot = 0.0_dp
        variance_dot = 0.0_dp
        mean_dot = 0.0_dp
        do j = 1, size(x, 1)
            do i = 1, self%latent%n_samples
                call self%latent%kernel%input_derivatives(x(j, :), self%latent%x_train(i, :), &
                    value, grad_x1, grad_x2, hessian, status)
                if (status%code /= FORTNUM_OK) return
                cross_dot(i, j) = dot_product(grad_x1, x_dot(j, :))
                mean_dot(j) = mean_dot(j) + cross_dot(i, j)*self%latent%alpha(i, 1)
            end do
            call self%latent%kernel%input_derivatives(x(j, :), x(j, :), value, grad_x1, &
                grad_x2, hessian, status)
            if (status%code /= FORTNUM_OK) return
            prior_dot = dot_product(grad_x1 + grad_x2, x_dot(j, :))
            variance_dot(j) = prior_dot - sum(cross_dot(:, j)*work(:, j))
        end do
        work_dot = cross_dot
        call self%latent%factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(x, 1)
            variance_dot(j) = variance_dot(j) - sum(cross(:, j)*work_dot(:, j))
        end do
        mean = latent_mean(:, 1)
        if (any(.not. ieee_is_finite(mean_dot)) .or. any(.not. ieee_is_finite(variance_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input JVP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_latent_input_jvp

    subroutine gp_ordinal_predict_latent_input_vjp(self, x, mean_bar, variance_bar, x_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_mean(:, :), variance(:)
        real(dp), allocatable :: cross(:, :), work(:, :)
        real(dp) :: value, grad_x1(self%n_features), grad_x2(self%n_features)
        real(dp), allocatable :: hessian(:, :)
        real(dp) :: prior_bar
        integer :: i, j, k

        x_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(mean_bar) /= size(x, 1) .or. &
                size(variance_bar) /= size(x, 1) .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input VJP: shape is invalid")
            return
        end if
        allocate(latent_mean(size(x, 1), 1), variance(size(x, 1)), &
            cross(self%latent%n_samples, size(x, 1)), &
            work(self%latent%n_samples, size(x, 1)), hessian(self%n_features, self%n_features))
        call self%latent%predict(x, latent_mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call self%latent%kernel%matrix(self%latent%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        work = cross
        call self%latent%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(x, 1)
            do k = 1, self%n_features
                x_bar(j, k) = 0.0_dp
                do i = 1, self%latent%n_samples
                    call self%latent%kernel%input_derivatives(x(j, :), self%latent%x_train(i, :), &
                        value, grad_x1, grad_x2, hessian, status)
                    if (status%code /= FORTNUM_OK) return
                    x_bar(j, k) = x_bar(j, k) + (mean_bar(j)*self%latent%alpha(i, 1) - &
                        2.0_dp*variance_bar(j)*work(i, j))*grad_x1(k)
                end do
            end do
            call self%latent%kernel%input_derivatives(x(j, :), x(j, :), value, grad_x1, &
                grad_x2, hessian, status)
            if (status%code /= FORTNUM_OK) return
            prior_bar = variance_bar(j)
            do k = 1, self%n_features
                x_bar(j, k) = x_bar(j, k) + prior_bar*(grad_x1(k) + grad_x2(k))
            end do
        end do
        if (any(.not. ieee_is_finite(x_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input VJP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_latent_input_vjp

    subroutine gp_ordinal_predict_proba_input_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)

        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_input_jvp(x, x_dot, mean, mean_dot, variance, variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probabilities_jvp(mean, mean_dot, variance, variance_dot, &
            self%cut_points, probabilities, probabilities_dot, status)
    end subroutine gp_ordinal_predict_proba_input_jvp

    subroutine gp_ordinal_predict_proba_input_vjp(self, x, probabilities_bar, x_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)

        x_bar = 0.0_dp
        allocate(mean(size(x, 1)), variance(size(x, 1)), mean_bar(size(x, 1)), &
            variance_bar(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probability_vjp(mean, variance, self%cut_points, probabilities_bar, &
            mean_bar, variance_bar, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_latent_input_vjp(x, mean_bar, variance_bar, x_bar, status)
    end subroutine gp_ordinal_predict_proba_input_vjp

    subroutine gp_ordinal_predict_proba_device(self, device, x, probabilities, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP device prediction: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP device prediction: device kind is invalid")
        end select
    end subroutine gp_ordinal_predict_proba_device

    subroutine gp_ordinal_predict_proba_parameter_vjp_device(self, device, x, &
            probabilities_bar, parameter_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP parameter VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_vjp(x, probabilities_bar, parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP parameter VJP device: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP parameter VJP device: device kind is invalid")
        end select
    end subroutine gp_ordinal_predict_proba_parameter_vjp_device

    subroutine gp_ordinal_predict_proba_input_vjp_device(self, device, x, &
            probabilities_bar, x_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        x_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP input VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_input_vjp(x, probabilities_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP input VJP device: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP input VJP device: device kind is invalid")
        end select
    end subroutine gp_ordinal_predict_proba_input_vjp_device

    function gp_ordinal_classes(self) result(labels)
        class(gp_ordinal_classification_t), intent(in) :: self
        integer, allocatable :: labels(:)

        allocate(labels(self%n_classes))
        if (self%n_classes > 0) labels = self%class_label
    end function gp_ordinal_classes

    function gp_ordinal_thresholds(self) result(thresholds)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), allocatable :: thresholds(:)

        allocate(thresholds(max(0, self%n_classes - 1)))
        if (self%n_classes > 1) thresholds = self%cut_points
    end function gp_ordinal_thresholds

    integer function gp_ordinal_class_count(self) result(count)
        class(gp_ordinal_classification_t), intent(in) :: self

        count = self%n_classes
    end function gp_ordinal_class_count

    integer function gp_ordinal_feature_count(self) result(count)
        class(gp_ordinal_classification_t), intent(in) :: self

        count = self%n_features
    end function gp_ordinal_feature_count

    integer function gp_ordinal_parameter_count(self) result(count)
        class(gp_ordinal_classification_t), intent(in) :: self

        count = 0
        if (self%is_fitted) count = self%latent%parameter_count()
    end function gp_ordinal_parameter_count

    function gp_ordinal_parameters(self) result(parameters)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (self%is_fitted) then
            parameters = self%latent%parameters()
        else
            allocate(parameters(0))
        end if
    end function gp_ordinal_parameters

    subroutine gp_ordinal_set_parameters(self, parameters, status)
        class(gp_ordinal_classification_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP set_parameters: model is not fitted")
            return
        end if
        call self%latent%set_parameters(parameters, status)
    end subroutine gp_ordinal_set_parameters

    subroutine gp_ordinal_log_marginal_likelihood(self, value, status)
        !! Return the exact latent-Gaussian GP log marginal likelihood.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log marginal likelihood: model is not fitted")
            return
        end if
        call self%latent%log_marginal_likelihood(value, status)
    end subroutine gp_ordinal_log_marginal_likelihood

    subroutine gp_ordinal_log_marginal_likelihood_jvp(self, direction, value_dot, status)
        !! Directional product of the exact latent-Gaussian GP evidence.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: value_dot
        type(fortnum_status_t), intent(out) :: status

        value_dot = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log marginal likelihood JVP: model is not fitted")
            return
        end if
        call self%latent%log_marginal_likelihood_jvp(direction, value_dot, status)
    end subroutine gp_ordinal_log_marginal_likelihood_jvp

    subroutine gp_ordinal_hyperparameter_gradient(self, gradient, status)
        !! Exact gradient over packed kernel and log-noise hyperparameters.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%hyperparameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter gradient: output shape is invalid")
            return
        end if
        call self%latent%hyperparameter_gradient(gradient, status)
    end subroutine gp_ordinal_hyperparameter_gradient

    subroutine gp_ordinal_hyperparameter_hvp(self, direction, parameter_hvp, status)
        !! Exact directional Hessian product over kernel and log-noise values.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_hvp = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter HVP: model is not fitted")
            return
        end if
        if (size(direction) /= self%hyperparameter_count() .or. &
                size(parameter_hvp) /= self%hyperparameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter HVP: parameter shape is invalid")
            return
        end if
        call self%latent%hyperparameter_hvp(direction, parameter_hvp, status)
    end subroutine gp_ordinal_hyperparameter_hvp

    subroutine gp_ordinal_hyperparameter_gradient_device(self, device, gradient, status)
        !! Device-dispatch wrapper; CUDA remains an explicit typed refusal.
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter gradient device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hyperparameter_gradient(gradient, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP hyperparameter gradient device: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter gradient device: device kind is invalid")
        end select
    end subroutine gp_ordinal_hyperparameter_gradient_device

    subroutine gp_ordinal_hyperparameter_hvp_device(self, device, direction, parameter_hvp, status)
        !! Device-dispatch wrapper; CUDA remains an explicit typed refusal.
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_hvp = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter HVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hyperparameter_hvp(direction, parameter_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP hyperparameter HVP device: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter HVP device: device kind is invalid")
        end select
    end subroutine gp_ordinal_hyperparameter_hvp_device

    logical function gp_ordinal_fitted(self) result(value)
        class(gp_ordinal_classification_t), intent(in) :: self

        value = self%is_fitted
    end function gp_ordinal_fitted

    logical function gp_ordinal_device_supported(self, device_kind) result(value)
        class(gp_ordinal_classification_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            value = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            value = .false.
        case default
            value = .false.
        end select
    end function gp_ordinal_device_supported

    subroutine ordinal_probabilities(mean, variance, cut_points, probabilities, status)
        real(dp), intent(in) :: mean(:), variance(:), cut_points(:)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: scale, upper, lower
        integer :: i, j, n, k

        n = size(mean)
        k = size(cut_points) + 1
        if (size(variance) /= n .or. any(shape(probabilities) /= [n, k])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probabilities: shape is invalid")
            return
        end if
        do i = 1, n
            if (.not. ieee_is_finite(mean(i)) .or. .not. ieee_is_finite(variance(i)) .or. &
                    variance(i) < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP probabilities: latent state is invalid")
                return
            end if
            scale = sqrt(max(1.0_dp + variance(i), MIN_SCALE))
            lower = 0.0_dp
            do j = 1, k
                if (j < k) then
                    upper = normal_cdf((cut_points(j) - mean(i))/scale)
                else
                    upper = 1.0_dp
                end if
                probabilities(i, j) = max(0.0_dp, upper - lower)
                lower = upper
            end do
            if (abs(sum(probabilities(i, :)) - 1.0_dp) > 2.0e-14_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP probabilities: normalization failed")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_probabilities

    subroutine ordinal_probabilities_jvp(mean, mean_dot, variance, variance_dot, cut_points, &
            probabilities, probabilities_dot, status)
        real(dp), intent(in) :: mean(:), mean_dot(:), variance(:), variance_dot(:), cut_points(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: scale, z, cdf, cdf_dot, lower, lower_dot
        integer :: i, j, n, k

        n = size(mean)
        k = size(cut_points) + 1
        call ordinal_probabilities(mean, variance, cut_points, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        if (size(mean_dot) /= n .or. size(variance_dot) /= n .or. &
                any(shape(probabilities_dot) /= [n, k])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability JVP: shape is invalid")
            return
        end if
        probabilities_dot = 0.0_dp
        do i = 1, n
            scale = sqrt(max(1.0_dp + variance(i), MIN_SCALE))
            lower = 0.0_dp
            lower_dot = 0.0_dp
            do j = 1, k
                if (j < k) then
                    z = (cut_points(j) - mean(i))/scale
                    cdf = normal_cdf(z)
                    cdf_dot = normal_pdf(z)*(-mean_dot(i)/scale - &
                        0.5_dp*(cut_points(j) - mean(i))*variance_dot(i)/(scale**3))
                else
                    cdf = 1.0_dp
                    cdf_dot = 0.0_dp
                end if
                probabilities_dot(i, j) = cdf_dot - lower_dot
                lower = cdf
                lower_dot = cdf_dot
            end do
        end do
        if (any(.not. ieee_is_finite(probabilities_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability JVP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_probabilities_jvp

    subroutine ordinal_probability_vjp(mean, variance, cut_points, probabilities_bar, &
            mean_bar, variance_bar, status)
        real(dp), intent(in) :: mean(:), variance(:), cut_points(:), probabilities_bar(:, :)
        real(dp), intent(out) :: mean_bar(:), variance_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: scale, z, density, boundary_bar
        integer :: i, j, n, k

        n = size(mean)
        k = size(cut_points) + 1
        mean_bar = 0.0_dp
        variance_bar = 0.0_dp
        if (size(variance) /= n .or. size(mean_bar) /= n .or. size(variance_bar) /= n .or. &
                any(shape(probabilities_bar) /= [n, k])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability VJP: shape is invalid")
            return
        end if
        do i = 1, n
            scale = sqrt(max(1.0_dp + variance(i), MIN_SCALE))
            do j = 1, k - 1
                z = (cut_points(j) - mean(i))/scale
                density = normal_pdf(z)
                boundary_bar = probabilities_bar(i, j) - probabilities_bar(i, j + 1)
                mean_bar(i) = mean_bar(i) - boundary_bar*density/scale
                variance_bar(i) = variance_bar(i) - boundary_bar*density* &
                    (cut_points(j) - mean(i))/(2.0_dp*scale**3)
            end do
        end do
        if (any(.not. ieee_is_finite(mean_bar)) .or. any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability VJP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_probability_vjp

    real(dp) function normal_cdf(value) result(output)
        real(dp), intent(in) :: value

        output = 0.5_dp*erfc(-value/SQRT_TWO)
    end function normal_cdf

    real(dp) function normal_pdf(value) result(output)
        real(dp), intent(in) :: value

        output = exp(-0.5_dp*value*value)/SQRT_TWO_PI
    end function normal_pdf

    logical function valid_options(options) result(value)
        type(gp_ordinal_classification_options_t), intent(in) :: options

        value = options%noise_variance > 0.0_dp .and. options%jitter > 0.0_dp
        if (value) value = ieee_is_finite(options%noise_variance) .and. &
            ieee_is_finite(options%jitter)
    end function valid_options

    subroutine sorted_unique(labels, unique_labels)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: unique_labels(:)
        integer, allocatable :: work(:)
        integer :: i, j, count, value

        allocate(work(size(labels)))
        work = labels
        do i = 2, size(work)
            value = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= value) exit
                work(j + 1) = work(j)
                j = j - 1
            end do
            work(j + 1) = value
        end do
        count = 0
        do i = 1, size(work)
            if (i == 1 .or. work(i) /= work(i - 1)) count = count + 1
        end do
        allocate(unique_labels(count))
        count = 0
        do i = 1, size(work)
            if (i == 1 .or. work(i) /= work(i - 1)) then
                count = count + 1
                unique_labels(count) = work(i)
            end if
        end do
    end subroutine sorted_unique

    integer function index_of(labels, value) result(index)
        integer, intent(in) :: labels(:), value
        integer :: i

        index = 0
        do i = 1, size(labels)
            if (labels(i) == value) then
                index = i
                return
            end if
        end do
    end function index_of

end module fortml_gp_ordinal_classification
