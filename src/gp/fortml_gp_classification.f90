module fortml_gp_classification
    !! Laplace-approximated binary Gaussian-process classification.
    !!
    !! The classifier supports Bernoulli logistic and probit likelihoods.  The
    !! latent mode is found with the standard damped Newton iteration and the
    !! posterior covariance is represented by the symmetric matrix
    !! ``I + sqrt(W) K sqrt(W)``.  Consequently prediction uses exactly the
    !! same kernel implementation as GP regression, including composite and
    !! user kernels that satisfy the kernel validation contract.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t
    implicit none
    private

    real(dp), parameter :: PI = 3.1415926535897932384626433832795_dp
    real(dp), parameter :: SQRT_TWO = 1.4142135623730950488016887242097_dp
    real(dp), parameter :: SQRT_TWO_PI = 2.506628274631000502415765284811_dp
    real(dp), parameter :: MIN_LIKELIHOOD_CURVATURE = 1.0e-12_dp

    integer, parameter, public :: GP_LIKELIHOOD_LOGISTIC = 1
    integer, parameter, public :: GP_LIKELIHOOD_PROBIT = 2

    type, public :: gp_classification_options_t
        integer :: likelihood = GP_LIKELIHOOD_LOGISTIC
        integer :: max_iterations = 50
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: jitter = 1.0e-8_dp
        real(dp) :: damping = 1.0_dp
    end type gp_classification_options_t

    type, public :: gp_classification_state_t
        integer :: iterations = 0
        real(dp) :: final_step_norm = huge(1.0_dp)
        real(dp) :: log_posterior = -huge(1.0_dp)
        logical :: converged = .false.
    end type gp_classification_state_t

    type, public :: gp_classification_t
        private
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: prior_factorization
        type(cholesky_factorization_t) :: posterior_factorization
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: covariance(:, :)
        real(dp), allocatable :: mode(:)
        real(dp), allocatable :: alpha(:)
        real(dp), allocatable :: sqrt_w(:)
        integer :: class_label(2) = 0
        integer :: n_samples = 0
        integer :: n_features = 0
        integer :: likelihood = GP_LIKELIHOOD_LOGISTIC
    contains
        procedure, public :: fit => gp_classification_fit
        procedure, public :: predict_latent => gp_classification_predict_latent
        procedure, public :: predict_latent_jvp => gp_classification_predict_latent_jvp
        procedure, public :: predict_proba => gp_classification_predict_proba
        procedure, public :: predict_proba_jvp => gp_classification_predict_proba_jvp
        procedure, public :: predict => gp_classification_predict
        procedure, public :: classes => gp_classification_classes
        procedure, public :: feature_count => gp_classification_feature_count
        procedure, public :: parameter_count => gp_classification_parameter_count
        procedure, public :: parameters => gp_classification_parameters
        procedure, public :: hyperparameter_gradient => &
            gp_classification_hyperparameter_gradient
        procedure, public :: fitted => gp_classification_fitted
        procedure, public :: likelihood_kind => gp_classification_likelihood
    end type gp_classification_t

    public :: gp_classification_fit
    public :: gp_classification_predict_latent
    public :: gp_classification_predict_latent_jvp
    public :: gp_classification_predict_proba
    public :: gp_classification_predict_proba_jvp
    public :: gp_classification_predict

contains

    subroutine gp_classification_fit(self, x, labels, kernel, status, options, state)
        class(gp_classification_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), intent(in) :: kernel
        type(fortnum_status_t), intent(out) :: status
        type(gp_classification_options_t), intent(in), optional :: options
        type(gp_classification_state_t), intent(out), optional :: state
        type(gp_classification_options_t) :: requested
        type(gp_classification_state_t) :: result
        real(dp), allocatable :: b(:), rhs(:), sqrt_w(:), mode_new(:)
        real(dp), allocatable :: matrix(:, :)
        real(dp) :: eta, probability, likelihood_gradient, curvature
        real(dp) :: step_norm, scale
        integer :: i, iteration

        result = gp_classification_state_t()
        if (present(state)) state = result
        requested = gp_classification_options_t()
        if (present(options)) requested = options
        if (.not. valid_options(requested)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: options are invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: inputs must be finite")
            return
        end if
        self%class_label = [minval(labels), maxval(labels)]
        if (self%class_label(1) == self%class_label(2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: exactly two classes are required")
            return
        end if
        do i = 1, size(labels)
            if (labels(i) /= self%class_label(1) .and. &
                labels(i) /= self%class_label(2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP classification fit: exactly two classes are required")
                return
            end if
        end do
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification fit: kernel dimension is invalid")
            return
        end if

        self%kernel = kernel
        allocate(self%x_train, source=x)
        self%n_samples = size(x, 1)
        self%n_features = size(x, 2)
        self%likelihood = requested%likelihood
        allocate(self%covariance(self%n_samples, self%n_samples))
        call self%kernel%matrix(self%x_train, self%x_train, self%covariance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, self%n_samples
            self%covariance(i, i) = self%covariance(i, i) + requested%jitter
        end do
        call self%prior_factorization%factorize(self%covariance, status)
        if (status%code /= FORTNUM_OK) return

        allocate(self%mode(self%n_samples), self%alpha(self%n_samples))
        allocate(self%sqrt_w(self%n_samples), b(self%n_samples), rhs(self%n_samples))
        allocate(mode_new(self%n_samples), sqrt_w(self%n_samples), &
            matrix(self%n_samples, self%n_samples))
        self%mode = 0.0_dp
        do iteration = 1, requested%max_iterations
            do i = 1, self%n_samples
                eta = encoded_label(labels(i), self%class_label)*self%mode(i)
                call likelihood_terms(eta, self%likelihood, probability, &
                    likelihood_gradient, curvature)
                sqrt_w(i) = sqrt(max(curvature, MIN_LIKELIHOOD_CURVATURE))
                b(i) = curvature*self%mode(i) + &
                    encoded_label(labels(i), self%class_label)*likelihood_gradient
            end do
            call posterior_system(self%covariance, sqrt_w, matrix)
            call self%posterior_factorization%factorize(matrix, status)
            if (status%code /= FORTNUM_OK) return
            rhs = sqrt_w*matmul(self%covariance, b)
            call self%posterior_factorization%solve(rhs, status)
            if (status%code /= FORTNUM_OK) return
            mode_new = matmul(self%covariance, b - sqrt_w*rhs)
            scale = max(1.0_dp, maxval(abs(self%mode)))
            step_norm = requested%damping*maxval(abs(mode_new - self%mode))/scale
            self%mode = self%mode + requested%damping*(mode_new - self%mode)
            result%iterations = iteration
            result%final_step_norm = step_norm
            if (step_norm <= requested%tolerance) then
                result%converged = .true.
                exit
            end if
        end do
        if (.not. result%converged) then
            if (present(state)) state = result
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification fit: Newton iteration limit reached")
            return
        end if

        ! Rebuild W and the posterior factorization at the damped final mode.
        do i = 1, self%n_samples
            eta = encoded_label(labels(i), self%class_label)*self%mode(i)
            call likelihood_terms(eta, self%likelihood, probability, &
                likelihood_gradient, curvature)
            self%sqrt_w(i) = sqrt(max(curvature, MIN_LIKELIHOOD_CURVATURE))
        end do
        call posterior_system(self%covariance, self%sqrt_w, matrix)
        call self%posterior_factorization%factorize(matrix, status)
        if (status%code /= FORTNUM_OK) return
        self%alpha = self%mode
        call self%prior_factorization%solve(self%alpha, status)
        if (status%code /= FORTNUM_OK) return
        result%log_posterior = log_posterior(self%mode, self%alpha, labels, &
            self%class_label, self%likelihood)
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_fit

    subroutine gp_classification_predict_latent(self, x, mean, variance, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status

        call predict_core(self, x, mean, variance, status)
    end subroutine gp_classification_predict_latent

    subroutine gp_classification_predict_latent_jvp( &
            self, x, x_dot, mean, mean_dot, variance, variance_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), work(:, :), work_dot(:, :)
        real(dp), allocatable :: prior(:), prior_dot(:)
        real(dp) :: value, value_dot
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        integer :: i, j

        if (.not. prediction_shapes(self, x, mean, variance, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. size(mean_dot) /= size(mean) .or. &
            size(variance_dot) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification JVP: input or output shape is invalid")
            return
        end if
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(cross_dot, mold=cross)
        allocate(work, mold=cross)
        allocate(work_dot, mold=cross)
        allocate(prior(size(x, 1)), prior_dot(size(x, 1)))
        allocate(gradient_x1(self%n_features), gradient_x2(self%n_features))
        allocate(mixed_hessian(self%n_features, self%n_features))
        do j = 1, size(x, 1)
            do i = 1, self%n_samples
                call self%kernel%input_derivatives(self%x_train(i, :), x(j, :), &
                    value, gradient_x1, gradient_x2, mixed_hessian, status)
                if (status%code /= FORTNUM_OK) return
                cross(i, j) = value
                cross_dot(i, j) = dot_product(gradient_x2, x_dot(j, :))
            end do
            call self%kernel%input_derivatives(x(j, :), x(j, :), value, &
                gradient_x1, gradient_x2, mixed_hessian, status)
            if (status%code /= FORTNUM_OK) return
            prior(j) = value
            prior_dot(j) = dot_product(gradient_x1 + gradient_x2, x_dot(j, :))
        end do
        mean = matmul(transpose(cross), self%alpha)
        mean_dot = matmul(transpose(cross_dot), self%alpha)
        call scale_rows(cross, self%sqrt_w, work)
        call scale_rows(cross_dot, self%sqrt_w, work_dot)
        call self%posterior_factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        call self%posterior_factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        variance = prior - sum(work*work, dim=1)
        variance_dot = prior_dot - 2.0_dp*sum(work*work_dot, dim=1)
        call clamp_variance(variance, status)
    end subroutine gp_classification_predict_latent_jvp

    subroutine gp_classification_predict_proba(self, x, probabilities, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)
        integer :: i

        if (.not. prediction_probability_shapes(self, x, probabilities, status)) return
        allocate(mean(size(x, 1)), variance(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities(i, 2) = predictive_probability(self%likelihood, mean(i), variance(i))
            probabilities(i, 1) = 1.0_dp - probabilities(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_proba

    subroutine gp_classification_predict_proba_jvp( &
            self, x, x_dot, probabilities, probabilities_dot, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        real(dp) :: p, p_mu, p_variance, scale, z, density
        integer :: i

        if (.not. prediction_probability_shapes(self, x, probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. any(shape(probabilities_dot) /= &
            shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification probability JVP: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_jvp(x, x_dot, mean, mean_dot, variance, &
            variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            p = predictive_probability(self%likelihood, mean(i), variance(i))
            if (self%likelihood == GP_LIKELIHOOD_LOGISTIC) then
                scale = sqrt(1.0_dp + PI*variance(i)/8.0_dp)
                z = mean(i)/scale
                p_mu = p*(1.0_dp - p)/scale
                p_variance = p*(1.0_dp - p)*(-mean(i)*PI/(16.0_dp*scale**3))
            else
                scale = sqrt(1.0_dp + variance(i))
                z = mean(i)/scale
                density = exp(-0.5_dp*z*z)/SQRT_TWO_PI
                p_mu = density/scale
                p_variance = density*(-mean(i)/(2.0_dp*scale**3))
            end if
            probabilities(i, 2) = p
            probabilities(i, 1) = 1.0_dp - p
            probabilities_dot(i, 2) = p_mu*mean_dot(i) + p_variance*variance_dot(i)
            probabilities_dot(i, 1) = -probabilities_dot(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict_proba_jvp

    subroutine gp_classification_predict(self, x, labels, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. prediction_input_valid(self, x, status)) return
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: label shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), 2))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            if (probabilities(i, 2) > probabilities(i, 1)) then
                labels(i) = self%class_label(2)
            else
                labels(i) = self%class_label(1)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_predict

    function gp_classification_classes(self) result(classes)
        class(gp_classification_t), intent(in) :: self
        integer :: classes(2)

        classes = self%class_label
    end function gp_classification_classes

    integer function gp_classification_feature_count(self) result(count)
        class(gp_classification_t), intent(in) :: self

        count = self%n_features
    end function gp_classification_feature_count

    integer function gp_classification_parameter_count(self) result(count)
        class(gp_classification_t), intent(in) :: self

        count = 0
        if (.not. self%fitted()) return
        count = self%kernel%parameter_count()
    end function gp_classification_parameter_count

    function gp_classification_parameters(self) result(parameters)
        class(gp_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (.not. self%fitted()) then
            allocate(parameters(0))
            return
        end if
        parameters = self%kernel%parameters()
    end function gp_classification_parameters

    subroutine gp_classification_hyperparameter_gradient(self, gradient, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: matrix_bar(:, :)
        integer :: i, j

        gradient = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification hyperparameter gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification hyperparameter gradient: output shape is invalid")
            return
        end if
        ! The reported objective is the Laplace mode log posterior without
        ! the optional evidence correction.  At a converged mode the envelope
        ! theorem removes the implicit mode derivative, leaving
        ! d log p(y,f_mode | theta) / d theta =
        ! 1/2 * alpha^T (dK/dtheta) alpha, where alpha = K^{-1} f_mode.
        ! The kernel VJP supplies this contraction for every supported kernel
        ! expression, including sums and products.
        allocate(matrix_bar(self%n_samples, self%n_samples))
        do j = 1, self%n_samples
            do i = 1, self%n_samples
                matrix_bar(i, j) = 0.5_dp*self%alpha(i)*self%alpha(j)
            end do
        end do
        call self%kernel%parameter_vjp(self%x_train, self%x_train, matrix_bar, &
            gradient, status)
    end subroutine gp_classification_hyperparameter_gradient

    logical function gp_classification_fitted(self) result(fitted)
        class(gp_classification_t), intent(in) :: self

        fitted = allocated(self%x_train) .and. allocated(self%covariance) .and. &
            allocated(self%mode) .and. allocated(self%alpha) .and. &
            allocated(self%sqrt_w) .and. self%n_samples > 0 .and. self%n_features > 0
    end function gp_classification_fitted

    integer function gp_classification_likelihood(self) result(kind)
        class(gp_classification_t), intent(in) :: self

        kind = self%likelihood
    end function gp_classification_likelihood

    subroutine predict_core(self, x, mean, variance, status)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), prior(:)
        integer :: i, j

        if (.not. prediction_shapes(self, x, mean, variance, status)) return
        allocate(cross(self%n_samples, size(x, 1)))
        allocate(work, mold=cross)
        allocate(prior(size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, self%n_samples
                cross(i, j) = self%kernel%value(self%x_train(i, :), x(j, :))
            end do
            prior(j) = self%kernel%value(x(j, :), x(j, :))
        end do
        mean = matmul(transpose(cross), self%alpha)
        call scale_rows(cross, self%sqrt_w, work)
        call self%posterior_factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        variance = prior - sum(work*work, dim=1)
        call clamp_variance(variance, status)
    end subroutine predict_core

    logical function prediction_shapes(self, x, mean, variance, status) result(valid)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            size(mean) /= size(x, 1) .or. &
            size(variance) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: input or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_shapes

    logical function prediction_probability_shapes(self, x, probabilities, status) &
            result(valid)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), 2]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: input or probability shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_probability_shapes

    subroutine clamp_variance(variance, status)
        real(dp), intent(inout) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        do i = 1, size(variance)
            if (.not. ieee_is_finite(variance(i))) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "GP classification prediction: variance is not finite")
                return
            end if
            if (variance(i) < 0.0_dp) then
                if (variance(i) < -1.0e-8_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "GP classification prediction: variance is not positive")
                    return
                end if
                variance(i) = 0.0_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine clamp_variance

    subroutine posterior_system(covariance, sqrt_w, matrix)
        real(dp), intent(in) :: covariance(:, :), sqrt_w(:)
        real(dp), intent(out) :: matrix(:, :)
        integer :: i, j

        matrix = 0.0_dp
        do i = 1, size(matrix, 1)
            matrix(i, i) = 1.0_dp
            do j = 1, size(matrix, 2)
                matrix(i, j) = matrix(i, j) + sqrt_w(i)*covariance(i, j)*sqrt_w(j)
            end do
        end do
    end subroutine posterior_system

    subroutine scale_rows(matrix, weights, scaled)
        real(dp), intent(in) :: matrix(:, :), weights(:)
        real(dp), intent(out) :: scaled(:, :)
        integer :: i

        do i = 1, size(matrix, 1)
            scaled(i, :) = weights(i)*matrix(i, :)
        end do
    end subroutine scale_rows

    subroutine likelihood_terms(eta, likelihood, probability, gradient, curvature)
        real(dp), intent(in) :: eta
        integer, intent(in) :: likelihood
        real(dp), intent(out) :: probability, gradient, curvature
        real(dp) :: ratio, density

        if (likelihood == GP_LIKELIHOOD_LOGISTIC) then
            probability = stable_sigmoid(eta)
            gradient = 1.0_dp - probability
            curvature = max(probability*(1.0_dp - probability), &
                MIN_LIKELIHOOD_CURVATURE)
        else
            probability = normal_cdf(eta)
            density = exp(-0.5_dp*eta*eta)/SQRT_TWO_PI
            if (probability > 1.0e-14_dp) then
                ratio = density/probability
            else
                ratio = max(1.0_dp, -eta) + 1.0_dp/max(1.0_dp, -eta)
            end if
            gradient = ratio
            curvature = max(ratio*(ratio + eta), MIN_LIKELIHOOD_CURVATURE)
        end if
    end subroutine likelihood_terms

    real(dp) function predictive_probability(likelihood, mean, variance) result(probability)
        integer, intent(in) :: likelihood
        real(dp), intent(in) :: mean, variance
        real(dp) :: scale

        if (likelihood == GP_LIKELIHOOD_LOGISTIC) then
            scale = sqrt(1.0_dp + PI*variance/8.0_dp)
            probability = stable_sigmoid(mean/scale)
        else
            scale = sqrt(1.0_dp + variance)
            probability = normal_cdf(mean/scale)
        end if
    end function predictive_probability

    real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        real(dp) :: exponential

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            exponential = exp(value)
            probability = exponential/(1.0_dp + exponential)
        end if
    end function stable_sigmoid

    real(dp) function normal_cdf(value) result(probability)
        real(dp), intent(in) :: value

        probability = 0.5_dp*erfc(-value/SQRT_TWO)
        probability = min(1.0_dp, max(0.0_dp, probability))
    end function normal_cdf

    real(dp) function encoded_label(label, classes) result(value)
        integer, intent(in) :: label, classes(2)

        value = merge(1.0_dp, -1.0_dp, label == classes(2))
    end function encoded_label

    real(dp) function log_posterior(mode, alpha, labels, classes, likelihood) result(value)
        real(dp), intent(in) :: mode(:), alpha(:)
        integer, intent(in) :: labels(:), classes(2), likelihood
        real(dp) :: eta, probability, gradient, curvature
        integer :: i

        value = -0.5_dp*dot_product(mode, alpha)
        do i = 1, size(mode)
            eta = encoded_label(labels(i), classes)*mode(i)
            call likelihood_terms(eta, likelihood, probability, gradient, curvature)
            value = value + log(max(probability, tiny(1.0_dp)))
        end do
    end function log_posterior

    logical function valid_options(options) result(valid)
        type(gp_classification_options_t), intent(in) :: options

        valid = (options%likelihood == GP_LIKELIHOOD_LOGISTIC .or. &
            options%likelihood == GP_LIKELIHOOD_PROBIT) .and. &
            options%max_iterations >= 1 .and. &
            ieee_is_finite(options%tolerance) .and. options%tolerance > 0.0_dp .and. &
            ieee_is_finite(options%jitter) .and. options%jitter >= 0.0_dp .and. &
            ieee_is_finite(options%damping) .and. options%damping > 0.0_dp .and. &
            options%damping <= 1.0_dp
    end function valid_options

    logical function prediction_input_valid(self, x, status) result(valid)
        class(gp_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification prediction: input dimension is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_input_valid

end module fortml_gp_classification
