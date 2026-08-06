module fortml_bnn
    !! Bayesian neural network with an explicit factorized Gaussian prior over
    !! the network weights, a reparameterized factorized Gaussian variational
    !! posterior, deterministic seeded Monte Carlo draws, and ELBO
    !! value/JVP/VJP/HVP products over the packed variational parameter vector.
    !!
    !! The variational parameter vector is `[mu(1:p), log_sigma(1:p)]` where
    !! `p` is the network parameter count. A weight draw uses the
    !! reparameterization `w_s = mu + exp(log_sigma)*eps_s` with fixed seeded
    !! standard-normal draws `eps_s`, so every product is a deterministic
    !! function of the variational parameters.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use, intrinsic :: iso_fortran_env, only: int64
    implicit none
    private

    real(dp), parameter :: PI = 3.141592653589793238462643_dp

    type, public :: bnn_t
        type(mlp_t) :: net
        real(dp), allocatable :: mu(:)
        real(dp), allocatable :: log_sigma(:)
        real(dp), allocatable :: noise(:, :)
        real(dp) :: prior_variance = 1.0_dp
        real(dp) :: noise_variance = 1.0_dp
        integer :: n_mc = 1
    contains
        procedure, public :: initialize => bnn_initialize
        procedure, public :: parameter_count => bnn_parameter_count
        procedure, public :: parameters => bnn_parameters
        procedure, public :: set_parameters => bnn_set_parameters
        procedure, public :: kl => bnn_kl
        procedure, public :: kl_gradient => bnn_kl_gradient
        procedure, public :: elbo => bnn_elbo
        procedure, public :: elbo_jvp => bnn_elbo_jvp
        procedure, public :: elbo_vjp => bnn_elbo_vjp
        procedure, public :: elbo_hvp => bnn_elbo_hvp
    end type bnn_t

contains

    subroutine bnn_initialize(self, layer_sizes, n_mc_samples, seed, status, &
            prior_variance, noise_variance, hidden_activation)
        !! Build the network, allocate the variational parameters, and draw the
        !! fixed standard-normal Monte Carlo table from `seed`.
        class(bnn_t), intent(out) :: self
        integer, intent(in) :: layer_sizes(:)
        integer, intent(in) :: n_mc_samples
        integer, intent(in) :: seed
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: prior_variance, noise_variance
        integer, intent(in), optional :: hidden_activation
        integer :: n_parameters, hidden_kind

        hidden_kind = MLP_TANH
        if (present(hidden_activation)) hidden_kind = hidden_activation
        self%prior_variance = 1.0_dp
        if (present(prior_variance)) self%prior_variance = prior_variance
        self%noise_variance = 1.0_dp
        if (present(noise_variance)) self%noise_variance = noise_variance

        if (n_mc_samples < 1 .or. seed == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN initialize: sample count or seed is invalid")
            return
        end if
        if (self%prior_variance <= 0.0_dp .or. self%noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN initialize: prior or noise variance is not positive")
            return
        end if
        call self%net%initialize(layer_sizes, status, &
            hidden_activation=hidden_kind, output_activation=MLP_LINEAR)
        if (status%code /= FORTNUM_OK) return

        n_parameters = self%net%parameter_count()
        self%n_mc = n_mc_samples
        allocate(self%mu(n_parameters))
        allocate(self%log_sigma(n_parameters))
        self%mu = 0.0_dp
        self%log_sigma = 0.0_dp
        allocate(self%noise(n_parameters, n_mc_samples))
        call seeded_normal_table(seed, self%noise, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine bnn_initialize

    integer function bnn_parameter_count(self) result(count)
        class(bnn_t), intent(in) :: self

        count = 0
        if (allocated(self%mu)) count = 2*size(self%mu)
    end function bnn_parameter_count

    function bnn_parameters(self) result(lambda)
        class(bnn_t), intent(in) :: self
        real(dp), allocatable :: lambda(:)
        integer :: n_parameters

        n_parameters = size(self%mu)
        allocate(lambda(2*n_parameters))
        lambda(1:n_parameters) = self%mu
        lambda(n_parameters + 1:2*n_parameters) = self%log_sigma
    end function bnn_parameters

    subroutine bnn_set_parameters(self, lambda, status)
        class(bnn_t), intent(inout) :: self
        real(dp), intent(in) :: lambda(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_parameters

        if (.not. allocated(self%mu)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN set_parameters: model is not initialized")
            return
        end if
        n_parameters = size(self%mu)
        if (size(lambda) /= 2*n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN set_parameters: parameter shape is invalid")
            return
        end if
        self%mu = lambda(1:n_parameters)
        self%log_sigma = lambda(n_parameters + 1:2*n_parameters)
        call status_set(status, FORTNUM_OK, "")
    end subroutine bnn_set_parameters

    subroutine bnn_kl(self, value, status)
        !! Analytic KL(q||p) for factorized Gaussians with zero-mean prior.
        class(bnn_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: variance
        integer :: i

        value = 0.0_dp
        if (.not. allocated(self%mu)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN kl: model is not initialized")
            return
        end if
        do i = 1, size(self%mu)
            variance = exp(2.0_dp*self%log_sigma(i))
            value = value + 0.5_dp*log(self%prior_variance) - self%log_sigma(i) &
                + 0.5_dp*(variance + self%mu(i)**2)/self%prior_variance - 0.5_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bnn_kl

    subroutine bnn_kl_gradient(self, gradient, status)
        class(bnn_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, n_parameters

        if (.not. allocated(self%mu)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN kl_gradient: model is not initialized")
            return
        end if
        n_parameters = size(self%mu)
        if (size(gradient) /= 2*n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN kl_gradient: gradient shape is invalid")
            return
        end if
        do i = 1, n_parameters
            gradient(i) = self%mu(i)/self%prior_variance
            gradient(n_parameters + i) = &
                exp(2.0_dp*self%log_sigma(i))/self%prior_variance - 1.0_dp
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bnn_kl_gradient

    subroutine bnn_elbo(self, x, y, value, status)
        !! Seeded Monte Carlo ELBO for a fixed-variance Gaussian likelihood.
        class(bnn_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: weights(:), prediction(:, :)
        real(dp) :: kl_value, likelihood
        integer :: s

        value = 0.0_dp
        if (.not. valid_data(self, x, y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN elbo: model or data shape is invalid")
            return
        end if
        allocate(weights(size(self%mu)))
        allocate(prediction(size(y, 1), size(y, 2)))
        likelihood = 0.0_dp
        do s = 1, self%n_mc
            call draw_weights(self, s, weights)
            call self%net%set_parameters(weights, status)
            if (status%code /= FORTNUM_OK) return
            call self%net%predict(x, prediction, status)
            if (status%code /= FORTNUM_OK) return
            likelihood = likelihood + log_likelihood(self, prediction, y)
        end do
        likelihood = likelihood/real(self%n_mc, dp)
        call self%kl(kl_value, status)
        if (status%code /= FORTNUM_OK) return
        value = likelihood - kl_value
    end subroutine bnn_elbo

    subroutine bnn_elbo_jvp(self, x, y, direction, value, tangent, status)
        !! Forward-mode directional derivative of the ELBO. The likelihood term
        !! is propagated through the network JVP, not through the gradient.
        class(bnn_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: weights(:), weight_tangent(:)
        real(dp), allocatable :: prediction(:, :), prediction_tangent(:, :)
        real(dp), allocatable :: zero_input(:, :), kl_grad(:)
        real(dp) :: kl_value, likelihood, likelihood_tangent
        integer :: s, n_parameters

        value = 0.0_dp
        tangent = 0.0_dp
        n_parameters = 0
        if (allocated(self%mu)) n_parameters = size(self%mu)
        if (.not. valid_data(self, x, y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN elbo_jvp: model or data shape is invalid")
            return
        end if
        if (size(direction) /= 2*n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN elbo_jvp: direction shape is invalid")
            return
        end if

        allocate(weights(n_parameters), weight_tangent(n_parameters))
        allocate(prediction(size(y, 1), size(y, 2)))
        allocate(prediction_tangent(size(y, 1), size(y, 2)))
        allocate(zero_input(size(x, 1), size(x, 2)))
        allocate(kl_grad(2*n_parameters))
        zero_input = 0.0_dp
        likelihood = 0.0_dp
        likelihood_tangent = 0.0_dp
        do s = 1, self%n_mc
            call draw_weights(self, s, weights)
            call weight_direction(self, s, direction, weight_tangent)
            call self%net%set_parameters(weights, status)
            if (status%code /= FORTNUM_OK) return
            call self%net%jvp(x, weight_tangent, zero_input, prediction, &
                prediction_tangent, status)
            if (status%code /= FORTNUM_OK) return
            likelihood = likelihood + log_likelihood(self, prediction, y)
            likelihood_tangent = likelihood_tangent &
                - sum((prediction - y)*prediction_tangent)/self%noise_variance
        end do
        likelihood = likelihood/real(self%n_mc, dp)
        likelihood_tangent = likelihood_tangent/real(self%n_mc, dp)

        call self%kl(kl_value, status)
        if (status%code /= FORTNUM_OK) return
        call self%kl_gradient(kl_grad, status)
        if (status%code /= FORTNUM_OK) return
        value = likelihood - kl_value
        tangent = likelihood_tangent - sum(kl_grad*direction)
    end subroutine bnn_elbo_jvp

    subroutine bnn_elbo_vjp(self, x, y, cotangent, gradient, status)
        !! Reverse-mode product for the scalar ELBO. `cotangent` scales the
        !! returned gradient with respect to the packed variational vector.
        class(bnn_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        real(dp), intent(in) :: cotangent
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: weights(:), weight_grad(:), kl_grad(:)
        real(dp), allocatable :: prediction(:, :), residual(:, :), input_bar(:, :)
        integer :: s, i, n_parameters
        real(dp) :: scale

        n_parameters = 0
        if (allocated(self%mu)) n_parameters = size(self%mu)
        if (.not. valid_data(self, x, y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN elbo_vjp: model or data shape is invalid")
            return
        end if
        if (size(gradient) /= 2*n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN elbo_vjp: gradient shape is invalid")
            return
        end if

        allocate(weights(n_parameters), weight_grad(n_parameters))
        allocate(kl_grad(2*n_parameters))
        allocate(prediction(size(y, 1), size(y, 2)))
        allocate(residual(size(y, 1), size(y, 2)))
        allocate(input_bar(size(x, 1), size(x, 2)))
        gradient = 0.0_dp
        scale = 1.0_dp/real(self%n_mc, dp)
        do s = 1, self%n_mc
            call draw_weights(self, s, weights)
            call self%net%set_parameters(weights, status)
            if (status%code /= FORTNUM_OK) return
            call self%net%predict(x, prediction, status)
            if (status%code /= FORTNUM_OK) return
            residual = -(prediction - y)/self%noise_variance
            call self%net%vjp(x, residual, weight_grad, input_bar, status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, n_parameters
                gradient(i) = gradient(i) + scale*weight_grad(i)
                gradient(n_parameters + i) = gradient(n_parameters + i) &
                    + scale*weight_grad(i)*exp(self%log_sigma(i))*self%noise(i, s)
            end do
        end do

        call self%kl_gradient(kl_grad, status)
        if (status%code /= FORTNUM_OK) return
        gradient = cotangent*(gradient - kl_grad)
    end subroutine bnn_elbo_vjp

    subroutine bnn_elbo_hvp(self, x, y, direction, product, status)
        !! Forward-over-reverse Hessian-vector product of the ELBO with respect
        !! to the packed variational vector.
        class(bnn_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: weights(:), weight_tangent(:)
        real(dp), allocatable :: weight_grad(:), weight_hvp(:), curvature(:)
        real(dp), allocatable :: prediction(:, :), prediction_tangent(:, :)
        real(dp), allocatable :: residual(:, :), cotangent_tangent(:, :)
        real(dp), allocatable :: zero_input(:, :), input_bar(:, :)
        integer :: s, i, n_parameters
        real(dp) :: scale, factor

        n_parameters = 0
        if (allocated(self%mu)) n_parameters = size(self%mu)
        if (.not. valid_data(self, x, y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN elbo_hvp: model or data shape is invalid")
            return
        end if
        if (size(direction) /= 2*n_parameters .or. &
            size(product) /= 2*n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BNN elbo_hvp: direction or product shape is invalid")
            return
        end if

        allocate(weights(n_parameters), weight_tangent(n_parameters))
        allocate(weight_grad(n_parameters), weight_hvp(n_parameters))
        allocate(curvature(n_parameters))
        allocate(prediction(size(y, 1), size(y, 2)))
        allocate(prediction_tangent(size(y, 1), size(y, 2)))
        allocate(residual(size(y, 1), size(y, 2)))
        allocate(cotangent_tangent(size(y, 1), size(y, 2)))
        allocate(zero_input(size(x, 1), size(x, 2)))
        allocate(input_bar(size(x, 1), size(x, 2)))
        zero_input = 0.0_dp
        product = 0.0_dp
        scale = 1.0_dp/real(self%n_mc, dp)

        do s = 1, self%n_mc
            call draw_weights(self, s, weights)
            call weight_direction(self, s, direction, weight_tangent)
            call self%net%set_parameters(weights, status)
            if (status%code /= FORTNUM_OK) return
            call self%net%jvp(x, weight_tangent, zero_input, prediction, &
                prediction_tangent, status)
            if (status%code /= FORTNUM_OK) return
            residual = -(prediction - y)/self%noise_variance
            call self%net%vjp(x, residual, weight_grad, input_bar, status)
            if (status%code /= FORTNUM_OK) return
            ! Differentiate `J^T u` with the cotangent `u` held fixed, then add
            ! the contribution of the changing cotangent `-J v / noise`.
            call self%net%hvp(x, residual, weight_tangent, zero_input, &
                weight_hvp, input_bar, status)
            if (status%code /= FORTNUM_OK) return
            cotangent_tangent = -prediction_tangent/self%noise_variance
            call self%net%vjp(x, cotangent_tangent, curvature, input_bar, status)
            if (status%code /= FORTNUM_OK) return
            weight_hvp = weight_hvp + curvature

            do i = 1, n_parameters
                factor = exp(self%log_sigma(i))*self%noise(i, s)
                product(i) = product(i) + scale*weight_hvp(i)
                product(n_parameters + i) = product(n_parameters + i) &
                    + scale*(weight_hvp(i)*factor &
                    + weight_grad(i)*factor*direction(n_parameters + i))
            end do
        end do

        do i = 1, n_parameters
            product(i) = product(i) - direction(i)/self%prior_variance
            product(n_parameters + i) = product(n_parameters + i) &
                - 2.0_dp*exp(2.0_dp*self%log_sigma(i))/self%prior_variance &
                *direction(n_parameters + i)
        end do
    end subroutine bnn_elbo_hvp

    subroutine draw_weights(self, sample, weights)
        class(bnn_t), intent(in) :: self
        integer, intent(in) :: sample
        real(dp), intent(out) :: weights(:)

        weights = self%mu + exp(self%log_sigma)*self%noise(:, sample)
    end subroutine draw_weights

    subroutine weight_direction(self, sample, direction, weight_tangent)
        !! Tangent of `w_s = mu + exp(log_sigma)*eps_s` in the packed direction.
        class(bnn_t), intent(in) :: self
        integer, intent(in) :: sample
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: weight_tangent(:)
        integer :: n_parameters

        n_parameters = size(self%mu)
        weight_tangent = direction(1:n_parameters) &
            + exp(self%log_sigma)*self%noise(:, sample) &
            *direction(n_parameters + 1:2*n_parameters)
    end subroutine weight_direction

    real(dp) function log_likelihood(self, prediction, y) result(value)
        class(bnn_t), intent(in) :: self
        real(dp), intent(in) :: prediction(:, :), y(:, :)

        value = -0.5_dp*sum((prediction - y)**2)/self%noise_variance &
            - 0.5_dp*real(size(y), dp)*log(2.0_dp*PI*self%noise_variance)
    end function log_likelihood

    logical function valid_data(self, x, y) result(valid)
        class(bnn_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)

        valid = allocated(self%mu) .and. allocated(self%noise)
        if (.not. valid) return
        valid = size(x, 1) > 0 .and. size(x, 1) == size(y, 1) .and. &
            size(x, 2) == self%net%layer_sizes(1) .and. &
            size(y, 2) == self%net%layer_sizes(size(self%net%layer_sizes))
    end function valid_data

    subroutine seeded_normal_table(seed, table, status)
        !! Deterministic standard-normal draws from the counter-based
        !! `fortnum_rng` stream. The same seed always gives the same table, so
        !! every Monte Carlo product is a reproducible function of the
        !! variational parameters.
        integer, intent(in) :: seed
        real(dp), intent(out) :: table(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(rng_t) :: generator
        integer :: i, j

        call rng_seed(generator, int(seed, int64), status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(table, 2)
            do i = 1, size(table, 1)
                call rng_normal(generator, table(i, j))
            end do
        end do
    end subroutine seeded_normal_table

end module fortml_bnn
