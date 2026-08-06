module fortml_vae
    !! Variational autoencoder over two explicit MLPs.
    !!
    !! The encoder maps each input to `2L` numbers, read as the mean and the
    !! log standard deviation of a diagonal Gaussian `q(z|x)`. The decoder maps
    !! a latent draw to the mean of a fixed-variance Gaussian likelihood. The
    !! draw uses the reparameterization `z = mu + exp(log_sigma)*eps` with a
    !! seeded noise table, so the ELBO and its gradient are deterministic
    !! functions of the parameters.
    !!
    !!     ELBO = sum_i [ log N(x_i | decoder(z_i), sigma^2)
    !!                    - KL(q(z|x_i) || N(0, I)) ]
    !!
    !! The gradient is one decoder VJP and one encoder VJP per batch: the
    !! decoder's input gradient is the cotangent that the reparameterization
    !! carries back to the encoder outputs, where the analytic KL gradient is
    !! added.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    use, intrinsic :: iso_fortran_env, only: int64
    implicit none
    private

    real(dp), parameter :: PI = 3.141592653589793238462643_dp

    type, public :: vae_t
        type(mlp_t) :: encoder
        type(mlp_t) :: decoder
        real(dp), allocatable :: noise(:, :)
        real(dp) :: likelihood_variance = 1.0_dp
        integer :: latent_dim = 0
        integer :: input_dim = 0
        integer :: batch_size = 0
    contains
        procedure, public :: initialize => vae_initialize
        procedure, public :: parameter_count => vae_parameter_count
        procedure, public :: parameters => vae_parameters
        procedure, public :: set_parameters => vae_set_parameters
        procedure, public :: elbo => vae_elbo
        procedure, public :: elbo_gradient => vae_elbo_gradient
        procedure, public :: reconstruct => vae_reconstruct
    end type vae_t

contains

    subroutine vae_initialize(self, input_dim, hidden_dim, latent_dim, &
            batch_size, seed, status, likelihood_variance)
        class(vae_t), intent(out) :: self
        integer, intent(in) :: input_dim, hidden_dim, latent_dim, batch_size, seed
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: likelihood_variance
        type(rng_t) :: generator
        integer :: i, j

        self%likelihood_variance = 1.0_dp
        if (present(likelihood_variance)) then
            self%likelihood_variance = likelihood_variance
        end if
        if (input_dim < 1 .or. hidden_dim < 1 .or. latent_dim < 1 .or. &
            batch_size < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "VAE: layer or batch shape is invalid")
            return
        end if
        if (self%likelihood_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "VAE: the likelihood variance must be positive")
            return
        end if

        call self%encoder%initialize([input_dim, hidden_dim, 2*latent_dim], &
            status, hidden_activation=MLP_TANH, output_activation=MLP_LINEAR)
        if (status%code /= FORTNUM_OK) return
        call self%decoder%initialize([latent_dim, hidden_dim, input_dim], &
            status, hidden_activation=MLP_TANH, output_activation=MLP_LINEAR)
        if (status%code /= FORTNUM_OK) return

        self%input_dim = input_dim
        self%latent_dim = latent_dim
        self%batch_size = batch_size
        allocate(self%noise(batch_size, latent_dim))
        call rng_seed(generator, int(seed, int64), status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, latent_dim
            do i = 1, batch_size
                call rng_normal(generator, self%noise(i, j))
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine vae_initialize

    integer function vae_parameter_count(self) result(count)
        class(vae_t), intent(in) :: self

        count = self%encoder%parameter_count() + self%decoder%parameter_count()
    end function vae_parameter_count

    function vae_parameters(self) result(theta)
        class(vae_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        integer :: split

        allocate(theta(self%parameter_count()))
        split = self%encoder%parameter_count()
        theta(1:split) = self%encoder%parameters()
        theta(split + 1:) = self%decoder%parameters()
    end function vae_parameters

    subroutine vae_set_parameters(self, theta, status)
        class(vae_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: split

        if (size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "VAE: parameter shape is invalid")
            return
        end if
        split = self%encoder%parameter_count()
        call self%encoder%set_parameters(theta(1:split), status)
        if (status%code /= FORTNUM_OK) return
        call self%decoder%set_parameters(theta(split + 1:), status)
    end subroutine vae_set_parameters

    subroutine vae_forward(self, x, code, latent, reconstruction, status)
        !! Encoder output, latent draw, and decoder output for a batch.
        class(vae_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: code(:, :), latent(:, :), reconstruction(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        call self%encoder%predict(x, code, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%latent_dim
            do i = 1, size(x, 1)
                latent(i, j) = code(i, j) + &
                    exp(code(i, self%latent_dim + j))*self%noise(i, j)
            end do
        end do
        call self%decoder%predict(latent, reconstruction, status)
    end subroutine vae_forward

    subroutine vae_elbo(self, x, value, status, expected_log_likelihood, kl_value)
        class(vae_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(out), optional :: expected_log_likelihood, kl_value

        real(dp), allocatable :: code(:, :), latent(:, :), reconstruction(:, :)
        real(dp) :: likelihood, divergence
        integer :: i, j

        value = 0.0_dp
        if (present(expected_log_likelihood)) expected_log_likelihood = 0.0_dp
        if (present(kl_value)) kl_value = 0.0_dp
        if (.not. valid_batch(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "VAE: batch shape is invalid")
            return
        end if

        allocate(code(size(x, 1), 2*self%latent_dim))
        allocate(latent(size(x, 1), self%latent_dim))
        allocate(reconstruction(size(x, 1), self%input_dim))
        call vae_forward(self, x, code, latent, reconstruction, status)
        if (status%code /= FORTNUM_OK) return

        likelihood = -0.5_dp*sum((x - reconstruction)**2)/self%likelihood_variance &
            - 0.5_dp*real(size(x), dp)*log(2.0_dp*PI*self%likelihood_variance)
        divergence = 0.0_dp
        do j = 1, self%latent_dim
            do i = 1, size(x, 1)
                divergence = divergence + 0.5_dp*( &
                    exp(2.0_dp*code(i, self%latent_dim + j)) &
                    + code(i, j)*code(i, j) - 1.0_dp &
                    - 2.0_dp*code(i, self%latent_dim + j))
            end do
        end do

        value = likelihood - divergence
        if (present(expected_log_likelihood)) expected_log_likelihood = likelihood
        if (present(kl_value)) kl_value = divergence
        call status_set(status, FORTNUM_OK, "")
    end subroutine vae_elbo

    subroutine vae_elbo_gradient(self, x, value, gradient, status)
        !! Gradient with respect to `[encoder parameters, decoder parameters]`.
        class(vae_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: code(:, :), latent(:, :), reconstruction(:, :)
        real(dp), allocatable :: decoder_cotangent(:, :), latent_bar(:, :)
        real(dp), allocatable :: code_bar(:, :), encoder_bar(:, :)
        real(dp), allocatable :: decoder_grad(:), encoder_grad(:)
        integer :: i, j, split, n_batch

        value = 0.0_dp
        if (.not. valid_batch(self, x) .or. &
            size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "VAE: gradient or batch shape is invalid")
            return
        end if
        n_batch = size(x, 1)
        allocate(code(n_batch, 2*self%latent_dim))
        allocate(latent(n_batch, self%latent_dim))
        allocate(reconstruction(n_batch, self%input_dim))
        call vae_forward(self, x, code, latent, reconstruction, status)
        if (status%code /= FORTNUM_OK) return

        ! Decoder: d loglik / d reconstruction.
        allocate(decoder_cotangent(n_batch, self%input_dim))
        allocate(latent_bar(n_batch, self%latent_dim))
        allocate(decoder_grad(self%decoder%parameter_count()))
        decoder_cotangent = (x - reconstruction)/self%likelihood_variance
        call self%decoder%vjp(latent, decoder_cotangent, decoder_grad, &
            latent_bar, status)
        if (status%code /= FORTNUM_OK) return

        ! Reparameterization plus the analytic KL gradient.
        allocate(code_bar(n_batch, 2*self%latent_dim))
        allocate(encoder_bar(n_batch, self%input_dim))
        allocate(encoder_grad(self%encoder%parameter_count()))
        do j = 1, self%latent_dim
            do i = 1, n_batch
                code_bar(i, j) = latent_bar(i, j) - code(i, j)
                code_bar(i, self%latent_dim + j) = &
                    latent_bar(i, j)*exp(code(i, self%latent_dim + j))* &
                    self%noise(i, j) &
                    - exp(2.0_dp*code(i, self%latent_dim + j)) + 1.0_dp
            end do
        end do
        call self%encoder%vjp(x, code_bar, encoder_grad, encoder_bar, status)
        if (status%code /= FORTNUM_OK) return

        split = self%encoder%parameter_count()
        gradient(1:split) = encoder_grad
        gradient(split + 1:) = decoder_grad
        call vae_elbo(self, x, value, status)
    end subroutine vae_elbo_gradient

    subroutine vae_reconstruct(self, x, reconstruction, status)
        class(vae_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: reconstruction(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: code(:, :), latent(:, :)

        if (.not. valid_batch(self, x) .or. &
            any(shape(reconstruction) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "VAE: reconstruction shape is invalid")
            return
        end if
        allocate(code(size(x, 1), 2*self%latent_dim))
        allocate(latent(size(x, 1), self%latent_dim))
        call vae_forward(self, x, code, latent, reconstruction, status)
    end subroutine vae_reconstruct

    logical function valid_batch(self, x) result(valid)
        class(vae_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)

        valid = self%latent_dim > 0 .and. allocated(self%noise)
        if (.not. valid) return
        valid = size(x, 1) == self%batch_size .and. size(x, 2) == self%input_dim
    end function valid_batch

end module fortml_vae
