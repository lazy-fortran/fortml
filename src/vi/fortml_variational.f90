module fortml_variational
    !! Reusable variational-inference contract for neural and Gaussian-process
    !! models.
    !!
    !! The variational posterior is a Gaussian `q = N(m, L L^T)` over one block
    !! of latent variables: neural-network weights, the inducing-value vector of
    !! a sparse GP, or any other block a model wants to integrate out. `L` is
    !! lower triangular with a positive diagonal, stored through its logarithm,
    !! so the packed vector is unconstrained and can be handed to `fortopt`
    !! directly. A model's remaining parameters - inducing-point locations,
    !! likelihood parameters, kernel hyperparameters - stay deterministic and
    !! travel beside the variational block as `extra`.
    !!
    !! The expected log likelihood uses the reparameterization
    !! `w_s = m + L eps_s` over a Monte Carlo table drawn once from a seed. The
    !! table is centred and whitened, so its empirical mean is zero and its
    !! empirical covariance is the identity to working precision. A log
    !! likelihood that is quadratic in `w` is then integrated exactly rather
    !! than approximately, which is what makes a conjugate model converge to its
    !! analytic posterior instead of to a sampling-noise neighbourhood of it.
    !!
    !! Minibatch scaling is explicit: the caller passes the factor
    !! `n_total/n_batch` that turns a minibatch log likelihood into an estimate
    !! of the full-data one. The KL term is never scaled.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use, intrinsic :: iso_fortran_env, only: int64
    implicit none
    private

    public :: gaussian_family_t
    public :: vi_log_likelihood_i
    public :: vi_elbo
    public :: vi_elbo_gradient

    type :: gaussian_family_t
        !! Full or diagonal Gaussian variational posterior over one block.
        integer :: dimension = 0
        integer :: n_mc = 1
        logical :: full_covariance = .true.
        real(dp) :: prior_variance = 1.0_dp
        real(dp), allocatable :: mean(:)
        real(dp), allocatable :: factor(:, :)
        real(dp), allocatable :: noise(:, :)
    contains
        procedure, public :: initialize => family_initialize
        procedure, public :: parameter_count => family_parameter_count
        procedure, public :: parameters => family_parameters
        procedure, public :: set_parameters => family_set_parameters
        procedure, public :: covariance => family_covariance
        procedure, public :: draw => family_draw
        procedure, public :: draw_tangent => family_draw_tangent
        procedure, public :: kl => family_kl
        procedure, public :: kl_gradient => family_kl_gradient
    end type gaussian_family_t

    abstract interface
        subroutine vi_log_likelihood_i(weights, extra, value, weight_gradient, &
                extra_gradient, status)
            !! Log likelihood of one latent draw and the deterministic
            !! parameters, with its gradients. The caller decides whether the
            !! data behind it is the full set or a minibatch.
            import :: dp, fortnum_status_t
            real(dp), intent(in) :: weights(:), extra(:)
            real(dp), intent(out) :: value
            real(dp), intent(out) :: weight_gradient(:), extra_gradient(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine vi_log_likelihood_i
    end interface

contains

    subroutine family_initialize(self, dimension, n_mc_samples, seed, status, &
            prior_variance, full_covariance)
        class(gaussian_family_t), intent(out) :: self
        integer, intent(in) :: dimension, n_mc_samples, seed
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: prior_variance
        logical, intent(in), optional :: full_covariance
        integer :: i

        self%prior_variance = 1.0_dp
        if (present(prior_variance)) self%prior_variance = prior_variance
        self%full_covariance = .true.
        if (present(full_covariance)) self%full_covariance = full_covariance

        if (dimension < 1 .or. n_mc_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational family: dimension or sample count is invalid")
            return
        end if
        if (self%prior_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational family: prior variance is not positive")
            return
        end if
        if (n_mc_samples <= dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational family: whitening needs more samples than the &
                &block dimension")
            return
        end if

        self%dimension = dimension
        self%n_mc = n_mc_samples
        allocate(self%mean(dimension))
        allocate(self%factor(dimension, dimension))
        self%mean = 0.0_dp
        self%factor = 0.0_dp
        do i = 1, dimension
            self%factor(i, i) = sqrt(self%prior_variance)
        end do
        allocate(self%noise(dimension, n_mc_samples))
        call whitened_normal_table(seed, self%noise, status)
    end subroutine family_initialize

    integer function family_parameter_count(self) result(count)
        class(gaussian_family_t), intent(in) :: self

        count = 0
        if (self%dimension < 1) return
        if (self%full_covariance) then
            count = self%dimension + self%dimension*(self%dimension + 1)/2
        else
            count = 2*self%dimension
        end if
    end function family_parameter_count

    function family_parameters(self) result(lambda)
        class(gaussian_family_t), intent(in) :: self
        real(dp), allocatable :: lambda(:)
        integer :: i, j, position

        allocate(lambda(self%parameter_count()))
        lambda(1:self%dimension) = self%mean
        position = self%dimension + 1
        do j = 1, self%dimension
            lambda(position) = log(self%factor(j, j))
            position = position + 1
            if (.not. self%full_covariance) cycle
            do i = j + 1, self%dimension
                lambda(position) = self%factor(i, j)
                position = position + 1
            end do
        end do
    end function family_parameters

    subroutine family_set_parameters(self, lambda, status)
        class(gaussian_family_t), intent(inout) :: self
        real(dp), intent(in) :: lambda(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, position

        if (self%dimension < 1 .or. size(lambda) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational family: parameter shape is invalid")
            return
        end if
        self%mean = lambda(1:self%dimension)
        self%factor = 0.0_dp
        position = self%dimension + 1
        do j = 1, self%dimension
            self%factor(j, j) = exp(lambda(position))
            position = position + 1
            if (.not. self%full_covariance) cycle
            do i = j + 1, self%dimension
                self%factor(i, j) = lambda(position)
                position = position + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine family_set_parameters

    subroutine family_covariance(self, matrix, status)
        class(gaussian_family_t), intent(in) :: self
        real(dp), intent(out) :: matrix(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (self%dimension < 1 .or. &
            any(shape(matrix) /= [self%dimension, self%dimension])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational family: covariance shape is invalid")
            return
        end if
        matrix = matmul(self%factor, transpose(self%factor))
        call status_set(status, FORTNUM_OK, "")
    end subroutine family_covariance

    subroutine family_draw(self, sample, weights)
        class(gaussian_family_t), intent(in) :: self
        integer, intent(in) :: sample
        real(dp), intent(out) :: weights(:)

        weights = self%mean + matmul(self%factor, self%noise(:, sample))
    end subroutine family_draw

    subroutine family_draw_tangent(self, sample, direction, tangent)
        !! Tangent of `w_s = m + L eps_s` in a packed parameter direction.
        class(gaussian_family_t), intent(in) :: self
        integer, intent(in) :: sample
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: tangent(:)
        real(dp), allocatable :: factor_tangent(:, :)
        integer :: i, j, position

        allocate(factor_tangent(self%dimension, self%dimension))
        factor_tangent = 0.0_dp
        position = self%dimension + 1
        do j = 1, self%dimension
            factor_tangent(j, j) = self%factor(j, j)*direction(position)
            position = position + 1
            if (.not. self%full_covariance) cycle
            do i = j + 1, self%dimension
                factor_tangent(i, j) = direction(position)
                position = position + 1
            end do
        end do
        tangent = direction(1:self%dimension) &
            + matmul(factor_tangent, self%noise(:, sample))
    end subroutine family_draw_tangent

    subroutine family_kl(self, value, status)
        !! KL(q || N(0, prior_variance I)), which is zero exactly when
        !! `m = 0` and `L L^T = prior_variance I`.
        class(gaussian_family_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer :: j

        value = 0.0_dp
        if (self%dimension < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational family: kl on an uninitialized family")
            return
        end if
        value = sum(self%factor**2)/self%prior_variance &
            + sum(self%mean**2)/self%prior_variance &
            - real(self%dimension, dp) &
            + real(self%dimension, dp)*log(self%prior_variance)
        do j = 1, self%dimension
            value = value - 2.0_dp*log(self%factor(j, j))
        end do
        value = 0.5_dp*value
        call status_set(status, FORTNUM_OK, "")
    end subroutine family_kl

    subroutine family_kl_gradient(self, gradient, status)
        class(gaussian_family_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, position

        if (self%dimension < 1 .or. size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational family: kl gradient shape is invalid")
            return
        end if
        gradient(1:self%dimension) = self%mean/self%prior_variance
        position = self%dimension + 1
        do j = 1, self%dimension
            gradient(position) = &
                self%factor(j, j)**2/self%prior_variance - 1.0_dp
            position = position + 1
            if (.not. self%full_covariance) cycle
            do i = j + 1, self%dimension
                gradient(position) = self%factor(i, j)/self%prior_variance
                position = position + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine family_kl_gradient

    subroutine vi_elbo(family, extra, log_likelihood, scale, value, &
            expected_log_likelihood, kl_value, status)
        !! ELBO and its two terms. Reporting the decomposition is part of the
        !! contract: an optimizer that only sees the sum cannot tell a
        !! likelihood improvement from a collapsing posterior.
        type(gaussian_family_t), intent(in) :: family
        real(dp), intent(in) :: extra(:)
        procedure(vi_log_likelihood_i) :: log_likelihood
        real(dp), intent(in) :: scale
        real(dp), intent(out) :: value, expected_log_likelihood, kl_value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: weights(:), weight_gradient(:), extra_gradient(:)
        real(dp) :: sample_value
        integer :: s

        value = 0.0_dp
        expected_log_likelihood = 0.0_dp
        kl_value = 0.0_dp
        if (family%dimension < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "vi elbo: family is not initialized")
            return
        end if
        allocate(weights(family%dimension), weight_gradient(family%dimension))
        allocate(extra_gradient(max(size(extra), 1)))
        do s = 1, family%n_mc
            call family%draw(s, weights)
            call log_likelihood(weights, extra, sample_value, weight_gradient, &
                extra_gradient(1:size(extra)), status)
            if (status%code /= FORTNUM_OK) return
            expected_log_likelihood = expected_log_likelihood + sample_value
        end do
        expected_log_likelihood = expected_log_likelihood/real(family%n_mc, dp)
        call family%kl(kl_value, status)
        if (status%code /= FORTNUM_OK) return
        value = scale*expected_log_likelihood - kl_value
    end subroutine vi_elbo

    subroutine vi_elbo_gradient(family, extra, log_likelihood, scale, value, &
            gradient, status)
        !! ELBO and its gradient with respect to the packed vector
        !! `[family parameters, extra]`.
        type(gaussian_family_t), intent(in) :: family
        real(dp), intent(in) :: extra(:)
        procedure(vi_log_likelihood_i) :: log_likelihood
        real(dp), intent(in) :: scale
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: weights(:), weight_gradient(:), extra_gradient(:)
        real(dp), allocatable :: mean_grad(:), factor_grad(:, :), kl_grad(:)
        real(dp) :: sample_value, expected_log_likelihood, kl_value, weight
        integer :: s, i, j, position, n_family

        value = 0.0_dp
        n_family = family%parameter_count()
        if (family%dimension < 1 .or. size(gradient) /= n_family + size(extra)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "vi elbo gradient: family or gradient shape is invalid")
            return
        end if

        allocate(weights(family%dimension), weight_gradient(family%dimension))
        allocate(extra_gradient(max(size(extra), 1)))
        allocate(mean_grad(family%dimension))
        allocate(factor_grad(family%dimension, family%dimension))
        allocate(kl_grad(n_family))
        mean_grad = 0.0_dp
        factor_grad = 0.0_dp
        expected_log_likelihood = 0.0_dp
        gradient(n_family + 1:) = 0.0_dp

        do s = 1, family%n_mc
            call family%draw(s, weights)
            call log_likelihood(weights, extra, sample_value, weight_gradient, &
                extra_gradient(1:size(extra)), status)
            if (status%code /= FORTNUM_OK) return
            expected_log_likelihood = expected_log_likelihood + sample_value
            mean_grad = mean_grad + weight_gradient
            do j = 1, family%dimension
                do i = j, family%dimension
                    factor_grad(i, j) = factor_grad(i, j) &
                        + weight_gradient(i)*family%noise(j, s)
                end do
            end do
            if (size(extra) > 0) then
                gradient(n_family + 1:) = gradient(n_family + 1:) &
                    + extra_gradient(1:size(extra))
            end if
        end do

        weight = scale/real(family%n_mc, dp)
        expected_log_likelihood = expected_log_likelihood/real(family%n_mc, dp)
        gradient(1:family%dimension) = weight*mean_grad
        position = family%dimension + 1
        do j = 1, family%dimension
            gradient(position) = weight*factor_grad(j, j)*family%factor(j, j)
            position = position + 1
            if (.not. family%full_covariance) cycle
            do i = j + 1, family%dimension
                gradient(position) = weight*factor_grad(i, j)
                position = position + 1
            end do
        end do
        gradient(n_family + 1:) = weight*gradient(n_family + 1:)

        call family%kl(kl_value, status)
        if (status%code /= FORTNUM_OK) return
        call family%kl_gradient(kl_grad, status)
        if (status%code /= FORTNUM_OK) return
        gradient(1:n_family) = gradient(1:n_family) - kl_grad
        value = scale*expected_log_likelihood - kl_value
    end subroutine vi_elbo_gradient

    subroutine whitened_normal_table(seed, table, status)
        !! Seeded standard normals, then centred and whitened so that the
        !! empirical mean is zero and the empirical covariance is the identity.
        integer, intent(in) :: seed
        real(dp), intent(out) :: table(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(cholesky_factorization_t) :: factorization
        type(rng_t) :: generator
        real(dp), allocatable :: covariance(:, :), column(:)
        integer :: i, j, s, dimension, n_samples

        dimension = size(table, 1)
        n_samples = size(table, 2)
        call rng_seed(generator, int(seed, int64), status)
        if (status%code /= FORTNUM_OK) return
        do s = 1, n_samples
            do i = 1, dimension
                call rng_normal(generator, table(i, s))
            end do
        end do
        do i = 1, dimension
            table(i, :) = table(i, :) - sum(table(i, :))/real(n_samples, dp)
        end do

        allocate(covariance(dimension, dimension), column(dimension))
        covariance = matmul(table, transpose(table))/real(n_samples, dp)
        call factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        do s = 1, n_samples
            column = table(:, s)
            do i = 1, dimension
                do j = 1, i - 1
                    column(i) = column(i) - factorization%lower(i, j)*column(j)
                end do
                column(i) = column(i)/factorization%lower(i, i)
            end do
            table(:, s) = column
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine whitened_normal_table

end module fortml_variational
