module fortml_deep_kernel_gp
    !! Deep kernel learning: a base kernel on a learned feature map.
    !!
    !! From Wilson, Hu, Salakhutdinov and Xing, *Deep Kernel Learning*
    !! (arXiv:1511.02222), read rather than recalled. Equation (5) is the whole
    !! construction:
    !!
    !!     k(x_i, x_j | theta)  ->  k(g(x_i, w), g(x_j, w) | theta, w)
    !!
    !! where `g` is a neural feature map with weights `w`. Everything else in
    !! the model is an ordinary GP; the kernel simply sees features instead of
    !! inputs.
    !!
    !! Three things the paper is explicit about that the one-line definition
    !! does not convey, and that a from-memory implementation would plausibly
    !! get wrong:
    !!
    !!   * **The base kernel keeps its own hyperparameters on the feature
    !!     space.** `theta` does not become redundant once a network is in
    !!     front of it. The paper carries the base lengthscale throughout and
    !!     differentiates the marginal likelihood with respect to it
    !!     separately from `w` (section 3, and the `d theta` term of the
    !!     gradient). Folding the lengthscale into the network's output scale
    !!     is tempting -- a linear output layer can absorb it -- but it changes
    !!     what the marginal likelihood is optimizing over and removes the
    !!     one hyperparameter that has a usable prior.
    !!   * **The weights are learned jointly with `theta` through the marginal
    !!     likelihood**, not pretrained and frozen. Section 3 is emphatic that
    !!     "compartmentalizing our model into a base kernel and deep
    !!     architecture is for pedagogical reasons"; the paper's own comparison
    !!     against a network trained separately and then handed to a GP is one
    !!     of its results, not an equivalent formulation.
    !!   * **The gradient factors through the features.** Equation (7) gives
    !!
    !!         dL/dK = (1/2) (K^-1 y y^T K^-1  -  K^-1)
    !!
    !!     with the noise absorbed into `K`, and the weight gradient is that
    !!     contracted with dK/dg and then backpropagated through `dg/dw`. This
    !!     module computes exactly that factorization rather than differencing
    !!     the likelihood, because the whole point of composing with a network
    !!     is that the network can be large.
    !!
    !! **Not implemented here: the KISS-GP approximation of equation (8).** The
    !! paper replaces `K` with a structured interpolation approximation for
    !! scalability, and its headline timings depend on that. This module is
    !! exact, so it inherits the cubic cost of a dense GP and is honest about
    !! it: `n` in the thousands, not the paper's millions. Reporting a deep
    !! kernel result against the paper's scaling claims without KISS-GP would
    !! be comparing different algorithms.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_kernels, only: kernel_t
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    implicit none
    private

    public :: deep_kernel_gp_t

    type :: deep_kernel_gp_t
        !! The feature map `g` of equation (5).
        type(mlp_t) :: features
        !! The base kernel, which lives on the *feature* space and keeps its
        !! own hyperparameters there.
        type(kernel_t) :: base_kernel
        !! An ordinary exact GP, conditioned on the mapped inputs. Composition
        !! rather than inheritance: a deep kernel GP *has* a GP on features, it
        !! is not a kind of GP on inputs, and the distinction is what keeps the
        !! feature map visible to the caller instead of hidden inside a
        !! subclass.
        type(gp_regression_t) :: model
        integer :: input_dimension = 0
        integer :: feature_dimension = 0
        real(dp) :: noise_variance = 1.0e-6_dp
        logical :: fitted = .false.
    contains
        procedure, public :: initialize => deep_initialize
        procedure, public :: fit => deep_fit
        procedure, public :: predict => deep_predict
        procedure, public :: transform => deep_transform
        procedure, public :: feature_gradient => deep_feature_gradient
        procedure, public :: log_marginal_likelihood => deep_log_marginal
        procedure, public :: weight_gradient => deep_weight_gradient
    end type deep_kernel_gp_t

contains

    !! Build the feature map and adopt a base kernel for the feature space.
    !!
    !! `layer_sizes` describes `g` end to end, so its first entry is the input
    !! width and its last is the feature width the base kernel will see. The
    !! output activation is linear: a squashing nonlinearity on the last layer
    !! would bound the feature space, and a bounded feature space silently caps
    !! how far apart the kernel can consider two inputs to be.
    subroutine deep_initialize(self, layer_sizes, base_kernel, status, &
            hidden_activation, initialization_seed)
        class(deep_kernel_gp_t), intent(out) :: self
        integer, intent(in) :: layer_sizes(:)
        type(kernel_t), intent(in) :: base_kernel
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_activation
        integer, intent(in), optional :: initialization_seed
        integer :: activation

        if (size(layer_sizes) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: the feature map needs at least an input and "// &
                "an output layer")
            return
        end if
        if (any(layer_sizes < 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: every layer needs at least one unit")
            return
        end if
        if (base_kernel%input_dim /= layer_sizes(size(layer_sizes))) then
            ! The base kernel lives on the feature space, not the input space.
            ! Getting this backwards is the single easiest way to build a
            ! model that runs and means nothing, so it is refused by name.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: the base kernel must take the feature width, "// &
                "not the input width")
            return
        end if

        activation = MLP_TANH
        if (present(hidden_activation)) activation = hidden_activation

        call self%features%initialize(layer_sizes, status, &
            hidden_activation=activation, output_activation=MLP_LINEAR, &
            initialization_seed=initialization_seed)
        if (status%code /= FORTNUM_OK) return

        self%base_kernel = base_kernel
        self%input_dimension = layer_sizes(1)
        self%feature_dimension = layer_sizes(size(layer_sizes))
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine deep_initialize

    !! The feature map itself, exposed rather than hidden.
    !!
    !! Public because the features are the interpretable part of the model --
    !! the paper's figures are drawn in this space -- and because a caller
    !! comparing against a separately trained network needs to be able to see
    !! them.
    subroutine deep_transform(self, x, features, status)
        class(deep_kernel_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: features(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (size(x, 2) /= self%input_dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: inputs are the wrong width")
            return
        end if
        if (size(features, 1) /= size(x, 1) .or. &
            size(features, 2) /= self%feature_dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: the feature buffer is the wrong shape")
            return
        end if
        call self%features%predict(x, features, status)
    end subroutine deep_transform

    !! Condition on data: map the inputs, then fit an exact GP on the features.
    subroutine deep_fit(self, x, y, noise_variance, status, jitter)
        class(deep_kernel_gp_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: jitter
        real(dp), allocatable :: features(:, :)

        if (self%input_dimension < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: fit before initialize")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%input_dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: training inputs are the wrong width")
            return
        end if
        if (size(y, 1) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: targets and inputs disagree in length")
            return
        end if
        if (noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: noise variance must be positive")
            return
        end if

        allocate (features(size(x, 1), self%feature_dimension))
        call self%features%predict(x, features, status)
        if (status%code /= FORTNUM_OK) return

        call self%model%fit(features, y, self%base_kernel, noise_variance, &
            status, jitter=jitter)
        if (status%code /= FORTNUM_OK) return

        self%noise_variance = noise_variance
        self%fitted = .true.
    end subroutine deep_fit

    !! Posterior at new inputs. The query points are mapped through the same
    !! `g`, which is what makes the model a kernel method rather than a network
    !! with a GP bolted on afterwards.
    subroutine deep_predict(self, x, mean, variance, status)
        class(deep_kernel_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:, :)
        real(dp), intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: features(:, :)

        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: predict before fit")
            return
        end if
        if (size(x, 2) /= self%input_dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: query inputs are the wrong width")
            return
        end if

        allocate (features(size(x, 1), self%feature_dimension))
        call self%features%predict(x, features, status)
        if (status%code /= FORTNUM_OK) return
        call self%model%predict(features, mean, variance, status)
    end subroutine deep_predict

    !! Log marginal likelihood of the current model, on the current features.
    subroutine deep_log_marginal(self, value, status)
        class(deep_kernel_gp_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: log marginal likelihood before fit")
            return
        end if
        call self%model%log_marginal_likelihood(value, status)
    end subroutine deep_log_marginal

    !! Gradient of the log marginal likelihood with respect to the *features*.
    !!
    !! This is the middle factor of the paper's chain rule: dL/dK contracted
    !! with dK/dg, holding the base kernel hyperparameters fixed. It is
    !! separated out because it is the only part specific to deep kernels --
    !! the term to its left is ordinary GP algebra and the term to its right is
    !! ordinary backpropagation.
    subroutine deep_feature_gradient(self, x, y, gradient, status)
        class(deep_kernel_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        real(dp), intent(out) :: gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: features(:, :), alpha(:), solved(:, :)
        real(dp), allocatable :: gram(:, :), factor(:, :), weight(:, :)
        real(dp), allocatable :: pair_gradient(:), inverse(:, :)
        real(dp), allocatable :: other_gradient(:), mixed(:, :)
        real(dp) :: entry_value
        integer :: n, d, i, j, c

        gradient = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: feature gradient before fit")
            return
        end if
        n = size(x, 1)
        d = self%feature_dimension
        if (size(gradient, 1) /= n .or. size(gradient, 2) /= d) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: the gradient buffer is the wrong shape")
            return
        end if
        if (size(y, 1) /= n .or. size(y, 2) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: the feature gradient is single-output")
            return
        end if

        allocate (features(n, d))
        call self%features%predict(x, features, status)
        if (status%code /= FORTNUM_OK) return

        allocate (gram(n, n), factor(n, n), inverse(n, n))
        call self%base_kernel%matrix(features, features, gram, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n
            gram(i, i) = gram(i, i) + self%noise_variance
        end do

        call cholesky(gram, factor, status)
        if (status%code /= FORTNUM_OK) return
        allocate (alpha(n), solved(n, n))
        call cholesky_solve_vector(factor, y(:, 1), alpha)
        call cholesky_inverse(factor, inverse)

        ! Equation (7): dL/dK = (1/2)(K^-1 y y^T K^-1 - K^-1), with the noise
        ! already absorbed into K above.
        allocate (weight(n, n))
        do j = 1, n
            do i = 1, n
                weight(i, j) = 0.5_dp*(alpha(i)*alpha(j) - inverse(i, j))
            end do
        end do

        ! Contract with dK/dg. The base kernel supplies the derivative of one
        ! entry with respect to its first argument; the second argument's
        ! contribution arrives when the pair is visited in the other order,
        ! which is why both (i,j) and (j,i) are accumulated into row i.
        allocate (pair_gradient(d), other_gradient(d), mixed(d, d))
        do i = 1, n
            do j = 1, n
                call self%base_kernel%input_derivatives(features(i, :), &
                    features(j, :), entry_value, pair_gradient, other_gradient, &
                    mixed, status)
                if (status%code /= FORTNUM_OK) return
                ! Both arguments. Feature row i enters K(i, j) as the first
                ! argument and K(j, i) as the second, and the derivative of the
                ! entry with respect to each is a different quantity even for a
                ! symmetric kernel. An earlier version accumulated only the
                ! first-argument term, reasoning that the second would arrive
                ! when the pair (j, i) came round -- but at that point the
                ! outer index is j, so the contribution lands in row j instead.
                ! The result was a gradient exactly half the true one, which
                ! the finite-difference oracle caught immediately and no
                ! amount of internal consistency would have.
                do c = 1, d
                    gradient(i, c) = gradient(i, c) &
                        + weight(i, j)*pair_gradient(c)
                    gradient(j, c) = gradient(j, c) &
                        + weight(i, j)*other_gradient(c)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine deep_feature_gradient

    !! Gradient of the log marginal likelihood with respect to the network
    !! weights: the feature gradient backpropagated through `g`.
    !!
    !! The paper's phrase for this last step is "computed using standard
    !! backpropagation", and that is exactly what it is -- the feature gradient
    !! is the seed adjoint handed to the network's own reverse pass. Nothing
    !! here differences the likelihood.
    subroutine deep_weight_gradient(self, x, y, gradient, status)
        class(deep_kernel_gp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: seed(:, :), input_bar(:, :)

        gradient = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: weight gradient before fit")
            return
        end if
        if (size(gradient) /= self%features%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "deep kernel GP: the weight gradient buffer is the wrong size")
            return
        end if

        allocate (seed(size(x, 1), self%feature_dimension))
        call deep_feature_gradient(self, x, y, seed, status)
        if (status%code /= FORTNUM_OK) return

        allocate (input_bar(size(x, 1), self%input_dimension))
        call self%features%vjp(x, seed, gradient, input_bar, status)
    end subroutine deep_weight_gradient

    !! Lower Cholesky factor.
    subroutine cholesky(matrix, factor, status)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), intent(out) :: factor(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: total
        integer :: n, i, j, k

        n = size(matrix, 1)
        factor = 0.0_dp
        do i = 1, n
            do j = 1, i
                total = matrix(i, j)
                do k = 1, j - 1
                    total = total - factor(i, k)*factor(j, k)
                end do
                if (i == j) then
                    if (total <= 0.0_dp) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "deep kernel GP: the Gram matrix is not positive "// &
                            "definite")
                        return
                    end if
                    factor(i, i) = sqrt(total)
                else
                    factor(i, j) = total/factor(j, j)
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine cholesky

    subroutine cholesky_solve_vector(factor, rhs, solution)
        real(dp), intent(in) :: factor(:, :), rhs(:)
        real(dp), intent(out) :: solution(:)
        real(dp), allocatable :: intermediate(:)
        real(dp) :: total
        integer :: n, i, k

        n = size(rhs)
        allocate (intermediate(n))
        do i = 1, n
            total = rhs(i)
            do k = 1, i - 1
                total = total - factor(i, k)*intermediate(k)
            end do
            intermediate(i) = total/factor(i, i)
        end do
        do i = n, 1, -1
            total = intermediate(i)
            do k = i + 1, n
                total = total - factor(k, i)*solution(k)
            end do
            solution(i) = total/factor(i, i)
        end do
    end subroutine cholesky_solve_vector

    !! Explicit inverse, which equation (7) genuinely needs -- the `-K^-1` term
    !! is not a solve against any particular right-hand side.
    subroutine cholesky_inverse(factor, inverse)
        real(dp), intent(in) :: factor(:, :)
        real(dp), intent(out) :: inverse(:, :)
        real(dp), allocatable :: unit_vector(:), column(:)
        integer :: n, j

        n = size(factor, 1)
        allocate (unit_vector(n), column(n))
        do j = 1, n
            unit_vector = 0.0_dp
            unit_vector(j) = 1.0_dp
            call cholesky_solve_vector(factor, unit_vector, column)
            inverse(:, j) = column
        end do
    end subroutine cholesky_inverse

end module fortml_deep_kernel_gp
