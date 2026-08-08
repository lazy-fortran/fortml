module fortml_mlp
    use fortnum_kinds, only: dp
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_pca, only: pca_t
    implicit none
    private

    integer, parameter, public :: MLP_LINEAR = 1
    integer, parameter, public :: MLP_TANH = 2
    integer, parameter, public :: MLP_RELU = 3
    integer, parameter, public :: MLP_GELU = 4
    integer, parameter, public :: MLP_SILU = 5
    integer, parameter, public :: MLP_ELU = 6
    integer, parameter, public :: MLP_SOFTPLUS = 7
    integer, parameter, public :: MLP_LEAKY_RELU = 8
    integer, parameter, public :: MLP_SIGMOID = 9
    integer, parameter, public :: MLP_MISH = 10
    real(dp), parameter :: gelu_c = 0.797884560802865355879_dp
    real(dp), parameter :: gelu_k = 0.044715_dp
    real(dp), parameter :: leaky_relu_slope = 0.01_dp

    type :: matrix_holder_t
        real(dp), allocatable :: value(:, :)
    end type matrix_holder_t

    type :: dense_layer_t
        real(dp), allocatable :: weight(:, :)
        real(dp), allocatable :: bias(:)
    end type dense_layer_t

    type, public :: mlp_parameter_block_t
        !! Stable descriptor for one packed MLP parameter or buffer block.
        !!
        !! `first:last` uses the one-based order returned by `parameters()`.  A
        !! dense MLP currently has only trainable parameter blocks, but the
        !! explicit `is_buffer` field keeps the tree contract extensible to
        !! non-trainable running state without changing the descriptor ABI.
        character(len=64) :: name = ""
        character(len=16) :: kind = ""
        integer :: first = 0
        integer :: last = -1
        integer :: rows = 0
        integer :: columns = 0
        logical :: trainable = .true.
        logical :: is_buffer = .false.
    end type mlp_parameter_block_t

    type, public :: mlp_t
        integer, allocatable :: layer_sizes(:)
        type(dense_layer_t), allocatable :: layer(:)
        integer :: hidden_activation = MLP_TANH
        integer :: output_activation = MLP_LINEAR
    contains
        procedure, public :: initialize => mlp_initialize
        procedure, public :: initialize_linear => mlp_initialize_linear
        procedure, public :: initialize_from_pca => mlp_initialize_from_pca
        procedure, public :: parameter_count => mlp_parameter_count
        procedure, public :: parameter_block_count => mlp_parameter_block_count
        procedure, public :: parameter_layout => mlp_parameter_layout
        procedure, public :: parameter_range => mlp_parameter_range
        procedure, public :: parameters => mlp_parameters
        procedure, public :: set_parameters => mlp_set_parameters
        procedure, public :: set_linear_parameters => mlp_set_linear_parameters
        procedure, public :: predict => mlp_predict
        procedure, public :: jvp => mlp_jvp
        procedure, public :: vjp => mlp_vjp
        procedure, public :: hvp => mlp_hvp
        procedure, public :: backprop => mlp_vjp
    end type mlp_t

    public :: mlp_jvp
    public :: mlp_vjp
    public :: mlp_hvp

contains

    subroutine mlp_initialize(self, layer_sizes, status, hidden_activation, &
            output_activation, initialization_seed)
        class(mlp_t), intent(out) :: self
        integer, intent(in) :: layer_sizes(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_activation, output_activation
        integer, intent(in), optional :: initialization_seed
        integer :: i, hidden_kind, output_kind, seed
        real(dp) :: scale

        hidden_kind = MLP_TANH
        if (present(hidden_activation)) hidden_kind = hidden_activation
        output_kind = MLP_LINEAR
        if (present(output_activation)) output_kind = output_activation
        seed = 17
        if (present(initialization_seed)) seed = initialization_seed

        if (size(layer_sizes) < 2 .or. any(layer_sizes < 1) .or. &
            .not. valid_activation(hidden_kind) .or. &
            .not. valid_activation(output_kind) .or. seed < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP initialize: invalid layer sizes or activation")
            return
        end if

        allocate(self%layer_sizes(size(layer_sizes)))
        self%layer_sizes = layer_sizes
        allocate(self%layer(size(layer_sizes) - 1))
        do i = 1, size(self%layer)
            allocate(self%layer(i)%weight(layer_sizes(i), layer_sizes(i + 1)))
            allocate(self%layer(i)%bias(layer_sizes(i + 1)))
            if ((hidden_kind == MLP_RELU .or. hidden_kind == MLP_LEAKY_RELU) .and. &
                i < size(self%layer)) then
                scale = sqrt(2.0_dp/real(layer_sizes(i), dp))
            else
                scale = sqrt(6.0_dp/real(layer_sizes(i) + layer_sizes(i + 1), dp))
            end if
            call initialize_layer(self%layer(i)%weight, self%layer(i)%bias, &
                scale, seed, i)
        end do
        self%hidden_activation = hidden_kind
        self%output_activation = output_kind
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_initialize

    subroutine mlp_initialize_linear(self, weight1, bias1, weight2, bias2, status)
        !! Initialize a two-layer MLP whose complete map is affine.
        !!
        !! The supplied arrays are copied into a topology with dimensions
        !! ``[size(weight1,1), size(weight1,2), size(weight2,2)]``.  Both
        !! layers use ``MLP_LINEAR``.  This is intentionally a finite,
        !! deterministic state-loading seam for linear/PCA warm starts; it
        !! does not claim an NNGP or GP-posterior equivalence.
        class(mlp_t), intent(out) :: self
        real(dp), intent(in) :: weight1(:, :), bias1(:)
        real(dp), intent(in) :: weight2(:, :), bias2(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. valid_linear_state(weight1, bias1, weight2, bias2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP initialize_linear: dimensions or values are invalid")
            return
        end if

        call self%initialize([size(weight1, 1), size(weight1, 2), size(weight2, 2)], &
            status, hidden_activation=MLP_LINEAR, output_activation=MLP_LINEAR)
        if (status%code /= FORTNUM_OK) return
        call self%set_linear_parameters(weight1, bias1, weight2, bias2, status)
    end subroutine mlp_initialize_linear

    subroutine mlp_initialize_from_pca(self, pca, status)
        !! Initialize a two-layer linear autoencoder from fitted PCA.
        !!
        !! The resulting map is exactly the fitted PCA transform followed by
        !! its inverse transform, including sample-variance whitening.  It is
        !! the finite linear/PCA reconstruction optimum, not an NNGP, NTK, or
        !! GP-posterior equivalence.
        class(mlp_t), intent(inout) :: self
        class(pca_t), intent(in) :: pca
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: components(:, :), center(:), variance(:)
        real(dp), allocatable :: weight1(:, :), bias1(:), weight2(:, :), bias2(:)
        integer :: n_components, n_features, j

        if (.not. pca%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP initialize_from_pca: PCA must be fitted")
            return
        end if
        components = pca%components()
        center = pca%mean()
        variance = pca%explained_variance()
        n_components = size(components, 1)
        n_features = size(components, 2)
        if (n_components < 1 .or. n_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP initialize_from_pca: PCA dimensions are invalid")
            return
        end if
        if (size(center) /= n_features .or. size(variance) /= n_components) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP initialize_from_pca: PCA metadata shape is invalid")
            return
        end if
        if (.not. all(ieee_is_finite(components))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP initialize_from_pca: PCA components are nonfinite")
            return
        end if
        if (.not. all(ieee_is_finite(center))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP initialize_from_pca: PCA mean is nonfinite")
            return
        end if
        if (.not. all(ieee_is_finite(variance)) .or. any(variance < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP initialize_from_pca: PCA variance is invalid")
            return
        end if

        allocate(weight1(n_features, n_components), bias1(n_components))
        allocate(weight2(n_components, n_features), bias2(n_features))
        weight1 = transpose(components)
        weight2 = components
        bias1 = -matmul(center, transpose(components))
        bias2 = center
        if (pca%whiten()) then
            do j = 1, n_components
                if (variance(j) > tiny(1.0_dp)) then
                    weight1(:, j) = weight1(:, j)/sqrt(variance(j))
                    bias1(j) = bias1(j)/sqrt(variance(j))
                    weight2(j, :) = weight2(j, :)*sqrt(variance(j))
                end if
            end do
        end if
        if (.not. all(ieee_is_finite(weight1)) .or. &
                .not. all(ieee_is_finite(bias1)) .or. &
                .not. all(ieee_is_finite(weight2)) .or. &
                .not. all(ieee_is_finite(bias2))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP initialize_from_pca: generated state is nonfinite")
            return
        end if
        call self%initialize_linear(weight1, bias1, weight2, bias2, status)
    end subroutine mlp_initialize_from_pca

    subroutine initialize_layer(weight, bias, scale, seed, layer_index)
        real(dp), intent(out) :: weight(:, :), bias(:)
        real(dp), intent(in) :: scale
        integer, intent(in) :: seed, layer_index
        integer :: i, j, index
        real(dp) :: phase

        ! A deterministic, platform-independent initializer is preferable to
        ! random_number here: callers can reproduce a training run without
        ! mutating the process-global RNG state.  The incommensurate phases
        ! break hidden-unit symmetry while retaining Xavier/He magnitudes.
        do j = 1, size(weight, 2)
            do i = 1, size(weight, 1)
                index = i + size(weight, 1)*(j - 1)
                phase = real(seed + 1009*layer_index + 9176*index, dp)
                weight(i, j) = scale*sin(phase)
            end do
        end do
        do j = 1, size(bias)
            phase = real(seed + 1009*layer_index + 7919*j, dp)
            bias(j) = 0.01_dp*scale*sin(phase)
        end do
    end subroutine initialize_layer

    integer function mlp_parameter_count(self) result(count)
        class(mlp_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. allocated(self%layer)) return
        do i = 1, size(self%layer)
            count = count + size(self%layer(i)%weight) + size(self%layer(i)%bias)
        end do
    end function mlp_parameter_count

    integer function mlp_parameter_block_count(self) result(count)
        !! Return the number of named parameter/buffer blocks in the tree.
        class(mlp_t), intent(in) :: self

        if (.not. allocated(self%layer)) then
            count = 0
        else
            count = 2*size(self%layer)
        end if
    end function mlp_parameter_block_count

    function mlp_parameter_layout(self) result(layout)
        !! Return the deterministic named tree for the packed parameter vector.
        !!
        !! Each dense layer contributes a `layer_n.weight` matrix followed by
        !! a `layer_n.bias` vector.  The ranges and shapes are metadata only;
        !! `parameters()` and `set_parameters()` remain the sole data path.
        class(mlp_t), intent(in) :: self
        type(mlp_parameter_block_t), allocatable :: layout(:)
        integer :: i, block, first, count

        count = self%parameter_block_count()
        allocate(layout(count))
        if (count == 0) return

        first = 1
        block = 0
        do i = 1, size(self%layer)
            block = block + 1
            layout(block)%name = ""
            write(layout(block)%name, '("layer_",i0,".weight")') i
            layout(block)%kind = "weight"
            layout(block)%first = first
            layout(block)%last = first + size(self%layer(i)%weight) - 1
            layout(block)%rows = size(self%layer(i)%weight, 1)
            layout(block)%columns = size(self%layer(i)%weight, 2)
            layout(block)%trainable = .true.
            layout(block)%is_buffer = .false.
            first = layout(block)%last + 1

            block = block + 1
            layout(block)%name = ""
            write(layout(block)%name, '("layer_",i0,".bias")') i
            layout(block)%kind = "bias"
            layout(block)%first = first
            layout(block)%last = first + size(self%layer(i)%bias) - 1
            layout(block)%rows = size(self%layer(i)%bias)
            layout(block)%columns = 1
            layout(block)%trainable = .true.
            layout(block)%is_buffer = .false.
            first = layout(block)%last + 1
        end do
    end function mlp_parameter_layout

    subroutine mlp_parameter_range(self, name, first, last, found)
        !! Resolve a stable tree path to its packed one-based range.
        class(mlp_t), intent(in) :: self
        character(*), intent(in) :: name
        integer, intent(out) :: first, last
        logical, intent(out) :: found
        type(mlp_parameter_block_t), allocatable :: layout(:)
        integer :: i

        first = 0
        last = -1
        found = .false.
        layout = self%parameter_layout()
        do i = 1, size(layout)
            if (trim(layout(i)%name) == trim(name)) then
                first = layout(i)%first
                last = layout(i)%last
                found = .true.
                return
            end if
        end do
    end subroutine mlp_parameter_range

    function mlp_parameters(self) result(theta)
        class(mlp_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        integer :: i, position, nweight, nbias

        allocate(theta(self%parameter_count()))
        position = 1
        do i = 1, size(self%layer)
            nweight = size(self%layer(i)%weight)
            nbias = size(self%layer(i)%bias)
            theta(position:position + nweight - 1) = &
                reshape(self%layer(i)%weight, [nweight])
            position = position + nweight
            theta(position:position + nbias - 1) = self%layer(i)%bias
            position = position + nbias
        end do
    end function mlp_parameters

    subroutine mlp_set_parameters(self, theta, status)
        class(mlp_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, position, nweight, nbias

        if (.not. model_allocated(self) .or. size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP set_parameters: model or parameter shape is invalid")
            return
        end if

        position = 1
        do i = 1, size(self%layer)
            nweight = size(self%layer(i)%weight)
            nbias = size(self%layer(i)%bias)
            self%layer(i)%weight = reshape(theta(position:position + nweight - 1), &
                shape(self%layer(i)%weight))
            position = position + nweight
            self%layer(i)%bias = theta(position:position + nbias - 1)
            position = position + nbias
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_set_parameters

    subroutine mlp_set_linear_parameters(self, weight1, bias1, weight2, bias2, status)
        !! Replace the complete state of an initialized two-layer linear MLP.
        !!
        !! Validation is completed before any layer is modified.  In
        !! particular, a nonlinear topology, a wrong shape, or a nonfinite
        !! supplied value is rejected with ``FORTNUM_DOMAIN_ERROR``.
        class(mlp_t), intent(inout) :: self
        real(dp), intent(in) :: weight1(:, :), bias1(:)
        real(dp), intent(in) :: weight2(:, :), bias2(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. valid_linear_state(weight1, bias1, weight2, bias2) .or. &
            .not. linear_model_matches(self, weight1, bias1, weight2, bias2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP set_linear_parameters: model, dimensions, or values are invalid")
            return
        end if

        self%layer(1)%weight = weight1
        self%layer(1)%bias = bias1
        self%layer(2)%weight = weight2
        self%layer(2)%bias = bias2
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_set_linear_parameters

    subroutine mlp_predict(self, x, y, status)
        class(mlp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(matrix_holder_t), allocatable :: activations(:), preactivations(:)

        if (.not. valid_batch_shape(self, x, y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP predict: model or batch shape is invalid")
            return
        end if
        call forward_cache(self, x, activations, preactivations, status)
        if (status%code /= FORTNUM_OK) return
        y = activations(size(activations))%value
    end subroutine mlp_predict

    subroutine mlp_jvp(self, x, dtheta, dx, y, dy, status)
        class(mlp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), dtheta(:), dx(:, :)
        real(dp), intent(out) :: y(:, :), dy(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(matrix_holder_t), allocatable :: activations(:), preactivations(:)
        real(dp), allocatable :: da(:, :), dz(:, :), next_da(:, :), derivative(:, :)
        real(dp), allocatable :: dweight(:, :), dbias(:)
        integer :: i, position, nweight, nbias

        if (.not. valid_batch_shape(self, x, y) .or. any(shape(dx) /= shape(x)) .or. &
            any(shape(dy) /= shape(y)) .or. size(dtheta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP jvp: model, tangent, or batch shape is invalid")
            return
        end if
        call forward_cache(self, x, activations, preactivations, status)
        if (status%code /= FORTNUM_OK) return

        allocate(da(size(x, 1), size(x, 2)))
        da = dx
        position = 1
        do i = 1, size(self%layer)
            nweight = size(self%layer(i)%weight)
            nbias = size(self%layer(i)%bias)
            allocate(dweight, mold=self%layer(i)%weight)
            allocate(dbias, mold=self%layer(i)%bias)
            dweight = reshape(dtheta(position:position + nweight - 1), &
                shape(dweight))
            position = position + nweight
            dbias = dtheta(position:position + nbias - 1)
            position = position + nbias

            allocate(dz(size(x, 1), size(self%layer(i)%bias)))
            dz = matmul(da, self%layer(i)%weight) + &
                matmul(activations(i)%value, dweight)
            dz = dz + spread(dbias, dim=1, ncopies=size(x, 1))
            allocate(derivative, mold=dz)
            call activation_derivative(preactivations(i)%value, derivative, &
                layer_activation(self, i))
            allocate(next_da, mold=dz)
            next_da = dz*derivative
            deallocate(dweight, dbias, dz, derivative, da)
            call move_alloc(next_da, da)
        end do
        y = activations(size(activations))%value
        dy = da
    end subroutine mlp_jvp

    subroutine mlp_vjp(self, x, u, parameter_bar, x_bar, status)
        class(mlp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: parameter_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(matrix_holder_t), allocatable :: activations(:), preactivations(:)
        real(dp), allocatable :: adjoint(:, :), dz(:, :), derivative(:, :)
        real(dp), allocatable :: weight_bar(:, :), bias_bar(:), previous_adjoint(:, :)
        integer :: i, position, nweight, nbias

        if (.not. valid_batch_shape(self, x, u) .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP vjp: model, cotangent, or output shape is invalid")
            return
        end if
        call forward_cache(self, x, activations, preactivations, status)
        if (status%code /= FORTNUM_OK) return

        allocate(adjoint, mold=u)
        adjoint = u
        parameter_bar = 0.0_dp
        position = self%parameter_count()
        do i = size(self%layer), 1, -1
            allocate(dz, mold=adjoint)
            dz = adjoint
            allocate(derivative, mold=dz)
            call activation_derivative(preactivations(i)%value, derivative, &
                layer_activation(self, i))
            dz = dz*derivative

            allocate(weight_bar, mold=self%layer(i)%weight)
            allocate(bias_bar, mold=self%layer(i)%bias)
            weight_bar = matmul(transpose(activations(i)%value), dz)
            bias_bar = sum(dz, dim=1)
            nweight = size(weight_bar)
            nbias = size(bias_bar)
            parameter_bar(position - nbias + 1:position) = bias_bar
            position = position - nbias
            parameter_bar(position - nweight + 1:position) = &
                reshape(weight_bar, [nweight])
            position = position - nweight

            allocate(previous_adjoint(size(x, 1), size(self%layer(i)%weight, 1)))
            previous_adjoint = matmul(dz, transpose(self%layer(i)%weight))
            deallocate(dz, derivative, weight_bar, bias_bar, adjoint)
            call move_alloc(previous_adjoint, adjoint)
        end do
        x_bar = adjoint
    end subroutine mlp_vjp

    subroutine mlp_hvp(self, x, u, dtheta, dx, parameter_hvp, x_hvp, status)
        !! Differentiate the VJP of the fixed output cotangent `u` in the
        !! joint direction `(dtheta, dx)`. This is a forward-over-reverse
        !! Hessian-vector product for the scalar contraction `sum(u*y)`.
        class(mlp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), dtheta(:), dx(:, :)
        real(dp), intent(out) :: parameter_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(matrix_holder_t), allocatable :: activations(:), preactivations(:)
        type(matrix_holder_t), allocatable :: tangent_activations(:), &
            tangent_preactivations(:)
        type(dense_layer_t), allocatable :: tangent_layer(:)
        real(dp), allocatable :: adjoint(:, :), adjoint_tangent(:, :)
        real(dp), allocatable :: dz(:, :), dz_tangent(:, :)
        real(dp), allocatable :: derivative(:, :), second_derivative(:, :)
        real(dp), allocatable :: weight_hvp(:, :), bias_hvp(:)
        real(dp), allocatable :: previous_adjoint(:, :), &
            previous_adjoint_tangent(:, :)
        integer :: i, position, nweight, nbias

        if (.not. valid_batch_shape(self, x, u)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hvp: model or batch shape is invalid")
            return
        end if
        if (any(shape(dx) /= shape(x)) .or. &
            any(shape(x_hvp) /= shape(x)) .or. &
            size(dtheta) /= self%parameter_count() .or. &
            size(parameter_hvp) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP hvp: direction or output shape is invalid")
            return
        end if

        call forward_cache(self, x, activations, preactivations, status)
        if (status%code /= FORTNUM_OK) return
        allocate(tangent_activations(size(activations)))
        allocate(tangent_preactivations(size(preactivations)))
        allocate(tangent_layer(size(self%layer)))
        allocate(tangent_activations(1)%value, source=dx)
        position = 1
        do i = 1, size(self%layer)
            nweight = size(self%layer(i)%weight)
            nbias = size(self%layer(i)%bias)
            allocate(tangent_layer(i)%weight, source=self%layer(i)%weight)
            allocate(tangent_layer(i)%bias, source=self%layer(i)%bias)
            tangent_layer(i)%weight = reshape( &
                dtheta(position:position + nweight - 1), &
                shape(tangent_layer(i)%weight))
            position = position + nweight
            tangent_layer(i)%bias = dtheta(position:position + nbias - 1)
            position = position + nbias

            allocate(tangent_preactivations(i)%value(size(x, 1), &
                size(self%layer(i)%bias)))
            tangent_preactivations(i)%value = &
                matmul(tangent_activations(i)%value, self%layer(i)%weight) + &
                matmul(activations(i)%value, tangent_layer(i)%weight) + &
                spread(tangent_layer(i)%bias, dim=1, ncopies=size(x, 1))
            allocate(tangent_activations(i + 1)%value, &
                mold=tangent_preactivations(i)%value)
            call activation_derivative(preactivations(i)%value, &
                tangent_activations(i + 1)%value, layer_activation(self, i))
            tangent_activations(i + 1)%value = &
                tangent_preactivations(i)%value * &
                tangent_activations(i + 1)%value
        end do

        allocate(adjoint, mold=u)
        allocate(adjoint_tangent, mold=u)
        adjoint = u
        adjoint_tangent = 0.0_dp
        parameter_hvp = 0.0_dp
        position = self%parameter_count()
        do i = size(self%layer), 1, -1
            allocate(derivative, mold=adjoint)
            allocate(second_derivative, mold=adjoint)
            call activation_derivative(preactivations(i)%value, derivative, &
                layer_activation(self, i))
            call activation_second_derivative(preactivations(i)%value, &
                second_derivative, layer_activation(self, i))
            allocate(dz, mold=adjoint)
            allocate(dz_tangent, mold=adjoint)
            dz = adjoint * derivative
            dz_tangent = adjoint_tangent * derivative + adjoint * &
                second_derivative * tangent_preactivations(i)%value

            allocate(weight_hvp, mold=self%layer(i)%weight)
            allocate(bias_hvp, mold=self%layer(i)%bias)
            weight_hvp = matmul(transpose(tangent_activations(i)%value), dz) + &
                matmul(transpose(activations(i)%value), dz_tangent)
            bias_hvp = sum(dz_tangent, dim=1)
            nweight = size(weight_hvp)
            nbias = size(bias_hvp)
            parameter_hvp(position - nbias + 1:position) = bias_hvp
            position = position - nbias
            parameter_hvp(position - nweight + 1:position) = &
                reshape(weight_hvp, [nweight])
            position = position - nweight

            allocate(previous_adjoint(size(x, 1), &
                size(self%layer(i)%weight, 1)))
            allocate(previous_adjoint_tangent(size(x, 1), &
                size(self%layer(i)%weight, 1)))
            previous_adjoint = matmul(dz, transpose(self%layer(i)%weight))
            previous_adjoint_tangent = matmul(dz_tangent, &
                transpose(self%layer(i)%weight)) + matmul(dz, &
                transpose(tangent_layer(i)%weight))
            deallocate(derivative, second_derivative, dz, dz_tangent, &
                weight_hvp, bias_hvp, adjoint, adjoint_tangent)
            call move_alloc(previous_adjoint, adjoint)
            call move_alloc(previous_adjoint_tangent, adjoint_tangent)
        end do
        x_hvp = adjoint_tangent
    end subroutine mlp_hvp

    subroutine forward_cache(self, x, activations, preactivations, status)
        class(mlp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(matrix_holder_t), allocatable, intent(out) :: activations(:), &
            preactivations(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, n_samples, n_outputs

        if (.not. model_allocated(self) .or. size(x, 2) /= self%layer_sizes(1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP forward: model or input shape is invalid")
            return
        end if

        n_samples = size(x, 1)
        allocate(activations(size(self%layer) + 1))
        allocate(preactivations(size(self%layer)))
        allocate(activations(1)%value, source=x)
        do i = 1, size(self%layer)
            n_outputs = self%layer_sizes(i + 1)
            allocate(preactivations(i)%value(n_samples, n_outputs))
            allocate(activations(i + 1)%value(n_samples, n_outputs))
            preactivations(i)%value = matmul(activations(i)%value, &
                self%layer(i)%weight) + &
                spread(self%layer(i)%bias, dim=1, ncopies=n_samples)
            call apply_activation(preactivations(i)%value, activations(i + 1)%value, &
                layer_activation(self, i))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine forward_cache

    integer function layer_activation(self, index) result(kind)
        class(mlp_t), intent(in) :: self
        integer, intent(in) :: index

        if (index < size(self%layer)) then
            kind = self%hidden_activation
        else
            kind = self%output_activation
        end if
    end function layer_activation

    logical function valid_batch_shape(self, x, y) result(valid)
        class(mlp_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)

        valid = model_allocated(self)
        if (.not. valid) return
        valid = size(x, 1) > 0 .and. size(x, 2) == self%layer_sizes(1) .and. &
            all(shape(y) == [size(x, 1), self%layer_sizes(size(self%layer_sizes))])
    end function valid_batch_shape

    logical function model_allocated(self) result(valid)
        class(mlp_t), intent(in) :: self

        valid = allocated(self%layer_sizes) .and. allocated(self%layer)
        if (.not. valid) return
        valid = size(self%layer_sizes) >= 2 .and. &
            size(self%layer) == size(self%layer_sizes) - 1
    end function model_allocated

    logical function valid_linear_state(weight1, bias1, weight2, bias2) result(valid)
        real(dp), intent(in) :: weight1(:, :), bias1(:)
        real(dp), intent(in) :: weight2(:, :), bias2(:)

        valid = size(weight1, 1) > 0 .and. size(weight1, 2) > 0 .and. &
            size(weight2, 1) > 0 .and. size(weight2, 2) > 0 .and. &
            size(bias1) == size(weight1, 2) .and. &
            size(weight2, 1) == size(weight1, 2) .and. &
            size(bias2) == size(weight2, 2)
        if (.not. valid) return
        valid = all(ieee_is_finite(weight1)) .and. all(ieee_is_finite(bias1)) .and. &
            all(ieee_is_finite(weight2)) .and. all(ieee_is_finite(bias2))
    end function valid_linear_state

    logical function linear_model_matches(self, weight1, bias1, weight2, bias2) result(valid)
        class(mlp_t), intent(in) :: self
        real(dp), intent(in) :: weight1(:, :), bias1(:)
        real(dp), intent(in) :: weight2(:, :), bias2(:)

        valid = model_allocated(self)
        if (.not. valid) return
        valid = size(self%layer) == 2 .and. self%hidden_activation == MLP_LINEAR .and. &
            self%output_activation == MLP_LINEAR
        if (.not. valid) return
        valid = size(self%layer(1)%weight, 1) == size(weight1, 1) .and. &
            size(self%layer(1)%weight, 2) == size(weight1, 2) .and. &
            size(self%layer(1)%bias) == size(bias1) .and. &
            size(self%layer(2)%weight, 1) == size(weight2, 1) .and. &
            size(self%layer(2)%weight, 2) == size(weight2, 2) .and. &
            size(self%layer(2)%bias) == size(bias2)
    end function linear_model_matches

    logical function valid_activation(kind) result(valid)
        integer, intent(in) :: kind

        valid = kind == MLP_LINEAR .or. kind == MLP_TANH .or. kind == MLP_RELU .or. &
            kind == MLP_GELU .or. kind == MLP_SILU .or. kind == MLP_ELU .or. &
            kind == MLP_SOFTPLUS .or. kind == MLP_LEAKY_RELU .or. &
            kind == MLP_SIGMOID .or. kind == MLP_MISH
    end function valid_activation

    subroutine apply_activation(input, output, kind)
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        integer, intent(in) :: kind
        integer :: i, j

        do j = 1, size(input, 2)
            do i = 1, size(input, 1)
                select case (kind)
                case (MLP_LINEAR)
                    output(i, j) = input(i, j)
                case (MLP_TANH)
                    output(i, j) = tanh(input(i, j))
                case (MLP_RELU)
                    output(i, j) = max(0.0_dp, input(i, j))
                case (MLP_GELU)
                    output(i, j) = gelu_value(input(i, j))
                case (MLP_SILU)
                    output(i, j) = input(i, j)*sigmoid_value(input(i, j))
                case (MLP_ELU)
                    if (input(i, j) > 0.0_dp) then
                        output(i, j) = input(i, j)
                    else
                        output(i, j) = exp(input(i, j)) - 1.0_dp
                    end if
                case (MLP_SOFTPLUS)
                    output(i, j) = softplus_value(input(i, j))
                case (MLP_LEAKY_RELU)
                    output(i, j) = merge(input(i, j), leaky_relu_slope*input(i, j), &
                        input(i, j) >= 0.0_dp)
                case (MLP_SIGMOID)
                    output(i, j) = sigmoid_value(input(i, j))
                case (MLP_MISH)
                    output(i, j) = mish_value(input(i, j))
                end select
            end do
        end do
    end subroutine apply_activation

    subroutine activation_derivative(input, output, kind)
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        integer, intent(in) :: kind
        integer :: i, j
        real(dp) :: value

        do j = 1, size(input, 2)
            do i = 1, size(input, 1)
                select case (kind)
                case (MLP_LINEAR)
                    output(i, j) = 1.0_dp
                case (MLP_TANH)
                    value = tanh(input(i, j))
                    output(i, j) = 1.0_dp - value*value
                case (MLP_RELU)
                    output(i, j) = merge(1.0_dp, 0.0_dp, input(i, j) > 0.0_dp)
                case (MLP_GELU)
                    output(i, j) = gelu_first_derivative(input(i, j))
                case (MLP_SILU)
                    value = sigmoid_value(input(i, j))
                    output(i, j) = value + input(i, j)*value*(1.0_dp - value)
                case (MLP_ELU)
                    if (input(i, j) > 0.0_dp) then
                        output(i, j) = 1.0_dp
                    else
                        output(i, j) = exp(input(i, j))
                    end if
                case (MLP_SOFTPLUS)
                    output(i, j) = sigmoid_value(input(i, j))
                case (MLP_LEAKY_RELU)
                    output(i, j) = merge(1.0_dp, leaky_relu_slope, &
                        input(i, j) >= 0.0_dp)
                case (MLP_SIGMOID)
                    value = sigmoid_value(input(i, j))
                    output(i, j) = value*(1.0_dp - value)
                case (MLP_MISH)
                    output(i, j) = mish_first_derivative(input(i, j))
                end select
            end do
        end do
    end subroutine activation_derivative

    subroutine activation_second_derivative(input, output, kind)
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        integer, intent(in) :: kind
        integer :: i, j
        real(dp) :: value

        do j = 1, size(input, 2)
            do i = 1, size(input, 1)
                select case (kind)
                case (MLP_LINEAR, MLP_RELU, MLP_LEAKY_RELU)
                    output(i, j) = 0.0_dp
                case (MLP_TANH)
                    value = tanh(input(i, j))
                    output(i, j) = -2.0_dp*value*(1.0_dp - value*value)
                case (MLP_GELU)
                    output(i, j) = gelu_second_derivative(input(i, j))
                case (MLP_SILU)
                    value = sigmoid_value(input(i, j))
                    output(i, j) = value*(1.0_dp - value)* &
                        (2.0_dp + input(i, j)*(1.0_dp - 2.0_dp*value))
                case (MLP_ELU)
                    if (input(i, j) > 0.0_dp) then
                        output(i, j) = 0.0_dp
                    else
                        output(i, j) = exp(input(i, j))
                    end if
                case (MLP_SOFTPLUS)
                    value = sigmoid_value(input(i, j))
                    output(i, j) = value*(1.0_dp - value)
                case (MLP_SIGMOID)
                    value = sigmoid_value(input(i, j))
                    output(i, j) = value*(1.0_dp - value)*(1.0_dp - 2.0_dp*value)
                case (MLP_MISH)
                    output(i, j) = mish_second_derivative(input(i, j))
                end select
            end do
        end do
    end subroutine activation_second_derivative

    pure real(dp) function sigmoid_value(x) result(value)
        real(dp), intent(in) :: x

        if (x >= 0.0_dp) then
            value = 1.0_dp/(1.0_dp + exp(-x))
        else
            value = exp(x)/(1.0_dp + exp(x))
        end if
    end function sigmoid_value

    pure real(dp) function softplus_value(x) result(value)
        real(dp), intent(in) :: x

        if (x > 0.0_dp) then
            value = x + log(1.0_dp + exp(-x))
        else
            value = log(1.0_dp + exp(x))
        end if
    end function softplus_value

    pure real(dp) function gelu_value(x) result(value)
        real(dp), intent(in) :: x
        real(dp) :: t, u

        u = gelu_c*(x + gelu_k*x*x*x)
        t = tanh(u)
        value = 0.5_dp*x*(1.0_dp + t)
    end function gelu_value

    pure real(dp) function gelu_first_derivative(x) result(value)
        real(dp), intent(in) :: x
        real(dp) :: t, q, u_prime, u

        u = gelu_c*(x + gelu_k*x*x*x)
        t = tanh(u)
        q = 1.0_dp - t*t
        u_prime = gelu_c*(1.0_dp + 3.0_dp*gelu_k*x*x)
        value = 0.5_dp*(1.0_dp + t) + 0.5_dp*x*q*u_prime
    end function gelu_first_derivative

    pure real(dp) function gelu_second_derivative(x) result(value)
        real(dp), intent(in) :: x
        real(dp) :: t, q, u_prime, u_second, u

        u = gelu_c*(x + gelu_k*x*x*x)
        t = tanh(u)
        q = 1.0_dp - t*t
        u_prime = gelu_c*(1.0_dp + 3.0_dp*gelu_k*x*x)
        u_second = gelu_c*6.0_dp*gelu_k*x
        value = q*u_prime + 0.5_dp*x*q*(u_second - 2.0_dp*t*u_prime*u_prime)
    end function gelu_second_derivative

    pure real(dp) function mish_value(x) result(value)
        real(dp), intent(in) :: x

        value = x*tanh(softplus_value(x))
    end function mish_value

    pure real(dp) function mish_first_derivative(x) result(value)
        real(dp), intent(in) :: x
        real(dp) :: softplus, tangent, sigmoid

        softplus = softplus_value(x)
        tangent = tanh(softplus)
        sigmoid = sigmoid_value(x)
        value = tangent + x*(1.0_dp - tangent*tangent)*sigmoid
    end function mish_first_derivative

    pure real(dp) function mish_second_derivative(x) result(value)
        real(dp), intent(in) :: x
        real(dp) :: softplus, tangent, sigmoid, first_factor

        softplus = softplus_value(x)
        tangent = tanh(softplus)
        sigmoid = sigmoid_value(x)
        first_factor = (1.0_dp - tangent*tangent)*sigmoid
        value = 2.0_dp*first_factor + x*first_factor* &
            (1.0_dp - sigmoid - 2.0_dp*tangent*sigmoid)
    end function mish_second_derivative

end module fortml_mlp
