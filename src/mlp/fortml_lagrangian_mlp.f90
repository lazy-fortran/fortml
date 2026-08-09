module fortml_lagrangian_mlp
    !! Scalar Lagrangian MLPs and Euler--Lagrange residuals.
    !!
    !! The network represents a time-independent scalar
    !! ``L(q,v)`` with input ``[q,v]``.  Its velocity Hessian is the
    !! mass matrix and the residual for a supplied acceleration is
    !! ``L_vq*v + L_vv*a - L_q``.  Pure input Hessian-vector products are
    !! obtained from the existing MLP reverse-over-forward product with a zero
    !! parameter direction; no finite-difference approximation is hidden in
    !! the production path.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    implicit none
    private

    type, public :: lagrangian_mlp_t
        private
        type(mlp_t) :: model
        integer :: n_coordinates = 0
        integer :: device_kind = FORTML_DEVICE_CPU
        logical :: ready = .false.
    contains
        procedure, public :: initialize => lagrangian_mlp_initialize
        procedure, public :: initialized => lagrangian_mlp_initialized
        procedure, public :: coordinate_count => lagrangian_mlp_coordinate_count
        procedure, public :: parameter_count => lagrangian_mlp_parameter_count
        procedure, public :: parameters => lagrangian_mlp_parameters
        procedure, public :: set_parameters => lagrangian_mlp_set_parameters
        procedure, public :: lagrangian => lagrangian_mlp_value
        procedure, public :: lagrangian_gradient => lagrangian_mlp_value_gradient
        procedure, public :: mass_matrix => lagrangian_mlp_mass_matrix
        procedure, public :: euler_lagrange_residual => &
            lagrangian_mlp_euler_lagrange_residual
        procedure, public :: device_supported => lagrangian_mlp_device_supported
        procedure, public :: select_device => lagrangian_mlp_select_device
    end type lagrangian_mlp_t

contains

    subroutine lagrangian_mlp_initialize(self, n_coordinates, layers, status, &
            hidden_activation, initialization_seed, device_kind)
        class(lagrangian_mlp_t), intent(out) :: self
        integer, intent(in) :: n_coordinates, layers(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_activation, initialization_seed, device_kind
        integer :: activation, seed, requested
        integer, allocatable :: full_layers(:)

        activation = MLP_TANH
        if (present(hidden_activation)) activation = hidden_activation
        seed = 17
        if (present(initialization_seed)) seed = initialization_seed
        requested = FORTML_DEVICE_CPU
        if (present(device_kind)) requested = device_kind
        self%ready = .false.
        self%n_coordinates = 0
        self%device_kind = FORTML_DEVICE_CPU
        if (n_coordinates < 1 .or. size(layers) < 1 .or. any(layers < 1) .or. &
            seed < 0 .or. requested < FORTML_DEVICE_CPU .or. &
            requested > FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP initialize: invalid coordinates, layers, seed, or device")
            return
        end if
        if (requested == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "Lagrangian MLP initialize: resident CUDA derivative graph is unavailable")
            return
        end if
        allocate(full_layers(size(layers)+1))
        full_layers(1) = 2*n_coordinates
        full_layers(2:) = layers
        if (full_layers(size(full_layers)) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP initialize: final layer must be scalar")
            return
        end if
        call self%model%initialize(full_layers, status, hidden_activation=activation, &
            output_activation=MLP_LINEAR, initialization_seed=seed)
        if (.not. status_ok(status)) return
        self%n_coordinates = n_coordinates
        self%device_kind = requested
        self%ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine lagrangian_mlp_initialize

    logical function lagrangian_mlp_initialized(self) result(yes)
        class(lagrangian_mlp_t), intent(in) :: self

        yes = self%ready .and. self%n_coordinates > 0 .and. &
            self%model%parameter_count() > 0
    end function lagrangian_mlp_initialized

    integer function lagrangian_mlp_coordinate_count(self) result(count)
        class(lagrangian_mlp_t), intent(in) :: self

        count = self%n_coordinates
    end function lagrangian_mlp_coordinate_count

    integer function lagrangian_mlp_parameter_count(self) result(count)
        class(lagrangian_mlp_t), intent(in) :: self

        count = 0
        if (self%initialized()) count = self%model%parameter_count()
    end function lagrangian_mlp_parameter_count

    function lagrangian_mlp_parameters(self) result(theta)
        class(lagrangian_mlp_t), intent(in) :: self
        real(dp), allocatable :: theta(:)

        theta = self%model%parameters()
    end function lagrangian_mlp_parameters

    subroutine lagrangian_mlp_set_parameters(self, theta, status)
        class(lagrangian_mlp_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP set_parameters: model is not initialized")
            return
        end if
        if (size(theta) /= self%model%parameter_count() .or. &
            any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP set_parameters: parameter shape is invalid")
            return
        end if
        call self%model%set_parameters(theta, status)
    end subroutine lagrangian_mlp_set_parameters

    subroutine lagrangian_mlp_value(self, q, v, values, status)
        class(lagrangian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: q(:, :), v(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: state(:, :), output(:, :)

        if (.not. valid_state(self, q, v, status)) return
        if (size(values) /= size(q, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP lagrangian: output shape is invalid")
            return
        end if
        call join_state(q, v, state)
        allocate(output(size(q, 1), 1))
        call self%model%predict(state, output, status)
        if (.not. status_ok(status)) return
        values = output(:, 1)
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP lagrangian: output is non-finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lagrangian_mlp_value

    subroutine lagrangian_mlp_value_gradient(self, q, v, values, gradient, status)
        class(lagrangian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: q(:, :), v(:, :)
        real(dp), intent(out) :: values(:), gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: state(:, :), output(:, :), one(:, :), parameter_bar(:)
        integer :: n, d

        if (.not. valid_state(self, q, v, status)) return
        n = size(q, 1)
        d = self%n_coordinates
        if (size(values) /= n .or. any(shape(gradient) /= [n, 2*d])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP gradient: output shape is invalid")
            return
        end if
        call join_state(q, v, state)
        allocate(output(n, 1), one(n, 1), parameter_bar(self%model%parameter_count()))
        one = 1.0_dp
        call self%model%predict(state, output, status)
        if (.not. status_ok(status)) return
        call self%model%vjp(state, one, parameter_bar, gradient, status)
        if (.not. status_ok(status)) return
        values = output(:, 1)
        if (any(.not. ieee_is_finite(values)) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP gradient: product is non-finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lagrangian_mlp_value_gradient

    subroutine lagrangian_mlp_mass_matrix(self, q, v, mass, status)
        class(lagrangian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: q(:, :), v(:, :)
        real(dp), intent(out) :: mass(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: hessian(:, :)
        integer :: i, d

        if (.not. valid_state(self, q, v, status)) return
        d = self%n_coordinates
        if (any(shape(mass) /= [size(q, 1), d, d])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP mass_matrix: output shape is invalid")
            return
        end if
        do i = 1, size(q, 1)
            call point_hessian(self, q(i, :), v(i, :), hessian, status)
            if (.not. status_ok(status)) return
            mass(i, :, :) = hessian(d+1:2*d, d+1:2*d)
            if (.not. matrix_nonsingular(mass(i, :, :))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Lagrangian MLP mass_matrix: velocity Hessian is singular")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lagrangian_mlp_mass_matrix

    subroutine lagrangian_mlp_euler_lagrange_residual(self, q, v, acceleration, &
            residual, status)
        class(lagrangian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: q(:, :), v(:, :), acceleration(:, :)
        real(dp), intent(out) :: residual(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:, :), hessian(:, :), values(:)
        integer :: i, d, n

        if (.not. valid_state(self, q, v, status)) return
        n = size(q, 1)
        d = self%n_coordinates
        if (any(shape(acceleration) /= [n, d]) .or. any(shape(residual) /= [n, d]) .or. &
            any(.not. ieee_is_finite(acceleration))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP residual: acceleration or output shape is invalid")
            return
        end if
        allocate(values(n), gradient(n, 2*d))
        call self%lagrangian_gradient(q, v, values, gradient, status)
        if (.not. status_ok(status)) return
        do i = 1, n
            call point_hessian(self, q(i, :), v(i, :), hessian, status)
            if (.not. status_ok(status)) return
            if (.not. matrix_nonsingular(hessian(d+1:2*d, d+1:2*d))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Lagrangian MLP residual: velocity Hessian is singular")
                return
            end if
            residual(i, :) = matmul(hessian(d+1:2*d, 1:d), v(i, :)) + &
                matmul(hessian(d+1:2*d, d+1:2*d), acceleration(i, :)) - &
                gradient(i, 1:d)
        end do
        if (any(.not. ieee_is_finite(residual))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP residual: result is non-finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lagrangian_mlp_euler_lagrange_residual

    logical function lagrangian_mlp_device_supported(self, device_kind) result(yes)
        class(lagrangian_mlp_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = self%initialized() .and. requested == FORTML_DEVICE_CPU
    end function lagrangian_mlp_device_supported

    subroutine lagrangian_mlp_select_device(self, device_kind, status)
        class(lagrangian_mlp_t), intent(inout) :: self
        integer, intent(in) :: device_kind
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP select_device: model is not initialized")
            return
        end if
        if (device_kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "Lagrangian MLP select_device: resident CUDA derivative graph is unavailable")
            return
        end if
        if (device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP select_device: invalid device kind")
            return
        end if
        self%device_kind = device_kind
        call status_set(status, FORTNUM_OK, "")
    end subroutine lagrangian_mlp_select_device

    subroutine point_hessian(self, q, v, hessian, status)
        class(lagrangian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: q(:), v(:)
        real(dp), allocatable, intent(out) :: hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: state(:, :), u(:, :), dtheta(:), dx(:, :)
        real(dp), allocatable :: parameter_hvp(:), x_hvp(:, :)
        integer :: d, j

        d = self%n_coordinates
        call join_point(q, v, state)
        allocate(hessian(2*d, 2*d), u(1, 1), dtheta(self%parameter_count()), &
            parameter_hvp(self%parameter_count()), dx(1, 2*d), x_hvp(1, 2*d))
        u = 1.0_dp
        dtheta = 0.0_dp
        do j = 1, 2*d
            dx = 0.0_dp
            dx(1, j) = 1.0_dp
            call self%model%hvp(state, u, dtheta, dx, parameter_hvp, x_hvp, status)
            if (.not. status_ok(status)) return
            hessian(:, j) = x_hvp(1, :)
        end do
        if (any(.not. ieee_is_finite(hessian))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP Hessian: product is non-finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine point_hessian

    logical function valid_state(self, q, v, status) result(valid)
        class(lagrangian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: q(:, :), v(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP: model is not initialized")
            return
        end if
        if (size(q, 1) < 1 .or. size(q, 2) /= self%n_coordinates .or. &
            any(shape(v) /= shape(q)) .or. any(.not. ieee_is_finite(q)) .or. &
            any(.not. ieee_is_finite(v))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Lagrangian MLP: coordinate or velocity shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_state

    subroutine join_state(q, v, state)
        real(dp), intent(in) :: q(:, :), v(:, :)
        real(dp), allocatable, intent(out) :: state(:, :)

        allocate(state(size(q, 1), size(q, 2)+size(v, 2)))
        state(:, 1:size(q, 2)) = q
        state(:, size(q, 2)+1:) = v
    end subroutine join_state

    subroutine join_point(q, v, state)
        real(dp), intent(in) :: q(:), v(:)
        real(dp), allocatable, intent(out) :: state(:, :)

        allocate(state(1, size(q)+size(v)))
        state(1, 1:size(q)) = q
        state(1, size(q)+1:) = v
    end subroutine join_point

    logical function matrix_nonsingular(matrix) result(ok)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), allocatable :: work(:, :)
        real(dp) :: pivot, scale, temporary
        integer :: n, i, j, pivot_row

        n = size(matrix, 1)
        ok = size(matrix, 2) == n
        if (.not. ok .or. n < 1) return
        allocate(work(n, n))
        work = matrix
        scale = max(1.0_dp, maxval(abs(work)))
        do j = 1, n
            pivot_row = j
            do i = j+1, n
                if (abs(work(i, j)) > abs(work(pivot_row, j))) pivot_row = i
            end do
            pivot = work(pivot_row, j)
            if (abs(pivot) <= 1.0e-12_dp*scale) then
                ok = .false.
                return
            end if
            if (pivot_row /= j) then
                do i = j, n
                    temporary = work(j, i)
                    work(j, i) = work(pivot_row, i)
                    work(pivot_row, i) = temporary
                end do
            end if
            do i = j+1, n
                work(i, j:n) = work(i, j:n) - work(i, j)/work(j, j)*work(j, j:n)
            end do
        end do
    end function matrix_nonsingular

end module fortml_lagrangian_mlp
