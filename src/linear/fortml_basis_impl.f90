module fortml_basis_impl
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_bspline, only: bspline_workspace_t, bspline_init, &
        bspline_set_knots, bspline_eval_basis, bspline_eval_deriv
    implicit none
    private

    abstract interface
        subroutine basis_value_callback(x, theta, phi, status)
            import :: dp, fortnum_status_t
            real(dp), intent(in) :: x(:, :), theta(:)
            real(dp), intent(out) :: phi(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine basis_value_callback

        subroutine basis_jvp_callback(x, theta, x_dot, theta_dot, phi, phi_dot, &
                status)
            import :: dp, fortnum_status_t
            real(dp), intent(in) :: x(:, :), theta(:), x_dot(:, :), theta_dot(:)
            real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine basis_jvp_callback

        subroutine basis_vjp_callback(x, theta, u, theta_bar, x_bar, status)
            import :: dp, fortnum_status_t
            real(dp), intent(in) :: x(:, :), theta(:), u(:, :)
            real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine basis_vjp_callback
    end interface

    type, abstract, public :: basis_impl_t
    contains
        procedure(basis_impl_input_count), deferred :: input_count
        procedure(basis_impl_feature_count), deferred :: feature_count
        procedure(basis_impl_parameter_count), deferred :: parameter_count
        procedure(basis_impl_parameters), deferred :: parameters
        procedure(basis_impl_set_parameters), deferred :: set_parameters
        procedure(basis_impl_evaluate), deferred :: evaluate
        procedure(basis_impl_jvp), deferred :: jvp
        procedure(basis_impl_vjp), deferred :: vjp
        procedure(basis_impl_valid), deferred :: valid
        procedure :: static_lowering_eligible => basis_impl_static_eligible
    end type basis_impl_t

    abstract interface
        integer function basis_impl_input_count(self) result(count)
            import :: basis_impl_t
            class(basis_impl_t), intent(in) :: self
        end function basis_impl_input_count

        integer function basis_impl_feature_count(self) result(count)
            import :: basis_impl_t
            class(basis_impl_t), intent(in) :: self
        end function basis_impl_feature_count

        integer function basis_impl_parameter_count(self) result(count)
            import :: basis_impl_t
            class(basis_impl_t), intent(in) :: self
        end function basis_impl_parameter_count

        function basis_impl_parameters(self) result(theta)
            import :: basis_impl_t, dp
            class(basis_impl_t), intent(in) :: self
            real(dp), allocatable :: theta(:)
        end function basis_impl_parameters

        subroutine basis_impl_set_parameters(self, theta, status)
            import :: basis_impl_t, dp, fortnum_status_t
            class(basis_impl_t), intent(inout) :: self
            real(dp), intent(in) :: theta(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine basis_impl_set_parameters

        subroutine basis_impl_evaluate(self, x, phi, status)
            import :: basis_impl_t, dp, fortnum_status_t
            class(basis_impl_t), intent(in) :: self
            real(dp), intent(in) :: x(:, :)
            real(dp), intent(out) :: phi(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine basis_impl_evaluate

        subroutine basis_impl_jvp(self, x, theta_dot, x_dot, phi, phi_dot, status)
            import :: basis_impl_t, dp, fortnum_status_t
            class(basis_impl_t), intent(in) :: self
            real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
            real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine basis_impl_jvp

        subroutine basis_impl_vjp(self, x, u, theta_bar, x_bar, status)
            import :: basis_impl_t, dp, fortnum_status_t
            class(basis_impl_t), intent(in) :: self
            real(dp), intent(in) :: x(:, :), u(:, :)
            real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine basis_impl_vjp

        logical function basis_impl_valid(self) result(valid)
            import :: basis_impl_t
            class(basis_impl_t), intent(in) :: self
        end function basis_impl_valid
    end interface

    type, extends(basis_impl_t) :: polynomial_basis_impl_t
        integer :: n_inputs = 0
        integer :: degree = 0
    contains
        procedure :: input_count => polynomial_input_count
        procedure :: feature_count => polynomial_feature_count
        procedure :: parameter_count => polynomial_parameter_count
        procedure :: parameters => polynomial_parameters
        procedure :: set_parameters => polynomial_set_parameters
        procedure :: evaluate => polynomial_evaluate
        procedure :: jvp => polynomial_jvp
        procedure :: vjp => polynomial_vjp
        procedure :: valid => polynomial_valid
    end type polynomial_basis_impl_t

    type, extends(basis_impl_t) :: fourier_basis_impl_t
        integer :: n_inputs = 0
        integer :: n_harmonics = 0
        real(dp), allocatable :: log_frequencies(:, :)
    contains
        procedure :: input_count => fourier_input_count
        procedure :: feature_count => fourier_feature_count
        procedure :: parameter_count => fourier_parameter_count
        procedure :: parameters => fourier_parameters
        procedure :: set_parameters => fourier_set_parameters
        procedure :: evaluate => fourier_evaluate
        procedure :: jvp => fourier_jvp
        procedure :: vjp => fourier_vjp
        procedure :: valid => fourier_valid
    end type fourier_basis_impl_t

    type, extends(basis_impl_t) :: radial_basis_impl_t
        integer :: n_inputs = 0
        integer :: n_centers = 0
        real(dp), allocatable :: centers(:, :)
        real(dp), allocatable :: log_scales(:, :)
    contains
        procedure :: input_count => radial_input_count
        procedure :: feature_count => radial_feature_count
        procedure :: parameter_count => radial_parameter_count
        procedure :: parameters => radial_parameters
        procedure :: set_parameters => radial_set_parameters
        procedure :: evaluate => radial_evaluate
        procedure :: jvp => radial_jvp
        procedure :: vjp => radial_vjp
        procedure :: valid => radial_valid
    end type radial_basis_impl_t

    type, extends(basis_impl_t) :: spline_basis_impl_t
        integer :: n_inputs = 0
        type(bspline_workspace_t), allocatable :: spline(:)
    contains
        procedure :: input_count => spline_input_count
        procedure :: feature_count => spline_feature_count
        procedure :: parameter_count => spline_parameter_count
        procedure :: parameters => spline_parameters
        procedure :: set_parameters => spline_set_parameters
        procedure :: evaluate => spline_evaluate
        procedure :: jvp => spline_jvp
        procedure :: vjp => spline_vjp
        procedure :: valid => spline_valid
    end type spline_basis_impl_t

    type, extends(basis_impl_t) :: callback_basis_impl_t
        integer :: n_inputs = 0
        integer :: n_features = 0
        real(dp), allocatable :: parameters_value(:)
        procedure(basis_value_callback), pointer, nopass :: value_callback => null()
        procedure(basis_jvp_callback), pointer, nopass :: jvp_callback => null()
        procedure(basis_vjp_callback), pointer, nopass :: vjp_callback => null()
    contains
        procedure :: input_count => callback_input_count
        procedure :: feature_count => callback_feature_count
        procedure :: parameter_count => callback_parameter_count
        procedure :: parameters => callback_parameters
        procedure :: set_parameters => callback_set_parameters
        procedure :: evaluate => callback_evaluate
        procedure :: jvp => callback_jvp
        procedure :: vjp => callback_vjp
        procedure :: valid => callback_valid
        procedure :: static_lowering_eligible => callback_static_eligible
    end type callback_basis_impl_t

    public :: basis_value_callback, basis_jvp_callback, basis_vjp_callback
    public :: create_polynomial_impl, create_fourier_impl, create_radial_impl
    public :: create_spline_impl, create_callback_impl

contains

    subroutine create_polynomial_impl(n_inputs, degree, impl, status)
        integer, intent(in) :: n_inputs, degree
        class(basis_impl_t), allocatable, intent(out) :: impl
        type(fortnum_status_t), intent(out) :: status
        type(polynomial_basis_impl_t) :: value

        if (n_inputs < 1 .or. degree < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: n_inputs and degree must be positive")
            return
        end if
        value%n_inputs = n_inputs
        value%degree = degree
        allocate(impl, source=value)
        call status_set(status, FORTNUM_OK, "")
    end subroutine create_polynomial_impl

    subroutine create_fourier_impl(n_inputs, frequencies, impl, status)
        integer, intent(in) :: n_inputs
        real(dp), intent(in) :: frequencies(:, :)
        class(basis_impl_t), allocatable, intent(out) :: impl
        type(fortnum_status_t), intent(out) :: status
        type(fourier_basis_impl_t) :: value

        if (n_inputs < 1 .or. size(frequencies, 2) /= n_inputs .or. &
            size(frequencies, 1) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis Fourier: frequency shape is invalid")
            return
        end if
        if (any(frequencies <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis Fourier: frequencies must be positive")
            return
        end if
        value%n_inputs = n_inputs
        value%n_harmonics = size(frequencies, 1)
        allocate(value%log_frequencies(value%n_harmonics, n_inputs))
        value%log_frequencies = log(frequencies)
        allocate(impl, source=value)
        call status_set(status, FORTNUM_OK, "")
    end subroutine create_fourier_impl

    subroutine create_radial_impl(n_inputs, centers, scales, impl, status)
        integer, intent(in) :: n_inputs
        real(dp), intent(in) :: centers(:, :), scales(:, :)
        class(basis_impl_t), allocatable, intent(out) :: impl
        type(fortnum_status_t), intent(out) :: status
        type(radial_basis_impl_t) :: value

        if (n_inputs < 1 .or. size(centers, 1) /= n_inputs .or. &
            size(centers, 2) < 1 .or. any(shape(scales) /= shape(centers))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis radial: center and scale shapes are invalid")
            return
        end if
        if (any(scales <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis radial: scales must be positive")
            return
        end if
        value%n_inputs = n_inputs
        value%n_centers = size(centers, 2)
        allocate(value%centers(n_inputs, value%n_centers))
        allocate(value%log_scales(n_inputs, value%n_centers))
        value%centers = centers
        value%log_scales = log(scales)
        allocate(impl, source=value)
        call status_set(status, FORTNUM_OK, "")
    end subroutine create_radial_impl

    subroutine create_spline_impl(n_inputs, order, breakpoints, impl, status)
        integer, intent(in) :: n_inputs, order
        real(dp), intent(in) :: breakpoints(:, :)
        class(basis_impl_t), allocatable, intent(out) :: impl
        type(fortnum_status_t), intent(out) :: status
        type(spline_basis_impl_t) :: value
        integer :: j

        if (n_inputs < 1 .or. size(breakpoints, 2) /= n_inputs .or. &
            size(breakpoints, 1) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline: breakpoint shape is invalid")
            return
        end if
        allocate(value%spline(n_inputs))
        do j = 1, n_inputs
            call bspline_init(value%spline(j), order, size(breakpoints, 1), &
                status)
            if (status%code /= FORTNUM_OK) return
            call bspline_set_knots(value%spline(j), breakpoints(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        value%n_inputs = n_inputs
        allocate(impl, source=value)
        call status_set(status, FORTNUM_OK, "")
    end subroutine create_spline_impl

    subroutine create_callback_impl(n_inputs, n_features, parameters, value_proc, &
            jvp_proc, vjp_proc, impl, status)
        integer, intent(in) :: n_inputs, n_features
        real(dp), intent(in) :: parameters(:)
        procedure(basis_value_callback) :: value_proc
        procedure(basis_jvp_callback) :: jvp_proc
        procedure(basis_vjp_callback) :: vjp_proc
        class(basis_impl_t), allocatable, intent(out) :: impl
        type(fortnum_status_t), intent(out) :: status
        type(callback_basis_impl_t) :: value

        if (n_inputs < 1 .or. n_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis callback: dimensions must be positive")
            return
        end if
        value%n_inputs = n_inputs
        value%n_features = n_features
        allocate(value%parameters_value(size(parameters)))
        value%parameters_value = parameters
        value%value_callback => value_proc
        value%jvp_callback => jvp_proc
        value%vjp_callback => vjp_proc
        allocate(impl, source=value)
        call status_set(status, FORTNUM_OK, "")
    end subroutine create_callback_impl

    logical function basis_impl_static_eligible(self) result(eligible)
        class(basis_impl_t), intent(in) :: self

        eligible = self%valid()
    end function basis_impl_static_eligible

    integer function polynomial_input_count(self) result(count)
        class(polynomial_basis_impl_t), intent(in) :: self
        count = self%n_inputs
    end function polynomial_input_count

    integer function polynomial_feature_count(self) result(count)
        class(polynomial_basis_impl_t), intent(in) :: self
        count = self%n_inputs*self%degree
    end function polynomial_feature_count

    integer function polynomial_parameter_count(self) result(count)
        class(polynomial_basis_impl_t), intent(in) :: self
        count = 0
    end function polynomial_parameter_count

    function polynomial_parameters(self) result(theta)
        class(polynomial_basis_impl_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        allocate(theta(0))
    end function polynomial_parameters

    subroutine polynomial_set_parameters(self, theta, status)
        class(polynomial_basis_impl_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(theta) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: parameter shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_set_parameters

    subroutine polynomial_evaluate(self, x, phi, status)
        class(polynomial_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, p, column

        if (size(x, 2) /= self%n_inputs .or. &
            size(phi, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: array shape is invalid")
            return
        end if
        phi = 0.0_dp
        column = 1
        do j = 1, self%n_inputs
            do p = 1, self%degree
                phi(:, column) = x(:, j)**p
                column = column + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_evaluate

    subroutine polynomial_jvp(self, x, theta_dot, x_dot, phi, phi_dot, status)
        class(polynomial_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, p, column

        if (size(theta_dot) /= 0 .or. any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: tangent shape is invalid")
            return
        end if
        call self%evaluate(x, phi, status)
        if (status%code /= FORTNUM_OK) return
        phi_dot = 0.0_dp
        column = 1
        do j = 1, self%n_inputs
            do p = 1, self%degree
                phi_dot(:, column) = real(p, dp)*x(:, j)**(p - 1)*x_dot(:, j)
                column = column + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_jvp

    subroutine polynomial_vjp(self, x, u, theta_bar, x_bar, status)
        class(polynomial_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, p, column

        if (size(u, 2) /= self%feature_count() .or. &
            size(theta_bar) /= 0 .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: cotangent shape is invalid")
            return
        end if
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        column = 1
        do j = 1, self%n_inputs
            do p = 1, self%degree
                x_bar(:, j) = x_bar(:, j) + &
                    u(:, column)*real(p, dp)*x(:, j)**(p - 1)
                column = column + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_vjp

    logical function polynomial_valid(self) result(valid)
        class(polynomial_basis_impl_t), intent(in) :: self
        valid = self%n_inputs > 0 .and. self%degree > 0
    end function polynomial_valid

    integer function fourier_input_count(self) result(count)
        class(fourier_basis_impl_t), intent(in) :: self
        count = self%n_inputs
    end function fourier_input_count

    integer function fourier_feature_count(self) result(count)
        class(fourier_basis_impl_t), intent(in) :: self
        count = 2*self%n_inputs*self%n_harmonics
    end function fourier_feature_count

    integer function fourier_parameter_count(self) result(count)
        class(fourier_basis_impl_t), intent(in) :: self
        count = self%n_inputs*self%n_harmonics
    end function fourier_parameter_count

    function fourier_parameters(self) result(theta)
        class(fourier_basis_impl_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        allocate(theta(self%parameter_count()))
        if (size(theta) > 0) theta = reshape(self%log_frequencies, shape(theta))
    end function fourier_parameters

    subroutine fourier_set_parameters(self, theta, status)
        class(fourier_basis_impl_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis Fourier: parameter shape is invalid")
            return
        end if
        self%log_frequencies = reshape(theta, shape(self%log_frequencies))
        call status_set(status, FORTNUM_OK, "")
    end subroutine fourier_set_parameters

    subroutine fourier_evaluate(self, x, phi, status)
        class(fourier_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, h, column
        real(dp) :: frequency

        if (size(x, 2) /= self%n_inputs .or. &
            size(phi, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis Fourier: array shape is invalid")
            return
        end if
        phi = 0.0_dp
        column = 1
        do j = 1, self%n_inputs
            do h = 1, self%n_harmonics
                frequency = exp(self%log_frequencies(h, j))
                phi(:, column) = sin(frequency*x(:, j))
                phi(:, column + 1) = cos(frequency*x(:, j))
                column = column + 2
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fourier_evaluate

    subroutine fourier_jvp(self, x, theta_dot, x_dot, phi, phi_dot, status)
        class(fourier_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, h, column, parameter_index
        real(dp) :: frequency
        real(dp) :: argument(size(x, 1)), argument_dot(size(x, 1))

        if (size(theta_dot) /= self%parameter_count() .or. &
            any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis Fourier: tangent shape is invalid")
            return
        end if
        call self%evaluate(x, phi, status)
        if (status%code /= FORTNUM_OK) return
        phi_dot = 0.0_dp
        column = 1
        parameter_index = 1
        do j = 1, self%n_inputs
            do h = 1, self%n_harmonics
                frequency = exp(self%log_frequencies(h, j))
                argument = frequency*x(:, j)
                argument_dot = frequency*(x_dot(:, j) + &
                    x(:, j)*theta_dot(parameter_index))
                phi_dot(:, column) = cos(argument)*argument_dot
                phi_dot(:, column + 1) = -sin(argument)*argument_dot
                column = column + 2
                parameter_index = parameter_index + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fourier_jvp

    subroutine fourier_vjp(self, x, u, theta_bar, x_bar, status)
        class(fourier_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, h, column, parameter_index
        real(dp) :: frequency
        real(dp) :: argument(size(x, 1)), z_bar(size(x, 1))

        if (size(u, 2) /= self%feature_count() .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis Fourier: cotangent shape is invalid")
            return
        end if
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        column = 1
        parameter_index = 1
        do j = 1, self%n_inputs
            do h = 1, self%n_harmonics
                frequency = exp(self%log_frequencies(h, j))
                argument = frequency*x(:, j)
                z_bar = u(:, column)*cos(argument) - &
                    u(:, column + 1)*sin(argument)
                x_bar(:, j) = x_bar(:, j) + frequency*z_bar
                theta_bar(parameter_index) = sum(frequency*x(:, j)*z_bar)
                column = column + 2
                parameter_index = parameter_index + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fourier_vjp

    logical function fourier_valid(self) result(valid)
        class(fourier_basis_impl_t), intent(in) :: self
        valid = self%n_inputs > 0 .and. self%n_harmonics > 0
        if (.not. valid) return
        valid = allocated(self%log_frequencies)
    end function fourier_valid

    integer function radial_input_count(self) result(count)
        class(radial_basis_impl_t), intent(in) :: self
        count = self%n_inputs
    end function radial_input_count

    integer function radial_feature_count(self) result(count)
        class(radial_basis_impl_t), intent(in) :: self
        count = self%n_centers
    end function radial_feature_count

    integer function radial_parameter_count(self) result(count)
        class(radial_basis_impl_t), intent(in) :: self
        count = 2*self%n_inputs*self%n_centers
    end function radial_parameter_count

    function radial_parameters(self) result(theta)
        class(radial_basis_impl_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        integer :: n

        n = self%n_inputs*self%n_centers
        allocate(theta(2*n))
        if (n > 0) then
            theta(:n) = reshape(self%centers, [n])
            theta(n + 1:) = reshape(self%log_scales, [n])
        end if
    end function radial_parameters

    subroutine radial_set_parameters(self, theta, status)
        class(radial_basis_impl_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n

        if (size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis radial: parameter shape is invalid")
            return
        end if
        n = self%n_inputs*self%n_centers
        self%centers = reshape(theta(:n), shape(self%centers))
        self%log_scales = reshape(theta(n + 1:), shape(self%log_scales))
        call status_set(status, FORTNUM_OK, "")
    end subroutine radial_set_parameters

    subroutine radial_evaluate(self, x, phi, status)
        class(radial_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, p, column
        real(dp) :: inverse_scale

        if (size(x, 2) /= self%n_inputs .or. &
            size(phi, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis radial: array shape is invalid")
            return
        end if
        phi = 0.0_dp
        column = 1
        do p = 1, self%n_centers
            do j = 1, self%n_inputs
                inverse_scale = exp(-self%log_scales(j, p))
                phi(:, column) = phi(:, column) + &
                    ((x(:, j) - self%centers(j, p))*inverse_scale)**2
            end do
            phi(:, column) = exp(-0.5_dp*phi(:, column))
            column = column + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine radial_evaluate

    subroutine radial_jvp(self, x, theta_dot, x_dot, phi, phi_dot, status)
        class(radial_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, p, column, center_index, n
        real(dp) :: q, q_dot, log_value_dot

        if (size(theta_dot) /= self%parameter_count() .or. &
            any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis radial: tangent shape is invalid")
            return
        end if
        call self%evaluate(x, phi, status)
        if (status%code /= FORTNUM_OK) return
        phi_dot = 0.0_dp
        column = 1
        n = self%n_inputs*self%n_centers
        do p = 1, self%n_centers
            do i = 1, size(x, 1)
                log_value_dot = 0.0_dp
                do j = 1, self%n_inputs
                    q = (x(i, j) - self%centers(j, p))* &
                        exp(-self%log_scales(j, p))
                    center_index = (p - 1)*self%n_inputs + j
                    q_dot = exp(-self%log_scales(j, p))* &
                        (x_dot(i, j) - theta_dot(center_index)) - &
                        q*theta_dot(n + center_index)
                    log_value_dot = log_value_dot - q*q_dot
                end do
                phi_dot(i, column) = phi(i, column)*log_value_dot
            end do
            column = column + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine radial_jvp

    subroutine radial_vjp(self, x, u, theta_bar, x_bar, status)
        class(radial_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, p, column, center_index, n
        real(dp) :: q, radial_value, inverse_scale

        if (size(u, 2) /= self%feature_count() .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis radial: cotangent shape is invalid")
            return
        end if
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        column = 1
        n = self%n_inputs*self%n_centers
        do p = 1, self%n_centers
            do i = 1, size(x, 1)
                radial_value = 0.0_dp
                do j = 1, self%n_inputs
                    q = (x(i, j) - self%centers(j, p))* &
                        exp(-self%log_scales(j, p))
                    radial_value = radial_value + q*q
                end do
                radial_value = exp(-0.5_dp*radial_value)
                do j = 1, self%n_inputs
                    q = (x(i, j) - self%centers(j, p))* &
                        exp(-self%log_scales(j, p))
                    inverse_scale = exp(-self%log_scales(j, p))
                    x_bar(i, j) = x_bar(i, j) - &
                        u(i, column)*radial_value*q*inverse_scale
                    center_index = (p - 1)*self%n_inputs + j
                    theta_bar(center_index) = theta_bar(center_index) + &
                        u(i, column)*radial_value*q*inverse_scale
                    theta_bar(n + center_index) = theta_bar(n + center_index) + &
                        u(i, column)*radial_value*q*q
                end do
            end do
            column = column + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine radial_vjp

    logical function radial_valid(self) result(valid)
        class(radial_basis_impl_t), intent(in) :: self
        valid = self%n_inputs > 0 .and. self%n_centers > 0
        if (.not. valid) return
        valid = allocated(self%centers)
        if (.not. valid) return
        valid = allocated(self%log_scales)
    end function radial_valid

    integer function spline_input_count(self) result(count)
        class(spline_basis_impl_t), intent(in) :: self
        count = self%n_inputs
    end function spline_input_count

    integer function spline_feature_count(self) result(count)
        class(spline_basis_impl_t), intent(in) :: self
        count = 0
        if (allocated(self%spline)) then
            if (size(self%spline) > 0) count = self%n_inputs*self%spline(1)%ncoef
        end if
    end function spline_feature_count

    integer function spline_parameter_count(self) result(count)
        class(spline_basis_impl_t), intent(in) :: self
        count = 0
    end function spline_parameter_count

    function spline_parameters(self) result(theta)
        class(spline_basis_impl_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        allocate(theta(0))
    end function spline_parameters

    subroutine spline_set_parameters(self, theta, status)
        class(spline_basis_impl_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(theta) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline: parameter shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine spline_set_parameters

    subroutine spline_evaluate(self, x, phi, status)
        class(spline_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, column, ncoef
        real(dp), allocatable :: values(:)

        if (size(x, 2) /= self%n_inputs .or. &
            size(phi, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline: array shape is invalid")
            return
        end if
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline: model is not initialized")
            return
        end if
        phi = 0.0_dp
        column = 1
        ncoef = self%spline(1)%ncoef
        allocate(values(ncoef))
        do j = 1, self%n_inputs
            do i = 1, size(x, 1)
                call bspline_eval_basis(self%spline(j), x(i, j), values, status)
                if (status%code /= FORTNUM_OK) return
                phi(i, column:column + ncoef - 1) = values
            end do
            column = column + ncoef
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine spline_evaluate

    subroutine spline_jvp(self, x, theta_dot, x_dot, phi, phi_dot, status)
        class(spline_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, column, ncoef
        real(dp), allocatable :: dvalues(:, :)

        if (size(theta_dot) /= 0 .or. any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline: tangent shape is invalid")
            return
        end if
        call self%evaluate(x, phi, status)
        if (status%code /= FORTNUM_OK) return
        phi_dot = 0.0_dp
        column = 1
        ncoef = self%spline(1)%ncoef
        allocate(dvalues(0:1, ncoef))
        do j = 1, self%n_inputs
            do i = 1, size(x, 1)
                call bspline_eval_deriv(self%spline(j), x(i, j), 1, dvalues, &
                    status)
                if (status%code /= FORTNUM_OK) return
                phi_dot(i, column:column + ncoef - 1) = &
                    dvalues(1, :)*x_dot(i, j)
            end do
            column = column + ncoef
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine spline_jvp

    subroutine spline_vjp(self, x, u, theta_bar, x_bar, status)
        class(spline_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, column, ncoef
        real(dp), allocatable :: dvalues(:, :)

        if (size(u, 2) /= self%feature_count() .or. &
            size(theta_bar) /= 0 .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline: cotangent shape is invalid")
            return
        end if
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline: model is not initialized")
            return
        end if
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        column = 1
        ncoef = self%spline(1)%ncoef
        allocate(dvalues(0:1, ncoef))
        do j = 1, self%n_inputs
            do i = 1, size(x, 1)
                call bspline_eval_deriv(self%spline(j), x(i, j), 1, dvalues, &
                    status)
                if (status%code /= FORTNUM_OK) return
                x_bar(i, j) = sum(u(i, column:column + ncoef - 1)* &
                    dvalues(1, :))
            end do
            column = column + ncoef
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine spline_vjp

    logical function spline_valid(self) result(valid)
        class(spline_basis_impl_t), intent(in) :: self
        valid = self%n_inputs > 0
        if (.not. valid) return
        valid = allocated(self%spline)
        if (.not. valid) return
        valid = size(self%spline) > 0
    end function spline_valid

    integer function callback_input_count(self) result(count)
        class(callback_basis_impl_t), intent(in) :: self
        count = self%n_inputs
    end function callback_input_count

    integer function callback_feature_count(self) result(count)
        class(callback_basis_impl_t), intent(in) :: self
        count = self%n_features
    end function callback_feature_count

    integer function callback_parameter_count(self) result(count)
        class(callback_basis_impl_t), intent(in) :: self
        count = 0
        if (allocated(self%parameters_value)) count = size(self%parameters_value)
    end function callback_parameter_count

    function callback_parameters(self) result(theta)
        class(callback_basis_impl_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        allocate(theta(self%parameter_count()))
        if (size(theta) > 0) theta = self%parameters_value
    end function callback_parameters

    subroutine callback_set_parameters(self, theta, status)
        class(callback_basis_impl_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis callback: parameter shape is invalid")
            return
        end if
        self%parameters_value = theta
        call status_set(status, FORTNUM_OK, "")
    end subroutine callback_set_parameters

    subroutine callback_evaluate(self, x, phi, status)
        class(callback_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (size(x, 2) /= self%n_inputs .or. &
            size(phi, 2) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis callback: array shape is invalid")
            return
        end if
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis callback: model is not initialized")
            return
        end if
        call self%value_callback(x, self%parameters_value, phi, status)
    end subroutine callback_evaluate

    subroutine callback_jvp(self, x, theta_dot, x_dot, phi, phi_dot, status)
        class(callback_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (size(theta_dot) /= self%parameter_count() .or. &
            any(shape(x_dot) /= shape(x)) .or. size(phi, 2) /= self%n_features &
            .or. any(shape(phi_dot) /= shape(phi))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis callback: tangent shape is invalid")
            return
        end if
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis callback: model is not initialized")
            return
        end if
        call self%jvp_callback(x, self%parameters_value, x_dot, theta_dot, phi, &
            phi_dot, status)
    end subroutine callback_jvp

    subroutine callback_vjp(self, x, u, theta_bar, x_bar, status)
        class(callback_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (size(u, 2) /= self%n_features .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis callback: cotangent shape is invalid")
            return
        end if
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis callback: model is not initialized")
            return
        end if
        call self%vjp_callback(x, self%parameters_value, u, theta_bar, x_bar, &
            status)
    end subroutine callback_vjp

    logical function callback_valid(self) result(valid)
        class(callback_basis_impl_t), intent(in) :: self
        valid = self%n_inputs > 0 .and. self%n_features > 0
        if (.not. valid) return
        valid = allocated(self%parameters_value)
        if (.not. valid) return
        valid = associated(self%value_callback)
        if (.not. valid) return
        valid = associated(self%jvp_callback)
        if (.not. valid) return
        valid = associated(self%vjp_callback)
    end function callback_valid

    logical function callback_static_eligible(self) result(eligible)
        class(callback_basis_impl_t), intent(in) :: self
        eligible = .false.
    end function callback_static_eligible

end module fortml_basis_impl
