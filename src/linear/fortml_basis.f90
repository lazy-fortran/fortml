module fortml_basis
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_basis_impl, only: basis_impl_t, basis_value_callback, &
        basis_jvp_callback, basis_vjp_callback, create_polynomial_impl, &
        create_fourier_impl, create_radial_impl, create_spline_impl, &
        create_callback_impl
    implicit none
    private

    integer, parameter, public :: BASIS_POLYNOMIAL = 1
    integer, parameter, public :: BASIS_FOURIER = 2
    integer, parameter, public :: BASIS_RADIAL = 3
    integer, parameter, public :: BASIS_SPLINE = 4
    integer, parameter, public :: BASIS_CALLBACK = 5

    type, public :: basis_map_t
        private
        logical :: include_intercept = .false.
        class(basis_impl_t), allocatable :: implementation
    contains
        procedure, public :: initialize_polynomial => basis_initialize_polynomial
        procedure, public :: initialize_fourier => basis_initialize_fourier
        procedure, public :: initialize_radial => basis_initialize_radial
        procedure, public :: initialize_spline => basis_initialize_spline
        procedure, public :: initialize_callback => basis_initialize_callback
        procedure, public :: input_count => basis_input_count
        procedure, public :: feature_count => basis_feature_count
        procedure, public :: parameter_count => basis_parameter_count
        procedure, public :: parameters => basis_parameters
        procedure, public :: set_parameters => basis_set_parameters
        procedure, public :: evaluate => basis_evaluate
        procedure, public :: jvp => basis_jvp
        procedure, public :: vjp => basis_vjp
        procedure, public :: valid => basis_valid
        procedure, public :: static_lowering_eligible => &
            basis_static_lowering_eligible
    end type basis_map_t

    public :: make_polynomial_basis
    public :: make_fourier_basis
    public :: make_radial_basis
    public :: make_spline_basis
    public :: basis_value_callback, basis_jvp_callback, basis_vjp_callback

contains

    function make_polynomial_basis(n_inputs, degree, status, include_intercept) &
            result(map)
        integer, intent(in) :: n_inputs, degree
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept
        type(basis_map_t) :: map

        call map%initialize_polynomial(n_inputs, degree, status, include_intercept)
    end function make_polynomial_basis

    function make_fourier_basis(n_inputs, frequencies, status, include_intercept) &
            result(map)
        integer, intent(in) :: n_inputs
        real(dp), intent(in) :: frequencies(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept
        type(basis_map_t) :: map

        call map%initialize_fourier(n_inputs, frequencies, status, &
            include_intercept)
    end function make_fourier_basis

    function make_radial_basis(n_inputs, centers, scales, status, include_intercept) &
            result(map)
        integer, intent(in) :: n_inputs
        real(dp), intent(in) :: centers(:, :), scales(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept
        type(basis_map_t) :: map

        call map%initialize_radial(n_inputs, centers, scales, status, &
            include_intercept)
    end function make_radial_basis

    function make_spline_basis(n_inputs, order, breakpoints, status, &
            include_intercept) result(map)
        integer, intent(in) :: n_inputs, order
        real(dp), intent(in) :: breakpoints(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept
        type(basis_map_t) :: map

        call map%initialize_spline(n_inputs, order, breakpoints, status, &
            include_intercept)
    end function make_spline_basis

    subroutine basis_initialize_polynomial(self, n_inputs, degree, status, &
            include_intercept)
        class(basis_map_t), intent(out) :: self
        integer, intent(in) :: n_inputs, degree
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept
        class(basis_impl_t), allocatable :: implementation

        self%include_intercept = .false.
        if (present(include_intercept)) self%include_intercept = include_intercept
        call create_polynomial_impl(n_inputs, degree, implementation, status)
        if (status%code /= FORTNUM_OK) return
        call move_alloc(implementation, self%implementation)
    end subroutine basis_initialize_polynomial

    subroutine basis_initialize_fourier(self, n_inputs, frequencies, status, &
            include_intercept)
        class(basis_map_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        real(dp), intent(in) :: frequencies(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept
        class(basis_impl_t), allocatable :: implementation

        self%include_intercept = .false.
        if (present(include_intercept)) self%include_intercept = include_intercept
        call create_fourier_impl(n_inputs, frequencies, implementation, status)
        if (status%code /= FORTNUM_OK) return
        call move_alloc(implementation, self%implementation)
    end subroutine basis_initialize_fourier

    subroutine basis_initialize_radial(self, n_inputs, centers, scales, status, &
            include_intercept)
        class(basis_map_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        real(dp), intent(in) :: centers(:, :), scales(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept
        class(basis_impl_t), allocatable :: implementation

        self%include_intercept = .false.
        if (present(include_intercept)) self%include_intercept = include_intercept
        call create_radial_impl(n_inputs, centers, scales, implementation, status)
        if (status%code /= FORTNUM_OK) return
        call move_alloc(implementation, self%implementation)
    end subroutine basis_initialize_radial

    subroutine basis_initialize_spline(self, n_inputs, order, breakpoints, status, &
            include_intercept)
        class(basis_map_t), intent(out) :: self
        integer, intent(in) :: n_inputs, order
        real(dp), intent(in) :: breakpoints(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept
        class(basis_impl_t), allocatable :: implementation

        self%include_intercept = .false.
        if (present(include_intercept)) self%include_intercept = include_intercept
        call create_spline_impl(n_inputs, order, breakpoints, implementation, &
            status)
        if (status%code /= FORTNUM_OK) return
        call move_alloc(implementation, self%implementation)
    end subroutine basis_initialize_spline

    subroutine basis_initialize_callback(self, n_inputs, n_features, parameters, &
            value, jvp, vjp, status)
        class(basis_map_t), intent(out) :: self
        integer, intent(in) :: n_inputs, n_features
        real(dp), intent(in) :: parameters(:)
        procedure(basis_value_callback) :: value
        procedure(basis_jvp_callback) :: jvp
        procedure(basis_vjp_callback) :: vjp
        type(fortnum_status_t), intent(out) :: status
        class(basis_impl_t), allocatable :: implementation

        self%include_intercept = .false.
        call create_callback_impl(n_inputs, n_features, parameters, value, jvp, &
            vjp, implementation, status)
        if (status%code /= FORTNUM_OK) return
        call move_alloc(implementation, self%implementation)
    end subroutine basis_initialize_callback

    integer function basis_feature_count(self) result(count)
        class(basis_map_t), intent(in) :: self

        count = 0
        if (.not. allocated(self%implementation)) return
        count = self%implementation%feature_count()
        if (self%include_intercept) count = count + 1
    end function basis_feature_count

    integer function basis_input_count(self) result(count)
        class(basis_map_t), intent(in) :: self

        count = 0
        if (.not. allocated(self%implementation)) return
        count = self%implementation%input_count()
    end function basis_input_count

    integer function basis_parameter_count(self) result(count)
        class(basis_map_t), intent(in) :: self

        count = 0
        if (.not. allocated(self%implementation)) return
        count = self%implementation%parameter_count()
    end function basis_parameter_count

    function basis_parameters(self) result(theta)
        class(basis_map_t), intent(in) :: self
        real(dp), allocatable :: theta(:)

        if (allocated(self%implementation)) then
            theta = self%implementation%parameters()
        else
            allocate(theta(0))
        end if
    end function basis_parameters

    subroutine basis_set_parameters(self, theta, status)
        class(basis_map_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. basis_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis set_parameters: model is not initialized")
            return
        end if
        if (size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis set_parameters: parameter shape is invalid")
            return
        end if
        call self%implementation%set_parameters(theta, status)
    end subroutine basis_set_parameters

    subroutine basis_evaluate(self, x, phi, status)
        class(basis_map_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: column

        if (.not. valid_shapes(self, x, phi)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis evaluate: model or array shape is invalid")
            return
        end if
        phi = 0.0_dp
        column = 1
        if (self%include_intercept) then
            phi(:, column) = 1.0_dp
            column = column + 1
        end if
        call self%implementation%evaluate(x, phi(:, column:), status)
    end subroutine basis_evaluate

    subroutine basis_jvp(self, x, theta_dot, x_dot, phi, phi_dot, status)
        class(basis_map_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: column

        if (.not. valid_shapes(self, x, phi)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis jvp: model or output shape is invalid")
            return
        end if
        if (any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis jvp: input tangent shape is invalid")
            return
        end if
        if (size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis jvp: parameter tangent shape is invalid")
            return
        end if
        if (any(shape(phi_dot) /= shape(phi))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis jvp: output tangent shape is invalid")
            return
        end if
        phi = 0.0_dp
        phi_dot = 0.0_dp
        column = 1
        if (self%include_intercept) column = column + 1
        call self%implementation%jvp(x, theta_dot, x_dot, phi(:, column:), &
            phi_dot(:, column:), status)
    end subroutine basis_jvp

    subroutine basis_vjp(self, x, u, theta_bar, x_bar, status)
        class(basis_map_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: column

        if (.not. valid_shapes(self, x, u)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis vjp: model or array shape is invalid")
            return
        end if
        if (size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis vjp: parameter cotangent shape is invalid")
            return
        end if
        if (any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis vjp: input cotangent shape is invalid")
            return
        end if
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        column = 1
        if (self%include_intercept) column = column + 1
        call self%implementation%vjp(x, u(:, column:), theta_bar, x_bar, status)
    end subroutine basis_vjp

    logical function basis_static_lowering_eligible(self) result(eligible)
        class(basis_map_t), intent(in) :: self

        eligible = .false.
        if (.not. basis_valid(self)) return
        eligible = self%implementation%static_lowering_eligible()
    end function basis_static_lowering_eligible

    logical function basis_valid(self) result(valid)
        class(basis_map_t), intent(in) :: self

        valid = allocated(self%implementation)
        if (.not. valid) return
        valid = self%implementation%valid()
    end function basis_valid

    logical function valid_shapes(self, x, output) result(valid)
        class(basis_map_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output(:, :)

        valid = basis_valid(self)
        if (.not. valid) return
        valid = size(x, 1) > 0
        if (.not. valid) return
        valid = size(x, 2) == self%implementation%input_count()
        if (.not. valid) return
        valid = size(output, 1) == size(x, 1)
        if (.not. valid) return
        valid = size(output, 2) == self%feature_count()
    end function valid_shapes

end module fortml_basis
