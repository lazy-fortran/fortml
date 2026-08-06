module fortml_basis
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_bspline, only: bspline_workspace_t, bspline_init, &
        bspline_set_knots, bspline_eval_basis, bspline_eval_deriv
    implicit none
    private

    integer, parameter, public :: BASIS_POLYNOMIAL = 1
    integer, parameter, public :: BASIS_FOURIER = 2
    integer, parameter, public :: BASIS_RADIAL = 3
    integer, parameter, public :: BASIS_SPLINE = 4

    type, public :: basis_map_t
        integer :: kind = 0
        integer :: n_inputs = 0
        integer :: degree = 0
        integer :: n_harmonics = 0
        integer :: n_centers = 0
        logical :: include_intercept = .false.
        real(dp), allocatable :: centers(:, :)
        real(dp), allocatable :: log_scales(:, :)
        real(dp), allocatable :: log_frequencies(:, :)
        type(bspline_workspace_t), allocatable :: spline(:)
    contains
        procedure, public :: initialize_polynomial => basis_initialize_polynomial
        procedure, public :: initialize_fourier => basis_initialize_fourier
        procedure, public :: initialize_radial => basis_initialize_radial
        procedure, public :: initialize_spline => basis_initialize_spline
        procedure, public :: feature_count => basis_feature_count
        procedure, public :: parameter_count => basis_parameter_count
        procedure, public :: parameters => basis_parameters
        procedure, public :: set_parameters => basis_set_parameters
        procedure, public :: evaluate => basis_evaluate
        procedure, public :: jvp => basis_jvp
        procedure, public :: vjp => basis_vjp
    end type basis_map_t

    public :: make_polynomial_basis
    public :: make_fourier_basis
    public :: make_radial_basis
    public :: make_spline_basis

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

        self%kind = 0
        self%n_inputs = 0
        self%degree = 0
        self%n_harmonics = 0
        self%n_centers = 0
        self%include_intercept = .false.
        if (allocated(self%log_frequencies)) deallocate (self%log_frequencies)
        if (allocated(self%centers)) deallocate (self%centers)
        if (allocated(self%log_scales)) deallocate (self%log_scales)
        if (allocated(self%spline)) deallocate (self%spline)
        if (present(include_intercept)) self%include_intercept = include_intercept
        if (n_inputs < 1 .or. degree < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: n_inputs and degree must be positive")
            return
        end if
        self%kind = BASIS_POLYNOMIAL
        self%n_inputs = n_inputs
        self%degree = degree
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_initialize_polynomial

    subroutine basis_initialize_fourier(self, n_inputs, frequencies, status, &
            include_intercept)
        class(basis_map_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        real(dp), intent(in) :: frequencies(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept

        self%kind = 0
        self%n_inputs = 0
        self%degree = 0
        self%n_harmonics = 0
        self%n_centers = 0
        self%include_intercept = .false.
        if (allocated(self%log_frequencies)) deallocate (self%log_frequencies)
        if (allocated(self%centers)) deallocate (self%centers)
        if (allocated(self%log_scales)) deallocate (self%log_scales)
        if (allocated(self%spline)) deallocate (self%spline)
        if (present(include_intercept)) self%include_intercept = include_intercept
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
        self%kind = BASIS_FOURIER
        self%n_inputs = n_inputs
        self%n_harmonics = size(frequencies, 1)
        allocate(self%log_frequencies(self%n_harmonics, n_inputs))
        self%log_frequencies = log(frequencies)
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_initialize_fourier

    subroutine basis_initialize_radial(self, n_inputs, centers, scales, status, &
            include_intercept)
        class(basis_map_t), intent(out) :: self
        integer, intent(in) :: n_inputs
        real(dp), intent(in) :: centers(:, :), scales(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept

        self%kind = 0
        self%n_inputs = 0
        self%degree = 0
        self%n_harmonics = 0
        self%n_centers = 0
        self%include_intercept = .false.
        if (allocated(self%log_frequencies)) deallocate (self%log_frequencies)
        if (allocated(self%centers)) deallocate (self%centers)
        if (allocated(self%log_scales)) deallocate (self%log_scales)
        if (allocated(self%spline)) deallocate (self%spline)
        if (present(include_intercept)) self%include_intercept = include_intercept
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
        self%kind = BASIS_RADIAL
        self%n_inputs = n_inputs
        self%n_centers = size(centers, 2)
        allocate(self%centers(n_inputs, self%n_centers))
        allocate(self%log_scales(n_inputs, self%n_centers))
        self%centers = centers
        self%log_scales = log(scales)
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_initialize_radial

    subroutine basis_initialize_spline(self, n_inputs, order, breakpoints, status, &
            include_intercept)
        class(basis_map_t), intent(out) :: self
        integer, intent(in) :: n_inputs, order
        real(dp), intent(in) :: breakpoints(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: include_intercept
        integer :: j

        self%kind = 0
        self%n_inputs = 0
        self%degree = 0
        self%n_harmonics = 0
        self%n_centers = 0
        self%include_intercept = .false.
        if (allocated(self%log_frequencies)) deallocate (self%log_frequencies)
        if (allocated(self%centers)) deallocate (self%centers)
        if (allocated(self%log_scales)) deallocate (self%log_scales)
        if (allocated(self%spline)) deallocate (self%spline)
        if (present(include_intercept)) self%include_intercept = include_intercept
        if (n_inputs < 1 .or. size(breakpoints, 2) /= n_inputs .or. &
            size(breakpoints, 1) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline: breakpoint shape is invalid")
            return
        end if
        allocate(self%spline(n_inputs))
        do j = 1, n_inputs
            call bspline_init(self%spline(j), order, size(breakpoints, 1), status)
            if (status%code /= FORTNUM_OK) return
            call bspline_set_knots(self%spline(j), breakpoints(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        self%kind = BASIS_SPLINE
        self%n_inputs = n_inputs
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_initialize_spline

    integer function basis_feature_count(self) result(count)
        class(basis_map_t), intent(in) :: self

        count = 0
        if (self%kind == BASIS_POLYNOMIAL) then
            count = self%n_inputs*self%degree
        else if (self%kind == BASIS_FOURIER) then
            count = 2*self%n_inputs*self%n_harmonics
        else if (self%kind == BASIS_RADIAL) then
            count = self%n_centers
        else if (self%kind == BASIS_SPLINE) then
            if (allocated(self%spline)) count = self%n_inputs*self%spline(1)%ncoef
        end if
        if (self%include_intercept) count = count + 1
    end function basis_feature_count

    integer function basis_parameter_count(self) result(count)
        class(basis_map_t), intent(in) :: self

        count = 0
        if (self%kind == BASIS_FOURIER) then
            count = self%n_inputs*self%n_harmonics
        else if (self%kind == BASIS_RADIAL) then
            count = 2*self%n_inputs*self%n_centers
        end if
    end function basis_parameter_count

    function basis_parameters(self) result(theta)
        class(basis_map_t), intent(in) :: self
        real(dp), allocatable :: theta(:)

        allocate(theta(self%parameter_count()))
        if (self%parameter_count() > 0) then
            if (self%kind == BASIS_FOURIER) then
                theta = reshape(self%log_frequencies, [self%parameter_count()])
            else if (self%kind == BASIS_RADIAL) then
                theta(1:self%n_inputs*self%n_centers) = &
                    reshape(self%centers, [self%n_inputs*self%n_centers])
                theta(self%n_inputs*self%n_centers + 1:) = &
                    reshape(self%log_scales, [self%n_inputs*self%n_centers])
            end if
        end if
    end function basis_parameters

    subroutine basis_set_parameters(self, theta, status)
        class(basis_map_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. valid_model(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis set_parameters: model is not initialized")
            return
        end if
        if (size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis set_parameters: parameter shape is invalid")
            return
        end if
        if (self%parameter_count() > 0) then
            if (self%kind == BASIS_FOURIER) then
                self%log_frequencies = reshape(theta, shape(self%log_frequencies))
            else if (self%kind == BASIS_RADIAL) then
                self%centers = reshape(theta(1:self%n_inputs*self%n_centers), &
                    shape(self%centers))
                self%log_scales = reshape(theta(self%n_inputs*self%n_centers + 1:), &
                    shape(self%log_scales))
            end if
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_set_parameters

    subroutine basis_evaluate(self, x, phi, status)
        class(basis_map_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, p, h, column, ncoef
        real(dp) :: frequency, inverse_scale
        real(dp), allocatable :: values(:)

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
        if (self%kind == BASIS_POLYNOMIAL) then
            do j = 1, self%n_inputs
                do p = 1, self%degree
                    phi(:, column) = x(:, j)**p
                    column = column + 1
                end do
            end do
        else if (self%kind == BASIS_FOURIER) then
            do j = 1, self%n_inputs
                do h = 1, self%n_harmonics
                    frequency = exp(self%log_frequencies(h, j))
                    phi(:, column) = sin(frequency*x(:, j))
                    phi(:, column + 1) = cos(frequency*x(:, j))
                    column = column + 2
                end do
            end do
        else if (self%kind == BASIS_RADIAL) then
            do p = 1, self%n_centers
                do j = 1, self%n_inputs
                    inverse_scale = exp(-self%log_scales(j, p))
                    phi(:, column) = phi(:, column) + &
                        ((x(:, j) - self%centers(j, p))*inverse_scale)**2
                end do
                phi(:, column) = exp(-0.5_dp*phi(:, column))
                column = column + 1
            end do
        else if (self%kind == BASIS_SPLINE) then
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
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_evaluate

    subroutine basis_jvp(self, x, theta_dot, x_dot, phi, phi_dot, status)
        class(basis_map_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, p, h, column, center_index, parameter_index, ncoef
        real(dp) :: frequency, q, q_dot, log_value_dot
        real(dp) :: argument(size(x, 1)), argument_dot(size(x, 1))
        real(dp), allocatable :: dvalues(:, :)

        if (.not. valid_shapes(self, x, phi)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis jvp: model or output shape is invalid")
            return
        end if
        if (any(shape(x_dot) /= shape(x)) .or. &
            size(theta_dot) /= self%parameter_count() .or. &
            any(shape(phi_dot) /= shape(phi))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis jvp: tangent shape is invalid")
            return
        end if
        call self%evaluate(x, phi, status)
        if (status%code /= FORTNUM_OK) return
        phi_dot = 0.0_dp
        column = 1
        if (self%include_intercept) column = column + 1
        if (self%kind == BASIS_POLYNOMIAL) then
            do j = 1, self%n_inputs
                do p = 1, self%degree
                    phi_dot(:, column) = real(p, dp)*x(:, j)**(p - 1)*x_dot(:, j)
                    column = column + 1
                end do
            end do
        else if (self%kind == BASIS_FOURIER) then
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
        else if (self%kind == BASIS_RADIAL) then
            do p = 1, self%n_centers
                do i = 1, size(x, 1)
                    log_value_dot = 0.0_dp
                    do j = 1, self%n_inputs
                        q = (x(i, j) - self%centers(j, p))* &
                            exp(-self%log_scales(j, p))
                        center_index = (p - 1)*self%n_inputs + j
                        q_dot = exp(-self%log_scales(j, p))* &
                            (x_dot(i, j) - theta_dot(center_index)) - &
                            q*theta_dot(self%n_inputs*self%n_centers + center_index)
                        log_value_dot = log_value_dot - q*q_dot
                    end do
                    phi_dot(i, column) = phi(i, column)*log_value_dot
                end do
                column = column + 1
            end do
        else if (self%kind == BASIS_SPLINE) then
            ncoef = self%spline(1)%ncoef
            allocate(dvalues(0:1, ncoef))
            do j = 1, self%n_inputs
                do i = 1, size(x, 1)
                    call bspline_eval_deriv(self%spline(j), x(i, j), 1, &
                        dvalues, status)
                    if (status%code /= FORTNUM_OK) return
                    phi_dot(i, column:column + ncoef - 1) = &
                        dvalues(1, :)*x_dot(i, j)
                end do
                column = column + ncoef
            end do
        end if
    end subroutine basis_jvp

    subroutine basis_vjp(self, x, u, theta_bar, x_bar, status)
        class(basis_map_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, p, h, column, center_index, parameter_index, ncoef
        real(dp) :: frequency, q, radial_value, inverse_scale
        real(dp) :: argument(size(x, 1)), z_bar(size(x, 1))
        real(dp), allocatable :: dvalues(:, :)

        if (.not. valid_shapes(self, x, u)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis vjp: model or array shape is invalid")
            return
        end if
        if (size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis vjp: cotangent shape is invalid")
            return
        end if
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        column = 1
        if (self%include_intercept) column = column + 1
        if (self%kind == BASIS_POLYNOMIAL) then
            do j = 1, self%n_inputs
                do p = 1, self%degree
                    x_bar(:, j) = x_bar(:, j) + &
                        u(:, column)*real(p, dp)*x(:, j)**(p - 1)
                    column = column + 1
                end do
            end do
        else if (self%kind == BASIS_FOURIER) then
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
        else if (self%kind == BASIS_RADIAL) then
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
                        theta_bar(self%n_inputs*self%n_centers + center_index) = &
                            theta_bar(self%n_inputs*self%n_centers + center_index) + &
                            u(i, column)*radial_value*q*q
                    end do
                end do
                column = column + 1
            end do
        else if (self%kind == BASIS_SPLINE) then
            ncoef = self%spline(1)%ncoef
            allocate(dvalues(0:1, ncoef))
            do j = 1, self%n_inputs
                do i = 1, size(x, 1)
                    call bspline_eval_deriv(self%spline(j), x(i, j), 1, &
                        dvalues, status)
                    if (status%code /= FORTNUM_OK) return
                    x_bar(i, j) = sum(u(i, column:column + ncoef - 1)* &
                        dvalues(1, :))
                end do
                column = column + ncoef
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_vjp

    logical function valid_model(self) result(valid)
        class(basis_map_t), intent(in) :: self

        valid = self%kind == BASIS_POLYNOMIAL .or. self%kind == BASIS_FOURIER .or. &
            self%kind == BASIS_RADIAL .or. self%kind == BASIS_SPLINE
        if (.not. valid) return
        valid = self%n_inputs > 0
        if (.not. valid) return
        if (self%kind == BASIS_POLYNOMIAL) then
            valid = self%degree > 0
        else if (self%kind == BASIS_FOURIER) then
            valid = self%n_harmonics > 0
            if (.not. valid) return
            valid = allocated(self%log_frequencies)
        else if (self%kind == BASIS_RADIAL) then
            valid = self%n_centers > 0
            if (.not. valid) return
            valid = allocated(self%centers) .and. allocated(self%log_scales)
        else
            valid = allocated(self%spline)
            if (.not. valid) return
            valid = size(self%spline) > 0
        end if
    end function valid_model

    logical function valid_shapes(self, x, output) result(valid)
        class(basis_map_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output(:, :)

        valid = valid_model(self)
        if (.not. valid) return
        valid = size(x, 1) > 0 .and. size(x, 2) == self%n_inputs
        if (.not. valid) return
        valid = size(output, 1) == size(x, 1) .and. &
            size(output, 2) == self%feature_count()
    end function valid_shapes

end module fortml_basis
