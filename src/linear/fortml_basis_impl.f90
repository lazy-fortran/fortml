module fortml_basis_impl
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
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
        ! HVP is a scalar-contraction Hessian-vector product.  Implementations
        ! that have a smooth analytic second derivative override the default;
        ! callbacks retain an explicit typed refusal instead of silently using
        ! finite differences.
        procedure :: hvp => basis_impl_hvp
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
        logical :: interactions = .false.
        integer, allocatable :: exponents(:, :)
    contains
        procedure :: input_count => polynomial_input_count
        procedure :: feature_count => polynomial_feature_count
        procedure :: parameter_count => polynomial_parameter_count
        procedure :: parameters => polynomial_parameters
        procedure :: set_parameters => polynomial_set_parameters
        procedure :: evaluate => polynomial_evaluate
        procedure :: jvp => polynomial_jvp
        procedure :: vjp => polynomial_vjp
        procedure :: hvp => polynomial_hvp
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
        procedure :: hvp => fourier_hvp
        procedure :: valid => fourier_valid
    end type fourier_basis_impl_t

    ! Fixed random Fourier features approximate a stationary kernel with
    ! ``sqrt(2/m)*cos(w_k dot x + b_k)``.  The frequencies and phases are
    ! intentionally part of the fitted state, not trainable parameters: this
    ! keeps the map a deterministic feature transform while retaining exact
    ! input JVP/VJP/HVP products.  A caller can sample ``frequencies`` and
    ! ``phases`` with any reproducible RNG before construction.
    type, extends(basis_impl_t) :: random_fourier_basis_impl_t
        integer :: n_inputs = 0
        integer :: n_components = 0
        real(dp), allocatable :: frequencies(:, :)
        real(dp), allocatable :: phases(:)
        real(dp) :: normalization = 0.0_dp
    contains
        procedure :: input_count => random_fourier_input_count
        procedure :: feature_count => random_fourier_feature_count
        procedure :: parameter_count => random_fourier_parameter_count
        procedure :: parameters => random_fourier_parameters
        procedure :: set_parameters => random_fourier_set_parameters
        procedure :: evaluate => random_fourier_evaluate
        procedure :: jvp => random_fourier_jvp
        procedure :: vjp => random_fourier_vjp
        procedure :: hvp => random_fourier_hvp
        procedure :: valid => random_fourier_valid
    end type random_fourier_basis_impl_t

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
        procedure :: hvp => radial_hvp
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
        procedure :: hvp => spline_hvp
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
    public :: create_polynomial_impl, create_fourier_impl, &
        create_random_fourier_impl, create_radial_impl
    public :: create_spline_impl, create_callback_impl

contains

    subroutine basis_impl_hvp(self, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        !! Default second-order product for basis implementations.
        !!
        !! `hvp` differentiates the VJP of the scalar contraction
        !! `sum(u*phi)` in the joint direction `(theta_dot, x_dot)`.  A
        !! callback cannot provide this product through the first-order
        !! callback ABI, so it fails explicitly rather than introducing a
        !! noisy finite-difference fallback.
        class(basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (size(theta_dot) /= self%parameter_count() .or. &
                size(theta_hvp) /= self%parameter_count() .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis hvp: direction or output shape is invalid")
            return
        end if
        theta_hvp = 0.0_dp
        x_hvp = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "basis hvp: second derivatives are unavailable for this map")
    end subroutine basis_impl_hvp

    subroutine create_polynomial_impl(n_inputs, degree, impl, status, interactions)
        integer, intent(in) :: n_inputs, degree
        class(basis_impl_t), allocatable, intent(out) :: impl
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: interactions
        type(polynomial_basis_impl_t) :: value

        if (n_inputs < 1 .or. degree < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: n_inputs and degree must be positive")
            return
        end if
        value%n_inputs = n_inputs
        value%degree = degree
        value%interactions = .false.
        if (present(interactions)) value%interactions = interactions
        if (value%interactions) call build_polynomial_exponents(value, status)
        if (status%code /= FORTNUM_OK) return
        allocate(impl, source=value)
        call status_set(status, FORTNUM_OK, "")
    end subroutine create_polynomial_impl

    subroutine build_polynomial_exponents(self, status)
        type(polynomial_basis_impl_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer :: d, n_terms, next, prefix(self%n_inputs)

        n_terms = 0
        do d = 1, self%degree
            n_terms = n_terms + polynomial_compositions(self%n_inputs, d)
        end do
        if (n_terms < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial interactions: term count overflow")
            return
        end if
        allocate(self%exponents(n_terms, self%n_inputs))
        self%exponents = 0
        prefix = 0
        next = 0
        do d = 1, self%degree
            call enumerate_polynomial_degree(self%exponents, next, 1, d, &
                self%n_inputs, prefix)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_polynomial_exponents

    integer function polynomial_compositions(n_inputs, degree) result(count)
        integer, intent(in) :: n_inputs, degree
        integer :: i, numerator, denominator

        count = 1
        do i = 1, degree
            numerator = n_inputs + i - 1
            denominator = i
            count = count*numerator/denominator
        end do
    end function polynomial_compositions

    recursive subroutine enumerate_polynomial_degree(exponents, next, position, &
            remaining, n_inputs, prefix)
        integer, intent(inout) :: exponents(:, :), next
        integer, intent(inout) :: prefix(:)
        integer, intent(in) :: position, remaining, n_inputs
        integer :: value

        if (position == n_inputs) then
            next = next + 1
            prefix(position) = remaining
            exponents(next, :) = prefix
            return
        end if
        do value = remaining, 0, -1
            prefix(position) = value
            call enumerate_polynomial_degree(exponents, next, position + 1, &
                remaining - value, n_inputs, prefix)
        end do
    end subroutine enumerate_polynomial_degree

    function monomial_partial(exponents, x, j) result(value)
        integer, intent(in) :: exponents(:), j
        real(dp), intent(in) :: x(:, :)
        real(dp) :: value(size(x, 1))
        integer :: k, exponent

        value = 1.0_dp
        if (exponents(j) == 0) then
            value = 0.0_dp
            return
        end if
        do k = 1, size(exponents)
            exponent = exponents(k)
            if (k == j) exponent = exponent - 1
            if (exponent < 0) then
                value = 0.0_dp
                return
            end if
            value = value*x(:, k)**exponent
        end do
    end function monomial_partial

    function monomial_partial_direction(exponents, x, x_dot, j) result(value)
        integer, intent(in) :: exponents(:), j
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp) :: value(size(x, 1)), term(size(x, 1))
        integer :: k, l, exponent

        value = 0.0_dp
        if (exponents(j) == 0) return
        do k = 1, size(exponents)
            exponent = exponents(k)
            if (k == j) exponent = exponent - 1
            if (exponent <= 0) cycle
            term = 1.0_dp
            do l = 1, size(exponents)
                exponent = exponents(l)
                if (l == j) exponent = exponent - 1
                if (l == k) exponent = exponent - 1
                if (exponent < 0) then
                    term = 0.0_dp
                    exit
                end if
                term = term*x(:, l)**exponent
            end do
            value = value + real(exponents(k) - merge(1, 0, k == j), dp)* &
                term*x_dot(:, k)
        end do
    end function monomial_partial_direction

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

    subroutine create_random_fourier_impl(n_inputs, frequencies, phases, impl, &
            status)
        integer, intent(in) :: n_inputs
        real(dp), intent(in) :: frequencies(:, :), phases(:)
        class(basis_impl_t), allocatable, intent(out) :: impl
        type(fortnum_status_t), intent(out) :: status
        type(random_fourier_basis_impl_t) :: value

        if (n_inputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: n_inputs must be positive")
            return
        end if
        if (size(frequencies, 1) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: at least one component is required")
            return
        end if
        if (size(frequencies, 2) /= n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: frequency shape is invalid")
            return
        end if
        if (size(phases) /= size(frequencies, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: phase shape is invalid")
            return
        end if
        value%n_inputs = n_inputs
        value%n_components = size(frequencies, 1)
        allocate(value%frequencies(value%n_components, n_inputs))
        allocate(value%phases(value%n_components))
        value%frequencies = frequencies
        value%phases = phases
        value%normalization = sqrt(2.0_dp/real(value%n_components, dp))
        allocate(impl, source=value)
        call status_set(status, FORTNUM_OK, "")
    end subroutine create_random_fourier_impl

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
        if (self%interactions) then
            if (allocated(self%exponents)) then
                count = size(self%exponents, 1)
            else
                count = 0
            end if
        else
            count = self%n_inputs*self%degree
        end if
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
        integer :: i, j, p, column

        if (size(x, 2) /= self%n_inputs .or. &
            size(phi, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: array shape is invalid")
            return
        end if
        phi = 0.0_dp
        if (self%interactions) then
            do i = 1, size(self%exponents, 1)
                phi(:, i) = 1.0_dp
                do j = 1, self%n_inputs
                    phi(:, i) = phi(:, i)*x(:, j)**self%exponents(i, j)
                end do
            end do
            call status_set(status, FORTNUM_OK, "")
            return
        end if
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
        integer :: i, j, p, column

        if (size(theta_dot) /= 0 .or. any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: tangent shape is invalid")
            return
        end if
        call self%evaluate(x, phi, status)
        if (status%code /= FORTNUM_OK) return
        phi_dot = 0.0_dp
        if (self%interactions) then
            do i = 1, size(self%exponents, 1)
                phi(:, i) = 1.0_dp
                phi_dot(:, i) = 0.0_dp
                do j = 1, self%n_inputs
                    phi(:, i) = phi(:, i)*x(:, j)**self%exponents(i, j)
                    phi_dot(:, i) = phi_dot(:, i) + &
                        real(self%exponents(i, j), dp)* &
                        monomial_partial(self%exponents(i, :), x, j)*x_dot(:, j)
                end do
            end do
            call status_set(status, FORTNUM_OK, "")
            return
        end if
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
        integer :: i, j, p, column

        if (size(u, 2) /= self%feature_count() .or. &
            size(theta_bar) /= 0 .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial: cotangent shape is invalid")
            return
        end if
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (self%interactions) then
            do i = 1, size(self%exponents, 1)
                do j = 1, self%n_inputs
                    x_bar(:, j) = x_bar(:, j) + u(:, i)* &
                        real(self%exponents(i, j), dp)* &
                        monomial_partial(self%exponents(i, :), x, j)
                end do
            end do
            call status_set(status, FORTNUM_OK, "")
            return
        end if
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

    subroutine polynomial_hvp(self, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        class(polynomial_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, p, column

        if (size(u, 2) /= self%feature_count() .or. &
                size(theta_dot) /= 0 .or. size(theta_hvp) /= 0 .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis polynomial hvp: array shape is invalid")
            return
        end if
        theta_hvp = 0.0_dp
        x_hvp = 0.0_dp
        if (self%interactions) then
            do i = 1, size(self%exponents, 1)
                do j = 1, self%n_inputs
                    x_hvp(:, j) = x_hvp(:, j) + u(:, i)* &
                        real(self%exponents(i, j), dp)* &
                        monomial_partial_direction(self%exponents(i, :), x, &
                        x_dot, j)
                end do
            end do
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        column = 1
        do j = 1, self%n_inputs
            do p = 1, self%degree
                if (p > 1) then
                    do i = 1, size(x, 1)
                        x_hvp(i, j) = x_hvp(i, j) + u(i, column)* &
                            real(p*(p - 1), dp)*x(i, j)**(p - 2)*x_dot(i, j)
                    end do
                end if
                column = column + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_hvp

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

    subroutine fourier_hvp(self, x, u, theta_dot, x_dot, theta_hvp, x_hvp, &
            status)
        class(fourier_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, h, column, parameter_index
        real(dp) :: frequency, frequency_dot
        real(dp) :: argument, argument_dot, z_bar, z_bar_dot
        real(dp) :: u_sine, u_cosine

        if (size(u, 2) /= self%feature_count() .or. &
                size(theta_dot) /= self%parameter_count() .or. &
                size(theta_hvp) /= self%parameter_count() .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis Fourier hvp: array shape is invalid")
            return
        end if
        theta_hvp = 0.0_dp
        x_hvp = 0.0_dp
        column = 1
        parameter_index = 1
        do j = 1, self%n_inputs
            do h = 1, self%n_harmonics
                frequency = exp(self%log_frequencies(h, j))
                frequency_dot = frequency*theta_dot(parameter_index)
                do i = 1, size(x, 1)
                    argument = frequency*x(i, j)
                    argument_dot = frequency*x_dot(i, j) + &
                        frequency_dot*x(i, j)
                    u_sine = u(i, column)
                    u_cosine = u(i, column + 1)
                    z_bar = u_sine*cos(argument) - u_cosine*sin(argument)
                    z_bar_dot = (-u_sine*sin(argument) - u_cosine*cos(argument))* &
                        argument_dot
                    x_hvp(i, j) = x_hvp(i, j) + &
                        frequency_dot*z_bar + frequency*z_bar_dot
                    theta_hvp(parameter_index) = theta_hvp(parameter_index) + &
                        argument_dot*z_bar + argument*z_bar_dot
                end do
                column = column + 2
                parameter_index = parameter_index + 1
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine fourier_hvp

    logical function fourier_valid(self) result(valid)
        class(fourier_basis_impl_t), intent(in) :: self
        valid = self%n_inputs > 0 .and. self%n_harmonics > 0
        if (.not. valid) return
        valid = allocated(self%log_frequencies)
    end function fourier_valid

    integer function random_fourier_input_count(self) result(count)
        class(random_fourier_basis_impl_t), intent(in) :: self
        count = self%n_inputs
    end function random_fourier_input_count

    integer function random_fourier_feature_count(self) result(count)
        class(random_fourier_basis_impl_t), intent(in) :: self
        count = self%n_components
    end function random_fourier_feature_count

    integer function random_fourier_parameter_count(self) result(count)
        class(random_fourier_basis_impl_t), intent(in) :: self
        count = 0
    end function random_fourier_parameter_count

    function random_fourier_parameters(self) result(theta)
        class(random_fourier_basis_impl_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        allocate(theta(0))
    end function random_fourier_parameters

    subroutine random_fourier_set_parameters(self, theta, status)
        class(random_fourier_basis_impl_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(theta) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: parameter shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_fourier_set_parameters

    subroutine random_fourier_evaluate(self, x, phi, status)
        class(random_fourier_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: phi(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: argument

        if (size(x, 2) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: input shape is invalid")
            return
        end if
        if (size(phi, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: output shape is invalid")
            return
        end if
        do k = 1, self%n_components
            do i = 1, size(x, 1)
                argument = self%phases(k)
                do j = 1, self%n_inputs
                    argument = argument + self%frequencies(k, j)*x(i, j)
                end do
                phi(i, k) = self%normalization*cos(argument)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_fourier_evaluate

    subroutine random_fourier_jvp(self, x, theta_dot, x_dot, phi, phi_dot, &
            status)
        class(random_fourier_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: phi(:, :), phi_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: argument, argument_dot

        if (size(theta_dot) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: parameter tangent shape is invalid")
            return
        end if
        if (any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: input tangent shape is invalid")
            return
        end if
        if (any(shape(phi_dot) /= shape(phi))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: output tangent shape is invalid")
            return
        end if
        call self%evaluate(x, phi, status)
        if (status%code /= FORTNUM_OK) return
        do k = 1, self%n_components
            do i = 1, size(x, 1)
                argument = self%phases(k)
                argument_dot = 0.0_dp
                do j = 1, self%n_inputs
                    argument = argument + self%frequencies(k, j)*x(i, j)
                    argument_dot = argument_dot + self%frequencies(k, j)* &
                        x_dot(i, j)
                end do
                phi_dot(i, k) = -self%normalization*sin(argument)*argument_dot
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_fourier_jvp

    subroutine random_fourier_vjp(self, x, u, theta_bar, x_bar, status)
        class(random_fourier_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: argument, coefficient

        if (size(u, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: cotangent shape is invalid")
            return
        end if
        if (size(theta_bar) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: parameter cotangent shape is invalid")
            return
        end if
        if (any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier: input cotangent shape is invalid")
            return
        end if
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        do k = 1, self%n_components
            do i = 1, size(x, 1)
                argument = self%phases(k)
                do j = 1, self%n_inputs
                    argument = argument + self%frequencies(k, j)*x(i, j)
                end do
                coefficient = -self%normalization*u(i, k)*sin(argument)
                do j = 1, self%n_inputs
                    x_bar(i, j) = x_bar(i, j) + coefficient* &
                        self%frequencies(k, j)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_fourier_vjp

    subroutine random_fourier_hvp(self, x, u, theta_dot, x_dot, theta_hvp, &
            x_hvp, status)
        class(random_fourier_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: argument, argument_dot, coefficient

        if (size(u, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier hvp: cotangent shape is invalid")
            return
        end if
        if (size(theta_dot) /= 0 .or. size(theta_hvp) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier hvp: parameter direction shape is invalid")
            return
        end if
        if (any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier hvp: input direction shape is invalid")
            return
        end if
        if (any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis random Fourier hvp: output shape is invalid")
            return
        end if
        x_hvp = 0.0_dp
        do k = 1, self%n_components
            do i = 1, size(x, 1)
                argument = self%phases(k)
                argument_dot = 0.0_dp
                do j = 1, self%n_inputs
                    argument = argument + self%frequencies(k, j)*x(i, j)
                    argument_dot = argument_dot + self%frequencies(k, j)* &
                        x_dot(i, j)
                end do
                coefficient = -self%normalization*u(i, k)*cos(argument)* &
                    argument_dot
                do j = 1, self%n_inputs
                    x_hvp(i, j) = x_hvp(i, j) + coefficient* &
                        self%frequencies(k, j)
                end do
            end do
        end do
        theta_hvp = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_fourier_hvp

    logical function random_fourier_valid(self) result(valid)
        class(random_fourier_basis_impl_t), intent(in) :: self
        valid = self%n_inputs > 0 .and. self%n_components > 0
        if (.not. valid) return
        valid = allocated(self%frequencies)
        if (.not. valid) return
        valid = allocated(self%phases)
    end function random_fourier_valid

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

    subroutine radial_hvp(self, x, u, theta_dot, x_dot, theta_hvp, x_hvp, &
            status)
        class(radial_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, p, column, center_index, n
        real(dp) :: q, q_dot, radial_value, radial_value_dot
        real(dp) :: inverse_scale, inverse_scale_dot, log_value_dot

        if (size(u, 2) /= self%feature_count() .or. &
                size(theta_dot) /= self%parameter_count() .or. &
                size(theta_hvp) /= self%parameter_count() .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis radial hvp: array shape is invalid")
            return
        end if
        theta_hvp = 0.0_dp
        x_hvp = 0.0_dp
        n = self%n_inputs*self%n_centers
        do p = 1, self%n_centers
            column = p
            do i = 1, size(x, 1)
                radial_value = 0.0_dp
                log_value_dot = 0.0_dp
                do j = 1, self%n_inputs
                    center_index = (p - 1)*self%n_inputs + j
                    inverse_scale = exp(-self%log_scales(j, p))
                    inverse_scale_dot = -inverse_scale*theta_dot(n + center_index)
                    q = (x(i, j) - self%centers(j, p))*inverse_scale
                    q_dot = inverse_scale*(x_dot(i, j) - theta_dot(center_index)) + &
                        inverse_scale_dot*(x(i, j) - self%centers(j, p))
                    radial_value = radial_value + q*q
                    log_value_dot = log_value_dot - q*q_dot
                end do
                radial_value = exp(-0.5_dp*radial_value)
                radial_value_dot = radial_value*log_value_dot
                do j = 1, self%n_inputs
                    center_index = (p - 1)*self%n_inputs + j
                    inverse_scale = exp(-self%log_scales(j, p))
                    inverse_scale_dot = -inverse_scale*theta_dot(n + center_index)
                    q = (x(i, j) - self%centers(j, p))*inverse_scale
                    q_dot = inverse_scale*(x_dot(i, j) - theta_dot(center_index)) + &
                        inverse_scale_dot*(x(i, j) - self%centers(j, p))
                    x_hvp(i, j) = x_hvp(i, j) - u(i, column)* &
                        (radial_value_dot*q*inverse_scale + radial_value*q_dot*inverse_scale + &
                         radial_value*q*inverse_scale_dot)
                    theta_hvp(center_index) = theta_hvp(center_index) + u(i, column)* &
                        (radial_value_dot*q*inverse_scale + radial_value*q_dot*inverse_scale + &
                         radial_value*q*inverse_scale_dot)
                    theta_hvp(n + center_index) = theta_hvp(n + center_index) + &
                        u(i, column)*(radial_value_dot*q*q + 2.0_dp*radial_value*q*q_dot)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine radial_hvp

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

    subroutine spline_hvp(self, x, u, theta_dot, x_dot, theta_hvp, x_hvp, &
            status)
        !! Spline HVP is analytic within a fixed knot span.  Crossing a knot
        !! is a non-smooth event and is rejected by the underlying evaluator.
        class(spline_basis_impl_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: theta_hvp(:), x_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, column, ncoef
        real(dp), allocatable :: dvalues(:, :)

        if (size(u, 2) /= self%feature_count() .or. &
                size(theta_dot) /= 0 .or. size(theta_hvp) /= 0 .or. &
                any(shape(x_dot) /= shape(x)) .or. &
                any(shape(x_hvp) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline hvp: array shape is invalid")
            return
        end if
        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis spline hvp: model is not initialized")
            return
        end if
        theta_hvp = 0.0_dp
        x_hvp = 0.0_dp
        column = 1
        ncoef = self%spline(1)%ncoef
        allocate(dvalues(0:2, ncoef))
        do j = 1, self%n_inputs
            do i = 1, size(x, 1)
                call bspline_eval_deriv(self%spline(j), x(i, j), 2, dvalues, &
                    status)
                if (status%code /= FORTNUM_OK) return
                x_hvp(i, j) = sum(u(i, column:column + ncoef - 1)* &
                    dvalues(2, :))*x_dot(i, j)
            end do
            column = column + ncoef
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine spline_hvp

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
