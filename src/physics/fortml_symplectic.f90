module fortml_symplectic
    !! Canonical symplectic-form residuals and derivative products.
    !!
    !! For a map with Jacobian ``A`` in canonical ``[q,p]`` coordinates the
    !! structure defect is ``D = A^T Omega A - Omega``, where
    !! ``Omega = [ 0 I ; -I 0 ]``.  The residual is the column-major packed
    !! defect matrix.  This module never estimates a derivative by finite
    !! differences: callers provide the map Jacobian and its JVP/VJP.
    !!
    !! ``symplectic_constraint_t`` is a small bridge to
    !! ``physics_constraint_t``.  A map provider owns its state and supplies
    !! exact Jacobian products; the bridge only performs the canonical form
    !! reduction and can therefore be reused by HNN, SympNet, Verlet, and GP
    !! flow-map providers without prescribing their representation.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_physics_objective, only: physics_constraint_t, &
        physics_residual_proc, physics_residual_jvp_proc, &
        physics_residual_vjp_proc
    implicit none
    private

    abstract interface
        subroutine symplectic_jacobian_proc(context, theta, jacobian, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: theta(:)
            real(dp), intent(out) :: jacobian(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine symplectic_jacobian_proc

        subroutine symplectic_jacobian_jvp_proc(context, theta, theta_dot, &
                jacobian, jacobian_dot, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: theta(:), theta_dot(:)
            real(dp), intent(out) :: jacobian(:, :), jacobian_dot(:, :)
            type(fortnum_status_t), intent(out) :: status
        end subroutine symplectic_jacobian_jvp_proc

        subroutine symplectic_jacobian_vjp_proc(context, theta, jacobian_bar, &
                theta_bar, status)
            import :: dp, fortnum_status_t
            class(*), pointer, intent(in) :: context
            real(dp), intent(in) :: theta(:), jacobian_bar(:, :)
            real(dp), intent(out) :: theta_bar(:)
            type(fortnum_status_t), intent(out) :: status
        end subroutine symplectic_jacobian_vjp_proc
    end interface

    type, public :: symplectic_form_diagnostic_t
        !! Canonical form residual and normalized least-squares diagnostic.
        private
        integer :: n_coordinates = 0
        integer :: state_size = 0
        integer :: device_kind = FORTML_DEVICE_CPU
        real(dp) :: weight = 1.0_dp
        logical :: ready = .false.
    contains
        procedure, public :: initialize => symplectic_diagnostic_initialize
        procedure, public :: initialized => symplectic_diagnostic_initialized
        procedure, public :: coordinate_count => symplectic_coordinate_count
        procedure, public :: state_dimension => symplectic_state_dimension
        procedure, public :: residual_count => symplectic_residual_count
        procedure, public :: residual_weight => symplectic_diagnostic_weight
        procedure, public :: device_supported => symplectic_device_supported
        procedure, public :: select_device => symplectic_select_device
        procedure, public :: residual => symplectic_diagnostic_residual
        procedure, public :: residual_jvp => symplectic_diagnostic_residual_jvp
        procedure, public :: residual_vjp => symplectic_diagnostic_residual_vjp
        procedure, public :: value => symplectic_diagnostic_value
        procedure, public :: value_jvp => symplectic_diagnostic_value_jvp
        procedure, public :: value_vjp => symplectic_diagnostic_value_vjp
        procedure, public :: is_symplectic => symplectic_diagnostic_is_symplectic
    end type symplectic_form_diagnostic_t

    type, public :: symplectic_constraint_t
        !! Jacobian-map adapter for ``physics_constraint_t``.
        private
        type(symplectic_form_diagnostic_t) :: diagnostic
        integer :: n_parameters = 0
        logical :: ready = .false.
        class(*), pointer :: context => null()
        procedure(symplectic_jacobian_proc), pointer, nopass :: jacobian_proc => null()
        procedure(symplectic_jacobian_jvp_proc), pointer, nopass :: jvp_proc => null()
        procedure(symplectic_jacobian_vjp_proc), pointer, nopass :: vjp_proc => null()
    contains
        procedure, public :: initialize => symplectic_constraint_initialize
        procedure, public :: initialized => symplectic_constraint_initialized
        procedure, public :: parameter_count => symplectic_constraint_parameter_count
        procedure, public :: residual_count => symplectic_constraint_residual_count
        procedure, public :: residual => symplectic_constraint_residual
        procedure, public :: residual_jvp => symplectic_constraint_residual_jvp
        procedure, public :: residual_vjp => symplectic_constraint_residual_vjp
        procedure, public :: value => symplectic_constraint_value
        procedure, public :: value_jvp => symplectic_constraint_value_jvp
        procedure, public :: value_vjp => symplectic_constraint_value_vjp
        procedure, public :: as_constraint => symplectic_constraint_as_constraint
        procedure, public :: device_supported => symplectic_constraint_device_supported
    end type symplectic_constraint_t

    public :: symplectic_form_matrix
    public :: symplectic_form_residual
    public :: symplectic_form_residual_jvp
    public :: symplectic_form_residual_vjp
    public :: symplectic_form_value
    public :: symplectic_form_value_jvp
    public :: symplectic_form_value_vjp
    public :: symplectic_form_is_symplectic
    public :: symplectic_jacobian_proc, symplectic_jacobian_jvp_proc
    public :: symplectic_jacobian_vjp_proc

contains

    subroutine symplectic_diagnostic_initialize(self, n_coordinates, status, &
            weight, device_kind)
        class(symplectic_form_diagnostic_t), intent(out) :: self
        integer, intent(in) :: n_coordinates
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: weight
        integer, intent(in), optional :: device_kind
        real(dp) :: requested_weight
        integer :: requested_device

        requested_weight = 1.0_dp
        if (present(weight)) requested_weight = weight
        requested_device = FORTML_DEVICE_CPU
        if (present(device_kind)) requested_device = device_kind
        self%n_coordinates = 0
        self%state_size = 0
        self%device_kind = FORTML_DEVICE_CPU
        self%weight = 1.0_dp
        self%ready = .false.
        if (n_coordinates < 1 .or. .not. ieee_is_finite(requested_weight) .or. &
                requested_weight <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid coordinates or weight")
            return
        end if
        if (requested_device == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "symplectic diagnostic: resident CUDA derivative kernel is not implemented")
            return
        end if
        if (requested_device /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid device kind")
            return
        end if
        self%n_coordinates = n_coordinates
        self%state_size = 2*n_coordinates
        self%device_kind = requested_device
        self%weight = requested_weight
        self%ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_diagnostic_initialize

    logical function symplectic_diagnostic_initialized(self) result(yes)
        class(symplectic_form_diagnostic_t), intent(in) :: self

        yes = self%ready .and. self%n_coordinates > 0 .and. &
            self%state_size == 2*self%n_coordinates .and. &
            self%device_kind == FORTML_DEVICE_CPU
    end function symplectic_diagnostic_initialized

    integer function symplectic_coordinate_count(self) result(count)
        class(symplectic_form_diagnostic_t), intent(in) :: self

        count = self%n_coordinates
    end function symplectic_coordinate_count

    integer function symplectic_state_dimension(self) result(count)
        class(symplectic_form_diagnostic_t), intent(in) :: self

        count = self%state_size
    end function symplectic_state_dimension

    integer function symplectic_residual_count(self) result(count)
        class(symplectic_form_diagnostic_t), intent(in) :: self

        count = self%state_size*self%state_size
    end function symplectic_residual_count

    real(dp) function symplectic_diagnostic_weight(self) result(weight)
        class(symplectic_form_diagnostic_t), intent(in) :: self

        weight = self%weight
    end function symplectic_diagnostic_weight

    logical function symplectic_device_supported(self, device_kind) result(yes)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = self%initialized() .and. &
            requested == FORTML_DEVICE_CPU
    end function symplectic_device_supported

    subroutine symplectic_select_device(self, device_kind, status)
        class(symplectic_form_diagnostic_t), intent(inout) :: self
        integer, intent(in) :: device_kind
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: object is not initialized")
            return
        end if
        if (device_kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "symplectic diagnostic: resident CUDA derivative kernel is not implemented")
            return
        end if
        if (device_kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid device kind")
            return
        end if
        self%device_kind = device_kind
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_select_device

    subroutine symplectic_diagnostic_residual(self, jacobian, residual, status)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        real(dp), intent(in) :: jacobian(:, :)
        real(dp), intent(out) :: residual(:)
        type(fortnum_status_t), intent(out) :: status

        call validate_diagnostic(self, jacobian, residual, status)
        if (.not. status_ok(status)) return
        call symplectic_form_residual(jacobian, residual, status)
    end subroutine symplectic_diagnostic_residual

    subroutine symplectic_diagnostic_residual_jvp(self, jacobian, jacobian_dot, &
            residual, residual_dot, status)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        real(dp), intent(in) :: jacobian(:, :), jacobian_dot(:, :)
        real(dp), intent(out) :: residual(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status

        call validate_diagnostic(self, jacobian, residual, status)
        if (.not. status_ok(status)) return
        if (any(shape(jacobian_dot) /= [self%state_size, self%state_size]) .or. &
                any(.not. ieee_is_finite(jacobian_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid Jacobian tangent")
            return
        end if
        if (size(residual_dot) /= self%residual_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid residual tangent shape")
            return
        end if
        call symplectic_form_residual_jvp(jacobian, jacobian_dot, residual, &
            residual_dot, status)
    end subroutine symplectic_diagnostic_residual_jvp

    subroutine symplectic_diagnostic_residual_vjp(self, jacobian, residual_bar, &
            jacobian_bar, status)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        real(dp), intent(in) :: jacobian(:, :), residual_bar(:)
        real(dp), intent(out) :: jacobian_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        call validate_diagnostic(self, jacobian, residual_bar, status)
        if (.not. status_ok(status)) return
        if (any(shape(jacobian_bar) /= [self%state_size, self%state_size])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid Jacobian cotangent shape")
            return
        end if
        call symplectic_form_residual_vjp(jacobian, residual_bar, jacobian_bar, status)
    end subroutine symplectic_diagnostic_residual_vjp

    subroutine symplectic_diagnostic_value(self, jacobian, value, status)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        real(dp), intent(in) :: jacobian(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        call validate_diagnostic_matrix(self, jacobian, status)
        if (.not. status_ok(status)) return
        call symplectic_form_value(jacobian, value, status, self%weight)
    end subroutine symplectic_diagnostic_value

    subroutine symplectic_diagnostic_value_jvp(self, jacobian, jacobian_dot, &
            value, value_dot, status)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        real(dp), intent(in) :: jacobian(:, :), jacobian_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status

        call validate_diagnostic_matrix(self, jacobian, status)
        if (.not. status_ok(status)) return
        if (any(shape(jacobian_dot) /= [self%state_size, self%state_size]) .or. &
                any(.not. ieee_is_finite(jacobian_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid Jacobian tangent")
            return
        end if
        call symplectic_form_value_jvp(jacobian, jacobian_dot, value, value_dot, &
            status, self%weight)
    end subroutine symplectic_diagnostic_value_jvp

    subroutine symplectic_diagnostic_value_vjp(self, jacobian, value_bar, &
            jacobian_bar, status)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        real(dp), intent(in) :: jacobian(:, :), value_bar
        real(dp), intent(out) :: jacobian_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        call validate_diagnostic_matrix(self, jacobian, status)
        if (.not. status_ok(status)) return
        if (.not. ieee_is_finite(value_bar) .or. &
                any(shape(jacobian_bar) /= [self%state_size, self%state_size])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid value cotangent")
            return
        end if
        call symplectic_form_value_vjp(jacobian, value_bar, jacobian_bar, status, &
            self%weight)
    end subroutine symplectic_diagnostic_value_vjp

    subroutine symplectic_diagnostic_is_symplectic(self, jacobian, tolerance, yes, &
            status)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        real(dp), intent(in) :: jacobian(:, :), tolerance
        logical, intent(out) :: yes
        type(fortnum_status_t), intent(out) :: status

        call validate_diagnostic_matrix(self, jacobian, status)
        if (.not. status_ok(status)) then
            yes = .false.
            return
        end if
        if (.not. ieee_is_finite(tolerance) .or. tolerance < 0.0_dp) then
            yes = .false.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid tolerance")
            return
        end if
        call symplectic_form_is_symplectic(jacobian, tolerance, yes, status)
    end subroutine symplectic_diagnostic_is_symplectic

    subroutine symplectic_form_matrix(n_coordinates, omega, status)
        integer, intent(in) :: n_coordinates
        real(dp), intent(out) :: omega(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: n

        if (n_coordinates < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: coordinate count must be positive")
            return
        end if
        n = 2*n_coordinates
        if (any(shape(omega) /= [n, n])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: matrix shape is invalid")
            return
        end if
        omega = 0.0_dp
        omega(1:n_coordinates, n_coordinates+1:n) = identity_matrix(n_coordinates)
        omega(n_coordinates+1:n, 1:n_coordinates) = -identity_matrix(n_coordinates)
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_form_matrix

    subroutine symplectic_form_residual(jacobian, residual, status)
        real(dp), intent(in) :: jacobian(:, :)
        real(dp), intent(out) :: residual(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: omega(:, :), defect(:, :)
        integer :: n, nq

        call validate_jacobian(jacobian, residual, n, nq, status)
        if (.not. status_ok(status)) return
        allocate(omega(n, n), defect(n, n))
        call symplectic_form_matrix(nq, omega, status)
        if (.not. status_ok(status)) return
        defect = matmul(transpose(jacobian), matmul(omega, jacobian)) - omega
        residual = reshape(defect, [n*n])
        if (any(.not. ieee_is_finite(residual))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: non-finite defect")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_form_residual

    subroutine symplectic_form_residual_jvp(jacobian, jacobian_dot, residual, &
            residual_dot, status)
        real(dp), intent(in) :: jacobian(:, :), jacobian_dot(:, :)
        real(dp), intent(out) :: residual(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: omega(:, :), defect_dot(:, :)
        integer :: n, nq

        call validate_jacobian(jacobian, residual, n, nq, status)
        if (.not. status_ok(status)) return
        if (any(shape(jacobian_dot) /= [n, n]) .or. &
                any(.not. ieee_is_finite(jacobian_dot)) .or. &
                size(residual_dot) /= n*n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: invalid Jacobian tangent or output")
            return
        end if
        allocate(omega(n, n), defect_dot(n, n))
        call symplectic_form_matrix(nq, omega, status)
        if (.not. status_ok(status)) return
        defect_dot = matmul(transpose(jacobian_dot), matmul(omega, jacobian)) + &
            matmul(transpose(jacobian), matmul(omega, jacobian_dot))
        call symplectic_form_residual(jacobian, residual, status)
        if (.not. status_ok(status)) return
        residual_dot = reshape(defect_dot, [n*n])
        if (any(.not. ieee_is_finite(residual_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: non-finite residual tangent")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_form_residual_jvp

    subroutine symplectic_form_residual_vjp(jacobian, residual_bar, jacobian_bar, &
            status)
        real(dp), intent(in) :: jacobian(:, :), residual_bar(:)
        real(dp), intent(out) :: jacobian_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: omega(:, :), bar(:, :)
        integer :: n, nq

        call validate_jacobian(jacobian, residual_bar, n, nq, status)
        if (.not. status_ok(status)) return
        if (any(shape(jacobian_bar) /= [n, n]) .or. &
                any(.not. ieee_is_finite(residual_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: invalid Jacobian cotangent or output")
            return
        end if
        allocate(omega(n, n), bar(n, n))
        call symplectic_form_matrix(nq, omega, status)
        if (.not. status_ok(status)) return
        bar = reshape(residual_bar, [n, n])
        jacobian_bar = matmul(omega, matmul(jacobian, transpose(bar))) + &
            matmul(transpose(omega), matmul(jacobian, bar))
        if (any(.not. ieee_is_finite(jacobian_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: non-finite Jacobian cotangent")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_form_residual_vjp

    subroutine symplectic_form_value(jacobian, value, status, weight)
        real(dp), intent(in) :: jacobian(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: weight
        real(dp), allocatable :: residual(:)
        real(dp) :: scale
        integer :: n

        scale = 1.0_dp
        if (present(weight)) scale = weight
        n = size(jacobian, 1)
        if (.not. ieee_is_finite(scale) .or. scale <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: weight must be finite and positive")
            return
        end if
        allocate(residual(max(1, n*n)))
        call symplectic_form_residual(jacobian, residual, status)
        if (.not. status_ok(status)) return
        value = scale*dot_product(residual, residual)/(2.0_dp*real(n*n, dp))
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: value is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_form_value

    subroutine symplectic_form_value_jvp(jacobian, jacobian_dot, value, value_dot, &
            status, weight)
        real(dp), intent(in) :: jacobian(:, :), jacobian_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: weight
        real(dp), allocatable :: residual(:), residual_dot(:)
        real(dp) :: scale
        integer :: n

        scale = 1.0_dp
        if (present(weight)) scale = weight
        n = size(jacobian, 1)
        if (.not. ieee_is_finite(scale) .or. scale <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: weight must be finite and positive")
            return
        end if
        allocate(residual(max(1, n*n)), residual_dot(max(1, n*n)))
        call symplectic_form_residual_jvp(jacobian, jacobian_dot, residual, &
            residual_dot, status)
        if (.not. status_ok(status)) return
        value = scale*dot_product(residual, residual)/(2.0_dp*real(n*n, dp))
        value_dot = scale*dot_product(residual, residual_dot)/real(n*n, dp)
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(value_dot)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: value tangent is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_form_value_jvp

    subroutine symplectic_form_value_vjp(jacobian, value_bar, jacobian_bar, status, &
            weight)
        real(dp), intent(in) :: jacobian(:, :), value_bar
        real(dp), intent(out) :: jacobian_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: weight
        real(dp), allocatable :: residual(:)
        real(dp) :: scale
        integer :: n

        scale = 1.0_dp
        if (present(weight)) scale = weight
        n = size(jacobian, 1)
        if (.not. ieee_is_finite(scale) .or. scale <= 0.0_dp .or. &
                .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: invalid value cotangent or weight")
            return
        end if
        allocate(residual(max(1, n*n)))
        call symplectic_form_residual(jacobian, residual, status)
        if (.not. status_ok(status)) return
        call symplectic_form_residual_vjp(jacobian, &
            value_bar*scale*residual/real(n*n, dp), jacobian_bar, status)
    end subroutine symplectic_form_value_vjp

    subroutine symplectic_form_is_symplectic(jacobian, tolerance, yes, status)
        real(dp), intent(in) :: jacobian(:, :), tolerance
        logical, intent(out) :: yes
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: residual(:)
        integer :: n

        yes = .false.
        n = size(jacobian, 1)
        allocate(residual(max(1, n*n)))
        call symplectic_form_residual(jacobian, residual, status)
        if (.not. status_ok(status)) return
        if (.not. ieee_is_finite(tolerance) .or. tolerance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: invalid tolerance")
            return
        end if
        yes = maxval(abs(residual)) <= tolerance
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_form_is_symplectic

    subroutine validate_jacobian(jacobian, residual, n, nq, status)
        real(dp), intent(in) :: jacobian(:, :), residual(:)
        integer, intent(out) :: n, nq
        type(fortnum_status_t), intent(out) :: status

        n = size(jacobian, 1)
        nq = n/2
        if (n < 2 .or. mod(n, 2) /= 0 .or. size(jacobian, 2) /= n .or. &
                size(residual) /= n*n .or. any(.not. ieee_is_finite(jacobian))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic form: Jacobian or residual shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_jacobian

    subroutine validate_diagnostic(self, jacobian, residual, status)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        real(dp), intent(in) :: jacobian(:, :), residual(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: object is not initialized")
            return
        end if
        if (any(shape(jacobian) /= [self%state_size, self%state_size]) .or. &
                size(residual) /= self%residual_count() .or. &
                any(.not. ieee_is_finite(jacobian))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid Jacobian or residual shape")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_diagnostic

    subroutine validate_diagnostic_matrix(self, jacobian, status)
        class(symplectic_form_diagnostic_t), intent(in) :: self
        real(dp), intent(in) :: jacobian(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: object is not initialized")
            return
        end if
        if (any(shape(jacobian) /= [self%state_size, self%state_size]) .or. &
                any(.not. ieee_is_finite(jacobian))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic diagnostic: invalid Jacobian shape or values")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_diagnostic_matrix

    subroutine symplectic_constraint_initialize(self, n_parameters, n_coordinates, &
            context, jacobian_proc, jvp_proc, vjp_proc, status, weight, device_kind)
        class(symplectic_constraint_t), intent(out), target :: self
        integer, intent(in) :: n_parameters, n_coordinates
        class(*), target, intent(inout), optional :: context
        procedure(symplectic_jacobian_proc) :: jacobian_proc
        procedure(symplectic_jacobian_jvp_proc) :: jvp_proc
        procedure(symplectic_jacobian_vjp_proc) :: vjp_proc
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: weight
        integer, intent(in), optional :: device_kind

        integer :: requested_device

        self%n_parameters = 0
        self%ready = .false.
        nullify(self%context, self%jacobian_proc, self%jvp_proc, self%vjp_proc)
        requested_device = FORTML_DEVICE_CPU
        if (present(device_kind)) requested_device = device_kind
        call self%diagnostic%initialize(n_coordinates, status, weight=weight, &
            device_kind=requested_device)
        if (.not. status_ok(status)) return
        if (n_parameters < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic constraint: parameter count must be positive")
            return
        end if
        self%n_parameters = n_parameters
        if (present(context)) self%context => context
        self%jacobian_proc => jacobian_proc
        self%jvp_proc => jvp_proc
        self%vjp_proc => vjp_proc
        self%ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine symplectic_constraint_initialize

    logical function symplectic_constraint_initialized(self) result(yes)
        class(symplectic_constraint_t), intent(in) :: self

        yes = self%ready .and. self%diagnostic%initialized() .and. &
            self%n_parameters > 0 .and. associated(self%jacobian_proc) .and. &
            associated(self%jvp_proc) .and. associated(self%vjp_proc)
    end function symplectic_constraint_initialized

    integer function symplectic_constraint_parameter_count(self) result(count)
        class(symplectic_constraint_t), intent(in) :: self

        count = self%n_parameters
    end function symplectic_constraint_parameter_count

    integer function symplectic_constraint_residual_count(self) result(count)
        class(symplectic_constraint_t), intent(in) :: self

        count = self%diagnostic%residual_count()
    end function symplectic_constraint_residual_count

    subroutine symplectic_constraint_residual(self, theta, residual, status)
        class(symplectic_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: residual(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: jacobian(:, :)

        if (.not. self%initialized() .or. size(theta) /= self%n_parameters .or. &
                size(residual) /= self%residual_count() .or. &
                any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic constraint: invalid state or output shape")
            return
        end if
        allocate(jacobian(self%diagnostic%state_dimension(), &
            self%diagnostic%state_dimension()))
        call self%jacobian_proc(self%context, theta, jacobian, status)
        if (.not. status_ok(status)) return
        call self%diagnostic%residual(jacobian, residual, status)
    end subroutine symplectic_constraint_residual

    subroutine symplectic_constraint_residual_jvp(self, theta, theta_dot, residual, &
            residual_dot, status)
        class(symplectic_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: residual(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: jacobian(:, :), jacobian_dot(:, :)

        if (.not. self%initialized() .or. size(theta) /= self%n_parameters .or. &
                size(theta_dot) /= self%n_parameters .or. &
                size(residual) /= self%residual_count() .or. &
                size(residual_dot) /= self%residual_count() .or. &
                any(.not. ieee_is_finite(theta)) .or. any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic constraint: invalid JVP shape or values")
            return
        end if
        allocate(jacobian(self%diagnostic%state_dimension(), &
            self%diagnostic%state_dimension()), jacobian_dot(&
            self%diagnostic%state_dimension(), self%diagnostic%state_dimension()))
        call self%jvp_proc(self%context, theta, theta_dot, jacobian, jacobian_dot, status)
        if (.not. status_ok(status)) return
        call self%diagnostic%residual_jvp(jacobian, jacobian_dot, residual, residual_dot, &
            status)
    end subroutine symplectic_constraint_residual_jvp

    subroutine symplectic_constraint_residual_vjp(self, theta, residual_bar, theta_bar, &
            status)
        class(symplectic_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), residual_bar(:)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: jacobian(:, :), jacobian_bar(:, :)

        if (.not. self%initialized() .or. size(theta) /= self%n_parameters .or. &
                size(residual_bar) /= self%residual_count() .or. &
                size(theta_bar) /= self%n_parameters .or. &
                any(.not. ieee_is_finite(theta)) .or. &
                any(.not. ieee_is_finite(residual_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic constraint: invalid VJP shape or values")
            return
        end if
        allocate(jacobian(self%diagnostic%state_dimension(), &
            self%diagnostic%state_dimension()), jacobian_bar(&
            self%diagnostic%state_dimension(), self%diagnostic%state_dimension()))
        call self%jacobian_proc(self%context, theta, jacobian, status)
        if (.not. status_ok(status)) return
        call self%diagnostic%residual_vjp(jacobian, residual_bar, jacobian_bar, status)
        if (.not. status_ok(status)) return
        call self%vjp_proc(self%context, theta, jacobian_bar, theta_bar, status)
    end subroutine symplectic_constraint_residual_vjp

    subroutine symplectic_constraint_value(self, theta, value, status)
        class(symplectic_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: jacobian(:, :)

        if (.not. self%initialized() .or. size(theta) /= self%n_parameters .or. &
                any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic constraint: invalid state")
            return
        end if
        allocate(jacobian(self%diagnostic%state_dimension(), &
            self%diagnostic%state_dimension()))
        call self%jacobian_proc(self%context, theta, jacobian, status)
        if (.not. status_ok(status)) return
        call self%diagnostic%value(jacobian, value, status)
    end subroutine symplectic_constraint_value

    subroutine symplectic_constraint_value_jvp(self, theta, theta_dot, value, &
            value_dot, status)
        class(symplectic_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: jacobian(:, :), jacobian_dot(:, :)

        if (.not. self%initialized() .or. size(theta) /= self%n_parameters .or. &
                size(theta_dot) /= self%n_parameters .or. &
                any(.not. ieee_is_finite(theta)) .or. any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic constraint: invalid value JVP inputs")
            return
        end if
        allocate(jacobian(self%diagnostic%state_dimension(), &
            self%diagnostic%state_dimension()), jacobian_dot(&
            self%diagnostic%state_dimension(), self%diagnostic%state_dimension()))
        call self%jvp_proc(self%context, theta, theta_dot, jacobian, jacobian_dot, status)
        if (.not. status_ok(status)) return
        call self%diagnostic%value_jvp(jacobian, jacobian_dot, value, value_dot, status)
    end subroutine symplectic_constraint_value_jvp

    subroutine symplectic_constraint_value_vjp(self, theta, value_bar, theta_bar, status)
        class(symplectic_constraint_t), intent(in) :: self
        real(dp), intent(in) :: theta(:), value_bar
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: jacobian(:, :), jacobian_bar(:, :)

        if (.not. self%initialized() .or. size(theta) /= self%n_parameters .or. &
                size(theta_bar) /= self%n_parameters .or. &
                .not. ieee_is_finite(value_bar) .or. &
                any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic constraint: invalid value VJP inputs")
            return
        end if
        allocate(jacobian(self%diagnostic%state_dimension(), &
            self%diagnostic%state_dimension()), jacobian_bar(&
            self%diagnostic%state_dimension(), self%diagnostic%state_dimension()))
        call self%jacobian_proc(self%context, theta, jacobian, status)
        if (.not. status_ok(status)) return
        call self%diagnostic%value_vjp(jacobian, value_bar, jacobian_bar, status)
        if (.not. status_ok(status)) return
        call self%vjp_proc(self%context, theta, jacobian_bar, theta_bar, status)
    end subroutine symplectic_constraint_value_vjp

    subroutine symplectic_constraint_as_constraint(self, constraint, status)
        class(symplectic_constraint_t), intent(inout), target :: self
        type(physics_constraint_t), intent(out) :: constraint
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "symplectic constraint: object is not initialized")
            return
        end if
        call constraint%initialize(self%n_parameters, self%diagnostic%residual_count(), &
            self%diagnostic%residual_weight(), self, symplectic_constraint_residual_wrapper, &
            symplectic_constraint_jvp_wrapper, symplectic_constraint_vjp_wrapper, status)
    end subroutine symplectic_constraint_as_constraint

    logical function symplectic_constraint_device_supported(self, device_kind) result(yes)
        class(symplectic_constraint_t), intent(in) :: self
        integer, intent(in), optional :: device_kind

        yes = self%initialized() .and. self%diagnostic%device_supported(device_kind)
    end function symplectic_constraint_device_supported

    subroutine symplectic_constraint_residual_wrapper(context, theta, residual, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: residual(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (symplectic_constraint_t)
                call context%residual(theta, residual, status)
            class default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "symplectic constraint: invalid callback context")
        end select
    end subroutine symplectic_constraint_residual_wrapper

    subroutine symplectic_constraint_jvp_wrapper(context, theta, theta_dot, residual, &
            residual_dot, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: residual(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (symplectic_constraint_t)
                call context%residual_jvp(theta, theta_dot, residual, residual_dot, status)
            class default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "symplectic constraint: invalid callback context")
        end select
    end subroutine symplectic_constraint_jvp_wrapper

    subroutine symplectic_constraint_vjp_wrapper(context, theta, residual_bar, theta_bar, &
            status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), residual_bar(:)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status

        select type (context)
            type is (symplectic_constraint_t)
                call context%residual_vjp(theta, residual_bar, theta_bar, status)
            class default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "symplectic constraint: invalid callback context")
        end select
    end subroutine symplectic_constraint_vjp_wrapper

    pure function identity_matrix(n) result(matrix)
        integer, intent(in) :: n
        real(dp) :: matrix(n, n)
        integer :: i

        matrix = 0.0_dp
        do i = 1, n
            matrix(i, i) = 1.0_dp
        end do
    end function identity_matrix

end module fortml_symplectic
