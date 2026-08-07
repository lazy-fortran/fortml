module fortml_hamiltonian_mlp
    !! Hamiltonian neural networks with exact derivative products.
    !!
    !! The model stores two scalar MLPs, V(q) and T(p), and defines
    !! H(q,p) = V(q) + T(p), or a general scalar H(q,p).  The split is
    !! intentional: the explicit leapfrog map below is symplectic for every
    !! differentiable V and T, whereas a general learned H(q,p) needs an
    !! implicit integrator.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_TANH
    implicit none
    private

    type, public :: hamiltonian_mlp_t
        private
        type(mlp_t) :: potential
        type(mlp_t) :: kinetic
        type(mlp_t) :: general
        integer :: n_coordinates = 0
        logical :: general_mode = .false.
    contains
        procedure, public :: initialize => hamiltonian_mlp_initialize
        procedure, public :: initialize_general => hamiltonian_mlp_initialize_general
        procedure, public :: parameter_count => hamiltonian_mlp_parameter_count
        procedure, public :: parameters => hamiltonian_mlp_parameters
        procedure, public :: set_parameters => hamiltonian_mlp_set_parameters
        procedure, public :: energy => hamiltonian_mlp_energy
        procedure, public :: energy_gradient => hamiltonian_mlp_energy_gradient
        procedure, public :: energy_jvp => hamiltonian_mlp_energy_jvp
        procedure, public :: energy_vjp => hamiltonian_mlp_energy_vjp
        procedure, public :: vector_field => hamiltonian_mlp_vector_field
        procedure, public :: vector_field_jvp => hamiltonian_mlp_vector_field_jvp
        procedure, public :: leapfrog => hamiltonian_mlp_leapfrog
        procedure, public :: coordinate_count => hamiltonian_mlp_coordinate_count
        procedure, public :: is_general => hamiltonian_mlp_is_general
    end type hamiltonian_mlp_t

contains

    subroutine hamiltonian_mlp_initialize(self, n_coordinates, potential_layers, &
            kinetic_layers, status, hidden_activation, initialization_seed)
        class(hamiltonian_mlp_t), intent(out) :: self
        integer, intent(in) :: n_coordinates, potential_layers(:), kinetic_layers(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_activation, initialization_seed
        integer :: activation, seed

        activation = MLP_TANH
        if (present(hidden_activation)) activation = hidden_activation
        seed = 17
        if (present(initialization_seed)) seed = initialization_seed
        if (n_coordinates < 1 .or. size(potential_layers) < 2 .or. &
            size(kinetic_layers) < 2 .or. potential_layers(1) /= n_coordinates .or. &
            kinetic_layers(1) /= n_coordinates .or. &
            potential_layers(size(potential_layers)) /= 1 .or. &
            kinetic_layers(size(kinetic_layers)) /= 1 .or. seed < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP initialize: invalid separable layer shapes")
            return
        end if

        call self%potential%initialize(potential_layers, status, &
            hidden_activation=activation, initialization_seed=seed)
        if (.not. status_ok(status)) return
        call self%kinetic%initialize(kinetic_layers, status, &
            hidden_activation=activation, initialization_seed=seed + 7919)
        if (.not. status_ok(status)) return
        self%n_coordinates = n_coordinates
        self%general_mode = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine hamiltonian_mlp_initialize

    subroutine hamiltonian_mlp_initialize_general(self, n_coordinates, layers, &
            status, hidden_activation, initialization_seed)
        class(hamiltonian_mlp_t), intent(out) :: self
        integer, intent(in) :: n_coordinates, layers(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_activation, initialization_seed
        integer :: activation, seed

        activation = MLP_TANH
        if (present(hidden_activation)) activation = hidden_activation
        seed = 17
        if (present(initialization_seed)) seed = initialization_seed
        if (n_coordinates < 1 .or. size(layers) < 2 .or. &
            layers(1) /= 2*n_coordinates .or. layers(size(layers)) /= 1 .or. &
            seed < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP initialize_general: invalid layer shapes")
            return
        end if

        call self%general%initialize(layers, status, hidden_activation=activation, &
            initialization_seed=seed)
        if (.not. status_ok(status)) return
        self%n_coordinates = n_coordinates
        self%general_mode = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine hamiltonian_mlp_initialize_general

    integer function hamiltonian_mlp_coordinate_count(self) result(count)
        class(hamiltonian_mlp_t), intent(in) :: self

        count = self%n_coordinates
    end function hamiltonian_mlp_coordinate_count

    logical function hamiltonian_mlp_is_general(self) result(general)
        class(hamiltonian_mlp_t), intent(in) :: self

        general = self%general_mode
    end function hamiltonian_mlp_is_general

    integer function hamiltonian_mlp_parameter_count(self) result(count)
        class(hamiltonian_mlp_t), intent(in) :: self

        if (self%general_mode) then
            count = self%general%parameter_count()
        else
            count = self%potential%parameter_count() + self%kinetic%parameter_count()
        end if
    end function hamiltonian_mlp_parameter_count

    function hamiltonian_mlp_parameters(self) result(theta)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        integer :: np, nk

        if (self%general_mode) then
            theta = self%general%parameters()
            return
        end if
        np = self%potential%parameter_count()
        nk = self%kinetic%parameter_count()
        allocate(theta(np + nk))
        if (np > 0) theta(1:np) = self%potential%parameters()
        if (nk > 0) theta(np + 1:np + nk) = self%kinetic%parameters()
    end function hamiltonian_mlp_parameters

    subroutine hamiltonian_mlp_set_parameters(self, theta, status)
        class(hamiltonian_mlp_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: np, nk

        if (self%general_mode) then
            if (size(theta) /= self%general%parameter_count() .or. &
                any(.not. ieee_is_finite(theta))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Hamiltonian MLP set_parameters: invalid parameter vector")
                return
            end if
            call self%general%set_parameters(theta, status)
            return
        end if
        np = self%potential%parameter_count()
        nk = self%kinetic%parameter_count()
        if (self%n_coordinates < 1 .or. size(theta) /= np + nk .or. &
            any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP set_parameters: invalid parameter vector")
            return
        end if
        call self%potential%set_parameters(theta(1:np), status)
        if (.not. status_ok(status)) return
        call self%kinetic%set_parameters(theta(np + 1:np + nk), status)
    end subroutine hamiltonian_mlp_set_parameters

    subroutine hamiltonian_mlp_energy(self, state, energy, status)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: state(:, :)
        real(dp), intent(out) :: energy(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: q(:, :), p(:, :), potential(:, :), kinetic(:, :)

        if (.not. valid_state(self, state) .or. any(shape(energy) /= [size(state, 1), 1])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP energy: invalid state or output shape")
            return
        end if
        if (self%general_mode) then
            call self%general%predict(state, energy, status)
            return
        end if
        call split_state(self, state, q, p)
        allocate(potential(size(state, 1), 1), kinetic(size(state, 1), 1))
        call self%potential%predict(q, potential, status)
        if (.not. status_ok(status)) return
        call self%kinetic%predict(p, kinetic, status)
        if (.not. status_ok(status)) return
        energy = potential + kinetic
    end subroutine hamiltonian_mlp_energy

    subroutine hamiltonian_mlp_energy_gradient(self, state, energy, gradient, status)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: state(:, :)
        real(dp), intent(out) :: energy(:, :), gradient(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: q(:, :), p(:, :), potential(:, :), kinetic(:, :)
        real(dp), allocatable :: q_bar(:, :), p_bar(:, :), one(:, :)
        real(dp), allocatable :: potential_bar(:), kinetic_bar(:)
        integer :: n, nq

        nq = self%n_coordinates
        n = size(state, 1)
        if (.not. valid_state(self, state) .or. any(shape(energy) /= [n, 1]) .or. &
            any(shape(gradient) /= shape(state))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP energy_gradient: invalid state or output shape")
            return
        end if
        if (self%general_mode) then
            allocate(one(n, 1), potential_bar(self%general%parameter_count()))
            one = 1.0_dp
            call self%general%predict(state, energy, status)
            if (.not. status_ok(status)) return
            call self%general%vjp(state, one, potential_bar, gradient, status)
            return
        end if
        call split_state(self, state, q, p)
        allocate(potential(n, 1), kinetic(n, 1), one(n, 1))
        allocate(q_bar(n, nq), p_bar(n, nq))
        allocate(potential_bar(self%potential%parameter_count()))
        allocate(kinetic_bar(self%kinetic%parameter_count()))
        one = 1.0_dp
        call self%potential%predict(q, potential, status)
        if (.not. status_ok(status)) return
        call self%kinetic%predict(p, kinetic, status)
        if (.not. status_ok(status)) return
        call self%potential%vjp(q, one, potential_bar, q_bar, status)
        if (.not. status_ok(status)) return
        call self%kinetic%vjp(p, one, kinetic_bar, p_bar, status)
        if (.not. status_ok(status)) return
        energy = potential + kinetic
        gradient(:, 1:nq) = q_bar
        gradient(:, nq + 1:2*nq) = p_bar
    end subroutine hamiltonian_mlp_energy_gradient

    subroutine hamiltonian_mlp_energy_jvp(self, state, dtheta, dstate, energy, &
            denergy, status)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: state(:, :), dtheta(:), dstate(:, :)
        real(dp), intent(out) :: energy(:, :), denergy(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: q(:, :), p(:, :), dq(:, :), dp_state(:, :)
        real(dp), allocatable :: potential(:, :), kinetic(:, :), dpotential(:, :), &
            dkinetic(:, :)
        integer :: np, nk, n

        n = size(state, 1)
        if (self%general_mode) then
            if (.not. valid_state(self, state) .or. any(shape(dstate) /= shape(state)) .or. &
                size(dtheta) /= self%general%parameter_count() .or. &
                any(.not. ieee_is_finite(dtheta)) .or. any(.not. ieee_is_finite(dstate)) .or. &
                any(shape(energy) /= [n, 1]) .or. any(shape(denergy) /= [n, 1])) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Hamiltonian MLP energy_jvp: invalid state, direction, or output")
                return
            end if
            call self%general%jvp(state, dtheta, dstate, energy, denergy, status)
            return
        end if
        np = self%potential%parameter_count()
        nk = self%kinetic%parameter_count()
        if (.not. valid_state(self, state) .or. any(shape(dstate) /= shape(state)) .or. &
            size(dtheta) /= np + nk .or. any(.not. ieee_is_finite(dtheta)) .or. &
            any(.not. ieee_is_finite(dstate)) .or. any(shape(energy) /= [n, 1]) .or. &
            any(shape(denergy) /= [n, 1])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP energy_jvp: invalid state, direction, or output")
            return
        end if
        call split_state(self, state, q, p)
        call split_state(self, dstate, dq, dp_state)
        allocate(potential(n, 1), kinetic(n, 1), dpotential(n, 1), dkinetic(n, 1))
        call self%potential%jvp(q, dtheta(1:np), dq, potential, dpotential, status)
        if (.not. status_ok(status)) return
        call self%kinetic%jvp(p, dtheta(np + 1:np + nk), dp_state, kinetic, dkinetic, status)
        if (.not. status_ok(status)) return
        energy = potential + kinetic
        denergy = dpotential + dkinetic
    end subroutine hamiltonian_mlp_energy_jvp

    subroutine hamiltonian_mlp_energy_vjp(self, state, energy_bar, parameter_bar, &
            state_bar, status)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: state(:, :), energy_bar(:)
        real(dp), intent(out) :: parameter_bar(:), state_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: q(:, :), p(:, :), cotangent(:, :), q_bar(:, :), p_bar(:, :)
        real(dp), allocatable :: potential_bar(:), kinetic_bar(:)
        integer :: np, nk, n, nq

        n = size(state, 1)
        nq = self%n_coordinates
        if (self%general_mode) then
            if (.not. valid_state(self, state) .or. size(energy_bar) /= n .or. &
                size(parameter_bar) /= self%general%parameter_count() .or. &
                any(shape(state_bar) /= shape(state)) .or. &
                any(.not. ieee_is_finite(energy_bar))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Hamiltonian MLP energy_vjp: invalid state, cotangent, or output")
                return
            end if
            allocate(cotangent(n, 1))
            cotangent(:, 1) = energy_bar
            call self%general%vjp(state, cotangent, parameter_bar, state_bar, status)
            return
        end if
        np = self%potential%parameter_count()
        nk = self%kinetic%parameter_count()
        if (.not. valid_state(self, state) .or. size(energy_bar) /= n .or. &
            size(parameter_bar) /= np + nk .or. any(shape(state_bar) /= shape(state)) .or. &
            any(.not. ieee_is_finite(energy_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP energy_vjp: invalid state, cotangent, or output")
            return
        end if
        call split_state(self, state, q, p)
        allocate(cotangent(n, 1), q_bar(n, nq), p_bar(n, nq))
        allocate(potential_bar(np), kinetic_bar(nk))
        cotangent(:, 1) = energy_bar
        call self%potential%vjp(q, cotangent, potential_bar, q_bar, status)
        if (.not. status_ok(status)) return
        call self%kinetic%vjp(p, cotangent, kinetic_bar, p_bar, status)
        if (.not. status_ok(status)) return
        parameter_bar(1:np) = potential_bar
        parameter_bar(np + 1:np + nk) = kinetic_bar
        state_bar(:, 1:nq) = q_bar
        state_bar(:, nq + 1:2*nq) = p_bar
    end subroutine hamiltonian_mlp_energy_vjp

    subroutine hamiltonian_mlp_vector_field(self, state, field, status)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: state(:, :)
        real(dp), intent(out) :: field(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: energy(:, :), gradient(:, :)
        integer :: nq

        if (.not. valid_state(self, state) .or. any(shape(field) /= shape(state))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP vector_field: invalid state or output shape")
            return
        end if
        nq = self%n_coordinates
        allocate(energy(size(state, 1), 1), gradient(size(state, 1), 2*nq))
        call self%energy_gradient(state, energy, gradient, status)
        if (.not. status_ok(status)) return
        field(:, 1:nq) = gradient(:, nq + 1:2*nq)
        field(:, nq + 1:2*nq) = -gradient(:, 1:nq)
    end subroutine hamiltonian_mlp_vector_field

    subroutine hamiltonian_mlp_vector_field_jvp(self, state, dtheta, dstate, field, &
            dfield, status)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: state(:, :), dtheta(:), dstate(:, :)
        real(dp), intent(out) :: field(:, :), dfield(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: q(:, :), p(:, :), dq(:, :), dp_state(:, :)
        real(dp), allocatable :: one(:, :), potential_bar(:), kinetic_bar(:)
        real(dp), allocatable :: q_hvp(:, :), p_hvp(:, :), potential_hvp(:), kinetic_hvp(:)
        real(dp), allocatable :: gradient(:, :), q_bar(:, :), p_bar(:, :)
        integer :: n, nq, np, nk

        n = size(state, 1)
        nq = self%n_coordinates
        if (self%general_mode) then
            if (.not. valid_state(self, state) .or. any(shape(dstate) /= shape(state)) .or. &
                any(shape(field) /= shape(state)) .or. any(shape(dfield) /= shape(state)) .or. &
                size(dtheta) /= self%general%parameter_count() .or. &
                any(.not. ieee_is_finite(dtheta)) .or. any(.not. ieee_is_finite(dstate))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "Hamiltonian MLP vector_field_jvp: invalid state or direction")
                return
            end if
            allocate(one(n, 1), potential_bar(self%general%parameter_count()))
            allocate(potential_hvp(self%general%parameter_count()), q_hvp(n, 2*nq))
            allocate(gradient(n, 2*nq))
            one = 1.0_dp
            call self%general%vjp(state, one, potential_bar, gradient, status)
            if (.not. status_ok(status)) return
            call self%general%hvp(state, one, dtheta, dstate, potential_hvp, &
                q_hvp, status)
            if (.not. status_ok(status)) return
            field(:, 1:nq) = gradient(:, nq + 1:2*nq)
            field(:, nq + 1:2*nq) = -gradient(:, 1:nq)
            dfield(:, 1:nq) = q_hvp(:, nq + 1:2*nq)
            dfield(:, nq + 1:2*nq) = -q_hvp(:, 1:nq)
            return
        end if
        np = self%potential%parameter_count()
        nk = self%kinetic%parameter_count()
        if (.not. valid_state(self, state) .or. any(shape(dstate) /= shape(state)) .or. &
            any(shape(field) /= shape(state)) .or. any(shape(dfield) /= shape(state)) .or. &
            size(dtheta) /= np + nk .or. any(.not. ieee_is_finite(dtheta)) .or. &
            any(.not. ieee_is_finite(dstate))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP vector_field_jvp: invalid state or direction")
            return
        end if
        call split_state(self, state, q, p)
        call split_state(self, dstate, dq, dp_state)
        allocate(one(n, 1), potential_bar(np), kinetic_bar(nk))
        allocate(q_hvp(n, nq), p_hvp(n, nq), potential_hvp(np), kinetic_hvp(nk))
        allocate(gradient(n, 2*nq), q_bar(n, nq), p_bar(n, nq))
        one = 1.0_dp
        call self%potential%vjp(q, one, potential_bar, q_bar, status)
        if (.not. status_ok(status)) return
        call self%kinetic%vjp(p, one, kinetic_bar, p_bar, status)
        if (.not. status_ok(status)) return
        call self%potential%hvp(q, one, dtheta(1:np), dq, potential_hvp, q_hvp, status)
        if (.not. status_ok(status)) return
        call self%kinetic%hvp(p, one, dtheta(np + 1:np + nk), dp_state, &
            kinetic_hvp, p_hvp, status)
        if (.not. status_ok(status)) return
        gradient(:, 1:nq) = q_bar
        gradient(:, nq + 1:2*nq) = p_bar
        field(:, 1:nq) = p_bar
        field(:, nq + 1:2*nq) = -q_bar
        dfield(:, 1:nq) = p_hvp
        dfield(:, nq + 1:2*nq) = -q_hvp
    end subroutine hamiltonian_mlp_vector_field_jvp

    subroutine hamiltonian_mlp_leapfrog(self, state, step, next_state, status)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: state(:, :), step
        real(dp), intent(out) :: next_state(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: q(:, :), p(:, :), q_next(:, :), p_half(:, :)
        real(dp), allocatable :: gq(:, :), gp(:, :)
        real(dp), allocatable :: one(:, :), potential_bar(:), kinetic_bar(:)
        integer :: n, nq

        n = size(state, 1)
        nq = self%n_coordinates
        if (.not. valid_state(self, state) .or. .not. ieee_is_finite(step) .or. &
            any(shape(next_state) /= shape(state))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian MLP leapfrog: invalid state, step, or output shape")
            return
        end if
        if (self%general_mode) then
            next_state = state
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "Hamiltonian MLP leapfrog: general H(q,p) requires an implicit integrator")
            return
        end if
        call split_state(self, state, q, p)
        allocate(gq(n, nq), gp(n, nq), q_next(n, nq), p_half(n, nq))
        allocate(one(n, 1), potential_bar(self%potential%parameter_count()))
        allocate(kinetic_bar(self%kinetic%parameter_count()))
        one = 1.0_dp
        call self%potential%vjp(q, one, potential_bar, gq, status)
        if (.not. status_ok(status)) return
        p_half = p - 0.5_dp*step*gq
        call self%kinetic%vjp(p_half, one, kinetic_bar, gp, status)
        if (.not. status_ok(status)) return
        q_next = q + step*gp
        call self%potential%vjp(q_next, one, potential_bar, gq, status)
        if (.not. status_ok(status)) return
        next_state(:, 1:nq) = q_next
        next_state(:, nq + 1:2*nq) = p_half - 0.5_dp*step*gq
    end subroutine hamiltonian_mlp_leapfrog

    logical function valid_state(self, state) result(valid)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: state(:, :)

        valid = self%n_coordinates > 0 .and. size(state, 1) > 0 .and. &
            size(state, 2) == 2*self%n_coordinates .and. all(ieee_is_finite(state))
    end function valid_state

    subroutine split_state(self, state, q, p)
        class(hamiltonian_mlp_t), intent(in) :: self
        real(dp), intent(in) :: state(:, :)
        real(dp), allocatable, intent(out) :: q(:, :), p(:, :)
        integer :: n, nq

        n = size(state, 1)
        nq = self%n_coordinates
        allocate(q(n, nq), p(n, nq))
        q = state(:, 1:nq)
        p = state(:, nq + 1:2*nq)
    end subroutine split_state

end module fortml_hamiltonian_mlp
