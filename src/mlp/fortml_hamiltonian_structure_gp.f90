module fortml_hamiltonian_structure_gp
    !! Finite-feature GP posterior initialization for separable Hamiltonian MLPs.
    !!
    !! Two independent scalar finite-feature kernel-ridge posteriors initialize
    !! ``V(q)`` and ``T(p)``.  Only the final affine layer of each component is
    !! changed; all hidden parameters, the separable topology, and the
    !! canonical vector-field construction remain exact.  This is a bounded
    !! finite-width warm start, not an infinite-width NNGP equivalence.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t
    use fortml_hamiltonian_mlp, only: hamiltonian_mlp_t
    use fortml_mlp_last_layer_gp, only: mlp_last_layer_gp_initializer_t
    implicit none
    private

    type, public :: hamiltonian_structure_gp_metadata_t
        !! Provenance and structure/error diagnostics for one fitted initializer.
        character(len=64) :: method = "finite-feature-separable-hamiltonian-gp"
        character(len=40) :: structure_scope = "separable-V-plus-T"
        integer :: potential_sample_count = 0
        integer :: kinetic_sample_count = 0
        integer :: potential_feature_dimension = 0
        integer :: kinetic_feature_dimension = 0
        integer :: potential_hidden_parameter_count = 0
        integer :: kinetic_hidden_parameter_count = 0
        real(dp) :: regularization = 0.0_dp
        real(dp) :: potential_fit_rmse = 0.0_dp
        real(dp) :: kinetic_fit_rmse = 0.0_dp
        real(dp) :: structure_defect = 0.0_dp
        real(dp) :: structure_tolerance = 0.0_dp
        logical :: exact_infinite_width = .false.
        logical :: hidden_parameters_frozen = .true.
        logical :: separability_preserved = .true.
        logical :: cuda_supported = .false.
    end type hamiltonian_structure_gp_metadata_t

    type, public :: hamiltonian_structure_gp_initializer_t
        private
        type(mlp_last_layer_gp_initializer_t) :: potential_posterior
        type(mlp_last_layer_gp_initializer_t) :: kinetic_posterior
        real(dp), allocatable :: potential_hidden_snapshot(:)
        real(dp), allocatable :: kinetic_hidden_snapshot(:)
        integer :: potential_parameter_count_value = 0
        integer :: kinetic_parameter_count_value = 0
        integer :: potential_hidden_count_value = 0
        integer :: kinetic_hidden_count_value = 0
        real(dp) :: regularization_value = 0.0_dp
        real(dp) :: potential_fit_rmse_value = 0.0_dp
        real(dp) :: kinetic_fit_rmse_value = 0.0_dp
        real(dp) :: structure_defect_value = 0.0_dp
        real(dp) :: structure_tolerance_value = 1.0e-12_dp
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => hamiltonian_structure_gp_fit
        procedure, public :: fit_apply => hamiltonian_structure_gp_fit_apply
        procedure, public :: apply => hamiltonian_structure_gp_apply
        procedure, public :: predict_components => hamiltonian_structure_gp_predict_components
        procedure, public :: potential_predict => hamiltonian_structure_gp_potential_predict
        procedure, public :: kinetic_predict => hamiltonian_structure_gp_kinetic_predict
        procedure, public :: apply_cuda => hamiltonian_structure_gp_apply_cuda
        procedure, public :: predict_cuda => hamiltonian_structure_gp_predict_cuda
        procedure, public :: fitted => hamiltonian_structure_gp_fitted
        procedure, public :: metadata => hamiltonian_structure_gp_metadata
        procedure, public :: regularization => hamiltonian_structure_gp_regularization
        procedure, public :: potential_coefficients => hamiltonian_structure_gp_potential_coefficients
        procedure, public :: kinetic_coefficients => hamiltonian_structure_gp_kinetic_coefficients
        procedure, public :: hidden_parameters => hamiltonian_structure_gp_hidden_parameters
    end type hamiltonian_structure_gp_initializer_t

contains

    subroutine hamiltonian_structure_gp_fit(self, model, q, potential_target, p, &
            kinetic_target, status, regularization, structure_tolerance)
        class(hamiltonian_structure_gp_initializer_t), intent(inout) :: self
        class(hamiltonian_mlp_t), intent(in) :: model
        real(dp), intent(in) :: q(:, :), potential_target(:, :), p(:, :), kinetic_target(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: regularization, structure_tolerance
        type(mlp_t) :: potential_model, kinetic_model
        real(dp), allocatable :: parameters(:), potential_prediction(:, :), kinetic_prediction(:, :)
        integer :: np, nk, fp, fk, hp, hk
        real(dp) :: lambda, tolerance

        lambda = 1.0e-6_dp
        if (present(regularization)) lambda = regularization
        tolerance = self%structure_tolerance_value
        if (present(structure_tolerance)) tolerance = structure_tolerance
        if (.not. ieee_is_finite(lambda) .or. lambda <= 0.0_dp .or. &
            .not. ieee_is_finite(tolerance) .or. tolerance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP fit: regularization or tolerance is invalid")
            return
        end if
        if (.not. valid_fit_inputs(model, q, potential_target, p, kinetic_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP fit: separable model or data is invalid")
            return
        end if

        potential_model = model%potential_model()
        kinetic_model = model%kinetic_model()
        call self%potential_posterior%fit(potential_model, q, potential_target, status, lambda)
        if (.not. status_ok(status)) return
        call self%kinetic_posterior%fit(kinetic_model, p, kinetic_target, status, lambda)
        if (.not. status_ok(status)) return

        call self%potential_posterior%predict(potential_model, q, potential_prediction, status)
        if (.not. status_ok(status)) return
        call self%kinetic_posterior%predict(kinetic_model, p, kinetic_prediction, status)
        if (.not. status_ok(status)) return
        if (.not. all(ieee_is_finite(potential_prediction)) .or. &
            .not. all(ieee_is_finite(kinetic_prediction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP fit: posterior predictions are nonfinite")
            return
        end if

        parameters = model%parameters()
        np = model%potential_parameter_count()
        nk = model%kinetic_parameter_count()
        fp = self%potential_posterior%feature_dimension()
        fk = self%kinetic_posterior%feature_dimension()
        hp = np - fp - 1
        hk = nk - fk - 1
        if (np < 1 .or. nk < 1 .or. hp < 0 .or. hk < 0 .or. &
            size(parameters) /= np + nk) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP fit: parameter layout is invalid")
            return
        end if

        if (allocated(self%potential_hidden_snapshot)) deallocate(self%potential_hidden_snapshot)
        if (allocated(self%kinetic_hidden_snapshot)) deallocate(self%kinetic_hidden_snapshot)
        allocate(self%potential_hidden_snapshot(hp), self%kinetic_hidden_snapshot(hk))
        if (hp > 0) self%potential_hidden_snapshot = parameters(1:hp)
        if (hk > 0) self%kinetic_hidden_snapshot = parameters(np + 1:np + hk)
        self%potential_parameter_count_value = np
        self%kinetic_parameter_count_value = nk
        self%potential_hidden_count_value = hp
        self%kinetic_hidden_count_value = hk
        self%regularization_value = lambda
        self%potential_fit_rmse_value = sqrt(sum((potential_prediction - potential_target)**2) / &
            real(size(potential_target), dp))
        self%kinetic_fit_rmse_value = sqrt(sum((kinetic_prediction - kinetic_target)**2) / &
            real(size(kinetic_target), dp))
        self%structure_defect_value = 0.0_dp
        self%structure_tolerance_value = tolerance
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine hamiltonian_structure_gp_fit

    subroutine hamiltonian_structure_gp_fit_apply(self, model, q, potential_target, p, &
            kinetic_target, status, regularization, structure_tolerance)
        class(hamiltonian_structure_gp_initializer_t), intent(inout) :: self
        class(hamiltonian_mlp_t), intent(inout) :: model
        real(dp), intent(in) :: q(:, :), potential_target(:, :), p(:, :), kinetic_target(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: regularization, structure_tolerance
        real(dp), allocatable :: old_parameters(:)

        if (model%is_general()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP fit_apply: general Hamiltonian is unsupported")
            return
        end if
        old_parameters = model%parameters()
        call self%fit(model, q, potential_target, p, kinetic_target, status, &
            regularization, structure_tolerance)
        if (.not. status_ok(status)) return
        call self%apply(model, status)
        if (.not. status_ok(status)) then
            call model%set_parameters(old_parameters, status)
        end if
    end subroutine hamiltonian_structure_gp_fit_apply

    subroutine hamiltonian_structure_gp_apply(self, model, status)
        class(hamiltonian_structure_gp_initializer_t), intent(inout) :: self
        class(hamiltonian_mlp_t), intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: old_parameters(:), parameters(:)
        real(dp), allocatable :: potential_weight(:, :), kinetic_weight(:, :)
        real(dp), allocatable :: potential_bias(:), kinetic_bias(:)
        real(dp), allocatable :: potential_coefficients(:, :), kinetic_coefficients(:, :)
        integer :: fp, fk

        if (.not. self%fitted_value .or. model%is_general()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP apply: initializer or separable model is invalid")
            return
        end if
        if (model%potential_parameter_count() /= self%potential_parameter_count_value .or. &
            model%kinetic_parameter_count() /= self%kinetic_parameter_count_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP apply: model topology changed")
            return
        end if
        parameters = model%parameters()
        if (.not. hidden_state_matches(self, parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP apply: hidden state changed")
            return
        end if
        fp = self%potential_posterior%feature_dimension()
        fk = self%kinetic_posterior%feature_dimension()
        potential_coefficients = self%potential_posterior%coefficients()
        kinetic_coefficients = self%kinetic_posterior%coefficients()
        allocate(potential_weight(fp, 1), potential_bias(1))
        allocate(kinetic_weight(fk, 1), kinetic_bias(1))
        potential_weight(:, 1) = potential_coefficients(1:fp, 1)
        potential_bias = potential_coefficients(fp + 1, 1)
        kinetic_weight(:, 1) = kinetic_coefficients(1:fk, 1)
        kinetic_bias = kinetic_coefficients(fk + 1, 1)
        old_parameters = parameters
        call model%set_potential_last_layer_parameters(potential_weight, potential_bias, status)
        if (.not. status_ok(status)) return
        call model%set_kinetic_last_layer_parameters(kinetic_weight, kinetic_bias, status)
        if (.not. status_ok(status)) then
            call model%set_parameters(old_parameters, status)
            return
        end if
        self%structure_defect_value = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine hamiltonian_structure_gp_apply

    subroutine hamiltonian_structure_gp_predict_components(self, model, q, p, &
            potential, kinetic, status)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        class(hamiltonian_mlp_t), intent(in) :: model
        real(dp), intent(in) :: q(:, :), p(:, :)
        real(dp), allocatable, intent(out) :: potential(:, :), kinetic(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(mlp_t) :: potential_model, kinetic_model

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        if (size(q, 1) < 1 .or. size(p, 1) < 1 .or. size(q, 2) /= model%coordinate_count() .or. &
            size(p, 2) /= model%coordinate_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP predict: state component shape is invalid")
            return
        end if
        potential_model = model%potential_model()
        kinetic_model = model%kinetic_model()
        call self%potential_posterior%predict(potential_model, q, potential, status)
        if (.not. status_ok(status)) return
        call self%kinetic_posterior%predict(kinetic_model, p, kinetic, status)
    end subroutine hamiltonian_structure_gp_predict_components

    subroutine hamiltonian_structure_gp_potential_predict(self, model, q, prediction, status)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        class(hamiltonian_mlp_t), intent(in) :: model
        real(dp), intent(in) :: q(:, :)
        real(dp), allocatable, intent(out) :: prediction(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: kinetic(:, :)

        call self%predict_components(model, q, zero_component(model, size(q, 1)), prediction, kinetic, status)
    end subroutine hamiltonian_structure_gp_potential_predict

    subroutine hamiltonian_structure_gp_kinetic_predict(self, model, p, prediction, status)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        class(hamiltonian_mlp_t), intent(in) :: model
        real(dp), intent(in) :: p(:, :)
        real(dp), allocatable, intent(out) :: prediction(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: potential(:, :)

        call self%predict_components(model, zero_component(model, size(p, 1)), p, potential, prediction, status)
    end subroutine hamiltonian_structure_gp_kinetic_predict

    subroutine hamiltonian_structure_gp_apply_cuda(self, model, status)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        class(hamiltonian_mlp_t), intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "Hamiltonian structure GP apply_cuda: resident CUDA GP/MLP state is unavailable")
    end subroutine hamiltonian_structure_gp_apply_cuda

    subroutine hamiltonian_structure_gp_predict_cuda(self, model, q, p, potential, kinetic, status)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        class(hamiltonian_mlp_t), intent(in) :: model
        real(dp), intent(in) :: q(:, :), p(:, :)
        real(dp), allocatable, intent(out) :: potential(:, :), kinetic(:, :)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "Hamiltonian structure GP predict_cuda: resident CUDA GP/MLP state is unavailable")
    end subroutine hamiltonian_structure_gp_predict_cuda

    logical function hamiltonian_structure_gp_fitted(self) result(value)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        value = self%fitted_value
    end function hamiltonian_structure_gp_fitted

    function hamiltonian_structure_gp_metadata(self) result(metadata)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        type(hamiltonian_structure_gp_metadata_t) :: metadata

        metadata%regularization = self%regularization_value
        metadata%potential_fit_rmse = self%potential_fit_rmse_value
        metadata%kinetic_fit_rmse = self%kinetic_fit_rmse_value
        metadata%structure_defect = self%structure_defect_value
        metadata%structure_tolerance = self%structure_tolerance_value
        metadata%potential_hidden_parameter_count = self%potential_hidden_count_value
        metadata%kinetic_hidden_parameter_count = self%kinetic_hidden_count_value
        if (self%fitted_value) then
            metadata%potential_sample_count = self%potential_posterior%sample_count()
            metadata%kinetic_sample_count = self%kinetic_posterior%sample_count()
            metadata%potential_feature_dimension = self%potential_posterior%feature_dimension()
            metadata%kinetic_feature_dimension = self%kinetic_posterior%feature_dimension()
        end if
    end function hamiltonian_structure_gp_metadata

    real(dp) function hamiltonian_structure_gp_regularization(self) result(value)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        value = self%regularization_value
    end function hamiltonian_structure_gp_regularization

    function hamiltonian_structure_gp_potential_coefficients(self) result(values)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        values = self%potential_posterior%coefficients()
    end function hamiltonian_structure_gp_potential_coefficients

    function hamiltonian_structure_gp_kinetic_coefficients(self) result(values)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        values = self%kinetic_posterior%coefficients()
    end function hamiltonian_structure_gp_kinetic_coefficients

    function hamiltonian_structure_gp_hidden_parameters(self) result(values)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: np, nk

        np = size(self%potential_hidden_snapshot)
        nk = size(self%kinetic_hidden_snapshot)
        allocate(values(np + nk))
        if (np > 0) values(1:np) = self%potential_hidden_snapshot
        if (nk > 0) values(np + 1:np + nk) = self%kinetic_hidden_snapshot
    end function hamiltonian_structure_gp_hidden_parameters

    subroutine check_structure(self, model, status)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        class(hamiltonian_mlp_t), intent(in) :: model
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: parameters(:)

        if (.not. self%fitted_value .or. model%is_general()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP query: initializer or model is invalid")
            return
        end if
        if (model%potential_parameter_count() /= self%potential_parameter_count_value .or. &
            model%kinetic_parameter_count() /= self%kinetic_parameter_count_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP query: model topology changed")
            return
        end if
        parameters = model%parameters()
        if (.not. hidden_state_matches(self, parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Hamiltonian structure GP query: hidden state changed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_structure

    logical function hidden_state_matches(self, parameters) result(matches)
        class(hamiltonian_structure_gp_initializer_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:)
        integer :: np, hp, hk
        real(dp) :: tolerance

        np = self%potential_parameter_count_value
        hp = self%potential_hidden_count_value
        hk = self%kinetic_hidden_count_value
        matches = size(parameters) == np + self%kinetic_parameter_count_value
        if (.not. matches) return
        tolerance = self%structure_tolerance_value
        if (hp > 0) matches = maxval(abs(parameters(1:hp) - self%potential_hidden_snapshot)) <= &
            tolerance*(1.0_dp + maxval(abs(self%potential_hidden_snapshot)))
        if (.not. matches .or. hk == 0) return
        matches = maxval(abs(parameters(np + 1:np + hk) - self%kinetic_hidden_snapshot)) <= &
            tolerance*(1.0_dp + maxval(abs(self%kinetic_hidden_snapshot)))
    end function hidden_state_matches

    logical function valid_fit_inputs(model, q, potential_target, p, kinetic_target) result(valid)
        class(hamiltonian_mlp_t), intent(in) :: model
        real(dp), intent(in) :: q(:, :), potential_target(:, :), p(:, :), kinetic_target(:, :)

        valid = .not. model%is_general() .and. model%coordinate_count() > 0 .and. &
            size(q, 1) > 0 .and. size(p, 1) > 0 .and. size(q, 2) == model%coordinate_count() .and. &
            size(p, 2) == model%coordinate_count() .and. size(potential_target, 1) == size(q, 1) .and. &
            size(kinetic_target, 1) == size(p, 1) .and. size(potential_target, 2) == 1 .and. &
            size(kinetic_target, 2) == 1 .and. all(ieee_is_finite(q)) .and. all(ieee_is_finite(p)) .and. &
            all(ieee_is_finite(potential_target)) .and. all(ieee_is_finite(kinetic_target))
    end function valid_fit_inputs

    function zero_component(model, n) result(component)
        class(hamiltonian_mlp_t), intent(in) :: model
        integer, intent(in) :: n
        real(dp), allocatable :: component(:, :)

        allocate(component(n, model%coordinate_count()))
        component = 0.0_dp
    end function zero_component

end module fortml_hamiltonian_structure_gp
