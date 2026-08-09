module fortml_pinn_structure_gp
    !! Finite-feature GP initialization for a named PINN residual objective.
    !!
    !! The initializer reuses the ordinary fixed-depth MLP posterior and
    !! changes only the final affine output layer.  A physics objective is
    !! evaluated before and after the candidate application, so the four
    !! named data/residual/boundary/conservation contributions remain
    !! observable.  This is a finite-width warm start; it is not an
    !! infinite-width GP or NNGP equivalence.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t
    use fortml_mlp_structure_gp, only: mlp_structure_gp_initializer_t
    use fortml_physics_objective, only: physics_objective_t
    implicit none
    private

    type, public :: pinn_structure_gp_metadata_t
        !! Fit provenance and named residual diagnostics.
        character(len=64) :: method = "finite-feature-pinn-structure-gp"
        character(len=48) :: structure_scope = "fixed-depth-affine-pinn-output"
        integer :: sample_count = 0
        integer :: feature_dimension = 0
        integer :: output_dimension = 0
        integer :: total_parameter_count = 0
        integer :: hidden_parameter_count = 0
        real(dp) :: regularization = 0.0_dp
        real(dp) :: hidden_scale = 0.0_dp
        real(dp) :: structure_tolerance = 0.0_dp
        real(dp) :: structure_defect = 0.0_dp
        real(dp) :: objective_before = 0.0_dp
        real(dp) :: objective_after = 0.0_dp
        real(dp) :: residual_term_before = 0.0_dp
        real(dp) :: residual_term_after = 0.0_dp
        real(dp) :: term_values_before(4) = 0.0_dp
        real(dp) :: term_values_after(4) = 0.0_dp
        logical :: exact_infinite_width = .false.
        logical :: hidden_parameters_frozen = .true.
        logical :: named_terms_preserved = .true.
        logical :: cuda_supported = .false.
    end type pinn_structure_gp_metadata_t

    type, public :: pinn_structure_gp_initializer_t
        !! Validated finite-feature PINN output-layer initializer.
        private
        type(mlp_structure_gp_initializer_t) :: posterior
        real(dp), allocatable :: hidden_snapshot(:)
        real(dp) :: hidden_scale_value = 0.0_dp
        real(dp) :: structure_tolerance_value = 1.0e-12_dp
        real(dp) :: terms_before_value(4) = 0.0_dp
        real(dp) :: terms_after_value(4) = 0.0_dp
        integer :: total_parameter_count_value = 0
        integer :: hidden_parameter_count_value = 0
        logical :: diagnostics_value = .false.
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => pinn_structure_gp_fit
        procedure, public :: fit_apply => pinn_structure_gp_fit_apply
        procedure, public :: apply => pinn_structure_gp_apply
        procedure, public :: predict => pinn_structure_gp_predict
        procedure, public :: jvp => pinn_structure_gp_jvp
        procedure, public :: predictive_variance => &
            pinn_structure_gp_predictive_variance
        procedure, public :: predictive_variance_jvp => &
            pinn_structure_gp_predictive_variance_jvp
        procedure, public :: objective_terms => pinn_structure_gp_objective_terms
        procedure, public :: apply_cuda => pinn_structure_gp_apply_cuda
        procedure, public :: predict_cuda => pinn_structure_gp_predict_cuda
        procedure, public :: jvp_cuda => pinn_structure_gp_jvp_cuda
        procedure, public :: predictive_variance_cuda => &
            pinn_structure_gp_predictive_variance_cuda
        procedure, public :: fitted => pinn_structure_gp_fitted
        procedure, public :: metadata => pinn_structure_gp_metadata
        procedure, public :: hidden_parameters => pinn_structure_gp_hidden_parameters
        procedure, public :: coefficients => pinn_structure_gp_coefficients
    end type pinn_structure_gp_initializer_t

contains

    subroutine pinn_structure_gp_fit(self, model, x, target, objective, status, &
            regularization, structure_tolerance)
        class(pinn_structure_gp_initializer_t), intent(inout) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(physics_objective_t), intent(in) :: objective
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: regularization, structure_tolerance
        type(mlp_structure_gp_initializer_t) :: candidate
        type(mlp_t) :: trial_model
        real(dp), allocatable :: before(:), after(:)
        real(dp) :: terms_before(4), terms_after(4), tolerance, scale
        integer :: hidden_count, feature_count, output_count

        tolerance = self%structure_tolerance_value
        if (present(structure_tolerance)) tolerance = structure_tolerance
        if (.not. ieee_is_finite(tolerance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP fit: structure tolerance is nonfinite")
            return
        end if
        if (tolerance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP fit: structure tolerance is negative")
            return
        end if
        if (.not. objective%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP fit: physics objective is uninitialized")
            return
        end if
        if (objective%parameter_count() /= model%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP fit: objective/model parameter counts differ")
            return
        end if

        call candidate%fit(model, x, target, status, regularization)
        if (.not. status_ok(status)) return
        before = model%parameters()
        call objective%term_values(before, terms_before, status)
        if (.not. status_ok(status)) return

        trial_model = model
        call candidate%apply(trial_model, status)
        if (.not. status_ok(status)) return
        after = trial_model%parameters()
        call objective%term_values(after, terms_after, status)
        if (.not. status_ok(status)) return
        if (.not. all(ieee_is_finite(terms_before)) .or. &
                .not. all(ieee_is_finite(terms_after))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP fit: objective diagnostics are nonfinite")
            return
        end if

        feature_count = candidate%feature_dimension()
        output_count = candidate%output_dimension()
        hidden_count = model%parameter_count() - feature_count*output_count - output_count
        if (hidden_count < 1 .or. hidden_count >= model%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP fit: hidden parameter layout is invalid")
            return
        end if
        scale = sqrt(sum(before(1:hidden_count)**2)/real(hidden_count, dp))
        if (.not. ieee_is_finite(scale)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP fit: hidden parameter scale is nonfinite")
            return
        end if

        self%posterior = candidate
        if (allocated(self%hidden_snapshot)) deallocate(self%hidden_snapshot)
        allocate(self%hidden_snapshot(hidden_count))
        self%hidden_snapshot = before(1:hidden_count)
        self%hidden_scale_value = scale
        self%structure_tolerance_value = tolerance
        self%terms_before_value = terms_before
        self%terms_after_value = terms_after
        self%total_parameter_count_value = size(before)
        self%hidden_parameter_count_value = hidden_count
        self%diagnostics_value = .true.
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine pinn_structure_gp_fit

    subroutine pinn_structure_gp_fit_apply(self, model, x, target, objective, status, &
            regularization, structure_tolerance)
        class(pinn_structure_gp_initializer_t), intent(inout) :: self
        class(mlp_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(physics_objective_t), intent(in) :: objective
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: regularization, structure_tolerance
        real(dp), allocatable :: old_parameters(:)

        old_parameters = model%parameters()
        call self%fit(model, x, target, objective, status, regularization, &
            structure_tolerance)
        if (.not. status_ok(status)) return
        call self%apply(model, status)
        if (.not. status_ok(status)) then
            call model%set_parameters(old_parameters, status)
        end if
    end subroutine pinn_structure_gp_fit_apply

    subroutine pinn_structure_gp_apply(self, model, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        call self%posterior%apply(model, status)
    end subroutine pinn_structure_gp_apply

    subroutine pinn_structure_gp_predict(self, model, x, y, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        call self%posterior%predict(model, x, y, status)
    end subroutine pinn_structure_gp_predict

    subroutine pinn_structure_gp_jvp(self, model, x, regularization_direction, &
            y, dy, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), regularization_direction
        real(dp), allocatable, intent(out) :: y(:, :), dy(:, :)
        type(fortnum_status_t), intent(out) :: status

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        call self%posterior%jvp(model, x, regularization_direction, y, dy, status)
    end subroutine pinn_structure_gp_jvp

    subroutine pinn_structure_gp_predictive_variance(self, model, x, variance, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        call self%posterior%predictive_variance(model, x, variance, status)
    end subroutine pinn_structure_gp_predictive_variance

    subroutine pinn_structure_gp_predictive_variance_jvp(self, model, x, &
            regularization_direction, variance, dvariance, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), regularization_direction
        real(dp), allocatable, intent(out) :: variance(:), dvariance(:)
        type(fortnum_status_t), intent(out) :: status

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        call self%posterior%predictive_variance_jvp(model, x, &
            regularization_direction, variance, dvariance, status)
    end subroutine pinn_structure_gp_predictive_variance_jvp

    subroutine pinn_structure_gp_objective_terms(self, objective, model, values, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        type(physics_objective_t), intent(in) :: objective
        class(mlp_t), intent(in) :: model
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: parameters(:)

        if (size(values) /= 4) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP objective_terms: output shape is invalid")
            return
        end if
        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        if (.not. objective%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP objective_terms: objective is uninitialized")
            return
        end if
        parameters = model%parameters()
        call objective%term_values(parameters, values, status)
    end subroutine pinn_structure_gp_objective_terms

    subroutine pinn_structure_gp_apply_cuda(self, model, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "PINN structure GP apply_cuda: resident CUDA residual graph is unavailable")
    end subroutine pinn_structure_gp_apply_cuda

    subroutine pinn_structure_gp_predict_cuda(self, model, x, y, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(inout) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "PINN structure GP predict_cuda: resident CUDA residual graph is unavailable")
    end subroutine pinn_structure_gp_predict_cuda

    subroutine pinn_structure_gp_jvp_cuda(self, model, x, regularization_direction, &
            y, dy, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), regularization_direction
        real(dp), allocatable, intent(inout) :: y(:, :), dy(:, :)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "PINN structure GP jvp_cuda: resident CUDA products are unavailable")
    end subroutine pinn_structure_gp_jvp_cuda

    subroutine pinn_structure_gp_predictive_variance_cuda(self, model, x, variance, &
            status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(inout) :: variance(:)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "PINN structure GP predictive_variance_cuda: resident CUDA products are unavailable")
    end subroutine pinn_structure_gp_predictive_variance_cuda

    logical function pinn_structure_gp_fitted(self) result(value)
        class(pinn_structure_gp_initializer_t), intent(in) :: self

        value = self%fitted_value
    end function pinn_structure_gp_fitted

    function pinn_structure_gp_metadata(self) result(metadata)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        type(pinn_structure_gp_metadata_t) :: metadata

        metadata%total_parameter_count = self%total_parameter_count_value
        metadata%hidden_parameter_count = self%hidden_parameter_count_value
        metadata%hidden_scale = self%hidden_scale_value
        metadata%structure_tolerance = self%structure_tolerance_value
        metadata%term_values_before = self%terms_before_value
        metadata%term_values_after = self%terms_after_value
        metadata%objective_before = sum(self%terms_before_value)
        metadata%objective_after = sum(self%terms_after_value)
        metadata%residual_term_before = self%terms_before_value(2)
        metadata%residual_term_after = self%terms_after_value(2)
        if (self%fitted_value) then
            metadata%sample_count = self%posterior%sample_count()
            metadata%feature_dimension = self%posterior%feature_dimension()
            metadata%output_dimension = self%posterior%output_dimension()
            metadata%regularization = self%posterior%regularization()
        end if
    end function pinn_structure_gp_metadata

    function pinn_structure_gp_hidden_parameters(self) result(values)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%hidden_snapshot)) then
            allocate(values, source=self%hidden_snapshot)
        else
            allocate(values(0))
        end if
    end function pinn_structure_gp_hidden_parameters

    function pinn_structure_gp_coefficients(self) result(values)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        values = self%posterior%coefficients()
    end function pinn_structure_gp_coefficients

    subroutine check_structure(self, model, status)
        class(pinn_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: parameters(:)
        real(dp) :: scale, tolerance

        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP query: initializer is unfitted")
            return
        end if
        if (.not. allocated(self%hidden_snapshot)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP query: hidden snapshot is unavailable")
            return
        end if
        if (model%parameter_count() /= self%total_parameter_count_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP query: model parameter layout changed")
            return
        end if
        parameters = model%parameters()
        if (size(parameters) /= self%total_parameter_count_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP query: packed parameter layout changed")
            return
        end if
        tolerance = self%structure_tolerance_value
        if (maxval(abs(parameters(1:self%hidden_parameter_count_value) - &
                self%hidden_snapshot)) > tolerance*(1.0_dp + &
                maxval(abs(self%hidden_snapshot)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP query: hidden parameters changed")
            return
        end if
        scale = sqrt(sum(parameters(1:self%hidden_parameter_count_value)**2)/ &
            real(self%hidden_parameter_count_value, dp))
        if (abs(scale - self%hidden_scale_value) > tolerance*(1.0_dp + &
                abs(self%hidden_scale_value))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "PINN structure GP query: hidden scale changed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_structure

end module fortml_pinn_structure_gp
