module fortml_mlp_structure_gp
    !! Structure-aware finite-feature GP initialization for ordinary MLPs.
    !!
    !! The initializer uses the existing finite-feature last-layer posterior,
    !! while making the structural boundary explicit: every parameter before
    !! the final affine layer is captured and must remain unchanged when the
    !! posterior is applied.  This gives a deterministic fixed-depth MLP
    !! warm-start that preserves the caller's topology, parameter ordering,
    !! and hidden-layer initialization scale.  It is not an infinite-width
    !! NNGP/NTK weight map.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_last_layer_gp, only: mlp_last_layer_gp_initializer_t
    implicit none
    private

    type, public :: mlp_structure_gp_metadata_t
        character(len=56) :: method = "finite-feature-structure-aware"
        character(len=48) :: structure_scope = "ordinary-fixed-depth-mlp"
        integer :: sample_count = 0
        integer :: total_parameter_count = 0
        integer :: hidden_parameter_count = 0
        integer :: feature_dimension = 0
        integer :: output_dimension = 0
        real(dp) :: regularization = 0.0_dp
        real(dp) :: hidden_scale = 0.0_dp
        real(dp) :: structure_tolerance = 0.0_dp
        logical :: exact_infinite_width = .false.
        logical :: cuda_supported = .false.
        logical :: hidden_parameters_frozen = .true.
    end type mlp_structure_gp_metadata_t

    type, public :: mlp_structure_gp_initializer_t
        private
        type(mlp_last_layer_gp_initializer_t) :: posterior
        real(dp), allocatable :: hidden_snapshot(:)
        integer :: total_parameter_count_value = 0
        integer :: hidden_parameter_count_value = 0
        real(dp) :: hidden_scale_value = 0.0_dp
        real(dp) :: structure_tolerance_value = 1.0e-12_dp
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => mlp_structure_gp_fit
        procedure, public :: fit_apply => mlp_structure_gp_fit_apply
        procedure, public :: apply => mlp_structure_gp_apply
        procedure, public :: predict => mlp_structure_gp_predict
        procedure, public :: jvp => mlp_structure_gp_jvp
        procedure, public :: predictive_variance => mlp_structure_gp_predictive_variance
        procedure, public :: predictive_variance_jvp => &
            mlp_structure_gp_predictive_variance_jvp
        procedure, public :: apply_cuda => mlp_structure_gp_apply_cuda
        procedure, public :: predict_cuda => mlp_structure_gp_predict_cuda
        procedure, public :: fitted => mlp_structure_gp_fitted
        procedure, public :: metadata => mlp_structure_gp_metadata
        procedure, public :: regularization => mlp_structure_gp_regularization
        procedure, public :: sample_count => mlp_structure_gp_sample_count
        procedure, public :: feature_dimension => mlp_structure_gp_feature_dimension
        procedure, public :: output_dimension => mlp_structure_gp_output_dimension
        procedure, public :: hidden_parameters => mlp_structure_gp_hidden_parameters
        procedure, public :: coefficients => mlp_structure_gp_coefficients
    end type mlp_structure_gp_initializer_t

contains

    subroutine mlp_structure_gp_fit(self, model, x, target, status, regularization, &
            structure_tolerance)
        class(mlp_structure_gp_initializer_t), intent(inout) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: regularization, structure_tolerance
        type(mlp_last_layer_gp_initializer_t) :: candidate
        real(dp), allocatable :: parameters(:)
        integer :: hidden_count
        real(dp) :: tolerance, scale

        tolerance = self%structure_tolerance_value
        if (present(structure_tolerance)) tolerance = structure_tolerance
        if (.not. ieee_is_finite(tolerance) .or. tolerance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP fit: structure tolerance is invalid")
            return
        end if
        if (.not. valid_model_structure(model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP fit: model must be fixed-depth with affine output")
            return
        end if
        call candidate%fit(model, x, target, status, regularization)
        if (.not. status_ok(status)) return

        parameters = model%parameters()
        hidden_count = hidden_parameter_count(model)
        if (hidden_count < 1 .or. hidden_count >= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP fit: hidden parameter layout is invalid")
            return
        end if
        scale = sqrt(sum(parameters(1:hidden_count)**2) / &
            real(hidden_count, dp))
        if (.not. ieee_is_finite(scale)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP fit: hidden parameter scale is nonfinite")
            return
        end if

        self%posterior = candidate
        allocate(self%hidden_snapshot(hidden_count))
        self%hidden_snapshot = parameters(1:hidden_count)
        self%total_parameter_count_value = size(parameters)
        self%hidden_parameter_count_value = hidden_count
        self%hidden_scale_value = scale
        self%structure_tolerance_value = tolerance
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_structure_gp_fit

    subroutine mlp_structure_gp_fit_apply(self, model, x, target, status, &
            regularization, structure_tolerance)
        class(mlp_structure_gp_initializer_t), intent(inout) :: self
        class(mlp_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: regularization, structure_tolerance
        real(dp), allocatable :: old_parameters(:)

        if (.not. valid_model_structure(model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP fit_apply: model is not fixed-depth with affine output")
            return
        end if
        old_parameters = model%parameters()
        call self%fit(model, x, target, status, regularization, structure_tolerance)
        if (.not. status_ok(status)) return
        call self%apply(model, status)
        if (.not. status_ok(status)) then
            call model%set_parameters(old_parameters, status)
        end if
    end subroutine mlp_structure_gp_fit_apply

    subroutine mlp_structure_gp_apply(self, model, status)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP apply: initializer is unfitted")
            return
        end if
        if (.not. valid_model_structure(model) .or. &
                model%parameter_count() /= self%total_parameter_count_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP apply: model topology or parameter layout changed")
            return
        end if
        if (.not. structure_matches(self, model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP apply: hidden parameters or scales changed")
            return
        end if
        call self%posterior%apply(model, status)
    end subroutine mlp_structure_gp_apply

    subroutine mlp_structure_gp_predict(self, model, x, y, status)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        call self%posterior%predict(model, x, y, status)
    end subroutine mlp_structure_gp_predict

    subroutine mlp_structure_gp_jvp(self, model, x, regularization_direction, &
            y, dy, status)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), regularization_direction
        real(dp), allocatable, intent(out) :: y(:, :), dy(:, :)
        type(fortnum_status_t), intent(out) :: status

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        call self%posterior%jvp(model, x, regularization_direction, y, dy, status)
    end subroutine mlp_structure_gp_jvp

    subroutine mlp_structure_gp_predictive_variance(self, model, x, variance, status)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: variance(:)
        type(fortnum_status_t), intent(out) :: status

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        call self%posterior%predictive_variance(model, x, variance, status)
    end subroutine mlp_structure_gp_predictive_variance

    subroutine mlp_structure_gp_predictive_variance_jvp(self, model, x, &
            regularization_direction, variance, dvariance, status)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), regularization_direction
        real(dp), allocatable, intent(out) :: variance(:), dvariance(:)
        type(fortnum_status_t), intent(out) :: status

        call check_structure(self, model, status)
        if (.not. status_ok(status)) return
        call self%posterior%predictive_variance_jvp(model, x, &
            regularization_direction, variance, dvariance, status)
    end subroutine mlp_structure_gp_predictive_variance_jvp

    subroutine mlp_structure_gp_apply_cuda(self, model, status)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "structure GP apply_cuda: resident CUDA GP/MLP state is unavailable")
    end subroutine mlp_structure_gp_apply_cuda

    subroutine mlp_structure_gp_predict_cuda(self, model, x, y, status)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "structure GP predict_cuda: resident CUDA GP/MLP state is unavailable")
    end subroutine mlp_structure_gp_predict_cuda

    logical function mlp_structure_gp_fitted(self) result(value)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        value = self%fitted_value
    end function mlp_structure_gp_fitted

    function mlp_structure_gp_metadata(self) result(metadata)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        type(mlp_structure_gp_metadata_t) :: metadata

        metadata%total_parameter_count = self%total_parameter_count_value
        metadata%hidden_parameter_count = self%hidden_parameter_count_value
        metadata%hidden_scale = self%hidden_scale_value
        metadata%structure_tolerance = self%structure_tolerance_value
        if (self%fitted_value) then
            metadata%sample_count = self%posterior%sample_count()
            metadata%feature_dimension = self%posterior%feature_dimension()
            metadata%output_dimension = self%posterior%output_dimension()
            metadata%regularization = self%posterior%regularization()
        end if
    end function mlp_structure_gp_metadata

    real(dp) function mlp_structure_gp_regularization(self) result(value)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        value = self%posterior%regularization()
    end function mlp_structure_gp_regularization

    integer function mlp_structure_gp_sample_count(self) result(value)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        value = self%posterior%sample_count()
    end function mlp_structure_gp_sample_count

    integer function mlp_structure_gp_feature_dimension(self) result(value)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        value = self%posterior%feature_dimension()
    end function mlp_structure_gp_feature_dimension

    integer function mlp_structure_gp_output_dimension(self) result(value)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        value = self%posterior%output_dimension()
    end function mlp_structure_gp_output_dimension

    function mlp_structure_gp_hidden_parameters(self) result(values)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%hidden_snapshot)) then
            allocate(values, source=self%hidden_snapshot)
        else
            allocate(values(0))
        end if
    end function mlp_structure_gp_hidden_parameters

    function mlp_structure_gp_coefficients(self) result(values)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        values = self%posterior%coefficients()
    end function mlp_structure_gp_coefficients

    subroutine check_structure(self, model, status)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP query: initializer is unfitted")
        else if (.not. valid_model_structure(model) .or. &
                model%parameter_count() /= self%total_parameter_count_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP query: model topology or parameter layout changed")
        else if (.not. structure_matches(self, model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "structure GP query: hidden parameters or scales changed")
        else
            call status_set(status, FORTNUM_OK, "")
        end if
    end subroutine check_structure

    logical function valid_model_structure(model) result(valid)
        class(mlp_t), intent(in) :: model

        valid = allocated(model%layer_sizes) .and. allocated(model%layer)
        if (.not. valid) return
        valid = size(model%layer_sizes) >= 3 .and. size(model%layer) >= 2
        if (.not. valid) return
        valid = model%output_activation == MLP_LINEAR
    end function valid_model_structure

    integer function hidden_parameter_count(model) result(count)
        class(mlp_t), intent(in) :: model
        integer :: total

        count = 0
        total = model%parameter_count()
        count = total - model%layer_sizes(size(model%layer_sizes) - 1) * &
            model%layer_sizes(size(model%layer_sizes)) - &
            model%layer_sizes(size(model%layer_sizes))
        if (count >= total) count = 0
    end function hidden_parameter_count

    logical function structure_matches(self, model) result(matches)
        class(mlp_structure_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), allocatable :: parameters(:)
        real(dp) :: scale, tolerance

        matches = allocated(self%hidden_snapshot)
        if (.not. matches) return
        parameters = model%parameters()
        if (size(parameters) /= self%total_parameter_count_value .or. &
                size(self%hidden_snapshot) /= self%hidden_parameter_count_value) then
            matches = .false.
            return
        end if
        tolerance = self%structure_tolerance_value
        matches = maxval(abs(parameters(1:self%hidden_parameter_count_value) - &
            self%hidden_snapshot)) <= tolerance*(1.0_dp + &
            maxval(abs(self%hidden_snapshot)))
        if (.not. matches) return
        scale = sqrt(sum(parameters(1:self%hidden_parameter_count_value)**2) / &
            real(self%hidden_parameter_count_value, dp))
        matches = abs(scale - self%hidden_scale_value) <= tolerance*(1.0_dp + &
            abs(self%hidden_scale_value))
    end function structure_matches

end module fortml_mlp_structure_gp
