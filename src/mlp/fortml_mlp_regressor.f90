module fortml_mlp_regressor
    !! Production dense MLP regressor facade.
    !!
    !! The facade keeps the public estimator contract separate from the
    !! low-level `mlp_t` and training products.  It supports deterministic
    !! Adam-family training with validation/checkpoint state, or the exact
    !! full-batch FortOpt L-BFGS-B objective.  Prediction and fixed-state
    !! parameter/input JVP/VJP products are delegated to the same network
    !! implementation used by the objective; no finite-difference fallback
    !! is hidden behind the estimator API.  CUDA is an explicit typed refusal
    !! until the complete resident trainer is linked.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_TANH
    use fortml_mlp_training, only: mlp_training_options_t, &
        mlp_training_state_t, mlp_training_checkpoint_t, &
        mlp_training_objective_t, mlp_lbfgsb_options_t, mlp_lbfgsb_result_t, &
        mlp_train, mlp_optimize_lbfgsb, mlp_loss_value_gradient, &
        mlp_loss_hvp
    implicit none
    private

    type, public :: mlp_regressor_options_t
        !! Estimator construction and training policy.
        integer, allocatable :: layer_sizes(:)
        integer :: hidden_activation = MLP_TANH
        integer :: initialization_seed = 17
        logical :: use_lbfgsb = .false.
        type(mlp_training_options_t) :: training
        type(mlp_lbfgsb_options_t) :: lbfgsb
    end type mlp_regressor_options_t

    type, public :: mlp_regressor_state_t
        type(mlp_training_state_t) :: training
        type(mlp_lbfgsb_result_t) :: lbfgsb
        logical :: used_lbfgsb = .false.
        logical :: converged = .false.
    end type mlp_regressor_state_t

    type, public :: mlp_regressor_t
        private
        type(mlp_t) :: model
        type(mlp_training_checkpoint_t) :: checkpoint_state
        type(mlp_regressor_state_t) :: fit_state
        integer :: n_features = 0
        integer :: n_outputs = 0
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => mlp_regressor_fit
        procedure, public :: predict => mlp_regressor_predict
        procedure, public :: predict_device => mlp_regressor_predict_device
        procedure, public :: predict_jvp => mlp_regressor_predict_jvp
        procedure, public :: predict_vjp => mlp_regressor_predict_vjp
        procedure, public :: loss_gradient => mlp_regressor_loss_gradient
        procedure, public :: loss_hvp => mlp_regressor_loss_hvp
        procedure, public :: parameters => mlp_regressor_parameters
        procedure, public :: set_parameters => mlp_regressor_set_parameters
        procedure, public :: parameter_count => mlp_regressor_parameter_count
        procedure, public :: feature_count => mlp_regressor_feature_count
        procedure, public :: output_count => mlp_regressor_output_count
        procedure, public :: state => mlp_regressor_state
        procedure, public :: training_checkpoint => mlp_regressor_checkpoint
        procedure, public :: fitted => mlp_regressor_fitted
        procedure, public :: device_supported => mlp_regressor_device_supported
    end type mlp_regressor_t

    public :: mlp_regressor_fit
    public :: mlp_regressor_predict
    public :: mlp_regressor_predict_device
    public :: mlp_regressor_predict_jvp
    public :: mlp_regressor_predict_vjp

contains

    subroutine mlp_regressor_fit(self, x, target, status, options, state, &
            validation_x, validation_target, checkpoint)
        class(mlp_regressor_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(mlp_regressor_options_t), intent(in) :: options
        type(mlp_regressor_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: validation_x(:, :), validation_target(:, :)
        type(mlp_training_checkpoint_t), intent(inout), optional :: checkpoint
        type(mlp_t) :: candidate
        type(mlp_regressor_state_t) :: candidate_state
        integer :: layers(2), n_layers
        logical :: valid_validation

        candidate_state%used_lbfgsb = .false.
        candidate_state%converged = .false.
        self%fitted_value = .false.
        valid_validation = present(validation_x) .eqv. present(validation_target)
        if (.not. valid_validation .or. size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(target, 1) /= size(x, 1) .or. size(target, 2) < 1 .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(target))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP regressor fit: finite row-oriented data and validation pairs are required")
            if (present(state)) state = candidate_state
            return
        end if
        if (allocated(options%layer_sizes)) then
            n_layers = size(options%layer_sizes)
            if (n_layers < 2 .or. any(options%layer_sizes < 1) .or. &
                options%layer_sizes(1) /= size(x, 2) .or. &
                options%layer_sizes(n_layers) /= size(target, 2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP regressor fit: layer topology does not match data")
                if (present(state)) state = candidate_state
                return
            end if
            call candidate%initialize(options%layer_sizes, status, &
                hidden_activation=options%hidden_activation, &
                initialization_seed=options%initialization_seed)
        else
            layers = [size(x, 2), size(target, 2)]
            call candidate%initialize(layers, status, &
                hidden_activation=options%hidden_activation, &
                initialization_seed=options%initialization_seed)
        end if
        if (.not. status_ok(status)) then
            if (present(state)) state = candidate_state
            return
        end if

        if (options%use_lbfgsb) then
            call mlp_optimize_lbfgsb(candidate, x, target, options%lbfgsb, &
                candidate_state%lbfgsb, status)
            candidate_state%used_lbfgsb = .true.
            candidate_state%converged = status_ok(status)
        else
            if (present(validation_x)) then
                if (present(checkpoint)) then
                    call mlp_train(candidate, x, target, status, options%training, &
                        candidate_state%training, validation_x, validation_target, checkpoint)
                else
                    call mlp_train(candidate, x, target, status, options%training, &
                        candidate_state%training, validation_x, validation_target)
                end if
            else if (present(checkpoint)) then
                call mlp_train(candidate, x, target, status, options%training, &
                    candidate_state%training, checkpoint=checkpoint)
            else
                call mlp_train(candidate, x, target, status, options%training, &
                    candidate_state%training)
            end if
            candidate_state%converged = candidate_state%training%converged .or. &
                status_ok(status)
        end if
        if (.not. status_ok(status)) then
            if (present(state)) state = candidate_state
            return
        end if
        self%model = candidate
        self%n_features = size(x, 2)
        self%n_outputs = size(target, 2)
        self%fit_state = candidate_state
        self%fitted_value = .true.
        if (present(checkpoint)) self%checkpoint_state = checkpoint
        if (present(state)) state = self%fit_state
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_regressor_fit

    subroutine mlp_regressor_predict(self, x, values, status)
        class(mlp_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        values = 0.0_dp
        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%n_features .or. &
            any(shape(values) /= [size(x, 1), self%n_outputs]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP regressor predict: model or array shape is invalid")
            return
        end if
        call self%model%predict(x, values, status)
    end subroutine mlp_regressor_predict

    subroutine mlp_regressor_predict_device(self, device, x, values, status)
        class(mlp_regressor_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP regressor device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, values, status)
        case (FORTML_DEVICE_CUDA)
            values = 0.0_dp
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP regressor device: resident CUDA trainer is not implemented")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP regressor device: device kind is invalid")
        end select
    end subroutine mlp_regressor_predict_device

    subroutine mlp_regressor_predict_jvp(self, x, theta_dot, x_dot, values, &
            values_dot, status)
        class(mlp_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: values(:, :), values_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        values = 0.0_dp
        values_dot = 0.0_dp
        if (.not. self%fitted_value .or. size(x, 2) /= self%n_features .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(values) /= [size(x, 1), self%n_outputs]) .or. &
            any(shape(values_dot) /= shape(values)) .or. &
            size(theta_dot) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP regressor JVP: model or tangent shape is invalid")
            return
        end if
        call self%model%jvp(x, theta_dot, x_dot, values, values_dot, status)
    end subroutine mlp_regressor_predict_jvp

    subroutine mlp_regressor_predict_vjp(self, x, values_bar, theta_bar, x_bar, status)
        class(mlp_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), values_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%fitted_value .or. size(x, 2) /= self%n_features .or. &
            any(shape(values_bar) /= [size(x, 1), self%n_outputs]) .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x)) .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(values_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP regressor VJP: model or cotangent shape is invalid")
            return
        end if
        call self%model%vjp(x, values_bar, theta_bar, x_bar, status)
    end subroutine mlp_regressor_predict_vjp

    subroutine mlp_regressor_loss_gradient(self, x, target, l2, value, gradient, &
            l2_gradient, status, sample_weight)
        class(mlp_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), target(:, :), l2
        real(dp), intent(out) :: value, gradient(:), l2_gradient
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP regressor loss: model is not fitted")
            return
        end if
        call mlp_loss_value_gradient(self%model, x, target, l2, value, gradient, &
            l2_gradient, status, sample_weight=sample_weight)
    end subroutine mlp_regressor_loss_gradient

    subroutine mlp_regressor_loss_hvp(self, x, target, l2, theta_dot, l2_direction, &
            parameter_hvp, l2_hvp, status, sample_weight)
        class(mlp_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), target(:, :), l2, theta_dot(:), l2_direction
        real(dp), intent(out) :: parameter_hvp(:), l2_hvp
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP regressor HVP: model is not fitted")
            return
        end if
        call mlp_loss_hvp(self%model, x, target, l2, theta_dot, l2_direction, &
            parameter_hvp, l2_hvp, status, sample_weight=sample_weight)
    end subroutine mlp_regressor_loss_hvp

    function mlp_regressor_parameters(self) result(parameters)
        class(mlp_regressor_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        if (self%fitted_value) then
            parameters = self%model%parameters()
        else
            allocate(parameters(0))
        end if
    end function mlp_regressor_parameters

    subroutine mlp_regressor_set_parameters(self, parameters, status)
        class(mlp_regressor_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. self%fitted_value .or. size(parameters) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP regressor set_parameters: model or parameter shape is invalid")
            return
        end if
        call self%model%set_parameters(parameters, status)
    end subroutine mlp_regressor_set_parameters

    integer function mlp_regressor_parameter_count(self) result(count)
        class(mlp_regressor_t), intent(in) :: self
        count = 0
        if (self%fitted_value) count = self%model%parameter_count()
    end function mlp_regressor_parameter_count

    integer function mlp_regressor_feature_count(self) result(count)
        class(mlp_regressor_t), intent(in) :: self
        count = self%n_features
    end function mlp_regressor_feature_count

    integer function mlp_regressor_output_count(self) result(count)
        class(mlp_regressor_t), intent(in) :: self
        count = self%n_outputs
    end function mlp_regressor_output_count

    function mlp_regressor_state(self) result(value)
        class(mlp_regressor_t), intent(in) :: self
        type(mlp_regressor_state_t) :: value
        value = self%fit_state
    end function mlp_regressor_state

    function mlp_regressor_checkpoint(self) result(value)
        class(mlp_regressor_t), intent(in) :: self
        type(mlp_training_checkpoint_t) :: value
        value = self%checkpoint_state
    end function mlp_regressor_checkpoint

    logical function mlp_regressor_fitted(self) result(value)
        class(mlp_regressor_t), intent(in) :: self
        value = self%fitted_value
    end function mlp_regressor_fitted

    logical function mlp_regressor_device_supported(self, device_kind) result(value)
        class(mlp_regressor_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = self%fitted_value .and. device_kind == FORTML_DEVICE_CPU
    end function mlp_regressor_device_supported

end module fortml_mlp_regressor
