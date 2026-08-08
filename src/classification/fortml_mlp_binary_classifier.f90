module fortml_mlp_binary_classifier
    !! Differentiable binary neural classifier with a single sigmoid head.
    !!
    !! Samples are rows and features are columns.  The final MLP output is
    !! one logit; probabilities are evaluated with the stable sigmoid from
    !! `fortml_losses`.  Fit uses deterministic Adam and supports sample and
    !! class weights, minibatches, early stopping, and L2 regularisation.
    !! Scores and probabilities expose exact parameter/input JVP and VJP
    !! products.  CUDA requests are typed refusals until a resident MLP kernel
    !! is linked; no hidden host fallback is used.
    use, intrinsic :: iso_fortran_env, only: int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_losses, only: stable_sigmoid, &
        multilabel_binary_cross_entropy_with_logits_value, &
        multilabel_binary_cross_entropy_with_logits_vjp, &
        multilabel_binary_cross_entropy_with_logits_hvp
    use fortml_mlp, only: mlp_t, MLP_TANH
    use fortopt_adam, only: adam_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: mlp_binary_classifier_options_t
        integer :: max_epochs = 1000
        integer :: batch_size = 0
        integer :: patience = 0
        integer :: shuffle_seed = 17
        integer :: initialization_seed = 17
        integer :: hidden_activation = MLP_TANH
        logical :: shuffle = .false.
        logical :: restore_best = .true.
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: tolerance = 1.0e-6_dp
        real(dp) :: min_delta = 0.0_dp
    end type mlp_binary_classifier_options_t

    type, public :: mlp_binary_classifier_state_t
        integer :: epochs = 0
        integer :: updates = 0
        integer :: best_epoch = 0
        logical :: converged = .false.
        logical :: early_stopped = .false.
        real(dp) :: initial_loss = huge(1.0_dp)
        real(dp) :: final_loss = huge(1.0_dp)
        real(dp) :: best_loss = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp), allocatable :: loss_history(:)
    end type mlp_binary_classifier_state_t

    type, public :: mlp_binary_training_objective_t
        !! Weighted BCE objective adapter for the generic FortOpt seam.
        !!
        !! The packed variable is the fitted network parameter vector.  When
        !! `optimize_l2` is enabled, one final coordinate is the non-negative
        !! L2 coefficient.  Every evaluation updates the live classifier, so
        !! direct products and an external trainer share one objective.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: features(:, :), targets(:), weights(:)
        real(dp) :: l2 = 0.0_dp
        logical :: optimize_l2 = .false.
    contains
        procedure, public :: initialize => mlp_binary_objective_initialize
        procedure, public :: parameter_count => mlp_binary_objective_parameter_count
        procedure, public :: parameters => mlp_binary_objective_parameters
        procedure, public :: value_gradient => mlp_binary_objective_value_gradient
        procedure, public :: jvp => mlp_binary_objective_jvp
        procedure, public :: vjp => mlp_binary_objective_vjp
        procedure, public :: hvp => mlp_binary_objective_hvp
        procedure, public :: fortopt => mlp_binary_objective_fortopt
    end type mlp_binary_training_objective_t

    type, public :: mlp_binary_lbfgsb_options_t
        !! Bounds and convergence controls for binary MLP L-BFGS-B.
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: l2_lower_bound = 0.0_dp
        real(dp) :: l2_upper_bound = 20.0_dp
        logical :: optimize_l2 = .false.
    end type mlp_binary_lbfgsb_options_t

    type, public :: mlp_binary_lbfgsb_result_t
        !! Diagnostics returned by `mlp_binary_optimize_lbfgsb`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: l2 = 0.0_dp
    end type mlp_binary_lbfgsb_result_t

    type, public :: mlp_binary_classifier_t
        private
        type(mlp_t) :: logits
        integer, allocatable :: class_label(:)
    contains
        procedure, public :: fit => mlp_binary_classifier_fit
        procedure, public :: decision_function => mlp_binary_decision
        procedure, public :: decision_function_device => &
            mlp_binary_decision_device
        procedure, public :: decision_function_jvp => mlp_binary_decision_jvp
        procedure, public :: decision_function_vjp => mlp_binary_decision_vjp
        procedure, public :: predict_proba => mlp_binary_predict_proba
        procedure, public :: predict_proba_device => &
            mlp_binary_predict_proba_device
        procedure, public :: predict_proba_jvp => mlp_binary_predict_proba_jvp
        procedure, public :: predict_proba_vjp => mlp_binary_predict_proba_vjp
        procedure, public :: predict => mlp_binary_predict
        procedure, public :: predict_device => mlp_binary_predict_device
        procedure, public :: classes => mlp_binary_classes
        procedure, public :: feature_count => mlp_binary_feature_count
        procedure, public :: parameter_count => mlp_binary_parameter_count
        procedure, public :: parameters => mlp_binary_parameters
        procedure, public :: set_parameters => mlp_binary_set_parameters
        procedure, public :: loss_gradient => mlp_binary_loss_gradient_model
        procedure, public :: loss_hvp => mlp_binary_loss_hvp_model
        procedure, public :: fitted => mlp_binary_fitted
        procedure, public :: device_supported => mlp_binary_device_supported
    end type mlp_binary_classifier_t

    public :: mlp_binary_classifier_fit
    public :: mlp_binary_decision
    public :: mlp_binary_decision_device
    public :: mlp_binary_decision_jvp
    public :: mlp_binary_decision_vjp
    public :: mlp_binary_predict_proba
    public :: mlp_binary_predict_proba_device
    public :: mlp_binary_predict_proba_jvp
    public :: mlp_binary_predict_proba_vjp
    public :: mlp_binary_predict
    public :: mlp_binary_predict_device
    public :: mlp_binary_optimize_lbfgsb

contains

    subroutine mlp_binary_objective_initialize(self, model, x, labels, l2, &
            status, optimize_l2, sample_weight, class_weight)
        !! Initialize a weighted sigmoid-head objective around a fitted model.
        class(mlp_binary_training_objective_t), intent(out) :: self
        class(mlp_binary_classifier_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), l2
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: optimize_l2
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        integer, allocatable :: encoded(:), classes(:)
        real(dp), allocatable :: effective_weight(:)
        real(dp) :: class_factors(2), weight_sum
        integer :: i

        self%l2 = 0.0_dp
        self%optimize_l2 = .false.
        if (present(optimize_l2)) self%optimize_l2 = optimize_l2
        if (.not. model%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective: model is not fitted")
            return
        end if
        if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp .or. &
            size(x, 1) < 1 .or. size(x, 2) /= model%feature_count() .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective: model, data, or L2 value is invalid")
            return
        end if
        classes = model%classes()
        if (size(classes) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective: fitted model has no binary classes")
            return
        end if
        call encode_model_labels(labels, classes, encoded, status)
        if (.not. status_ok(status)) return

        allocate(effective_weight(size(labels)))
        effective_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP binary objective: sample weights are invalid")
                return
            end if
            effective_weight = sample_weight
        end if
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= 2 .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP binary objective: class weights are invalid")
                return
            end if
            class_factors = class_weight
        end if
        do i = 1, size(effective_weight)
            effective_weight(i) = effective_weight(i)*class_factors(encoded(i) + 1)
        end do
        weight_sum = sum(effective_weight)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective: effective weights have no positive mass")
            return
        end if

        self%model => model%logits
        allocate(self%features, source=x)
        allocate(self%targets(size(encoded)), self%weights(size(encoded)))
        self%targets = real(encoded, dp)
        self%weights = effective_weight
        self%l2 = l2
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_objective_initialize

    integer function mlp_binary_objective_parameter_count(self) result(count)
        class(mlp_binary_training_objective_t), intent(in) :: self

        count = 0
        if (.not. associated(self%model)) return
        count = self%model%parameter_count()
        if (self%optimize_l2) count = count + 1
    end function mlp_binary_objective_parameter_count

    function mlp_binary_objective_parameters(self) result(parameters)
        class(mlp_binary_training_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: n_model

        allocate(parameters(self%parameter_count()))
        parameters = 0.0_dp
        if (.not. associated(self%model)) return
        n_model = self%model%parameter_count()
        parameters(:n_model) = self%model%parameters()
        if (self%optimize_l2) parameters(n_model + 1) = self%l2
    end function mlp_binary_objective_parameters

    subroutine mlp_binary_objective_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_binary_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: l2
        integer :: n_model

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective: parameter or gradient shape is invalid")
            return
        end if
        l2 = self%l2
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP binary objective: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        call mlp_binary_loss_gradient(self%model, self%features, self%targets, &
            l2, value, gradient(:n_model), status, self%weights)
        if (.not. status_ok(status)) return
        if (self%optimize_l2) gradient(n_model + 1) = &
            0.5_dp*sum(parameters(:n_model)*parameters(:n_model))
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_objective_value_gradient

    subroutine mlp_binary_objective_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_binary_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective JVP: direction shape or values are invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective JVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_objective_jvp

    subroutine mlp_binary_objective_vjp(self, parameters, output_bar, gradient, &
            status)
        class(mlp_binary_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective VJP: output cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        gradient = output_bar*gradient
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_objective_vjp

    subroutine mlp_binary_objective_hvp(self, parameters, direction, product, &
            status)
        class(mlp_binary_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta(:), theta_hvp(:)
        real(dp) :: l2, l2_direction
        integer :: n_model

        product = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective HVP: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
            size(direction) /= size(parameters) .or. &
            size(product) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective HVP: parameter or direction shape is invalid")
            return
        end if
        l2 = self%l2
        l2_direction = 0.0_dp
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            l2_direction = direction(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP binary objective HVP: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        allocate(theta(n_model), theta_hvp(n_model))
        theta = parameters(:n_model)
        call mlp_binary_loss_hvp(self%model, self%features, self%targets, l2, &
            direction(:n_model), theta_hvp, status, self%weights)
        if (.not. status_ok(status)) return
        product(:n_model) = theta_hvp
        if (self%optimize_l2) then
            product(:n_model) = product(:n_model) + l2_direction*theta
            product(n_model + 1) = dot_product(theta, direction(:n_model))
        end if
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_objective_hvp

    subroutine mlp_binary_objective_fortopt(self, objective, status)
        class(mlp_binary_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            mlp_binary_objective_context_callback, status)
    end subroutine mlp_binary_objective_fortopt

    subroutine mlp_binary_objective_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_binary_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary objective: context has the wrong type")
        end select
    end subroutine mlp_binary_objective_context_callback

    subroutine mlp_binary_optimize_lbfgsb(model, x, labels, options, result, status, &
            sample_weight, class_weight)
        !! Optimize a weighted binary MLP BCE objective with FortOpt L-BFGS-B.
        class(mlp_binary_classifier_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(mlp_binary_lbfgsb_options_t), intent(in) :: options
        type(mlp_binary_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        type(mlp_binary_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_model, n_parameters

        result = mlp_binary_lbfgsb_result_t()
        if (.not. valid_lbfgsb_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary L-BFGS-B: options are invalid")
            return
        end if
        call adapter%initialize(model, x, labels, options%l2, status, &
            optimize_l2=options%optimize_l2, sample_weight=sample_weight, &
            class_weight=class_weight)
        if (.not. status_ok(status)) return
        n_model = model%parameter_count()
        n_parameters = adapter%parameter_count()
        parameters = adapter%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower(:n_model) = options%lower_bound
        upper(:n_model) = options%upper_bound
        if (options%optimize_l2) then
            lower(n_model + 1) = options%l2_lower_bound
            upper(n_model + 1) = options%l2_upper_bound
            parameters(n_model + 1) = min(max(options%l2, lower(n_model + 1)), &
                upper(n_model + 1))
        end if
        call adapter%fortopt(objective, status)
        if (.not. status_ok(status)) return
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (.not. status_ok(status)) return
        call model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) return
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = sqrt(sum(gradient*gradient))
        result%l2 = options%l2
        if (options%optimize_l2) result%l2 = parameters(n_model + 1)
        if (.not. ieee_is_finite(result%objective) .or. &
            .not. ieee_is_finite(result%gradient_norm) .or. &
            .not. ieee_is_finite(result%l2)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP binary L-BFGS-B: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP binary L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_optimize_lbfgsb

    subroutine mlp_binary_classifier_fit(self, x, labels, status, &
            hidden_layer_sizes, options, state, sample_weight, class_weight)
        class(mlp_binary_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_layer_sizes(:)
        type(mlp_binary_classifier_options_t), intent(in), optional :: options
        type(mlp_binary_classifier_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:)
        type(mlp_binary_classifier_options_t) :: config
        type(mlp_binary_classifier_state_t) :: result
        type(adam_t) :: optimizer
        integer, allocatable :: layer_sizes(:), order(:), encoded(:)
        integer, allocatable :: x_indices(:)
        real(dp), allocatable :: theta(:), best_theta(:), gradient(:)
        real(dp), allocatable :: x_batch(:, :), batch_targets(:), batch_weights(:)
        real(dp), allocatable :: effective_weight(:), class_factors(:)
        real(dp) :: loss, gradient_norm, best_loss, improvement
        integer :: n_samples, n_features, n_hidden, batch, first, last
        integer :: n_batch, epoch, stale_epochs, i, negative_label, positive_label
        integer(int64) :: shuffle_state

        if (present(options)) config = options
        if (.not. valid_options(config)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary classifier fit: invalid optimizer options")
            if (present(state)) state = result
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary classifier fit: input dimensions or values are invalid")
            if (present(state)) state = result
            return
        end if

        negative_label = minval(labels)
        positive_label = maxval(labels)
        if (negative_label == positive_label) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary classifier fit: two distinct classes are required")
            if (present(state)) state = result
            return
        end if
        do i = 1, size(labels)
            if (labels(i) /= negative_label .and. labels(i) /= positive_label) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP binary classifier fit: labels must contain exactly two classes")
                if (present(state)) state = result
                return
            end if
        end do

        allocate(encoded(size(labels)), effective_weight(size(labels)), &
            class_factors(2))
        encoded = 0
        where (labels == positive_label) encoded = 1
        effective_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP binary classifier fit: sample weights must be finite and nonnegative")
                if (present(state)) state = result
                return
            end if
            effective_weight = sample_weight
        end if
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= 2 .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP binary classifier fit: class weights must be positive and finite")
                if (present(state)) state = result
                return
            end if
            class_factors = class_weight
        end if
        do i = 1, size(encoded)
            effective_weight(i) = effective_weight(i)*class_factors(encoded(i) + 1)
        end do
        if (.not. ieee_is_finite(sum(effective_weight)) .or. &
            sum(effective_weight) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary classifier fit: effective weights have no positive mass")
            if (present(state)) state = result
            return
        end if

        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_hidden = 0
        if (present(hidden_layer_sizes)) then
            n_hidden = size(hidden_layer_sizes)
            if (n_hidden > 0 .and. any(hidden_layer_sizes < 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP binary classifier fit: hidden layer sizes must be positive")
                if (present(state)) state = result
                return
            end if
        end if
        allocate(layer_sizes(n_hidden + 2))
        layer_sizes(1) = n_features
        if (n_hidden > 0) layer_sizes(2:n_hidden + 1) = hidden_layer_sizes
        layer_sizes(n_hidden + 2) = 1
        call self%logits%initialize(layer_sizes, status, &
            hidden_activation=config%hidden_activation, initialization_seed=&
            config%initialization_seed)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        allocate(self%class_label(2))
        self%class_label = [negative_label, positive_label]

        theta = self%logits%parameters()
        ! Two statements: a sourced ALLOCATE names exactly one object, so mixing
        ! a sourced and an unsourced allocation is invalid. gfortran accepted
        ! it; nvfortran rejects it, which is what surfaced this.
        allocate(best_theta, source=theta)
        allocate(gradient(size(theta)))
        allocate(order(n_samples), x_indices(n_samples))
        allocate(result%loss_history(config%max_epochs))
        call mlp_binary_loss_gradient(self%logits, x, real(encoded, dp), config%l2, loss, &
            gradient, status, effective_weight)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        result%initial_loss = loss
        result%best_loss = loss
        best_loss = loss
        stale_epochs = 0
        shuffle_state = int(config%shuffle_seed, int64)
        if (shuffle_state <= 0_int64) shuffle_state = 1_int64
        batch = config%batch_size
        if (batch == 0) batch = n_samples
        batch = min(batch, n_samples)
        call optimizer%initialize(size(theta), status, &
            learning_rate=config%learning_rate, beta1=config%beta1, &
            beta2=config%beta2, epsilon=config%epsilon)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if

        do epoch = 1, config%max_epochs
            order = [(i, i=1, n_samples)]
            if (config%shuffle) call shuffle_order(order, shuffle_state)
            do first = 1, n_samples, batch
                last = min(first + batch - 1, n_samples)
                n_batch = last - first + 1
                allocate(x_batch(n_batch, n_features), batch_targets(n_batch), &
                    batch_weights(n_batch))
                x_indices = order
                x_batch = x(x_indices(first:last), :)
                batch_targets = real(encoded(x_indices(first:last)), dp)
                batch_weights = effective_weight(x_indices(first:last))
                if (sum(batch_weights) <= 0.0_dp) then
                    deallocate(x_batch, batch_targets, batch_weights)
                    cycle
                end if
                call mlp_binary_loss_gradient(self%logits, x_batch, batch_targets, &
                    config%l2, loss, gradient, status, batch_weights)
                deallocate(x_batch, batch_targets, batch_weights)
                if (.not. status_ok(status)) then
                    if (present(state)) state = result
                    return
                end if
                call optimizer%step(theta, gradient, status)
                if (.not. status_ok(status)) then
                    if (present(state)) state = result
                    return
                end if
                call self%logits%set_parameters(theta, status)
                if (.not. status_ok(status)) then
                    if (present(state)) state = result
                    return
                end if
                result%updates = result%updates + 1
            end do

            call mlp_binary_loss_gradient(self%logits, x, real(encoded, dp), config%l2, &
                loss, gradient, status, effective_weight)
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
            gradient_norm = sqrt(sum(gradient*gradient))
            result%epochs = epoch
            result%loss_history(epoch) = loss
            result%gradient_norm = gradient_norm
            improvement = best_loss - loss
            if (improvement > config%min_delta .or. epoch == 1) then
                best_loss = loss
                result%best_loss = loss
                result%best_epoch = epoch
                best_theta = theta
                stale_epochs = 0
            else
                stale_epochs = stale_epochs + 1
            end if
            if (gradient_norm <= config%tolerance) then
                result%converged = .true.
                exit
            end if
            if (config%patience > 0 .and. stale_epochs >= config%patience) then
                result%early_stopped = .true.
                exit
            end if
        end do

        call shrink_history(result%loss_history, result%epochs)
        if (config%restore_best .and. result%best_epoch > 0 .and. &
            result%best_epoch < result%epochs) then
            theta = best_theta
            call self%logits%set_parameters(theta, status)
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
            call mlp_binary_loss_gradient(self%logits, x, real(encoded, dp), config%l2, &
                loss, gradient, status, effective_weight)
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
            result%gradient_norm = sqrt(sum(gradient*gradient))
        end if
        result%final_loss = loss
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_classifier_fit

    subroutine mlp_binary_loss_gradient(model, x, targets, l2, value, gradient, &
            status, sample_weight)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), targets(:), l2
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: logits(:, :), target_matrix(:, :), logits_bar(:, :)
        real(dp), allocatable :: x_bar(:, :), theta(:)

        value = 0.0_dp
        gradient = 0.0_dp
        if (l2 < 0.0_dp .or. .not. ieee_is_finite(l2) .or. &
            size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(targets) /= size(x, 1) .or. &
            size(gradient) /= model%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary loss: model, data, or gradient shape is invalid")
            return
        end if
        allocate(logits(size(x, 1), 1), target_matrix(size(x, 1), 1), &
            logits_bar(size(x, 1), 1), x_bar(size(x, 1), size(x, 2)))
        target_matrix(:, 1) = targets
        call model%predict(x, logits, status)
        if (.not. status_ok(status)) return
        if (present(sample_weight)) then
            call multilabel_binary_cross_entropy_with_logits_value(logits, &
                target_matrix, value, status, sample_weight=sample_weight)
        else
            call multilabel_binary_cross_entropy_with_logits_value(logits, &
                target_matrix, value, status)
        end if
        if (.not. status_ok(status)) return
        if (present(sample_weight)) then
            call multilabel_binary_cross_entropy_with_logits_vjp(logits, &
                target_matrix, 1.0_dp, logits_bar, status, sample_weight=sample_weight)
        else
            call multilabel_binary_cross_entropy_with_logits_vjp(logits, &
                target_matrix, 1.0_dp, logits_bar, status)
        end if
        if (.not. status_ok(status)) return
        call model%vjp(x, logits_bar, gradient, x_bar, status)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        value = value + 0.5_dp*l2*sum(theta*theta)
        gradient = gradient + l2*theta
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary loss: objective or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_loss_gradient

    subroutine mlp_binary_loss_hvp(model, x, targets, l2, theta_dot, hvp, status, &
            sample_weight)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), targets(:), l2, theta_dot(:)
        real(dp), intent(out) :: hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: logits(:, :), target_matrix(:, :), logits_bar(:, :)
        real(dp), allocatable :: logits_dot(:, :), logits_hvp(:, :), x_bar(:, :)
        real(dp), allocatable :: x_zero(:, :), x_hvp(:, :), direct_hvp(:), theta(:)

        hvp = 0.0_dp
        if (l2 < 0.0_dp .or. .not. ieee_is_finite(l2) .or. &
            size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(targets) /= size(x, 1) .or. &
            size(theta_dot) /= model%parameter_count() .or. &
            size(hvp) /= model%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary HVP: model, data, or direction shape is invalid")
            return
        end if
        allocate(logits(size(x, 1), 1), target_matrix(size(x, 1), 1), &
            logits_bar(size(x, 1), 1), logits_dot(size(x, 1), 1), &
            logits_hvp(size(x, 1), 1), x_zero(size(x, 1), size(x, 2)), &
            x_bar(size(x, 1), size(x, 2)), x_hvp(size(x, 1), size(x, 2)), &
            direct_hvp(size(theta_dot)))
        x_zero = 0.0_dp
        target_matrix(:, 1) = targets
        call model%predict(x, logits, status)
        if (.not. status_ok(status)) return
        if (present(sample_weight)) then
            call multilabel_binary_cross_entropy_with_logits_vjp(logits, &
                target_matrix, 1.0_dp, logits_bar, status, sample_weight=sample_weight)
        else
            call multilabel_binary_cross_entropy_with_logits_vjp(logits, &
                target_matrix, 1.0_dp, logits_bar, status)
        end if
        if (.not. status_ok(status)) return
        call model%jvp(x, theta_dot, x_zero, logits, logits_dot, status)
        if (.not. status_ok(status)) return
        if (present(sample_weight)) then
            call multilabel_binary_cross_entropy_with_logits_hvp(logits, &
                target_matrix, logits_dot, logits_hvp, status, sample_weight=sample_weight)
        else
            call multilabel_binary_cross_entropy_with_logits_hvp(logits, &
                target_matrix, logits_dot, logits_hvp, status)
        end if
        if (.not. status_ok(status)) return
        call model%hvp(x, logits_bar, theta_dot, x_zero, direct_hvp, x_hvp, status)
        if (.not. status_ok(status)) return
        call model%vjp(x, logits_hvp, hvp, x_bar, status)
        if (.not. status_ok(status)) return
        hvp = hvp + direct_hvp
        theta = model%parameters()
        hvp = hvp + l2*theta_dot
        if (any(.not. ieee_is_finite(hvp))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_loss_hvp

    subroutine mlp_binary_loss_gradient_model(self, x, labels, l2, value, &
            gradient, status, sample_weight)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), l2
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: encoded(:)

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary loss_gradient: model is not fitted")
            return
        end if
        call encode_model_labels(labels, self%class_label, encoded, status)
        if (.not. status_ok(status)) return
        if (present(sample_weight)) then
            call mlp_binary_loss_gradient(self%logits, x, real(encoded, dp), l2, &
                value, gradient, status, sample_weight)
        else
            call mlp_binary_loss_gradient(self%logits, x, real(encoded, dp), l2, &
                value, gradient, status)
        end if
    end subroutine mlp_binary_loss_gradient_model

    subroutine mlp_binary_loss_hvp_model(self, x, labels, l2, theta_dot, hvp, &
            status, sample_weight)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), l2, theta_dot(:)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: encoded(:)

        hvp = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary loss_hvp: model is not fitted")
            return
        end if
        call encode_model_labels(labels, self%class_label, encoded, status)
        if (.not. status_ok(status)) return
        if (present(sample_weight)) then
            call mlp_binary_loss_hvp(self%logits, x, real(encoded, dp), l2, theta_dot, &
                hvp, status, sample_weight)
        else
            call mlp_binary_loss_hvp(self%logits, x, real(encoded, dp), l2, theta_dot, &
                hvp, status)
        end if
    end subroutine mlp_binary_loss_hvp_model

    subroutine mlp_binary_decision(self, x, scores, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: logits(:, :)

        scores = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary decision_function: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
            size(scores) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary decision_function: input or output shape is invalid")
            return
        end if
        allocate(logits(size(x, 1), 1))
        call self%logits%predict(x, logits, status)
        if (status_ok(status)) scores = logits(:, 1)
    end subroutine mlp_binary_decision

    subroutine mlp_binary_decision_device(self, device, x, scores, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP binary device decision: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary device decision: device kind is invalid")
        end select
    end subroutine mlp_binary_decision_device

    subroutine mlp_binary_decision_jvp(self, x, theta_dot, x_dot, scores, &
            scores_dot, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:), scores_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: logits(:, :), logits_dot(:, :)

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary decision JVP: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
            any(shape(x_dot) /= shape(x)) .or. size(scores) /= size(x, 1) .or. &
            size(scores_dot) /= size(scores) .or. &
            size(theta_dot) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary decision JVP: model, tangent, or output shape is invalid")
            return
        end if
        allocate(logits(size(x, 1), 1), logits_dot(size(x, 1), 1))
        call self%logits%jvp(x, theta_dot, x_dot, logits, logits_dot, status)
        if (status_ok(status)) then
            scores = logits(:, 1)
            scores_dot = logits_dot(:, 1)
        end if
    end subroutine mlp_binary_decision_jvp

    subroutine mlp_binary_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: score_matrix(:, :)

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary decision VJP: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
            size(scores_bar) /= size(x, 1) .or. size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x)) .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary decision VJP: model, cotangent, or output shape is invalid")
            return
        end if
        allocate(score_matrix(size(x, 1), 1))
        score_matrix(:, 1) = scores_bar
        call self%logits%vjp(x, score_matrix, theta_bar, x_bar, status)
    end subroutine mlp_binary_decision_vjp

    subroutine mlp_binary_predict_proba(self, x, probabilities, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)

        probabilities = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary predict_proba: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), 2])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary predict_proba: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        probabilities(:, 2) = stable_sigmoid(scores)
        probabilities(:, 1) = 1.0_dp - probabilities(:, 2)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_predict_proba

    subroutine mlp_binary_predict_proba_device(self, device, x, probabilities, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP binary device probability: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary device probability: device kind is invalid")
        end select
    end subroutine mlp_binary_predict_proba_device

    subroutine mlp_binary_predict_proba_jvp(self, x, theta_dot, x_dot, &
            probabilities, probabilities_dot, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:), scores_dot(:), positive(:)

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (any(shape(probabilities) /= [size(x, 1), 2]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary probability JVP: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)), scores_dot(size(x, 1)))
        call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        if (.not. status_ok(status)) return
        allocate(positive(size(x, 1)))
        positive = stable_sigmoid(scores)
        probabilities(:, 2) = positive
        probabilities(:, 1) = 1.0_dp - positive
        probabilities_dot(:, 2) = positive*(1.0_dp - positive)*scores_dot
        probabilities_dot(:, 1) = -probabilities_dot(:, 2)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_predict_proba_jvp

    subroutine mlp_binary_predict_proba_vjp(self, x, probabilities_bar, theta_bar, &
            x_bar, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:), score_bar(:), probability(:)

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (any(shape(probabilities_bar) /= [size(x, 1), 2])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary probability VJP: cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary probability VJP: cotangent must be finite")
            return
        end if
        allocate(scores(size(x, 1)), score_bar(size(x, 1)), probability(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        probability = stable_sigmoid(scores)
        score_bar = (probabilities_bar(:, 2) - probabilities_bar(:, 1))* &
            probability*(1.0_dp - probability)
        call self%decision_function_vjp(x, score_bar, theta_bar, x_bar, status)
    end subroutine mlp_binary_predict_proba_vjp

    subroutine mlp_binary_predict(self, x, labels, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)

        labels = 0
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary predict: model is not fitted")
            return
        end if
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary predict: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        where (scores >= 0.0_dp)
            labels = self%class_label(2)
        elsewhere
            labels = self%class_label(1)
        end where
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_binary_predict

    subroutine mlp_binary_predict_device(self, device, x, labels, status)
        class(mlp_binary_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP binary device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary device prediction: device kind is invalid")
        end select
    end subroutine mlp_binary_predict_device

    function mlp_binary_classes(self) result(classes)
        class(mlp_binary_classifier_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function mlp_binary_classes

    integer function mlp_binary_feature_count(self) result(count)
        class(mlp_binary_classifier_t), intent(in) :: self

        count = 0
        if (self%fitted()) count = self%logits%layer_sizes(1)
    end function mlp_binary_feature_count

    integer function mlp_binary_parameter_count(self) result(count)
        class(mlp_binary_classifier_t), intent(in) :: self

        count = 0
        if (self%fitted()) count = self%logits%parameter_count()
    end function mlp_binary_parameter_count

    function mlp_binary_parameters(self) result(values)
        class(mlp_binary_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (self%fitted()) then
            values = self%logits%parameters()
        else
            allocate(values(0))
        end if
    end function mlp_binary_parameters

    subroutine mlp_binary_set_parameters(self, values, status)
        class(mlp_binary_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary set_parameters: model is not fitted")
            return
        end if
        call self%logits%set_parameters(values, status)
    end subroutine mlp_binary_set_parameters

    logical function mlp_binary_fitted(self) result(is_fitted)
        class(mlp_binary_classifier_t), intent(in) :: self

        is_fitted = allocated(self%class_label)
        if (.not. is_fitted) return
        is_fitted = size(self%class_label) == 2 .and. &
            self%logits%parameter_count() > 0
    end function mlp_binary_fitted

    logical function mlp_binary_device_supported(self, device_kind) result(supported)
        class(mlp_binary_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%fitted()
        case default
            supported = .false.
        end select
    end function mlp_binary_device_supported

    logical function valid_options(options) result(valid)
        type(mlp_binary_classifier_options_t), intent(in) :: options

        valid = options%max_epochs >= 1 .and. options%batch_size >= 0 .and. &
            options%patience >= 0 .and. options%learning_rate > 0.0_dp .and. &
            options%beta1 >= 0.0_dp .and. options%beta1 < 1.0_dp .and. &
            options%beta2 >= 0.0_dp .and. options%beta2 < 1.0_dp .and. &
            options%epsilon > 0.0_dp .and. options%l2 >= 0.0_dp .and. &
            options%tolerance >= 0.0_dp .and. options%min_delta >= 0.0_dp .and. &
            options%initialization_seed >= 0
        if (options%shuffle) valid = valid .and. options%shuffle_seed > 0
        valid = valid .and. ieee_is_finite(options%learning_rate) .and. &
            ieee_is_finite(options%beta1) .and. ieee_is_finite(options%beta2) .and. &
            ieee_is_finite(options%epsilon) .and. ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%tolerance) .and. ieee_is_finite(options%min_delta)
    end function valid_options

    logical function valid_lbfgsb_options(options) result(valid)
        type(mlp_binary_lbfgsb_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. options%lower_bound <= &
            options%upper_bound .and. options%l2_lower_bound <= &
            options%l2_upper_bound .and. options%l2 >= 0.0_dp .and. &
            options%l2_lower_bound >= 0.0_dp
        valid = valid .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            ieee_is_finite(options%lower_bound) .and. &
            ieee_is_finite(options%upper_bound) .and. &
            ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%l2_lower_bound) .and. &
            ieee_is_finite(options%l2_upper_bound) .and. &
            options%gradient_tolerance >= 0.0_dp .and. &
            options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp
        if (options%optimize_l2) valid = valid .and. &
            options%l2 >= options%l2_lower_bound .and. &
            options%l2 <= options%l2_upper_bound
    end function valid_lbfgsb_options

    subroutine encode_model_labels(labels, classes, encoded, status)
        integer, intent(in) :: labels(:), classes(:)
        integer, allocatable, intent(out) :: encoded(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        allocate(encoded(size(labels)))
        encoded = -1
        do i = 1, size(labels)
            if (labels(i) == classes(1)) encoded(i) = 0
            if (labels(i) == classes(2)) encoded(i) = 1
        end do
        if (any(encoded < 0)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP binary labels: label is not present in fitted classes")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine encode_model_labels

    subroutine shuffle_order(order, generator)
        integer, intent(inout) :: order(:)
        integer(int64), intent(inout) :: generator
        integer :: i, j, temporary

        do i = size(order), 2, -1
            generator = mod(48271_int64*generator, 2147483647_int64)
            j = 1 + int(mod(generator, int(i, int64)))
            temporary = order(i)
            order(i) = order(j)
            order(j) = temporary
        end do
    end subroutine shuffle_order

    subroutine shrink_history(history, length)
        real(dp), allocatable, intent(inout) :: history(:)
        integer, intent(in) :: length
        real(dp), allocatable :: compact(:)

        if (size(history) == length) return
        allocate(compact(max(0, length)))
        if (length > 0) compact = history(:length)
        call move_alloc(compact, history)
    end subroutine shrink_history

end module fortml_mlp_binary_classifier
