module fortml_mlp_classifier
    !! Multiclass neural classifier backed by the differentiable `mlp_t`.
    !!
    !! Samples are rows and features are columns.  The final MLP layer emits
    !! one logit per class; probabilities and the cross-entropy objective use
    !! the stable shared implementations in `fortml_losses`.  The optimizer
    !! is deterministic Adam with an explicit, local shuffle stream.
    use, intrinsic :: iso_fortran_env, only: int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_TANH
    use fortml_losses, only: softmax_value, softmax_jvp, softmax_vjp
    use fortopt_adam, only: adam_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: mlp_classifier_options_t
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
    end type mlp_classifier_options_t

    type, public :: mlp_classifier_state_t
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
    end type mlp_classifier_state_t

    type, public :: mlp_classifier_t
        private
        type(mlp_t) :: logits
        integer, allocatable :: class_label(:)
    contains
        procedure, public :: fit => mlp_classifier_fit
        procedure, public :: decision_function => mlp_classifier_decision
        procedure, public :: decision_function_device => &
            mlp_classifier_decision_device
        procedure, public :: decision_function_jvp => mlp_classifier_decision_jvp
        procedure, public :: decision_function_vjp => mlp_classifier_decision_vjp
        procedure, public :: predict_proba => mlp_classifier_predict_proba
        procedure, public :: predict_proba_device => &
            mlp_classifier_predict_proba_device
        procedure, public :: predict_proba_jvp => mlp_classifier_predict_proba_jvp
        procedure, public :: predict_proba_vjp => mlp_classifier_predict_proba_vjp
        procedure, public :: predict => mlp_classifier_predict
        procedure, public :: predict_device => mlp_classifier_predict_device
        procedure, public :: classes => mlp_classifier_classes
        procedure, public :: feature_count => mlp_classifier_feature_count
        procedure, public :: class_count => mlp_classifier_class_count
        procedure, public :: parameter_count => mlp_classifier_parameter_count
        procedure, public :: parameters => mlp_classifier_parameters
        procedure, public :: set_parameters => mlp_classifier_set_parameters
        procedure, public :: loss_gradient => mlp_classifier_model_loss_gradient
        procedure, public :: fitted => mlp_classifier_fitted
        procedure, public :: device_supported => mlp_classifier_device_supported
    end type mlp_classifier_t

    type, public :: mlp_classifier_training_objective_t
        !! Weighted multiclass cross-entropy objective around a fitted MLP.
        !!
        !! The packed variable is the logits-network parameter vector.  With
        !! `optimize_l2`, one final non-negative coordinate is appended for the
        !! L2 coefficient.  Value/gradient, JVP, VJP, and analytic HVP products
        !! all share the same weighted objective and update the live model, so
        !! FortOpt and direct derivative checks cannot drift apart.
        private
        type(mlp_t), pointer :: model => null()
        real(dp), allocatable :: features(:, :)
        integer, allocatable :: targets(:)
        real(dp), allocatable :: weights(:)
        real(dp) :: l2 = 0.0_dp
        logical :: optimize_l2 = .false.
    contains
        procedure, public :: initialize => mlp_classifier_objective_initialize
        procedure, public :: parameter_count => &
            mlp_classifier_objective_parameter_count
        procedure, public :: parameters => mlp_classifier_objective_parameters
        procedure, public :: value_gradient => &
            mlp_classifier_objective_value_gradient
        procedure, public :: jvp => mlp_classifier_objective_jvp
        procedure, public :: vjp => mlp_classifier_objective_vjp
        procedure, public :: hvp => mlp_classifier_objective_hvp
        procedure, public :: fortopt => mlp_classifier_objective_fortopt
    end type mlp_classifier_training_objective_t

    type, public :: mlp_classifier_lbfgsb_options_t
        !! Bounds and convergence controls for multiclass MLP L-BFGS-B.
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
    end type mlp_classifier_lbfgsb_options_t

    type, public :: mlp_classifier_lbfgsb_result_t
        !! Diagnostics returned by `mlp_classifier_optimize_lbfgsb`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: l2 = 0.0_dp
    end type mlp_classifier_lbfgsb_result_t

    public :: mlp_classifier_decision_device
    public :: mlp_classifier_decision_jvp
    public :: mlp_classifier_decision_vjp
    public :: mlp_classifier_predict_proba_device
    public :: mlp_classifier_predict_proba_jvp
    public :: mlp_classifier_predict_proba_vjp
    public :: mlp_classifier_predict_device
    public :: mlp_classifier_optimize_lbfgsb

contains

    subroutine mlp_classifier_objective_initialize(self, classifier, x, labels, &
            l2, status, optimize_l2, sample_weight, class_weight)
        !! Initialize a weighted multiclass cross-entropy objective.
        class(mlp_classifier_training_objective_t), intent(out) :: self
        class(mlp_classifier_t), target, intent(inout) :: classifier
        real(dp), intent(in) :: x(:, :), l2
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: optimize_l2
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        integer, allocatable :: classes(:), encoded(:)
        real(dp), allocatable :: effective_weight(:), class_factors(:)
        real(dp) :: weight_sum
        integer :: i

        self%l2 = 0.0_dp
        self%optimize_l2 = .false.
        if (present(optimize_l2)) self%optimize_l2 = optimize_l2
        if (.not. classifier%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: model is not fitted")
            return
        end if
        if (.not. ieee_is_finite(l2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: L2 value is not finite")
            return
        end if
        if (l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: L2 value must be non-negative")
            return
        end if
        if (size(x, 1) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: feature rows are empty")
            return
        end if
        if (size(x, 2) /= classifier%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: feature shape is invalid")
            return
        end if
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: label shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: features must be finite")
            return
        end if

        classes = classifier%classes()
        call encode_labels(labels, classes, encoded, status)
        if (.not. status_ok(status)) return
        allocate(effective_weight(size(labels)))
        effective_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier objective: sample weights shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier objective: sample weights must be finite")
                return
            end if
            if (any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier objective: sample weights must be non-negative")
                return
            end if
            effective_weight = sample_weight
        end if
        allocate(class_factors(size(classes)))
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= size(classes)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier objective: class weights shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(class_weight))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier objective: class weights must be finite")
                return
            end if
            if (any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier objective: class weights must be positive")
                return
            end if
            class_factors = class_weight
        end if
        do i = 1, size(effective_weight)
            effective_weight(i) = effective_weight(i)*class_factors(encoded(i))
        end do
        weight_sum = sum(effective_weight)
        if (.not. ieee_is_finite(weight_sum)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: effective weights overflowed")
            return
        end if
        if (weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: effective weights have no positive mass")
            return
        end if

        self%model => classifier%logits
        allocate(self%features, source=x)
        allocate(self%targets, source=encoded)
        allocate(self%weights, source=effective_weight)
        self%l2 = l2
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_classifier_objective_initialize

    integer function mlp_classifier_objective_parameter_count(self) result(count)
        class(mlp_classifier_training_objective_t), intent(in) :: self

        count = 0
        if (.not. associated(self%model)) return
        count = self%model%parameter_count()
        if (self%optimize_l2) count = count + 1
    end function mlp_classifier_objective_parameter_count

    function mlp_classifier_objective_parameters(self) result(parameters)
        class(mlp_classifier_training_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: n_model

        allocate(parameters(self%parameter_count()))
        parameters = 0.0_dp
        if (.not. associated(self%model)) return
        n_model = self%model%parameter_count()
        parameters(:n_model) = self%model%parameters()
        if (self%optimize_l2) parameters(n_model + 1) = self%l2
    end function mlp_classifier_objective_parameters

    subroutine mlp_classifier_objective_value_gradient(self, parameters, value, &
            gradient, status)
        class(mlp_classifier_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: l2
        integer :: n_model

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: parameter shape is invalid")
            return
        end if
        if (size(gradient) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: gradient shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: parameters must be finite")
            return
        end if
        l2 = self%l2
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            if (l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier objective: optimized L2 must be non-negative")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return
        call mlp_classifier_loss_gradient(self%model, self%features, self%targets, &
            l2, value, gradient(:n_model), status, self%weights)
        if (.not. status_ok(status)) return
        if (self%optimize_l2) gradient(n_model + 1) = &
            0.5_dp*sum(parameters(:n_model)*parameters(:n_model))
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: value is not finite")
            return
        end if
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_classifier_objective_value_gradient

    subroutine mlp_classifier_objective_jvp(self, parameters, direction, value, &
            tangent, status)
        class(mlp_classifier_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective JVP: direction shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective JVP: direction is not finite")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective JVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_classifier_objective_jvp

    subroutine mlp_classifier_objective_vjp(self, parameters, output_bar, gradient, &
            status)
        class(mlp_classifier_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective VJP: output cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        gradient = output_bar*gradient
        if (any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_classifier_objective_vjp

    subroutine mlp_classifier_objective_hvp(self, parameters, direction, product, &
            status)
        class(mlp_classifier_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), scores_dot(:, :), probabilities(:, :)
        real(dp), allocatable :: probabilities_dot(:, :), logits_bar(:, :)
        real(dp), allocatable :: logits_bar_dot(:, :), x_dot(:, :), x_hvp(:, :)
        real(dp), allocatable :: theta_hvp(:), theta_extra(:), x_bar(:, :)
        real(dp) :: l2, l2_direction, weight_sum
        integer :: n_model, row, column

        product = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective HVP: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective HVP: parameter shape is invalid")
            return
        end if
        if (size(direction) /= size(parameters) .or. size(product) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective HVP: direction shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective HVP: parameters are not finite")
            return
        end if
        if (any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective HVP: direction is not finite")
            return
        end if
        l2 = self%l2
        l2_direction = 0.0_dp
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            l2_direction = direction(n_model + 1)
            if (l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier objective HVP: optimized L2 is negative")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (.not. status_ok(status)) return

        allocate(scores(size(self%features, 1), &
            self%model%layer_sizes(size(self%model%layer_sizes))))
        allocate(scores_dot, mold=scores)
        allocate(probabilities, mold=scores)
        allocate(probabilities_dot, mold=scores)
        allocate(logits_bar, mold=scores)
        allocate(logits_bar_dot, mold=scores)
        allocate(x_dot, mold=self%features)
        allocate(x_hvp, mold=self%features)
        allocate(x_bar, mold=self%features)
        allocate(theta_hvp(n_model), theta_extra(n_model))
        x_dot = 0.0_dp
        call self%model%predict(self%features, scores, status)
        if (.not. status_ok(status)) return
        call softmax_value(scores, probabilities, status)
        if (.not. status_ok(status)) return
        call self%model%jvp(self%features, direction(:n_model), x_dot, &
            scores, scores_dot, status)
        if (.not. status_ok(status)) return
        call softmax_jvp(scores, scores_dot, probabilities, probabilities_dot, status)
        if (.not. status_ok(status)) return
        weight_sum = sum(self%weights)
        logits_bar = 0.0_dp
        logits_bar_dot = 0.0_dp
        do row = 1, size(scores, 1)
            do column = 1, size(scores, 2)
                logits_bar(row, column) = self%weights(row)*probabilities(row, column)
                logits_bar_dot(row, column) = &
                    self%weights(row)*probabilities_dot(row, column)
            end do
            logits_bar(row, self%targets(row)) = &
                logits_bar(row, self%targets(row)) - self%weights(row)
        end do
        logits_bar = logits_bar/weight_sum
        logits_bar_dot = logits_bar_dot/weight_sum
        call self%model%hvp(self%features, logits_bar, direction(:n_model), &
            x_dot, theta_hvp, x_hvp, status)
        if (.not. status_ok(status)) return
        call self%model%vjp(self%features, logits_bar_dot, theta_extra, x_bar, status)
        if (.not. status_ok(status)) return
        product(:n_model) = theta_hvp + theta_extra + l2*direction(:n_model)
        if (self%optimize_l2) then
            product(:n_model) = product(:n_model) + l2_direction*parameters(:n_model)
            product(n_model + 1) = dot_product(parameters(:n_model), direction(:n_model))
        end if
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_classifier_objective_hvp

    subroutine mlp_classifier_objective_fortopt(self, objective, status)
        class(mlp_classifier_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            mlp_classifier_objective_context_callback, status)
    end subroutine mlp_classifier_objective_fortopt

    subroutine mlp_classifier_objective_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (mlp_classifier_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier objective: context has the wrong type")
        end select
    end subroutine mlp_classifier_objective_context_callback

    subroutine mlp_classifier_optimize_lbfgsb(model, x, labels, options, result, &
            status, sample_weight, class_weight)
        !! Optimize multiclass cross-entropy with FortOpt L-BFGS-B.
        class(mlp_classifier_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(mlp_classifier_lbfgsb_options_t), intent(in) :: options
        type(mlp_classifier_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        type(mlp_classifier_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_model, n_parameters

        result = mlp_classifier_lbfgsb_result_t()
        if (.not. valid_lbfgsb_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier L-BFGS-B: options are invalid")
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
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier L-BFGS-B: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_classifier_optimize_lbfgsb

    subroutine mlp_classifier_fit(self, x, labels, status, hidden_layer_sizes, &
            options, state, sample_weight, class_weight)
        class(mlp_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_layer_sizes(:)
        type(mlp_classifier_options_t), intent(in), optional :: options
        type(mlp_classifier_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:)
        type(mlp_classifier_options_t) :: config
        type(mlp_classifier_state_t) :: result
        type(adam_t) :: optimizer
        integer, allocatable :: layer_sizes(:), classes(:), encoded(:), order(:)
        integer, allocatable :: batch_labels(:)
        real(dp), allocatable :: theta(:), best_theta(:), gradient(:)
        real(dp), allocatable :: x_batch(:, :), sample_weights(:), batch_weights(:)
        real(dp), allocatable :: class_factors(:)
        real(dp) :: loss, gradient_norm, best_loss, improvement
        integer :: n_samples, n_features, n_classes, n_hidden
        integer :: batch, first, last, n_batch, epoch, stale_epochs
        integer(int64) :: shuffle_state
        logical :: stop_now

        if (present(options)) config = options
        if (.not. valid_options(config)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier fit: invalid optimizer options")
            if (present(state)) state = result
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier fit: input dimensions or values are invalid")
            if (present(state)) state = result
            return
        end if

        call sorted_classes(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier fit: at least two classes are required")
            if (present(state)) state = result
            return
        end if
        call encode_labels(labels, classes, encoded, status)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        allocate(class_factors(n_classes))
        class_factors = 1.0_dp
        allocate(sample_weights(size(labels)))
        sample_weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier fit: sample weights must match samples")
                if (present(state)) state = result
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier fit: sample weights must be finite and nonnegative")
                if (present(state)) state = result
                return
            end if
            sample_weights = sample_weight
        end if
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier fit: class weights must match sorted classes")
                if (present(state)) state = result
                return
            end if
            if (any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier fit: class weights must be finite and positive")
                if (present(state)) state = result
                return
            end if
            class_factors = class_weight
        end if
        do first = 1, size(labels)
            sample_weights(first) = sample_weights(first)*class_factors(encoded(first))
        end do
        if (any(.not. ieee_is_finite(sample_weights)) .or. &
            .not. ieee_is_finite(sum(sample_weights)) .or. &
            sum(sample_weights) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier fit: effective sample weights must have positive mass")
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
                    "MLP classifier fit: hidden layer sizes must be positive")
                if (present(state)) state = result
                return
            end if
        end if
        allocate(layer_sizes(n_hidden + 2))
        layer_sizes(1) = n_features
        if (n_hidden > 0) layer_sizes(2:n_hidden + 1) = hidden_layer_sizes
        layer_sizes(n_hidden + 2) = n_classes
        call self%logits%initialize(layer_sizes, status, &
            hidden_activation=config%hidden_activation, initialization_seed=&
            config%initialization_seed)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        allocate(self%class_label(n_classes))
        self%class_label = classes

        theta = self%logits%parameters()
        allocate(best_theta, source=theta)
        allocate(gradient(size(theta)))
        allocate(order(n_samples))
        allocate(result%loss_history(config%max_epochs))
        call mlp_classifier_loss_gradient(self%logits, x, encoded, config%l2, &
            loss, gradient, status, sample_weights)
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
            order = [(first, first=1, n_samples)]
            if (config%shuffle) call shuffle_order(order, shuffle_state)
            do first = 1, n_samples, batch
                last = min(first + batch - 1, n_samples)
                n_batch = last - first + 1
                allocate(x_batch(n_batch, n_features), batch_labels(n_batch), &
                    batch_weights(n_batch))
                x_batch = x(order(first:last), :)
                batch_labels = encoded(order(first:last))
                batch_weights = sample_weights(order(first:last))
                if (sum(batch_weights) <= 0.0_dp) then
                    deallocate(x_batch, batch_labels, batch_weights)
                    cycle
                end if
                call mlp_classifier_loss_gradient(self%logits, x_batch, &
                    batch_labels, config%l2, loss, gradient, status, batch_weights)
                deallocate(x_batch, batch_labels, batch_weights)
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

            call mlp_classifier_loss_gradient(self%logits, x, encoded, config%l2, &
                loss, gradient, status, sample_weights)
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
            gradient_norm = sqrt(sum(gradient*gradient))
            result%epochs = epoch
            result%loss_history(epoch) = loss
            result%gradient_norm = gradient_norm
            improvement = best_loss - loss
            if (improvement > config%min_delta) then
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
        if (config%restore_best .and. result%best_epoch < result%epochs) then
            theta = best_theta
            call self%logits%set_parameters(theta, status)
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
            call mlp_classifier_loss_gradient(self%logits, x, encoded, config%l2, &
                loss, gradient, status, sample_weights)
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
            result%gradient_norm = sqrt(sum(gradient*gradient))
        end if
        result%final_loss = loss
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_classifier_fit

    subroutine mlp_classifier_loss_gradient(model, x, labels, l2, value, &
            gradient, status, sample_weight)
        !! Cross-entropy value and parameter gradient for a logits MLP.
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), l2
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: logits(:, :), logits_bar(:, :), x_bar(:, :)
        real(dp), allocatable :: theta(:), weights(:)
        real(dp) :: weight_sum, maximum, normalizer
        integer :: row, column

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. allocated(model%layer_sizes)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: model is not initialized")
            return
        end if
        if (l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: model, data, or gradient shape is invalid")
            return
        end if
        if (.not. ieee_is_finite(l2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: model, data, or gradient shape is invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: model, data, or gradient shape is invalid")
            return
        end if
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: model, data, or gradient shape is invalid")
            return
        end if
        if (size(gradient) /= model%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: model, data, or gradient shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: model, data, or gradient shape is invalid")
            return
        end if
        allocate(weights(size(x, 1)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier loss: sample-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP classifier loss: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: sample weights must have positive mass")
            return
        end if
        allocate(logits(size(x, 1), model%layer_sizes(size(model%layer_sizes))))
        allocate(logits_bar(size(x, 1), size(logits, 2)))
        allocate(x_bar(size(x, 1), size(x, 2)))
        call model%predict(x, logits, status)
        if (.not. status_ok(status)) return
        if (any(.not. ieee_is_finite(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: logits must be finite")
            return
        end if
        if (size(logits, 2) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: logits need at least two classes")
            return
        end if
        if (any(labels < 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: class indices are invalid")
            return
        end if
        if (any(labels > size(logits, 2))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: class indices are invalid")
            return
        end if
        value = 0.0_dp
        logits_bar = 0.0_dp
        do row = 1, size(logits, 1)
            maximum = maxval(logits(row, :))
            normalizer = 0.0_dp
            do column = 1, size(logits, 2)
                logits_bar(row, column) = exp(logits(row, column) - maximum)
                normalizer = normalizer + logits_bar(row, column)
            end do
            value = value + weights(row)*(maximum - logits(row, labels(row)) + &
                log(normalizer))
            logits_bar(row, :) = weights(row)*logits_bar(row, :)/normalizer
            logits_bar(row, labels(row)) = logits_bar(row, labels(row)) - weights(row)
        end do
        value = value/weight_sum
        logits_bar = logits_bar/weight_sum
        call model%vjp(x, logits_bar, gradient, x_bar, status)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        value = value + 0.5_dp*l2*sum(theta*theta)
        gradient = gradient + l2*theta
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_classifier_loss_gradient

    subroutine mlp_classifier_model_loss_gradient(self, x, labels, l2, value, &
            gradient, status, sample_weight)
        class(mlp_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), l2
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: encoded(:)

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. mlp_classifier_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss_gradient: model is not fitted")
            return
        end if
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier loss_gradient: label shape is invalid")
            return
        end if
        call encode_labels(labels, self%class_label, encoded, status)
        if (.not. status_ok(status)) return
        call mlp_classifier_loss_gradient(self%logits, x, encoded, l2, value, &
            gradient, status, sample_weight)
    end subroutine mlp_classifier_model_loss_gradient

    subroutine mlp_classifier_decision(self, x, scores, status)
        class(mlp_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. mlp_classifier_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier decision_function: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
            any(shape(scores) /= [size(x, 1), self%class_count()]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier decision_function: input or output shape is invalid")
            return
        end if
        call self%logits%predict(x, scores, status)
    end subroutine mlp_classifier_decision

    subroutine mlp_classifier_decision_device(self, device, x, scores, status)
        !! Explicit device contract for MLP logits.
        !!
        !! The MLP classifier currently has no resident CUDA lowering.  CPU
        !! dispatch is the ordinary host implementation; an accelerator
        !! request returns `FORTNUM_NOT_IMPLEMENTED` instead of silently
        !! copying the batch back to the host.
        class(mlp_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP classifier device decision: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier device decision: device kind is invalid")
        end select
    end subroutine mlp_classifier_decision_device

    subroutine mlp_classifier_decision_jvp(self, x, theta_dot, x_dot, scores, &
            scores_dot, status)
        !! Exact joint JVP of logits with respect to parameters and inputs.
        class(mlp_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:, :), scores_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. mlp_classifier_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier decision JVP: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(scores) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(scores_dot) /= shape(scores)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier decision JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier decision JVP: inputs and tangents must be finite")
            return
        end if
        call self%logits%jvp(x, theta_dot, x_dot, scores, scores_dot, status)
    end subroutine mlp_classifier_decision_jvp

    subroutine mlp_classifier_decision_vjp(self, x, scores_bar, theta_bar, &
            x_bar, status)
        !! Exact VJP of logits with respect to packed parameters and inputs.
        class(mlp_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. mlp_classifier_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier decision VJP: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
            any(shape(scores_bar) /= [size(x, 1), self%class_count()]) .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier decision VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier decision VJP: inputs and cotangents must be finite")
            return
        end if
        call self%logits%vjp(x, scores_bar, theta_bar, x_bar, status)
    end subroutine mlp_classifier_decision_vjp

    subroutine mlp_classifier_predict_proba(self, x, probabilities, status)
        class(mlp_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :)

        if (.not. mlp_classifier_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier predict_proba: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier predict_proba: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%class_count()))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        call softmax_value(scores, probabilities, status)
    end subroutine mlp_classifier_predict_proba

    subroutine mlp_classifier_predict_proba_device(self, device, x, probabilities, &
            status)
        class(mlp_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP classifier device probability: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier device probability: device kind is invalid")
        end select
    end subroutine mlp_classifier_predict_proba_device

    subroutine mlp_classifier_predict_proba_jvp(self, x, theta_dot, x_dot, &
            probabilities, probabilities_dot, status)
        !! Exact joint JVP of softmax probabilities.
        class(mlp_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), scores_dot(:, :)

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. mlp_classifier_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier probability JVP: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier probability JVP: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%class_count()))
        allocate(scores_dot(size(x, 1), self%class_count()))
        call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        if (status%code /= FORTNUM_OK) return
        call softmax_jvp(scores, scores_dot, probabilities, probabilities_dot, status)
    end subroutine mlp_classifier_predict_proba_jvp

    subroutine mlp_classifier_predict_proba_vjp(self, x, probabilities_bar, &
            theta_bar, x_bar, status)
        !! Exact VJP of softmax probabilities with respect to parameters/input.
        class(mlp_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), logits_bar(:, :)

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. mlp_classifier_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier probability VJP: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
            any(shape(probabilities_bar) /= [size(x, 1), self%class_count()]) .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier probability VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier probability VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(scores(size(x, 1), self%class_count()))
        allocate(logits_bar(size(x, 1), self%class_count()))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        call softmax_vjp(scores, probabilities_bar, logits_bar, status)
        if (status%code /= FORTNUM_OK) return
        call self%decision_function_vjp(x, logits_bar, theta_bar, x_bar, status)
    end subroutine mlp_classifier_predict_proba_vjp

    subroutine mlp_classifier_predict(self, x, labels, status)
        class(mlp_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, class_index

        if (.not. mlp_classifier_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier predict: model is not fitted")
            return
        end if
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%class_count()))
        call self%predict_proba(x, probabilities, status)
        if (.not. status_ok(status)) return
        do i = 1, size(labels)
            class_index = maxloc(probabilities(i, :), dim=1)
            labels(i) = self%class_label(class_index)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_classifier_predict

    subroutine mlp_classifier_predict_device(self, device, x, labels, status)
        class(mlp_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP classifier device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier device prediction: device kind is invalid")
        end select
    end subroutine mlp_classifier_predict_device

    function mlp_classifier_classes(self) result(classes)
        class(mlp_classifier_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function mlp_classifier_classes

    integer function mlp_classifier_feature_count(self) result(count)
        class(mlp_classifier_t), intent(in) :: self

        count = 0
        if (self%fitted()) count = self%logits%layer_sizes(1)
    end function mlp_classifier_feature_count

    integer function mlp_classifier_class_count(self) result(count)
        class(mlp_classifier_t), intent(in) :: self

        count = 0
        if (allocated(self%class_label)) count = size(self%class_label)
    end function mlp_classifier_class_count

    integer function mlp_classifier_parameter_count(self) result(count)
        class(mlp_classifier_t), intent(in) :: self

        count = 0
        if (self%fitted()) count = self%logits%parameter_count()
    end function mlp_classifier_parameter_count

    function mlp_classifier_parameters(self) result(values)
        class(mlp_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (self%fitted()) then
            values = self%logits%parameters()
        else
            allocate(values(0))
        end if
    end function mlp_classifier_parameters

    subroutine mlp_classifier_set_parameters(self, values, status)
        class(mlp_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier set_parameters: model is not fitted")
            return
        end if
        call self%logits%set_parameters(values, status)
    end subroutine mlp_classifier_set_parameters

    logical function mlp_classifier_fitted(self) result(is_fitted)
        class(mlp_classifier_t), intent(in) :: self

        is_fitted = .false.
        if (.not. allocated(self%class_label)) return
        if (size(self%class_label) < 2) return
        is_fitted = self%logits%parameter_count() > 0
    end function mlp_classifier_fitted

    logical function mlp_classifier_device_supported(self, device_kind) &
            result(supported)
        class(mlp_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%fitted()
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function mlp_classifier_device_supported

    logical function valid_options(options) result(valid)
        type(mlp_classifier_options_t), intent(in) :: options

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
        type(mlp_classifier_lbfgsb_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. options%gradient_tolerance >= 0.0_dp .and. &
            options%step_tolerance >= 0.0_dp .and. options%objective_tolerance >= 0.0_dp .and. &
            options%lower_bound <= options%upper_bound .and. options%l2 >= 0.0_dp .and. &
            options%l2_lower_bound >= 0.0_dp .and. &
            options%l2_lower_bound <= options%l2_upper_bound
        valid = valid .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            ieee_is_finite(options%lower_bound) .and. ieee_is_finite(options%upper_bound) .and. &
            ieee_is_finite(options%l2) .and. ieee_is_finite(options%l2_lower_bound) .and. &
            ieee_is_finite(options%l2_upper_bound)
        if (options%optimize_l2) valid = valid .and. &
            options%l2 >= options%l2_lower_bound .and. &
            options%l2 <= options%l2_upper_bound
    end function valid_lbfgsb_options

    subroutine encode_labels(labels, classes, encoded, status)
        integer, intent(in) :: labels(:), classes(:)
        integer, allocatable, intent(out) :: encoded(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        allocate(encoded(size(labels)))
        encoded = 0
        do i = 1, size(labels)
            do j = 1, size(classes)
                if (labels(i) == classes(j)) then
                    encoded(i) = j
                    exit
                end if
            end do
        end do
        if (any(encoded < 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP classifier fit: could not encode class labels")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine encode_labels

    subroutine sorted_classes(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, n, temporary

        allocate(work, source=labels)
        do i = 2, size(work)
            temporary = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= temporary) exit
                work(j + 1) = work(j)
                j = j - 1
            end do
            work(j + 1) = temporary
        end do
        n = 1
        do i = 2, size(work)
            if (work(i) /= work(n)) then
                n = n + 1
                work(n) = work(i)
            end if
        end do
        allocate(classes(n))
        classes = work(:n)
    end subroutine sorted_classes

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

end module fortml_mlp_classifier
