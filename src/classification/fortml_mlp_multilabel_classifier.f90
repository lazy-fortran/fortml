module fortml_mlp_multilabel_classifier
    !! Dense multilabel neural classifier with a shared differentiable MLP.
    !!
    !! The final layer emits one logit per indicator column.  Every column is
    !! trained with weighted binary cross entropy and the same MLP parameters;
    !! fitting is deterministic full-batch Adam.  The prediction and loss
    !! products are exact with respect to both packed parameters and inputs.
    !! CUDA requests are explicit typed refusals until resident kernels are
    !! linked; no host fallback is performed.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_losses, only: stable_sigmoid, &
        multilabel_binary_cross_entropy_with_logits_value, &
        multilabel_binary_cross_entropy_with_logits_vjp, &
        multilabel_binary_cross_entropy_with_logits_hvp
    use fortml_mlp, only: mlp_t, MLP_TANH
    use fortopt_adam, only: adam_t
    implicit none
    private

    type, public :: mlp_multilabel_classifier_options_t
        integer :: max_epochs = 1000
        integer :: initialization_seed = 17
        integer :: hidden_activation = MLP_TANH
        logical :: restore_best = .true.
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: tolerance = 1.0e-6_dp
        real(dp) :: min_delta = 0.0_dp
    end type mlp_multilabel_classifier_options_t

    type, public :: mlp_multilabel_classifier_state_t
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
    end type mlp_multilabel_classifier_state_t

    type, public :: mlp_multilabel_classifier_t
        private
        type(mlp_t) :: logits
        integer :: n_features = 0
        integer :: n_labels = 0
        real(dp), allocatable :: decision_threshold(:)
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => mlp_multilabel_classifier_fit
        procedure, public :: decision_function => mlp_multilabel_decision
        procedure, public :: decision_function_device => &
            mlp_multilabel_decision_device
        procedure, public :: decision_function_jvp => mlp_multilabel_decision_jvp
        procedure, public :: decision_function_vjp => mlp_multilabel_decision_vjp
        procedure, public :: predict_proba => mlp_multilabel_predict_proba
        procedure, public :: predict_proba_device => &
            mlp_multilabel_predict_proba_device
        procedure, public :: predict_proba_jvp => mlp_multilabel_predict_proba_jvp
        procedure, public :: predict_proba_vjp => mlp_multilabel_predict_proba_vjp
        procedure, public :: predict => mlp_multilabel_predict
        procedure, public :: predict_device => mlp_multilabel_predict_device
        procedure, public :: set_parameters => mlp_multilabel_set_parameters
        procedure, public :: parameters => mlp_multilabel_parameters
        procedure, public :: parameter_count => mlp_multilabel_parameter_count
        procedure, public :: label_count => mlp_multilabel_label_count
        procedure, public :: feature_count => mlp_multilabel_feature_count
        procedure, public :: thresholds => mlp_multilabel_thresholds
        procedure, public :: set_thresholds => mlp_multilabel_set_thresholds
        procedure, public :: loss_gradient => &
            mlp_multilabel_loss_gradient_model
        procedure, public :: loss_hvp => mlp_multilabel_loss_hvp_model
        procedure, public :: fitted => mlp_multilabel_fitted
        procedure, public :: device_supported => mlp_multilabel_device_supported
    end type mlp_multilabel_classifier_t

    public :: mlp_multilabel_classifier_fit
    public :: mlp_multilabel_decision
    public :: mlp_multilabel_decision_device
    public :: mlp_multilabel_decision_jvp
    public :: mlp_multilabel_decision_vjp
    public :: mlp_multilabel_predict_proba
    public :: mlp_multilabel_predict_proba_device
    public :: mlp_multilabel_predict_proba_jvp
    public :: mlp_multilabel_predict_proba_vjp
    public :: mlp_multilabel_predict
    public :: mlp_multilabel_predict_device

contains

    subroutine mlp_multilabel_classifier_fit(self, x, indicators, status, &
            hidden_layer_sizes, options, state, sample_weight, class_weight, &
            thresholds)
        class(mlp_multilabel_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_layer_sizes(:)
        type(mlp_multilabel_classifier_options_t), intent(in), optional :: options
        type(mlp_multilabel_classifier_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:, :)
        real(dp), intent(in), optional :: thresholds(:)
        type(mlp_multilabel_classifier_options_t) :: config
        type(mlp_multilabel_classifier_state_t) :: result
        type(adam_t) :: optimizer
        integer, allocatable :: layer_sizes(:)
        real(dp), allocatable :: targets(:, :), weights(:), theta(:), best_theta(:)
        real(dp), allocatable :: gradient(:)
        real(dp) :: loss, gradient_norm, best_loss, improvement
        integer :: n_samples, n_features, n_labels, n_hidden, epoch

        if (present(options)) config = options
        if (.not. valid_options(config)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel fit: invalid optimizer options")
            if (present(state)) state = result
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(indicators, 1) /= size(x, 1) .or. size(indicators, 2) < 1 .or. &
            any(.not. ieee_is_finite(x)) .or. &
            any((indicators /= 0) .and. (indicators /= 1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel fit: input dimensions or indicator values are invalid")
            if (present(state)) state = result
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_labels = size(indicators, 2)
        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel fit: sample weights must be finite and nonnegative")
                if (present(state)) state = result
                return
            end if
            weights = sample_weight
        end if
        if (.not. ieee_is_finite(sum(weights)) .or. sum(weights) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel fit: sample weights have no positive mass")
            if (present(state)) state = result
            return
        end if
        if (present(class_weight)) then
            if (any(shape(class_weight) /= [2, n_labels]) .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel fit: class_weight must have shape (2,n_labels)")
                if (present(state)) state = result
                return
            end if
        end if
        if (present(thresholds)) then
            if (size(thresholds) /= n_labels .or. &
                any(.not. ieee_is_finite(thresholds)) .or. &
                any(thresholds <= 0.0_dp) .or. any(thresholds >= 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel fit: thresholds must be finite in (0,1)")
                if (present(state)) state = result
                return
            end if
        end if
        n_hidden = 0
        if (present(hidden_layer_sizes)) then
            n_hidden = size(hidden_layer_sizes)
            if (n_hidden > 0 .and. any(hidden_layer_sizes < 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel fit: hidden layer sizes must be positive")
                if (present(state)) state = result
                return
            end if
        end if
        allocate(layer_sizes(n_hidden + 2))
        layer_sizes(1) = n_features
        if (n_hidden > 0) layer_sizes(2:n_hidden + 1) = hidden_layer_sizes
        layer_sizes(n_hidden + 2) = n_labels
        call self%logits%initialize(layer_sizes, status, &
            hidden_activation=config%hidden_activation, initialization_seed=&
            config%initialization_seed)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        self%n_features = n_features
        self%n_labels = n_labels
        allocate(self%decision_threshold(n_labels))
        self%decision_threshold = 0.5_dp
        if (present(thresholds)) self%decision_threshold = thresholds
        self%is_fitted = .true.
        allocate(targets(n_samples, n_labels))
        targets = real(indicators, dp)
        theta = self%logits%parameters()
        allocate(best_theta, source=theta, gradient(size(theta)))
        allocate(result%loss_history(config%max_epochs))
        call mlp_multilabel_loss_gradient(self%logits, x, targets, config%l2, &
            loss, gradient, status, weights, class_weight)
        if (.not. status_ok(status)) then
            self%is_fitted = .false.
            if (present(state)) state = result
            return
        end if
        result%initial_loss = loss
        result%best_loss = loss
        best_loss = loss
        call optimizer%initialize(size(theta), status, &
            learning_rate=config%learning_rate, beta1=config%beta1, &
            beta2=config%beta2, epsilon=config%epsilon)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        do epoch = 1, config%max_epochs
            call mlp_multilabel_loss_gradient(self%logits, x, targets, config%l2, &
                loss, gradient, status, weights, class_weight)
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
            call mlp_multilabel_loss_gradient(self%logits, x, targets, config%l2, &
                loss, gradient, status, weights, class_weight)
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
            end if
            if (gradient_norm <= config%tolerance) then
                result%converged = .true.
                exit
            end if
            if (epoch > 1 .and. improvement <= config%min_delta .and. &
                config%min_delta > 0.0_dp) then
                result%early_stopped = .true.
                exit
            end if
        end do
        if (result%epochs < size(result%loss_history)) then
            call shrink_history(result%loss_history, result%epochs)
        end if
        if (config%restore_best .and. result%best_epoch > 0 .and. &
            result%best_epoch < result%epochs) then
            theta = best_theta
            call self%logits%set_parameters(theta, status)
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
            call mlp_multilabel_loss_gradient(self%logits, x, targets, config%l2, &
                loss, gradient, status, weights, class_weight)
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
            result%gradient_norm = sqrt(sum(gradient*gradient))
        end if
        result%final_loss = loss
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_classifier_fit

    subroutine mlp_multilabel_loss_gradient(model, x, targets, l2, value, gradient, &
            status, sample_weight, class_weight)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :), l2
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:, :)
        real(dp), allocatable :: logits(:, :), logits_bar(:, :), x_bar(:, :)
        real(dp), allocatable :: col_logits(:, :), col_targets(:, :), col_bar(:, :)
        real(dp), allocatable :: weights(:), theta(:)
        real(dp) :: column_value
        integer :: j

        value = 0.0_dp
        gradient = 0.0_dp
        if (l2 < 0.0_dp .or. .not. ieee_is_finite(l2) .or. &
            size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            any(shape(targets) /= [size(x, 1), size(model%layer(size(model%layer))%bias)]) .or. &
            size(gradient) /= model%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets)) .or. &
            any(targets < 0.0_dp) .or. any(targets > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss: model, data, or gradient shape is invalid")
            return
        end if
        if (present(class_weight)) then
            if (any(shape(class_weight) /= [2, size(targets, 2)])) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel loss: class weight shape is invalid")
                return
            end if
        end if
        allocate(logits(size(x, 1), size(targets, 2)), &
            logits_bar(size(x, 1), size(targets, 2)), x_bar(size(x, 1), size(x, 2)), &
            col_logits(size(x, 1), 1), col_targets(size(x, 1), 1), &
            col_bar(size(x, 1), 1), weights(size(x, 1)))
        call model%predict(x, logits, status)
        if (.not. status_ok(status)) return
        logits_bar = 0.0_dp
        do j = 1, size(targets, 2)
            col_logits(:, 1) = logits(:, j)
            col_targets(:, 1) = targets(:, j)
            call make_column_weights(targets(:, j), j, sample_weight, class_weight, &
                weights, status)
            if (.not. status_ok(status)) return
            call multilabel_binary_cross_entropy_with_logits_value(col_logits, &
                col_targets, column_value, status, sample_weight=weights)
            if (.not. status_ok(status)) return
            call multilabel_binary_cross_entropy_with_logits_vjp(col_logits, &
                col_targets, 1.0_dp, col_bar, status, sample_weight=weights)
            if (.not. status_ok(status)) return
            value = value + column_value
            logits_bar(:, j) = col_bar(:, 1)
        end do
        value = value/real(size(targets, 2), dp)
        logits_bar = logits_bar/real(size(targets, 2), dp)
        call model%vjp(x, logits_bar, gradient, x_bar, status)
        if (.not. status_ok(status)) return
        theta = model%parameters()
        value = value + 0.5_dp*l2*sum(theta*theta)
        gradient = gradient + l2*theta
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss: objective or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_loss_gradient

    subroutine mlp_multilabel_loss_hvp(model, x, targets, l2, theta_dot, hvp, &
            status, sample_weight, class_weight)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :), l2, theta_dot(:)
        real(dp), intent(out) :: hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:, :)
        real(dp), allocatable :: logits(:, :), logits_dot(:, :), logits_bar(:, :)
        real(dp), allocatable :: logits_hvp(:, :), x_zero(:, :), x_bar(:, :), x_hvp(:, :)
        real(dp), allocatable :: col_logits(:, :), col_targets(:, :), col_dot(:, :)
        real(dp), allocatable :: col_bar(:, :), col_hvp(:, :), weights(:), direct(:)
        real(dp), allocatable :: theta(:)
        integer :: j

        hvp = 0.0_dp
        if (l2 < 0.0_dp .or. .not. ieee_is_finite(l2) .or. &
            size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            any(shape(targets) /= [size(x, 1), size(model%layer(size(model%layer))%bias)]) .or. &
            size(theta_dot) /= model%parameter_count() .or. &
            size(hvp) /= model%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets)) .or. &
            any(targets < 0.0_dp) .or. any(targets > 1.0_dp) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel HVP: model, data, or direction shape is invalid")
            return
        end if
        if (present(class_weight)) then
            if (any(shape(class_weight) /= [2, size(targets, 2)])) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel HVP: class weight shape is invalid")
                return
            end if
        end if
        allocate(logits(size(x, 1), size(targets, 2)), &
            logits_dot(size(x, 1), size(targets, 2)), &
            logits_bar(size(x, 1), size(targets, 2)), &
            logits_hvp(size(x, 1), size(targets, 2)), x_zero(size(x, 1), size(x, 2)), &
            x_bar(size(x, 1), size(x, 2)), x_hvp(size(x, 1), size(x, 2)), &
            col_logits(size(x, 1), 1), col_targets(size(x, 1), 1), &
            col_dot(size(x, 1), 1), col_bar(size(x, 1), 1), &
            col_hvp(size(x, 1), 1), weights(size(x, 1)), direct(size(theta_dot)))
        x_zero = 0.0_dp
        call model%predict(x, logits, status)
        if (.not. status_ok(status)) return
        call model%jvp(x, theta_dot, x_zero, logits, logits_dot, status)
        if (.not. status_ok(status)) return
        logits_bar = 0.0_dp
        logits_hvp = 0.0_dp
        do j = 1, size(targets, 2)
            col_logits(:, 1) = logits(:, j)
            col_dot(:, 1) = logits_dot(:, j)
            col_targets(:, 1) = targets(:, j)
            call make_column_weights(targets(:, j), j, sample_weight, class_weight, &
                weights, status)
            if (.not. status_ok(status)) return
            call multilabel_binary_cross_entropy_with_logits_vjp(col_logits, &
                col_targets, 1.0_dp, col_bar, status, sample_weight=weights)
            if (.not. status_ok(status)) return
            call multilabel_binary_cross_entropy_with_logits_hvp(col_logits, &
                col_targets, col_dot, col_hvp, status, sample_weight=weights)
            if (.not. status_ok(status)) return
            logits_bar(:, j) = col_bar(:, 1)
            logits_hvp(:, j) = col_hvp(:, 1)
        end do
        logits_bar = logits_bar/real(size(targets, 2), dp)
        logits_hvp = logits_hvp/real(size(targets, 2), dp)
        call model%hvp(x, logits_bar, theta_dot, x_zero, direct, x_hvp, status)
        if (.not. status_ok(status)) return
        call model%vjp(x, logits_hvp, hvp, x_bar, status)
        if (.not. status_ok(status)) return
        hvp = hvp + direct
        theta = model%parameters()
        hvp = hvp + l2*theta_dot
        if (any(.not. ieee_is_finite(hvp))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_loss_hvp

    subroutine mlp_multilabel_loss_gradient_model(self, x, indicators, l2, value, &
            gradient, status, sample_weight, class_weight)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), l2
        integer, intent(in) :: indicators(:, :)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:, :)
        real(dp), allocatable :: targets(:, :)

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss_gradient: model is not fitted")
            return
        end if
        if (size(indicators, 1) /= size(x, 1) .or. &
            size(indicators, 2) /= self%label_count() .or. &
            any((indicators /= 0) .and. (indicators /= 1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss_gradient: indicator shape or values are invalid")
            return
        end if
        allocate(targets(size(indicators, 1), size(indicators, 2)))
        targets = real(indicators, dp)
        call mlp_multilabel_loss_gradient(self%logits, x, targets, l2, value, gradient, &
            status, sample_weight, class_weight)
    end subroutine mlp_multilabel_loss_gradient_model

    subroutine mlp_multilabel_loss_hvp_model(self, x, indicators, l2, theta_dot, hvp, &
            status, sample_weight, class_weight)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), l2, theta_dot(:)
        integer, intent(in) :: indicators(:, :)
        real(dp), intent(out) :: hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:, :)
        real(dp), allocatable :: targets(:, :)

        hvp = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss_hvp: model is not fitted")
            return
        end if
        if (size(indicators, 1) /= size(x, 1) .or. &
            size(indicators, 2) /= self%label_count() .or. &
            any((indicators /= 0) .and. (indicators /= 1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss_hvp: indicator shape or values are invalid")
            return
        end if
        allocate(targets(size(indicators, 1), size(indicators, 2)))
        targets = real(indicators, dp)
        call mlp_multilabel_loss_hvp(self%logits, x, targets, l2, theta_dot, hvp, status, &
            sample_weight, class_weight)
    end subroutine mlp_multilabel_loss_hvp_model

    subroutine mlp_multilabel_decision(self, x, scores, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        scores = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(scores) /= [size(x, 1), self%n_labels]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision: input or output shape is invalid")
            return
        end if
        call self%logits%predict(x, scores, status)
    end subroutine mlp_multilabel_decision

    subroutine mlp_multilabel_decision_device(self, device, x, scores, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP multilabel device decision: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device decision: device kind is invalid")
        end select
    end subroutine mlp_multilabel_decision_device

    subroutine mlp_multilabel_decision_jvp(self, x, theta_dot, x_dot, scores, &
            scores_dot, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:, :), scores_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision JVP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. any(shape(x_dot) /= shape(x)) .or. &
            any(shape(scores) /= [size(x, 1), self%n_labels]) .or. &
            any(shape(scores_dot) /= shape(scores)) .or. &
            size(theta_dot) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision JVP: tangent or output shape is invalid")
            return
        end if
        call self%logits%jvp(x, theta_dot, x_dot, scores, scores_dot, status)
    end subroutine mlp_multilabel_decision_jvp

    subroutine mlp_multilabel_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(scores_bar) /= [size(x, 1), self%n_labels]) .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x)) .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision VJP: cotangent or output shape is invalid")
            return
        end if
        call self%logits%vjp(x, scores_bar, theta_bar, x_bar, status)
    end subroutine mlp_multilabel_decision_vjp

    subroutine mlp_multilabel_predict_proba(self, x, probabilities, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :)

        probabilities = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel predict_proba: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%n_labels])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel predict_proba: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%n_labels))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        probabilities = stable_sigmoid(scores)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_predict_proba

    subroutine mlp_multilabel_predict_proba_device(self, device, x, probabilities, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP multilabel device probability: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device probability: device kind is invalid")
        end select
    end subroutine mlp_multilabel_predict_proba_device

    subroutine mlp_multilabel_predict_proba_jvp(self, x, theta_dot, x_dot, &
            probabilities, probabilities_dot, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), scores_dot(:, :), positive(:, :)

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel probability JVP: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%n_labels]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel probability JVP: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%n_labels), &
            scores_dot(size(x, 1), self%n_labels))
        call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        if (.not. status_ok(status)) return
        allocate(positive(size(x, 1), self%n_labels))
        positive = stable_sigmoid(scores)
        probabilities = positive
        probabilities_dot = positive*(1.0_dp - positive)*scores_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_predict_proba_jvp

    subroutine mlp_multilabel_predict_proba_vjp(self, x, probabilities_bar, &
            theta_bar, x_bar, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), positive(:, :), score_bar(:, :)

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel probability VJP: model is not fitted")
            return
        end if
        if (any(shape(probabilities_bar) /= [size(x, 1), self%n_labels]) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel probability VJP: cotangent shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%n_labels), &
            positive(size(x, 1), self%n_labels), &
            score_bar(size(x, 1), self%n_labels))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        positive = stable_sigmoid(scores)
        score_bar = probabilities_bar*positive*(1.0_dp - positive)
        call self%decision_function_vjp(x, score_bar, theta_bar, x_bar, status)
    end subroutine mlp_multilabel_predict_proba_vjp

    subroutine mlp_multilabel_predict(self, x, indicators, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: j

        indicators = 0
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel predict: model is not fitted")
            return
        end if
        if (any(shape(indicators) /= [size(x, 1), self%n_labels])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_labels))
        call self%predict_proba(x, probabilities, status)
        if (.not. status_ok(status)) return
        do j = 1, self%n_labels
            where (probabilities(:, j) >= self%decision_threshold(j))
                indicators(:, j) = 1
            elsewhere
                indicators(:, j) = 0
            end where
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_predict

    subroutine mlp_multilabel_predict_device(self, device, x, indicators, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: indicators(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, indicators, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP multilabel device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device prediction: device kind is invalid")
        end select
    end subroutine mlp_multilabel_predict_device

    function mlp_multilabel_parameters(self) result(values)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (self%fitted()) then
            values = self%logits%parameters()
        else
            allocate(values(0))
        end if
    end function mlp_multilabel_parameters

    subroutine mlp_multilabel_set_parameters(self, values, status)
        class(mlp_multilabel_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel set_parameters: model is not fitted")
            return
        end if
        call self%logits%set_parameters(values, status)
    end subroutine mlp_multilabel_set_parameters

    integer function mlp_multilabel_parameter_count(self) result(count)
        class(mlp_multilabel_classifier_t), intent(in) :: self

        count = 0
        if (self%fitted()) count = self%logits%parameter_count()
    end function mlp_multilabel_parameter_count

    integer function mlp_multilabel_label_count(self) result(count)
        class(mlp_multilabel_classifier_t), intent(in) :: self

        count = 0
        if (self%fitted()) count = self%n_labels
    end function mlp_multilabel_label_count

    integer function mlp_multilabel_feature_count(self) result(count)
        class(mlp_multilabel_classifier_t), intent(in) :: self

        count = 0
        if (self%fitted()) count = self%n_features
    end function mlp_multilabel_feature_count

    function mlp_multilabel_thresholds(self) result(values)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%decision_threshold)) then
            values = self%decision_threshold
        else
            allocate(values(0))
        end if
    end function mlp_multilabel_thresholds

    subroutine mlp_multilabel_set_thresholds(self, values, status)
        class(mlp_multilabel_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted() .or. size(values) /= self%n_labels .or. &
            any(.not. ieee_is_finite(values)) .or. any(values <= 0.0_dp) .or. &
            any(values >= 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel set_thresholds: finite values in (0,1) are required")
            return
        end if
        self%decision_threshold = values
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_set_thresholds

    logical function mlp_multilabel_fitted(self) result(value)
        class(mlp_multilabel_classifier_t), intent(in) :: self

        value = self%is_fitted .and. allocated(self%decision_threshold) .and. &
            self%n_features > 0 .and. self%n_labels > 0 .and. &
            self%logits%parameter_count() > 0
    end function mlp_multilabel_fitted

    logical function mlp_multilabel_device_supported(self, device_kind) result(value)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            value = self%fitted()
        case default
            value = .false.
        end select
    end function mlp_multilabel_device_supported

    logical function valid_options(options) result(value)
        type(mlp_multilabel_classifier_options_t), intent(in) :: options

        value = options%max_epochs >= 1 .and. options%learning_rate > 0.0_dp .and. &
            options%beta1 >= 0.0_dp .and. options%beta1 < 1.0_dp .and. &
            options%beta2 >= 0.0_dp .and. options%beta2 < 1.0_dp .and. &
            options%epsilon > 0.0_dp .and. options%l2 >= 0.0_dp .and. &
            options%tolerance >= 0.0_dp .and. options%min_delta >= 0.0_dp .and. &
            options%initialization_seed >= 0 .and. &
            ieee_is_finite(options%learning_rate) .and. &
            ieee_is_finite(options%beta1) .and. ieee_is_finite(options%beta2) .and. &
            ieee_is_finite(options%epsilon) .and. ieee_is_finite(options%l2) .and. &
            ieee_is_finite(options%tolerance) .and. ieee_is_finite(options%min_delta)
    end function valid_options

    subroutine make_column_weights(target, column, sample_weight, class_weight, weights, status)
        real(dp), intent(in) :: target(:)
        integer, intent(in) :: column
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:, :)
        real(dp), intent(out) :: weights(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(target)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel loss: sample weight shape is invalid")
                return
            end if
            weights = sample_weight
        end if
        if (present(class_weight)) then
            if (size(class_weight, 1) /= 2 .or. column < 1 .or. &
                column > size(class_weight, 2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel loss: class weight shape is invalid")
                return
            end if
            do i = 1, size(target)
                if (target(i) >= 0.5_dp) then
                    weights(i) = weights(i)*class_weight(2, column)
                else
                    weights(i) = weights(i)*class_weight(1, column)
                end if
            end do
        end if
        if (any(.not. ieee_is_finite(weights)) .or. any(weights < 0.0_dp) .or. &
            sum(weights) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss: effective weights have no positive mass")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine make_column_weights

    subroutine shrink_history(history, length)
        real(dp), allocatable, intent(inout) :: history(:)
        integer, intent(in) :: length
        real(dp), allocatable :: compact(:)

        allocate(compact(max(0, length)))
        if (length > 0) compact = history(:length)
        call move_alloc(compact, history)
    end subroutine shrink_history

end module fortml_mlp_multilabel_classifier
