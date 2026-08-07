module fortml_mlp_ordinal_classifier
    !! Ordered cumulative-logit neural classifier.
    !!
    !! The network maps each sample to one latent score ``eta`` and the output
    !! head uses ordered thresholds ``t``:
    !! ``P(Y <= k) = sigmoid(t(k)-eta)``.  Threshold increments are optimized
    !! in log coordinates, so every fit preserves the ordered-label contract.
    !! Public packed parameters contain the MLP parameters followed by the
    !! actual (strictly increasing) thresholds.  Prediction JVP/VJP products
    !! are analytic and cover both network parameters and thresholds.
    !!
    !! Training is deterministic full-batch L-BFGS-B on CPU.  CUDA dispatch is
    !! explicit and refuses until a resident ordinal neural kernel is linked;
    !! it never silently copies data back to the host.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_TANH
    implicit none
    private

    type, public :: mlp_ordinal_classifier_options_t
        integer :: max_iterations = 500
        integer :: initialization_seed = 17
        integer :: hidden_activation = MLP_TANH
        real(dp) :: l2 = 1.0e-3_dp
        real(dp) :: tolerance = 1.0e-7_dp
    end type mlp_ordinal_classifier_options_t

    type, public :: mlp_ordinal_classifier_state_t
        integer :: iterations = 0
        logical :: converged = .false.
        real(dp) :: initial_loss = huge(1.0_dp)
        real(dp) :: final_loss = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type mlp_ordinal_classifier_state_t

    type, public :: mlp_ordinal_classifier_t
        private
        type(mlp_t) :: network
        real(dp), allocatable :: threshold(:)
        integer, allocatable :: class_label(:)
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => mlp_ordinal_fit
        procedure, public :: decision_function => mlp_ordinal_decision
        procedure, public :: decision_function_device => mlp_ordinal_decision_device
        procedure, public :: decision_function_jvp => mlp_ordinal_decision_jvp
        procedure, public :: decision_function_vjp => mlp_ordinal_decision_vjp
        procedure, public :: predict_proba => mlp_ordinal_predict_proba
        procedure, public :: predict_proba_device => &
            mlp_ordinal_predict_proba_device
        procedure, public :: predict_proba_jvp => mlp_ordinal_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            mlp_ordinal_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => mlp_ordinal_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            mlp_ordinal_predict_proba_parameter_vjp
        procedure, public :: predict => mlp_ordinal_predict
        procedure, public :: predict_device => mlp_ordinal_predict_device
        procedure, public :: classes => mlp_ordinal_classes
        procedure, public :: class_count => mlp_ordinal_class_count
        procedure, public :: feature_count => mlp_ordinal_feature_count
        procedure, public :: parameter_count => mlp_ordinal_parameter_count
        procedure, public :: parameters => mlp_ordinal_parameters
        procedure, public :: set_parameters => mlp_ordinal_set_parameters
        procedure, public :: thresholds => mlp_ordinal_thresholds
        procedure, public :: fitted => mlp_ordinal_fitted
        procedure, public :: device_supported => mlp_ordinal_device_supported
    end type mlp_ordinal_classifier_t

    public :: mlp_ordinal_fit
    public :: mlp_ordinal_decision
    public :: mlp_ordinal_predict_proba
    public :: mlp_ordinal_predict_proba_jvp
    public :: mlp_ordinal_predict_proba_parameter_jvp
    public :: mlp_ordinal_predict_proba_vjp
    public :: mlp_ordinal_predict_proba_parameter_vjp
    public :: mlp_ordinal_predict

contains

    subroutine mlp_ordinal_fit(self, x, labels, status, hidden_layer_sizes, &
            options, state, sample_weight)
        class(mlp_ordinal_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_layer_sizes(:)
        type(mlp_ordinal_classifier_options_t), intent(in), optional :: options
        type(mlp_ordinal_classifier_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:)
        type(mlp_ordinal_classifier_options_t) :: config
        type(mlp_ordinal_classifier_state_t) :: result
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: weights(:), encoded(:), theta(:), lower(:), upper(:)
        real(dp), allocatable :: classes_real(:), raw_initial(:)
        integer, allocatable :: classes(:), layer_sizes(:)
        integer :: n_samples, n_features, n_classes, n_thresholds
        integer :: n_network, n_parameters, i, j, class_index, iterations
        real(dp) :: weight_sum, cumulative, requested_tolerance
        logical :: valid

        result = mlp_ordinal_classifier_state_t()
        self%is_fitted = .false.
        if (allocated(self%threshold)) deallocate(self%threshold)
        if (allocated(self%class_label)) deallocate(self%class_label)
        if (present(options)) config = options
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal fit: input dimensions or values are invalid")
            if (present(state)) state = result
            return
        end if
        if (.not. valid_options(config)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal fit: optimizer options are invalid")
            if (present(state)) state = result
            return
        end if
        call sorted_classes(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal fit: at least two ordered classes are required")
            if (present(state)) state = result
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_thresholds = n_classes - 1
        allocate(weights(n_samples), encoded(n_samples), classes_real(n_classes))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal fit: sample weights must be finite and nonnegative")
                if (present(state)) state = result
                return
            end if
            weights = sample_weight
        end if
        do i = 1, n_samples
            class_index = 0
            do j = 1, n_classes
                if (labels(i) == classes(j)) then
                    class_index = j
                    exit
                end if
            end do
            encoded(i) = real(class_index, dp)
        end do
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal fit: sample weights must have positive mass")
            if (present(state)) state = result
            return
        end if
        do j = 1, n_classes
            if (sum(weights, mask=nint(encoded) == j) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal fit: every ordered class needs positive weight")
                if (present(state)) state = result
                return
            end if
        end do

        if (present(hidden_layer_sizes)) then
            if (size(hidden_layer_sizes) < 1 .or. any(hidden_layer_sizes < 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal fit: hidden layer sizes are invalid")
                if (present(state)) state = result
                return
            end if
            allocate(layer_sizes(size(hidden_layer_sizes) + 2))
            layer_sizes(1) = n_features
            layer_sizes(2:size(hidden_layer_sizes) + 1) = hidden_layer_sizes
            layer_sizes(size(layer_sizes)) = 1
        else
            allocate(layer_sizes(3))
            layer_sizes = [n_features, 8, 1]
        end if
        call self%network%initialize(layer_sizes, status, &
            hidden_activation=config%hidden_activation, initialization_seed=config%initialization_seed)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        n_network = self%network%parameter_count()
        n_parameters = n_network + n_thresholds
        allocate(theta(n_parameters), lower(n_parameters), upper(n_parameters), &
            raw_initial(n_thresholds))
        theta(:n_network) = self%network%parameters()
        cumulative = 0.0_dp
        do j = 1, n_thresholds
            cumulative = cumulative + sum(weights, mask=nint(encoded) == j)/weight_sum
            cumulative = min(max(cumulative, 0.05_dp), 0.95_dp)
            classes_real(j) = log(cumulative/(1.0_dp-cumulative))
        end do
        raw_initial(1) = classes_real(1)
        do j = 2, n_thresholds
            raw_initial(j) = log(max(classes_real(j)-classes_real(j-1), 1.0e-3_dp))
        end do
        theta(n_network+1:) = raw_initial
        lower = -huge(1.0_dp)
        upper = huge(1.0_dp)
        call objective%initialize(n_parameters, ordinal_objective, status)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        iterations = config%max_iterations
        requested_tolerance = config%tolerance
        optimizer_options%max_iterations = iterations
        optimizer_options%gradient_tolerance = requested_tolerance
        optimizer_options%step_tolerance = min(1.0e-12_dp, requested_tolerance)
        optimizer_options%objective_tolerance = min(1.0e-12_dp, requested_tolerance)
        call optimizer%minimize(objective, theta, lower, upper, optimizer_options, &
            optimizer_result, status)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        result%iterations = optimizer_result%state%iteration
        result%converged = optimizer_result%state%converged
        result%final_loss = optimizer_result%state%value
        result%gradient_norm = optimizer_result%state%gradient_norm
        result%initial_loss = result%final_loss
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP ordinal fit: optimizer reached its iteration limit")
            if (present(state)) state = result
            return
        end if
        if (any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "MLP ordinal fit: optimizer returned nonfinite parameters")
            if (present(state)) state = result
            return
        end if
        call self%network%set_parameters(theta(:n_network), status)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        allocate(self%threshold(n_thresholds), self%class_label(n_classes))
        call raw_to_threshold(theta(n_network+1:), self%threshold, status)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        self%class_label = classes
        self%is_fitted = .true.
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")

    contains

        subroutine ordinal_objective(parameters, value, gradient, objective_status)
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: value, gradient(:)
            type(fortnum_status_t), intent(out) :: objective_status
            real(dp), allocatable :: threshold_local(:), eta(:, :), eta_bar(:, :)
            real(dp), allocatable :: threshold_bar(:), network_gradient(:), x_bar(:, :)
            real(dp) :: probability, q_upper, q_lower, s_upper, s_lower
            real(dp) :: d_eta, d_threshold, row_weight, sample_loss
            integer :: row, class_position, threshold_position

            value = 0.0_dp
            gradient = 0.0_dp
            if (size(parameters) /= n_parameters .or. size(gradient) /= n_parameters) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal objective: parameter shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(parameters))) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal objective: parameters are nonfinite")
                return
            end if
            allocate(threshold_local(n_thresholds))
            call raw_to_threshold(parameters(n_network+1:), threshold_local, &
                objective_status)
            if (.not. status_ok(objective_status)) return
            call self%network%set_parameters(parameters(:n_network), objective_status)
            if (.not. status_ok(objective_status)) return
            allocate(eta(n_samples, 1), eta_bar(n_samples, 1), &
                threshold_bar(n_thresholds), network_gradient(n_network), x_bar(n_samples, n_features))
            call self%network%predict(x, eta, objective_status)
            if (.not. status_ok(objective_status)) return
            eta_bar = 0.0_dp
            threshold_bar = 0.0_dp
            do row = 1, n_samples
                class_position = nint(encoded(row))
                q_upper = 1.0_dp
                s_upper = 0.0_dp
                if (class_position <= n_thresholds) then
                    q_upper = stable_sigmoid(threshold_local(class_position)-eta(row,1))
                    s_upper = q_upper*(1.0_dp-q_upper)
                end if
                q_lower = 0.0_dp
                s_lower = 0.0_dp
                if (class_position > 1) then
                    q_lower = stable_sigmoid(threshold_local(class_position-1)-eta(row,1))
                    s_lower = q_lower*(1.0_dp-q_lower)
                end if
                probability = q_upper-q_lower
                if (.not. ieee_is_finite(probability) .or. probability <= tiny(1.0_dp)) then
                    call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                        "MLP ordinal objective: class probability underflowed")
                    return
                end if
                row_weight = weights(row)/weight_sum
                sample_loss = -log(probability)
                value = value + row_weight*sample_loss
                d_eta = row_weight*(s_upper-s_lower)/probability
                eta_bar(row,1) = d_eta
                if (class_position <= n_thresholds) then
                    d_threshold = -row_weight*s_upper/probability
                    threshold_bar(class_position) = threshold_bar(class_position)+d_threshold
                end if
                if (class_position > 1) then
                    d_threshold = row_weight*s_lower/probability
                    threshold_bar(class_position-1) = threshold_bar(class_position-1)+d_threshold
                end if
            end do
            call self%network%vjp(x, eta_bar, network_gradient, x_bar, objective_status)
            if (.not. status_ok(objective_status)) return
            gradient(:n_network) = network_gradient + config%l2*parameters(:n_network)
            call threshold_raw_gradient(parameters(n_network+1:), threshold_bar, &
                gradient(n_network+1:))
            value = value + 0.5_dp*config%l2*sum(parameters(:n_network)**2)
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal objective: value or gradient is nonfinite")
                return
            end if
            call status_set(objective_status, FORTNUM_OK, "")
        end subroutine ordinal_objective

    end subroutine mlp_ordinal_fit

    subroutine mlp_ordinal_decision(self, x, scores, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal decision: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
            any(shape(scores) /= [size(x, 1), 1]) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal decision: input or output shape is invalid")
            return
        end if
        call self%network%predict(x, scores, status)
    end subroutine mlp_ordinal_decision

    subroutine mlp_ordinal_decision_device(self, device, x, scores, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal decision device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP ordinal decision device: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal decision device: device kind is invalid")
        end select
    end subroutine mlp_ordinal_decision_device

    subroutine mlp_ordinal_decision_jvp(self, x, theta_dot, x_dot, scores, scores_dot, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:, :), scores_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: network_dot(:)
        if (.not. self%is_fitted .or. size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal decision JVP: model or parameter shape is invalid")
            return
        end if
        if (any(shape(scores) /= [size(x,1),1]) .or. any(shape(scores_dot) /= shape(scores)) .or. &
            any(shape(x_dot) /= shape(x)) .or. any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal decision JVP: tangent or output shape is invalid")
            return
        end if
        allocate(network_dot(self%network%parameter_count()))
        network_dot = theta_dot(:size(network_dot))
        call self%network%jvp(x, network_dot, x_dot, scores, scores_dot, status)
    end subroutine mlp_ordinal_decision_jvp

    subroutine mlp_ordinal_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: network_bar(:)
        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal decision VJP: model or parameter shape is invalid")
            return
        end if
        if (any(shape(scores_bar) /= [size(x,1),1]) .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal decision VJP: cotangent or output shape is invalid")
            return
        end if
        allocate(network_bar(self%network%parameter_count()))
        call self%network%vjp(x, scores_bar, network_bar, x_bar, status)
        if (.not. status_ok(status)) return
        theta_bar(:size(network_bar)) = network_bar
    end subroutine mlp_ordinal_decision_vjp

    subroutine mlp_ordinal_predict_proba(self, x, probabilities, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :)
        integer :: row, j
        probabilities = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal probability: model is not fitted")
            return
        end if
        if (size(x,1) < 1 .or. size(x,2) /= self%feature_count() .or. &
            any(shape(probabilities) /= [size(x,1),self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal probability: input or output shape is invalid")
            return
        end if
        allocate(scores(size(x,1),1))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        do row = 1, size(x,1)
            do j = 1, self%class_count()-1
                probabilities(row,j) = stable_sigmoid(self%threshold(j)-scores(row,1))
            end do
            probabilities(row,1) = probabilities(row,1)
            do j = 2, self%class_count()-1
                probabilities(row,j) = probabilities(row,j)-probabilities(row,j-1)
            end do
            probabilities(row,self%class_count()) = 1.0_dp - &
                stable_sigmoid(self%threshold(self%class_count()-1)-scores(row,1))
        end do
        if (any(probabilities < -1.0e-12_dp) .or. any(.not. ieee_is_finite(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal probability: nonfinite or negative probability")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_ordinal_predict_proba

    subroutine mlp_ordinal_predict_proba_device(self, device, x, probabilities, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal probability device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP ordinal probability device: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal probability device: device kind is invalid")
        end select
    end subroutine mlp_ordinal_predict_proba_device

    subroutine mlp_ordinal_predict_proba_jvp(self, x, theta_dot, x_dot, probabilities, &
            probabilities_dot, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        call mlp_ordinal_predict_proba_parameter_jvp(self, x, theta_dot, probabilities, &
            probabilities_dot, status, x_dot)
    end subroutine mlp_ordinal_predict_proba_jvp

    subroutine mlp_ordinal_predict_proba_parameter_jvp(self, x, theta_dot, probabilities, &
            probabilities_dot, status, x_dot)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: x_dot(:, :)
        real(dp), allocatable :: scores(:, :), scores_dot(:, :), q(:, :), q_dot(:, :)
        real(dp), allocatable :: network_dot(:), input_dot(:, :)
        integer :: row, j, k
        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. self%is_fitted .or. size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal probability JVP: model or parameter shape is invalid")
            return
        end if
        if (size(x,1) < 1 .or. size(x,2) /= self%feature_count() .or. &
            any(shape(probabilities) /= [size(x,1),self%class_count()]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal probability JVP: input or output shape is invalid")
            return
        end if
        allocate(network_dot(self%network%parameter_count()), input_dot(size(x,1),size(x,2)))
        network_dot = theta_dot(:size(network_dot))
        input_dot = 0.0_dp
        if (present(x_dot)) then
            if (any(shape(x_dot) /= shape(x))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal probability JVP: input tangent shape is invalid")
                return
            end if
            input_dot = x_dot
        end if
        allocate(scores(size(x,1),1), scores_dot(size(x,1),1), &
            q(size(x,1),self%class_count()-1), q_dot(size(x,1),self%class_count()-1))
        call self%network%jvp(x, network_dot, input_dot, scores, scores_dot, status)
        if (.not. status_ok(status)) return
        do row = 1, size(x,1)
            do j = 1, self%class_count()-1
                q(row,j) = stable_sigmoid(self%threshold(j)-scores(row,1))
                q_dot(row,j) = q(row,j)*(1.0_dp-q(row,j))* &
                    (theta_dot(size(network_dot)+j)-scores_dot(row,1))
            end do
            probabilities(row,1) = q(row,1)
            probabilities_dot(row,1) = q_dot(row,1)
            do k = 2, self%class_count()-1
                probabilities(row,k) = q(row,k)-q(row,k-1)
                probabilities_dot(row,k) = q_dot(row,k)-q_dot(row,k-1)
            end do
            probabilities(row,self%class_count()) = 1.0_dp-q(row,self%class_count()-1)
            probabilities_dot(row,self%class_count()) = -q_dot(row,self%class_count()-1)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_ordinal_predict_proba_parameter_jvp

    subroutine mlp_ordinal_predict_proba_vjp(self, x, probabilities_bar, theta_bar, &
            x_bar, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        call mlp_ordinal_predict_proba_parameter_vjp(self, x, probabilities_bar, theta_bar, &
            status, x_bar)
    end subroutine mlp_ordinal_predict_proba_vjp

    subroutine mlp_ordinal_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status, x_bar)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(out), optional :: x_bar(:, :)
        real(dp), allocatable :: scores(:, :), score_bar(:, :), network_bar(:), local_x_bar(:, :)
        real(dp), allocatable :: q(:), q_bar(:)
        integer :: row, j, k, network_count
        theta_bar = 0.0_dp
        if (present(x_bar)) x_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal probability VJP: model or parameter shape is invalid")
            return
        end if
        if (size(x,1) < 1 .or. size(x,2) /= self%feature_count() .or. &
            any(shape(probabilities_bar) /= [size(x,1),self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal probability VJP: input or cotangent shape is invalid")
            return
        end if
        network_count = self%network%parameter_count()
        allocate(scores(size(x,1),1), score_bar(size(x,1),1), &
            network_bar(network_count), local_x_bar(size(x,1),size(x,2)), &
            q(self%class_count()-1), q_bar(self%class_count()-1))
        call self%network%predict(x, scores, status)
        if (.not. status_ok(status)) return
        score_bar = 0.0_dp
        do row = 1, size(x,1)
            do j = 1, self%class_count()-1
                q(j) = stable_sigmoid(self%threshold(j)-scores(row,1))
            end do
            q_bar = 0.0_dp
            q_bar(1) = probabilities_bar(row,1)
            do k = 1, self%class_count()-2
                q_bar(k) = probabilities_bar(row,k)-probabilities_bar(row,k+1)
            end do
            q_bar(self%class_count()-1) = probabilities_bar(row,self%class_count()-1)- &
                probabilities_bar(row,self%class_count())
            do j = 1, self%class_count()-1
                score_bar(row,1) = score_bar(row,1) - q(j)*(1.0_dp-q(j))*q_bar(j)
                theta_bar(network_count+j) = theta_bar(network_count+j) + &
                    q(j)*(1.0_dp-q(j))*q_bar(j)
            end do
        end do
        call self%network%vjp(x, score_bar, network_bar, local_x_bar, status)
        if (.not. status_ok(status)) return
        theta_bar(:network_count) = network_bar
        if (present(x_bar)) then
            if (any(shape(x_bar) /= shape(x))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal probability VJP: input cotangent shape is invalid")
                return
            end if
            x_bar = local_x_bar
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_ordinal_predict_proba_parameter_vjp

    subroutine mlp_ordinal_predict(self, x, labels, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: row, best, j
        if (.not. self%is_fitted .or. size(labels) /= size(x,1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal predict: model or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x,1),self%class_count()))
        call self%predict_proba(x, probabilities, status)
        if (.not. status_ok(status)) return
        do row = 1, size(x,1)
            best = 1
            do j = 2, self%class_count()
                if (probabilities(row,j) > probabilities(row,best)) best = j
            end do
            labels(row) = self%class_label(best)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_ordinal_predict

    subroutine mlp_ordinal_predict_device(self, device, x, labels, status)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal predict device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP ordinal predict device: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal predict device: device kind is invalid")
        end select
    end subroutine mlp_ordinal_predict_device

    function mlp_ordinal_classes(self) result(values)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        integer, allocatable :: values(:)
        if (allocated(self%class_label)) then
            allocate(values(size(self%class_label)))
            values = self%class_label
        else
            allocate(values(0))
        end if
    end function mlp_ordinal_classes

    integer function mlp_ordinal_class_count(self) result(value)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        value = size_or_zero(self%class_label)
    end function mlp_ordinal_class_count

    integer function mlp_ordinal_feature_count(self) result(value)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        if (allocated(self%network%layer_sizes)) then
            value = self%network%layer_sizes(1)
        else
            value = 0
        end if
    end function mlp_ordinal_feature_count

    integer function mlp_ordinal_parameter_count(self) result(value)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        value = self%network%parameter_count()+size_or_zero_real(self%threshold)
    end function mlp_ordinal_parameter_count

    function mlp_ordinal_parameters(self) result(values)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:), network_values(:)
        integer :: n_network
        n_network = self%network%parameter_count()
        allocate(values(n_network+size_or_zero_real(self%threshold)))
        if (n_network > 0) then
            network_values = self%network%parameters()
            values(:n_network) = network_values
        end if
        if (allocated(self%threshold)) values(n_network+1:) = self%threshold
    end function mlp_ordinal_parameters

    subroutine mlp_ordinal_set_parameters(self, values, status)
        class(mlp_ordinal_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_network
        if (.not. allocated(self%threshold)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal set_parameters: shape or values are invalid")
            return
        end if
        if (size(values) /= self%parameter_count() .or. any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal set_parameters: shape or values are invalid")
            return
        end if
        n_network = self%network%parameter_count()
        if (size(self%threshold) > 1) then
            if (any(values(n_network+2:) <= values(n_network+1:n_network+size(self%threshold)-1))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal set_parameters: thresholds must be strictly increasing")
                return
            end if
        end if
        call self%network%set_parameters(values(:n_network), status)
        if (.not. status_ok(status)) return
        self%threshold = values(n_network+1:)
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_ordinal_set_parameters

    function mlp_ordinal_thresholds(self) result(values)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        if (allocated(self%threshold)) then
            allocate(values(size(self%threshold)))
            values = self%threshold
        else
            allocate(values(0))
        end if
    end function mlp_ordinal_thresholds

    logical function mlp_ordinal_fitted(self) result(value)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        value = self%is_fitted
    end function mlp_ordinal_fitted

    logical function mlp_ordinal_device_supported(self, device) result(value)
        class(mlp_ordinal_classifier_t), intent(in) :: self
        integer, intent(in) :: device
        value = self%is_fitted .and. device == FORTML_DEVICE_CPU
    end function mlp_ordinal_device_supported

    logical function valid_options(config) result(value)
        type(mlp_ordinal_classifier_options_t), intent(in) :: config
        value = config%max_iterations >= 1 .and. config%initialization_seed >= 0 .and. &
            config%l2 >= 0.0_dp .and. ieee_is_finite(config%l2) .and. &
            config%tolerance > 0.0_dp .and. ieee_is_finite(config%tolerance)
    end function valid_options

    subroutine raw_to_threshold(raw, threshold, status)
        real(dp), intent(in) :: raw(:)
        real(dp), intent(out) :: threshold(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: j
        if (size(raw) /= size(threshold) .or. size(raw) < 1 .or. &
            any(.not. ieee_is_finite(raw))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal thresholds: raw coordinates are invalid")
            return
        end if
        threshold(1) = raw(1)
        do j = 2, size(raw)
            if (raw(j) > log(huge(1.0_dp))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP ordinal thresholds: increment overflow")
                return
            end if
            threshold(j) = threshold(j-1)+exp(raw(j))
        end do
        if (any(.not. ieee_is_finite(threshold))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP ordinal thresholds: result is nonfinite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine raw_to_threshold

    subroutine threshold_raw_gradient(raw, threshold_gradient, raw_gradient)
        real(dp), intent(in) :: raw(:), threshold_gradient(:)
        real(dp), intent(out) :: raw_gradient(:)
        integer :: j
        real(dp) :: cumulative
        raw_gradient = 0.0_dp
        raw_gradient(1) = sum(threshold_gradient)
        do j = 2, size(raw)
            cumulative = sum(threshold_gradient(j:))
            raw_gradient(j) = exp(raw(j))*cumulative
        end do
    end subroutine threshold_raw_gradient

    pure real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp+exp(-value))
        else
            probability = exp(value)/(1.0_dp+exp(value))
        end if
    end function stable_sigmoid

    subroutine sorted_classes(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, n, temporary
        allocate(work, source=labels)
        do i = 2, size(work)
            temporary = work(i)
            j = i-1
            do while (j >= 1)
                if (work(j) <= temporary) exit
                work(j+1) = work(j)
                j = j-1
            end do
            work(j+1) = temporary
        end do
        n = 1
        do i = 2, size(work)
            if (work(i) /= work(n)) then
                n = n+1
                work(n) = work(i)
            end if
        end do
        allocate(classes(n))
        classes = work(:n)
    end subroutine sorted_classes

    integer function size_or_zero(values) result(value)
        integer, allocatable, intent(in) :: values(:)
        if (allocated(values)) then
            value = size(values)
        else
            value = 0
        end if
    end function size_or_zero

    integer function size_or_zero_real(values) result(value)
        real(dp), allocatable, intent(in) :: values(:)
        if (allocated(values)) then
            value = size(values)
        else
            value = 0
        end if
    end function size_or_zero_real

end module fortml_mlp_ordinal_classifier
