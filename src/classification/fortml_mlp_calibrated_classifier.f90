module fortml_mlp_calibrated_classifier
    !! Calibrated neural classification with a differentiable MLP base.
    !!
    !! The estimator first fits ``mlp_classifier_t`` and then fits a
    !! deterministic calibration head on its training logits.  Binary heads
    !! use the shared ``probability_calibrator_t`` (sigmoid, temperature, or
    !! weighted isotonic).  Multiclass heads use a single positive softmax
    !! temperature, which is the same scalar-temperature model used by
    !! modern neural calibration libraries.  The packed parameter vector is
    !! the MLP vector followed by the smooth calibration parameters.
    !!
    !! Sigmoid and temperature products are analytic and cover both input and
    !! packed-parameter directions.  Isotonic fitting and prediction are
    !! supported for binary models, but all derivative products explicitly
    !! refuse because the PAVA active set is discrete.  Device requests are
    !! explicit; CUDA currently returns a typed refusal rather than silently
    !! copying the batch to the host.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_losses, only: softmax_value
    use fortml_mlp_classifier, only: mlp_classifier_t, &
        mlp_classifier_options_t, mlp_classifier_state_t
    use fortml_probability_calibration, only: probability_calibrator_t, &
        probability_calibration_options_t, probability_calibration_state_t, &
        CALIBRATION_SIGMOID, CALIBRATION_ISOTONIC, CALIBRATION_TEMPERATURE
    implicit none
    private

    integer, parameter, public :: MLP_CALIBRATION_SIGMOID = CALIBRATION_SIGMOID
    integer, parameter, public :: MLP_CALIBRATION_ISOTONIC = CALIBRATION_ISOTONIC
    integer, parameter, public :: MLP_CALIBRATION_TEMPERATURE = CALIBRATION_TEMPERATURE

    type, public :: mlp_calibrated_classifier_options_t
        !! Options for the base neural classifier and calibration head.
        type(mlp_classifier_options_t) :: classifier
        type(probability_calibration_options_t) :: calibration
    end type mlp_calibrated_classifier_options_t

    type, public :: mlp_calibrated_classifier_state_t
        type(mlp_classifier_state_t) :: classifier
        type(probability_calibration_state_t) :: calibration
        integer :: calibration_iterations = 0
        logical :: classifier_converged = .false.
        logical :: calibration_converged = .false.
        logical :: converged = .false.
        real(dp) :: calibration_objective = huge(1.0_dp)
        real(dp) :: temperature = 1.0_dp
    end type mlp_calibrated_classifier_state_t

    type, public :: mlp_calibrated_classifier_t
        private
        type(mlp_classifier_t) :: classifier
        type(probability_calibrator_t) :: binary_calibrator
        integer :: calibration_method_code = CALIBRATION_TEMPERATURE
        real(dp) :: multiclass_temperature = 1.0_dp
        logical :: multiclass_temperature_fitted = .false.
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => mlp_calibrated_classifier_fit
        procedure, public :: decision_function => mlp_calibrated_decision
        procedure, public :: decision_function_device => mlp_calibrated_decision_device
        procedure, public :: decision_function_jvp => mlp_calibrated_decision_jvp
        procedure, public :: decision_function_vjp => mlp_calibrated_decision_vjp
        procedure, public :: predict_proba => mlp_calibrated_predict_proba
        procedure, public :: predict_proba_device => mlp_calibrated_predict_proba_device
        procedure, public :: predict_proba_jvp => mlp_calibrated_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            mlp_calibrated_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => mlp_calibrated_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            mlp_calibrated_predict_proba_parameter_vjp
        procedure, public :: predict => mlp_calibrated_predict
        procedure, public :: predict_device => mlp_calibrated_predict_device
        procedure, public :: classes => mlp_calibrated_classes
        procedure, public :: class_count => mlp_calibrated_class_count
        procedure, public :: feature_count => mlp_calibrated_feature_count
        procedure, public :: parameter_count => mlp_calibrated_parameter_count
        procedure, public :: parameters => mlp_calibrated_parameters
        procedure, public :: set_parameters => mlp_calibrated_set_parameters
        procedure, public :: fitted => mlp_calibrated_fitted
        procedure, public :: device_supported => mlp_calibrated_device_supported
        procedure, public :: calibration_method => mlp_calibrated_method
        procedure, public :: temperature => mlp_calibrated_temperature
    end type mlp_calibrated_classifier_t

    public :: mlp_calibrated_classifier_fit
    public :: mlp_calibrated_decision
    public :: mlp_calibrated_decision_device
    public :: mlp_calibrated_decision_jvp
    public :: mlp_calibrated_decision_vjp
    public :: mlp_calibrated_predict_proba
    public :: mlp_calibrated_predict_proba_device
    public :: mlp_calibrated_predict_proba_jvp
    public :: mlp_calibrated_predict_proba_parameter_jvp
    public :: mlp_calibrated_predict_proba_vjp
    public :: mlp_calibrated_predict_proba_parameter_vjp
    public :: mlp_calibrated_predict
    public :: mlp_calibrated_predict_device

contains

    subroutine mlp_calibrated_classifier_fit(self, x, labels, status, hidden_layer_sizes, &
            options, state, sample_weight, class_weight)
        class(mlp_calibrated_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_layer_sizes(:)
        type(mlp_calibrated_classifier_options_t), intent(in), optional :: options
        type(mlp_calibrated_classifier_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:)
        type(mlp_calibrated_classifier_options_t) :: config
        type(mlp_calibrated_classifier_state_t) :: result
        type(mlp_classifier_state_t) :: classifier_state
        type(probability_calibration_state_t) :: calibration_state
        real(dp), allocatable :: scores(:, :), margin(:), weights(:), calibration_parameters(:)
        integer, allocatable :: classes(:), encoded(:)
        integer :: n_classes
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(mlp_calibrated_classifier_options_t) :: mlp_calibrated_classifier_options_t_default
        type(mlp_calibrated_classifier_state_t) :: mlp_calibrated_classifier_state_t_default

        self%is_fitted = .false.
        self%multiclass_temperature_fitted = .false.
        self%multiclass_temperature = 1.0_dp
        config = mlp_calibrated_classifier_options_t_default
        if (present(options)) config = options
        result = mlp_calibrated_classifier_state_t_default
        if (.not. valid_calibration_options(config%calibration)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP fit: calibration options are invalid")
            if (present(state)) state = result
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(labels) /= size(x, 1) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP fit: input dimensions or values are invalid")
            if (present(state)) state = result
            return
        end if
        classes = sorted_classes(labels)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP fit: at least two classes are required")
            if (present(state)) state = result
            return
        end if
        if (n_classes > 2 .and. config%calibration%method /= CALIBRATION_TEMPERATURE) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "calibrated MLP fit: multiclass calibration currently supports temperature only")
            if (present(state)) state = result
            return
        end if

        if (present(hidden_layer_sizes)) then
            call self%classifier%fit(x, labels, status, hidden_layer_sizes=hidden_layer_sizes, &
                options=config%classifier, state=classifier_state, sample_weight=sample_weight, &
                class_weight=class_weight)
        else
            call self%classifier%fit(x, labels, status, options=config%classifier, &
                state=classifier_state, sample_weight=sample_weight, class_weight=class_weight)
        end if
        result%classifier = classifier_state
        result%classifier_converged = classifier_state%converged
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        allocate(scores(size(x, 1), n_classes))
        call self%classifier%decision_function(x, scores, status)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        call effective_weights(labels, classes, sample_weight, class_weight, weights, status)
        if (.not. status_ok(status)) then
            if (present(state)) state = result
            return
        end if
        self%calibration_method_code = config%calibration%method
        if (n_classes == 2) then
            allocate(margin(size(labels)))
            margin = scores(:, 2) - scores(:, 1)
            call self%binary_calibrator%fit(margin, labels, status, &
                options=config%calibration, sample_weight=weights, state=calibration_state)
            result%calibration = calibration_state
            result%calibration_iterations = calibration_state%iterations
            result%calibration_converged = calibration_state%converged
            result%calibration_objective = calibration_state%objective
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
            if (config%calibration%method == CALIBRATION_TEMPERATURE) then
                calibration_parameters = self%binary_calibrator%parameters()
                result%temperature = calibration_parameters(1)
            else
                result%temperature = 1.0_dp
            end if
        else
            call fit_multiclass_temperature(self, scores, labels, classes, weights, &
                config%calibration, result, status)
            if (.not. status_ok(status)) then
                if (present(state)) state = result
                return
            end if
        end if
        self%is_fitted = .true.
        result%converged = result%classifier_converged .and. result%calibration_converged
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_calibrated_classifier_fit

    subroutine mlp_calibrated_decision(self, x, scores, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP decision: model is not fitted")
            return
        end if
        call self%classifier%decision_function(x, scores, status)
    end subroutine mlp_calibrated_decision

    subroutine mlp_calibrated_decision_device(self, device, x, scores, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "calibrated MLP device decision: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP device decision: device kind is invalid")
        end select
    end subroutine mlp_calibrated_decision_device

    subroutine mlp_calibrated_decision_jvp(self, x, theta_dot, x_dot, scores, scores_dot, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:, :), scores_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_network
        real(dp), allocatable :: network_dot(:)

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP decision JVP: model is not fitted")
            return
        end if
        n_network = self%classifier%parameter_count()
        if (size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP decision JVP: parameter tangent has invalid size")
            return
        end if
        allocate(network_dot(n_network))
        network_dot = theta_dot(:n_network)
        call self%classifier%decision_function_jvp(x, network_dot, x_dot, scores, scores_dot, status)
    end subroutine mlp_calibrated_decision_jvp

    subroutine mlp_calibrated_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_network
        real(dp), allocatable :: network_bar(:)

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP decision VJP: model is not fitted")
            return
        end if
        n_network = self%classifier%parameter_count()
        if (size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP decision VJP: parameter cotangent has invalid size")
            return
        end if
        allocate(network_bar(n_network))
        call self%classifier%decision_function_vjp(x, scores_bar, network_bar, x_bar, status)
        if (status_ok(status)) theta_bar(:n_network) = network_bar
    end subroutine mlp_calibrated_decision_vjp

    subroutine mlp_calibrated_predict_proba(self, x, probabilities, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), scaled(:, :), margin(:), binary(:,:)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP probability: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP probability: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%class_count()))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        if (self%class_count() == 2) then
            allocate(margin(size(x, 1)), binary(size(x, 1), 2))
            margin = scores(:, 2) - scores(:, 1)
            call self%binary_calibrator%predict_proba(margin, binary, status)
            if (status_ok(status)) probabilities = binary
        else
            allocate(scaled(size(x, 1), self%class_count()))
            scaled = scores/self%multiclass_temperature
            call softmax_value(scaled, probabilities, status)
        end if
    end subroutine mlp_calibrated_predict_proba

    subroutine mlp_calibrated_predict_proba_device(self, device, x, probabilities, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "calibrated MLP device probability: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP device probability: device kind is invalid")
        end select
    end subroutine mlp_calibrated_predict_proba_device

    subroutine mlp_calibrated_predict_proba_jvp(self, x, theta_dot, x_dot, probabilities, &
            probabilities_dot, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), scores_dot(:, :), margin(:), margin_dot(:)
        real(dp), allocatable :: binary(:,:), binary_dot(:,:), calibration_dot(:,:), &
            cal_dot(:), scaled(:,:)
        integer :: n_network, i
        real(dp) :: alpha, alpha_dot, p_dot, dotp

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP probability JVP: model is not fitted")
            return
        end if
        if (size(theta_dot) /= self%parameter_count() .or. any(shape(probabilities) /= &
            [size(x, 1), self%class_count()]) .or. any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP probability JVP: input or output shape is invalid")
            return
        end if
        if (self%calibration_method_code == CALIBRATION_ISOTONIC) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "calibrated MLP probability JVP: isotonic PAVA active set is discrete")
            return
        end if
        allocate(scores(size(x, 1), self%class_count()), scores_dot(size(x, 1), self%class_count()))
        call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        if (.not. status_ok(status)) return
        n_network = self%classifier%parameter_count()
        if (self%class_count() == 2) then
            allocate(margin(size(x, 1)), margin_dot(size(x, 1)), binary(size(x, 1), 2), &
                binary_dot(size(x, 1), 2), calibration_dot(size(x, 1), 2), &
                cal_dot(max(1, self%binary_calibrator%parameter_count())))
            margin = scores(:, 2) - scores(:, 1)
            margin_dot = scores_dot(:, 2) - scores_dot(:, 1)
            call self%binary_calibrator%predict_proba_jvp(margin, margin_dot, binary, &
                binary_dot, status)
            if (.not. status_ok(status)) return
            if (self%binary_calibrator%parameter_count() > 0) then
                cal_dot(:self%binary_calibrator%parameter_count()) = theta_dot(n_network+1:)
                call self%binary_calibrator%predict_proba_parameter_jvp(margin, cal_dot, binary, &
                    calibration_dot, status)
                if (.not. status_ok(status)) return
                binary_dot = binary_dot + calibration_dot
            end if
            probabilities = binary
            probabilities_dot = binary_dot
        else
            alpha = 1.0_dp/self%multiclass_temperature
            alpha_dot = 0.0_dp
            if (self%parameter_count() > n_network) alpha_dot = -theta_dot(n_network+1)/ &
                self%multiclass_temperature**2
            allocate(scaled(size(x, 1), self%class_count()))
            scaled = alpha*scores
            call softmax_value(scaled, probabilities, status)
            if (.not. status_ok(status)) return
            do i = 1, size(x, 1)
                dotp = sum(probabilities(i, :)*(alpha*scores_dot(i, :) + alpha_dot*scores(i, :)))
                probabilities_dot(i, :) = probabilities(i, :)*(alpha*scores_dot(i, :) + &
                    alpha_dot*scores(i, :) - dotp)
            end do
        end if
    end subroutine mlp_calibrated_predict_proba_jvp

    subroutine mlp_calibrated_predict_proba_parameter_jvp(self, x, theta_dot, probabilities, &
            probabilities_dot, status, x_dot)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: x_dot(:, :)
        real(dp), allocatable :: input_dot(:, :)

        allocate(input_dot(size(x, 1), size(x, 2)))
        input_dot = 0.0_dp
        if (present(x_dot)) then
            if (any(shape(x_dot) /= shape(x))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated MLP parameter JVP: input tangent shape is invalid")
                probabilities = 0.0_dp
                probabilities_dot = 0.0_dp
                return
            end if
            input_dot = x_dot
        end if
        call self%predict_proba_jvp(x, theta_dot, input_dot, probabilities, probabilities_dot, status)
    end subroutine mlp_calibrated_predict_proba_parameter_jvp

    subroutine mlp_calibrated_predict_proba_vjp(self, x, probabilities_bar, theta_bar, x_bar, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), logits_bar(:, :), margin(:), margin_bar(:)
        real(dp), allocatable :: binary(:,:), cal_bar(:), network_bar(:)
        integer :: n_network, i
        real(dp) :: alpha, alpha_bar, dotp

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP probability VJP: model is not fitted")
            return
        end if
        if (size(theta_bar) /= self%parameter_count() .or. any(shape(probabilities_bar) /= &
            [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP probability VJP: input or output shape is invalid")
            return
        end if
        if (self%calibration_method_code == CALIBRATION_ISOTONIC) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "calibrated MLP probability VJP: isotonic PAVA active set is discrete")
            return
        end if
        allocate(scores(size(x, 1), self%class_count()), logits_bar(size(x, 1), self%class_count()))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        n_network = self%classifier%parameter_count()
        if (self%class_count() == 2) then
            allocate(margin(size(x, 1)), margin_bar(size(x, 1)), binary(size(x, 1), 2), &
                cal_bar(max(1, self%binary_calibrator%parameter_count())), network_bar(n_network))
            margin = scores(:, 2)-scores(:, 1)
            call self%binary_calibrator%predict_proba(margin, binary, status)
            if (.not. status_ok(status)) return
            call self%binary_calibrator%predict_proba_vjp(margin, probabilities_bar, margin_bar, status)
            if (.not. status_ok(status)) return
            logits_bar(:, 1) = -margin_bar
            logits_bar(:, 2) = margin_bar
            call self%classifier%decision_function_vjp(x, logits_bar, network_bar, x_bar, status)
            if (.not. status_ok(status)) return
            theta_bar(:n_network) = network_bar
            if (self%binary_calibrator%parameter_count() > 0) then
                call self%binary_calibrator%predict_proba_parameter_vjp(margin, probabilities_bar, &
                    cal_bar, status)
                if (.not. status_ok(status)) return
                theta_bar(n_network+1:) = cal_bar(:self%binary_calibrator%parameter_count())
            end if
        else
            alpha = 1.0_dp/self%multiclass_temperature
            call scaled_softmax_vjp(scores, probabilities_bar, alpha, logits_bar, alpha_bar, status)
            if (.not. status_ok(status)) return
            allocate(network_bar(n_network))
            call self%classifier%decision_function_vjp(x, logits_bar, network_bar, x_bar, status)
            if (.not. status_ok(status)) return
            theta_bar(:n_network) = network_bar
            if (self%parameter_count() > n_network) theta_bar(n_network+1) = -alpha_bar/ &
                self%multiclass_temperature**2
        end if
    end subroutine mlp_calibrated_predict_proba_vjp

    subroutine mlp_calibrated_predict_proba_parameter_vjp(self, x, probabilities_bar, theta_bar, &
            status, x_bar)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(out), optional :: x_bar(:, :)
        real(dp), allocatable :: local_x_bar(:, :)

        allocate(local_x_bar(size(x, 1), size(x, 2)))
        call self%predict_proba_vjp(x, probabilities_bar, theta_bar, local_x_bar, status)
        if (present(x_bar)) then
            if (any(shape(x_bar) /= shape(x))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated MLP parameter VJP: input cotangent shape is invalid")
                return
            end if
            x_bar = local_x_bar
        end if
    end subroutine mlp_calibrated_predict_proba_parameter_vjp

    subroutine mlp_calibrated_predict(self, x, labels, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer, allocatable :: classes(:)
        integer :: i, j

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%class_count()))
        call self%predict_proba(x, probabilities, status)
        if (.not. status_ok(status)) return
        classes = self%classes()
        do i = 1, size(labels)
            j = maxloc(probabilities(i, :), dim=1)
            labels(i) = classes(j)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_calibrated_predict

    subroutine mlp_calibrated_predict_device(self, device, x, labels, status)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP device predict: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "calibrated MLP device predict: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP device predict: device kind is invalid")
        end select
    end subroutine mlp_calibrated_predict_device

    function mlp_calibrated_classes(self) result(classes)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        integer, allocatable :: classes(:)
        classes = self%classifier%classes()
    end function mlp_calibrated_classes

    integer function mlp_calibrated_class_count(self) result(count)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        count = self%classifier%class_count()
    end function mlp_calibrated_class_count

    integer function mlp_calibrated_feature_count(self) result(count)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        count = self%classifier%feature_count()
    end function mlp_calibrated_feature_count

    integer function mlp_calibrated_parameter_count(self) result(count)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        if (.not. self%is_fitted) then
            count = 0
        else if (self%class_count() == 2) then
            count = self%classifier%parameter_count() + self%binary_calibrator%parameter_count()
        else if (self%multiclass_temperature_fitted) then
            count = self%classifier%parameter_count() + 1
        else
            count = self%classifier%parameter_count()
        end if
    end function mlp_calibrated_parameter_count

    function mlp_calibrated_parameters(self) result(values)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:), network(:), calibration(:)
        integer :: n_network, n_calibration

        if (.not. self%is_fitted) then
            allocate(values(0))
            return
        end if
        network = self%classifier%parameters()
        n_network = size(network)
        if (self%class_count() == 2) then
            calibration = self%binary_calibrator%parameters()
        else
            allocate(calibration(1))
            calibration = self%multiclass_temperature
        end if
        n_calibration = size(calibration)
        allocate(values(n_network+n_calibration))
        values(:n_network) = network
        if (n_calibration > 0) values(n_network+1:) = calibration
    end function mlp_calibrated_parameters

    subroutine mlp_calibrated_set_parameters(self, values, status)
        class(mlp_calibrated_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_network

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP set_parameters: model or parameter vector is invalid")
            return
        end if
        n_network = self%classifier%parameter_count()
        call self%classifier%set_parameters(values(:n_network), status)
        if (.not. status_ok(status)) return
        if (self%class_count() == 2) then
            if (self%binary_calibrator%parameter_count() > 0) then
                call self%binary_calibrator%set_parameters(values(n_network+1:), status)
            end if
        else
            if (values(n_network+1) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated MLP set_parameters: temperature must be positive")
                return
            end if
            self%multiclass_temperature = values(n_network+1)
        end if
    end subroutine mlp_calibrated_set_parameters

    logical function mlp_calibrated_fitted(self) result(fitted)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        fitted = self%is_fitted .and. self%classifier%fitted()
        if (fitted .and. self%class_count() == 2) fitted = self%binary_calibrator%fitted()
        if (fitted .and. self%class_count() > 2) fitted = self%multiclass_temperature_fitted
    end function mlp_calibrated_fitted

    logical function mlp_calibrated_device_supported(self, device_kind) result(supported)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%fitted()
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function mlp_calibrated_device_supported

    integer function mlp_calibrated_method(self) result(method)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        method = self%calibration_method_code
    end function mlp_calibrated_method

    real(dp) function mlp_calibrated_temperature(self) result(value)
        class(mlp_calibrated_classifier_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        if (self%class_count() == 2 .and. self%calibration_method_code == CALIBRATION_TEMPERATURE) then
            parameters = self%binary_calibrator%parameters()
            value = parameters(1)
        else if (self%class_count() > 2 .and. self%multiclass_temperature_fitted) then
            value = self%multiclass_temperature
        else
            value = huge(1.0_dp)
        end if
    end function mlp_calibrated_temperature

    subroutine fit_multiclass_temperature(self, scores, labels, classes, weights, options, result, status)
        class(mlp_calibrated_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: scores(:, :), weights(:)
        integer, intent(in) :: labels(:), classes(:)
        type(probability_calibration_options_t), intent(in) :: options
        type(mlp_calibrated_classifier_state_t), intent(inout) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), encoded(:)
        real(dp) :: alpha, alpha_trial, gradient, hessian, step, objective, objective_trial
        real(dp) :: scale, step_norm, alpha_floor
        integer :: iteration, line_search, i, j, index

        allocate(probabilities(size(scores, 1), size(scores, 2)), encoded(size(labels)))
        do i = 1, size(labels)
            index = 0
            do j = 1, size(classes)
                if (labels(i) == classes(j)) then
                    index = j
                    exit
                end if
            end do
            if (index < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated MLP temperature fit: could not encode labels")
                return
            end if
            encoded(i) = real(index, dp)
        end do
        alpha_floor = sqrt(tiny(1.0_dp))
        alpha = 1.0_dp
        objective = multiclass_temperature_objective(scores, encoded, weights, alpha, options%l2, probabilities)
        result%calibration_iterations = 0
        result%calibration_converged = .false.
        do iteration = 1, options%max_iterations
            call multiclass_temperature_derivatives(scores, encoded, weights, alpha, options%l2, &
                probabilities, gradient, hessian)
            if (.not. ieee_is_finite(gradient) .or. .not. ieee_is_finite(hessian) .or. hessian <= 0.0_dp) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "calibrated MLP temperature fit: invalid Newton curvature")
                return
            end if
            step = gradient/hessian
            scale = 1.0_dp
            alpha_trial = max(alpha_floor, alpha-scale*options%damping*step)
            objective_trial = multiclass_temperature_objective(scores, encoded, weights, alpha_trial, &
                options%l2, probabilities)
            do line_search = 1, 30
                if (objective_trial <= objective .or. scale <= 1.0e-8_dp) exit
                scale = 0.5_dp*scale
                alpha_trial = max(alpha_floor, alpha-scale*options%damping*step)
                objective_trial = multiclass_temperature_objective(scores, encoded, weights, alpha_trial, &
                    options%l2, probabilities)
            end do
            step_norm = abs(alpha_trial-alpha)/max(1.0_dp, abs(alpha))
            alpha = alpha_trial
            objective = objective_trial
            result%calibration_iterations = iteration
            if (step_norm <= options%tolerance .or. abs(gradient) <= options%tolerance) then
                result%calibration_converged = .true.
                exit
            end if
        end do
        if (.not. result%calibration_converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "calibrated MLP temperature fit: iteration limit reached")
            return
        end if
        self%multiclass_temperature = 1.0_dp/alpha
        self%multiclass_temperature_fitted = .true.
        result%calibration%method = CALIBRATION_TEMPERATURE
        result%calibration%iterations = result%calibration_iterations
        result%calibration%converged = .true.
        result%calibration%final_step_norm = step_norm
        result%calibration%objective = objective
        result%calibration_objective = objective
        result%temperature = self%multiclass_temperature
        call status_set(status, FORTNUM_OK, "")
    end subroutine fit_multiclass_temperature

    real(dp) function multiclass_temperature_objective(scores, encoded, weights, alpha, l2, probabilities) result(value)
        real(dp), intent(in) :: scores(:, :), encoded(:), weights(:), alpha, l2
        real(dp), intent(out) :: probabilities(:, :)
        real(dp), allocatable :: scaled(:, :)
        real(dp) :: max_score, normalizer
        integer :: i, j, class_index

        value = 0.0_dp
        allocate(scaled(size(scores, 1), size(scores, 2)))
        scaled = alpha*scores
        do i = 1, size(scores, 1)
            max_score = maxval(scaled(i, :))
            normalizer = sum(exp(scaled(i, :)-max_score))
            probabilities(i, :) = exp(scaled(i, :)-max_score)/normalizer
            class_index = int(encoded(i))
            value = value + weights(i)*(log(normalizer)+max_score-scaled(i, class_index))
        end do
        value = value/sum(weights) + 0.5_dp*l2*alpha*alpha
    end function multiclass_temperature_objective

    subroutine multiclass_temperature_derivatives(scores, encoded, weights, alpha, l2, probabilities, gradient, hessian)
        real(dp), intent(in) :: scores(:, :), encoded(:), weights(:), alpha, l2
        real(dp), intent(in) :: probabilities(:, :)
        real(dp), intent(out) :: gradient, hessian
        integer :: i, class_index
        real(dp) :: mean_score, mean_square

        gradient = l2*alpha
        hessian = l2
        do i = 1, size(scores, 1)
            class_index = int(encoded(i))
            mean_score = sum(probabilities(i, :)*scores(i, :))
            mean_square = sum(probabilities(i, :)*scores(i, :)*scores(i, :))
            gradient = gradient + weights(i)*(mean_score-scores(i, class_index))/sum(weights)
            hessian = hessian + weights(i)*(mean_square-mean_score*mean_score)/sum(weights)
        end do
    end subroutine multiclass_temperature_derivatives

    subroutine scaled_softmax_vjp(scores, probabilities_bar, alpha, scores_bar, alpha_bar, status)
        real(dp), intent(in) :: scores(:, :), probabilities_bar(:, :), alpha
        real(dp), intent(out) :: scores_bar(:, :), alpha_bar
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        real(dp) :: dotp, eta_bar
        integer :: i, j

        if (any(shape(probabilities_bar) /= shape(scores)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "calibrated MLP scaled softmax VJP: shape is invalid")
            return
        end if
        allocate(probabilities(size(scores, 1), size(scores, 2)))
        call softmax_value(alpha*scores, probabilities, status)
        if (.not. status_ok(status)) return
        scores_bar = 0.0_dp
        alpha_bar = 0.0_dp
        do i = 1, size(scores, 1)
            dotp = sum(probabilities(i, :)*probabilities_bar(i, :))
            do j = 1, size(scores, 2)
                eta_bar = probabilities(i, j)*(probabilities_bar(i, j)-dotp)
                scores_bar(i, j) = alpha*eta_bar
                alpha_bar = alpha_bar + scores(i, j)*eta_bar
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine scaled_softmax_vjp

    subroutine effective_weights(labels, classes, sample_weight, class_weight, weights, status)
        integer, intent(in) :: labels(:), classes(:)
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        real(dp), allocatable, intent(out) :: weights(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        allocate(weights(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated MLP fit: sample weights are invalid")
                return
            end if
            weights = sample_weight
        end if
        if (present(class_weight)) then
            if (size(class_weight) /= size(classes) .or. any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "calibrated MLP fit: class weights are invalid")
                return
            end if
            do i = 1, size(labels)
                do j = 1, size(classes)
                    if (labels(i) == classes(j)) then
                        weights(i) = weights(i)*class_weight(j)
                        exit
                    end if
                end do
            end do
        end if
        if (any(.not. ieee_is_finite(weights)) .or. .not. ieee_is_finite(sum(weights)) .or. &
            sum(weights) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "calibrated MLP fit: effective weights have no positive mass")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine effective_weights

    function sorted_classes(labels) result(classes)
        integer, intent(in) :: labels(:)
        integer, allocatable :: classes(:)
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
    end function sorted_classes

    logical function valid_calibration_options(options) result(valid)
        type(probability_calibration_options_t), intent(in) :: options
        valid = options%method == CALIBRATION_SIGMOID .or. options%method == CALIBRATION_ISOTONIC .or. &
            options%method == CALIBRATION_TEMPERATURE
        valid = valid .and. options%max_iterations >= 1 .and. options%tolerance >= 0.0_dp .and. &
            options%damping > 0.0_dp .and. options%l2 >= 0.0_dp
        valid = valid .and. ieee_is_finite(options%tolerance) .and. ieee_is_finite(options%damping) .and. &
            ieee_is_finite(options%l2)
    end function valid_calibration_options

end module fortml_mlp_calibrated_classifier
