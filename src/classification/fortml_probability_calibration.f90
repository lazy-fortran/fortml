module fortml_probability_calibration
    !! Binary probability calibration with explicit derivative contracts.
    !!
    !! ``probability_calibrator_t`` implements positive-temperature scaling,
    !! Platt sigmoid calibration, and weighted isotonic calibration for a
    !! scalar decision score.  Labels are
    !! arbitrary integers; the stored class order is ascending and column two
    !! is the calibrated positive probability.  Temperature and sigmoid fits
    !! are smooth and expose score and parameter products.  Isotonic prediction is linearly
    !! interpolated between weighted PAVA knots and exposes the exact score
    !! derivative away from knots; products at a knot are refused because the
    !! active interpolation segment is not unique.
    !!
    !! Calibration fitting is intentionally a host operation.  Prediction is
    !! available on a selected CPU context; CUDA returns a typed refusal until
    !! a resident calibration kernel is linked.  No hidden host fallback is
    !! performed for accelerator requests.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    integer, parameter, public :: CALIBRATION_SIGMOID = 1
    integer, parameter, public :: CALIBRATION_ISOTONIC = 2
    integer, parameter, public :: CALIBRATION_TEMPERATURE = 3

    type, public :: probability_calibration_options_t
        integer :: method = CALIBRATION_SIGMOID
        integer :: max_iterations = 100
        real(dp) :: tolerance = 1.0e-10_dp
        real(dp) :: damping = 1.0_dp
        real(dp) :: l2 = 1.0e-8_dp
    end type probability_calibration_options_t

    type, public :: probability_calibration_state_t
        integer :: method = CALIBRATION_SIGMOID
        integer :: iterations = 0
        integer :: knot_count = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: final_step_norm = huge(1.0_dp)
        logical :: converged = .false.
    end type probability_calibration_state_t

    type, public :: probability_calibrator_t
        private
        integer :: calibration_method = CALIBRATION_SIGMOID
        integer :: class_label(2) = 0
        real(dp) :: temperature = 1.0_dp
        real(dp) :: sigmoid_slope = 0.0_dp
        real(dp) :: sigmoid_intercept = 0.0_dp
        real(dp), allocatable :: knots(:)
        real(dp), allocatable :: knot_values(:)
        integer :: n_knots = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => probability_calibration_fit
        procedure, public :: predict_proba => probability_calibration_predict_proba
        procedure, public :: predict_proba_device => &
            probability_calibration_predict_proba_device
        procedure, public :: predict_proba_jvp => probability_calibration_predict_proba_jvp
        procedure, public :: predict_proba_vjp => probability_calibration_predict_proba_vjp
        procedure, public :: predict_proba_parameter_jvp => &
            probability_calibration_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            probability_calibration_predict_proba_parameter_vjp
        procedure, public :: predict => probability_calibration_predict
        procedure, public :: set_parameters => probability_calibration_set_parameters
        procedure, public :: parameters => probability_calibration_parameters
        procedure, public :: parameter_count => probability_calibration_parameter_count
        procedure, public :: classes => probability_calibration_classes
        procedure, public :: method => probability_calibration_method
        procedure, public :: fitted => probability_calibration_fitted
        procedure, public :: device_supported => probability_calibration_device_supported
    end type probability_calibrator_t

    type, public :: multiclass_probability_calibrator_t
        !! Multiclass positive-temperature calibration for logit matrices.
        !!
        !! The input rows are logits in ascending sorted-class column order.
        !! Fit learns one positive scalar temperature from a weighted softmax
        !! NLL.  All products are analytic for fixed fitted state; Platt and
        !! isotonic multiclass policies remain explicit capability refusals.
        private
        real(dp) :: temperature = 1.0_dp
        integer, allocatable :: class_label(:)
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => multiclass_probability_calibration_fit
        procedure, public :: predict_proba => &
            multiclass_probability_calibration_predict_proba
        procedure, public :: predict_proba_device => &
            multiclass_probability_calibration_predict_proba_device
        procedure, public :: predict_proba_jvp => &
            multiclass_probability_calibration_predict_proba_jvp
        procedure, public :: predict_proba_vjp => &
            multiclass_probability_calibration_predict_proba_vjp
        procedure, public :: predict_proba_parameter_jvp => &
            multiclass_probability_calibration_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            multiclass_probability_calibration_predict_proba_parameter_vjp
        procedure, public :: predict => multiclass_probability_calibration_predict
        procedure, public :: set_parameters => &
            multiclass_probability_calibration_set_parameters
        procedure, public :: parameters => multiclass_probability_calibration_parameters
        procedure, public :: parameter_count => &
            multiclass_probability_calibration_parameter_count
        procedure, public :: classes => multiclass_probability_calibration_classes
        procedure, public :: method => multiclass_probability_calibration_method
        procedure, public :: fitted => multiclass_probability_calibration_fitted
        procedure, public :: device_supported => &
            multiclass_probability_calibration_device_supported
    end type multiclass_probability_calibrator_t

    public :: probability_calibration_fit
    public :: probability_calibration_predict_proba
    public :: probability_calibration_predict_proba_device
    public :: probability_calibration_predict_proba_jvp
    public :: probability_calibration_predict_proba_vjp
    public :: probability_calibration_predict_proba_parameter_jvp
    public :: probability_calibration_predict_proba_parameter_vjp
    public :: probability_calibration_predict
    public :: multiclass_probability_calibration_fit
    public :: multiclass_probability_calibration_predict_proba
    public :: multiclass_probability_calibration_predict_proba_device
    public :: multiclass_probability_calibration_predict_proba_jvp
    public :: multiclass_probability_calibration_predict_proba_vjp
    public :: multiclass_probability_calibration_predict_proba_parameter_jvp
    public :: multiclass_probability_calibration_predict_proba_parameter_vjp
    public :: multiclass_probability_calibration_predict

contains

    subroutine multiclass_probability_calibration_fit(self, scores, labels, status, &
            options, sample_weight, state)
        class(multiclass_probability_calibrator_t), intent(out) :: self
        real(dp), intent(in) :: scores(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(probability_calibration_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        type(probability_calibration_state_t), intent(out), optional :: state
        type(probability_calibration_options_t) :: requested
        type(probability_calibration_state_t) :: result
        integer, allocatable :: classes(:), encoded(:)
        real(dp), allocatable :: weights(:)
        real(dp) :: total_weight
        integer :: i, j

        self%is_fitted = .false.
        self%temperature = 1.0_dp
        if (allocated(self%class_label)) deallocate(self%class_label)
        requested = probability_calibration_options_t()
        if (present(options)) requested = options
        result = probability_calibration_state_t()
        result%method = CALIBRATION_TEMPERATURE
        if (present(state)) state = result
        if (.not. valid_options(requested)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration fit: options are invalid")
            return
        end if
        if (requested%method /= CALIBRATION_TEMPERATURE) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multiclass probability calibration fit: only temperature scaling is implemented")
            return
        end if
        if (size(scores, 1) < 1 .or. size(scores, 2) < 2 .or. &
            size(labels) /= size(scores, 1) .or. any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration fit: scores and labels have invalid shape")
            return
        end if
        call sorted_unique_labels(labels, classes)
        if (size(classes) /= size(scores, 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration fit: logit columns must match sorted classes")
            return
        end if
        allocate(weights(size(labels)), encoded(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multiclass probability calibration fit: weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multiclass probability calibration fit: weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        total_weight = sum(weights)
        if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration fit: weights need positive total mass")
            return
        end if
        do j = 1, size(classes)
            if (sum(weights, mask=labels == classes(j)) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multiclass probability calibration fit: every class needs positive weight")
                return
            end if
        end do
        do i = 1, size(labels)
            encoded(i) = 0
            do j = 1, size(classes)
                if (labels(i) == classes(j)) then
                    encoded(i) = j
                    exit
                end if
            end do
            if (encoded(i) == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multiclass probability calibration fit: label encoding failed")
                return
            end if
        end do
        allocate(self%class_label(size(classes)))
        self%class_label = classes
        call fit_multiclass_temperature(self, scores, encoded, weights, total_weight, &
            requested, result, status)
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if
        self%is_fitted = .true.
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_probability_calibration_fit

    subroutine fit_multiclass_temperature(self, scores, encoded, weights, total_weight, &
            options, state, status)
        class(multiclass_probability_calibrator_t), intent(inout) :: self
        real(dp), intent(in) :: scores(:, :), weights(:), total_weight
        integer, intent(in) :: encoded(:)
        type(probability_calibration_options_t), intent(in) :: options
        type(probability_calibration_state_t), intent(inout) :: state
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        real(dp) :: alpha, alpha_trial, objective, objective_trial
        real(dp) :: gradient, hessian, step, step_scale, step_norm
        real(dp) :: alpha_floor
        integer :: iteration, line_search

        allocate(probabilities(size(scores, 1), size(scores, 2)))
        alpha_floor = sqrt(tiny(1.0_dp))
        alpha = 1.0_dp
        call multiclass_temperature_objective(scores, encoded, weights, total_weight, &
            alpha, options%l2, probabilities, objective, status)
        if (status%code /= FORTNUM_OK) return
        state%iterations = 0
        state%final_step_norm = huge(1.0_dp)
        state%converged = .false.
        do iteration = 1, options%max_iterations
            call multiclass_temperature_derivatives(scores, encoded, weights, total_weight, &
                alpha, options%l2, probabilities, gradient, hessian)
            if (.not. ieee_is_finite(gradient) .or. .not. ieee_is_finite(hessian) .or. &
                hessian <= 0.0_dp) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "multiclass probability calibration fit: invalid Newton curvature")
                return
            end if
            step = gradient/hessian
            step_scale = 1.0_dp
            alpha_trial = max(alpha_floor, alpha - step_scale*options%damping*step)
            call multiclass_temperature_objective(scores, encoded, weights, total_weight, &
                alpha_trial, options%l2, probabilities, objective_trial, status)
            if (status%code /= FORTNUM_OK) return
            do line_search = 1, 30
                if (objective_trial <= objective .or. step_scale <= 1.0e-8_dp) exit
                step_scale = 0.5_dp*step_scale
                alpha_trial = max(alpha_floor, alpha - step_scale*options%damping*step)
                call multiclass_temperature_objective(scores, encoded, weights, total_weight, &
                    alpha_trial, options%l2, probabilities, objective_trial, status)
                if (status%code /= FORTNUM_OK) return
            end do
            step_norm = abs(alpha_trial-alpha)/max(1.0_dp, abs(alpha))
            alpha = alpha_trial
            objective = objective_trial
            state%iterations = iteration
            state%final_step_norm = step_norm
            if (step_norm <= options%tolerance .or. abs(gradient) <= options%tolerance) then
                state%converged = .true.
                exit
            end if
        end do
        if (.not. state%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multiclass probability calibration fit: iteration limit reached")
            return
        end if
        self%temperature = 1.0_dp/alpha
        state%objective = objective
        state%method = CALIBRATION_TEMPERATURE
        state%knot_count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine fit_multiclass_temperature

    subroutine multiclass_temperature_objective(scores, encoded, weights, total_weight, &
            alpha, l2, probabilities, value, status)
        real(dp), intent(in) :: scores(:, :), weights(:), total_weight, alpha, l2
        integer, intent(in) :: encoded(:)
        real(dp), intent(out) :: probabilities(:, :), value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: max_score, normalizer
        integer :: i, class_index

        value = 0.0_dp
        do i = 1, size(scores, 1)
            max_score = maxval(alpha*scores(i, :))
            normalizer = sum(exp(alpha*scores(i, :) - max_score))
            if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multiclass probability calibration fit: softmax normalizer is invalid")
                return
            end if
            probabilities(i, :) = exp(alpha*scores(i, :) - max_score)/normalizer
            class_index = encoded(i)
            value = value + weights(i)*(log(normalizer) + max_score - &
                alpha*scores(i, class_index))
        end do
        value = value/total_weight + 0.5_dp*l2*alpha*alpha
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration fit: objective is nonfinite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_temperature_objective

    subroutine multiclass_temperature_derivatives(scores, encoded, weights, total_weight, &
            alpha, l2, probabilities, gradient, hessian)
        real(dp), intent(in) :: scores(:, :), weights(:), total_weight, alpha, l2
        integer, intent(in) :: encoded(:)
        real(dp), intent(in) :: probabilities(:, :)
        real(dp), intent(out) :: gradient, hessian
        real(dp) :: mean_score, mean_square
        integer :: i, class_index

        gradient = l2*alpha
        hessian = l2
        do i = 1, size(scores, 1)
            class_index = encoded(i)
            mean_score = sum(probabilities(i, :)*scores(i, :))
            mean_square = sum(probabilities(i, :)*scores(i, :)*scores(i, :))
            gradient = gradient + weights(i)*(mean_score - scores(i, class_index))/total_weight
            hessian = hessian + weights(i)*(mean_square - mean_score*mean_score)/total_weight
        end do
    end subroutine multiclass_temperature_derivatives

    subroutine multiclass_probability_calibration_predict_proba(self, scores, &
            probabilities, status)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. multiclass_prediction_valid(self, scores, probabilities, status)) return
        call multiclass_softmax(self, scores, probabilities, status)
    end subroutine multiclass_probability_calibration_predict_proba

    subroutine multiclass_softmax(self, scores, probabilities, status)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: alpha, max_score, normalizer
        integer :: i

        alpha = 1.0_dp/self%temperature
        do i = 1, size(scores, 1)
            max_score = maxval(alpha*scores(i, :))
            normalizer = sum(exp(alpha*scores(i, :) - max_score))
            if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multiclass probability calibration prediction: softmax normalizer is invalid")
                return
            end if
            probabilities(i, :) = exp(alpha*scores(i, :) - max_score)/normalizer
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_softmax

    subroutine multiclass_probability_calibration_predict_proba_device(self, device, &
            scores, probabilities, status)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: scores(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(scores, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multiclass probability calibration device: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration device: device kind is invalid")
        end select
    end subroutine multiclass_probability_calibration_predict_proba_device

    subroutine multiclass_probability_calibration_predict_proba_jvp(self, scores, &
            scores_dot, probabilities, probabilities_dot, status)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :), scores_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: alpha, tangent, dot_product
        integer :: i, j

        if (.not. multiclass_prediction_valid(self, scores, probabilities, status)) return
        if (any(shape(scores_dot) /= shape(scores)) .or. &
            any(.not. ieee_is_finite(scores_dot)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration JVP: tangent or output shape is invalid")
            return
        end if
        call multiclass_softmax(self, scores, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        alpha = 1.0_dp/self%temperature
        do i = 1, size(scores, 1)
            tangent = alpha*scores_dot(i, 1)
            dot_product = probabilities(i, 1)*tangent
            probabilities_dot(i, 1) = probabilities(i, 1)*tangent
            do j = 2, size(scores, 2)
                tangent = alpha*scores_dot(i, j)
                dot_product = dot_product + probabilities(i, j)*tangent
                probabilities_dot(i, j) = probabilities(i, j)*tangent
            end do
            probabilities_dot(i, :) = probabilities_dot(i, :) - &
                probabilities(i, :)*dot_product
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_probability_calibration_predict_proba_jvp

    subroutine multiclass_probability_calibration_predict_proba_vjp(self, scores, &
            probabilities_bar, scores_bar, status)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: scores_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        real(dp) :: dot_product, eta_bar, alpha
        integer :: i, j

        if (.not. multiclass_vjp_valid(self, scores, probabilities_bar, scores_bar, status)) return
        allocate(probabilities(size(scores, 1), size(scores, 2)))
        call multiclass_softmax(self, scores, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        alpha = 1.0_dp/self%temperature
        scores_bar = 0.0_dp
        do i = 1, size(scores, 1)
            dot_product = sum(probabilities(i, :)*probabilities_bar(i, :))
            do j = 1, size(scores, 2)
                eta_bar = probabilities(i, j)*(probabilities_bar(i, j)-dot_product)
                scores_bar(i, j) = alpha*eta_bar
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_probability_calibration_predict_proba_vjp

    subroutine multiclass_probability_calibration_predict_proba_parameter_jvp(self, &
            scores, parameters_dot, probabilities, probabilities_dot, status)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :), parameters_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: alpha, alpha_dot, tangent, dot_product
        integer :: i

        if (.not. multiclass_prediction_valid(self, scores, probabilities, status)) return
        if (size(parameters_dot) /= 1 .or. any(.not. ieee_is_finite(parameters_dot)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration parameter JVP: tangent or output shape is invalid")
            return
        end if
        call multiclass_softmax(self, scores, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        alpha = 1.0_dp/self%temperature
        alpha_dot = -parameters_dot(1)/self%temperature**2
        do i = 1, size(scores, 1)
            probabilities_dot(i, :) = (alpha_dot*scores(i, :))*probabilities(i, :)
            dot_product = sum(probabilities_dot(i, :))
            probabilities_dot(i, :) = probabilities_dot(i, :) - &
                probabilities(i, :)*dot_product
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_probability_calibration_predict_proba_parameter_jvp

    subroutine multiclass_probability_calibration_predict_proba_parameter_vjp(self, scores, &
            probabilities_bar, parameters_bar, status)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameters_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        real(dp) :: dot_product, eta_bar, alpha_bar
        integer :: i, j

        if (size(parameters_bar) > 0) parameters_bar = 0.0_dp
        if (.not. multiclass_parameter_vjp_valid(self, scores, probabilities_bar, &
            parameters_bar, status)) return
        allocate(probabilities(size(scores, 1), size(scores, 2)))
        call multiclass_softmax(self, scores, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        alpha_bar = 0.0_dp
        do i = 1, size(scores, 1)
            dot_product = sum(probabilities(i, :)*probabilities_bar(i, :))
            do j = 1, size(scores, 2)
                eta_bar = probabilities(i, j)*(probabilities_bar(i, j)-dot_product)
                alpha_bar = alpha_bar + scores(i, j)*eta_bar
            end do
        end do
        parameters_bar(1) = -alpha_bar/self%temperature**2
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_probability_calibration_predict_proba_parameter_vjp

    subroutine multiclass_probability_calibration_predict(self, scores, labels, status)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, j, best_class

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration predict: model is not fitted")
            return
        end if
        if (.not. allocated(self%class_label)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration predict: class metadata is absent")
            return
        end if
        if (size(labels) /= size(scores, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(scores, 1), size(self%class_label)))
        call self%predict_proba(scores, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(scores, 1)
            best_class = 1
            do j = 2, size(self%class_label)
                if (probabilities(i, j) > probabilities(i, best_class)) best_class = j
            end do
            labels(i) = self%class_label(best_class)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_probability_calibration_predict

    subroutine multiclass_probability_calibration_set_parameters(self, parameters, status)
        class(multiclass_probability_calibrator_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration set_parameters: model is not fitted")
            return
        end if
        if (size(parameters) /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration set_parameters: temperature must be positive")
            return
        end if
        if (.not. ieee_is_finite(parameters(1)) .or. parameters(1) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration set_parameters: temperature must be positive")
            return
        end if
        self%temperature = parameters(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine multiclass_probability_calibration_set_parameters

    function multiclass_probability_calibration_parameters(self) result(parameters)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (self%is_fitted) then
            allocate(parameters(1))
            parameters = [self%temperature]
        else
            allocate(parameters(0))
        end if
    end function multiclass_probability_calibration_parameters

    integer function multiclass_probability_calibration_parameter_count(self) result(count)
        class(multiclass_probability_calibrator_t), intent(in) :: self

        if (self%is_fitted) then
            count = 1
        else
            count = 0
        end if
    end function multiclass_probability_calibration_parameter_count

    function multiclass_probability_calibration_classes(self) result(classes)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function multiclass_probability_calibration_classes

    integer function multiclass_probability_calibration_method(self) result(method)
        class(multiclass_probability_calibrator_t), intent(in) :: self

        method = CALIBRATION_TEMPERATURE
    end function multiclass_probability_calibration_method

    logical function multiclass_probability_calibration_fitted(self) result(fitted)
        class(multiclass_probability_calibrator_t), intent(in) :: self

        fitted = self%is_fitted
    end function multiclass_probability_calibration_fitted

    logical function multiclass_probability_calibration_device_supported(self, device_kind) &
            result(supported)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function multiclass_probability_calibration_device_supported

    logical function multiclass_prediction_valid(self, scores, probabilities, status) &
            result(valid)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration prediction: model is not fitted")
            return
        end if
        if (.not. allocated(self%class_label)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration prediction: class metadata is absent")
            return
        end if
        if (size(scores, 1) < 1 .or. size(scores, 2) /= size(self%class_label) .or. &
            any(shape(probabilities) /= [size(scores, 1), size(self%class_label)]) .or. &
            any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration prediction: shape or logits are invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function multiclass_prediction_valid

    logical function multiclass_vjp_valid(self, scores, probabilities_bar, scores_bar, status) &
            result(valid)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: scores_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. multiclass_prediction_valid(self, scores, probabilities_bar, status)) return
        if (any(shape(scores_bar) /= shape(scores)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration VJP: shape or cotangent is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function multiclass_vjp_valid

    logical function multiclass_parameter_vjp_valid(self, scores, probabilities_bar, &
            parameters_bar, status) result(valid)
        class(multiclass_probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameters_bar(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. multiclass_prediction_valid(self, scores, probabilities_bar, status)) return
        if (size(parameters_bar) /= 1 .or. any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multiclass probability calibration parameter VJP: shape or cotangent is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function multiclass_parameter_vjp_valid

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: sorted(:)
        integer :: i, j, key, n_unique

        allocate(sorted(size(labels)))
        sorted = labels
        do i = 2, size(sorted)
            key = sorted(i)
            j = i - 1
            do while (j >= 1)
                if (sorted(j) <= key) exit
                sorted(j + 1) = sorted(j)
                j = j - 1
            end do
            sorted(j + 1) = key
        end do
        n_unique = 1
        do i = 2, size(sorted)
            if (sorted(i) /= sorted(i - 1)) n_unique = n_unique + 1
        end do
        allocate(classes(n_unique))
        classes(1) = sorted(1)
        n_unique = 1
        do i = 2, size(sorted)
            if (sorted(i) /= sorted(i - 1)) then
                n_unique = n_unique + 1
                classes(n_unique) = sorted(i)
            end if
        end do
    end subroutine sorted_unique_labels

    subroutine probability_calibration_fit(self, scores, labels, status, options, &
            sample_weight, state)
        class(probability_calibrator_t), intent(out) :: self
        real(dp), intent(in) :: scores(:)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(probability_calibration_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        type(probability_calibration_state_t), intent(out), optional :: state
        type(probability_calibration_options_t) :: requested
        type(probability_calibration_state_t) :: result
        real(dp), allocatable :: weights(:)
        integer :: label_min, label_max

        self%is_fitted = .false.
        self%n_knots = 0
        if (allocated(self%knots)) deallocate(self%knots)
        if (allocated(self%knot_values)) deallocate(self%knot_values)
        result = probability_calibration_state_t()
        if (present(state)) state = result
        requested = probability_calibration_options_t()
        if (present(options)) requested = options
        if (.not. valid_options(requested)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration fit: options are invalid")
            return
        end if
        if (size(scores) < 1 .or. size(labels) /= size(scores) .or. &
            any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration fit: scores and labels have invalid shape")
            return
        end if
        label_min = minval(labels)
        label_max = maxval(labels)
        if (label_min == label_max .or. any((labels /= label_min) .and. &
            (labels /= label_max))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration fit: two target classes are required")
            return
        end if
        allocate(weights(size(scores)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(scores) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "probability calibration fit: weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        if (.not. ieee_is_finite(sum(weights)) .or. sum(weights) <= 0.0_dp .or. &
            sum(weights, mask=labels == label_min) <= 0.0_dp .or. &
            sum(weights, mask=labels == label_max) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration fit: each class needs positive weight")
            return
        end if

        self%class_label = [label_min, label_max]
        self%calibration_method = requested%method
        select case (requested%method)
        case (CALIBRATION_SIGMOID)
            call fit_sigmoid(self, scores, labels, weights, requested, result, status)
        case (CALIBRATION_ISOTONIC)
            call fit_isotonic(self, scores, labels, weights, result, status)
        case (CALIBRATION_TEMPERATURE)
            call fit_temperature(self, scores, labels, weights, requested, result, status)
        end select
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if
        result%method = self%calibration_method
        if (present(state)) state = result
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine probability_calibration_fit

    subroutine fit_sigmoid(self, scores, labels, weights, options, state, status)
        class(probability_calibrator_t), intent(inout) :: self
        real(dp), intent(in) :: scores(:), weights(:)
        integer, intent(in) :: labels(:)
        type(probability_calibration_options_t), intent(in) :: options
        type(probability_calibration_state_t), intent(inout) :: state
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: positive_mass, total_mass, mean_label, slope, intercept
        real(dp) :: objective_old, objective_new, gradient_slope, gradient_intercept
        real(dp) :: hessian_ss, hessian_si, hessian_ii, determinant
        real(dp) :: step_slope, step_intercept, step_scale, step_norm
        real(dp) :: trial_slope, trial_intercept
        integer :: iteration, line_search

        total_mass = sum(weights)
        positive_mass = sum(weights, mask=labels == self%class_label(2))
        mean_label = min(max(positive_mass/total_mass, 1.0e-8_dp), 1.0_dp - 1.0e-8_dp)
        slope = 0.0_dp
        intercept = log(mean_label/(1.0_dp - mean_label))
        objective_old = sigmoid_objective(scores, labels, weights, &
            self%class_label, slope, intercept, options%l2)
        state%iterations = 0
        state%final_step_norm = huge(1.0_dp)
        state%converged = .false.
        do iteration = 1, options%max_iterations
            call sigmoid_derivatives(scores, labels, weights, self%class_label, &
                slope, intercept, options%l2, gradient_slope, gradient_intercept, &
                hessian_ss, hessian_si, hessian_ii)
            determinant = hessian_ss*hessian_ii - hessian_si*hessian_si
            if (.not. ieee_is_finite(determinant) .or. determinant <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "probability calibration sigmoid fit: singular Newton system")
                return
            end if
            step_slope = (hessian_ii*gradient_slope - hessian_si*gradient_intercept)/determinant
            step_intercept = (-hessian_si*gradient_slope + hessian_ss*gradient_intercept)/determinant
            step_scale = 1.0_dp
            trial_slope = slope - step_scale*options%damping*step_slope
            trial_intercept = intercept - step_scale*options%damping*step_intercept
            objective_new = sigmoid_objective(scores, labels, weights, self%class_label, &
                trial_slope, trial_intercept, options%l2)
            do line_search = 1, 30
                if (objective_new <= objective_old .or. step_scale <= 1.0e-8_dp) exit
                step_scale = 0.5_dp*step_scale
                trial_slope = slope - step_scale*options%damping*step_slope
                trial_intercept = intercept - step_scale*options%damping*step_intercept
                objective_new = sigmoid_objective(scores, labels, weights, self%class_label, &
                    trial_slope, trial_intercept, options%l2)
            end do
            step_norm = max(abs(trial_slope - slope), abs(trial_intercept - intercept))/ &
                max(1.0_dp, max(abs(slope), abs(intercept)))
            slope = trial_slope
            intercept = trial_intercept
            objective_old = objective_new
            state%iterations = iteration
            state%final_step_norm = step_norm
            if (step_norm <= options%tolerance .or. &
                max(abs(gradient_slope), abs(gradient_intercept)) <= options%tolerance) then
                state%converged = .true.
                exit
            end if
        end do
        if (.not. state%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "probability calibration sigmoid fit: iteration limit reached")
            return
        end if
        self%sigmoid_slope = slope
        self%sigmoid_intercept = intercept
        state%objective = objective_old
        state%knot_count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine fit_sigmoid

    subroutine fit_temperature(self, scores, labels, weights, options, state, status)
        !! Fit a positive scalar temperature for already-oriented binary logits.
        !!
        !! The optimized coordinate is ``alpha = 1 / temperature``.  The
        !! weighted logistic objective is convex in alpha, so a damped Newton
        !! step with a positive-domain line search is sufficient and keeps the
        !! public parameterization physically meaningful.
        class(probability_calibrator_t), intent(inout) :: self
        real(dp), intent(in) :: scores(:), weights(:)
        integer, intent(in) :: labels(:)
        type(probability_calibration_options_t), intent(in) :: options
        type(probability_calibration_state_t), intent(inout) :: state
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: alpha, alpha_trial, objective_old, objective_new
        real(dp) :: gradient, hessian, step, step_scale, step_norm
        real(dp) :: alpha_floor
        integer :: iteration, line_search

        alpha_floor = sqrt(tiny(1.0_dp))
        alpha = 1.0_dp
        objective_old = temperature_objective(scores, labels, weights, self%class_label, &
            alpha, options%l2)
        state%iterations = 0
        state%final_step_norm = huge(1.0_dp)
        state%converged = .false.
        do iteration = 1, options%max_iterations
            call temperature_derivatives(scores, labels, weights, self%class_label, alpha, &
                options%l2, gradient, hessian)
            if (.not. ieee_is_finite(gradient) .or. .not. ieee_is_finite(hessian) .or. &
                hessian <= 0.0_dp) then
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "probability calibration temperature fit: invalid Newton curvature")
                return
            end if
            step = gradient/hessian
            step_scale = 1.0_dp
            alpha_trial = max(alpha_floor, alpha - step_scale*options%damping*step)
            objective_new = temperature_objective(scores, labels, weights, self%class_label, &
                alpha_trial, options%l2)
            do line_search = 1, 30
                if (objective_new <= objective_old .or. step_scale <= 1.0e-8_dp) exit
                step_scale = 0.5_dp*step_scale
                alpha_trial = max(alpha_floor, alpha - step_scale*options%damping*step)
                objective_new = temperature_objective(scores, labels, weights, self%class_label, &
                    alpha_trial, options%l2)
            end do
            step_norm = abs(alpha_trial - alpha)/max(1.0_dp, abs(alpha))
            alpha = alpha_trial
            objective_old = objective_new
            state%iterations = iteration
            state%final_step_norm = step_norm
            if (step_norm <= options%tolerance .or. abs(gradient) <= options%tolerance) then
                state%converged = .true.
                exit
            end if
        end do
        if (.not. state%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "probability calibration temperature fit: iteration limit reached")
            return
        end if
        self%temperature = 1.0_dp/alpha
        state%objective = objective_old
        state%knot_count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine fit_temperature

    subroutine fit_isotonic(self, scores, labels, weights, state, status)
        class(probability_calibrator_t), intent(inout) :: self
        real(dp), intent(in) :: scores(:), weights(:)
        integer, intent(in) :: labels(:)
        type(probability_calibration_state_t), intent(inout) :: state
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: sorted_scores(:), unique_scores(:), unique_weights(:)
        real(dp), allocatable :: unique_positive(:), block_x(:), block_w(:), block_y(:)
        integer, allocatable :: order(:)
        integer :: i, j, m, n_blocks, key, n
        real(dp) :: label_value, previous_mean, current_mean, objective

        n = size(scores)
        allocate(order(n), sorted_scores(n))
        order = [(i, i=1, n)]
        do i = 2, n
            key = order(i)
            j = i - 1
            do while (j >= 1)
                if (scores(order(j)) <= scores(key)) exit
                order(j + 1) = order(j)
                j = j - 1
            end do
            order(j + 1) = key
        end do
        sorted_scores = scores(order)
        allocate(unique_scores(n), unique_weights(n), unique_positive(n))
        m = 0
        do i = 1, n
            if (weights(order(i)) <= 0.0_dp) cycle
            if (m == 0 .or. sorted_scores(i) /= unique_scores(m)) then
                m = m + 1
                unique_scores(m) = sorted_scores(i)
                unique_weights(m) = weights(order(i))
                unique_positive(m) = weights(order(i))* &
                    merge(1.0_dp, 0.0_dp, labels(order(i)) == self%class_label(2))
            else
                unique_weights(m) = unique_weights(m) + weights(order(i))
                unique_positive(m) = unique_positive(m) + weights(order(i))* &
                    merge(1.0_dp, 0.0_dp, labels(order(i)) == self%class_label(2))
            end if
        end do
        if (m < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration isotonic fit: no positive-weight scores")
            return
        end if
        allocate(block_x(m), block_w(m), block_y(m))
        n_blocks = 0
        do i = 1, m
            n_blocks = n_blocks + 1
            block_x(n_blocks) = unique_scores(i)
            block_w(n_blocks) = unique_weights(i)
            block_y(n_blocks) = unique_positive(i)
            do while (n_blocks >= 2)
                previous_mean = block_y(n_blocks - 1)/block_w(n_blocks - 1)
                current_mean = block_y(n_blocks)/block_w(n_blocks)
                if (previous_mean <= current_mean) exit
                block_x(n_blocks - 1) = (block_w(n_blocks - 1)*block_x(n_blocks - 1) + &
                    block_w(n_blocks)*block_x(n_blocks))/ &
                    (block_w(n_blocks - 1) + block_w(n_blocks))
                block_y(n_blocks - 1) = block_y(n_blocks - 1) + block_y(n_blocks)
                block_w(n_blocks - 1) = block_w(n_blocks - 1) + block_w(n_blocks)
                n_blocks = n_blocks - 1
            end do
        end do
        allocate(self%knots(n_blocks), self%knot_values(n_blocks))
        self%knots = block_x(:n_blocks)
        self%knot_values = block_y(:n_blocks)/block_w(:n_blocks)
        self%n_knots = n_blocks
        objective = 0.0_dp
        do i = 1, n
            if (weights(i) <= 0.0_dp) cycle
            call isotonic_probability(self, scores(i), label_value, status)
            if (status%code /= FORTNUM_OK) return
            label_value = merge(1.0_dp, 0.0_dp, labels(i) == self%class_label(2))
            objective = objective + weights(i)*(label_value - &
                isotonic_value(self, scores(i)))**2
        end do
        state%iterations = 1
        state%final_step_norm = 0.0_dp
        state%converged = .true.
        state%objective = objective/sum(weights)
        state%knot_count = n_blocks
        call status_set(status, FORTNUM_OK, "")
    end subroutine fit_isotonic

    subroutine probability_calibration_predict_proba(self, scores, probabilities, status)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i
        real(dp) :: positive

        if (.not. prediction_valid(self, scores, probabilities, status)) return
        do i = 1, size(scores)
            call calibration_probability(self, scores(i), positive, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(i, 1) = 1.0_dp - positive
            probabilities(i, 2) = positive
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine probability_calibration_predict_proba

    subroutine probability_calibration_predict_proba_device(self, device, scores, &
            probabilities, status)
        class(probability_calibrator_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: scores(:)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(scores, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "probability calibration device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration device prediction: device kind is invalid")
        end select
    end subroutine probability_calibration_predict_proba_device

    subroutine probability_calibration_predict_proba_jvp(self, scores, scores_dot, &
            probabilities, probabilities_dot, status)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:), scores_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i
        real(dp) :: positive, positive_dot

        if (.not. prediction_valid(self, scores, probabilities, status)) return
        if (size(scores_dot) /= size(scores) .or. any(.not. ieee_is_finite(scores_dot)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration JVP: tangent or output shape is invalid")
            return
        end if
        do i = 1, size(scores)
            call calibration_probability_derivative(self, scores(i), positive, &
                positive_dot, status)
            if (status%code /= FORTNUM_OK) return
            positive_dot = positive_dot*scores_dot(i)
            probabilities(i, 1) = 1.0_dp - positive
            probabilities(i, 2) = positive
            probabilities_dot(i, 1) = -positive_dot
            probabilities_dot(i, 2) = positive_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine probability_calibration_predict_proba_jvp

    subroutine probability_calibration_predict_proba_vjp(self, scores, probabilities_bar, &
            scores_bar, status)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:), probabilities_bar(:, :)
        real(dp), intent(out) :: scores_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: positive, positive_dot
        integer :: i

        scores_bar = 0.0_dp
        if (.not. prediction_valid_bar(self, scores, probabilities_bar, scores_bar, status)) return
        do i = 1, size(scores)
            call calibration_probability_derivative(self, scores(i), positive, &
                positive_dot, status)
            if (status%code /= FORTNUM_OK) return
            scores_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*positive_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine probability_calibration_predict_proba_vjp

    subroutine probability_calibration_predict_proba_parameter_jvp(self, scores, &
            parameters_dot, probabilities, probabilities_dot, status)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:), parameters_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i
        real(dp) :: positive, positive_dot, eta

        if (.not. prediction_valid(self, scores, probabilities, status)) return
        if ((self%calibration_method /= CALIBRATION_SIGMOID .and. &
            self%calibration_method /= CALIBRATION_TEMPERATURE) .or. &
            size(parameters_dot) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(parameters_dot)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "probability calibration parameter JVP: method has no smooth parameters")
            return
        end if
        if (self%calibration_method == CALIBRATION_SIGMOID) then
            do i = 1, size(scores)
                eta = self%sigmoid_slope*scores(i) + self%sigmoid_intercept
                positive = sigmoid(eta)
                positive_dot = positive*(1.0_dp - positive)* &
                    (scores(i)*parameters_dot(1) + parameters_dot(2))
                probabilities(i, :) = [1.0_dp - positive, positive]
                probabilities_dot(i, :) = [-positive_dot, positive_dot]
            end do
        else
            do i = 1, size(scores)
                eta = scores(i)/self%temperature
                positive = sigmoid(eta)
                positive_dot = positive*(1.0_dp - positive)* &
                    (-scores(i)/self%temperature**2)*parameters_dot(1)
                probabilities(i, :) = [1.0_dp - positive, positive]
                probabilities_dot(i, :) = [-positive_dot, positive_dot]
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine probability_calibration_predict_proba_parameter_jvp

    subroutine probability_calibration_predict_proba_parameter_vjp(self, scores, &
            probabilities_bar, parameters_bar, status)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:), probabilities_bar(:, :)
        real(dp), intent(out) :: parameters_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: positive, positive_dot, factor
        integer :: i

        parameters_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(probabilities_bar, 1) /= size(scores) .or. &
            size(probabilities_bar, 2) /= 2 .or. &
            size(parameters_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(scores)) .or. any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration parameter VJP: input or output shape is invalid")
            return
        end if
        if (self%calibration_method /= CALIBRATION_SIGMOID .and. &
            self%calibration_method /= CALIBRATION_TEMPERATURE) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "probability calibration parameter VJP: method parameters are discrete")
            return
        end if
        if (size(parameters_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration parameter VJP: output shape is invalid")
            return
        end if
        if (self%calibration_method == CALIBRATION_SIGMOID) then
            do i = 1, size(scores)
                positive = sigmoid(self%sigmoid_slope*scores(i) + self%sigmoid_intercept)
                positive_dot = positive*(1.0_dp - positive)
                factor = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*positive_dot
                parameters_bar(1) = parameters_bar(1) + factor*scores(i)
                parameters_bar(2) = parameters_bar(2) + factor
            end do
        else
            do i = 1, size(scores)
                positive = sigmoid(scores(i)/self%temperature)
                positive_dot = positive*(1.0_dp - positive)
                factor = (probabilities_bar(i, 2) - probabilities_bar(i, 1))*positive_dot
                parameters_bar(1) = parameters_bar(1) - factor*scores(i)/self%temperature**2
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine probability_calibration_predict_proba_parameter_vjp

    subroutine probability_calibration_predict(self, scores, labels, status)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. self%is_fitted .or. size(labels) /= size(scores)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration predict: model or output shape is invalid")
            return
        end if
        allocate(probabilities(size(scores), 2))
        call self%predict_proba(scores, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(scores)
            if (probabilities(i, 2) > probabilities(i, 1)) then
                labels(i) = self%class_label(2)
            else
                labels(i) = self%class_label(1)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine probability_calibration_predict

    subroutine probability_calibration_set_parameters(self, parameters, status)
        class(probability_calibrator_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted .or. &
            ((self%calibration_method /= CALIBRATION_SIGMOID .and. &
            self%calibration_method /= CALIBRATION_TEMPERATURE)) .or. &
            size(parameters) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration set_parameters: fitted smooth state is invalid")
            return
        end if
        if (self%calibration_method == CALIBRATION_SIGMOID) then
            self%sigmoid_slope = parameters(1)
            self%sigmoid_intercept = parameters(2)
        else
            if (parameters(1) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "probability calibration set_parameters: temperature must be positive")
                return
            end if
            self%temperature = parameters(1)
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine probability_calibration_set_parameters

    function probability_calibration_parameters(self) result(parameters)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (.not. self%is_fitted .or. (self%calibration_method /= CALIBRATION_SIGMOID .and. &
            self%calibration_method /= CALIBRATION_TEMPERATURE)) then
            allocate(parameters(0))
        else if (self%calibration_method == CALIBRATION_TEMPERATURE) then
            allocate(parameters(1))
            parameters = [self%temperature]
        else
            allocate(parameters(2))
            parameters = [self%sigmoid_slope, self%sigmoid_intercept]
        end if
    end function probability_calibration_parameters

    integer function probability_calibration_parameter_count(self) result(count)
        class(probability_calibrator_t), intent(in) :: self

        if (self%is_fitted .and. self%calibration_method == CALIBRATION_TEMPERATURE) then
            count = 1
        else if (self%is_fitted .and. self%calibration_method == CALIBRATION_SIGMOID) then
            count = 2
        else
            count = 0
        end if
    end function probability_calibration_parameter_count

    function probability_calibration_classes(self) result(classes)
        class(probability_calibrator_t), intent(in) :: self
        integer :: classes(2)

        classes = self%class_label
    end function probability_calibration_classes

    integer function probability_calibration_method(self) result(method)
        class(probability_calibrator_t), intent(in) :: self

        method = self%calibration_method
    end function probability_calibration_method

    logical function probability_calibration_fitted(self) result(fitted)
        class(probability_calibrator_t), intent(in) :: self

        fitted = self%is_fitted
    end function probability_calibration_fitted

    logical function probability_calibration_device_supported(self, device_kind) result(supported)
        class(probability_calibrator_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function probability_calibration_device_supported

    logical function prediction_valid(self, scores, probabilities, status) result(valid)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%is_fitted .or. size(scores) < 1 .or. &
            any(shape(probabilities) /= [size(scores), 2]) .or. &
            any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration prediction: model, shape, or scores are invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_valid

    logical function prediction_valid_bar(self, scores, probabilities_bar, scores_bar, status) &
            result(valid)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: scores(:), probabilities_bar(:, :), scores_bar(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%is_fitted .or. size(scores) < 1 .or. &
            size(probabilities_bar, 1) /= size(scores) .or. &
            size(probabilities_bar, 2) /= 2 .or. size(scores_bar) /= size(scores) .or. &
            any(.not. ieee_is_finite(scores)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration VJP: model, shape, or inputs are invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_valid_bar

    subroutine calibration_probability(self, score, positive, status)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: score
        real(dp), intent(out) :: positive
        type(fortnum_status_t), intent(out) :: status

        if (.not. ieee_is_finite(score)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration prediction: score is not finite")
            return
        end if
        select case (self%calibration_method)
        case (CALIBRATION_SIGMOID)
            positive = sigmoid(self%sigmoid_slope*score + self%sigmoid_intercept)
        case (CALIBRATION_TEMPERATURE)
            positive = sigmoid(score/self%temperature)
        case (CALIBRATION_ISOTONIC)
            positive = isotonic_value(self, score)
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration prediction: method is invalid")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine calibration_probability

    subroutine isotonic_probability(self, score, positive, status)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: score
        real(dp), intent(out) :: positive
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted .and. self%n_knots < 1) then
            positive = 0.0_dp
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        positive = isotonic_value(self, score)
        call status_set(status, FORTNUM_OK, "")
    end subroutine isotonic_probability

    real(dp) function isotonic_value(self, score) result(value)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: score
        integer :: i
        real(dp) :: fraction

        if (self%n_knots <= 0) then
            value = 0.5_dp
        else if (self%n_knots == 1 .or. score <= self%knots(1)) then
            value = self%knot_values(1)
        else if (score >= self%knots(self%n_knots)) then
            value = self%knot_values(self%n_knots)
        else
            i = 1
            do while (i < self%n_knots - 1 .and. score > self%knots(i + 1))
                i = i + 1
            end do
            fraction = (score - self%knots(i))/(self%knots(i + 1) - self%knots(i))
            value = self%knot_values(i) + fraction* &
                (self%knot_values(i + 1) - self%knot_values(i))
        end if
        value = min(max(value, 0.0_dp), 1.0_dp)
    end function isotonic_value

    subroutine calibration_probability_derivative(self, score, positive, derivative, status)
        class(probability_calibrator_t), intent(in) :: self
        real(dp), intent(in) :: score
        real(dp), intent(out) :: positive, derivative
        type(fortnum_status_t), intent(out) :: status
        integer :: i
        real(dp) :: left_slope, right_slope, tol

        call calibration_probability(self, score, positive, status)
        if (status%code /= FORTNUM_OK) return
        select case (self%calibration_method)
        case (CALIBRATION_SIGMOID)
            derivative = positive*(1.0_dp - positive)*self%sigmoid_slope
        case (CALIBRATION_TEMPERATURE)
            derivative = positive*(1.0_dp - positive)/self%temperature
        case (CALIBRATION_ISOTONIC)
            derivative = 0.0_dp
            if (self%n_knots <= 1) then
                call status_set(status, FORTNUM_OK, "")
                return
            end if
            tol = 64.0_dp*epsilon(1.0_dp)*max(1.0_dp, abs(score))
            if (abs(score - self%knots(1)) <= tol) then
                if (self%knot_values(2) /= self%knot_values(1)) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "probability calibration isotonic JVP: knot derivative is undefined")
                    return
                end if
            else if (abs(score - self%knots(self%n_knots)) <= tol) then
                if (self%knot_values(self%n_knots) /= &
                    self%knot_values(self%n_knots - 1)) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "probability calibration isotonic JVP: knot derivative is undefined")
                    return
                end if
            else
                i = 1
                do while (i < self%n_knots - 1 .and. score > self%knots(i + 1))
                    i = i + 1
                end do
                if (abs(score - self%knots(i)) <= tol) then
                    left_slope = (self%knot_values(i) - self%knot_values(i - 1))/ &
                        (self%knots(i) - self%knots(i - 1))
                    right_slope = (self%knot_values(i + 1) - self%knot_values(i))/ &
                        (self%knots(i + 1) - self%knots(i))
                    if (left_slope /= right_slope) then
                        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                            "probability calibration isotonic JVP: knot derivative is undefined")
                        return
                    end if
                else if (abs(score - self%knots(i + 1)) <= tol) then
                    left_slope = (self%knot_values(i + 1) - self%knot_values(i))/ &
                        (self%knots(i + 1) - self%knots(i))
                    right_slope = (self%knot_values(i + 2) - self%knot_values(i + 1))/ &
                        (self%knots(i + 2) - self%knots(i + 1))
                    if (left_slope /= right_slope) then
                        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                            "probability calibration isotonic JVP: knot derivative is undefined")
                        return
                    end if
                end if
                derivative = (self%knot_values(i + 1) - self%knot_values(i))/ &
                    (self%knots(i + 1) - self%knots(i))
            end if
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "probability calibration derivative: method is invalid")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine calibration_probability_derivative

    logical function valid_options(options) result(valid)
        type(probability_calibration_options_t), intent(in) :: options

        valid = (options%method == CALIBRATION_SIGMOID .or. &
            options%method == CALIBRATION_ISOTONIC .or. &
            options%method == CALIBRATION_TEMPERATURE) .and. &
            options%max_iterations >= 1 .and. ieee_is_finite(options%tolerance) .and. &
            options%tolerance > 0.0_dp .and. ieee_is_finite(options%damping) .and. &
            options%damping > 0.0_dp .and. options%damping <= 1.0_dp .and. &
            ieee_is_finite(options%l2) .and. options%l2 >= 0.0_dp
    end function valid_options

    real(dp) function sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        real(dp) :: exponential

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            exponential = exp(value)
            probability = exponential/(1.0_dp + exponential)
        end if
    end function sigmoid

    real(dp) function sigmoid_objective(scores, labels, weights, classes, slope, intercept, l2) &
            result(objective)
        real(dp), intent(in) :: scores(:), weights(:), slope, intercept, l2
        integer, intent(in) :: labels(:), classes(2)
        real(dp) :: eta, probability, target
        integer :: i

        objective = 0.0_dp
        do i = 1, size(scores)
            eta = slope*scores(i) + intercept
            target = merge(1.0_dp, 0.0_dp, labels(i) == classes(2))
            if (eta >= 0.0_dp) then
                objective = objective + weights(i)*(log(1.0_dp + exp(-eta)) + &
                    (1.0_dp - target)*eta)
            else
                objective = objective + weights(i)*(log(1.0_dp + exp(eta)) - target*eta)
            end if
        end do
        objective = objective + 0.5_dp*l2*(slope*slope + intercept*intercept)
    end function sigmoid_objective

    real(dp) function temperature_objective(scores, labels, weights, classes, alpha, l2) &
            result(objective)
        real(dp), intent(in) :: scores(:), weights(:), alpha, l2
        integer, intent(in) :: labels(:), classes(2)
        real(dp) :: eta, target
        integer :: i

        objective = 0.0_dp
        do i = 1, size(scores)
            eta = alpha*scores(i)
            target = merge(1.0_dp, 0.0_dp, labels(i) == classes(2))
            if (eta >= 0.0_dp) then
                objective = objective + weights(i)*(log(1.0_dp + exp(-eta)) + &
                    (1.0_dp - target)*eta)
            else
                objective = objective + weights(i)*(log(1.0_dp + exp(eta)) - target*eta)
            end if
        end do
        objective = objective + 0.5_dp*l2*alpha*alpha
    end function temperature_objective

    subroutine temperature_derivatives(scores, labels, weights, classes, alpha, l2, &
            gradient, hessian)
        real(dp), intent(in) :: scores(:), weights(:), alpha, l2
        integer, intent(in) :: labels(:), classes(2)
        real(dp), intent(out) :: gradient, hessian
        real(dp) :: eta, probability, curvature, target, residual
        integer :: i

        gradient = l2*alpha
        hessian = l2
        do i = 1, size(scores)
            eta = alpha*scores(i)
            probability = sigmoid(eta)
            target = merge(1.0_dp, 0.0_dp, labels(i) == classes(2))
            residual = probability - target
            curvature = max(probability*(1.0_dp - probability), 1.0e-14_dp)
            gradient = gradient + weights(i)*residual*scores(i)
            hessian = hessian + weights(i)*curvature*scores(i)*scores(i)
        end do
    end subroutine temperature_derivatives

    subroutine sigmoid_derivatives(scores, labels, weights, classes, slope, intercept, l2, &
            gradient_slope, gradient_intercept, hessian_ss, hessian_si, hessian_ii)
        real(dp), intent(in) :: scores(:), weights(:), slope, intercept, l2
        integer, intent(in) :: labels(:), classes(2)
        real(dp), intent(out) :: gradient_slope, gradient_intercept, hessian_ss, hessian_si, hessian_ii
        real(dp) :: eta, probability, curvature, target, residual
        integer :: i

        gradient_slope = l2*slope
        gradient_intercept = l2*intercept
        hessian_ss = l2
        hessian_si = 0.0_dp
        hessian_ii = l2
        do i = 1, size(scores)
            eta = slope*scores(i) + intercept
            probability = sigmoid(eta)
            target = merge(1.0_dp, 0.0_dp, labels(i) == classes(2))
            residual = probability - target
            curvature = max(probability*(1.0_dp - probability), 1.0e-14_dp)
            gradient_slope = gradient_slope + weights(i)*residual*scores(i)
            gradient_intercept = gradient_intercept + weights(i)*residual
            hessian_ss = hessian_ss + weights(i)*curvature*scores(i)*scores(i)
            hessian_si = hessian_si + weights(i)*curvature*scores(i)
            hessian_ii = hessian_ii + weights(i)*curvature
        end do
    end subroutine sigmoid_derivatives

end module fortml_probability_calibration
