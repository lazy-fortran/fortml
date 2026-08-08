module fortml_rbf_svm_classifier
    !! Dense RBF binary kernel SVM with a deterministic primal RKHS fit.
    !!
    !! The finite training set is used as the RBF feature basis.  Fitting
    !! minimizes the weighted squared-hinge RKHS objective
    !!
    !!   0.5*c^T K c + C/sum(w) * sum_i w_i [1-y_i(Kc+b)]_+^2 .
    !!
    !! `c` is therefore an explicit dual/kernel-expansion coefficient vector,
    !! although the bounded FortOpt L-BFGS-B solve is performed in this
    !! equivalent primal coordinate system.  The objective is convex and the
    !! squared hinge is C1, but its second derivative changes at margin one;
    !! fit-state derivatives are intentionally not exposed.  Fixed-state
    !! decision and probability products are analytic and include the
    !! coefficient, intercept, log-gamma, and query-input tangents.
    !! Prediction uses score >= 0 for the sorted positive class.  Probabilities
    !! are the explicitly documented uncalibrated sigmoid of the decision
    !! score, not a fitted Platt model.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    real(dp), parameter :: SUPPORT_TOLERANCE = 1.0e-10_dp

    type, public :: rbf_svm_classifier_t
        private
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: coefficient(:)
        integer :: class_label(2) = 0
        real(dp) :: intercept_value = 0.0_dp
        real(dp) :: gamma_value = 1.0_dp
        real(dp) :: c_value = 1.0_dp
        integer :: n_features = 0
        integer :: n_samples = 0
        integer :: iterations_value = 0
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => rbf_svm_fit
        procedure, public :: decision_function => rbf_svm_decision_function
        procedure, public :: predict => rbf_svm_predict
        procedure, public :: predict_proba => rbf_svm_predict_proba
        procedure, public :: decision_function_device => rbf_svm_decision_device
        procedure, public :: predict_device => rbf_svm_predict_device
        procedure, public :: predict_proba_device => rbf_svm_predict_proba_device
        procedure, public :: decision_function_jvp_device => rbf_svm_decision_jvp_device
        procedure, public :: decision_function_vjp_device => rbf_svm_decision_vjp_device
        procedure, public :: predict_proba_jvp_device => rbf_svm_predict_proba_jvp_device
        procedure, public :: predict_proba_vjp_device => rbf_svm_predict_proba_vjp_device
        procedure, public :: device_supported => rbf_svm_device_supported
        procedure, public :: decision_function_jvp => rbf_svm_decision_jvp
        procedure, public :: decision_function_vjp => rbf_svm_decision_vjp
        procedure, public :: predict_proba_jvp => rbf_svm_predict_proba_jvp
        procedure, public :: predict_proba_vjp => rbf_svm_predict_proba_vjp
        procedure, public :: jvp => rbf_svm_decision_jvp
        procedure, public :: vjp => rbf_svm_decision_vjp
        procedure, public :: coefficients => rbf_svm_coefficients
        procedure, public :: intercept => rbf_svm_intercept
        procedure, public :: classes => rbf_svm_classes
        procedure, public :: gamma => rbf_svm_gamma
        procedure, public :: c_parameter => rbf_svm_c_parameter
        procedure, public :: parameter_count => rbf_svm_parameter_count
        procedure, public :: parameters => rbf_svm_parameters
        procedure, public :: set_parameters => rbf_svm_set_parameters
        procedure, public :: support_vector_count => rbf_svm_support_count
        procedure, public :: feature_count => rbf_svm_feature_count
        procedure, public :: sample_count => rbf_svm_sample_count
        procedure, public :: iterations => rbf_svm_iterations
        procedure, public :: fitted => rbf_svm_fitted
    end type rbf_svm_classifier_t

    public :: rbf_svm_fit
    public :: rbf_svm_decision_function
    public :: rbf_svm_predict

contains

    subroutine rbf_svm_fit(self, x, labels, status, c, gamma, max_iterations, &
            tolerance, sample_weight)
        class(rbf_svm_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: c, gamma, tolerance, sample_weight(:)
        integer, intent(in), optional :: max_iterations
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: options
        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: kernel(:, :), theta(:), lower(:), upper(:)
        real(dp), allocatable :: encoded(:), weights(:)
        real(dp) :: requested_c, requested_gamma, requested_tolerance
        real(dp) :: weight_mass
        integer :: requested_iterations, negative_label, positive_label
        integer :: n_samples, n_features, n_parameters, i

        self%fitted_value = .false.
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1 .or. size(labels) /= n_samples .or. &
                any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM fit: finite input dimensions and labels are required")
            return
        end if
        negative_label = minval(labels)
        positive_label = maxval(labels)
        if (negative_label == positive_label) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM fit: exactly two distinct classes are required")
            return
        end if
        do i = 1, n_samples
            if (labels(i) /= negative_label .and. labels(i) /= positive_label) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM fit: exactly two distinct classes are required")
                return
            end if
        end do

        requested_c = 1.0_dp
        if (present(c)) requested_c = c
        requested_gamma = 1.0_dp/real(n_features, dp)
        if (present(gamma)) requested_gamma = gamma
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        requested_iterations = 500
        if (present(max_iterations)) requested_iterations = max_iterations
        if (.not. ieee_is_finite(requested_c) .or. requested_c <= 0.0_dp .or. &
                .not. ieee_is_finite(requested_gamma) .or. requested_gamma <= 0.0_dp .or. &
                .not. ieee_is_finite(requested_tolerance) .or. requested_tolerance <= 0.0_dp .or. &
                requested_iterations < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM fit: C, gamma, tolerance, and iterations must be valid")
            return
        end if

        allocate(weights(n_samples), encoded(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                    any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        weight_mass = sum(weights)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM fit: sample weights must have positive mass")
            return
        end if
        encoded = -1.0_dp
        where (labels == positive_label) encoded = 1.0_dp

        allocate(kernel(n_samples, n_samples))
        call rbf_kernel_matrix(x, requested_gamma, kernel, status)
        if (status%code /= FORTNUM_OK) return
        ! Fit only the convex kernel coefficients and intercept.  log(gamma)
        ! is part of the fixed-state prediction parameter pack below, but
        ! allowing the optimizer to move it would require refitting K and
        ! differentiating the active margin set.
        n_parameters = n_samples + 1
        allocate(theta(n_parameters), lower(n_parameters), upper(n_parameters))
        theta = 0.0_dp
        lower = -huge(1.0_dp)
        upper = huge(1.0_dp)
        call objective%initialize(n_parameters, rbf_svm_fit_objective, status)
        if (status%code /= FORTNUM_OK) return
        options%max_iterations = requested_iterations
        ! A tiny requested tolerance below the reliable Armijo resolution of
        ! the dense kernel solve otherwise turns an already converged iterate
        ! into a line-search refusal.  Keep the public tolerance validation,
        ! but use a numerically meaningful floor for this finite-basis fit.
        options%gradient_tolerance = max(requested_tolerance, 1.0e-6_dp)
        options%step_tolerance = min(1.0e-12_dp, requested_tolerance)
        options%objective_tolerance = min(1.0e-12_dp, requested_tolerance)
        options%max_line_search = 100
        options%armijo_constant = 1.0e-8_dp
        call optimizer%minimize(objective, theta, lower, upper, options, result, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. result%state%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "RBF SVM fit: FortOpt reached its iteration limit")
            return
        end if
        if (any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "RBF SVM fit: optimizer returned nonfinite parameters")
            return
        end if

        allocate(self%x_train(n_samples, n_features), self%coefficient(n_samples))
        self%x_train = x
        self%coefficient = theta(:n_samples)
        self%intercept_value = theta(n_samples + 1)
        self%class_label = [negative_label, positive_label]
        self%gamma_value = requested_gamma
        self%c_value = requested_c
        self%n_features = n_features
        self%n_samples = n_samples
        self%iterations_value = result%state%iteration
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")

    contains

        subroutine rbf_svm_fit_objective(parameters, value, gradient, objective_status)
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: value, gradient(:)
            type(fortnum_status_t), intent(out) :: objective_status
            real(dp), allocatable :: scores(:), residual(:), score_gradient(:)
            value = 0.0_dp
            gradient = 0.0_dp
            if (size(parameters) /= n_parameters .or. size(gradient) /= n_parameters) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM objective: parameter shape is invalid")
                return
            end if
            allocate(scores(n_samples), residual(n_samples), score_gradient(n_samples))
            scores = matmul(kernel, parameters(:n_samples)) + parameters(n_samples + 1)
            residual = max(0.0_dp, 1.0_dp - encoded*scores)
            score_gradient = -2.0_dp*requested_c/weight_mass*weights*encoded*residual
            value = 0.5_dp*dot_product(parameters(:n_samples), &
                matmul(kernel, parameters(:n_samples))) + &
                requested_c/weight_mass*sum(weights*residual*residual)
            gradient(:n_samples) = matmul(kernel, parameters(:n_samples)) + &
                matmul(kernel, score_gradient)
            gradient(n_samples + 1) = sum(score_gradient)
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM objective: value or gradient is nonfinite")
                return
            end if
            call status_set(objective_status, FORTNUM_OK, "")
        end subroutine rbf_svm_fit_objective

    end subroutine rbf_svm_fit

    subroutine rbf_svm_decision_function(self, x, scores, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: d2, delta

        scores = 0.0_dp
        if (.not. rbf_svm_fitted(self) .or. size(x, 1) < 1 .or. &
                size(x, 2) /= self%n_features .or. size(scores) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM decision: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM decision: inputs must be finite")
            return
        end if
        do i = 1, size(x, 1)
            scores(i) = self%intercept_value
            do j = 1, self%n_samples
                d2 = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                end do
                if (.not. ieee_is_finite(d2)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "RBF SVM decision: distance overflow")
                    return
                end if
                scores(i) = scores(i) + self%coefficient(j)*exp(-self%gamma_value*d2)
            end do
        end do
        if (any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM decision: score overflow")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_decision_function

    subroutine rbf_svm_predict(self, x, labels, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i

        labels = 0
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM predict: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            if (scores(i) >= 0.0_dp) then
                labels(i) = self%class_label(2)
            else
                labels(i) = self%class_label(1)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_predict

    subroutine rbf_svm_predict_proba(self, x, probabilities, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i

        if (size(probabilities, 1) /= size(x, 1) .or. size(probabilities, 2) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM predict_proba: output shape must be (n_samples,2)")
            return
        end if
        allocate(scores(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(scores)
            probabilities(i, 2) = stable_sigmoid(scores(i))
            probabilities(i, 1) = 1.0_dp - probabilities(i, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_predict_proba

    subroutine rbf_svm_decision_device(self, device, x, scores, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status

        scores = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM device decision: resident CUDA RBF SVM kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device decision: device kind is invalid")
        end select
    end subroutine rbf_svm_decision_device

    subroutine rbf_svm_predict_device(self, device, x, labels, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        labels = 0
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM device prediction: resident CUDA RBF SVM kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device prediction: device kind is invalid")
        end select
    end subroutine rbf_svm_predict_device

    subroutine rbf_svm_predict_proba_device(self, device, x, probabilities, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM device probability: resident CUDA RBF SVM kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device probability: device kind is invalid")
        end select
    end subroutine rbf_svm_predict_proba_device

    subroutine rbf_svm_decision_jvp_device(self, device, x, theta_dot, x_dot, &
            scores, scores_dot, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:), scores_dot(:)
        type(fortnum_status_t), intent(out) :: status

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device decision JVP: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM device decision JVP: resident CUDA derivative kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device decision JVP: device kind is invalid")
        end select
    end subroutine rbf_svm_decision_jvp_device

    subroutine rbf_svm_decision_vjp_device(self, device, x, scores_bar, theta_bar, &
            x_bar, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), scores_bar(:)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device decision VJP: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function_vjp(x, scores_bar, theta_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM device decision VJP: resident CUDA derivative kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device decision VJP: device kind is invalid")
        end select
    end subroutine rbf_svm_decision_vjp_device

    subroutine rbf_svm_predict_proba_jvp_device(self, device, x, theta_dot, x_dot, &
            probabilities, probabilities_dot, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device probability JVP: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_jvp(x, theta_dot, x_dot, probabilities, probabilities_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM device probability JVP: resident CUDA derivative kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device probability JVP: device kind is invalid")
        end select
    end subroutine rbf_svm_predict_proba_jvp_device

    subroutine rbf_svm_predict_proba_vjp_device(self, device, x, probabilities_bar, &
            theta_bar, x_bar, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device probability VJP: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_vjp(x, probabilities_bar, theta_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM device probability VJP: resident CUDA derivative kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM device probability VJP: device kind is invalid")
        end select
    end subroutine rbf_svm_predict_proba_vjp_device

    subroutine rbf_svm_decision_jvp(self, x, theta_dot, x_dot, scores, scores_dot, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:), scores_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k, n
        real(dp) :: d2, delta, direction, kernel_value, dkernel_x, dloggamma

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. rbf_svm_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM decision JVP: model is not fitted")
            return
        end if
        n = self%n_samples
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
                any(shape(x_dot) /= shape(x)) .or. size(scores) /= size(x, 1) .or. &
                size(scores_dot) /= size(scores) .or. size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM decision JVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM decision JVP: inputs and tangents must be finite")
            return
        end if
        do i = 1, size(x, 1)
            scores(i) = self%intercept_value
            scores_dot(i) = theta_dot(n + 1)
            do j = 1, n
                d2 = 0.0_dp
                direction = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                    direction = direction + delta*x_dot(i, k)
                end do
                kernel_value = exp(-self%gamma_value*d2)
                dkernel_x = -2.0_dp*self%gamma_value*kernel_value*direction
                dloggamma = -self%gamma_value*d2*kernel_value
                scores(i) = scores(i) + self%coefficient(j)*kernel_value
                scores_dot(i) = scores_dot(i) + kernel_value*theta_dot(j) + &
                    self%coefficient(j)*(dkernel_x + dloggamma*theta_dot(n + 2))
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_decision_jvp

    subroutine rbf_svm_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k, n
        real(dp) :: d2, delta, kernel_value, dloggamma, coefficient

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. rbf_svm_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM decision VJP: model is not fitted")
            return
        end if
        n = self%n_samples
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
                size(scores_bar) /= size(x, 1) .or. size(theta_bar) /= self%parameter_count() .or. &
                any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM decision VJP: cotangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM decision VJP: inputs and cotangents must be finite")
            return
        end if
        theta_bar(n + 1) = sum(scores_bar)
        do i = 1, size(x, 1)
            do j = 1, n
                d2 = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                end do
                kernel_value = exp(-self%gamma_value*d2)
                coefficient = scores_bar(i)*self%coefficient(j)
                theta_bar(j) = theta_bar(j) + scores_bar(i)*kernel_value
                dloggamma = -self%gamma_value*d2*kernel_value
                theta_bar(n + 2) = theta_bar(n + 2) + coefficient*dloggamma
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    x_bar(i, k) = x_bar(i, k) - 2.0_dp*self%gamma_value* &
                        coefficient*kernel_value*delta
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_decision_vjp

    subroutine rbf_svm_predict_proba_jvp(self, x, theta_dot, x_dot, probabilities, &
            probabilities_dot, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:), scores_dot(:)
        real(dp) :: positive, positive_dot
        integer :: i

        if (size(probabilities, 1) /= size(x, 1) .or. size(probabilities, 2) /= 2 .or. &
                any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM probability JVP: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)), scores_dot(size(x, 1)))
        call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(scores)
            positive = stable_sigmoid(scores(i))
            positive_dot = positive*(1.0_dp - positive)*scores_dot(i)
            probabilities(i, 2) = positive
            probabilities(i, 1) = 1.0_dp - positive
            probabilities_dot(i, 2) = positive_dot
            probabilities_dot(i, 1) = -positive_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_predict_proba_jvp

    subroutine rbf_svm_predict_proba_vjp(self, x, probabilities_bar, theta_bar, x_bar, status)
        class(rbf_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), scores_bar(:)
        integer :: i

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (size(probabilities_bar, 1) /= size(x, 1) .or. size(probabilities_bar, 2) /= 2 .or. &
                any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM probability VJP: cotangent shape or values are invalid")
            return
        end if
        allocate(probabilities(size(x, 1), 2), scores_bar(size(x, 1)))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            scores_bar(i) = (probabilities_bar(i, 2) - probabilities_bar(i, 1))* &
                probabilities(i, 1)*probabilities(i, 2)
        end do
        call self%decision_function_vjp(x, scores_bar, theta_bar, x_bar, status)
    end subroutine rbf_svm_predict_proba_vjp

    logical function rbf_svm_device_supported(self, device_kind) result(supported)
        class(rbf_svm_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%fitted_value .and. device_kind == FORTML_DEVICE_CPU
    end function rbf_svm_device_supported

    function rbf_svm_coefficients(self) result(values)
        class(rbf_svm_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%coefficient)) then
            values = self%coefficient
        else
            allocate(values(0))
        end if
    end function rbf_svm_coefficients

    real(dp) function rbf_svm_intercept(self) result(value)
        class(rbf_svm_classifier_t), intent(in) :: self
        value = self%intercept_value
    end function rbf_svm_intercept

    function rbf_svm_classes(self) result(values)
        class(rbf_svm_classifier_t), intent(in) :: self
        integer :: values(2)

        values = self%class_label
    end function rbf_svm_classes

    real(dp) function rbf_svm_gamma(self) result(value)
        class(rbf_svm_classifier_t), intent(in) :: self
        value = self%gamma_value
    end function rbf_svm_gamma

    real(dp) function rbf_svm_c_parameter(self) result(value)
        class(rbf_svm_classifier_t), intent(in) :: self
        value = self%c_value
    end function rbf_svm_c_parameter

    integer function rbf_svm_parameter_count(self) result(count)
        class(rbf_svm_classifier_t), intent(in) :: self

        count = 0
        if (allocated(self%coefficient)) count = size(self%coefficient) + 2
    end function rbf_svm_parameter_count

    function rbf_svm_parameters(self) result(values)
        class(rbf_svm_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: n

        if (.not. allocated(self%coefficient)) then
            allocate(values(0))
            return
        end if
        n = size(self%coefficient)
        allocate(values(n + 2))
        values(:n) = self%coefficient
        values(n + 1) = self%intercept_value
        values(n + 2) = log(self%gamma_value)
    end function rbf_svm_parameters

    subroutine rbf_svm_set_parameters(self, values, status)
        class(rbf_svm_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n
        real(dp) :: gamma

        if (.not. rbf_svm_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM set_parameters: model is not fitted")
            return
        end if
        if (size(values) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM set_parameters: parameter shape or values are invalid")
            return
        end if
        gamma = exp(values(self%n_samples + 2))
        if (.not. ieee_is_finite(gamma) .or. gamma <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM set_parameters: log-gamma overflows or is nonpositive")
            return
        end if
        n = self%n_samples
        self%coefficient = values(:n)
        self%intercept_value = values(n + 1)
        self%gamma_value = gamma
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_set_parameters

    integer function rbf_svm_support_count(self) result(value)
        class(rbf_svm_classifier_t), intent(in) :: self

        value = 0
        if (allocated(self%coefficient)) value = count(abs(self%coefficient) > SUPPORT_TOLERANCE)
    end function rbf_svm_support_count

    integer function rbf_svm_feature_count(self) result(value)
        class(rbf_svm_classifier_t), intent(in) :: self
        value = self%n_features
    end function rbf_svm_feature_count

    integer function rbf_svm_sample_count(self) result(value)
        class(rbf_svm_classifier_t), intent(in) :: self
        value = self%n_samples
    end function rbf_svm_sample_count

    integer function rbf_svm_iterations(self) result(value)
        class(rbf_svm_classifier_t), intent(in) :: self
        value = self%iterations_value
    end function rbf_svm_iterations

    logical function rbf_svm_fitted(self) result(value)
        class(rbf_svm_classifier_t), intent(in) :: self
        value = self%fitted_value .and. allocated(self%coefficient) .and. &
            allocated(self%x_train)
    end function rbf_svm_fitted

    subroutine rbf_kernel_matrix(x, gamma, kernel, status)
        real(dp), intent(in) :: x(:, :), gamma
        real(dp), intent(out) :: kernel(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: d2, delta

        kernel = 0.0_dp
        if (any(shape(kernel) /= [size(x, 1), size(x, 1)])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM kernel: output shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            do j = 1, size(x, 1)
                d2 = 0.0_dp
                do k = 1, size(x, 2)
                    delta = x(i, k) - x(j, k)
                    d2 = d2 + delta*delta
                end do
                if (.not. ieee_is_finite(d2)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "RBF SVM kernel: squared distance overflow")
                    return
                end if
                kernel(i, j) = exp(-gamma*d2)
            end do
        end do
        if (any(.not. ieee_is_finite(kernel))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM kernel: nonfinite values")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_kernel_matrix

    pure real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

end module fortml_rbf_svm_classifier
