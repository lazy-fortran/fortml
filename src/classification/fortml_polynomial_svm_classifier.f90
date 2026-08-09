module fortml_polynomial_svm_classifier
    !! Dense polynomial-kernel binary SVM.
    !!
    !! The finite training set is the polynomial feature basis.  Fitting
    !! minimizes the weighted squared-hinge RKHS objective with FortOpt
    !! L-BFGS-B.  The fixed-state score, probability, JVP, and VJP products
    !! are analytic.  CUDA dispatch is an explicit refusal until a resident
    !! polynomial-SVM kernel is linked.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    real(dp), parameter :: SUPPORT_TOLERANCE = 1.0e-10_dp

    type, public :: polynomial_svm_classifier_t
        private
        real(dp), allocatable :: x_train(:, :), coefficient(:)
        integer :: class_label(2) = 0
        real(dp) :: intercept_value = 0.0_dp
        real(dp) :: gamma_value = 1.0_dp
        real(dp) :: coef0_value = 1.0_dp
        real(dp) :: c_value = 1.0_dp
        integer :: degree_value = 3
        integer :: n_features = 0, n_samples = 0, iterations_value = 0
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => polynomial_svm_fit
        procedure, public :: decision_function => polynomial_svm_decision_function
        procedure, public :: predict => polynomial_svm_predict
        procedure, public :: predict_proba => polynomial_svm_predict_proba
        procedure, public :: decision_function_device => polynomial_svm_decision_device
        procedure, public :: predict_device => polynomial_svm_predict_device
        procedure, public :: predict_proba_device => polynomial_svm_predict_proba_device
        procedure, public :: decision_function_jvp => polynomial_svm_decision_jvp
        procedure, public :: decision_function_vjp => polynomial_svm_decision_vjp
        procedure, public :: predict_proba_jvp => polynomial_svm_predict_proba_jvp
        procedure, public :: predict_proba_vjp => polynomial_svm_predict_proba_vjp
        procedure, public :: decision_function_jvp_device => polynomial_svm_decision_jvp_device
        procedure, public :: decision_function_vjp_device => polynomial_svm_decision_vjp_device
        procedure, public :: predict_proba_jvp_device => polynomial_svm_predict_proba_jvp_device
        procedure, public :: predict_proba_vjp_device => polynomial_svm_predict_proba_vjp_device
        procedure, public :: device_supported => polynomial_svm_device_supported
        procedure, public :: coefficients => polynomial_svm_coefficients
        procedure, public :: intercept => polynomial_svm_intercept
        procedure, public :: classes => polynomial_svm_classes
        procedure, public :: gamma => polynomial_svm_gamma
        procedure, public :: coef0 => polynomial_svm_coef0
        procedure, public :: degree => polynomial_svm_degree
        procedure, public :: c_parameter => polynomial_svm_c_parameter
        procedure, public :: parameter_count => polynomial_svm_parameter_count
        procedure, public :: parameters => polynomial_svm_parameters
        procedure, public :: set_parameters => polynomial_svm_set_parameters
        procedure, public :: support_vector_count => polynomial_svm_support_count
        procedure, public :: feature_count => polynomial_svm_feature_count
        procedure, public :: sample_count => polynomial_svm_sample_count
        procedure, public :: iterations => polynomial_svm_iterations
        procedure, public :: fitted => polynomial_svm_fitted
    end type polynomial_svm_classifier_t

    public :: polynomial_svm_fit, polynomial_svm_decision_function, polynomial_svm_predict

contains

    subroutine polynomial_svm_fit(self, x, labels, status, c, gamma, degree, coef0, &
            max_iterations, tolerance, sample_weight)
        class(polynomial_svm_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: c, gamma, coef0, tolerance, sample_weight(:)
        integer, intent(in), optional :: degree, max_iterations
        type(polynomial_svm_classifier_t) :: candidate
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: options
        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: kernel(:, :), theta(:), lower(:), upper(:)
        real(dp), allocatable :: encoded(:), weights(:)
        real(dp) :: requested_c, requested_gamma, requested_coef0, requested_tolerance
        real(dp) :: weight_mass
        integer :: requested_degree, requested_iterations
        integer :: negative_label, positive_label, n_samples, n_features
        integer :: n_parameters, i

        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1 .or. size(labels) /= n_samples .or. &
                any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "polynomial SVM fit: finite input dimensions and labels are required")
            return
        end if
        negative_label = minval(labels)
        positive_label = maxval(labels)
        if (negative_label == positive_label) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "polynomial SVM fit: exactly two distinct classes are required")
            return
        end if
        do i = 1, n_samples
            if (labels(i) /= negative_label .and. labels(i) /= positive_label) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "polynomial SVM fit: exactly two distinct classes are required")
                return
            end if
        end do
        requested_c = 1.0_dp
        if (present(c)) requested_c = c
        requested_gamma = 1.0_dp/real(n_features, dp)
        if (present(gamma)) requested_gamma = gamma
        requested_degree = 3
        if (present(degree)) requested_degree = degree
        requested_coef0 = 1.0_dp
        if (present(coef0)) requested_coef0 = coef0
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        requested_iterations = 500
        if (present(max_iterations)) requested_iterations = max_iterations
        if (.not. ieee_is_finite(requested_c) .or. requested_c <= 0.0_dp .or. &
                .not. ieee_is_finite(requested_gamma) .or. requested_gamma <= 0.0_dp .or. &
                requested_degree < 1 .or. .not. ieee_is_finite(requested_coef0) .or. &
                requested_coef0 < 0.0_dp .or. .not. ieee_is_finite(requested_tolerance) .or. &
                requested_tolerance <= 0.0_dp .or. requested_iterations < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "polynomial SVM fit: C, gamma, degree, coef0, tolerance, and iterations must be valid")
            return
        end if
        allocate(weights(n_samples), encoded(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                    any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "polynomial SVM fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        weight_mass = sum(weights)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "polynomial SVM fit: sample weights must have positive mass")
            return
        end if
        encoded = -1.0_dp
        where (labels == positive_label) encoded = 1.0_dp

        allocate(kernel(n_samples, n_samples))
        call polynomial_kernel_matrix(x, requested_gamma, requested_degree, requested_coef0, &
            kernel, status)
        if (status%code /= FORTNUM_OK) return
        n_parameters = n_samples + 3
        allocate(theta(n_parameters), lower(n_parameters), upper(n_parameters))
        theta = 0.0_dp
        theta(n_samples + 2) = log(requested_gamma)
        theta(n_samples + 3) = requested_coef0
        lower = -huge(1.0_dp)
        upper = huge(1.0_dp)
        call objective%initialize(n_parameters, polynomial_svm_fit_objective, status)
        if (status%code /= FORTNUM_OK) return
        options%max_iterations = requested_iterations
        options%gradient_tolerance = max(requested_tolerance, 1.0e-6_dp)
        options%step_tolerance = min(1.0e-12_dp, requested_tolerance)
        options%objective_tolerance = min(1.0e-12_dp, requested_tolerance)
        options%max_line_search = 100
        options%armijo_constant = 1.0e-8_dp
        call optimizer%minimize(objective, theta, lower, upper, options, result, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. result%state%converged .or. any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "polynomial SVM fit: FortOpt did not return a finite converged state")
            return
        end if
        allocate(candidate%x_train(n_samples, n_features), candidate%coefficient(n_samples))
        candidate%x_train = x
        candidate%coefficient = theta(:n_samples)
        candidate%intercept_value = theta(n_samples + 1)
        candidate%gamma_value = requested_gamma
        candidate%coef0_value = requested_coef0
        candidate%degree_value = requested_degree
        candidate%c_value = requested_c
        candidate%class_label = [negative_label, positive_label]
        candidate%n_features = n_features
        candidate%n_samples = n_samples
        candidate%iterations_value = result%state%iteration
        candidate%fitted_value = .true.
        self%x_train = candidate%x_train
        self%coefficient = candidate%coefficient
        self%intercept_value = candidate%intercept_value
        self%gamma_value = candidate%gamma_value
        self%coef0_value = candidate%coef0_value
        self%c_value = candidate%c_value
        self%degree_value = candidate%degree_value
        self%class_label = candidate%class_label
        self%n_features = candidate%n_features
        self%n_samples = candidate%n_samples
        self%iterations_value = candidate%iterations_value
        self%fitted_value = candidate%fitted_value
        call status_set(status, FORTNUM_OK, "")

    contains

        subroutine polynomial_svm_fit_objective(parameters, value, gradient, objective_status)
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: value, gradient(:)
            type(fortnum_status_t), intent(out) :: objective_status
            real(dp), allocatable :: scores(:), residual(:), score_gradient(:)
            value = 0.0_dp
            gradient = 0.0_dp
            if (size(parameters) /= n_parameters .or. size(gradient) /= n_parameters) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "polynomial SVM objective: parameter shape is invalid")
                return
            end if
            if (.not. ieee_is_finite(parameters(n_samples + 3))) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "polynomial SVM objective: coef0 is nonfinite")
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
            gradient(n_samples + 2) = 0.0_dp
            gradient(n_samples + 3) = 0.0_dp
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "polynomial SVM objective: value or gradient is nonfinite")
                return
            end if
            call status_set(objective_status, FORTNUM_OK, "")
        end subroutine polynomial_svm_fit_objective
    end subroutine polynomial_svm_fit

    subroutine polynomial_svm_decision_function(self, x, scores, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j
        real(dp) :: base
        scores = 0.0_dp
        if (.not. polynomial_svm_fitted(self) .or. size(x, 1) < 1 .or. &
                size(x, 2) /= self%n_features .or. size(scores) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "polynomial SVM decision: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM decision: inputs must be finite")
            return
        end if
        do i = 1, size(x, 1)
            scores(i) = self%intercept_value
            do j = 1, self%n_samples
                base = self%gamma_value*dot_product(x(i, :), self%x_train(j, :)) + self%coef0_value
                scores(i) = scores(i) + self%coefficient(j)*base**self%degree_value
            end do
        end do
        if (any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM decision: score overflow")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_svm_decision_function

    subroutine polynomial_svm_predict(self, x, labels, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i
        labels = 0
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM predict: output shape is invalid")
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
    end subroutine polynomial_svm_predict

    subroutine polynomial_svm_predict_proba(self, x, probabilities, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i
        probabilities = 0.0_dp
        if (size(probabilities, 1) /= size(x, 1) .or. size(probabilities, 2) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "polynomial SVM predict_proba: output shape must be (n_samples,2)")
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
    end subroutine polynomial_svm_predict_proba

    subroutine polynomial_svm_decision_jvp(self, x, theta_dot, x_dot, scores, scores_dot, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:), scores_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j
        real(dp) :: base, dot, dbase, factor, kval
        scores = 0.0_dp; scores_dot = 0.0_dp
        if (.not. polynomial_svm_valid_derivative_shapes(self, x, theta_dot, x_dot, scores, scores_dot, status)) return
        do i = 1, size(x, 1)
            scores(i) = self%intercept_value
            scores_dot(i) = theta_dot(self%n_samples + 1)
            do j = 1, self%n_samples
                dot = dot_product(x(i, :), self%x_train(j, :))
                base = self%gamma_value*dot + self%coef0_value
                kval = base**self%degree_value
                factor = real(self%degree_value, dp)*base**(self%degree_value - 1)
                dbase = self%gamma_value*dot_product(x_dot(i, :), self%x_train(j, :)) + &
                    self%gamma_value*dot*theta_dot(self%n_samples + 2) + theta_dot(self%n_samples + 3)
                scores(i) = scores(i) + self%coefficient(j)*kval
                scores_dot(i) = scores_dot(i) + kval*theta_dot(j) + &
                    self%coefficient(j)*factor*dbase
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_svm_decision_jvp

    subroutine polynomial_svm_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: base, dot, factor, kval, scale
        theta_bar = 0.0_dp; x_bar = 0.0_dp
        if (.not. polynomial_svm_valid_vjp_shapes(self, x, scores_bar, theta_bar, x_bar, status)) return
        theta_bar(self%n_samples + 1) = sum(scores_bar)
        do i = 1, size(x, 1)
            do j = 1, self%n_samples
                dot = dot_product(x(i, :), self%x_train(j, :))
                base = self%gamma_value*dot + self%coef0_value
                kval = base**self%degree_value
                factor = real(self%degree_value, dp)*base**(self%degree_value - 1)
                scale = scores_bar(i)*self%coefficient(j)*factor
                theta_bar(j) = theta_bar(j) + scores_bar(i)*kval
                theta_bar(self%n_samples + 2) = theta_bar(self%n_samples + 2) + scale*self%gamma_value*dot
                theta_bar(self%n_samples + 3) = theta_bar(self%n_samples + 3) + scale
                do k = 1, self%n_features
                    x_bar(i, k) = x_bar(i, k) + scale*self%gamma_value*self%x_train(j, k)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_svm_decision_vjp

    subroutine polynomial_svm_predict_proba_jvp(self, x, theta_dot, x_dot, probabilities, probabilities_dot, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:), scores_dot(:)
        real(dp) :: p, p_dot
        integer :: i
        probabilities = 0.0_dp; probabilities_dot = 0.0_dp
        if (size(probabilities, 1) /= size(x, 1) .or. size(probabilities, 2) /= 2 .or. &
                any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM probability JVP: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)), scores_dot(size(x, 1)))
        call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(scores)
            p = stable_sigmoid(scores(i)); p_dot = p*(1.0_dp-p)*scores_dot(i)
            probabilities(i, :) = [1.0_dp-p, p]
            probabilities_dot(i, :) = [-p_dot, p_dot]
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_svm_predict_proba_jvp

    subroutine polynomial_svm_predict_proba_vjp(self, x, probabilities_bar, theta_bar, x_bar, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), scores_bar(:)
        integer :: i
        theta_bar = 0.0_dp; x_bar = 0.0_dp
        if (size(probabilities_bar, 1) /= size(x, 1) .or. size(probabilities_bar, 2) /= 2 .or. &
                any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM probability VJP: cotangent shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), 2), scores_bar(size(x, 1)))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            scores_bar(i) = (probabilities_bar(i, 2)-probabilities_bar(i, 1))* &
                probabilities(i, 1)*probabilities(i, 2)
        end do
        call self%decision_function_vjp(x, scores_bar, theta_bar, x_bar, status)
    end subroutine polynomial_svm_predict_proba_vjp

    subroutine polynomial_svm_decision_device(self, device, x, scores, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        scores = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device decision: device is not selected"); return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU); call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA); call status_set(status, FORTNUM_NOT_IMPLEMENTED, "polynomial SVM device decision: resident CUDA kernel is not linked")
        case default; call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device decision: device kind is invalid")
        end select
    end subroutine polynomial_svm_decision_device

    subroutine polynomial_svm_predict_device(self, device, x, labels, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        labels = 0
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device prediction: device is not selected"); return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU); call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA); call status_set(status, FORTNUM_NOT_IMPLEMENTED, "polynomial SVM device prediction: resident CUDA kernel is not linked")
        case default; call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device prediction: device kind is invalid")
        end select
    end subroutine polynomial_svm_predict_device

    subroutine polynomial_svm_predict_proba_device(self, device, x, probabilities, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        probabilities = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device probability: device is not selected"); return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU); call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA); call status_set(status, FORTNUM_NOT_IMPLEMENTED, "polynomial SVM device probability: resident CUDA kernel is not linked")
        case default; call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device probability: device kind is invalid")
        end select
    end subroutine polynomial_svm_predict_proba_device

    subroutine polynomial_svm_decision_jvp_device(self, device, x, theta_dot, x_dot, scores, scores_dot, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:), scores_dot(:)
        type(fortnum_status_t), intent(out) :: status
        scores = 0.0_dp; scores_dot = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device decision JVP: device is not selected"); return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU); call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        case (FORTML_DEVICE_CUDA); call status_set(status, FORTNUM_NOT_IMPLEMENTED, "polynomial SVM device decision JVP: resident CUDA kernel is not linked")
        case default; call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device decision JVP: device kind is invalid")
        end select
    end subroutine polynomial_svm_decision_jvp_device

    subroutine polynomial_svm_decision_vjp_device(self, device, x, scores_bar, theta_bar, x_bar, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), scores_bar(:)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        theta_bar = 0.0_dp; x_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device decision VJP: device is not selected"); return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU); call self%decision_function_vjp(x, scores_bar, theta_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA); call status_set(status, FORTNUM_NOT_IMPLEMENTED, "polynomial SVM device decision VJP: resident CUDA kernel is not linked")
        case default; call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device decision VJP: device kind is invalid")
        end select
    end subroutine polynomial_svm_decision_vjp_device

    subroutine polynomial_svm_predict_proba_jvp_device(self, device, x, theta_dot, x_dot, probabilities, probabilities_dot, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        probabilities = 0.0_dp; probabilities_dot = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device probability JVP: device is not selected"); return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU); call self%predict_proba_jvp(x, theta_dot, x_dot, probabilities, probabilities_dot, status)
        case (FORTML_DEVICE_CUDA); call status_set(status, FORTNUM_NOT_IMPLEMENTED, "polynomial SVM device probability JVP: resident CUDA kernel is not linked")
        case default; call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device probability JVP: device kind is invalid")
        end select
    end subroutine polynomial_svm_predict_proba_jvp_device

    subroutine polynomial_svm_predict_proba_vjp_device(self, device, x, probabilities_bar, theta_bar, x_bar, status)
        class(polynomial_svm_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        theta_bar = 0.0_dp; x_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device probability VJP: device is not selected"); return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU); call self%predict_proba_vjp(x, probabilities_bar, theta_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA); call status_set(status, FORTNUM_NOT_IMPLEMENTED, "polynomial SVM device probability VJP: resident CUDA kernel is not linked")
        case default; call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM device probability VJP: device kind is invalid")
        end select
    end subroutine polynomial_svm_predict_proba_vjp_device

    logical function polynomial_svm_device_supported(self, device_kind) result(supported)
        class(polynomial_svm_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        supported = self%fitted_value .and. device_kind == FORTML_DEVICE_CPU
    end function polynomial_svm_device_supported

    function polynomial_svm_coefficients(self) result(values)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        if (allocated(self%coefficient)) then; values = self%coefficient
        else; allocate(values(0)); end if
    end function polynomial_svm_coefficients
    real(dp) function polynomial_svm_intercept(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self; value = self%intercept_value
    end function polynomial_svm_intercept
    function polynomial_svm_classes(self) result(values)
        class(polynomial_svm_classifier_t), intent(in) :: self; integer :: values(2); values = self%class_label
    end function polynomial_svm_classes
    real(dp) function polynomial_svm_gamma(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self; value = self%gamma_value
    end function polynomial_svm_gamma
    real(dp) function polynomial_svm_coef0(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self; value = self%coef0_value
    end function polynomial_svm_coef0
    integer function polynomial_svm_degree(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self; value = self%degree_value
    end function polynomial_svm_degree
    real(dp) function polynomial_svm_c_parameter(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self; value = self%c_value
    end function polynomial_svm_c_parameter
    integer function polynomial_svm_parameter_count(self) result(count)
        class(polynomial_svm_classifier_t), intent(in) :: self
        count = 0; if (allocated(self%coefficient)) count = size(self%coefficient) + 3
    end function polynomial_svm_parameter_count
    function polynomial_svm_parameters(self) result(values)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:); integer :: n
        if (.not. allocated(self%coefficient)) then; allocate(values(0)); return; end if
        n = size(self%coefficient); allocate(values(n + 3)); values(:n) = self%coefficient
        values(n + 1) = self%intercept_value; values(n + 2) = log(self%gamma_value)
        values(n + 3) = self%coef0_value
    end function polynomial_svm_parameters
    subroutine polynomial_svm_set_parameters(self, values, status)
        class(polynomial_svm_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:); type(fortnum_status_t), intent(out) :: status
        real(dp) :: gamma; integer :: n
        if (.not. polynomial_svm_fitted(self) .or. size(values) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM set_parameters: model or values are invalid"); return
        end if
        n = self%n_samples; gamma = exp(values(n + 2))
        if (.not. ieee_is_finite(gamma) .or. gamma <= 0.0_dp .or. values(n + 3) < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM set_parameters: gamma or coef0 is invalid"); return
        end if
        self%coefficient = values(:n); self%intercept_value = values(n + 1)
        self%gamma_value = gamma; self%coef0_value = values(n + 3)
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_svm_set_parameters
    integer function polynomial_svm_support_count(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self
        value = 0; if (allocated(self%coefficient)) value = count(abs(self%coefficient) > SUPPORT_TOLERANCE)
    end function polynomial_svm_support_count
    integer function polynomial_svm_feature_count(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self; value = self%n_features
    end function polynomial_svm_feature_count
    integer function polynomial_svm_sample_count(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self; value = self%n_samples
    end function polynomial_svm_sample_count
    integer function polynomial_svm_iterations(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self; value = self%iterations_value
    end function polynomial_svm_iterations
    logical function polynomial_svm_fitted(self) result(value)
        class(polynomial_svm_classifier_t), intent(in) :: self
        value = self%fitted_value .and. allocated(self%coefficient) .and. allocated(self%x_train)
    end function polynomial_svm_fitted

    subroutine polynomial_kernel_matrix(x, gamma, degree, coef0, kernel, status)
        real(dp), intent(in) :: x(:, :), gamma, coef0
        integer, intent(in) :: degree
        real(dp), intent(out) :: kernel(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j
        real(dp) :: base
        kernel = 0.0_dp
        if (any(shape(kernel) /= [size(x, 1), size(x, 1)])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM kernel: output shape is invalid"); return
        end if
        do i = 1, size(x, 1)
            do j = 1, size(x, 1)
                base = gamma*dot_product(x(i, :), x(j, :)) + coef0
                kernel(i, j) = base**degree
                if (.not. ieee_is_finite(kernel(i, j))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM kernel: value overflow"); return
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_kernel_matrix

    logical function polynomial_svm_valid_derivative_shapes(self, x, theta_dot, x_dot, scores, scores_dot, status) result(valid)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :), scores(:), scores_dot(:)
        type(fortnum_status_t), intent(out) :: status
        valid = .false.
        if (.not. polynomial_svm_fitted(self) .or. size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
                any(shape(x_dot) /= shape(x)) .or. size(scores) /= size(x, 1) .or. &
                size(scores_dot) /= size(scores) .or. size(theta_dot) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM JVP: model, shapes, or values are invalid"); return
        end if
        valid = .true.
    end function polynomial_svm_valid_derivative_shapes
    logical function polynomial_svm_valid_vjp_shapes(self, x, scores_bar, theta_bar, x_bar, status) result(valid)
        class(polynomial_svm_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:), theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        valid = .false.
        if (.not. polynomial_svm_fitted(self) .or. size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
                size(scores_bar) /= size(x, 1) .or. size(theta_bar) /= self%parameter_count() .or. &
                any(shape(x_bar) /= shape(x)) .or. any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "polynomial SVM VJP: model, shapes, or values are invalid"); return
        end if
        valid = .true.
    end function polynomial_svm_valid_vjp_shapes

    pure real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        if (value >= 0.0_dp) then; probability = 1.0_dp/(1.0_dp + exp(-value))
        else; probability = exp(value)/(1.0_dp + exp(value)); end if
    end function stable_sigmoid

end module fortml_polynomial_svm_classifier
