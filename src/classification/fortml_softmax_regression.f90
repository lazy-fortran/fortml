module fortml_softmax_regression
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    use fortml_losses, only: softmax_value, softmax_jvp, softmax_vjp
    implicit none
    private

    type, public :: softmax_regression_t
        private
        real(dp), allocatable :: coefficient(:, :)
        real(dp), allocatable :: intercept(:)
        integer, allocatable :: class_label(:)
        real(dp) :: l2 = 1.0_dp
        logical :: fit_intercept = .true.
    contains
        procedure, public :: fit => softmax_fit
        procedure, public :: decision_function => softmax_decision_function
        procedure, public :: jvp => softmax_decision_jvp
        procedure, public :: vjp => softmax_decision_vjp
        procedure, public :: decision_function_jvp => softmax_decision_jvp
        procedure, public :: decision_function_vjp => softmax_decision_vjp
        procedure, public :: predict_proba => softmax_predict_proba
        procedure, public :: predict_proba_jvp => softmax_predict_proba_jvp
        procedure, public :: predict_proba_vjp => softmax_predict_proba_vjp
        procedure, public :: predict => softmax_predict
        procedure, public :: coefficients => softmax_coefficients
        procedure, public :: intercept_values => softmax_intercepts
        procedure, public :: classes => softmax_classes
        procedure, public :: feature_count => softmax_feature_count
        procedure, public :: class_count => softmax_class_count
        procedure, public :: parameter_count => softmax_parameter_count
        procedure, public :: parameters => softmax_parameters
        procedure, public :: set_parameters => softmax_set_parameters
        procedure, public :: fitted => softmax_fitted
    end type softmax_regression_t

contains

    subroutine softmax_fit(self, x, labels, status, l2, fit_intercept, &
            max_iterations, tolerance, sample_weight, class_weight)
        class(softmax_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:)
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: options
        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: theta(:), lower(:), upper(:), encoded(:)
        real(dp), allocatable :: weights(:)
        real(dp), allocatable :: class_factors(:)
        real(dp) :: penalty, requested_tolerance, weight_sum
        integer :: iterations, n_features, n_classes, n_parameters
        integer :: i, j, position
        logical :: include_intercept

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax fit: input dimensions are invalid")
            return
        end if
        allocate(weights(size(labels)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "softmax fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax fit: inputs must be finite")
            return
        end if
        penalty = 1.0_dp
        if (present(l2)) penalty = l2
        if (.not. ieee_is_finite(penalty) .or. penalty < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax fit: L2 penalty must be finite and nonnegative")
            return
        end if
        include_intercept = .true.
        if (present(fit_intercept)) include_intercept = fit_intercept
        iterations = 200
        if (present(max_iterations)) iterations = max_iterations
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        if (iterations < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax fit: invalid optimizer controls")
            return
        end if
        if (requested_tolerance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax fit: invalid optimizer controls")
            return
        end if
        if (.not. ieee_is_finite(requested_tolerance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax fit: invalid optimizer controls")
            return
        end if

        call sorted_classes(labels, self%class_label)
        n_features = size(x, 2)
        n_classes = size(self%class_label)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax fit: at least two distinct classes are required")
            return
        end if
        allocate(class_factors(n_classes))
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "softmax fit: class weights must match sorted classes")
                return
            end if
            if (any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "softmax fit: class weights must be finite and positive")
                return
            end if
            class_factors = class_weight
        end if
        do i = 1, size(labels)
            do j = 1, n_classes
                if (labels(i) == self%class_label(j)) then
                    weights(i) = weights(i)*class_factors(j)
                    exit
                end if
            end do
        end do
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax fit: effective weights must have positive mass")
            return
        end if
        n_parameters = n_features*n_classes
        if (include_intercept) n_parameters = n_parameters + n_classes
        allocate(theta(n_parameters), lower(n_parameters), upper(n_parameters))
        allocate(encoded(size(labels)))
        theta = 0.0_dp
        lower = -huge(1.0_dp)
        upper = huge(1.0_dp)
        do i = 1, size(labels)
            do j = 1, n_classes
                if (labels(i) == self%class_label(j)) then
                    encoded(i) = real(j, dp)
                    exit
                end if
            end do
        end do

        call objective%initialize(n_parameters, softmax_objective, status)
        if (status%code /= FORTNUM_OK) return
        options%max_iterations = iterations
        options%gradient_tolerance = requested_tolerance
        options%step_tolerance = min(1.0e-12_dp, requested_tolerance)
        options%objective_tolerance = min(1.0e-12_dp, requested_tolerance)
        call optimizer%minimize(objective, theta, lower, upper, options, result, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. result%state%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "softmax fit: optimizer reached its iteration limit")
            return
        end if

        allocate(self%coefficient(n_features, n_classes))
        position = 1
        self%coefficient = reshape(theta(position:position + n_features*n_classes - 1), &
            shape(self%coefficient))
        position = position + n_features*n_classes
        allocate(self%intercept(n_classes))
        self%intercept = 0.0_dp
        if (include_intercept) self%intercept = theta(position:position + n_classes - 1)
        self%l2 = penalty
        self%fit_intercept = include_intercept
        call status_set(status, FORTNUM_OK, "")

    contains

        subroutine softmax_objective(parameters, value, gradient, objective_status)
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: value
            real(dp), intent(out) :: gradient(:)
            type(fortnum_status_t), intent(out) :: objective_status
            real(dp) :: logits(size(x, 1), n_classes)
            real(dp) :: logits_bar(size(x, 1), n_classes)
            real(dp) :: maximum, normalizer
            integer :: row, column, offset

            offset = n_features*n_classes
            do row = 1, size(x, 1)
                do column = 1, n_classes
                    logits(row, column) = dot_product(x(row, :), &
                        parameters((column - 1)*n_features + 1:column*n_features))
                    if (include_intercept) logits(row, column) = &
                        logits(row, column) + parameters(offset + column)
                end do
            end do
            value = 0.0_dp
            logits_bar = 0.0_dp
            do row = 1, size(x, 1)
                maximum = maxval(logits(row, :))
                normalizer = 0.0_dp
                do column = 1, n_classes
                    logits_bar(row, column) = exp(logits(row, column) - maximum)
                    normalizer = normalizer + logits_bar(row, column)
                end do
                value = value + weights(row)*(maximum - &
                    logits(row, int(encoded(row))) + log(normalizer))
                logits_bar(row, :) = weights(row)*logits_bar(row, :)/normalizer
                logits_bar(row, int(encoded(row))) = &
                    logits_bar(row, int(encoded(row))) - weights(row)
            end do
            value = value/weight_sum
            logits_bar = logits_bar/weight_sum
            gradient = 0.0_dp
            do column = 1, n_classes
                do row = 1, n_features
                    gradient((column - 1)*n_features + row) = &
                        dot_product(x(:, row), logits_bar(:, column)) + &
                        penalty*parameters((column - 1)*n_features + row)
                end do
                if (include_intercept) gradient(offset + column) = sum(logits_bar(:, column))
            end do
            value = value + 0.5_dp*penalty*sum(parameters(:offset)**2)
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(objective_status, FORTNUM_DOMAIN_ERROR, &
                    "softmax objective: result is not finite")
                return
            end if
            call status_set(objective_status, FORTNUM_OK, "")
        end subroutine softmax_objective

    end subroutine softmax_fit

    subroutine softmax_decision_function(self, x, scores, status)
        class(softmax_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. softmax_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision_function: model is not fitted")
            return
        end if
        if (size(x, 2) /= size(self%coefficient, 1) .or. &
            any(shape(scores) /= [size(x, 1), size(self%coefficient, 2)])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision_function: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision_function: inputs must be finite")
            return
        end if
        scores = matmul(x, self%coefficient) + spread(self%intercept, dim=1, ncopies=size(x, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_decision_function

    subroutine softmax_decision_jvp(self, x, theta_dot, x_dot, scores, &
            scores_dot, status)
        class(softmax_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:, :), scores_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coefficient_dot(:, :)
        integer :: n_features, n_classes, offset

        if (.not. softmax_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision JVP: model is not fitted")
            return
        end if
        n_features = size(self%coefficient, 1)
        n_classes = size(self%coefficient, 2)
        if (size(x, 2) /= n_features .or. size(x_dot, 1) /= size(x, 1) .or. &
            size(x_dot, 2) /= size(x, 2) .or. any(shape(scores) /= &
            [size(x, 1), n_classes]) .or. any(shape(scores_dot) /= shape(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision JVP: parameter tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision JVP: tangents must be finite")
            return
        end if
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        offset = n_features*n_classes
        allocate(coefficient_dot(n_features, n_classes))
        coefficient_dot = reshape(theta_dot(:offset), shape(coefficient_dot))
        scores_dot = matmul(x_dot, self%coefficient) + matmul(x, coefficient_dot)
        if (self%fit_intercept) then
            scores_dot = scores_dot + spread(theta_dot(offset + 1:offset + n_classes), &
                dim=1, ncopies=size(x, 1))
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_decision_jvp

    subroutine softmax_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(softmax_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coefficient_bar(:, :)
        integer :: n_features, n_classes, offset

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. softmax_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision VJP: model is not fitted")
            return
        end if
        n_features = size(self%coefficient, 1)
        n_classes = size(self%coefficient, 2)
        if (size(x, 2) /= n_features .or. size(x_bar, 1) /= size(x, 1) .or. &
            size(x_bar, 2) /= size(x, 2) .or. any(shape(scores_bar) /= &
            [size(x, 1), n_classes]) .or. size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax decision VJP: inputs and cotangents must be finite")
            return
        end if
        offset = n_features*n_classes
        allocate(coefficient_bar(n_features, n_classes))
        coefficient_bar = matmul(transpose(x), scores_bar)
        theta_bar(:offset) = reshape(coefficient_bar, [offset])
        if (self%fit_intercept) theta_bar(offset + 1:offset + n_classes) = &
            sum(scores_bar, dim=1)
        x_bar = matmul(scores_bar, transpose(self%coefficient))
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_decision_vjp

    subroutine softmax_predict_proba(self, x, probabilities, status)
        class(softmax_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :)

        allocate(scores(size(x, 1), self%class_count()))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        call softmax_value(scores, probabilities, status)
    end subroutine softmax_predict_proba

    subroutine softmax_predict_proba_jvp(self, x, theta_dot, x_dot, probabilities, &
            probabilities_dot, status)
        class(softmax_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), scores_dot(:, :)
        integer :: n_classes

        if (.not. softmax_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax probability JVP: model is not fitted")
            return
        end if
        n_classes = size(self%coefficient, 2)
        if (size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= n_classes .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax probability JVP: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), n_classes), scores_dot(size(x, 1), n_classes))
        call self%decision_function_jvp(x, theta_dot, x_dot, scores, scores_dot, status)
        if (status%code /= FORTNUM_OK) return
        call softmax_jvp(scores, scores_dot, probabilities, probabilities_dot, status)
    end subroutine softmax_predict_proba_jvp

    subroutine softmax_predict_proba_vjp(self, x, probabilities_bar, theta_bar, &
            x_bar, status)
        class(softmax_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :), logits_bar(:, :)
        integer :: n_classes

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. softmax_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax probability VJP: model is not fitted")
            return
        end if
        n_classes = size(self%coefficient, 2)
        if (size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= n_classes) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax probability VJP: cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax probability VJP: cotangent must be finite")
            return
        end if
        allocate(scores(size(x, 1), n_classes), logits_bar(size(x, 1), n_classes))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        call softmax_vjp(scores, probabilities_bar, logits_bar, status)
        if (status%code /= FORTNUM_OK) return
        call self%decision_function_vjp(x, logits_bar, theta_bar, x_bar, status)
    end subroutine softmax_predict_proba_vjp

    subroutine softmax_predict(self, x, labels, status)
        class(softmax_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, class_index

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%class_count()))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            class_index = maxloc(probabilities(i, :), dim=1)
            labels(i) = self%class_label(class_index)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_predict

    function softmax_coefficients(self) result(values)
        class(softmax_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%coefficient)) then
            values = self%coefficient
        else
            allocate(values(0, 0))
        end if
    end function softmax_coefficients

    integer function softmax_parameter_count(self) result(count)
        class(softmax_regression_t), intent(in) :: self

        count = 0
        if (.not. allocated(self%coefficient)) return
        count = size(self%coefficient)
        if (self%fit_intercept) count = count + size(self%intercept)
    end function softmax_parameter_count

    function softmax_parameters(self) result(values)
        class(softmax_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: offset

        if (.not. allocated(self%coefficient)) then
            allocate(values(0))
            return
        end if
        offset = size(self%coefficient)
        allocate(values(self%parameter_count()))
        values(:offset) = reshape(self%coefficient, [offset])
        if (self%fit_intercept) values(offset + 1:) = self%intercept
    end function softmax_parameters

    subroutine softmax_set_parameters(self, values, status)
        class(softmax_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: offset

        if (.not. softmax_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax set_parameters: model is not fitted")
            return
        end if
        if (size(values) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax set_parameters: parameter shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax set_parameters: parameters must be finite")
            return
        end if
        offset = size(self%coefficient)
        self%coefficient = reshape(values(:offset), shape(self%coefficient))
        if (self%fit_intercept) self%intercept = values(offset + 1:)
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_set_parameters

    function softmax_intercepts(self) result(values)
        class(softmax_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%intercept)) then
            values = self%intercept
        else
            allocate(values(0))
        end if
    end function softmax_intercepts

    function softmax_classes(self) result(labels)
        class(softmax_regression_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function softmax_classes

    integer function softmax_feature_count(self) result(count)
        class(softmax_regression_t), intent(in) :: self

        count = 0
        if (allocated(self%coefficient)) count = size(self%coefficient, 1)
    end function softmax_feature_count

    integer function softmax_class_count(self) result(count)
        class(softmax_regression_t), intent(in) :: self

        count = 0
        if (allocated(self%class_label)) count = size(self%class_label)
    end function softmax_class_count

    logical function softmax_fitted(self) result(is_fitted)
        class(softmax_regression_t), intent(in) :: self

        is_fitted = allocated(self%coefficient) .and. allocated(self%intercept) .and. &
            allocated(self%class_label)
    end function softmax_fitted

    subroutine sorted_classes(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, n
        integer :: temporary

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

end module fortml_softmax_regression
