module fortml_softmax_training
    !! Exact weighted softmax objective products and FortOpt training.
    !!
    !! `softmax_training_objective_t` keeps a fitted multinomial model and a
    !! copy of its labelled data.  The packed optimizer variable contains the
    !! model coefficients/intercepts and may append a non-negative L2
    !! coefficient.  Value, gradient, and Hessian-vector products are
    !! evaluated analytically; no finite differences are used by the adapter.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortml_softmax_regression, only: softmax_regression_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: softmax_training_objective_t
        !! Weighted multinomial cross-entropy value/gradient/HVP adapter.
        !!
        !! Coefficients are packed column-major followed by intercepts.  With
        !! `optimize_l2=.true.`, the final coordinate is the L2 coefficient
        !! and its derivative is the exact squared coefficient norm block.
        private
        type(softmax_regression_t), pointer :: model => null()
        real(dp), allocatable :: features(:, :), weights(:)
        integer, allocatable :: targets(:)
        real(dp) :: weight_sum = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        logical :: optimize_l2 = .false.
    contains
        procedure, public :: initialize => softmax_objective_initialize
        procedure, public :: parameter_count => softmax_objective_parameter_count
        procedure, public :: parameters => softmax_objective_parameters
        procedure, public :: value_gradient => softmax_objective_value_gradient
        procedure, public :: hvp => softmax_objective_hvp
        procedure, public :: fortopt => softmax_objective_fortopt
    end type softmax_training_objective_t

    type, public :: softmax_lbfgsb_options_t
        !! Bounds and convergence controls for softmax L-BFGS-B.
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
    end type softmax_lbfgsb_options_t

    type, public :: softmax_lbfgsb_result_t
        !! Diagnostics returned by `softmax_optimize_lbfgsb`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: l2 = 0.0_dp
    end type softmax_lbfgsb_result_t

    public :: softmax_optimize_lbfgsb

contains

    subroutine softmax_objective_initialize(self, model, x, labels, l2, status, &
            optimize_l2, sample_weight, class_weight)
        class(softmax_training_objective_t), intent(out) :: self
        type(softmax_regression_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), l2
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: optimize_l2
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        integer, allocatable :: classes(:)
        real(dp), allocatable :: class_factors(:)
        integer :: i, j, n_classes

        self%l2 = 0.0_dp
        self%optimize_l2 = .false.
        if (present(optimize_l2)) self%optimize_l2 = optimize_l2
        if (.not. model%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= model%feature_count() .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: model, input, or label shape is invalid")
            return
        end if
        if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: L2 coefficient is invalid")
            return
        end if

        classes = model%classes()
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: fitted model must have at least two classes")
            return
        end if
        allocate(class_factors(n_classes))
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "softmax objective: class weights are invalid")
                return
            end if
            class_factors = class_weight
        end if

        self%model => model
        allocate(self%features, source=x)
        allocate(self%targets(size(labels)), self%weights(size(labels)))
        self%targets = 0
        self%weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "softmax objective: sample weights are invalid")
                return
            end if
            self%weights = sample_weight
        end if
        do i = 1, size(labels)
            do j = 1, n_classes
                if (labels(i) == classes(j)) then
                    self%targets(i) = j
                    self%weights(i) = self%weights(i)*class_factors(j)
                    exit
                end if
            end do
            if (self%targets(i) == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "softmax objective: labels do not match fitted classes")
                return
            end if
        end do
        self%weight_sum = sum(self%weights)
        if (.not. ieee_is_finite(self%weight_sum) .or. self%weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: effective weights have no positive mass")
            return
        end if
        self%l2 = l2
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_objective_initialize

    integer function softmax_objective_parameter_count(self) result(count)
        class(softmax_training_objective_t), intent(in) :: self

        count = 0
        if (.not. associated(self%model)) return
        count = self%model%parameter_count()
        if (self%optimize_l2) count = count + 1
    end function softmax_objective_parameter_count

    function softmax_objective_parameters(self) result(parameters)
        class(softmax_training_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: n_model

        allocate(parameters(self%parameter_count()))
        parameters = 0.0_dp
        if (.not. associated(self%model)) return
        n_model = self%model%parameter_count()
        parameters(:n_model) = self%model%parameters()
        if (self%optimize_l2) parameters(n_model + 1) = self%l2
    end function softmax_objective_parameters

    subroutine softmax_objective_value_gradient(self, parameters, value, gradient, &
            status)
        class(softmax_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: logits(:, :), probabilities(:, :)
        real(dp) :: l2, maximum, normalizer, residual
        integer :: i, j, k, n_features, n_classes, n_model, offset

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: parameter or gradient shape is invalid")
            return
        end if
        n_features = self%model%feature_count()
        n_classes = self%model%class_count()
        offset = n_features*n_classes
        l2 = self%l2
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "softmax objective: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        call self%model%set_regularization(l2, status)
        if (status%code /= FORTNUM_OK) return

        allocate(logits(size(self%features, 1), n_classes), &
            probabilities(size(self%features, 1), n_classes))
        do i = 1, size(self%features, 1)
            do j = 1, n_classes
                logits(i, j) = dot_product(self%features(i, :), &
                    parameters((j - 1)*n_features + 1:j*n_features))
                if (n_model > offset) logits(i, j) = logits(i, j) + &
                    parameters(offset + j)
            end do
            maximum = maxval(logits(i, :))
            normalizer = 0.0_dp
            do j = 1, n_classes
                probabilities(i, j) = exp(logits(i, j) - maximum)
                normalizer = normalizer + probabilities(i, j)
            end do
            probabilities(i, :) = probabilities(i, :)/normalizer
            value = value + self%weights(i)*(maximum - &
                logits(i, self%targets(i)) + log(normalizer))
            do j = 1, n_classes
                residual = self%weights(i)/self%weight_sum * &
                    (probabilities(i, j) - merge(1.0_dp, 0.0_dp, &
                    j == self%targets(i)))
                do k = 1, n_features
                    gradient((j - 1)*n_features + k) = &
                        gradient((j - 1)*n_features + k) + residual*self%features(i, k)
                end do
                if (n_model > offset) gradient(offset + j) = &
                    gradient(offset + j) + residual
            end do
        end do
        value = value/self%weight_sum + 0.5_dp*l2*sum( &
            parameters(:offset)**2)
        gradient(:offset) = gradient(:offset) + l2*parameters(:offset)
        if (self%optimize_l2) gradient(n_model + 1) = 0.5_dp*sum( &
            parameters(:offset)**2)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_objective_value_gradient

    subroutine softmax_objective_hvp(self, parameters, direction, product, status)
        !! Exact Hessian-vector product for the joint `(theta,l2)` block.
        class(softmax_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: logits(:, :), probabilities(:, :), logits_dot(:)
        real(dp) :: l2, l2_direction, maximum, normalizer, mean_dot, residual_dot
        integer :: i, j, k, n_features, n_classes, n_model, offset

        product = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective HVP: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
            size(direction) /= size(parameters) .or. size(product) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective HVP: parameter or direction shape is invalid")
            return
        end if
        n_features = self%model%feature_count()
        n_classes = self%model%class_count()
        offset = n_features*n_classes
        l2 = self%l2
        l2_direction = 0.0_dp
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            l2_direction = direction(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "softmax objective HVP: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        call self%model%set_regularization(l2, status)
        if (status%code /= FORTNUM_OK) return

        allocate(logits(size(self%features, 1), n_classes), &
            probabilities(size(self%features, 1), n_classes), &
            logits_dot(n_classes))
        do i = 1, size(self%features, 1)
            do j = 1, n_classes
                logits(i, j) = dot_product(self%features(i, :), &
                    parameters((j - 1)*n_features + 1:j*n_features))
                logits_dot(j) = dot_product(self%features(i, :), &
                    direction((j - 1)*n_features + 1:j*n_features))
                if (n_model > offset) then
                    logits(i, j) = logits(i, j) + parameters(offset + j)
                    logits_dot(j) = logits_dot(j) + direction(offset + j)
                end if
            end do
            maximum = maxval(logits(i, :))
            normalizer = 0.0_dp
            do j = 1, n_classes
                probabilities(i, j) = exp(logits(i, j) - maximum)
                normalizer = normalizer + probabilities(i, j)
            end do
            probabilities(i, :) = probabilities(i, :)/normalizer
            mean_dot = dot_product(probabilities(i, :), logits_dot)
            do j = 1, n_classes
                residual_dot = self%weights(i)/self%weight_sum * &
                    probabilities(i, j)*(logits_dot(j) - mean_dot)
                do k = 1, n_features
                    product((j - 1)*n_features + k) = &
                        product((j - 1)*n_features + k) + &
                        residual_dot*self%features(i, k)
                end do
                if (n_model > offset) product(offset + j) = &
                    product(offset + j) + residual_dot
            end do
        end do
        product(:offset) = product(:offset) + l2*direction(:offset) + &
            l2_direction*parameters(:offset)
        if (self%optimize_l2) product(n_model + 1) = dot_product( &
            parameters(:offset), direction(:offset))
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_objective_hvp

    subroutine softmax_objective_fortopt(self, objective, status)
        class(softmax_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            softmax_objective_context_callback, status)
    end subroutine softmax_objective_fortopt

    subroutine softmax_objective_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (softmax_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax objective: context has the wrong type")
        end select
    end subroutine softmax_objective_context_callback

    subroutine softmax_optimize_lbfgsb(model, x, labels, options, result, status, &
            sample_weight, class_weight)
        !! Optimize a fitted multinomial softmax objective with L-BFGS-B.
        type(softmax_regression_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(softmax_lbfgsb_options_t), intent(in) :: options
        type(softmax_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        type(softmax_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_model, n_parameters

        result = softmax_lbfgsb_result_t()
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax L-BFGS-B: options are invalid")
            return
        end if
        call adapter%initialize(model, x, labels, options%l2, status, &
            optimize_l2=options%optimize_l2, sample_weight=sample_weight, &
            class_weight=class_weight)
        if (status%code /= FORTNUM_OK) return
        n_model = model%parameter_count()
        n_parameters = adapter%parameter_count()
        parameters = adapter%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower(:n_model) = options%lower_bound
        upper(:n_model) = options%upper_bound
        if (options%optimize_l2) then
            lower(n_model + 1) = options%l2_lower_bound
            upper(n_model + 1) = options%l2_upper_bound
        end if
        call adapter%fortopt(objective, status)
        if (status%code /= FORTNUM_OK) return
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (status%code /= FORTNUM_OK) return
        call model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        if (options%optimize_l2) then
            call model%set_regularization(parameters(n_model + 1), status)
        else
            call model%set_regularization(options%l2, status)
        end if
        if (status%code /= FORTNUM_OK) return
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (status%code /= FORTNUM_OK) return
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
                "softmax L-BFGS-B: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "softmax L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_optimize_lbfgsb

    logical function valid_options(options) result(valid)
        type(softmax_lbfgsb_options_t), intent(in) :: options

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
    end function valid_options

end module fortml_softmax_training
