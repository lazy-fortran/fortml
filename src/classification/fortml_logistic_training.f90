module fortml_logistic_training
    !! Exact objective products and FortOpt training for binary logistic models.
    !!
    !! The adapter keeps the packed coefficient/intercept vector as the
    !! optimizer variable and optionally appends the non-negative L2
    !! coefficient.  Value, gradient, and Hessian-vector products are
    !! evaluated analytically from the weighted logistic objective.  The
    !! optional L2 coordinate is a joint objective coordinate, not a
    !! finite-difference approximation to an optimizer trajectory.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortml_logistic_regression, only: logistic_regression_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: logistic_training_objective_t
        !! Weighted binary logistic value/gradient/HVP adapter.
        !!
        !! The packed variable is the fitted model's parameters.  With
        !! `optimize_l2=.true.`, one final scalar coordinate is appended and
        !! receives the exact derivative and HVP of the L2 coefficient.
        private
        type(logistic_regression_t), pointer :: model => null()
        real(dp), allocatable :: features(:, :), targets(:), weights(:)
        real(dp) :: weight_sum = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        logical :: optimize_l2 = .false.
    contains
        procedure, public :: initialize => logistic_objective_initialize
        procedure, public :: parameter_count => logistic_objective_parameter_count
        procedure, public :: parameters => logistic_objective_parameters
        procedure, public :: value_gradient => logistic_objective_value_gradient
        procedure, public :: hvp => logistic_objective_hvp
        procedure, public :: fortopt => logistic_objective_fortopt
    end type logistic_training_objective_t

    type, public :: logistic_lbfgsb_options_t
        !! Bounds and convergence controls for logistic L-BFGS-B.
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
    end type logistic_lbfgsb_options_t

    type, public :: logistic_lbfgsb_result_t
        !! Diagnostics returned by `logistic_optimize_lbfgsb`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: l2 = 0.0_dp
    end type logistic_lbfgsb_result_t

    public :: logistic_optimize_lbfgsb

contains

    subroutine logistic_objective_initialize(self, model, x, labels, l2, status, &
            optimize_l2, sample_weight, class_weight)
        class(logistic_training_objective_t), intent(out) :: self
        type(logistic_regression_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), l2
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: optimize_l2
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        integer, allocatable :: classes(:)
        real(dp) :: class_factors(2)
        integer :: i

        self%l2 = 0.0_dp
        self%optimize_l2 = .false.
        if (present(optimize_l2)) self%optimize_l2 = optimize_l2
        if (.not. model%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= model%feature_count() .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: model, input, or label shape is invalid")
            return
        end if
        if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: L2 coefficient is invalid")
            return
        end if
        classes = model%classes()
        if (size(classes) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: fitted model must have two classes")
            return
        end if
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= 2 .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic objective: class weights are invalid")
                return
            end if
            class_factors = class_weight
        end if

        self%model => model
        allocate(self%features, source=x)
        allocate(self%targets(size(labels)), self%weights(size(labels)))
        self%targets = 0.0_dp
        self%weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic objective: sample weights are invalid")
                return
            end if
            self%weights = sample_weight
        end if
        do i = 1, size(labels)
            if (labels(i) == classes(1)) then
                self%weights(i) = self%weights(i)*class_factors(1)
            else if (labels(i) == classes(2)) then
                self%targets(i) = 1.0_dp
                self%weights(i) = self%weights(i)*class_factors(2)
            else
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic objective: labels do not match fitted classes")
                return
            end if
        end do
        self%weight_sum = sum(self%weights)
        if (.not. ieee_is_finite(self%weight_sum) .or. self%weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: effective weights have no positive mass")
            return
        end if
        self%l2 = l2
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_objective_initialize

    integer function logistic_objective_parameter_count(self) result(count)
        class(logistic_training_objective_t), intent(in) :: self

        count = 0
        if (.not. associated(self%model)) return
        count = self%model%parameter_count()
        if (self%optimize_l2) count = count + 1
    end function logistic_objective_parameter_count

    function logistic_objective_parameters(self) result(parameters)
        class(logistic_training_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: n_model

        allocate(parameters(self%parameter_count()))
        parameters = 0.0_dp
        if (.not. associated(self%model)) return
        n_model = self%model%parameter_count()
        parameters(:n_model) = self%model%parameters()
        if (self%optimize_l2) parameters(n_model + 1) = self%l2
    end function logistic_objective_parameters

    subroutine logistic_objective_value_gradient(self, parameters, value, gradient, &
            status)
        class(logistic_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: l2, probability, residual, score
        integer :: i, j, n_features, n_model

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: parameter or gradient shape is invalid")
            return
        end if
        n_features = self%model%feature_count()
        l2 = self%l2
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic objective: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        call self%model%set_regularization(l2, status)
        if (status%code /= FORTNUM_OK) return

        do i = 1, size(self%features, 1)
            score = 0.0_dp
            do j = 1, n_features
                score = score + self%features(i, j)*parameters(j)
            end do
            if (n_model > n_features) score = score + parameters(n_model)
            probability = stable_sigmoid(score)
            residual = probability - self%targets(i)
            if (self%targets(i) > 0.5_dp) then
                value = value + self%weights(i)*stable_softplus(-score)
            else
                value = value + self%weights(i)*stable_softplus(score)
            end if
            do j = 1, n_features
                gradient(j) = gradient(j) + self%weights(i)*residual* &
                    self%features(i, j)
            end do
            if (n_model > n_features) gradient(n_model) = gradient(n_model) + &
                self%weights(i)*residual
        end do
        value = value/self%weight_sum
        gradient = gradient/self%weight_sum
        value = value + 0.5_dp*l2*sum(parameters(:n_features)**2)
        gradient(:n_features) = gradient(:n_features) + &
            l2*parameters(:n_features)
        if (self%optimize_l2) gradient(n_model + 1) = &
            0.5_dp*sum(parameters(:n_features)**2)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_objective_value_gradient

    subroutine logistic_objective_hvp(self, parameters, direction, product, status)
        !! Exact Hessian-vector product for the joint `(theta,l2)` block.
        class(logistic_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: l2, l2_direction, probability, curvature
        real(dp) :: score, score_direction
        integer :: i, j, n_features, n_model

        product = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective HVP: adapter is not initialized")
            return
        end if
        n_model = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. &
            size(direction) /= size(parameters) .or. &
            size(product) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective HVP: parameter or direction shape is invalid")
            return
        end if
        n_features = self%model%feature_count()
        l2 = self%l2
        l2_direction = 0.0_dp
        if (self%optimize_l2) then
            l2 = parameters(n_model + 1)
            l2_direction = direction(n_model + 1)
            if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "logistic objective HVP: optimized L2 coefficient is invalid")
                return
            end if
        end if
        call self%model%set_parameters(parameters(:n_model), status)
        if (status%code /= FORTNUM_OK) return
        call self%model%set_regularization(l2, status)
        if (status%code /= FORTNUM_OK) return

        do i = 1, size(self%features, 1)
            score = 0.0_dp
            score_direction = 0.0_dp
            do j = 1, n_features
                score = score + self%features(i, j)*parameters(j)
                score_direction = score_direction + self%features(i, j)*direction(j)
            end do
            if (n_model > n_features) then
                score = score + parameters(n_model)
                score_direction = score_direction + direction(n_model)
            end if
            probability = stable_sigmoid(score)
            curvature = self%weights(i)/self%weight_sum*probability* &
                (1.0_dp - probability)*score_direction
            do j = 1, n_features
                product(j) = product(j) + curvature*self%features(i, j)
            end do
            if (n_model > n_features) product(n_model) = product(n_model) + curvature
        end do
        product(:n_features) = product(:n_features) + l2*direction(:n_features) + &
            l2_direction*parameters(:n_features)
        if (self%optimize_l2) product(n_model + 1) = &
            dot_product(parameters(:n_features), direction(:n_features))
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_objective_hvp

    subroutine logistic_objective_fortopt(self, objective, status)
        class(logistic_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            logistic_objective_context_callback, status)
    end subroutine logistic_objective_fortopt

    subroutine logistic_objective_context_callback(context, parameters, value, &
            gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (logistic_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic objective: context has the wrong type")
        end select
    end subroutine logistic_objective_context_callback

    subroutine logistic_optimize_lbfgsb(model, x, labels, options, result, status, &
            sample_weight, class_weight)
        !! Optimize a fitted binary logistic objective with FortOpt L-BFGS-B.
        !!
        !! The model parameter block is always optimized.  With
        !! `options%optimize_l2`, a bounded L2 coordinate is appended and
        !! receives analytic value/gradient/HVP products from the objective.
        type(logistic_regression_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(logistic_lbfgsb_options_t), intent(in) :: options
        type(logistic_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:)
        type(logistic_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_model, n_parameters
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(logistic_lbfgsb_result_t) :: logistic_lbfgsb_result_t_default

        result = logistic_lbfgsb_result_t_default
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "logistic L-BFGS-B: options are invalid")
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
            if (status%code /= FORTNUM_OK) return
        else
            call model%set_regularization(options%l2, status)
            if (status%code /= FORTNUM_OK) return
        end if
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
                "logistic L-BFGS-B: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "logistic L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine logistic_optimize_lbfgsb

    logical function valid_options(options) result(valid)
        type(logistic_lbfgsb_options_t), intent(in) :: options

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

    pure real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        real(dp) :: exponential

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            exponential = exp(value)
            probability = exponential/(1.0_dp + exponential)
        end if
    end function stable_sigmoid

    pure real(dp) function stable_softplus(value) result(softplus)
        real(dp), intent(in) :: value

        softplus = max(value, 0.0_dp) + log(1.0_dp + exp(-abs(value)))
    end function stable_softplus

end module fortml_logistic_training
