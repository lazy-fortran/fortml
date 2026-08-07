module fortml_gp_classification_training
    !! FortOpt L-BFGS-B adapters for Laplace GP classification.
    !!
    !! Each objective evaluation refits the Laplace mode at the proposed
    !! kernel-log parameters and then consumes the classifier's analytic
    !! envelope gradient.  The objective is the negative converged mode
    !! log-posterior; it is deliberately not the full Laplace evidence.
    !! The multiclass adapter uses one shared kernel parameter vector for all
    !! one-vs-rest models and sums their independent gradients.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortml_kernels, only: kernel_t
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t, &
        gp_multiclass_classification_options_t, &
        gp_multiclass_classification_state_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: gp_classification_hyperparameter_options_t
        !! Bounds and convergence controls for binary Laplace GP training.
        type(gp_classification_options_t) :: fit
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
    end type gp_classification_hyperparameter_options_t

    type, public :: gp_multiclass_hyperparameter_options_t
        !! Bounds and convergence controls for shared-kernel OVR GP training.
        type(gp_multiclass_classification_options_t) :: fit
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
    end type gp_multiclass_hyperparameter_options_t

    type, public :: gp_classification_hyperparameter_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        real(dp) :: negative_log_posterior = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type gp_classification_hyperparameter_result_t

    type, public :: gp_multiclass_hyperparameter_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        real(dp) :: negative_log_posterior = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type gp_multiclass_hyperparameter_result_t

    type :: binary_context_t
        type(gp_classification_t), pointer :: model => null()
        type(kernel_t), pointer :: kernel => null()
        real(dp), allocatable :: x(:, :)
        integer, allocatable :: labels(:)
        type(gp_classification_options_t) :: fit
    end type binary_context_t

    type :: multiclass_context_t
        type(gp_multiclass_classification_t), pointer :: model => null()
        type(kernel_t), pointer :: kernel => null()
        real(dp), allocatable :: x(:, :)
        integer, allocatable :: labels(:)
        type(gp_multiclass_classification_options_t) :: fit
    end type multiclass_context_t

    public :: gp_classification_optimize_hyperparameters
    public :: gp_multiclass_optimize_hyperparameters

contains

    subroutine gp_classification_optimize_hyperparameters( &
            model, x, labels, kernel, options, result, status)
        type(gp_classification_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), target, intent(inout) :: kernel
        type(gp_classification_hyperparameter_options_t), intent(in) :: options
        type(gp_classification_hyperparameter_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(binary_context_t), target :: context
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_parameters

        result = gp_classification_hyperparameter_result_t()
        if (.not. valid_binary_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification training: options are invalid")
            return
        end if
        if (.not. valid_binary_data(x, labels, kernel)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification training: model data or kernel is invalid")
            return
        end if
        n_parameters = kernel%parameter_count()
        if (n_parameters < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification training: kernel has no parameters")
            return
        end if

        context%model => model
        context%kernel => kernel
        allocate(context%x, source=x)
        allocate(context%labels, source=labels)
        context%fit = options%fit
        parameters = kernel%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower = options%lower_bound
        upper = options%upper_bound
        call objective%initialize_context(n_parameters, context, &
            binary_objective, status)
        if (status%code /= FORTNUM_OK) return
        call copy_lbfgsb_options(options%memory, options%max_iterations, &
            options%max_line_search, options%gradient_tolerance, &
            options%step_tolerance, options%objective_tolerance, optimizer_options)
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (status%code /= FORTNUM_OK) return
        call binary_objective(context, parameters, result%negative_log_posterior, &
            gradient, status)
        if (status%code /= FORTNUM_OK) return

        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%gradient_norm = sqrt(sum(gradient*gradient))
        if (.not. ieee_is_finite(result%negative_log_posterior) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification training: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP classification training: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_classification_optimize_hyperparameters

    subroutine gp_multiclass_optimize_hyperparameters( &
            model, x, labels, kernel, options, result, status)
        type(gp_multiclass_classification_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), target, intent(inout) :: kernel
        type(gp_multiclass_hyperparameter_options_t), intent(in) :: options
        type(gp_multiclass_hyperparameter_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(multiclass_context_t), target :: context
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_parameters

        result = gp_multiclass_hyperparameter_result_t()
        if (.not. valid_multiclass_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass training: options are invalid")
            return
        end if
        if (.not. valid_multiclass_data(x, labels, kernel)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass training: model data or kernel is invalid")
            return
        end if
        n_parameters = kernel%parameter_count()
        if (n_parameters < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass training: kernel has no parameters")
            return
        end if

        context%model => model
        context%kernel => kernel
        allocate(context%x, source=x)
        allocate(context%labels, source=labels)
        context%fit = options%fit
        parameters = kernel%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower = options%lower_bound
        upper = options%upper_bound
        call objective%initialize_context(n_parameters, context, &
            multiclass_objective, status)
        if (status%code /= FORTNUM_OK) return
        call copy_lbfgsb_options(options%memory, options%max_iterations, &
            options%max_line_search, options%gradient_tolerance, &
            options%step_tolerance, options%objective_tolerance, optimizer_options)
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (status%code /= FORTNUM_OK) return
        call multiclass_objective(context, parameters, &
            result%negative_log_posterior, gradient, status)
        if (status%code /= FORTNUM_OK) return

        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%gradient_norm = sqrt(sum(gradient*gradient))
        if (.not. ieee_is_finite(result%negative_log_posterior) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass training: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP multiclass training: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_optimize_hyperparameters

    subroutine binary_objective(context_any, parameters, value, gradient, status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        type(gp_classification_state_t) :: state

        value = huge(1.0_dp)
        gradient = 0.0_dp
        select type (context => context_any)
            type is (binary_context_t)
            if (.not. associated(context%model) .or. .not. associated(context%kernel) .or. &
                size(parameters) /= context%kernel%parameter_count() .or. &
                size(gradient) /= size(parameters)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP classification objective: parameter shape is invalid")
                return
            end if
            call context%kernel%set_parameters(parameters, status)
            if (status%code /= FORTNUM_OK) return
            call context%model%fit(context%x, context%labels, context%kernel, status, &
                context%fit, state)
            if (status%code /= FORTNUM_OK) return
            call context%model%hyperparameter_gradient(gradient, status)
            if (status%code /= FORTNUM_OK) return
            value = -state%log_posterior
            gradient = -gradient
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP classification objective: value is not finite")
                return
            end if
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP classification objective: context has the wrong type")
        end select
    end subroutine binary_objective

    subroutine multiclass_objective(context_any, parameters, value, gradient, status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        type(gp_multiclass_classification_state_t) :: state
        real(dp), allocatable :: packed_gradient(:)
        integer :: block_count, class_count, first, last, i

        value = huge(1.0_dp)
        gradient = 0.0_dp
        select type (context => context_any)
            type is (multiclass_context_t)
            if (.not. associated(context%model) .or. .not. associated(context%kernel) .or. &
                size(parameters) /= context%kernel%parameter_count() .or. &
                size(gradient) /= size(parameters)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass objective: parameter shape is invalid")
                return
            end if
            call context%kernel%set_parameters(parameters, status)
            if (status%code /= FORTNUM_OK) return
            call context%model%fit(context%x, context%labels, context%kernel, status, &
                context%fit, state)
            if (status%code /= FORTNUM_OK) return
            block_count = context%kernel%parameter_count()
            class_count = context%model%class_count()
            allocate(packed_gradient(block_count*class_count))
            call context%model%hyperparameter_gradient(packed_gradient, status)
            if (status%code /= FORTNUM_OK) return
            first = 1
            gradient = 0.0_dp
            do i = 1, class_count
                last = first + block_count - 1
                gradient = gradient + packed_gradient(first:last)
                first = last + 1
            end do
            value = -state%log_posterior
            gradient = -gradient
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass objective: value is not finite")
                return
            end if
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass objective: context has the wrong type")
        end select
    end subroutine multiclass_objective

    subroutine copy_lbfgsb_options(memory, max_iterations, max_line_search, &
            gradient_tolerance, step_tolerance, objective_tolerance, output)
        integer, intent(in) :: memory, max_iterations, max_line_search
        real(dp), intent(in) :: gradient_tolerance, step_tolerance, objective_tolerance
        type(lbfgsb_options_t), intent(out) :: output

        output%memory = memory
        output%max_iterations = max_iterations
        output%max_line_search = max_line_search
        output%gradient_tolerance = gradient_tolerance
        output%step_tolerance = step_tolerance
        output%objective_tolerance = objective_tolerance
    end subroutine copy_lbfgsb_options

    logical function valid_binary_options(options) result(valid)
        type(gp_classification_hyperparameter_options_t), intent(in) :: options

        valid = valid_optimizer_options(options%memory, options%max_iterations, &
            options%max_line_search, options%gradient_tolerance, options%step_tolerance, &
            options%objective_tolerance, options%lower_bound, options%upper_bound) .and. &
            options%fit%max_iterations >= 1 .and. options%fit%tolerance > 0.0_dp .and. &
            options%fit%jitter >= 0.0_dp .and. options%fit%damping > 0.0_dp .and. &
            options%fit%damping <= 1.0_dp
    end function valid_binary_options

    logical function valid_multiclass_options(options) result(valid)
        type(gp_multiclass_hyperparameter_options_t), intent(in) :: options

        valid = valid_optimizer_options(options%memory, options%max_iterations, &
            options%max_line_search, options%gradient_tolerance, options%step_tolerance, &
            options%objective_tolerance, options%lower_bound, options%upper_bound) .and. &
            options%fit%max_iterations >= 1 .and. options%fit%tolerance > 0.0_dp .and. &
            options%fit%jitter >= 0.0_dp .and. options%fit%damping > 0.0_dp .and. &
            options%fit%damping <= 1.0_dp
    end function valid_multiclass_options

    logical function valid_optimizer_options(memory, max_iterations, max_line_search, &
            gradient_tolerance, step_tolerance, objective_tolerance, lower_bound, &
            upper_bound) result(valid)
        integer, intent(in) :: memory, max_iterations, max_line_search
        real(dp), intent(in) :: gradient_tolerance, step_tolerance, objective_tolerance
        real(dp), intent(in) :: lower_bound, upper_bound

        valid = memory >= 1 .and. max_iterations >= 1 .and. max_line_search >= 1 .and. &
            ieee_is_finite(gradient_tolerance) .and. ieee_is_finite(step_tolerance) .and. &
            ieee_is_finite(objective_tolerance) .and. ieee_is_finite(lower_bound) .and. &
            ieee_is_finite(upper_bound) .and. gradient_tolerance >= 0.0_dp .and. &
            step_tolerance >= 0.0_dp .and. objective_tolerance >= 0.0_dp .and. &
            lower_bound <= upper_bound
    end function valid_optimizer_options

    logical function valid_binary_data(x, labels, kernel) result(valid)
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), intent(in) :: kernel

        valid = size(x, 1) >= 1 .and. size(x, 2) >= 1 .and. &
            size(labels) == size(x, 1) .and. all(ieee_is_finite(x)) .and. &
            kernel%input_dim == size(x, 2) .and. kernel%parameter_count() >= 1 .and. &
            count_unique(labels) == 2
    end function valid_binary_data

    logical function valid_multiclass_data(x, labels, kernel) result(valid)
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), intent(in) :: kernel

        valid = size(x, 1) >= 1 .and. size(x, 2) >= 1 .and. &
            size(labels) == size(x, 1) .and. all(ieee_is_finite(x)) .and. &
            kernel%input_dim == size(x, 2) .and. kernel%parameter_count() >= 1 .and. &
            count_unique(labels) >= 2
    end function valid_multiclass_data

    integer function count_unique(values) result(count)
        integer, intent(in) :: values(:)
        integer :: i, j
        logical :: found

        count = 0
        do i = 1, size(values)
            found = .false.
            do j = 1, i - 1
                if (values(j) == values(i)) then
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) count = count + 1
        end do
    end function count_unique

end module fortml_gp_classification_training
