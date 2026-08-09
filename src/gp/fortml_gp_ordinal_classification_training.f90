module fortml_gp_ordinal_classification_training
    !! FortOpt L-BFGS-B training for the latent-Gaussian ordinal GP state.
    !!
    !! The ordinal classifier owns an exact latent `gp_regression_t`.  This
    !! adapter optimizes its packed kernel and log-noise coordinates against
    !! the analytic marginal-likelihood gradient.  The objective therefore
    !! shares the same factorization and derivative path as prediction and
    !! never estimates a gradient by finite differences.  Exact factorization
    !! and this control-plane optimizer are CPU-only; CUDA requests are typed
    !! refusals until resident ordinal kernels are available.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_gp_ordinal_classification, only: gp_ordinal_classification_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: gp_ordinal_hyperparameter_options_t
        !! Bounds and convergence controls for ordinal-GP evidence training.
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
    end type gp_ordinal_hyperparameter_options_t

    type, public :: gp_ordinal_hyperparameter_result_t
        !! Diagnostics returned by `gp_ordinal_optimize_hyperparameters`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: negative_log_marginal_likelihood = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type gp_ordinal_hyperparameter_result_t

    public :: gp_ordinal_optimize_hyperparameters

    type :: ordinal_context_t
        type(gp_ordinal_classification_t), pointer :: model => null()
    end type ordinal_context_t

contains

    subroutine gp_ordinal_optimize_hyperparameters(model, options, result, status, device)
        !! Minimize negative exact latent-Gaussian evidence with FortOpt.
        type(gp_ordinal_classification_t), target, intent(inout) :: model
        type(gp_ordinal_hyperparameter_options_t), intent(in) :: options
        type(gp_ordinal_hyperparameter_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(fortml_device_t), intent(in), optional :: device
        type(ordinal_context_t), target :: context
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        type(fortnum_status_t) :: restore_status
        real(dp), allocatable :: parameters(:), initial_parameters(:), lower(:), upper(:)
        real(dp), allocatable :: gradient(:)
        integer :: n_parameters
        !! Declared default instance keeps nvfortran compatibility (it rejects
        !! empty structure constructors such as `T()`).
        type(gp_ordinal_hyperparameter_result_t) :: result_default

        result = result_default
        if (present(device)) then
            if (.not. device%selected .or. .not. device%available) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP hyperparameter training: selected device is unavailable")
                return
            end if
            select case (device%kind)
            case (FORTML_DEVICE_CPU)
                continue
            case (FORTML_DEVICE_CUDA)
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "ordinal GP hyperparameter training: resident CUDA optimizer is not linked")
                return
            case default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP hyperparameter training: device kind is invalid")
                return
            end select
        end if
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter training: options are invalid")
            return
        end if
        if (.not. model%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter training: model is not fitted")
            return
        end if
        n_parameters = model%hyperparameter_count()
        if (n_parameters < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter training: model has no parameters")
            return
        end if

        initial_parameters = model%hyperparameters()
        allocate(parameters(n_parameters), lower(n_parameters), upper(n_parameters), &
            gradient(n_parameters))
        parameters = initial_parameters
        lower = options%lower_bound
        upper = options%upper_bound
        context%model => model
        call objective%initialize_context(n_parameters, context, &
            ordinal_negative_lml_objective, status)
        if (status%code /= FORTNUM_OK) return
        call copy_lbfgsb_options(options, optimizer_options)
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (status%code /= FORTNUM_OK .and. status%code /= FORTNUM_CONVERGENCE_ERROR) then
            call model%set_hyperparameters(initial_parameters, restore_status)
            return
        end if
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%converged = optimizer_result%state%converged

        call ordinal_negative_lml_objective(context, parameters, &
            result%negative_log_marginal_likelihood, gradient, status)
        if (status%code /= FORTNUM_OK) then
            call model%set_hyperparameters(initial_parameters, restore_status)
            return
        end if
        result%gradient_norm = sqrt(sum(gradient*gradient))
        if (.not. ieee_is_finite(result%negative_log_marginal_likelihood) .or. &
                .not. ieee_is_finite(result%gradient_norm)) then
            call model%set_hyperparameters(initial_parameters, restore_status)
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP hyperparameter training: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call model%set_hyperparameters(initial_parameters, restore_status)
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP hyperparameter training: iteration limit reached")
            return
        end if
        call model%set_hyperparameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_optimize_hyperparameters

    subroutine ordinal_negative_lml_objective(context_any, parameters, value, gradient, status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: likelihood

        value = huge(1.0_dp)
        gradient = 0.0_dp
        select type (context => context_any)
            type is (ordinal_context_t)
            if (.not. associated(context%model) .or. size(parameters) /= &
                    context%model%hyperparameter_count() .or. size(gradient) /= size(parameters)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP hyperparameter objective: parameter shape is invalid")
                return
            end if
            call context%model%set_hyperparameters(parameters, status)
            if (status%code /= FORTNUM_OK) return
            call context%model%log_marginal_likelihood(likelihood, status)
            if (status%code /= FORTNUM_OK) return
            call context%model%hyperparameter_gradient(gradient, status)
            if (status%code /= FORTNUM_OK) return
            value = -likelihood
            gradient = -gradient
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP hyperparameter objective: value is not finite")
                return
            end if
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter objective: context has the wrong type")
        end select
    end subroutine ordinal_negative_lml_objective

    subroutine copy_lbfgsb_options(options, output)
        type(gp_ordinal_hyperparameter_options_t), intent(in) :: options
        type(lbfgsb_options_t), intent(out) :: output

        output%memory = options%memory
        output%max_iterations = options%max_iterations
        output%max_line_search = options%max_line_search
        output%gradient_tolerance = options%gradient_tolerance
        output%step_tolerance = options%step_tolerance
        output%objective_tolerance = options%objective_tolerance
    end subroutine copy_lbfgsb_options

    logical function valid_options(options) result(valid)
        type(gp_ordinal_hyperparameter_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. &
            ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. &
            ieee_is_finite(options%lower_bound) .and. &
            ieee_is_finite(options%upper_bound) .and. &
            options%gradient_tolerance >= 0.0_dp .and. &
            options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp .and. &
            options%lower_bound <= options%upper_bound
    end function valid_options

end module fortml_gp_ordinal_classification_training
