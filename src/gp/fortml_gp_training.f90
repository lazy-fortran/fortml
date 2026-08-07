module fortml_gp_training
    !! FortOpt-backed hyperparameter training for exact Gaussian processes.
    !!
    !! The model owns the covariance factorization.  Each L-BFGS-B objective
    !! evaluation updates the packed log kernel/noise parameters through the
    !! model setter, refactors the covariance, and obtains the analytic
    !! marginal-likelihood gradient.  This keeps optimization and derivative
    !! contracts on the same code path as prediction.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortml_gaussian_process, only: gp_regression_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: gp_hyperparameter_options_t
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
    end type gp_hyperparameter_options_t

    type, public :: gp_hyperparameter_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        real(dp) :: negative_log_marginal_likelihood = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type gp_hyperparameter_result_t

    public :: gp_optimize_hyperparameters

contains

    subroutine gp_optimize_hyperparameters(model, options, result, status)
        type(gp_regression_t), target, intent(inout) :: model
        type(gp_hyperparameter_options_t), intent(in) :: options
        type(gp_hyperparameter_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_parameters

        result = gp_hyperparameter_result_t()
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP hyperparameter training: options are invalid")
            return
        end if
        n_parameters = model%parameter_count()
        if (n_parameters < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP hyperparameter training: model has no parameters")
            return
        end if

        parameters = model%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower = options%lower_bound
        upper = options%upper_bound
        call objective%initialize_context(n_parameters, model, &
            gp_negative_lml_objective, status)
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
        call model%set_parameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        call gp_negative_lml_objective(model, parameters, &
            result%negative_log_marginal_likelihood, gradient, status)
        if (status%code /= FORTNUM_OK) return

        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%gradient_norm = sqrt(sum(gradient**2))
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP hyperparameter training: iteration limit reached")
            return
        end if
        if (.not. ieee_is_finite(result%negative_log_marginal_likelihood) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP hyperparameter training: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_optimize_hyperparameters

    subroutine gp_negative_lml_objective(context, parameters, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: likelihood

        value = huge(1.0_dp)
        gradient = 0.0_dp
        select type (model => context)
            type is (gp_regression_t)
            if (size(parameters) /= model%parameter_count() .or. &
                size(gradient) /= size(parameters)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP hyperparameter objective: parameter shape is invalid")
                return
            end if
            call model%set_parameters(parameters, status)
            if (status%code /= FORTNUM_OK) return
            call model%log_marginal_likelihood(likelihood, status)
            if (status%code /= FORTNUM_OK) return
            call model%hyperparameter_gradient(gradient, status)
            if (status%code /= FORTNUM_OK) return
            value = -likelihood
            gradient = -gradient
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP hyperparameter objective: value is not finite")
                return
            end if
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP hyperparameter objective: context has the wrong type")
        end select
    end subroutine gp_negative_lml_objective

    logical function valid_options(options) result(valid)
        type(gp_hyperparameter_options_t), intent(in) :: options

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

end module fortml_gp_training
