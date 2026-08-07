module fortml_gp_training
    !! FortOpt-backed hyperparameter training for exact Gaussian processes.
    !!
    !! The model owns the covariance factorization.  Each L-BFGS-B objective
    !! evaluation updates the packed log kernel/noise parameters through the
    !! model setter, refactors the covariance, and obtains the analytic
    !! marginal-likelihood gradient.  This keeps optimization and derivative
    !! contracts on the same code path as prediction.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_gaussian_process, only: gp_regression_t
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
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
        integer :: starts = 1
        integer(int64) :: seed = 17_int64
        logical :: include_current = .true.
    end type gp_hyperparameter_options_t

    type, public :: gp_hyperparameter_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        real(dp) :: negative_log_marginal_likelihood = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        integer :: start_count = 0
        integer :: successful_starts = 0
        integer :: best_start = 0
        integer :: objective_evaluations = 0
    end type gp_hyperparameter_result_t

    public :: gp_optimize_hyperparameters
    public :: gp_optimize_hyperparameters_multistart

contains

    subroutine gp_optimize_hyperparameters(model, options, result, status, device)
        type(gp_regression_t), target, intent(inout) :: model
        type(gp_hyperparameter_options_t), intent(in) :: options
        type(gp_hyperparameter_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(fortml_device_t), intent(in), optional :: device

        call gp_optimize_hyperparameters_multistart(model, options, result, status, device)
    end subroutine gp_optimize_hyperparameters

    subroutine gp_optimize_hyperparameters_multistart(model, options, result, status, device)
        !! Optimize exact-GP log parameters from deterministic seeded starts.
        !!
        !! The first start may reuse the fitted state.  Remaining starts are
        !! uniform draws in the closed bound box from `options%seed`.  Only
        !! converged finite runs compete for retention; after all starts the
        !! model is restored to the best parameter vector.  CUDA is refused
        !! explicitly because exact GP factorization and this optimizer are
        !! not resident-device implementations yet.
        type(gp_regression_t), target, intent(inout) :: model
        type(gp_hyperparameter_options_t), intent(in) :: options
        type(gp_hyperparameter_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(fortml_device_t), intent(in), optional :: device
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        type(rng_t) :: generator
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        real(dp), allocatable :: best_parameters(:), initial_parameters(:)
        real(dp) :: local_value, local_gradient_norm, uniform
        integer :: n_parameters, start, j
        logical :: run_converged

        result = gp_hyperparameter_result_t()
        result%start_count = options%starts
        if (present(device)) then
            if (.not. device%selected .or. .not. device%available) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP hyperparameter training: selected device is unavailable")
                return
            end if
            if (device%kind == FORTML_DEVICE_CUDA) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "GP hyperparameter training: no resident CUDA exact-GP optimizer")
                return
            end if
            if (device%kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP hyperparameter training: device kind is invalid")
                return
            end if
        end if
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
        initial_parameters = model%parameters()
        allocate(parameters(n_parameters), best_parameters(n_parameters), &
            lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower = options%lower_bound
        upper = options%upper_bound
        best_parameters = initial_parameters
        call objective%initialize_context(n_parameters, model, &
            gp_negative_lml_objective, status)
        if (status%code /= FORTNUM_OK) return

        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call rng_seed(generator, options%seed, status)
        if (status%code /= FORTNUM_OK) return
        do start = 1, options%starts
            if (start == 1 .and. options%include_current) then
                parameters = initial_parameters
            else
                do j = 1, n_parameters
                    call rng_uniform(generator, uniform)
                    parameters(j) = lower(j) + (upper(j) - lower(j))*uniform
                end do
            end if
            call optimizer%minimize(objective, parameters, lower, upper, &
                optimizer_options, optimizer_result, status)
            result%objective_evaluations = result%objective_evaluations + &
                optimizer_result%line_search_evaluations + 1
            run_converged = optimizer_result%state%converged
            if (status%code == FORTNUM_CONVERGENCE_ERROR) then
                ! Armijo failure or iteration exhaustion is a failed start,
                ! not a reason to discard finite results from other starts.
                call status_set(status, FORTNUM_OK, "")
            else if (status%code /= FORTNUM_OK) then
                return
            end if
            call gp_negative_lml_objective(model, parameters, local_value, gradient, status)
            if (status%code /= FORTNUM_OK) return
            if (.not. ieee_is_finite(local_value) .or. &
                any(.not. ieee_is_finite(gradient))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP hyperparameter training: start returned a nonfinite state")
                return
            end if
            local_gradient_norm = sqrt(sum(gradient**2))
            if (run_converged) then
                result%successful_starts = result%successful_starts + 1
                if (result%successful_starts == 1 .or. &
                    local_value < result%negative_log_marginal_likelihood) then
                    result%negative_log_marginal_likelihood = local_value
                    result%gradient_norm = local_gradient_norm
                    result%iterations = optimizer_result%state%iteration
                    result%best_start = start
                    best_parameters = parameters
                end if
            end if
        end do

        if (result%successful_starts < 1) then
            call model%set_parameters(initial_parameters, status)
            if (status%code /= FORTNUM_OK) return
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "GP hyperparameter training: no multistart run converged")
            return
        end if
        call model%set_parameters(best_parameters, status)
        if (status%code /= FORTNUM_OK) return
        call gp_negative_lml_objective(model, best_parameters, &
            result%negative_log_marginal_likelihood, gradient, status)
        if (status%code /= FORTNUM_OK) return
        result%gradient_norm = sqrt(sum(gradient**2))
        result%converged = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_optimize_hyperparameters_multistart

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
            options%lower_bound <= options%upper_bound .and. &
            options%starts >= 1 .and. options%seed > 0_int64
    end function valid_options

end module fortml_gp_training
