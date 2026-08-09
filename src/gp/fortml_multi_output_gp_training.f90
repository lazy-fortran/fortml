module fortml_multi_output_gp_training
    !! FortOpt hyperparameter training for intrinsic-coregionalization GPs.
    !!
    !! `multi_output_gp_t` already provides analytic likelihood gradients and
    !! Hessian-vector products for its packed kernel/noise/coregionalization
    !! state.  This adapter makes that state a first-class FortOpt objective:
    !! every L-BFGS-B trial is applied through the model's transactional setter,
    !! the exact negative log marginal likelihood is returned, and a failed
    !! trial cannot leave a partially refactored posterior behind.
    !!
    !! The bounds distinguish log kernel/noise coordinates, unconstrained
    !! latent-output weights, and non-negative independent output variances.
    !! CUDA is an explicit capability boundary: exact ICM factorization and
    !! this optimizer are CPU implementations until a resident backend exists.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_multi_output_gp, only: multi_output_gp_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: multi_output_gp_hyperparameter_options_t
        !! Bounds and convergence controls for exact ICM L-BFGS-B.
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: kernel_lower_bound = -20.0_dp
        real(dp) :: kernel_upper_bound = 20.0_dp
        real(dp) :: noise_lower_bound = -20.0_dp
        real(dp) :: noise_upper_bound = 20.0_dp
        real(dp) :: weight_lower_bound = -20.0_dp
        real(dp) :: weight_upper_bound = 20.0_dp
        real(dp) :: independent_lower_bound = 0.0_dp
        real(dp) :: independent_upper_bound = 20.0_dp
        integer :: starts = 1
        integer(int64) :: seed = 17_int64
        logical :: include_current = .true.
    end type multi_output_gp_hyperparameter_options_t

    type, public :: multi_output_gp_hyperparameter_result_t
        !! Diagnostics for the retained exact-Icm optimization run.
        logical :: converged = .false.
        integer :: iterations = 0
        real(dp) :: negative_log_marginal_likelihood = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        integer :: start_count = 0
        integer :: successful_starts = 0
        integer :: best_start = 0
        integer :: objective_evaluations = 0
    end type multi_output_gp_hyperparameter_result_t

    public :: multi_output_gp_optimize_hyperparameters

contains

    subroutine multi_output_gp_optimize_hyperparameters(model, options, result, status, device)
        !! Optimize all exact ICM hyperparameters using FortOpt L-BFGS-B.
        type(multi_output_gp_t), target, intent(inout) :: model
        type(multi_output_gp_hyperparameter_options_t), intent(in) :: options
        type(multi_output_gp_hyperparameter_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(fortml_device_t), intent(in), optional :: device

        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        type(rng_t) :: generator
        real(dp), allocatable :: parameters(:), initial_parameters(:), best_parameters(:)
        real(dp), allocatable :: lower(:), upper(:), gradient(:)
        real(dp) :: local_value, local_gradient_norm, uniform
        integer :: n_parameters, kernel_count, rank, output_count
        integer :: independent_start, start, j
        logical :: run_converged
        type(multi_output_gp_hyperparameter_result_t) :: result_default

        result = result_default
        result%start_count = options%starts

        if (present(device)) then
            if (.not. device%selected .or. .not. device%available) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multi-output GP training: selected device is unavailable")
                return
            end if
            select case (device%kind)
            case (FORTML_DEVICE_CPU)
                continue
            case (FORTML_DEVICE_CUDA)
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "multi-output GP training: no resident CUDA ICM optimizer")
                return
            case default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multi-output GP training: device kind is invalid")
                return
            end select
        end if
        if (.not. model%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP training: model must be fitted first")
            return
        end if
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP training: options are invalid")
            return
        end if

        n_parameters = model%parameter_count()
        kernel_count = model%kernel%parameter_count()
        rank = size(model%weights, 2)
        output_count = model%n_outputs
        independent_start = kernel_count + 2 + output_count*rank
        if (n_parameters < 1 .or. independent_start + output_count - 1 /= n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP training: packed model state is invalid")
            return
        end if

        initial_parameters = model%parameters()
        allocate(parameters(n_parameters), best_parameters(n_parameters), &
            lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        if (kernel_count > 0) then
            lower(:kernel_count) = options%kernel_lower_bound
            upper(:kernel_count) = options%kernel_upper_bound
        end if
        lower(kernel_count + 1) = options%noise_lower_bound
        upper(kernel_count + 1) = options%noise_upper_bound
        if (output_count*rank > 0) then
            lower(kernel_count + 2:independent_start - 1) = options%weight_lower_bound
            upper(kernel_count + 2:independent_start - 1) = options%weight_upper_bound
        end if
        lower(independent_start:) = options%independent_lower_bound
        upper(independent_start:) = options%independent_upper_bound
        best_parameters = initial_parameters

        call objective%initialize_context(n_parameters, model, &
            multi_output_gp_negative_lml_objective, status)
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
                call status_set(status, FORTNUM_OK, "")
            else if (status%code /= FORTNUM_OK) then
                return
            end if
            call multi_output_gp_negative_lml_objective(model, parameters, &
                local_value, gradient, status)
            if (status%code /= FORTNUM_OK) return
            local_gradient_norm = sqrt(sum(gradient*gradient))
            if (.not. ieee_is_finite(local_value) .or. &
                .not. ieee_is_finite(local_gradient_norm)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multi-output GP training: start returned a nonfinite state")
                return
            end if
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
                "multi-output GP training: no optimization start converged")
            return
        end if
        call model%set_parameters(best_parameters, status)
        if (status%code /= FORTNUM_OK) return
        call multi_output_gp_negative_lml_objective(model, best_parameters, &
            result%negative_log_marginal_likelihood, gradient, status)
        if (status%code /= FORTNUM_OK) return
        result%gradient_norm = sqrt(sum(gradient*gradient))
        result%converged = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_gp_optimize_hyperparameters

    subroutine multi_output_gp_negative_lml_objective(context, parameters, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: likelihood

        value = huge(1.0_dp)
        gradient = 0.0_dp
        select type (model => context)
            type is (multi_output_gp_t)
            if (size(parameters) /= model%parameter_count() .or. &
                size(gradient) /= size(parameters)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multi-output GP objective: parameter shape is invalid")
                return
            end if
            call model%set_parameters(parameters, status)
            if (status%code /= FORTNUM_OK) return
            call model%log_marginal_likelihood(model%targets, likelihood, status)
            if (status%code /= FORTNUM_OK) return
            call model%hyperparameter_gradient(gradient, status)
            if (status%code /= FORTNUM_OK) return
            value = -likelihood
            gradient = -gradient
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "multi-output GP objective: value or gradient is not finite")
                return
            end if
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP objective: context has the wrong type")
        end select
    end subroutine multi_output_gp_negative_lml_objective

    logical function valid_options(options) result(valid)
        type(multi_output_gp_hyperparameter_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. options%starts >= 1 .and. options%seed > 0_int64
        valid = valid .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. ieee_is_finite(options%objective_tolerance)
        valid = valid .and. options%gradient_tolerance >= 0.0_dp .and. &
            options%step_tolerance >= 0.0_dp .and. options%objective_tolerance >= 0.0_dp
        valid = valid .and. finite_ordered(options%kernel_lower_bound, options%kernel_upper_bound)
        valid = valid .and. finite_ordered(options%noise_lower_bound, options%noise_upper_bound)
        valid = valid .and. finite_ordered(options%weight_lower_bound, options%weight_upper_bound)
        valid = valid .and. finite_ordered(options%independent_lower_bound, &
            options%independent_upper_bound)
        valid = valid .and. options%independent_lower_bound >= 0.0_dp
    end function valid_options

    logical function finite_ordered(lower, upper) result(valid)
        real(dp), intent(in) :: lower, upper

        valid = ieee_is_finite(lower) .and. ieee_is_finite(upper) .and. lower <= upper
    end function finite_ordered

end module fortml_multi_output_gp_training
