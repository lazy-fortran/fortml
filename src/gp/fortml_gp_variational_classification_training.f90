module fortml_gp_variational_classification_training
    !! FortOpt L-BFGS-B training for the packed variational Bernoulli-GP state.
    !!
    !! The objective is the deterministic ELBO owned by
    !! `gp_variational_classification_t`; this adapter minimizes its negative
    !! value and writes the best packed mean/log-Cholesky vector back to the
    !! model.  It is intentionally CPU-only until the inducing solve,
    !! likelihood table, and reduction are resident on CUDA.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_gp_variational_classification, only: &
        gp_variational_classification_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: gp_variational_classification_lbfgsb_options_t
        !! Bounds and convergence controls for variational-GP ELBO training.
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
    end type gp_variational_classification_lbfgsb_options_t

    type, public :: gp_variational_classification_lbfgsb_result_t
        !! Diagnostics returned by `gp_variational_classification_optimize`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: elbo = -huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type gp_variational_classification_lbfgsb_result_t

    type :: variational_context_t
        type(gp_variational_classification_t), pointer :: model => null()
        real(dp), allocatable :: x(:, :)
        integer, allocatable :: labels(:)
    end type variational_context_t

    public :: gp_variational_classification_optimize

contains

    subroutine gp_variational_classification_optimize(model, x, labels, options, &
            result, status, device)
        !! Maximize the deterministic variational ELBO with FortOpt L-BFGS-B.
        !!
        !! The packed vector follows `model%parameters()`: inducing means,
        !! log-Cholesky diagonals, and strict lower-triangular entries.  Bounds
        !! apply to every packed coordinate; CUDA is rejected explicitly.
        type(gp_variational_classification_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(gp_variational_classification_lbfgsb_options_t), intent(in) :: options
        type(gp_variational_classification_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(fortml_device_t), intent(in), optional :: device
        type(variational_context_t), target :: context
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_parameters

        result = gp_variational_classification_lbfgsb_result_t()
        if (present(device)) then
            if (.not. device%selected .or. .not. device%available) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP training: selected device is unavailable")
                return
            end if
            select case (device%kind)
            case (FORTML_DEVICE_CPU)
                continue
            case (FORTML_DEVICE_CUDA)
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "variational GP training: resident CUDA optimizer is not linked")
                return
            case default
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP training: device kind is invalid")
                return
            end select
        end if
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP training: options are invalid")
            return
        end if
        if (.not. valid_data(model, x, labels)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP training: data or labels are invalid")
            return
        end if
        n_parameters = model%parameter_count()
        if (n_parameters < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP training: model has no parameters")
            return
        end if

        context%model => model
        allocate(context%x, source=x)
        allocate(context%labels, source=labels)
        parameters = model%parameters()
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower = options%lower_bound
        upper = options%upper_bound
        call objective%initialize_context(n_parameters, context, &
            negative_elbo_objective, status)
        if (status%code /= FORTNUM_OK) return
        call copy_lbfgsb_options(options, optimizer_options)
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (status%code /= FORTNUM_OK .and. status%code /= FORTNUM_CONVERGENCE_ERROR) return

        call negative_elbo_objective(context, parameters, result%elbo, gradient, status)
        if (status%code /= FORTNUM_OK) return
        result%elbo = -result%elbo
        result%gradient_norm = sqrt(sum(gradient*gradient))
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        if (.not. ieee_is_finite(result%elbo) .or. &
                .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "variational GP training: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "variational GP training: iteration limit reached")
            return
        end if
        call model%set_parameters(parameters, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_variational_classification_optimize

    subroutine negative_elbo_objective(context_any, parameters, value, gradient, status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: elbo

        value = huge(1.0_dp)
        gradient = 0.0_dp
        select type (context => context_any)
            type is (variational_context_t)
            if (.not. associated(context%model) .or. size(parameters) /= &
                    context%model%parameter_count() .or. size(gradient) /= size(parameters)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP objective: parameter shape is invalid")
                return
            end if
            call context%model%set_parameters(parameters, status)
            if (status%code /= FORTNUM_OK) return
            call context%model%elbo_gradient(context%x, context%labels, elbo, &
                gradient, status)
            if (status%code /= FORTNUM_OK) return
            value = -elbo
            gradient = -gradient
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP objective: value is not finite")
                return
            end if
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP objective: context has the wrong type")
        end select
    end subroutine negative_elbo_objective

    subroutine copy_lbfgsb_options(options, output)
        type(gp_variational_classification_lbfgsb_options_t), intent(in) :: options
        type(lbfgsb_options_t), intent(out) :: output

        output%memory = options%memory
        output%max_iterations = options%max_iterations
        output%max_line_search = options%max_line_search
        output%gradient_tolerance = options%gradient_tolerance
        output%step_tolerance = options%step_tolerance
        output%objective_tolerance = options%objective_tolerance
    end subroutine copy_lbfgsb_options

    logical function valid_options(options) result(valid)
        type(gp_variational_classification_lbfgsb_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. ieee_is_finite(options%objective_tolerance) .and. &
            ieee_is_finite(options%lower_bound) .and. ieee_is_finite(options%upper_bound) .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp .and. options%lower_bound <= options%upper_bound
    end function valid_options

    logical function valid_data(model, x, labels) result(valid)
        type(gp_variational_classification_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        integer :: i, n_positive, n_negative

        n_positive = 0
        n_negative = 0
        do i = 1, size(labels)
            if (labels(i) == 0) then
                n_negative = n_negative + 1
            else if (labels(i) == 1) then
                n_positive = n_positive + 1
            end if
        end do
        valid = model%parameter_count() >= 1 .and. size(x, 1) >= 1 .and. &
            size(x, 2) >= 1 .and. size(labels) == size(x, 1) .and. &
            model%parameter_count() >= 1 .and. all(ieee_is_finite(x)) .and. &
            n_positive > 0 .and. n_negative > 0
    end function valid_data

end module fortml_gp_variational_classification_training
