module fortml_gp_ordinal_cutpoint_training
    !! Fixed-latent cut-point calibration for ordered GP classifiers.
    !!
    !! The latent GP fit remains fixed.  FortOpt L-BFGS-B minimizes the
    !! weighted ordered-logit or ordered-probit negative log likelihood over a
    !! location coordinate followed by log gap coordinates.  The transform
    !! makes every trial cut-point vector strictly increasing.  The fitted
    !! model is mutated only after convergence, so every refusal preserves its
    !! previous thresholds.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_gp_ordinal_classification, only: gp_ordinal_classification_t, &
        GP_ORDINAL_LIKELIHOOD_LOGISTIC, GP_ORDINAL_LIKELIHOOD_PROBIT, &
        gp_ordinal_log_likelihood_value, gp_ordinal_log_likelihood_vjp, &
        gp_ordinal_log_likelihood_hvp
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: gp_ordinal_cutpoint_options_t
        integer :: likelihood = GP_ORDINAL_LIKELIHOOD_PROBIT
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-7_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: minimum_gap = 1.0e-6_dp
        real(dp) :: location_lower = -20.0_dp
        real(dp) :: location_upper = 20.0_dp
        real(dp) :: log_gap_lower = -20.0_dp
        real(dp) :: log_gap_upper = 20.0_dp
    end type gp_ordinal_cutpoint_options_t

    type, public :: gp_ordinal_cutpoint_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: initial_negative_log_likelihood = huge(1.0_dp)
        real(dp) :: negative_log_likelihood = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type gp_ordinal_cutpoint_result_t

    type :: cutpoint_context_t
        real(dp), allocatable :: eta(:)
        real(dp), allocatable :: weight(:)
        integer, allocatable :: rank(:)
        integer :: likelihood = GP_ORDINAL_LIKELIHOOD_PROBIT
        real(dp) :: minimum_gap = 1.0e-6_dp
    end type cutpoint_context_t

    public :: gp_ordinal_cutpoint_value_gradient
    public :: gp_ordinal_cutpoint_hvp
    public :: gp_ordinal_cutpoint_value_gradient_device
    public :: gp_ordinal_cutpoint_device_supported
    public :: gp_ordinal_optimize_cutpoints

contains

    subroutine gp_ordinal_cutpoint_value_gradient(model, x, labels, thresholds, &
            likelihood, value, gradient, status, sample_weight)
        !! Evaluate the weighted fixed-posterior-mean cut-point objective.
        type(gp_ordinal_classification_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), thresholds(:)
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        type(cutpoint_context_t) :: context

        value = 0.0_dp
        gradient = 0.0_dp
        call prepare_context(model, x, labels, likelihood, context, status, sample_weight)
        if (status%code /= FORTNUM_OK) return
        call evaluate_threshold_objective(context, thresholds, value, gradient, status)
    end subroutine gp_ordinal_cutpoint_value_gradient

    subroutine gp_ordinal_cutpoint_hvp(model, x, labels, thresholds, likelihood, &
            direction, value, gradient, hessian_product, status, sample_weight)
        !! Return the exact threshold Hessian applied to `direction`.
        type(gp_ordinal_classification_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), thresholds(:), direction(:)
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: value, gradient(:), hessian_product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        type(cutpoint_context_t) :: context

        value = 0.0_dp
        gradient = 0.0_dp
        hessian_product = 0.0_dp
        call prepare_context(model, x, labels, likelihood, context, status, sample_weight)
        if (status%code /= FORTNUM_OK) return
        if (size(direction) /= size(thresholds) .or. &
            size(hessian_product) /= size(thresholds) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point HVP: direction or output shape is invalid")
            return
        end if
        call evaluate_threshold_objective(context, thresholds, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        call evaluate_threshold_hvp(context, thresholds, direction, hessian_product, status)
    end subroutine gp_ordinal_cutpoint_hvp

    subroutine gp_ordinal_cutpoint_value_gradient_device(device, model, x, labels, &
            thresholds, likelihood, value, gradient, status, sample_weight)
        type(fortml_device_t), intent(in) :: device
        type(gp_ordinal_classification_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), thresholds(:)
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)

        value = 0.0_dp
        gradient = 0.0_dp
        call cutpoint_device_dispatch(device, status, &
            "ordinal GP cut-point objective device")
        if (status%code /= FORTNUM_OK) return
        call gp_ordinal_cutpoint_value_gradient(model, x, labels, thresholds, likelihood, &
            value, gradient, status, sample_weight)
    end subroutine gp_ordinal_cutpoint_value_gradient_device

    logical function gp_ordinal_cutpoint_device_supported(device_kind) result(supported)
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function gp_ordinal_cutpoint_device_supported

    subroutine gp_ordinal_optimize_cutpoints(model, x, labels, options, result, &
            status, device, sample_weight)
        !! Calibrate strictly ordered cut points while holding the latent GP fixed.
        type(gp_ordinal_classification_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(gp_ordinal_cutpoint_options_t), intent(in) :: options
        type(gp_ordinal_cutpoint_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(fortml_device_t), intent(in), optional :: device
        real(dp), intent(in), optional :: sample_weight(:)
        type(cutpoint_context_t), target :: context
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: initial_thresholds(:), thresholds(:)
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        type(gp_ordinal_cutpoint_result_t) :: default_result

        result = default_result
        if (present(device)) then
            call cutpoint_device_dispatch(device, status, &
                "ordinal GP cut-point training device")
            if (status%code /= FORTNUM_OK) return
        end if
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point training: options are invalid")
            return
        end if
        call prepare_context(model, x, labels, options%likelihood, context, status, &
            sample_weight)
        if (status%code /= FORTNUM_OK) return
        initial_thresholds = model%thresholds()
        allocate(parameters(size(initial_thresholds)), lower(size(initial_thresholds)), &
            upper(size(initial_thresholds)), thresholds(size(initial_thresholds)), &
            gradient(size(initial_thresholds)))
        call encode_thresholds(initial_thresholds, options%minimum_gap, parameters, status)
        if (status%code /= FORTNUM_OK) return
        lower = options%log_gap_lower
        upper = options%log_gap_upper
        lower(1) = options%location_lower
        upper(1) = options%location_upper
        context%minimum_gap = options%minimum_gap
        call evaluate_threshold_objective(context, initial_thresholds, &
            result%initial_negative_log_likelihood, gradient, status)
        if (status%code /= FORTNUM_OK) return
        call objective%initialize_context(size(parameters), context, &
            transformed_objective, status)
        if (status%code /= FORTNUM_OK) return
        call copy_options(options, optimizer_options)
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (status%code /= FORTNUM_OK .and. &
            status%code /= FORTNUM_CONVERGENCE_ERROR) return
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP cut-point training: iteration limit reached")
            return
        end if
        call decode_thresholds(parameters, options%minimum_gap, thresholds, status)
        if (status%code /= FORTNUM_OK) return
        call evaluate_threshold_objective(context, thresholds, &
            result%negative_log_likelihood, gradient, status)
        if (status%code /= FORTNUM_OK) return
        result%gradient_norm = sqrt(sum(gradient*gradient))
        if (.not. ieee_is_finite(result%negative_log_likelihood) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP cut-point training: result is not finite")
            return
        end if
        call model%set_thresholds(thresholds, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_optimize_cutpoints

    subroutine transformed_objective(context_any, parameters, value, gradient, status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: thresholds(size(parameters)), threshold_gradient(size(parameters))
        integer :: j

        value = huge(1.0_dp)
        gradient = 0.0_dp
        select type (context => context_any)
            type is (cutpoint_context_t)
            if (size(gradient) /= size(parameters) .or. size(parameters) < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP cut-point objective: parameter shape is invalid")
                return
            end if
            call decode_thresholds(parameters, context%minimum_gap, thresholds, status)
            if (status%code /= FORTNUM_OK) return
            call evaluate_threshold_objective(context, thresholds, value, &
                threshold_gradient, status)
            if (status%code /= FORTNUM_OK) return
            gradient(1) = sum(threshold_gradient)
            do j = 2, size(parameters)
                gradient(j) = exp(parameters(j))*sum(threshold_gradient(j:))
            end do
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point objective: context has the wrong type")
        end select
    end subroutine transformed_objective

    subroutine evaluate_threshold_objective(context, thresholds, value, gradient, status)
        type(cutpoint_context_t), intent(in) :: context
        real(dp), intent(in) :: thresholds(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: eta_bar(1), local_gradient(size(thresholds)), row_value
        real(dp) :: eta_row(1), weight_mass
        integer :: i, rank_row(1)

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. valid_thresholds(thresholds) .or. &
            size(gradient) /= size(thresholds)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point objective: thresholds are invalid")
            return
        end if
        weight_mass = sum(context%weight)
        do i = 1, size(context%eta)
            eta_row(1) = context%eta(i)
            rank_row(1) = context%rank(i)
            call gp_ordinal_log_likelihood_value(eta_row, rank_row, thresholds, &
                context%likelihood, row_value, status)
            if (status%code /= FORTNUM_OK) return
            call gp_ordinal_log_likelihood_vjp(eta_row, rank_row, thresholds, &
                context%likelihood, -context%weight(i)/weight_mass, eta_bar, &
                local_gradient, status)
            if (status%code /= FORTNUM_OK) return
            value = value - context%weight(i)*row_value/weight_mass
            gradient = gradient + local_gradient
        end do
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP cut-point objective: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine evaluate_threshold_objective

    subroutine evaluate_threshold_hvp(context, thresholds, direction, product, status)
        type(cutpoint_context_t), intent(in) :: context
        real(dp), intent(in) :: thresholds(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: eta_direction(1), eta_product(1), eta_row(1)
        real(dp) :: local_product(size(thresholds)), weight_mass
        integer :: i, rank_row(1)

        product = 0.0_dp
        eta_direction = 0.0_dp
        weight_mass = sum(context%weight)
        do i = 1, size(context%eta)
            eta_row(1) = context%eta(i)
            rank_row(1) = context%rank(i)
            call gp_ordinal_log_likelihood_hvp(eta_row, rank_row, thresholds, &
                context%likelihood, -context%weight(i)/weight_mass, eta_direction, &
                direction, eta_product, local_product, status)
            if (status%code /= FORTNUM_OK) return
            product = product + local_product
        end do
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP cut-point HVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine evaluate_threshold_hvp

    subroutine prepare_context(model, x, labels, likelihood, context, status, sample_weight)
        type(gp_ordinal_classification_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:), likelihood
        type(cutpoint_context_t), intent(out) :: context
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: classes(:)
        real(dp), allocatable :: variance(:), class_mass(:)
        integer :: i, j

        if (.not. model%fitted() .or. size(x, 1) < 1 .or. &
            size(x, 2) /= model%feature_count() .or. size(labels) /= size(x, 1) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point objective: model or data are invalid")
            return
        end if
        if (likelihood /= GP_ORDINAL_LIKELIHOOD_LOGISTIC .and. &
            likelihood /= GP_ORDINAL_LIKELIHOOD_PROBIT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point objective: likelihood is invalid")
            return
        end if
        allocate(context%eta(size(labels)), variance(size(labels)), &
            context%rank(size(labels)), context%weight(size(labels)))
        call model%predict_latent(x, context%eta, variance, status)
        if (status%code /= FORTNUM_OK) return
        context%weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP cut-point objective: sample weights are invalid")
                return
            end if
            context%weight = sample_weight
        end if
        if (sum(context%weight) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point objective: sample weights have no positive mass")
            return
        end if
        classes = model%classes()
        allocate(class_mass(size(classes)))
        class_mass = 0.0_dp
        do i = 1, size(labels)
            context%rank(i) = 0
            do j = 1, size(classes)
                if (labels(i) == classes(j)) then
                    context%rank(i) = j
                    class_mass(j) = class_mass(j) + context%weight(i)
                    exit
                end if
            end do
            if (context%rank(i) == 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP cut-point objective: labels differ from fitted classes")
                return
            end if
        end do
        if (any(class_mass <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point objective: every class needs positive weight")
            return
        end if
        context%likelihood = likelihood
        call status_set(status, FORTNUM_OK, "")
    end subroutine prepare_context

    subroutine encode_thresholds(thresholds, minimum_gap, parameters, status)
        real(dp), intent(in) :: thresholds(:), minimum_gap
        real(dp), intent(out) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: gap
        integer :: j

        parameters = 0.0_dp
        if (size(parameters) /= size(thresholds) .or. .not. valid_thresholds(thresholds)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point encoding: thresholds are invalid")
            return
        end if
        parameters(1) = thresholds(1)
        do j = 2, size(thresholds)
            gap = thresholds(j) - thresholds(j - 1) - minimum_gap
            if (.not. ieee_is_finite(gap) .or. gap <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP cut-point encoding: an initial gap is too small")
                return
            end if
            parameters(j) = log(gap)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine encode_thresholds

    subroutine decode_thresholds(parameters, minimum_gap, thresholds, status)
        real(dp), intent(in) :: parameters(:), minimum_gap
        real(dp), intent(out) :: thresholds(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: j

        thresholds = 0.0_dp
        if (size(thresholds) /= size(parameters) .or. size(parameters) < 1 .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point decoding: parameters are invalid")
            return
        end if
        thresholds(1) = parameters(1)
        do j = 2, size(parameters)
            thresholds(j) = thresholds(j - 1) + minimum_gap + exp(parameters(j))
        end do
        if (.not. valid_thresholds(thresholds)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP cut-point decoding: transformed thresholds are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine decode_thresholds

    logical function valid_thresholds(thresholds) result(valid)
        real(dp), intent(in) :: thresholds(:)

        valid = size(thresholds) >= 1
        if (.not. valid) return
        valid = all(ieee_is_finite(thresholds))
        if (.not. valid) return
        if (size(thresholds) > 1) then
            valid = all(thresholds(2:) > thresholds(:size(thresholds) - 1))
        end if
    end function valid_thresholds

    logical function valid_options(options) result(valid)
        type(gp_ordinal_cutpoint_options_t), intent(in) :: options

        valid = options%likelihood == GP_ORDINAL_LIKELIHOOD_LOGISTIC .or. &
            options%likelihood == GP_ORDINAL_LIKELIHOOD_PROBIT
        if (.not. valid) return
        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1
        if (.not. valid) return
        valid = all(ieee_is_finite([options%gradient_tolerance, &
            options%step_tolerance, options%objective_tolerance, options%minimum_gap, &
            options%location_lower, options%location_upper, options%log_gap_lower, &
            options%log_gap_upper]))
        if (.not. valid) return
        valid = options%gradient_tolerance >= 0.0_dp .and. &
            options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp .and. options%minimum_gap >= 0.0_dp .and. &
            options%location_lower <= options%location_upper .and. &
            options%log_gap_lower <= options%log_gap_upper
    end function valid_options

    subroutine copy_options(options, output)
        type(gp_ordinal_cutpoint_options_t), intent(in) :: options
        type(lbfgsb_options_t), intent(out) :: output

        output%memory = options%memory
        output%max_iterations = options%max_iterations
        output%max_line_search = options%max_line_search
        output%gradient_tolerance = options%gradient_tolerance
        output%step_tolerance = options%step_tolerance
        output%objective_tolerance = options%objective_tolerance
    end subroutine copy_options

    subroutine cutpoint_device_dispatch(device, status, operation)
        type(fortml_device_t), intent(in) :: device
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": selected device is unavailable")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call status_set(status, FORTNUM_OK, "")
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, trim(operation)// &
                ": resident ordinal likelihood reduction is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": device kind is invalid")
        end select
    end subroutine cutpoint_device_dispatch

end module fortml_gp_ordinal_cutpoint_training
