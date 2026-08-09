!> Fixed-structure boosted-tree objective products.
module fortml_boosted_leaf_objective
    !! Objective and derivative products for the continuous coordinates of a
    !! fitted XGBoost- or LightGBM-style tree ensemble.
    !!
    !! A fitted tree's split thresholds, missing-value routes, categorical
    !! partitions, and tree count are discrete state.  This adapter freezes
    !! that state and exposes the packed `[base_score, leaf weights]`
    !! coordinates already used by the tree leaf-product APIs.  In this
    !! coordinate chart the margin is affine, so weighted squared and binary
    !! logistic losses have exact value/gradient/JVP/VJP/HVP products.  The
    !! bounded FortOpt adapter consumes the same callback; it never
    !! finite-differences a tree coordinate.
    !!
    !! The current operation graph is CPU-resident.  CUDA requests return
    !! `FORTNUM_NOT_IMPLEMENTED`; no host fallback is counted as GPU support.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t
    use fortml_lightgbm, only: lightgbm_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: BOOSTED_LEAF_LOSS_SQUARED = 1
    integer, parameter, public :: BOOSTED_LEAF_LOSS_LOGISTIC = 2
    integer, parameter, public :: BOOSTED_LEAF_MODEL_XGBOOST = 1
    integer, parameter, public :: BOOSTED_LEAF_MODEL_LIGHTGBM = 2

    !> A scalar loss on the fixed leaf/base-coordinate design matrix.
    type, public :: boosted_leaf_objective_t
        private
        real(dp), allocatable :: design(:, :)
        real(dp), allocatable :: initial_parameters(:)
        real(dp), allocatable :: targets(:), weights(:)
        real(dp) :: weight_sum = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        integer :: loss_code = 0
        integer :: model_code = 0
        integer :: device_kind = FORTML_DEVICE_CPU
    contains
        procedure, public :: initialize_xgboost => boosted_leaf_initialize_xgb
        procedure, public :: initialize_lightgbm => boosted_leaf_initialize_lgbm
        generic, public :: initialize => initialize_xgboost, initialize_lightgbm
        procedure, public :: initialized => boosted_leaf_initialized
        procedure, public :: device_supported => boosted_leaf_device_supported
        procedure, public :: parameter_count => boosted_leaf_parameter_count
        procedure, public :: parameters => boosted_leaf_parameters
        procedure, public :: model_kind => boosted_leaf_model_kind
        procedure, public :: loss_kind => boosted_leaf_loss_kind
        procedure, public :: value_gradient => boosted_leaf_value_gradient
        procedure, public :: jvp => boosted_leaf_jvp
        procedure, public :: vjp => boosted_leaf_vjp
        procedure, public :: hvp => boosted_leaf_hvp
        procedure, public :: fortopt => boosted_leaf_fortopt
    end type boosted_leaf_objective_t

    !> Bounds and convergence controls for the fixed-structure objective.
    type, public :: boosted_leaf_lbfgsb_options_t
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
        integer :: device_kind = FORTML_DEVICE_CPU
        real(dp) :: l2 = 0.0_dp
    end type boosted_leaf_lbfgsb_options_t

    !> Result of bounded fixed-structure leaf-coordinate optimization.
    type, public :: boosted_leaf_lbfgsb_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp), allocatable :: parameters(:)
    end type boosted_leaf_lbfgsb_result_t

    public :: boosted_leaf_optimize_lbfgsb_xgb
    public :: boosted_leaf_optimize_lbfgsb_lgbm
    interface boosted_leaf_optimize_lbfgsb
        module procedure boosted_leaf_optimize_lbfgsb_xgb
        module procedure boosted_leaf_optimize_lbfgsb_lgbm
    end interface
    public :: boosted_leaf_optimize_lbfgsb

contains

    subroutine boosted_leaf_initialize_xgb(self, model, x, targets, loss, status, &
            sample_weight, l2, device_kind)
        class(boosted_leaf_objective_t), intent(out) :: self
        type(xgboost_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:)
        integer, intent(in) :: loss
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), l2
        integer, intent(in), optional :: device_kind
        real(dp), allocatable :: design(:, :), initial_parameters(:)
        real(dp) :: regularization
        integer :: requested_device

        regularization = 0.0_dp
        if (present(l2)) regularization = l2
        requested_device = FORTML_DEVICE_CPU
        if (present(device_kind)) requested_device = device_kind
        if (requested_device == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "boosted leaf objective: resident CUDA tree objective is not implemented")
            return
        end if
        if (requested_device /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: device kind is invalid")
            return
        end if
        call xgb_design(model, x, design, initial_parameters, status)
        if (.not. status_ok(status)) return
        call initialize_from_design(self, design, initial_parameters, targets, loss, &
            status, sample_weight, regularization, BOOSTED_LEAF_MODEL_XGBOOST)
        if (status_ok(status)) self%device_kind = requested_device
    end subroutine boosted_leaf_initialize_xgb

    subroutine boosted_leaf_initialize_lgbm(self, model, x, targets, loss, status, &
            sample_weight, l2, device_kind)
        class(boosted_leaf_objective_t), intent(out) :: self
        type(lightgbm_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:)
        integer, intent(in) :: loss
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), l2
        integer, intent(in), optional :: device_kind
        real(dp), allocatable :: design(:, :), initial_parameters(:)
        real(dp) :: regularization
        integer :: requested_device

        regularization = 0.0_dp
        if (present(l2)) regularization = l2
        requested_device = FORTML_DEVICE_CPU
        if (present(device_kind)) requested_device = device_kind
        if (requested_device == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "boosted leaf objective: resident CUDA tree objective is not implemented")
            return
        end if
        if (requested_device /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: device kind is invalid")
            return
        end if
        call lgbm_design(model, x, design, initial_parameters, status)
        if (.not. status_ok(status)) return
        call initialize_from_design(self, design, initial_parameters, targets, loss, &
            status, sample_weight, regularization, BOOSTED_LEAF_MODEL_LIGHTGBM)
        if (status_ok(status)) self%device_kind = requested_device
    end subroutine boosted_leaf_initialize_lgbm

    subroutine xgb_design(model, x, design, initial_parameters, status)
        type(xgboost_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: design(:, :), initial_parameters(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: direction(:), prediction(:), tangent(:)
        integer :: n, p, j

        n = size(x, 1)
        p = model%leaf_parameter_count()
        if (n < 1 .or. size(x, 2) < 1 .or. p < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: XGBoost model or query shape is invalid")
            return
        end if
        allocate(design(n, p), initial_parameters(p), direction(p), prediction(n), tangent(n))
        initial_parameters = model%leaf_parameters(status)
        if (.not. status_ok(status)) return
        do j = 1, p
            direction = 0.0_dp
            direction(j) = 1.0_dp
            call model%predict_leaf_jvp(x, direction, prediction, tangent, status)
            if (.not. status_ok(status)) then
                return
            end if
            design(:, j) = tangent
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_design

    subroutine lgbm_design(model, x, design, initial_parameters, status)
        type(lightgbm_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: design(:, :), initial_parameters(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: direction(:), prediction(:), tangent(:)
        integer :: n, p, j

        n = size(x, 1)
        p = model%leaf_parameter_count()
        if (n < 1 .or. size(x, 2) < 1 .or. p < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: LightGBM model or query shape is invalid")
            return
        end if
        allocate(design(n, p), initial_parameters(p), direction(p), prediction(n), tangent(n))
        initial_parameters = model%leaf_parameters(status)
        if (.not. status_ok(status)) return
        do j = 1, p
            direction = 0.0_dp
            direction(j) = 1.0_dp
            call model%predict_leaf_jvp(x, direction, prediction, tangent, status)
            if (.not. status_ok(status)) then
                return
            end if
            design(:, j) = tangent
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_design

    subroutine initialize_from_design(self, design, initial_parameters, targets, loss, status, &
            sample_weight, l2, model_code)
        class(boosted_leaf_objective_t), intent(out) :: self
        real(dp), intent(in) :: design(:, :), initial_parameters(:), targets(:)
        integer, intent(in) :: loss, model_code
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), l2
        real(dp), allocatable :: weights(:)
        real(dp) :: regularization, weight_sum

        regularization = 0.0_dp
        if (present(l2)) regularization = l2
        if (size(design, 1) < 1 .or. size(design, 2) < 1 .or. &
            size(initial_parameters) /= size(design, 2) .or. &
            size(targets) /= size(design, 1) .or. &
            any(.not. ieee_is_finite(design)) .or. &
            any(.not. ieee_is_finite(targets)) .or. &
            (loss /= BOOSTED_LEAF_LOSS_SQUARED .and. &
            loss /= BOOSTED_LEAF_LOSS_LOGISTIC) .or. &
            (loss == BOOSTED_LEAF_LOSS_LOGISTIC .and. &
            (any(targets < 0.0_dp) .or. any(targets > 1.0_dp))) .or. &
            .not. ieee_is_finite(regularization) .or. regularization < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: design, targets, loss, or L2 is invalid")
            return
        end if
        allocate(weights(size(targets)))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(targets) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "boosted leaf objective: sample weights are invalid")
                return
            end if
            weights = sample_weight
        end if
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: effective weights have no positive mass")
            return
        end if
        allocate(self%design, source=design)
        allocate(self%initial_parameters, source=initial_parameters)
        allocate(self%targets, source=targets)
        call move_alloc(weights, self%weights)
        self%weight_sum = weight_sum
        self%l2 = regularization
        self%loss_code = loss
        self%model_code = model_code
        self%device_kind = FORTML_DEVICE_CPU
        call status_set(status, FORTNUM_OK, "")
    end subroutine initialize_from_design

    logical function boosted_leaf_initialized(self) result(yes)
        class(boosted_leaf_objective_t), intent(in) :: self

        yes = allocated(self%design) .and. allocated(self%targets) .and. &
            allocated(self%weights) .and. size(self%design, 1) > 0 .and. &
            size(self%design, 2) > 0 .and. self%weight_sum > 0.0_dp
    end function boosted_leaf_initialized

    logical function boosted_leaf_device_supported(self, device_kind) result(yes)
        class(boosted_leaf_objective_t), intent(in) :: self
        integer, intent(in), optional :: device_kind
        integer :: requested

        requested = self%device_kind
        if (present(device_kind)) requested = device_kind
        yes = requested == FORTML_DEVICE_CPU
    end function boosted_leaf_device_supported

    integer function boosted_leaf_parameter_count(self) result(n)
        class(boosted_leaf_objective_t), intent(in) :: self

        n = 0
        if (allocated(self%design)) n = size(self%design, 2)
    end function boosted_leaf_parameter_count

    function boosted_leaf_parameters(self) result(parameters)
        class(boosted_leaf_objective_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        allocate(parameters(self%parameter_count()))
        if (allocated(self%initial_parameters)) then
            parameters = self%initial_parameters
        else
            parameters = 0.0_dp
        end if
    end function boosted_leaf_parameters

    integer function boosted_leaf_model_kind(self) result(code)
        class(boosted_leaf_objective_t), intent(in) :: self
        code = self%model_code
    end function boosted_leaf_model_kind

    integer function boosted_leaf_loss_kind(self) result(code)
        class(boosted_leaf_objective_t), intent(in) :: self
        code = self%loss_code
    end function boosted_leaf_loss_kind

    subroutine boosted_leaf_value_gradient(self, parameters, value, gradient, status)
        class(boosted_leaf_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:), response(:), curvature(:)

        value = huge(1.0_dp)
        gradient = 0.0_dp
        if (.not. valid_parameters(self, parameters, gradient, status)) return
        allocate(margin(size(self%targets)), response(size(self%targets)), &
            curvature(size(self%targets)))
        call objective_terms(self, parameters, margin, value, response, curvature, status)
        if (.not. status_ok(status)) return
        gradient = matmul(transpose(self%design), response)/self%weight_sum + &
            self%l2*parameters
        value = value/self%weight_sum + 0.5_dp*self%l2*dot_product(parameters, parameters)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosted_leaf_value_gradient

    subroutine boosted_leaf_jvp(self, parameters, direction, value, tangent, status)
        class(boosted_leaf_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        value = huge(1.0_dp)
        tangent = 0.0_dp
        if (size(direction) /= size(parameters) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective JVP: direction is invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        tangent = dot_product(gradient, direction)
        if (.not. ieee_is_finite(tangent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective JVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosted_leaf_jvp

    subroutine boosted_leaf_vjp(self, parameters, output_bar, gradient, status)
        class(boosted_leaf_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), output_bar
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value

        gradient = 0.0_dp
        if (.not. ieee_is_finite(output_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective VJP: output cotangent is invalid")
            return
        end if
        call self%value_gradient(parameters, value, gradient, status)
        if (.not. status_ok(status)) return
        gradient = output_bar*gradient
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosted_leaf_vjp

    subroutine boosted_leaf_hvp(self, parameters, direction, product, status)
        class(boosted_leaf_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:), response(:), curvature(:), tangent(:)
        real(dp) :: base_value

        product = 0.0_dp
        if (.not. valid_parameters(self, parameters, product, status) .or. &
            size(direction) /= size(parameters) .or. &
            any(.not. ieee_is_finite(direction))) then
            if (status%code == FORTNUM_OK) call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective HVP: direction is invalid")
            return
        end if
        allocate(margin(size(self%targets)), response(size(self%targets)), &
            curvature(size(self%targets)), tangent(size(self%targets)))
        call objective_terms(self, parameters, margin, base_value, response, curvature, status)
        if (.not. status_ok(status)) return
        tangent = matmul(self%design, direction)
        product = matmul(transpose(self%design), curvature*tangent)/self%weight_sum + &
            self%l2*direction
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosted_leaf_hvp

    subroutine boosted_leaf_fortopt(self, objective, status)
        class(boosted_leaf_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: adapter is not initialized")
            return
        end if
        call objective%initialize_context(self%parameter_count(), self, &
            boosted_leaf_context_callback, status)
    end subroutine boosted_leaf_fortopt

    subroutine boosted_leaf_context_callback(context, parameters, value, gradient, status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (boosted_leaf_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: context has the wrong type")
        end select
    end subroutine boosted_leaf_context_callback

    subroutine boosted_leaf_optimize_lbfgsb_xgb(model, x, targets, loss, options, &
            result, status, sample_weight)
        type(xgboost_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:)
        integer, intent(in) :: loss
        type(boosted_leaf_lbfgsb_options_t), intent(in) :: options
        type(boosted_leaf_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        type(boosted_leaf_objective_t), target :: adapter

        call result_defaults(result)
        call validate_lbfgsb_options(options, status)
        if (.not. status_ok(status)) return
        call adapter%initialize_xgboost(model, x, targets, loss, status, &
            sample_weight=sample_weight, l2=options%l2, device_kind=options%device_kind)
        if (.not. status_ok(status)) return
        call optimize_adapter(adapter, options, result, status)
    end subroutine boosted_leaf_optimize_lbfgsb_xgb

    subroutine boosted_leaf_optimize_lbfgsb_lgbm(model, x, targets, loss, options, &
            result, status, sample_weight)
        type(lightgbm_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:)
        integer, intent(in) :: loss
        type(boosted_leaf_lbfgsb_options_t), intent(in) :: options
        type(boosted_leaf_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        type(boosted_leaf_objective_t), target :: adapter

        call result_defaults(result)
        call validate_lbfgsb_options(options, status)
        if (.not. status_ok(status)) return
        call adapter%initialize_lightgbm(model, x, targets, loss, status, &
            sample_weight=sample_weight, l2=options%l2, device_kind=options%device_kind)
        if (.not. status_ok(status)) return
        call optimize_adapter(adapter, options, result, status)
    end subroutine boosted_leaf_optimize_lbfgsb_lgbm

    subroutine optimize_adapter(adapter, options, result, status)
        type(boosted_leaf_objective_t), target, intent(inout) :: adapter
        type(boosted_leaf_lbfgsb_options_t), intent(in) :: options
        type(boosted_leaf_lbfgsb_result_t), intent(inout) :: result
        type(fortnum_status_t), intent(out) :: status
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n

        n = adapter%parameter_count()
        allocate(parameters(n), lower(n), upper(n), gradient(n))
        parameters = adapter%parameters()
        lower = options%lower_bound
        upper = options%upper_bound
        parameters = min(max(parameters, lower), upper)
        call adapter%fortopt(objective, status)
        if (.not. status_ok(status)) return
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        call optimizer%minimize(objective, parameters, lower, upper, &
            optimizer_options, optimizer_result, status)
        if (.not. status_ok(status)) return
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) return
        result%parameters = parameters
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        result%gradient_norm = sqrt(sum(gradient*gradient))
        if (.not. ieee_is_finite(result%objective) .or. &
            .not. ieee_is_finite(result%gradient_norm)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "boosted leaf L-BFGS-B: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "boosted leaf L-BFGS-B: iteration limit reached")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine optimize_adapter

    subroutine result_defaults(result)
        type(boosted_leaf_lbfgsb_result_t), intent(out) :: result

        result%converged = .false.
        result%iterations = 0
        result%line_search_evaluations = 0
        result%objective = huge(1.0_dp)
        result%gradient_norm = huge(1.0_dp)
    end subroutine result_defaults

    subroutine validate_lbfgsb_options(options, status)
        type(boosted_leaf_lbfgsb_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status

        if (options%memory < 1 .or. options%max_iterations < 1 .or. &
            options%max_line_search < 1 .or. &
            .not. ieee_is_finite(options%gradient_tolerance) .or. &
            .not. ieee_is_finite(options%step_tolerance) .or. &
            .not. ieee_is_finite(options%objective_tolerance) .or. &
            options%gradient_tolerance < 0.0_dp .or. &
            options%step_tolerance < 0.0_dp .or. &
            options%objective_tolerance < 0.0_dp .or. &
            .not. ieee_is_finite(options%lower_bound) .or. &
            .not. ieee_is_finite(options%upper_bound) .or. &
            options%lower_bound > options%upper_bound .or. &
            .not. ieee_is_finite(options%l2) .or. options%l2 < 0.0_dp .or. &
            (options%device_kind /= FORTML_DEVICE_CPU .and. &
            options%device_kind /= FORTML_DEVICE_CUDA)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf L-BFGS-B: options or bounds are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_lbfgsb_options

    logical function valid_parameters(self, parameters, output, status) result(valid)
        class(boosted_leaf_objective_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(inout) :: output(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        output = 0.0_dp
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: adapter is not initialized")
            return
        end if
        if (size(parameters) /= self%parameter_count() .or. &
            size(output) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: parameter or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_parameters

    subroutine objective_terms(self, parameters, margin, value, response, curvature, status)
        class(boosted_leaf_objective_t), intent(in) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: margin(:), value, response(:), curvature(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: residual, probability, loss_value
        integer :: i

        value = 0.0_dp
        margin = matmul(self%design, parameters)
        response = 0.0_dp
        curvature = 0.0_dp
        do i = 1, size(margin)
            if (self%loss_code == BOOSTED_LEAF_LOSS_SQUARED) then
                residual = margin(i) - self%targets(i)
                loss_value = 0.5_dp*residual*residual
                response(i) = self%weights(i)*residual
                curvature(i) = self%weights(i)
            else
                probability = stable_sigmoid(margin(i))
                loss_value = stable_softplus(margin(i)) - &
                    self%targets(i)*margin(i)
                response(i) = self%weights(i)*(probability - self%targets(i))
                curvature(i) = self%weights(i)*probability*(1.0_dp-probability)
            end if
            value = value + self%weights(i)*loss_value
        end do
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(margin)) .or. &
            any(.not. ieee_is_finite(response)) .or. &
            any(.not. ieee_is_finite(curvature))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted leaf objective: loss products are not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine objective_terms

    pure real(dp) function stable_sigmoid(value) result(output)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            output = 1.0_dp/(1.0_dp + exp(-value))
        else
            output = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

    pure real(dp) function stable_softplus(value) result(output)
        real(dp), intent(in) :: value

        output = max(value, 0.0_dp) + log(1.0_dp + exp(-abs(value)))
    end function stable_softplus

end module fortml_boosted_leaf_objective
