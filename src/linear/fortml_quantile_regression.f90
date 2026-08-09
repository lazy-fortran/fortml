module fortml_quantile_regression
    !! Weighted multi-output linear quantile regression.
    !!
    !! Each output column has one quantile level.  Fit uses the exact weighted
    !! pinball objective and FortOpt L-BFGS-B.  Quantile levels are sorted once
    !! at initialization, making packed parameters and output metadata
    !! deterministic.  The fitted affine predictor has analytic input and
    !! parameter JVP/VJP products.  Objective HVPs are zero away from pinball
    !! residual kinks (apart from the feature L2 block) and refuse a residual
    !! kink instead of fabricating a second derivative.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_parameter_registry, only: parameter_registry_t, &
        parameter_block_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    real(dp), parameter :: QUANTILE_DEFAULT_BOUND = 30.0_dp
    real(dp), parameter :: QUANTILE_DEFAULT_LEVEL = 0.5_dp
    real(dp), parameter :: QUANTILE_DEFAULT_KINK_TOLERANCE = 1.0e-10_dp

    !> A weighted affine predictor with one fitted output per quantile level.
    type, public :: quantile_regression_t
        private
        real(dp), allocatable :: coefficient(:, :)
        real(dp), allocatable :: quantile_level_value(:)
        integer :: feature_count_value = 0
        integer :: output_count_value = 0
        real(dp) :: l2_value = 0.0_dp
        logical :: fit_intercept_value = .true.
        logical :: fitted_value = .false.
    contains
        procedure, public :: initialize => quantile_initialize
        procedure, public :: fit_matrix => quantile_fit_matrix
        procedure, public :: fit_vector => quantile_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: predict_matrix => quantile_predict_matrix
        procedure, public :: predict_vector => quantile_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_device => quantile_predict_device
        procedure, public :: device_supported => quantile_device_supported
        procedure, public :: predict_jvp => quantile_predict_jvp
        procedure, public :: predict_vjp => quantile_predict_vjp
        procedure, public :: jvp => quantile_predict_jvp
        procedure, public :: vjp => quantile_predict_vjp
        procedure, public :: coefficients => quantile_coefficients
        procedure, public :: parameters => quantile_parameters
        procedure, public :: set_parameters => quantile_set_parameters
        procedure, public :: parameter_count => quantile_parameter_count
        procedure, public :: feature_count => quantile_feature_count
        procedure, public :: output_count => quantile_output_count
        procedure, public :: quantile_levels => quantile_levels_value
        procedure, public :: quantiles => quantile_levels_value
        procedure, public :: regularization => quantile_regularization
        procedure, public :: fit_intercept => quantile_fit_intercept
        procedure, public :: fitted => quantile_fitted
    end type quantile_regression_t

    !> FortOpt-facing pinball objective for a fixed design and target set.
    type, public :: quantile_training_objective_t
        private
        type(quantile_regression_t), pointer :: model => null()
        type(parameter_registry_t) :: registry
        real(dp), allocatable :: features(:, :), targets(:, :), weights(:)
        real(dp) :: weight_sum = 0.0_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: kink_tolerance = QUANTILE_DEFAULT_KINK_TOLERANCE
        real(dp) :: smoothing = 0.0_dp
    contains
        procedure, public :: initialize => quantile_objective_initialize
        procedure, public :: parameter_count => quantile_objective_parameter_count
        procedure, public :: parameters => quantile_objective_parameters
        procedure, public :: value_gradient => quantile_objective_value_gradient
        procedure, public :: jvp => quantile_objective_jvp
        procedure, public :: vjp => quantile_objective_vjp
        procedure, public :: hvp => quantile_objective_hvp
        procedure, public :: fortopt => quantile_objective_fortopt
    end type quantile_training_objective_t

    !> Bounded L-BFGS-B controls for a quantile fit.
    type, public :: quantile_lbfgsb_options_t
        integer :: memory = 10
        integer :: max_iterations = 500
        integer :: max_line_search = 80
        real(dp) :: gradient_tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -QUANTILE_DEFAULT_BOUND
        real(dp) :: upper_bound = QUANTILE_DEFAULT_BOUND
        real(dp) :: l2 = 0.0_dp
        real(dp) :: kink_tolerance = QUANTILE_DEFAULT_KINK_TOLERANCE
        real(dp) :: fit_smoothing = 1.0e-1_dp
        integer :: device_kind = FORTML_DEVICE_CPU
    end type quantile_lbfgsb_options_t

    !> Diagnostics returned by `quantile_optimize_lbfgsb`.
    type, public :: quantile_lbfgsb_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: exact_gradient_norm = huge(1.0_dp)
        real(dp) :: fit_smoothing = 0.0_dp
    end type quantile_lbfgsb_result_t

    public :: quantile_optimize_lbfgsb

contains

    subroutine quantile_initialize(self, n_features, n_outputs, levels, status, &
            fit_intercept)
        class(quantile_regression_t), intent(out) :: self
        integer, intent(in) :: n_features, n_outputs
        real(dp), intent(in) :: levels(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fit_intercept
        logical :: include_intercept
        integer :: n_parameters
        real(dp), allocatable :: sorted_levels(:)

        if (n_features < 1 .or. n_outputs < 1 .or. size(levels) /= n_outputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile initialize: dimensions and level count are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(levels)) .or. any(levels <= 0.0_dp) .or. &
            any(levels >= 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile initialize: levels must be finite and strictly between zero and one")
            return
        end if
        include_intercept = .true.
        if (present(fit_intercept)) include_intercept = fit_intercept
        allocate(sorted_levels(n_outputs))
        call sort_levels(levels, sorted_levels, status)
        if (.not. status_ok(status)) return
        n_parameters = n_features + merge(1, 0, include_intercept)
        allocate(self%coefficient(n_parameters, n_outputs))
        self%coefficient = 0.0_dp
        call move_alloc(sorted_levels, self%quantile_level_value)
        self%feature_count_value = n_features
        self%output_count_value = n_outputs
        self%l2_value = 0.0_dp
        self%fit_intercept_value = include_intercept
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_initialize

    subroutine quantile_fit_matrix(self, x, y, status, quantile_levels, l2, &
            fit_intercept, sample_weight, max_iterations, tolerance)
        class(quantile_regression_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: quantile_levels(:), l2, sample_weight(:), &
            tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), allocatable :: levels(:), sorted_targets(:, :)
        integer, allocatable :: order(:)
        integer :: i, iterations
        real(dp) :: requested_l2, requested_tolerance
        logical :: include_intercept
        type(quantile_regression_t), target :: candidate
        type(quantile_lbfgsb_options_t) :: options
        type(quantile_lbfgsb_result_t) :: result

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(y, 1) /= size(x, 1) .or. &
            size(y, 2) < 1 .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile fit: finite inputs and matching dimensions are required")
            return
        end if
        allocate(levels(size(y, 2)))
        levels = QUANTILE_DEFAULT_LEVEL
        if (present(quantile_levels)) then
            if (size(quantile_levels) /= size(y, 2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "quantile fit: level count must match target columns")
                return
            end if
            levels = quantile_levels
        end if
        if (any(.not. ieee_is_finite(levels)) .or. any(levels <= 0.0_dp) .or. &
            any(levels >= 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile fit: levels must be finite and strictly between zero and one")
            return
        end if
        allocate(order(size(levels)), sorted_targets(size(y, 1), size(y, 2)))
        call sort_levels_with_order(levels, order, status)
        if (.not. status_ok(status)) return
        do i = 1, size(order)
            sorted_targets(:, i) = y(:, order(i))
        end do
        include_intercept = .true.
        if (present(fit_intercept)) include_intercept = fit_intercept
        call candidate%initialize(size(x, 2), size(y, 2), levels(order), status, &
            include_intercept)
        if (.not. status_ok(status)) return
        requested_l2 = 0.0_dp
        if (present(l2)) requested_l2 = l2
        requested_tolerance = options%gradient_tolerance
        if (present(tolerance)) requested_tolerance = tolerance
        iterations = options%max_iterations
        if (present(max_iterations)) iterations = max_iterations
        if (.not. ieee_is_finite(requested_l2) .or. requested_l2 < 0.0_dp .or. &
            iterations < 1 .or. .not. ieee_is_finite(requested_tolerance) .or. &
            requested_tolerance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile fit: L2, iteration limit, or tolerance is invalid")
            return
        end if
        options%l2 = requested_l2
        options%max_iterations = iterations
        options%gradient_tolerance = requested_tolerance
        call quantile_optimize_lbfgsb(candidate, x, sorted_targets, options, result, &
            status, sample_weight)
        if (status_ok(status)) then
            candidate%l2_value = requested_l2
            call commit_quantile_state(self, candidate)
        end if
    end subroutine quantile_fit_matrix

    subroutine quantile_fit_vector(self, x, y, status, quantile_level, l2, &
            fit_intercept, sample_weight, max_iterations, tolerance)
        class(quantile_regression_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: quantile_level, l2, sample_weight(:), &
            tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp) :: levels(1), requested_level
        real(dp), allocatable :: target(:, :)

        requested_level = QUANTILE_DEFAULT_LEVEL
        if (present(quantile_level)) requested_level = quantile_level
        levels(1) = requested_level
        allocate(target(size(y), 1))
        target(:, 1) = y
        call self%fit_matrix(x, target, status, levels, l2, fit_intercept, &
            sample_weight, max_iterations, tolerance)
    end subroutine quantile_fit_vector

    subroutine commit_quantile_state(self, candidate)
        class(quantile_regression_t), intent(inout) :: self
        type(quantile_regression_t), intent(inout) :: candidate

        if (allocated(self%coefficient)) deallocate(self%coefficient)
        if (allocated(self%quantile_level_value)) then
            deallocate(self%quantile_level_value)
        end if
        call move_alloc(candidate%coefficient, self%coefficient)
        call move_alloc(candidate%quantile_level_value, self%quantile_level_value)
        self%feature_count_value = candidate%feature_count_value
        self%output_count_value = candidate%output_count_value
        self%l2_value = candidate%l2_value
        self%fit_intercept_value = candidate%fit_intercept_value
        self%fitted_value = candidate%fitted_value
    end subroutine commit_quantile_state

    subroutine quantile_predict_matrix(self, x, y, status)
        class(quantile_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, o, offset

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile predict: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile predict: inputs must be finite")
            return
        end if
        offset = merge(1, 0, self%fit_intercept_value)
        y = 0.0_dp
        do o = 1, self%output_count_value
            do i = 1, size(x, 1)
                if (offset == 1) y(i, o) = self%coefficient(1, o)
                do j = 1, self%feature_count_value
                    y(i, o) = y(i, o) + x(i, j)* &
                        self%coefficient(offset + j, o)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_predict_matrix

    subroutine quantile_predict_vector(self, x, y, status)
        class(quantile_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:, :)

        if (size(y) /= size(x, 1) .or. self%output_count_value /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile predict: vector output requires one quantile")
            return
        end if
        allocate(values(size(y), 1))
        call self%predict_matrix(x, values, status)
        if (status_ok(status)) y = values(:, 1)
    end subroutine quantile_predict_vector

    subroutine quantile_predict_device(self, device, x, y, status)
        class(quantile_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (device%kind == FORTML_DEVICE_CPU) then
            call self%predict_matrix(x, y, status)
        else
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "quantile prediction: CUDA kernel is not resident")
        end if
    end subroutine quantile_predict_device

    logical function quantile_device_supported(self, device_kind) result(supported)
        class(quantile_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function quantile_device_supported

    subroutine quantile_predict_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(quantile_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, o, offset
        real(dp), allocatable :: tangent(:, :)

        if (.not. self%fitted_value .or. size(x, 2) /= self%feature_count_value .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value]) .or. &
            any(shape(y_dot) /= shape(y)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile JVP: inputs and tangents must be finite")
            return
        end if
        allocate(tangent, mold=self%coefficient)
        tangent = reshape(theta_dot, shape(tangent))
        offset = merge(1, 0, self%fit_intercept_value)
        y = 0.0_dp
        y_dot = 0.0_dp
        do o = 1, self%output_count_value
            do i = 1, size(x, 1)
                if (offset == 1) then
                    y(i, o) = self%coefficient(1, o)
                    y_dot(i, o) = tangent(1, o)
                end if
                do j = 1, self%feature_count_value
                    y(i, o) = y(i, o) + x(i, j)*self%coefficient(offset+j, o)
                    y_dot(i, o) = y_dot(i, o) + x_dot(i, j)* &
                        self%coefficient(offset+j, o) + x(i, j)*tangent(offset+j, o)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_predict_jvp

    subroutine quantile_predict_vjp(self, x, u, theta_bar, x_bar, status)
        class(quantile_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coefficient_bar(:, :)
        integer :: i, j, o, offset

        if (.not. self%fitted_value .or. size(x, 2) /= self%feature_count_value .or. &
            any(shape(u) /= [size(x, 1), self%output_count_value]) .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(u))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(coefficient_bar, mold=self%coefficient)
        coefficient_bar = 0.0_dp
        offset = merge(1, 0, self%fit_intercept_value)
        do o = 1, self%output_count_value
            if (offset == 1) coefficient_bar(1, o) = sum(u(:, o))
            do j = 1, self%feature_count_value
                coefficient_bar(offset+j, o) = dot_product(x(:, j), u(:, o))
            end do
        end do
        theta_bar = reshape(coefficient_bar, [size(theta_bar)])
        x_bar = 0.0_dp
        do i = 1, size(x, 1)
            do j = 1, self%feature_count_value
                do o = 1, self%output_count_value
                    x_bar(i, j) = x_bar(i, j) + u(i, o)* &
                        self%coefficient(offset+j, o)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_predict_vjp

    function quantile_coefficients(self) result(values)
        class(quantile_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%coefficient)) then
            allocate(values, mold=self%coefficient)
            values = self%coefficient
        else
            allocate(values(0, 0))
        end if
    end function quantile_coefficients

    function quantile_parameters(self) result(values)
        class(quantile_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        allocate(values(self%parameter_count()))
        if (size(values) > 0) values = reshape(self%coefficient, [size(values)])
    end function quantile_parameters

    subroutine quantile_set_parameters(self, values, status)
        class(quantile_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted_value .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile set_parameters: model or packed values are invalid")
            return
        end if
        self%coefficient = reshape(values, shape(self%coefficient))
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_set_parameters

    integer function quantile_parameter_count(self) result(count)
        class(quantile_regression_t), intent(in) :: self

        if (allocated(self%coefficient)) then
            count = size(self%coefficient)
        else
            count = 0
        end if
    end function quantile_parameter_count

    integer function quantile_feature_count(self) result(count)
        class(quantile_regression_t), intent(in) :: self
        count = self%feature_count_value
    end function quantile_feature_count

    integer function quantile_output_count(self) result(count)
        class(quantile_regression_t), intent(in) :: self
        count = self%output_count_value
    end function quantile_output_count

    function quantile_levels_value(self) result(values)
        class(quantile_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%quantile_level_value)) then
            allocate(values, mold=self%quantile_level_value)
            values = self%quantile_level_value
        else
            allocate(values(0))
        end if
    end function quantile_levels_value

    real(dp) function quantile_regularization(self) result(value)
        class(quantile_regression_t), intent(in) :: self
        value = self%l2_value
    end function quantile_regularization

    logical function quantile_fit_intercept(self) result(value)
        class(quantile_regression_t), intent(in) :: self
        value = self%fit_intercept_value
    end function quantile_fit_intercept

    logical function quantile_fitted(self) result(value)
        class(quantile_regression_t), intent(in) :: self
        value = self%fitted_value
    end function quantile_fitted

    subroutine quantile_objective_initialize(self, model, x, targets, l2, status, &
            sample_weight, kink_tolerance, device_kind)
        class(quantile_training_objective_t), intent(out) :: self
        type(quantile_regression_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :), l2
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:), kink_tolerance
        integer, intent(in), optional :: device_kind
        type(parameter_block_t) :: block
        real(dp) :: mass
        integer :: requested_device

        requested_device = FORTML_DEVICE_CPU
        if (present(device_kind)) requested_device = device_kind
        if (requested_device == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "quantile objective: CUDA kernel is not resident")
            return
        end if
        if (.not. model%fitted() .or. size(x, 1) < 1 .or. &
            size(x, 2) /= model%feature_count() .or. &
            any(shape(targets) /= [size(x, 1), model%output_count()]) .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets)) .or. &
            .not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile objective: model, data, or L2 is invalid")
            return
        end if
        self%model => model
        self%l2 = l2
        self%kink_tolerance = QUANTILE_DEFAULT_KINK_TOLERANCE
        if (present(kink_tolerance)) self%kink_tolerance = kink_tolerance
        if (.not. ieee_is_finite(self%kink_tolerance) .or. &
            self%kink_tolerance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile objective: kink tolerance is invalid")
            return
        end if
        allocate(self%features, source=x)
        allocate(self%targets, source=targets)
        allocate(self%weights(size(x, 1)))
        self%weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "quantile objective: sample weights are invalid")
                return
            end if
            self%weights = sample_weight
        end if
        mass = sum(self%weights)
        if (.not. ieee_is_finite(mass) .or. mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile objective: sample weights have no positive mass")
            return
        end if
        self%weight_sum = mass
        call block%initialize("coefficients", model%parameter_count(), model, &
            quantile_parameter_get, quantile_parameter_set, status)
        if (.not. status_ok(status)) return
        call self%registry%clear()
        call self%registry%add(block, status)
    end subroutine quantile_objective_initialize

    integer function quantile_objective_parameter_count(self) result(count)
        class(quantile_training_objective_t), intent(in) :: self
        count = self%registry%parameter_count()
    end function quantile_objective_parameter_count

    function quantile_objective_parameters(self) result(values)
        class(quantile_training_objective_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        type(fortnum_status_t) :: status

        allocate(values(self%parameter_count()))
        if (size(values) > 0) call self%registry%pack(values, status)
    end function quantile_objective_parameters

    subroutine quantile_objective_value_gradient(self, parameters, value, gradient, &
            status)
        class(quantile_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coefficients(:), levels(:)
        real(dp) :: prediction, residual, loss, derivative, norm
        integer :: i, j, o, p, n_rows, offset, index

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile objective: adapter is not initialized")
            return
        end if
        p = self%model%parameter_count()
        offset = merge(1, 0, self%model%fit_intercept())
        n_rows = self%model%feature_count() + offset
        if (size(parameters) /= self%parameter_count() .or. &
            size(gradient) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile objective: parameter shape is invalid")
            return
        end if
        allocate(coefficients(p), levels(self%model%output_count()))
        coefficients = parameters
        levels = self%model%quantile_levels()
        call self%registry%unpack(coefficients, status)
        if (.not. status_ok(status)) return
        norm = self%weight_sum*real(self%model%output_count(), dp)
        do o = 1, self%model%output_count()
            do i = 1, size(self%features, 1)
                prediction = 0.0_dp
                do j = 1, self%model%feature_count()
                    prediction = prediction + self%features(i, j)* &
                        coefficients((o-1)*n_rows + offset + j)
                end do
                if (offset == 1) prediction = prediction + coefficients((o-1)*n_rows+1)
                residual = prediction - self%targets(i, o)
                if (self%smoothing > 0.0_dp .and. &
                    residual >= self%smoothing) then
                    loss = levels(o)*residual
                    derivative = levels(o)
                else if (self%smoothing > 0.0_dp .and. &
                        residual <= -self%smoothing) then
                    loss = (levels(o)-1.0_dp)*residual
                    derivative = levels(o)-1.0_dp
                else if (self%smoothing > 0.0_dp .and. residual >= 0.0_dp) then
                    loss = levels(o)*(0.5_dp*residual**2/self%smoothing + &
                        0.5_dp*self%smoothing)
                    derivative = levels(o)*residual/self%smoothing
                else if (self%smoothing > 0.0_dp) then
                    loss = (1.0_dp-levels(o))*(0.5_dp*residual**2/&
                        self%smoothing + 0.5_dp*self%smoothing)
                    derivative = (1.0_dp-levels(o))*residual/self%smoothing
                else if (residual > self%kink_tolerance) then
                    loss = levels(o)*residual
                    derivative = levels(o)
                else if (residual < -self%kink_tolerance) then
                    loss = (levels(o)-1.0_dp)*residual
                    derivative = levels(o)-1.0_dp
                else
                    loss = levels(o)*max(residual, 0.0_dp) + &
                        (levels(o)-1.0_dp)*min(residual, 0.0_dp)
                    derivative = 0.0_dp
                end if
                value = value + self%weights(i)*loss/norm
                do j = 1, n_rows
                    index = (o-1)*n_rows+j
                    if (j == 1 .and. offset == 1) then
                        gradient(index) = gradient(index) + &
                            self%weights(i)*derivative/norm
                    else
                        gradient(index) = gradient(index) + self%weights(i)* &
                            derivative*self%features(i, j-offset)/norm
                    end if
                end do
            end do
        end do
        do o = 1, self%model%output_count()
            do j = 1+offset, n_rows
                index = (o-1)*n_rows+j
                value = value + 0.5_dp*self%l2*coefficients(index)**2
                gradient(index) = gradient(index) + self%l2*coefficients(index)
            end do
        end do
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile objective: value or gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_objective_value_gradient

    subroutine quantile_objective_jvp(self, parameters, direction, value, tangent, &
            status)
        class(quantile_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)

        if (size(direction) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile JVP: direction shape is invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (status_ok(status)) tangent = dot_product(gradient, direction)
    end subroutine quantile_objective_jvp

    subroutine quantile_objective_vjp(self, parameters, value_bar, parameter_bar, &
            status)
        class(quantile_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)
        real(dp) :: value

        if (size(parameter_bar) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile VJP: output shape is invalid")
            return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (status_ok(status)) parameter_bar = value_bar*gradient
    end subroutine quantile_objective_vjp

    subroutine quantile_objective_hvp(self, parameters, direction, product, status)
        class(quantile_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coefficients(:), levels(:)
        real(dp) :: prediction, residual, norm
        integer :: i, j, o, p, n_rows, offset, index

        product = 0.0_dp
        if (.not. associated(self%model) .or. size(parameters) /= self%parameter_count() .or. &
            size(direction) /= size(parameters) .or. size(product) /= size(parameters) .or. &
            any(.not. ieee_is_finite(parameters)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile HVP: parameter or direction shape is invalid")
            return
        end if
        p = self%model%parameter_count()
        offset = merge(1, 0, self%model%fit_intercept())
        n_rows = self%model%feature_count() + offset
        allocate(coefficients(p), levels(self%model%output_count()))
        coefficients = parameters
        levels = self%model%quantile_levels()
        call self%registry%unpack(coefficients, status)
        if (.not. status_ok(status)) return
        norm = self%weight_sum*real(self%model%output_count(), dp)
        do o = 1, self%model%output_count()
            do i = 1, size(self%features, 1)
                prediction = 0.0_dp
                do j = 1, self%model%feature_count()
                    prediction = prediction + self%features(i, j)* &
                        coefficients((o-1)*n_rows+offset+j)
                end do
                if (offset == 1) prediction = prediction + coefficients((o-1)*n_rows+1)
                residual = prediction-self%targets(i, o)
                if (self%weights(i) > 0.0_dp .and. abs(residual) <= &
                    self%kink_tolerance) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "quantile HVP: residual lies on a pinball kink")
                    return
                end if
            end do
        end do
        do o = 1, self%model%output_count()
            do j = 1+offset, n_rows
                index = (o-1)*n_rows+j
                product(index) = self%l2*direction(index)
            end do
        end do
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_objective_hvp

    subroutine quantile_objective_fortopt(self, objective, status)
        class(quantile_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status

        call objective%initialize_context(self%parameter_count(), self, &
            quantile_objective_callback, status)
    end subroutine quantile_objective_fortopt

    subroutine quantile_objective_callback(context, parameters, value, gradient, &
            status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (adapter => context)
            type is (quantile_training_objective_t)
            call adapter%value_gradient(parameters, value, gradient, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile objective: callback context has wrong type")
        end select
    end subroutine quantile_objective_callback

    subroutine quantile_optimize_lbfgsb(model, x, targets, options, result, status, &
            sample_weight)
        type(quantile_regression_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :)
        type(quantile_lbfgsb_options_t), intent(in) :: options
        type(quantile_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        type(quantile_training_objective_t), target :: adapter
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        type(fortnum_status_t) :: stage_status
        real(dp), allocatable :: parameters(:), backup(:), lower(:), upper(:), &
            gradient(:)
        real(dp) :: schedule(3), continuation_scale
        integer :: n, stage, successful_stages
        type(quantile_lbfgsb_result_t) :: result_default

        result = result_default
        if (.not. valid_options(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile L-BFGS-B: options are invalid")
            return
        end if
        call adapter%initialize(model, x, targets, options%l2, status, &
            sample_weight, options%kink_tolerance, options%device_kind)
        if (.not. status_ok(status)) return
        n = adapter%parameter_count()
        allocate(parameters(n), backup(n), lower(n), upper(n), gradient(n))
        lower = options%lower_bound
        upper = options%upper_bound
        optimizer_options%memory = options%memory
        optimizer_options%max_iterations = options%max_iterations
        optimizer_options%max_line_search = options%max_line_search
        optimizer_options%gradient_tolerance = options%gradient_tolerance
        optimizer_options%step_tolerance = options%step_tolerance
        optimizer_options%objective_tolerance = options%objective_tolerance
        ! The public objective remains exact pinball.  A descending C1
        ! continuation gives L-BFGS-B a deterministic Armijo callback at a
        ! residual kink; if a later, sharper stage hits the nonsmooth boundary,
        ! retain the last successful stage and report its width explicitly.
        continuation_scale = max(1.0_dp, 2.0_dp*maxval(abs(targets)))
        schedule = [max(options%fit_smoothing, continuation_scale), &
            max(1.0e-4_dp, 0.1_dp*max(options%fit_smoothing, continuation_scale)), &
            max(1.0e-5_dp, 0.01_dp*max(options%fit_smoothing, continuation_scale))]
        if (options%fit_smoothing == 0.0_dp) schedule = 0.0_dp
        successful_stages = 0
        result%iterations = 0
        result%line_search_evaluations = 0
        do stage = 1, 3
            if (schedule(stage) == 0.0_dp .and. stage > 1) exit
            backup = adapter%parameters()
            parameters = backup
            adapter%smoothing = schedule(stage)
            call adapter%fortopt(objective, stage_status)
            if (.not. status_ok(stage_status)) then
                if (successful_stages == 0) then
                    status = stage_status
                    return
                end if
                call model%set_parameters(backup, status)
                exit
            end if
            call optimizer%minimize(objective, parameters, lower, upper, &
                optimizer_options, optimizer_result, stage_status)
            if (.not. status_ok(stage_status)) then
                call model%set_parameters(backup, status)
                if (successful_stages == 0) then
                    status = stage_status
                    return
                end if
                exit
            end if
            call model%set_parameters(parameters, status)
            if (.not. status_ok(status)) return
            successful_stages = successful_stages + 1
            result%gradient_norm = optimizer_result%state%gradient_norm
            result%fit_smoothing = schedule(stage)
            result%iterations = result%iterations + optimizer_result%state%iteration
            result%line_search_evaluations = result%line_search_evaluations + &
                optimizer_result%line_search_evaluations
            if (.not. optimizer_result%state%converged .and. stage == 3) exit
        end do
        if (successful_stages == 0) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "quantile L-BFGS-B: no continuation stage converged")
            return
        end if
        parameters = model%parameters()
        adapter%smoothing = 0.0_dp
        call adapter%value_gradient(parameters, result%objective, gradient, status)
        if (.not. status_ok(status)) return
        result%exact_gradient_norm = sqrt(sum(gradient*gradient))
        result%converged = successful_stages > 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_optimize_lbfgsb

    logical function valid_options(options) result(valid)
        type(quantile_lbfgsb_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. options%lower_bound < &
            options%upper_bound .and. options%l2 >= 0.0_dp
        if (options%device_kind /= FORTML_DEVICE_CPU .and. &
            options%device_kind /= FORTML_DEVICE_CUDA) valid = .false.
        if (.not. ieee_is_finite(options%gradient_tolerance) .or. &
            .not. ieee_is_finite(options%step_tolerance) .or. &
            .not. ieee_is_finite(options%objective_tolerance) .or. &
            .not. ieee_is_finite(options%lower_bound) .or. &
            .not. ieee_is_finite(options%upper_bound) .or. &
            .not. ieee_is_finite(options%l2) .or. &
            .not. ieee_is_finite(options%kink_tolerance) .or. &
            .not. ieee_is_finite(options%fit_smoothing)) valid = .false.
        if (options%gradient_tolerance < 0.0_dp .or. &
            options%step_tolerance < 0.0_dp .or. &
            options%objective_tolerance < 0.0_dp .or. &
            options%kink_tolerance < 0.0_dp .or. &
            options%fit_smoothing < 0.0_dp) valid = .false.
    end function valid_options

    subroutine sort_levels(levels, sorted, status)
        real(dp), intent(in) :: levels(:)
        real(dp), intent(out) :: sorted(:)
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: order(:)

        allocate(order(size(levels)))
        call sort_levels_with_order(levels, order, status)
        if (.not. status_ok(status)) return
        sorted = levels(order)
    end subroutine sort_levels

    subroutine sort_levels_with_order(levels, order, status)
        real(dp), intent(in) :: levels(:)
        integer, intent(out) :: order(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, key

        if (size(order) /= size(levels) .or. size(levels) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile levels: order shape is invalid")
            return
        end if
        order = [(i, i=1, size(levels))]
        do i = 2, size(order)
            key = order(i)
            j = i - 1
            do while (j >= 1)
                if (levels(order(j)) <= levels(key)) exit
                order(j+1) = order(j)
                j = j - 1
            end do
            order(j+1) = key
        end do
        do i = 2, size(order)
            if (levels(order(i)) == levels(order(i-1))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "quantile levels: duplicate levels are ambiguous")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sort_levels_with_order

    subroutine quantile_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (quantile_regression_t)
            if (size(values) /= model%parameter_count()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "quantile registry getter: shape is invalid")
                return
            end if
            values = reshape(model%coefficient, [size(values)])
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile registry getter: context type is invalid")
        end select
    end subroutine quantile_parameter_get

    subroutine quantile_parameter_set(context, values, status)
        class(*), pointer, intent(inout) :: context
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        select type (model => context)
            type is (quantile_regression_t)
            call model%set_parameters(values, status)
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "quantile registry setter: context type is invalid")
        end select
    end subroutine quantile_parameter_set

end module fortml_quantile_regression
