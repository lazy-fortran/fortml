module fortml_huber_regression
    !! Weighted linear Huber regression and its packed objective products.
    !!
    !! The coefficient block is owned by a `parameter_registry_t`.  The
    !! objective may append bounded L2 and Huber-delta coordinates, so the
    !! same value/gradient path is usable by FortOpt and outer searches.  The
    !! Hessian-vector product is exact for a fixed residual branch; a typed
    !! refusal is returned at a Huber kink rather than inventing a second
    !! derivative.
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

    real(dp), parameter :: HUBER_DEFAULT_DELTA = 1.0_dp
    real(dp), parameter :: HUBER_DEFAULT_BOUND = 30.0_dp
    real(dp), parameter :: HUBER_DEFAULT_KINK_TOLERANCE = 1.0e-10_dp

    type, public :: huber_regression_t
        private
        real(dp), allocatable :: coefficient(:, :)
        integer :: feature_count_value = 0
        integer :: output_count_value = 0
        real(dp) :: delta_value = HUBER_DEFAULT_DELTA
        real(dp) :: l2_value = 0.0_dp
        logical :: fit_intercept_value = .true.
        logical :: fitted_value = .false.
    contains
        procedure, public :: initialize => huber_initialize
        procedure, public :: fit_matrix => huber_fit_matrix
        procedure, public :: fit_vector => huber_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: predict_matrix => huber_predict_matrix
        procedure, public :: predict_vector => huber_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_device => huber_predict_device
        procedure, public :: device_supported => huber_device_supported
        procedure, public :: predict_jvp => huber_predict_jvp
        procedure, public :: predict_vjp => huber_predict_vjp
        procedure, public :: jvp => huber_predict_jvp
        procedure, public :: vjp => huber_predict_vjp
        procedure, public :: coefficients => huber_coefficients
        procedure, public :: parameters => huber_parameters
        procedure, public :: set_parameters => huber_set_parameters
        procedure, public :: parameter_count => huber_parameter_count
        procedure, public :: feature_count => huber_feature_count
        procedure, public :: output_count => huber_output_count
        procedure, public :: delta => huber_delta
        procedure, public :: regularization => huber_regularization
        procedure, public :: fit_intercept => huber_fit_intercept
        procedure, public :: fitted => huber_fitted
    end type huber_regression_t

    type, public :: huber_training_objective_t
        private
        type(huber_regression_t), pointer :: model => null()
        type(parameter_registry_t) :: registry
        real(dp), allocatable :: features(:, :), targets(:, :), weights(:)
        real(dp) :: weight_sum = 0.0_dp
        real(dp) :: delta = HUBER_DEFAULT_DELTA
        real(dp) :: l2 = 0.0_dp
        real(dp) :: kink_tolerance = HUBER_DEFAULT_KINK_TOLERANCE
        logical :: optimize_l2 = .false.
        logical :: optimize_delta = .false.
    contains
        procedure, public :: initialize => huber_objective_initialize
        procedure, public :: parameter_count => huber_objective_parameter_count
        procedure, public :: parameters => huber_objective_parameters
        procedure, public :: value_gradient => huber_objective_value_gradient
        procedure, public :: jvp => huber_objective_jvp
        procedure, public :: vjp => huber_objective_vjp
        procedure, public :: hvp => huber_objective_hvp
        procedure, public :: fortopt => huber_objective_fortopt
    end type huber_training_objective_t

    type, public :: huber_lbfgsb_options_t
        integer :: memory = 10
        integer :: max_iterations = 200
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -HUBER_DEFAULT_BOUND
        real(dp) :: upper_bound = HUBER_DEFAULT_BOUND
        real(dp) :: l2 = 0.0_dp
        real(dp) :: l2_lower_bound = 0.0_dp
        real(dp) :: l2_upper_bound = 20.0_dp
        real(dp) :: delta = HUBER_DEFAULT_DELTA
        real(dp) :: delta_lower_bound = 1.0e-6_dp
        real(dp) :: delta_upper_bound = 20.0_dp
        real(dp) :: kink_tolerance = HUBER_DEFAULT_KINK_TOLERANCE
        logical :: optimize_l2 = .false.
        logical :: optimize_delta = .false.
        integer :: device_kind = FORTML_DEVICE_CPU
    end type huber_lbfgsb_options_t

    type, public :: huber_lbfgsb_result_t
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: objective = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: l2 = 0.0_dp
        real(dp) :: delta = HUBER_DEFAULT_DELTA
    end type huber_lbfgsb_result_t

    public :: huber_optimize_lbfgsb

contains

    subroutine huber_initialize(self, n_features, n_outputs, status, fit_intercept)
        class(huber_regression_t), intent(out) :: self
        integer, intent(in) :: n_features, n_outputs
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fit_intercept
        logical :: intercept
        integer :: n_parameters

        intercept = .true.
        if (present(fit_intercept)) intercept = fit_intercept
        if (n_features < 1 .or. n_outputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber initialize: dimensions must be positive")
            return
        end if
        n_parameters = n_features + merge(1, 0, intercept)
        allocate(self%coefficient(n_parameters, n_outputs))
        self%coefficient = 0.0_dp
        self%feature_count_value = n_features
        self%output_count_value = n_outputs
        self%fit_intercept_value = intercept
        self%delta_value = HUBER_DEFAULT_DELTA
        self%l2_value = 0.0_dp
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_initialize

    subroutine huber_fit_matrix(self, x, y, status, delta, l2, fit_intercept, &
            sample_weight, max_iterations, tolerance)
        class(huber_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: delta, l2, sample_weight(:), tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        type(huber_lbfgsb_options_t) :: options
        type(huber_lbfgsb_result_t) :: result
        integer :: iterations
        real(dp) :: requested_delta, requested_l2, requested_tolerance
        logical :: intercept

        requested_delta = HUBER_DEFAULT_DELTA
        if (present(delta)) requested_delta = delta
        requested_l2 = 0.0_dp
        if (present(l2)) requested_l2 = l2
        requested_tolerance = options%gradient_tolerance
        if (present(tolerance)) requested_tolerance = tolerance
        iterations = options%max_iterations
        if (present(max_iterations)) iterations = max_iterations
        intercept = .true.
        if (present(fit_intercept)) intercept = fit_intercept
        call self%initialize(size(x, 2), size(y, 2), status, intercept)
        if (.not. status_ok(status)) return
        options%delta = requested_delta
        options%l2 = requested_l2
        options%max_iterations = iterations
        options%gradient_tolerance = requested_tolerance
        call huber_optimize_lbfgsb(self, x, y, options, result, status, &
            sample_weight)
    end subroutine huber_fit_matrix

    subroutine huber_fit_vector(self, x, y, status, delta, l2, fit_intercept, &
            sample_weight, max_iterations, tolerance)
        class(huber_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: delta, l2, sample_weight(:), tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), allocatable :: target(:, :)

        allocate(target(size(y), 1))
        target(:, 1) = y
        call self%fit_matrix(x, target, status, delta, l2, fit_intercept, &
            sample_weight, max_iterations, tolerance)
    end subroutine huber_fit_vector

    subroutine huber_predict_matrix(self, x, y, status)
        class(huber_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :)

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber predict: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber predict: inputs must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        call make_design(x, self%fit_intercept_value, design)
        y = matmul(design, self%coefficient)
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_predict_matrix

    subroutine huber_predict_vector(self, x, y, status)
        class(huber_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:, :)
        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber predict: output shape is invalid")
            return
        end if
        allocate(values(size(y), 1))
        call self%predict_matrix(x, values, status)
        if (status_ok(status)) y = values(:, 1)
    end subroutine huber_predict_vector

    subroutine huber_predict_device(self, device, x, y, status)
        class(huber_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (device%kind == FORTML_DEVICE_CPU) then
            call self%predict_matrix(x, y, status)
        else
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "Huber prediction: CUDA kernel is not resident")
        end if
    end subroutine huber_predict_device

    logical function huber_device_supported(self, device_kind) result(supported)
        class(huber_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind
        supported = device_kind == FORTML_DEVICE_CPU
    end function huber_device_supported

    subroutine huber_predict_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(huber_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), design_dot(:, :), coefficient_dot(:, :)
        if (.not. self%fitted_value .or. size(x, 2) /= self%feature_count_value .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value]) .or. &
            any(shape(y_dot) /= shape(y)) .or. size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber JVP: inputs and tangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(design_dot, mold=design)
        allocate(coefficient_dot, mold=self%coefficient)
        call make_design(x, self%fit_intercept_value, design)
        call make_tangent_design(x_dot, self%fit_intercept_value, design_dot)
        coefficient_dot = reshape(theta_dot, shape(coefficient_dot))
        y = matmul(design, self%coefficient)
        y_dot = matmul(design_dot, self%coefficient) + matmul(design, coefficient_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_predict_jvp

    subroutine huber_predict_vjp(self, x, u, theta_bar, x_bar, status)
        class(huber_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), coefficient_bar(:, :)
        integer :: j, offset
        if (.not. self%fitted_value .or. size(x, 2) /= self%feature_count_value .or. &
            any(shape(u) /= [size(x, 1), self%output_count_value]) .or. &
            size(theta_bar) /= self%parameter_count() .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(u))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(coefficient_bar, mold=self%coefficient)
        call make_design(x, self%fit_intercept_value, design)
        coefficient_bar = matmul(transpose(design), u)
        theta_bar = reshape(coefficient_bar, [size(theta_bar)])
        x_bar = 0.0_dp
        offset = merge(1, 0, self%fit_intercept_value)
        do j = 1, self%feature_count_value
            x_bar(:, j) = matmul(u, self%coefficient(offset + j, :))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_predict_vjp

    function huber_coefficients(self) result(values)
        class(huber_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        if (allocated(self%coefficient)) then
            allocate(values, mold=self%coefficient); values = self%coefficient
        else
            allocate(values(0, 0))
        end if
    end function huber_coefficients

    function huber_parameters(self) result(values)
        class(huber_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        allocate(values(self%parameter_count()))
        if (size(values) > 0) values = reshape(self%coefficient, [size(values)])
    end function huber_parameters

    subroutine huber_set_parameters(self, values, status)
        class(huber_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        if (.not. self%fitted_value .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber set_parameters: model or packed values are invalid")
            return
        end if
        self%coefficient = reshape(values, shape(self%coefficient))
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_set_parameters

    integer function huber_parameter_count(self) result(count)
        class(huber_regression_t), intent(in) :: self
        if (allocated(self%coefficient)) then
            count = size(self%coefficient)
        else
            count = 0
        end if
    end function huber_parameter_count
    integer function huber_feature_count(self) result(count)
        class(huber_regression_t), intent(in) :: self; count = self%feature_count_value
    end function huber_feature_count
    integer function huber_output_count(self) result(count)
        class(huber_regression_t), intent(in) :: self; count = self%output_count_value
    end function huber_output_count
    real(dp) function huber_delta(self) result(value)
        class(huber_regression_t), intent(in) :: self; value = self%delta_value
    end function huber_delta
    real(dp) function huber_regularization(self) result(value)
        class(huber_regression_t), intent(in) :: self; value = self%l2_value
    end function huber_regularization
    logical function huber_fit_intercept(self) result(value)
        class(huber_regression_t), intent(in) :: self; value = self%fit_intercept_value
    end function huber_fit_intercept
    logical function huber_fitted(self) result(value)
        class(huber_regression_t), intent(in) :: self; value = self%fitted_value
    end function huber_fitted

    subroutine huber_objective_initialize(self, model, x, targets, delta, l2, status, &
            optimize_l2, optimize_delta, sample_weight, kink_tolerance, device_kind)
        class(huber_training_objective_t), intent(out) :: self
        type(huber_regression_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :), delta, l2
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: optimize_l2, optimize_delta
        real(dp), intent(in), optional :: sample_weight(:), kink_tolerance
        integer, intent(in), optional :: device_kind
        type(parameter_block_t) :: block
        real(dp) :: mass
        integer :: requested_device
        requested_device = FORTML_DEVICE_CPU
        if (present(device_kind)) requested_device = device_kind
        if (requested_device == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "Huber objective: CUDA kernel is not resident")
            return
        end if
        if (.not. model%fitted() .or. size(x, 1) < 1 .or. size(x, 2) /= model%feature_count() .or. &
            any(shape(targets) /= [size(x, 1), model%output_count()]) .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets)) .or. &
            .not. ieee_is_finite(delta) .or. delta <= 0.0_dp .or. .not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber objective: model, data, delta, or L2 is invalid")
            return
        end if
        self%model => model; self%delta = delta; self%l2 = l2
        self%optimize_l2 = .false.; if (present(optimize_l2)) self%optimize_l2 = optimize_l2
        self%optimize_delta = .false.; if (present(optimize_delta)) self%optimize_delta = optimize_delta
        self%kink_tolerance = HUBER_DEFAULT_KINK_TOLERANCE
        if (present(kink_tolerance)) self%kink_tolerance = kink_tolerance
        if (.not. ieee_is_finite(self%kink_tolerance) .or. self%kink_tolerance < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber objective: kink tolerance is invalid")
            return
        end if
        allocate(self%features, source=x); allocate(self%targets, source=targets)
        allocate(self%weights(size(x, 1))); self%weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1) .or. any(.not. ieee_is_finite(sample_weight)) .or. any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber objective: sample weights are invalid"); return
            end if
            self%weights = sample_weight
        end if
        mass = sum(self%weights)
        if (.not. ieee_is_finite(mass) .or. mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber objective: sample weights have no positive mass"); return
        end if
        self%weight_sum = mass
        call block%initialize("coefficients", model%parameter_count(), model, &
            huber_parameter_get, huber_parameter_set, status)
        if (.not. status_ok(status)) return
        call self%registry%clear(); call self%registry%add(block, status)
    end subroutine huber_objective_initialize

    integer function huber_objective_parameter_count(self) result(count)
        class(huber_training_objective_t), intent(in) :: self
        count = self%registry%parameter_count()
        if (self%optimize_l2) count = count + 1
        if (self%optimize_delta) count = count + 1
    end function huber_objective_parameter_count

    function huber_objective_parameters(self) result(values)
        class(huber_training_objective_t), intent(in) :: self
        real(dp), allocatable :: values(:), coefficients(:)
        type(fortnum_status_t) :: status
        integer :: n
        n = self%registry%parameter_count(); allocate(values(self%parameter_count())); values = 0.0_dp
        if (n > 0) then
            allocate(coefficients(n)); call self%registry%pack(coefficients, status)
            if (status_ok(status)) values(:n) = coefficients
        end if
        if (self%optimize_l2) values(n + 1) = self%l2
        if (self%optimize_delta) values(n + 1 + merge(1, 0, self%optimize_l2)) = self%delta
    end function huber_objective_parameters

    subroutine huber_objective_value_gradient(self, parameters, value, gradient, status)
        class(huber_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coefficients(:)
        real(dp) :: delta, l2, residual, prediction, loss, derivative, norm
        integer :: i, j, o, p, n, offset, index
        value = 0.0_dp; gradient = 0.0_dp
        if (.not. associated(self%model)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber objective: adapter is not initialized"); return
        end if
        n = self%registry%parameter_count(); p = self%model%parameter_count()
        if (size(parameters) /= self%parameter_count() .or. size(gradient) /= size(parameters) .or. any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber objective: parameter shape is invalid"); return
        end if
        allocate(coefficients(p)); coefficients = parameters(:p)
        call self%registry%unpack(coefficients, status); if (.not. status_ok(status)) return
        l2 = self%l2; if (self%optimize_l2) l2 = parameters(n + 1)
        delta = self%delta; if (self%optimize_delta) delta = parameters(n + 1 + merge(1, 0, self%optimize_l2))
        if (.not. ieee_is_finite(l2) .or. l2 < 0.0_dp .or. .not. ieee_is_finite(delta) .or. delta <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber objective: packed hyperparameters are invalid"); return
        end if
        norm = self%weight_sum*real(self%model%output_count(), dp)
        offset = merge(1, 0, self%model%fit_intercept())
        do o = 1, self%model%output_count()
            do i = 1, size(self%features, 1)
                prediction = 0.0_dp
                if (offset == 1) prediction = coefficients((o-1)*p+1)
                do j = 1, self%model%feature_count()
                    prediction = prediction + self%features(i,j)* &
                        coefficients((o-1)*p+offset+j)
                end do
                residual = prediction - self%targets(i, o)
                if (abs(residual) <= delta) then
                    loss = 0.5_dp*residual*residual; derivative = residual
                else
                    loss = delta*(abs(residual)-0.5_dp*delta); derivative = delta*sign(1.0_dp, residual)
                end if
                value = value + self%weights(i)*loss/norm
                do j = 1, p
                    index = (o-1)*p+j
                    if (j == 1 .and. offset == 1) then
                        gradient(index) = gradient(index) + self%weights(i)*derivative/norm
                    else
                        gradient(index) = gradient(index) + self%weights(i)*derivative*self%features(i,j-offset)/norm
                    end if
                end do
                if (self%optimize_delta .and. abs(residual) > delta) gradient(n + 1 + merge(1,0,self%optimize_l2)) = &
                    gradient(n + 1 + merge(1,0,self%optimize_l2)) + self%weights(i)*(abs(residual)-delta)/norm
            end do
        end do
        do o = 1, self%model%output_count()
            do j = 1+offset, p
                index = (o-1)*p+j; value = value + 0.5_dp*l2*coefficients(index)**2
                gradient(index) = gradient(index) + l2*coefficients(index)
            end do
        end do
        if (self%optimize_l2) then
            gradient(n+1) = 0.0_dp
            do o = 1, self%model%output_count()
                gradient(n+1) = gradient(n+1) + 0.5_dp*sum(&
                    coefficients((o-1)*p+1+offset:o*p)**2)
            end do
        end if
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber objective: value or gradient is not finite"); return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_objective_value_gradient

    subroutine huber_objective_jvp(self, parameters, direction, value, tangent, status)
        class(huber_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:)
        if (size(direction) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber JVP: direction shape is invalid"); return
        end if
        allocate(gradient(size(parameters)))
        call self%value_gradient(parameters, value, gradient, status)
        if (status_ok(status)) tangent = dot_product(gradient, direction)
    end subroutine huber_objective_jvp

    subroutine huber_objective_vjp(self, parameters, value_bar, parameter_bar, status)
        class(huber_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), value_bar
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient(:); real(dp) :: value
        if (size(parameter_bar) /= size(parameters)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber VJP: output shape is invalid"); return
        end if
        allocate(gradient(size(parameters))); call self%value_gradient(parameters, value, gradient, status)
        if (status_ok(status)) parameter_bar = value_bar*gradient
    end subroutine huber_objective_vjp

    subroutine huber_objective_hvp(self, parameters, direction, product, status)
        class(huber_training_objective_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:), direction(:)
        real(dp), intent(out) :: product(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coefficients(:)
        real(dp) :: delta, l2, residual, prediction, pred_dot, curvature, sign_r, norm, delta_dot, l2_dot
        integer :: i,j,o,p,n,offset,index,delta_index
        product = 0.0_dp
        if (.not. associated(self%model) .or. size(parameters) /= self%parameter_count() .or. &
            size(direction) /= size(parameters) .or. size(product) /= size(parameters) .or. any(.not. ieee_is_finite(parameters)) .or. any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber HVP: parameter or direction shape is invalid"); return
        end if
        n = self%registry%parameter_count(); p = self%model%parameter_count(); allocate(coefficients(p)); coefficients = parameters(:p)
        call self%registry%unpack(coefficients, status); if (.not. status_ok(status)) return
        l2 = self%l2; l2_dot = 0.0_dp; if (self%optimize_l2) then; l2=parameters(n+1); l2_dot=direction(n+1); end if
        delta_index = n + 1 + merge(1,0,self%optimize_l2); delta=self%delta; delta_dot=0.0_dp
        if (self%optimize_delta) then; delta=parameters(delta_index); delta_dot=direction(delta_index); end if
        if (.not. ieee_is_finite(delta) .or. delta <= 0.0_dp .or. .not. ieee_is_finite(l2) .or. l2 < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber HVP: packed hyperparameters are invalid"); return
        end if
        norm = self%weight_sum*real(self%model%output_count(), dp); offset=merge(1,0,self%model%fit_intercept())
        do o=1,self%model%output_count(); do i=1,size(self%features,1)
            prediction=0.0_dp; pred_dot=0.0_dp
            do j=1,p
                if (j==1 .and. offset==1) then
                    prediction=prediction+coefficients((o-1)*p+j); pred_dot=pred_dot+direction((o-1)*p+j)
                else
                    prediction=prediction+self%features(i,j-offset)*coefficients((o-1)*p+j); pred_dot=pred_dot+self%features(i,j-offset)*direction((o-1)*p+j)
                end if
            end do
            residual=prediction-self%targets(i,o)
            if (abs(abs(residual)-delta) <= self%kink_tolerance*max(1.0_dp,delta)) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, "Huber HVP: residual lies on a kink"); return
            end if
            if (abs(residual) < delta) then
                curvature = 1.0_dp; sign_r = 0.0_dp
            else
                curvature = 0.0_dp; sign_r = sign(1.0_dp,residual)
            end if
            do j=1,p
                index=(o-1)*p+j
                if (j==1 .and. offset==1) then
                    product(index)=product(index)+self%weights(i)/norm*(curvature*pred_dot + merge(delta_dot*sign_r,0.0_dp,self%optimize_delta))
                else
                    product(index)=product(index)+self%weights(i)/norm*(curvature*pred_dot + merge(delta_dot*sign_r,0.0_dp,self%optimize_delta))*self%features(i,j-offset)
                end if
            end do
            if (self%optimize_delta .and. curvature==0.0_dp) product(delta_index)=product(delta_index)+self%weights(i)/norm*(sign_r*pred_dot-delta_dot)
        end do; end do
        do o=1,self%model%output_count(); do j=1+offset,p
            index=(o-1)*p+j; product(index)=product(index)+l2*direction(index)+l2_dot*coefficients(index)
        end do; end do
        if (self%optimize_l2) then
            product(n+1) = 0.0_dp
            do o = 1, self%model%output_count()
                product(n+1) = product(n+1) + dot_product(&
                    coefficients((o-1)*p+1+offset:o*p), &
                    direction((o-1)*p+1+offset:o*p))
            end do
        end if
        if (any(.not. ieee_is_finite(product))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "Huber HVP: product is not finite"); return
        end if
        call status_set(status,FORTNUM_OK,"")
    end subroutine huber_objective_hvp

    subroutine huber_objective_fortopt(self, objective, status)
        class(huber_training_objective_t), target, intent(inout) :: self
        type(objective_t), intent(out) :: objective
        type(fortnum_status_t), intent(out) :: status
        call objective%initialize_context(self%parameter_count(), self, huber_objective_callback, status)
    end subroutine huber_objective_fortopt

    subroutine huber_objective_callback(context, parameters, value, gradient, status)
        class(*), intent(inout) :: context; real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:); type(fortnum_status_t), intent(out) :: status
        select type(adapter=>context); type is(huber_training_objective_t)
            call adapter%value_gradient(parameters,value,gradient,status)
        class default; call status_set(status,FORTNUM_DOMAIN_ERROR,"Huber objective: callback context has wrong type")
        end select
    end subroutine huber_objective_callback

    subroutine huber_optimize_lbfgsb(model, x, targets, options, result, status, sample_weight)
        type(huber_regression_t), target, intent(inout) :: model
        real(dp), intent(in) :: x(:, :), targets(:, :)
        type(huber_lbfgsb_options_t), intent(in) :: options
        type(huber_lbfgsb_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        type(huber_training_objective_t), target :: adapter
        type(objective_t) :: objective; type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options; type(lbfgsb_result_t) :: optimizer_result
        real(dp), allocatable :: parameters(:), lower(:), upper(:), gradient(:)
        integer :: n,p,nparams; result=huber_lbfgsb_result_t()
        if (.not. valid_options(options)) then; call status_set(status,FORTNUM_DOMAIN_ERROR,"Huber L-BFGS-B: options are invalid"); return; end if
        call adapter%initialize(model,x,targets,options%delta,options%l2,status,optimize_l2=options%optimize_l2,optimize_delta=options%optimize_delta,sample_weight=sample_weight,kink_tolerance=options%kink_tolerance,device_kind=options%device_kind)
        if (.not. status_ok(status)) return
        n=model%parameter_count(); p=adapter%parameter_count(); parameters=adapter%parameters(); allocate(lower(p),upper(p),gradient(p)); lower(:n)=options%lower_bound; upper(:n)=options%upper_bound
        nparams=n; if(options%optimize_l2)then; lower(nparams+1)=options%l2_lower_bound;upper(nparams+1)=options%l2_upper_bound;nparams=nparams+1;end if
        if(options%optimize_delta)then; lower(nparams+1)=options%delta_lower_bound;upper(nparams+1)=options%delta_upper_bound;end if
        call adapter%fortopt(objective,status); if(.not.status_ok(status))return
        optimizer_options%memory=options%memory;optimizer_options%max_iterations=options%max_iterations;optimizer_options%max_line_search=options%max_line_search;optimizer_options%gradient_tolerance=options%gradient_tolerance;optimizer_options%step_tolerance=options%step_tolerance;optimizer_options%objective_tolerance=options%objective_tolerance
        call optimizer%minimize(objective,parameters,lower,upper,optimizer_options,optimizer_result,status); if(.not.status_ok(status))return
        call model%set_parameters(parameters(:n),status); if(.not.status_ok(status))return
        if (options%optimize_l2) then
            model%l2_value = parameters(n + 1)
        else
            model%l2_value = options%l2
        end if
        if (options%optimize_delta) then
            model%delta_value = parameters(n + 1 + merge(1, 0, options%optimize_l2))
        else
            model%delta_value = options%delta
        end if
        call adapter%value_gradient(parameters,result%objective,gradient,status); if(.not.status_ok(status))return
        result%converged=optimizer_result%state%converged;result%iterations=optimizer_result%state%iteration;result%line_search_evaluations=optimizer_result%line_search_evaluations;result%gradient_norm=sqrt(sum(gradient*gradient));result%l2=model%l2_value;result%delta=model%delta_value
        if(.not.result%converged)then;call status_set(status,FORTNUM_CONVERGENCE_ERROR,"Huber L-BFGS-B: iteration limit reached");return;end if
        call status_set(status,FORTNUM_OK,"")
    end subroutine huber_optimize_lbfgsb

    logical function valid_options(options) result(valid)
        type(huber_lbfgsb_options_t), intent(in) :: options
        valid=options%memory>=1.and.options%max_iterations>=1.and.options%max_line_search>=1.and.options%lower_bound<options%upper_bound.and.options%l2_lower_bound>=0.0_dp.and.options%l2_lower_bound<=options%l2_upper_bound.and.options%delta_lower_bound>0.0_dp.and.options%delta_lower_bound<=options%delta_upper_bound.and.options%delta>0.0_dp.and.options%l2>=0.0_dp.and.(options%device_kind==FORTML_DEVICE_CPU.or.options%device_kind==FORTML_DEVICE_CUDA)
        valid=valid.and.ieee_is_finite(options%gradient_tolerance).and.ieee_is_finite(options%step_tolerance).and.ieee_is_finite(options%objective_tolerance).and.ieee_is_finite(options%lower_bound).and.ieee_is_finite(options%upper_bound).and.ieee_is_finite(options%l2).and.ieee_is_finite(options%delta).and.options%gradient_tolerance>=0.0_dp.and.options%step_tolerance>=0.0_dp.and.options%objective_tolerance>=0.0_dp
        if(options%optimize_l2) valid=valid.and.options%l2>=options%l2_lower_bound.and.options%l2<=options%l2_upper_bound
        if(options%optimize_delta) valid=valid.and.options%delta>=options%delta_lower_bound.and.options%delta<=options%delta_upper_bound
    end function valid_options

    subroutine make_design(x, intercept, design)
        real(dp), intent(in) :: x(:, :); logical, intent(in) :: intercept; real(dp), intent(out) :: design(:, :)
        design=0.0_dp; if(intercept)then;design(:,1)=1.0_dp;design(:,2:)=x;else;design=x;end if
    end subroutine make_design
    subroutine make_tangent_design(x, intercept, design)
        real(dp), intent(in)::x(:, :);logical,intent(in)::intercept;real(dp),intent(out)::design(:, :)
        design=0.0_dp;if(intercept)then;design(:,2:)=x;else;design=x;end if
    end subroutine make_tangent_design

    subroutine huber_parameter_get(context, values, status)
        class(*), pointer, intent(in) :: context; real(dp), intent(out)::values(:); type(fortnum_status_t),intent(out)::status
        select type(model=>context);type is(huber_regression_t)
            if(size(values)/=model%parameter_count())then;call status_set(status,FORTNUM_DOMAIN_ERROR,"Huber registry getter: shape invalid");return;end if
            values=reshape(model%coefficient,[size(values)]);call status_set(status,FORTNUM_OK,"")
        class default;call status_set(status,FORTNUM_DOMAIN_ERROR,"Huber registry getter: context type invalid")
        end select
    end subroutine huber_parameter_get
    subroutine huber_parameter_set(context, values, status)
        class(*), pointer, intent(inout)::context;real(dp),intent(in)::values(:);type(fortnum_status_t),intent(out)::status
        select type(model=>context);type is(huber_regression_t)
            call model%set_parameters(values,status)
        class default;call status_set(status,FORTNUM_DOMAIN_ERROR,"Huber registry setter: context type invalid")
        end select
    end subroutine huber_parameter_set

end module fortml_huber_regression
