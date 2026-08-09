module fortml_operator_gaussian_process
    !! Exact dense GPs with a registered first-order linear differential
    !! observation operator.
    !!
    !! A registry column is ordered as ``[value, d/dx_1, ..., d/dx_d]``.
    !! An observation therefore represents
    !! ``a_0 f(x) + sum_i a_i d_i f(x)`` rather than forcing callers to
    !! duplicate rows for every derivative component.  The covariance is
    !! assembled from the kernel value, both input gradients, and the mixed
    !! input Hessian supplied by ``kernel_t``.  This module deliberately keeps
    !! the operator state separate from ``gp_derivative_regression_t``: the
    !! latter remains the row-wise component API, while this registry gives
    !! physics and boundary-condition code a named linear-operator seam.
    !!
    !! CPU is the reference backend.  CUDA entry points are explicit typed
    !! refusals until a resident operator covariance/factorization graph is
    !! available; no hidden host fallback is used.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, clone_kernel_into
    implicit none
    private

    integer, parameter, public :: OPERATOR_NAME_LENGTH = 64

    type, public :: linear_differential_operator_registry_t
        integer :: input_dim = 0
        integer :: n_operators = 0
        real(dp), allocatable :: coefficients(:, :) ! [value, gradients] x op
        character(len=OPERATOR_NAME_LENGTH), allocatable :: names(:)
    contains
        procedure, public :: initialize => operator_registry_initialize
        procedure, public :: set_operator => operator_registry_set_operator
        procedure, public :: operator_count => operator_registry_count
        procedure, public :: parameter_count => operator_registry_parameter_count
        procedure, public :: operator_coefficients => operator_registry_coefficients
    end type linear_differential_operator_registry_t

    type, public :: gp_linear_operator_regression_t
        type(kernel_t) :: kernel
        type(cholesky_factorization_t) :: factorization
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: operators_train(:, :) ! [value, gradients] x op
        real(dp), allocatable :: y_train(:, :)
        real(dp), allocatable :: alpha(:, :)
        character(len=OPERATOR_NAME_LENGTH), allocatable :: operator_names(:)
        real(dp) :: noise_variance = 1.0e-6_dp
        real(dp) :: jitter = 1.0e-10_dp
        integer :: n_observations = 0
        integer :: n_features = 0
        integer :: n_outputs = 0
    contains
        procedure, public :: fit => gp_operator_fit
        procedure, public :: predict => gp_operator_predict
        procedure, public :: predict_device => gp_operator_predict_device
        procedure, public :: joint_covariance => gp_operator_joint_covariance
        procedure, public :: joint_covariance_device => gp_operator_joint_covariance_device
        procedure, public :: predict_operator_jvp => gp_operator_predict_operator_jvp
        procedure, public :: predict_operator_vjp => gp_operator_predict_operator_vjp
        procedure, public :: predict_operator_jvp_device => &
            gp_operator_predict_operator_jvp_device
        procedure, public :: predict_operator_vjp_device => &
            gp_operator_predict_operator_vjp_device
        procedure, public :: device_supported => gp_operator_device_supported
        procedure, public :: observation_count => gp_operator_observation_count
        procedure, public :: parameter_count => gp_operator_parameter_count
        procedure, public :: parameters => gp_operator_parameters
    end type gp_linear_operator_regression_t

    public :: gp_operator_fit
    public :: gp_operator_predict
    public :: gp_operator_predict_device
    public :: gp_operator_joint_covariance
    public :: gp_operator_joint_covariance_device
    public :: gp_operator_predict_operator_jvp
    public :: gp_operator_predict_operator_vjp
    public :: gp_operator_predict_operator_jvp_device
    public :: gp_operator_predict_operator_vjp_device

contains

    subroutine operator_registry_initialize(self, input_dim, n_operators, status)
        class(linear_differential_operator_registry_t), intent(out) :: self
        integer, intent(in) :: input_dim, n_operators
        type(fortnum_status_t), intent(out) :: status

        if (input_dim < 1 .or. n_operators < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator registry: dimensions must be positive")
            return
        end if
        self%input_dim = input_dim
        self%n_operators = n_operators
        allocate(self%coefficients(input_dim + 1, n_operators))
        allocate(self%names(n_operators))
        self%coefficients = 0.0_dp
        self%names = "unnamed"
        call status_set(status, FORTNUM_OK, "")
    end subroutine operator_registry_initialize

    subroutine operator_registry_set_operator(self, index, name, coefficients, status)
        class(linear_differential_operator_registry_t), intent(inout) :: self
        integer, intent(in) :: index
        character(len=*), intent(in) :: name
        real(dp), intent(in) :: coefficients(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. allocated(self%coefficients) .or. index < 1 .or. &
            index > self%n_operators .or. size(coefficients) /= self%input_dim + 1 .or. &
            len_trim(name) < 1 .or. len_trim(name) > OPERATOR_NAME_LENGTH .or. &
            any(.not. ieee_is_finite(coefficients))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator registry: operator shape, name, or coefficients are invalid")
            return
        end if
        self%coefficients(:, index) = coefficients
        self%names(index) = " "
        self%names(index)(:len_trim(name)) = name(:len_trim(name))
        call status_set(status, FORTNUM_OK, "")
    end subroutine operator_registry_set_operator

    integer function operator_registry_count(self) result(count)
        class(linear_differential_operator_registry_t), intent(in) :: self
        count = self%n_operators
    end function operator_registry_count

    integer function operator_registry_parameter_count(self) result(count)
        class(linear_differential_operator_registry_t), intent(in) :: self
        count = self%input_dim + 1
    end function operator_registry_parameter_count

    function operator_registry_coefficients(self) result(coefficients)
        class(linear_differential_operator_registry_t), intent(in) :: self
        real(dp), allocatable :: coefficients(:, :)
        if (allocated(self%coefficients)) then
            allocate(coefficients, source=self%coefficients)
        else
            allocate(coefficients(0, 0))
        end if
    end function operator_registry_coefficients

    subroutine gp_operator_fit(self, x, operators, y, kernel, noise_variance, status, jitter)
        class(gp_linear_operator_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: jitter
        real(dp), allocatable :: covariance(:, :)
        integer :: i

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(y, 1) /= size(x, 1) .or. &
            size(y, 2) < 1 .or. noise_variance <= 0.0_dp .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP fit: data dimensions, finiteness, or noise is invalid")
            return
        end if
        if (.not. valid_registry(operators, size(x, 2))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP fit: registry dimensions are invalid")
            return
        end if
        if (operators%n_operators /= size(x, 1) .or. &
            kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP fit: observation or kernel dimensions are invalid")
            return
        end if

        call clone_kernel_into(kernel, self%kernel)
        allocate(self%x_train, source=x)
        allocate(self%operators_train, source=operators%coefficients)
        allocate(self%operator_names, source=operators%names)
        allocate(self%y_train, source=y)
        self%n_observations = size(x, 1)
        self%n_features = size(x, 2)
        self%n_outputs = size(y, 2)
        self%noise_variance = noise_variance
        self%jitter = 1.0e-10_dp
        if (present(jitter)) self%jitter = jitter
        if (.not. ieee_is_finite(self%jitter) .or. self%jitter < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP fit: jitter must be finite and nonnegative")
            return
        end if
        allocate(covariance(self%n_observations, self%n_observations))
        do i = 1, self%n_observations
            call operator_covariance_matrix_row(self, i, covariance(i, :), status)
            if (status%code /= FORTNUM_OK) return
            covariance(i, i) = covariance(i, i) + self%noise_variance + self%jitter
        end do
        call self%factorization%factorize(covariance, status)
        if (status%code /= FORTNUM_OK) return
        allocate(self%alpha, source=self%y_train)
        call self%factorization%solve(self%alpha, status)
    end subroutine gp_operator_fit

    subroutine gp_operator_predict(self, x, operators, mean, variance, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        real(dp), intent(out) :: mean(:, :), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :)
        real(dp) :: prior
        integer :: i, j

        if (.not. operator_gp_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction: model is not fitted")
            return
        end if
        if (.not. valid_query(self, x, operators, mean, variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction: query or output shape is invalid")
            return
        end if
        allocate(cross(self%n_observations, operators%n_operators))
        do j = 1, operators%n_operators
            do i = 1, self%n_observations
                call operator_covariance(self%kernel, self%x_train(i, :), &
                    self%operators_train(:, i), x(j, :), operators%coefficients(:, j), &
                    cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        mean = matmul(transpose(cross), self%alpha)
        allocate(work, source=cross)
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, operators%n_operators
            call operator_covariance(self%kernel, x(j, :), operators%coefficients(:, j), &
                x(j, :), operators%coefficients(:, j), prior, status)
            if (status%code /= FORTNUM_OK) return
            variance(j) = prior - dot_product(cross(:, j), work(:, j))
            if (variance(j) < 0.0_dp) then
                if (variance(j) < -1.0e-9_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "operator GP prediction: posterior variance is not positive")
                    return
                end if
                variance(j) = 0.0_dp
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_operator_predict

    subroutine gp_operator_predict_device(self, device, x, operators, mean, variance, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        real(dp), intent(out) :: mean(:, :), variance(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, operators, mean, variance, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "operator GP device prediction: resident operator covariance is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP device prediction: device kind is invalid")
        end select
    end subroutine gp_operator_predict_device

    subroutine gp_operator_joint_covariance(self, x, operators, covariance, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        if (.not. operator_gp_fitted(self) .or. .not. valid_query_covariance(self, x, &
            operators, covariance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP joint covariance: model, query, or output is invalid")
            return
        end if
        call operator_posterior_covariance(self, x, operators, covariance, status)
    end subroutine gp_operator_joint_covariance

    subroutine gp_operator_joint_covariance_device(self, device, x, operators, covariance, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        real(dp), intent(out) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP covariance device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%joint_covariance(x, operators, covariance, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "operator GP covariance device: resident operator covariance is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP covariance device: device kind is invalid")
        end select
    end subroutine gp_operator_joint_covariance_device

    subroutine gp_operator_predict_operator_jvp(self, x, operators, direction, mean, &
            mean_dot, variance, variance_dot, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:, :)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), work(:, :)
        real(dp) :: prior, prior_dot
        integer :: i, j

        if (.not. operator_gp_fitted(self) .or. .not. valid_query(self, x, operators, mean, variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction JVP: model or query is invalid")
            return
        end if
        if (any(shape(direction) /= shape(operators%coefficients)) .or. &
            any(.not. ieee_is_finite(direction)) .or. any(shape(mean_dot) /= shape(mean)) .or. &
            size(variance_dot) /= size(variance)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction JVP: direction or output shape is invalid")
            return
        end if
        allocate(cross(self%n_observations, operators%n_operators))
        allocate(cross_dot, mold=cross)
        do j = 1, operators%n_operators
            do i = 1, self%n_observations
                call operator_covariance(self%kernel, self%x_train(i, :), &
                    self%operators_train(:, i), x(j, :), operators%coefficients(:, j), &
                    cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
                call operator_covariance_direction(self%kernel, self%x_train(i, :), &
                    self%operators_train(:, i), x(j, :), operators%coefficients(:, j), &
                    direction(:, j), cross_dot(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
            call operator_covariance(self%kernel, x(j, :), operators%coefficients(:, j), &
                x(j, :), operators%coefficients(:, j), prior, status)
            if (status%code /= FORTNUM_OK) return
            call operator_covariance_two_directions(self%kernel, x(j, :), &
                operators%coefficients(:, j), direction(:, j), prior_dot, status)
            if (status%code /= FORTNUM_OK) return
            variance(j) = prior
            variance_dot(j) = prior_dot
        end do
        mean = matmul(transpose(cross), self%alpha)
        mean_dot = matmul(transpose(cross_dot), self%alpha)
        allocate(work, source=cross)
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        variance = variance - sum(cross*work, dim=1)
        variance_dot = variance_dot - 2.0_dp*sum(cross_dot*work, dim=1)
        do j = 1, size(variance)
            if (variance(j) < 0.0_dp) then
                if (variance(j) < -1.0e-9_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "operator GP prediction JVP: posterior variance is not positive")
                    return
                end if
                variance(j) = 0.0_dp
            end if
        end do
        if (any(.not. ieee_is_finite(mean_dot)) .or. any(.not. ieee_is_finite(variance_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "operator GP prediction JVP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_operator_predict_operator_jvp

    subroutine gp_operator_predict_operator_vjp(self, x, operators, mean_bar, variance_bar, &
            operator_bar, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:, :), variance_bar(:)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        real(dp), intent(out) :: operator_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), work(:, :), cross_bar(:, :), basis(:)
        real(dp) :: covariance, prior_left, prior_right
        integer :: i, j, q

        operator_bar = 0.0_dp
        if (.not. operator_gp_fitted(self) .or. .not. valid_query(self, x, operators, &
            mean_bar, variance_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction VJP: model or query is invalid")
            return
        end if
        if (any(shape(operator_bar) /= shape(operators%coefficients)) .or. &
            any(.not. ieee_is_finite(mean_bar)) .or. any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction VJP: cotangent or output shape is invalid")
            return
        end if
        allocate(cross(self%n_observations, operators%n_operators), work(self%n_observations, &
            operators%n_operators), cross_bar(self%n_observations, operators%n_operators))
        do j = 1, operators%n_operators
            do i = 1, self%n_observations
                call operator_covariance(self%kernel, self%x_train(i, :), &
                    self%operators_train(:, i), x(j, :), operators%coefficients(:, j), &
                    cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        cross_bar = matmul(self%alpha, transpose(mean_bar)) - 2.0_dp*work* &
            spread(variance_bar, dim=1, ncopies=self%n_observations)
        allocate(basis(size(operators%coefficients, 1)))
        operator_bar = 0.0_dp
        do j = 1, operators%n_operators
            do q = 1, size(basis)
                basis = 0.0_dp
                basis(q) = 1.0_dp
                do i = 1, self%n_observations
                    call operator_covariance(self%kernel, self%x_train(i, :), &
                        self%operators_train(:, i), x(j, :), basis, covariance, status)
                    if (status%code /= FORTNUM_OK) return
                    operator_bar(q, j) = operator_bar(q, j) + cross_bar(i, j)*covariance
                end do
                call operator_covariance(self%kernel, x(j, :), operators%coefficients(:, j), &
                    x(j, :), basis, prior_left, status)
                if (status%code /= FORTNUM_OK) return
                call operator_covariance(self%kernel, x(j, :), basis, x(j, :), &
                    operators%coefficients(:, j), prior_right, status)
                if (status%code /= FORTNUM_OK) return
                operator_bar(q, j) = operator_bar(q, j) + variance_bar(j)* &
                    (prior_left + prior_right)
            end do
        end do
        if (any(.not. ieee_is_finite(operator_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "operator GP prediction VJP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_operator_predict_operator_vjp

    subroutine gp_operator_predict_operator_jvp_device(self, device, x, operators, direction, &
            mean, mean_dot, variance, variance_dot, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), direction(:, :)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_operator_jvp(x, operators, direction, mean, mean_dot, &
                variance, variance_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "operator GP prediction JVP device: resident operator covariance is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction JVP device: device kind is invalid")
        end select
    end subroutine gp_operator_predict_operator_jvp_device

    subroutine gp_operator_predict_operator_vjp_device(self, device, x, operators, mean_bar, &
            variance_bar, operator_bar, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), mean_bar(:, :), variance_bar(:)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        real(dp), intent(out) :: operator_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_operator_vjp(x, operators, mean_bar, variance_bar, operator_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "operator GP prediction VJP device: resident operator covariance is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP prediction VJP device: device kind is invalid")
        end select
    end subroutine gp_operator_predict_operator_vjp_device

    logical function gp_operator_device_supported(self, device_kind) result(supported)
        class(gp_linear_operator_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind
        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = operator_gp_fitted(self)
        case default
            supported = .false.
        end select
    end function gp_operator_device_supported

    integer function gp_operator_observation_count(self) result(count)
        class(gp_linear_operator_regression_t), intent(in) :: self
        count = self%n_observations
    end function gp_operator_observation_count

    integer function gp_operator_parameter_count(self) result(count)
        class(gp_linear_operator_regression_t), intent(in) :: self
        count = self%kernel%parameter_count() + 1
    end function gp_operator_parameter_count

    function gp_operator_parameters(self) result(parameters)
        class(gp_linear_operator_regression_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: n
        n = self%kernel%parameter_count()
        allocate(parameters(n + 1))
        if (n > 0) parameters(:n) = self%kernel%parameters()
        parameters(n + 1) = log(self%noise_variance)
    end function gp_operator_parameters

    subroutine operator_covariance_matrix_row(self, row, output, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        integer, intent(in) :: row
        real(dp), intent(out) :: output(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: j
        if (size(output) /= self%n_observations) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator GP covariance row: output shape is invalid")
            return
        end if
        do j = 1, self%n_observations
            call operator_covariance(self%kernel, self%x_train(row, :), &
                self%operators_train(:, row), self%x_train(j, :), self%operators_train(:, j), &
                output(j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine operator_covariance_matrix_row

    subroutine operator_posterior_covariance(self, x, operators, covariance, status)
        class(gp_linear_operator_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        real(dp), intent(inout) :: covariance(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), prior(:, :), work(:, :)
        integer :: i, j, n

        n = operators%n_operators
        allocate(cross(self%n_observations, n), prior(n, n), work(self%n_observations, n))
        do j = 1, n
            do i = 1, self%n_observations
                call operator_covariance(self%kernel, self%x_train(i, :), &
                    self%operators_train(:, i), x(j, :), operators%coefficients(:, j), &
                    cross(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
            do i = 1, n
                call operator_covariance(self%kernel, x(i, :), operators%coefficients(:, i), &
                    x(j, :), operators%coefficients(:, j), prior(i, j), status)
                if (status%code /= FORTNUM_OK) return
            end do
        end do
        work = cross
        call self%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        covariance = prior - matmul(transpose(cross), work)
        covariance = 0.5_dp*(covariance + transpose(covariance))
        do i = 1, n
            if (covariance(i, i) < 0.0_dp) then
                if (covariance(i, i) < -1.0e-9_dp) then
                    call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                        "operator GP covariance: posterior variance is not positive")
                    return
                end if
                covariance(i, i) = 0.0_dp
            end if
        end do
        if (any(.not. ieee_is_finite(covariance))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "operator GP covariance: nonfinite result")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine operator_posterior_covariance

    subroutine operator_covariance(kernel, x1, weights1, x2, weights2, covariance, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), weights1(:), x2(:), weights2(:)
        real(dp), intent(out) :: covariance
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        real(dp) :: value
        integer :: i, j, d

        covariance = 0.0_dp
        d = size(x1)
        if (size(x2) /= d .or. size(weights1) /= d + 1 .or. size(weights2) /= d + 1 .or. &
            kernel%input_dim /= d .or. any(.not. ieee_is_finite(x1)) .or. &
            any(.not. ieee_is_finite(x2)) .or. any(.not. ieee_is_finite(weights1)) .or. &
            any(.not. ieee_is_finite(weights2))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator covariance: input or coefficient shape is invalid")
            return
        end if
        if (all(weights1(2:) == 0.0_dp) .and. all(weights2(2:) == 0.0_dp)) then
            covariance = weights1(1)*weights2(1)*kernel%value(x1, x2)
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        allocate(gradient_x1(d), gradient_x2(d), mixed_hessian(d, d))
        call kernel%input_derivatives(x1, x2, value, gradient_x1, gradient_x2, mixed_hessian, status)
        if (status%code /= FORTNUM_OK) return
        covariance = weights1(1)*weights2(1)*value
        do i = 1, d
            covariance = covariance + weights1(i + 1)*weights2(1)*gradient_x1(i) + &
                weights1(1)*weights2(i + 1)*gradient_x2(i)
            do j = 1, d
                covariance = covariance + weights1(i + 1)*weights2(j + 1)*mixed_hessian(i, j)
            end do
        end do
        if (.not. ieee_is_finite(covariance)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "operator covariance: nonfinite result")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine operator_covariance

    subroutine operator_covariance_direction(kernel, x1, weights1, x2, weights2, direction, &
            covariance_dot, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x1(:), weights1(:), x2(:), weights2(:), direction(:)
        real(dp), intent(out) :: covariance_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: basis(:)
        real(dp) :: value
        integer :: q

        covariance_dot = 0.0_dp
        if (size(direction) /= size(weights2) .or. any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "operator covariance direction: shape is invalid")
            return
        end if
        allocate(basis(size(direction)))
        do q = 1, size(direction)
            if (direction(q) == 0.0_dp) cycle
            basis = 0.0_dp
            basis(q) = 1.0_dp
            call operator_covariance(kernel, x1, weights1, x2, basis, value, status)
            if (status%code /= FORTNUM_OK) return
            covariance_dot = covariance_dot + direction(q)*value
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine operator_covariance_direction

    subroutine operator_covariance_two_directions(kernel, x, weights, direction, value_dot, status)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: x(:), weights(:), direction(:)
        real(dp), intent(out) :: value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: left, right

        call operator_covariance(kernel, x, direction, x, weights, left, status)
        if (status%code /= FORTNUM_OK) return
        call operator_covariance(kernel, x, weights, x, direction, right, status)
        if (status%code /= FORTNUM_OK) return
        value_dot = left + right
        call status_set(status, FORTNUM_OK, "")
    end subroutine operator_covariance_two_directions

    logical function valid_registry(registry, input_dim) result(valid)
        type(linear_differential_operator_registry_t), intent(in) :: registry
        integer, intent(in) :: input_dim
        integer :: i
        valid = .false.
        if (.not. allocated(registry%coefficients)) return
        if (.not. allocated(registry%names)) return
        if (registry%input_dim /= input_dim) return
        if (registry%n_operators < 1) return
        if (.not. all(shape(registry%coefficients) == [input_dim + 1, registry%n_operators])) return
        if (size(registry%names) /= registry%n_operators) return
        do i = 1, registry%n_operators
            if (len_trim(registry%names(i)) < 1) return
        end do
        if (any(.not. ieee_is_finite(registry%coefficients))) return
        valid = .true.
    end function valid_registry

    logical function valid_query(self, x, operators, mean, variance) result(valid)
        class(gp_linear_operator_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean(:, :), variance(:)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        valid = .false.
        if (.not. operator_gp_fitted(self)) return
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features) return
        if (.not. valid_registry(operators, self%n_features)) return
        if (size(mean, 1) /= operators%n_operators .or. size(mean, 2) /= self%n_outputs) return
        if (size(variance) /= operators%n_operators) return
        if (any(.not. ieee_is_finite(x))) return
        valid = .true.
    end function valid_query

    logical function valid_query_covariance(self, x, operators, covariance) result(valid)
        class(gp_linear_operator_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), covariance(:, :)
        type(linear_differential_operator_registry_t), intent(in) :: operators
        valid = .false.
        if (.not. operator_gp_fitted(self)) return
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features) return
        if (.not. valid_registry(operators, self%n_features)) return
        if (.not. all(shape(covariance) == [operators%n_operators, operators%n_operators])) return
        if (any(.not. ieee_is_finite(x))) return
        valid = .true.
    end function valid_query_covariance

    logical function operator_gp_fitted(self) result(fitted)
        class(gp_linear_operator_regression_t), intent(in) :: self
        fitted = allocated(self%x_train) .and. allocated(self%operators_train) .and. &
            allocated(self%y_train) .and. allocated(self%alpha) .and. self%n_observations > 0 .and. &
            self%n_features > 0 .and. self%n_outputs > 0
    end function operator_gp_fitted

end module fortml_operator_gaussian_process
