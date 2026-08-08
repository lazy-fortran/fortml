module fortml_kernels
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortml_generated_matern12_products, only: fortml_matern12_hvp
    use fortml_generated_matern32_products, only: fortml_matern32_hvp
    use fortml_generated_matern52_products, only: fortml_matern52_hvp
    use fortml_generated_rbf_products, only: fortml_rbf_hvp
    use fortml_kernel_formula, only: kernel_formula_t, MAX_FORMULA_STACK, &
        OPCODE_PUSH_R2, OPCODE_PUSH_R, OPCODE_PUSH_DOT, OPCODE_PUSH_CONST, &
        OPCODE_ADD, OPCODE_SUBTRACT, OPCODE_MULTIPLY, OPCODE_NEGATE, &
        OPCODE_EXP, OPCODE_DIVIDE_CONST
    implicit none
    private

    interface
        subroutine fortml_generated_rbf_leaf_fortran( &
                variance, distance, lengthscale, output)
            use, intrinsic :: iso_fortran_env, only: real64
            real(real64), intent(in) :: variance, distance, lengthscale
            real(real64), intent(out) :: output
        end subroutine fortml_generated_rbf_leaf_fortran

        subroutine fortml_generated_rbf_leaf_derivatives( &
                variance, distance, lengthscale, value, dvariance, &
                ddistance, dlengthscale)
            use, intrinsic :: iso_fortran_env, only: real64
            real(real64), intent(in) :: variance, distance, lengthscale
            real(real64), intent(out) :: value, dvariance, ddistance, &
                dlengthscale
        end subroutine fortml_generated_rbf_leaf_derivatives
    end interface

    integer, parameter, public :: KERNEL_RBF = 1
    integer, parameter, public :: KERNEL_MATERN12 = 2
    integer, parameter, public :: KERNEL_MATERN32 = 3
    integer, parameter, public :: KERNEL_MATERN52 = 4
    integer, parameter, public :: KERNEL_LINEAR = 5
    integer, parameter, public :: KERNEL_CONSTANT = 6
    integer, parameter, public :: KERNEL_WHITE_NOISE = 7
    integer, parameter, public :: KERNEL_SUM = 8
    integer, parameter, public :: KERNEL_PRODUCT = 9
    integer, parameter, public :: KERNEL_USER = 10
    integer, parameter, public :: KERNEL_PERIODIC = 11
    integer, parameter, public :: KERNEL_RATIONAL_QUADRATIC = 12
    integer, parameter, public :: KERNEL_COSINE = 13
    integer, parameter, public :: KERNEL_POLYNOMIAL = 14
    !! Squared-exponential kernel with one positive length scale per feature.
    !! Parameters are packed as [log_variance, log_lengthscale(:)].
    integer, parameter, public :: KERNEL_RBF_ARD = 15
    !! GPyTorch-style spectral mixture kernel.  Parameters are packed per
    !! mixture as [log_weight, log_scale(:), mean(:)].  Scales and weights
    !! use logarithmic unconstrained coordinates while frequencies (means)
    !! remain signed physical coordinates.
    integer, parameter, public :: KERNEL_SPECTRAL_MIXTURE = 16
    !! Product of a squared-exponential envelope and a periodic factor.
    !! Parameters are packed as [log_variance, log_envelope_lengthscale,
    !! log_periodic_lengthscale, log_period].
    integer, parameter, public :: KERNEL_LOCAL_PERIODIC = 17
    !! Smooth change-point covariance.  The two child kernels are blended by
    !! logistic gates in one input feature; parameters are the child
    !! parameters followed by [log_transition_width, transition_center].
    integer, parameter, public :: KERNEL_CHANGE_POINT = 18

    type, public :: kernel_t
        integer :: kind = 0
        integer :: input_dim = 0
        real(dp), allocatable :: log_parameters(:)
        type(kernel_t), pointer :: left => null()
        type(kernel_t), pointer :: right => null()
        integer :: change_feature = 1
        !! A validated user formula, present only for KERNEL_USER leaves.
        type(kernel_formula_t), allocatable :: formula
    contains
        procedure, public :: parameter_count => kernel_parameter_count
        procedure, public :: parameters => kernel_parameters
        procedure, public :: set_parameters => kernel_set_parameters
        procedure, public :: value => kernel_value
        procedure, public :: input_derivatives => kernel_input_derivatives
        procedure, public :: matrix => kernel_matrix
        procedure, public :: matrix_jvp => kernel_matrix_jvp
        procedure, public :: parameter_vjp => kernel_parameter_vjp
        procedure, public :: parameter_hvp => kernel_parameter_hvp
    end type kernel_t

    public :: make_rbf_kernel
    public :: make_rbf_ard_kernel
    public :: make_spectral_mixture_kernel
    public :: make_ard_rbf_kernel
    public :: make_matern12_kernel
    public :: make_matern32_kernel
    public :: make_matern52_kernel
    public :: make_linear_kernel
    public :: make_constant_kernel
    public :: make_white_noise_kernel
    public :: make_periodic_kernel
    public :: make_local_periodic_kernel
    public :: make_change_point_kernel
    public :: make_rational_quadratic_kernel
    public :: make_cosine_kernel
    public :: make_polynomial_kernel
    public :: make_user_kernel
    public :: kernel_add
    public :: kernel_multiply
    public :: clone_kernel
    public :: clone_kernel_into
    public :: kernel_input_derivatives

contains

    function make_rbf_kernel(input_dim, variance, lengthscale, status) result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, lengthscale
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_leaf(kernel, KERNEL_RBF, input_dim, variance, lengthscale, status)
    end function make_rbf_kernel

    function make_rbf_ard_kernel(input_dim, variance, lengthscales, status) result(kernel)
        !! Construct an anisotropic squared-exponential (ARD RBF) kernel.
        !!
        !! The positive ``lengthscales`` vector is copied into the kernel and
        !! represented internally by logarithms.  Keeping the logarithmic
        !! parameterization makes the ordinary kernel parameter products and
        !! GP hyperparameter optimization unconstrained, just as for the
        !! isotropic ``make_rbf_kernel`` constructor.
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, lengthscales(:)
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        if (input_dim < 1 .or. variance <= 0.0_dp .or. &
            size(lengthscales) /= input_dim .or. any(lengthscales <= 0.0_dp) .or. &
            .not. ieee_is_finite(variance) .or. any(.not. ieee_is_finite(lengthscales))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ARD RBF constructor: dimensions and scales must be positive and finite")
            return
        end if
        kernel%kind = KERNEL_RBF_ARD
        kernel%input_dim = input_dim
        allocate(kernel%log_parameters(input_dim + 1))
        kernel%log_parameters(1) = log(variance)
        kernel%log_parameters(2:) = log(lengthscales)
        call status_set(status, FORTNUM_OK, "")
    end function make_rbf_ard_kernel

    function make_ard_rbf_kernel(input_dim, variance, lengthscales, status) result(kernel)
        !! Alias with the conventional ``ARD_RBF`` word ordering.
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, lengthscales(:)
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        kernel = make_rbf_ard_kernel(input_dim, variance, lengthscales, status)
    end function make_ard_rbf_kernel

    function make_spectral_mixture_kernel(input_dim, num_mixtures, weights, means, &
            scales, status) result(kernel)
        !! Construct a stationary spectral-mixture kernel.
        !!
        !! For lag ``tau`` the q-th component is
        !! ``w_q prod_d exp(-2*pi^2*tau_d^2*s_qd**2)*cos(2*pi*tau_d*mu_qd)``.
        !! ``weights`` and frequency standard deviations ``scales`` are positive
        !! physical values and are packed as logarithms; ``means`` are signed
        !! frequencies.  This is
        !! the same parameterization used by GPyTorch's
        !! ``SpectralMixtureKernel`` and composes with the ordinary sum and
        !! product kernel trees.
        integer, intent(in) :: input_dim, num_mixtures
        real(dp), intent(in) :: weights(:), means(:, :), scales(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel
        integer :: q, feature, base, block

        if (input_dim < 1 .or. num_mixtures < 1 .or. size(weights) /= num_mixtures .or. &
            size(means, 1) /= num_mixtures .or. size(means, 2) /= input_dim .or. &
            size(scales, 1) /= num_mixtures .or. size(scales, 2) /= input_dim .or. &
            any(weights <= 0.0_dp) .or. any(scales <= 0.0_dp) .or. &
            any(.not. ieee_is_finite(weights)) .or. any(.not. ieee_is_finite(means)) .or. &
            any(.not. ieee_is_finite(scales))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "spectral mixture constructor: dimensions and scales must be valid")
            return
        end if
        kernel%kind = KERNEL_SPECTRAL_MIXTURE
        kernel%input_dim = input_dim
        block = 1 + 2*input_dim
        allocate(kernel%log_parameters(num_mixtures*block))
        do q = 1, num_mixtures
            base = (q - 1)*block
            kernel%log_parameters(base + 1) = log(weights(q))
            do feature = 1, input_dim
                kernel%log_parameters(base + 1 + feature) = log(scales(q, feature))
                kernel%log_parameters(base + 1 + input_dim + feature) = means(q, feature)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end function make_spectral_mixture_kernel

    function make_matern12_kernel(input_dim, variance, lengthscale, status) result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, lengthscale
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_leaf(kernel, KERNEL_MATERN12, input_dim, variance, lengthscale, status)
    end function make_matern12_kernel

    function make_matern32_kernel(input_dim, variance, lengthscale, status) result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, lengthscale
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_leaf(kernel, KERNEL_MATERN32, input_dim, variance, lengthscale, status)
    end function make_matern32_kernel

    function make_matern52_kernel(input_dim, variance, lengthscale, status) result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, lengthscale
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_leaf(kernel, KERNEL_MATERN52, input_dim, variance, lengthscale, status)
    end function make_matern52_kernel

    function make_linear_kernel(input_dim, variance, status) result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_leaf(kernel, KERNEL_LINEAR, input_dim, variance, 1.0_dp, status)
    end function make_linear_kernel

    function make_constant_kernel(input_dim, variance, status) result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_leaf(kernel, KERNEL_CONSTANT, input_dim, variance, 1.0_dp, status)
    end function make_constant_kernel

    function make_white_noise_kernel(input_dim, variance, status) result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_leaf(kernel, KERNEL_WHITE_NOISE, input_dim, variance, 1.0_dp, status)
    end function make_white_noise_kernel

    function make_periodic_kernel(input_dim, variance, lengthscale, period, status) &
            result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, lengthscale, period
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_three_parameter_leaf(kernel, KERNEL_PERIODIC, input_dim, &
            variance, lengthscale, period, status)
    end function make_periodic_kernel

    function make_local_periodic_kernel(input_dim, variance, envelope_lengthscale, &
            periodic_lengthscale, period, status) result(kernel)
        !! Construct the locally-periodic covariance
        !!
        !! ``k(x,x') = variance * exp(-||x-x'||^2/(2 ell_e^2)) *
        !! exp(-2 sin^2(pi ||x-x'|| / period) / ell_p^2)``.
        !! All four parameters are positive and stored in logarithmic
        !! coordinates, so the kernel composes directly with exact GP
        !! inference and its hyperparameter products.
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, envelope_lengthscale
        real(dp), intent(in) :: periodic_lengthscale, period
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        if (input_dim < 1 .or. variance <= 0.0_dp .or. envelope_lengthscale <= 0.0_dp .or. &
            periodic_lengthscale <= 0.0_dp .or. period <= 0.0_dp .or. &
            .not. ieee_is_finite(variance) .or. .not. ieee_is_finite(envelope_lengthscale) .or. &
            .not. ieee_is_finite(periodic_lengthscale) .or. .not. ieee_is_finite(period)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "local periodic constructor: dimensions and scales must be positive and finite")
            return
        end if
        kernel%kind = KERNEL_LOCAL_PERIODIC
        kernel%input_dim = input_dim
        allocate(kernel%log_parameters(4))
        kernel%log_parameters = [log(variance), log(envelope_lengthscale), &
            log(periodic_lengthscale), log(period)]
        call status_set(status, FORTNUM_OK, "")
    end function make_local_periodic_kernel

    function make_change_point_kernel(left, right, feature, center, width, status) &
            result(kernel)
        !! Construct a smooth change-point kernel from two valid child kernels.
        !!
        !! For feature ``d`` the gate is
        !! ``s(x)=1/2(1+tanh((x_d-center)/width))`` and the covariance is
        !! ``s(x)s(x') k_left + (1-s(x))(1-s(x')) k_right``.  This is a
        !! positive-semidefinite sum of gated child covariances and remains a
        !! first-class kernel expression for exact GP inference.
        type(kernel_t), intent(in) :: left, right
        integer, intent(in) :: feature
        real(dp), intent(in) :: center, width
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        if (.not. kernel_valid(left) .or. .not. kernel_valid(right) .or. &
            left%input_dim /= right%input_dim .or. feature < 1 .or. &
            feature > left%input_dim .or. width <= 0.0_dp .or. &
            .not. ieee_is_finite(center) .or. .not. ieee_is_finite(width)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "change-point constructor: children, feature, center, and width are invalid")
            return
        end if
        kernel%kind = KERNEL_CHANGE_POINT
        kernel%input_dim = left%input_dim
        kernel%change_feature = feature
        allocate(kernel%left, source=left)
        allocate(kernel%right, source=right)
        allocate(kernel%log_parameters(2))
        kernel%log_parameters = [log(width), center]
        call status_set(status, FORTNUM_OK, "")
    end function make_change_point_kernel

    function make_rational_quadratic_kernel( &
            input_dim, variance, lengthscale, alpha, status) result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, lengthscale, alpha
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_three_parameter_leaf(kernel, KERNEL_RATIONAL_QUADRATIC, input_dim, &
            variance, lengthscale, alpha, status)
    end function make_rational_quadratic_kernel

    function make_cosine_kernel(input_dim, variance, lengthscale, status) result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, lengthscale
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_leaf(kernel, KERNEL_COSINE, input_dim, variance, lengthscale, status)
    end function make_cosine_kernel

    function make_polynomial_kernel(input_dim, variance, scale, offset, degree, status) &
            result(kernel)
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance, scale, offset, degree
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        if (input_dim < 1 .or. variance <= 0.0_dp .or. scale <= 0.0_dp .or. &
            offset <= 0.0_dp .or. degree < 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "polynomial kernel constructor: dimensions and scales must be positive")
            return
        end if
        kernel%kind = KERNEL_POLYNOMIAL
        kernel%input_dim = input_dim
        allocate(kernel%log_parameters(4))
        kernel%log_parameters = [log(variance), log(scale), log(offset), log(degree)]
        call status_set(status, FORTNUM_OK, "")
    end function make_polynomial_kernel

    function make_user_kernel(input_dim, variance, formula, status) result(kernel)
        !! Build a leaf from a user formula. The formula must already validate:
        !! this is the refusal boundary, so nothing downstream has to decide
        !! whether a user expression is safe to lower.
        integer, intent(in) :: input_dim
        real(dp), intent(in) :: variance
        type(kernel_formula_t), intent(in) :: formula
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        if (.not. formula%static_lowering_eligible()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel constructor: the user formula is not validated")
            return
        end if
        if (input_dim < 1 .or. variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel constructor: dimensions and scales must be positive")
            return
        end if
        kernel%kind = KERNEL_USER
        kernel%input_dim = input_dim
        allocate(kernel%log_parameters(1))
        kernel%log_parameters(1) = log(variance)
        allocate(kernel%formula, source=formula)
        call status_set(status, FORTNUM_OK, "")
    end function make_user_kernel

    function kernel_add(left, right, status) result(kernel)
        type(kernel_t), intent(in) :: left, right
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_composite(kernel, KERNEL_SUM, left, right, status)
    end function kernel_add

    function kernel_multiply(left, right, status) result(kernel)
        type(kernel_t), intent(in) :: left, right
        type(fortnum_status_t), intent(out) :: status
        type(kernel_t) :: kernel

        call make_composite(kernel, KERNEL_PRODUCT, left, right, status)
    end function kernel_multiply

    recursive function clone_kernel(source) result(copy)
        !! Return an independent copy of a kernel expression tree.
        !!
        !! Composite kernels own their children through pointers.  Intrinsic
        !! assignment therefore aliases those children, which is surprising
        !! for any derivative or optimizer probe that temporarily changes a
        !! parameter.  This explicit clone preserves the tree while making
        !! every mutable node independent.
        type(kernel_t), intent(in) :: source
        type(kernel_t) :: copy

        call clone_kernel_into(source, copy)
    end function clone_kernel

    recursive subroutine clone_kernel_into(source, copy)
        type(kernel_t), intent(in) :: source
        type(kernel_t), intent(out) :: copy

        copy%kind = source%kind
        copy%input_dim = source%input_dim
        copy%change_feature = source%change_feature
        if (allocated(source%log_parameters)) then
            allocate(copy%log_parameters, source=source%log_parameters)
        end if
        if (allocated(source%formula)) then
            allocate(copy%formula, source=source%formula)
        end if
        if (associated(source%left)) then
            allocate(copy%left)
            call clone_kernel_into(source%left, copy%left)
        end if
        if (associated(source%right)) then
            allocate(copy%right)
            call clone_kernel_into(source%right, copy%right)
        end if
    end subroutine clone_kernel_into

    recursive integer function kernel_parameter_count(self) result(count)
        class(kernel_t), intent(in) :: self

        count = 0
        select case (self%kind)
        case (KERNEL_RBF_ARD)
            count = self%input_dim + 1
        case (KERNEL_SPECTRAL_MIXTURE)
            count = size(self%log_parameters)
        case (KERNEL_RBF, KERNEL_MATERN12, KERNEL_MATERN32, KERNEL_MATERN52, &
                KERNEL_COSINE)
            count = 2
        case (KERNEL_PERIODIC, KERNEL_RATIONAL_QUADRATIC)
            count = 3
        case (KERNEL_LOCAL_PERIODIC)
            count = 4
        case (KERNEL_CHANGE_POINT)
            if (associated(self%left)) count = count + self%left%parameter_count()
            if (associated(self%right)) count = count + self%right%parameter_count()
            count = count + 2
        case (KERNEL_POLYNOMIAL)
            count = 4
        case (KERNEL_LINEAR, KERNEL_CONSTANT, KERNEL_WHITE_NOISE, KERNEL_USER)
            count = 1
        case (KERNEL_SUM, KERNEL_PRODUCT)
            if (associated(self%left)) count = count + self%left%parameter_count()
            if (associated(self%right)) count = count + self%right%parameter_count()
        end select
    end function kernel_parameter_count

    recursive function kernel_parameters(self) result(parameters)
        class(kernel_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: left_count, right_count

        allocate(parameters(self%parameter_count()))
        if (self%kind == KERNEL_SUM .or. self%kind == KERNEL_PRODUCT .or. &
            self%kind == KERNEL_CHANGE_POINT) then
            left_count = self%left%parameter_count()
            parameters(1:left_count) = self%left%parameters()
            right_count = self%right%parameter_count()
            parameters(left_count + 1:left_count + right_count) = self%right%parameters()
            if (self%kind == KERNEL_CHANGE_POINT) then
                parameters(left_count + right_count + 1:) = self%log_parameters
            end if
        else if (size(parameters) > 0) then
            parameters = self%log_parameters
        end if
    end function kernel_parameters

    recursive subroutine kernel_set_parameters(self, parameters, status)
        class(kernel_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: left_count, right_count

        if (.not. kernel_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel set_parameters: kernel is not initialized")
            return
        end if
        if (size(parameters) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel set_parameters: parameter shape is invalid")
            return
        end if

        if (self%kind == KERNEL_SUM .or. self%kind == KERNEL_PRODUCT) then
            left_count = self%left%parameter_count()
            call self%left%set_parameters(parameters(:left_count), status)
            if (status%code /= FORTNUM_OK) return
            call self%right%set_parameters(parameters(left_count + 1:), status)
        else if (self%kind == KERNEL_CHANGE_POINT) then
            left_count = self%left%parameter_count()
            right_count = self%right%parameter_count()
            call self%left%set_parameters(parameters(:left_count), status)
            if (status%code /= FORTNUM_OK) return
            call self%right%set_parameters(parameters(left_count + 1:left_count + right_count), &
                status)
            if (status%code /= FORTNUM_OK) return
            self%log_parameters = parameters(left_count + right_count + 1:)
        else
            self%log_parameters = parameters
            call status_set(status, FORTNUM_OK, "")
        end if
    end subroutine kernel_set_parameters

    subroutine kernel_matrix(self, x1, x2, matrix, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :)
        real(dp), intent(out) :: matrix(:, :)
        type(fortnum_status_t), intent(out) :: status

        call check_matrix_shapes(self, x1, x2, matrix, status)
        if (status%code /= FORTNUM_OK) return
        call kernel_matrix_impl(self, x1, x2, matrix)
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_matrix

    recursive function kernel_value(self, x1, x2) result(value)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp) :: value
        real(dp) :: variance, lengthscale, third_parameter, r2, length_derivative
        real(dp) :: polynomial_scale, polynomial_offset, polynomial_degree
        real(dp) :: parameter_derivative_1, parameter_derivative_2

        value = 0.0_dp
        if (size(x1) /= self%input_dim) return
        if (size(x2) /= self%input_dim) return
        if (.not. kernel_valid(self)) return
        select case (self%kind)
        case (KERNEL_SUM)
            value = self%left%value(x1, x2) + self%right%value(x1, x2)
        case (KERNEL_PRODUCT)
            value = self%left%value(x1, x2)*self%right%value(x1, x2)
        case (KERNEL_RBF_ARD)
            value = ard_rbf_value(self, x1, x2)
        case (KERNEL_SPECTRAL_MIXTURE)
            value = spectral_value(self, x1, x2)
        case (KERNEL_LOCAL_PERIODIC)
            value = local_periodic_value(self, x1, x2)
        case (KERNEL_CHANGE_POINT)
            value = change_point_value(self, x1, x2)
        case default
            variance = exp(self%log_parameters(1))
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52 .or. &
                self%kind == KERNEL_PERIODIC .or. self%kind == KERNEL_COSINE .or. &
                self%kind == KERNEL_RATIONAL_QUADRATIC) then
                lengthscale = exp(self%log_parameters(2))
            else
                lengthscale = 1.0_dp
            end if
            third_parameter = 1.0_dp
            if (self%kind == KERNEL_PERIODIC .or. &
                self%kind == KERNEL_RATIONAL_QUADRATIC) then
                third_parameter = exp(self%log_parameters(3))
            end if
            r2 = sum((x1 - x2)**2)
            value = 0.0_dp
            if (self%kind == KERNEL_RBF) then
                call fortml_generated_rbf_leaf_fortran( &
                    variance, r2, lengthscale, value)
            else if (self%kind == KERNEL_PERIODIC) then
                call periodic_value_derivatives(variance, lengthscale, third_parameter, &
                    r2, value, parameter_derivative_1, parameter_derivative_2, &
                    length_derivative)
            else if (self%kind == KERNEL_RATIONAL_QUADRATIC) then
                call rational_quadratic_value_derivatives(variance, lengthscale, &
                    third_parameter, r2, value, parameter_derivative_1, &
                    parameter_derivative_2, length_derivative)
            else if (self%kind == KERNEL_COSINE) then
                call cosine_value_derivatives(variance, lengthscale, r2, value, &
                    parameter_derivative_1, parameter_derivative_2)
            else if (self%kind == KERNEL_POLYNOMIAL) then
                polynomial_scale = exp(self%log_parameters(2))
                polynomial_offset = exp(self%log_parameters(3))
                polynomial_degree = exp(self%log_parameters(4))
                call polynomial_value_derivatives(variance, polynomial_scale, &
                    polynomial_offset, polynomial_degree, dot_product(x1, x2), value, &
                    parameter_derivative_1, parameter_derivative_2, length_derivative, &
                    third_parameter)
            else if (self%kind /= KERNEL_USER) then
                call leaf_value_and_length_derivative(self%kind, variance, &
                    lengthscale, r2, value, length_derivative)
            end if
            if (self%kind == KERNEL_USER) then
                value = variance*self%formula%evaluate(r2, dot_product(x1, x2))
            else if (self%kind == KERNEL_LINEAR) then
                value = variance*dot_product(x1, x2)
            else if (self%kind == KERNEL_CONSTANT) then
                value = variance
            else if (self%kind == KERNEL_WHITE_NOISE) then
                value = variance*merge(1.0_dp, 0.0_dp, all(x1 == x2))
            end if
        end select
    end function kernel_value

    subroutine change_point_gate(self, x1, x2, gate_left, gate_right, &
            gate_left_dot, gate_right_dot, direction)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: gate_left, gate_right
        real(dp), intent(out), optional :: gate_left_dot, gate_right_dot
        real(dp), intent(in), optional :: direction(:)
        real(dp) :: width, center, z1, z2, s1, s2, zdot1, zdot2
        integer :: feature

        feature = self%change_feature
        width = exp(self%log_parameters(1))
        center = self%log_parameters(2)
        z1 = (x1(feature) - center)/width
        z2 = (x2(feature) - center)/width
        s1 = 0.5_dp*(1.0_dp + tanh(z1))
        s2 = 0.5_dp*(1.0_dp + tanh(z2))
        gate_left = s1*s2
        gate_right = (1.0_dp - s1)*(1.0_dp - s2)
        if (present(gate_left_dot) .and. present(gate_right_dot) .and. &
            present(direction)) then
            zdot1 = -z1*direction(1) - direction(2)/width
            zdot2 = -z2*direction(1) - direction(2)/width
            gate_left_dot = 0.5_dp*(1.0_dp - tanh(z1)**2)*zdot1*s2 + &
                s1*0.5_dp*(1.0_dp - tanh(z2)**2)*zdot2
            gate_right_dot = -0.5_dp*(1.0_dp - tanh(z1)**2)*zdot1*(1.0_dp - s2) - &
                (1.0_dp - s1)*0.5_dp*(1.0_dp - tanh(z2)**2)*zdot2
        end if
    end subroutine change_point_gate

    real(dp) function change_point_value(self, x1, x2) result(value)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp) :: gate_left, gate_right

        call change_point_gate(self, x1, x2, gate_left, gate_right)
        value = gate_left*self%left%value(x1, x2) + gate_right*self%right%value(x1, x2)
    end function change_point_value

    recursive subroutine kernel_input_derivatives( &
            self, x1, x2, value, gradient_x1, gradient_x2, mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        gradient_x1 = 0.0_dp
        gradient_x2 = 0.0_dp
        mixed_hessian = 0.0_dp
        if (.not. kernel_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel input_derivatives: kernel is not initialized")
            return
        end if
        if (size(x1) /= self%input_dim .or. size(x2) /= self%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel input_derivatives: input dimension is invalid")
            return
        end if
        if (size(gradient_x1) /= self%input_dim .or. &
            size(gradient_x2) /= self%input_dim .or. &
            any(shape(mixed_hessian) /= [self%input_dim, self%input_dim])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel input_derivatives: output shape is invalid")
            return
        end if
        call kernel_input_derivatives_impl( &
            self, x1, x2, value, gradient_x1, gradient_x2, mixed_hessian, status)
    end subroutine kernel_input_derivatives

    subroutine kernel_matrix_jvp(self, x1, x2, direction, matrix, matrix_dot, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), direction(:)
        real(dp), intent(out) :: matrix(:, :), matrix_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        call check_matrix_shapes(self, x1, x2, matrix, status)
        if (status%code /= FORTNUM_OK) return
        if (any(shape(matrix_dot) /= shape(matrix))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel matrix_jvp: tangent output shape is invalid")
            return
        end if
        if (size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel matrix_jvp: parameter tangent shape is invalid")
            return
        end if
        call kernel_matrix_jvp_impl(self, x1, x2, direction, matrix, matrix_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_matrix_jvp

    subroutine kernel_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        call check_matrix_shapes(self, x1, x2, matrix_bar, status)
        if (status%code /= FORTNUM_OK) return
        if (size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel parameter_vjp: output shape is invalid")
            return
        end if
        parameter_bar = 0.0_dp
        call kernel_parameter_vjp_impl(self, x1, x2, matrix_bar, parameter_bar, 1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_parameter_vjp

    subroutine kernel_parameter_hvp( &
            self, x1, x2, matrix_bar, direction, parameter_bar, &
            parameter_bar_dot, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :), direction(:)
        real(dp), intent(out) :: parameter_bar(:), parameter_bar_dot(:)
        type(fortnum_status_t), intent(out) :: status

        call check_matrix_shapes(self, x1, x2, matrix_bar, status)
        if (status%code /= FORTNUM_OK) return
        if (size(direction) /= self%parameter_count() .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            size(parameter_bar_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel parameter_hvp: parameter shape is invalid")
            return
        end if
        parameter_bar = 0.0_dp
        parameter_bar_dot = 0.0_dp
        call kernel_parameter_hvp_impl(self, x1, x2, matrix_bar, direction, &
            parameter_bar, parameter_bar_dot, 1, status)
    end subroutine kernel_parameter_hvp

    recursive subroutine kernel_parameter_hvp_impl( &
            self, x1, x2, matrix_bar, direction, parameter_bar, &
            parameter_bar_dot, offset, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :), direction(:)
        real(dp), intent(inout) :: parameter_bar(:), parameter_bar_dot(:)
        integer, intent(in) :: offset
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: left_matrix(:, :), left_matrix_dot(:, :)
        real(dp), allocatable :: right_matrix(:, :), right_matrix_dot(:, :)
        real(dp) :: variance_log, lengthscale_log, r2, distance
        real(dp) :: value, value_dot, x1_bar, x1_bar_dot
        real(dp) :: x2_bar, x2_bar_dot, variance_bar, variance_bar_dot
        real(dp) :: lengthscale_bar, lengthscale_bar_dot
        real(dp) :: distance_bar, distance_bar_dot
        real(dp) :: third_log, third_log_dot
        real(dp) :: derivative_1, derivative_2, derivative_3, derivative_4
        real(dp) :: derivative_1_dot, derivative_2_dot, derivative_3_dot, derivative_4_dot
        real(dp) :: polynomial_scale, polynomial_offset, polynomial_degree, inner_product
        integer :: i, j, left_count

        select case (self%kind)
        case (KERNEL_SUM)
            left_count = self%left%parameter_count()
            call kernel_parameter_hvp_impl( &
                self%left, x1, x2, matrix_bar, direction(:left_count), &
                parameter_bar, parameter_bar_dot, offset, status)
            if (status%code /= FORTNUM_OK) return
            call kernel_parameter_hvp_impl( &
                self%right, x1, x2, matrix_bar, direction(left_count + 1:), &
                parameter_bar, parameter_bar_dot, offset + left_count, status)
            return
        case (KERNEL_PRODUCT)
            left_count = self%left%parameter_count()
            allocate(left_matrix(size(x1, 1), size(x2, 1)))
            allocate(left_matrix_dot, mold=left_matrix)
            allocate(right_matrix, mold=left_matrix)
            allocate(right_matrix_dot, mold=left_matrix)
            call kernel_matrix_jvp_impl(self%left, x1, x2, direction(:left_count), &
                left_matrix, left_matrix_dot)
            call kernel_matrix_jvp_impl(self%right, x1, x2, direction(left_count + 1:), &
                right_matrix, right_matrix_dot)
            call kernel_parameter_hvp_impl( &
                self%left, x1, x2, matrix_bar*right_matrix, direction(:left_count), &
                parameter_bar, parameter_bar_dot, offset, status)
            if (status%code /= FORTNUM_OK) return
            call kernel_parameter_vjp_impl( &
                self%left, x1, x2, matrix_bar*right_matrix_dot, parameter_bar_dot, offset)
            call kernel_parameter_hvp_impl( &
                self%right, x1, x2, matrix_bar*left_matrix, direction(left_count + 1:), &
                parameter_bar, parameter_bar_dot, offset + left_count, status)
            if (status%code /= FORTNUM_OK) return
            call kernel_parameter_vjp_impl( &
                self%right, x1, x2, matrix_bar*left_matrix_dot, parameter_bar_dot, &
                offset + left_count)
            call status_set(status, FORTNUM_OK, "")
            return
        case (KERNEL_RBF_ARD)
            call ard_rbf_parameter_hvp(self, x1, x2, matrix_bar, direction, &
                parameter_bar, parameter_bar_dot, offset, status)
            return
        case (KERNEL_SPECTRAL_MIXTURE)
            call spectral_parameter_hvp(self, x1, x2, matrix_bar, direction, &
                parameter_bar, parameter_bar_dot, offset, status)
            return
        case (KERNEL_LOCAL_PERIODIC)
            call local_periodic_parameter_hvp(self, x1, x2, matrix_bar, direction, &
                parameter_bar, parameter_bar_dot, offset, status)
            return
        case (KERNEL_CHANGE_POINT)
            call change_point_parameter_hvp(self, x1, x2, matrix_bar, direction, &
                parameter_bar, parameter_bar_dot, offset, status)
            return
        case default
            continue
        end select

        variance_log = self%log_parameters(1)
        if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
            self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52 .or. &
            self%kind == KERNEL_PERIODIC .or. self%kind == KERNEL_COSINE .or. &
            self%kind == KERNEL_RATIONAL_QUADRATIC) then
            lengthscale_log = self%log_parameters(2)
        else
            lengthscale_log = 0.0_dp
        end if
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                r2 = sum((x1(i, :) - x2(j, :))**2)
                distance = sqrt(r2)
                select case (self%kind)
                case (KERNEL_RBF)
                    call fortml_rbf_hvp( &
                        distance, 0.0_dp, 0.0_dp, 0.0_dp, variance_log, direction(1), &
                        lengthscale_log, direction(2), value, value_dot, matrix_bar(i, j), &
                        x1_bar, x1_bar_dot, x2_bar, x2_bar_dot, variance_bar, &
                        variance_bar_dot, lengthscale_bar, lengthscale_bar_dot)
                    parameter_bar(offset) = parameter_bar(offset) + variance_bar
                    parameter_bar(offset + 1) = parameter_bar(offset + 1) + lengthscale_bar
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + variance_bar_dot
                    parameter_bar_dot(offset + 1) = parameter_bar_dot(offset + 1) + &
                        lengthscale_bar_dot
                case (KERNEL_MATERN12)
                    call fortml_matern12_hvp( &
                        distance, 0.0_dp, variance_log, direction(1), lengthscale_log, &
                        direction(2), value, value_dot, matrix_bar(i, j), distance_bar, &
                        distance_bar_dot, variance_bar, variance_bar_dot, lengthscale_bar, &
                        lengthscale_bar_dot)
                    parameter_bar(offset) = parameter_bar(offset) + variance_bar
                    parameter_bar(offset + 1) = parameter_bar(offset + 1) + lengthscale_bar
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + variance_bar_dot
                    parameter_bar_dot(offset + 1) = parameter_bar_dot(offset + 1) + &
                        lengthscale_bar_dot
                case (KERNEL_MATERN32)
                    call fortml_matern32_hvp( &
                        distance, 0.0_dp, variance_log, direction(1), lengthscale_log, &
                        direction(2), value, value_dot, matrix_bar(i, j), distance_bar, &
                        distance_bar_dot, variance_bar, variance_bar_dot, lengthscale_bar, &
                        lengthscale_bar_dot)
                    parameter_bar(offset) = parameter_bar(offset) + variance_bar
                    parameter_bar(offset + 1) = parameter_bar(offset + 1) + lengthscale_bar
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + variance_bar_dot
                    parameter_bar_dot(offset + 1) = parameter_bar_dot(offset + 1) + &
                        lengthscale_bar_dot
                case (KERNEL_MATERN52)
                    call fortml_matern52_hvp( &
                        distance, 0.0_dp, variance_log, direction(1), lengthscale_log, &
                        direction(2), value, value_dot, matrix_bar(i, j), distance_bar, &
                        distance_bar_dot, variance_bar, variance_bar_dot, lengthscale_bar, &
                        lengthscale_bar_dot)
                    parameter_bar(offset) = parameter_bar(offset) + variance_bar
                    parameter_bar(offset + 1) = parameter_bar(offset + 1) + lengthscale_bar
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + variance_bar_dot
                    parameter_bar_dot(offset + 1) = parameter_bar_dot(offset + 1) + &
                        lengthscale_bar_dot
                case (KERNEL_PERIODIC)
                    third_log = self%log_parameters(3)
                    third_log_dot = direction(3)
                    call periodic_value_hvp(variance_log, lengthscale_log, third_log, &
                        direction(1), direction(2), third_log_dot, r2, value, value_dot, &
                        derivative_1, derivative_1_dot, derivative_2, derivative_2_dot, &
                        derivative_3, derivative_3_dot)
                    parameter_bar(offset) = parameter_bar(offset) + &
                        matrix_bar(i, j)*derivative_1
                    parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                        matrix_bar(i, j)*derivative_2
                    parameter_bar(offset + 2) = parameter_bar(offset + 2) + &
                        matrix_bar(i, j)*derivative_3
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + &
                        matrix_bar(i, j)*derivative_1_dot
                    parameter_bar_dot(offset + 1) = parameter_bar_dot(offset + 1) + &
                        matrix_bar(i, j)*derivative_2_dot
                    parameter_bar_dot(offset + 2) = parameter_bar_dot(offset + 2) + &
                        matrix_bar(i, j)*derivative_3_dot
                case (KERNEL_RATIONAL_QUADRATIC)
                    third_log = self%log_parameters(3)
                    third_log_dot = direction(3)
                    call rational_quadratic_value_hvp(variance_log, lengthscale_log, &
                        third_log, direction(1), direction(2), third_log_dot, r2, value, &
                        value_dot, derivative_1, derivative_1_dot, derivative_2, &
                        derivative_2_dot, derivative_3, derivative_3_dot)
                    parameter_bar(offset) = parameter_bar(offset) + &
                        matrix_bar(i, j)*derivative_1
                    parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                        matrix_bar(i, j)*derivative_2
                    parameter_bar(offset + 2) = parameter_bar(offset + 2) + &
                        matrix_bar(i, j)*derivative_3
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + &
                        matrix_bar(i, j)*derivative_1_dot
                    parameter_bar_dot(offset + 1) = parameter_bar_dot(offset + 1) + &
                        matrix_bar(i, j)*derivative_2_dot
                    parameter_bar_dot(offset + 2) = parameter_bar_dot(offset + 2) + &
                        matrix_bar(i, j)*derivative_3_dot
                case (KERNEL_COSINE)
                    call cosine_value_hvp(variance_log, lengthscale_log, direction(1), &
                        direction(2), r2, value, value_dot, derivative_1, derivative_1_dot, &
                        derivative_2, derivative_2_dot)
                    parameter_bar(offset) = parameter_bar(offset) + &
                        matrix_bar(i, j)*derivative_1
                    parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                        matrix_bar(i, j)*derivative_2
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + &
                        matrix_bar(i, j)*derivative_1_dot
                    parameter_bar_dot(offset + 1) = parameter_bar_dot(offset + 1) + &
                        matrix_bar(i, j)*derivative_2_dot
                case (KERNEL_POLYNOMIAL)
                    polynomial_scale = exp(self%log_parameters(2))
                    polynomial_offset = exp(self%log_parameters(3))
                    polynomial_degree = exp(self%log_parameters(4))
                    inner_product = dot_product(x1(i, :), x2(j, :))
                    call polynomial_value_hvp(variance_log, self%log_parameters(2), &
                        self%log_parameters(3), self%log_parameters(4), direction(1), &
                        direction(2), direction(3), direction(4), inner_product, value, &
                        value_dot, derivative_1, derivative_1_dot, derivative_2, &
                        derivative_2_dot, derivative_3, derivative_3_dot, derivative_4, &
                        derivative_4_dot)
                    parameter_bar(offset) = parameter_bar(offset) + &
                        matrix_bar(i, j)*derivative_1
                    parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                        matrix_bar(i, j)*derivative_2
                    parameter_bar(offset + 2) = parameter_bar(offset + 2) + &
                        matrix_bar(i, j)*derivative_3
                    parameter_bar(offset + 3) = parameter_bar(offset + 3) + &
                        matrix_bar(i, j)*derivative_4
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + &
                        matrix_bar(i, j)*derivative_1_dot
                    parameter_bar_dot(offset + 1) = parameter_bar_dot(offset + 1) + &
                        matrix_bar(i, j)*derivative_2_dot
                    parameter_bar_dot(offset + 2) = parameter_bar_dot(offset + 2) + &
                        matrix_bar(i, j)*derivative_3_dot
                    parameter_bar_dot(offset + 3) = parameter_bar_dot(offset + 3) + &
                        matrix_bar(i, j)*derivative_4_dot
                case (KERNEL_LINEAR)
                    value = exp(variance_log)*dot_product(x1(i, :), x2(j, :))
                    parameter_bar(offset) = parameter_bar(offset) + matrix_bar(i, j)*value
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + &
                        matrix_bar(i, j)*value*direction(1)
                case (KERNEL_CONSTANT)
                    value = exp(variance_log)
                    parameter_bar(offset) = parameter_bar(offset) + matrix_bar(i, j)*value
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + &
                        matrix_bar(i, j)*value*direction(1)
                case (KERNEL_WHITE_NOISE)
                    value = exp(variance_log)*merge(1.0_dp, 0.0_dp, &
                        same_row(x1, i, x2, j))
                    parameter_bar(offset) = parameter_bar(offset) + matrix_bar(i, j)*value
                    parameter_bar_dot(offset) = parameter_bar_dot(offset) + &
                        matrix_bar(i, j)*value*direction(1)
                case default
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel parameter_hvp: unsupported kernel kind")
                    return
                end select
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine kernel_parameter_hvp_impl

    subroutine change_point_parameter_hvp(self, x1, x2, matrix_bar, direction, &
            parameter_bar, parameter_bar_dot, offset, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :), direction(:)
        real(dp), intent(inout) :: parameter_bar(:), parameter_bar_dot(:)
        integer, intent(in) :: offset
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: left_matrix(:, :), right_matrix(:, :)
        real(dp), allocatable :: left_matrix_dot(:, :), right_matrix_dot(:, :)
        real(dp), allocatable :: gate_left_matrix(:, :), gate_right_matrix(:, :)
        real(dp), allocatable :: gate_left_matrix_dot(:, :), gate_right_matrix_dot(:, :)
        real(dp) :: gate_left, gate_right, gate_left_dot, gate_right_dot
        real(dp) :: gate_gradient_left(2), gate_gradient_right(2)
        real(dp) :: gate_gradient_left_dot(2), gate_gradient_right_dot(2)
        real(dp) :: left_value, right_value, left_value_dot, right_value_dot
        integer :: left_count, right_count, i, j

        left_count = self%left%parameter_count()
        right_count = self%right%parameter_count()
        allocate(left_matrix(size(x1, 1), size(x2, 1)))
        allocate(right_matrix, mold=left_matrix)
        allocate(left_matrix_dot, mold=left_matrix)
        allocate(right_matrix_dot, mold=left_matrix)
        allocate(gate_left_matrix, mold=left_matrix)
        allocate(gate_right_matrix, mold=left_matrix)
        allocate(gate_left_matrix_dot, mold=left_matrix)
        allocate(gate_right_matrix_dot, mold=left_matrix)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                call change_point_gate(self, x1(i, :), x2(j, :), gate_left_matrix(i, j), &
                    gate_right_matrix(i, j), gate_left_matrix_dot(i, j), &
                    gate_right_matrix_dot(i, j), direction(left_count + right_count + 1: &
                    left_count + right_count + 2))
            end do
        end do
        call kernel_matrix_jvp_impl(self%left, x1, x2, direction(:left_count), &
            left_matrix, left_matrix_dot)
        call kernel_matrix_jvp_impl(self%right, x1, x2, &
            direction(left_count + 1:left_count + right_count), right_matrix, right_matrix_dot)
        call kernel_parameter_hvp_impl(self%left, x1, x2, matrix_bar*gate_left_matrix, &
            direction(:left_count), parameter_bar, parameter_bar_dot, offset, status)
        if (status%code /= FORTNUM_OK) return
        call kernel_parameter_vjp_impl(self%left, x1, x2, matrix_bar*gate_left_matrix_dot, &
            parameter_bar_dot, offset)
        call kernel_parameter_hvp_impl(self%right, x1, x2, matrix_bar*gate_right_matrix, &
            direction(left_count + 1:left_count + right_count), parameter_bar, &
            parameter_bar_dot, offset + left_count, status)
        if (status%code /= FORTNUM_OK) return
        call kernel_parameter_vjp_impl(self%right, x1, x2, matrix_bar*gate_right_matrix_dot, &
            parameter_bar_dot, offset + left_count)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                left_value = left_matrix(i, j)
                right_value = right_matrix(i, j)
                left_value_dot = left_matrix_dot(i, j)
                right_value_dot = right_matrix_dot(i, j)
                call change_point_parameter_gate_hvp(self, x1(i, :), x2(j, :), &
                    direction(left_count + right_count + 1:left_count + right_count + 2), &
                    gate_left, gate_right, gate_left_dot, gate_right_dot, &
                    gate_gradient_left, gate_gradient_right, gate_gradient_left_dot, &
                    gate_gradient_right_dot)
                parameter_bar(offset + left_count + right_count) = &
                    parameter_bar(offset + left_count + right_count) + matrix_bar(i, j)* &
                    (gate_gradient_left(1)*left_value + gate_gradient_right(1)*right_value)
                parameter_bar(offset + left_count + right_count + 1) = &
                    parameter_bar(offset + left_count + right_count + 1) + matrix_bar(i, j)* &
                    (gate_gradient_left(2)*left_value + gate_gradient_right(2)*right_value)
                parameter_bar_dot(offset + left_count + right_count) = &
                    parameter_bar_dot(offset + left_count + right_count) + matrix_bar(i, j)* &
                    (gate_gradient_left_dot(1)*left_value + gate_gradient_left(1)*left_value_dot + &
                    gate_gradient_right_dot(1)*right_value + gate_gradient_right(1)*right_value_dot)
                parameter_bar_dot(offset + left_count + right_count + 1) = &
                    parameter_bar_dot(offset + left_count + right_count + 1) + matrix_bar(i, j)* &
                    (gate_gradient_left_dot(2)*left_value + gate_gradient_left(2)*left_value_dot + &
                    gate_gradient_right_dot(2)*right_value + gate_gradient_right(2)*right_value_dot)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine change_point_parameter_hvp

    subroutine change_point_parameter_gate_hvp(self, x1, x2, direction, gate_left, &
            gate_right, gate_left_dot, gate_right_dot, gate_gradient_left, &
            gate_gradient_right, gate_gradient_left_dot, gate_gradient_right_dot)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:), direction(2)
        real(dp), intent(out) :: gate_left, gate_right, gate_left_dot, gate_right_dot
        real(dp), intent(out) :: gate_gradient_left(2), gate_gradient_right(2)
        real(dp), intent(out) :: gate_gradient_left_dot(2), gate_gradient_right_dot(2)
        real(dp) :: width, center, z1, z2, s1, s2, q1, q2, s1_dot, s2_dot
        real(dp) :: z1_dot, z2_dot, q1_dot, q2_dot, inv_width
        real(dp) :: gradient_1(2), gradient_2(2), gradient_1_dot(2), gradient_2_dot(2)
        integer :: feature

        width = exp(self%log_parameters(1))
        inv_width = 1.0_dp/width
        center = self%log_parameters(2)
        feature = self%change_feature
        z1 = (x1(feature) - center)*inv_width
        z2 = (x2(feature) - center)*inv_width
        s1 = 0.5_dp*(1.0_dp + tanh(z1))
        s2 = 0.5_dp*(1.0_dp + tanh(z2))
        q1 = 0.5_dp*(1.0_dp - tanh(z1)**2)
        q2 = 0.5_dp*(1.0_dp - tanh(z2)**2)
        z1_dot = -z1*direction(1) - inv_width*direction(2)
        z2_dot = -z2*direction(1) - inv_width*direction(2)
        s1_dot = q1*z1_dot
        s2_dot = q2*z2_dot
        q1_dot = -2.0_dp*tanh(z1)*q1*z1_dot
        q2_dot = -2.0_dp*tanh(z2)*q2*z2_dot
        gradient_1 = q1*[-z1, -inv_width]
        gradient_2 = q2*[-z2, -inv_width]
        gradient_1_dot = [q1_dot*(-z1) + q1*(-z1_dot), &
            q1_dot*(-inv_width) + q1*inv_width*direction(1)]
        gradient_2_dot = [q2_dot*(-z2) + q2*(-z2_dot), &
            q2_dot*(-inv_width) + q2*inv_width*direction(1)]
        gate_left = s1*s2
        gate_right = (1.0_dp - s1)*(1.0_dp - s2)
        gate_left_dot = s1_dot*s2 + s1*s2_dot
        gate_right_dot = -s1_dot*(1.0_dp - s2) - (1.0_dp - s1)*s2_dot
        gate_gradient_left = gradient_1*s2 + s1*gradient_2
        gate_gradient_right = -gradient_1*(1.0_dp - s2) - (1.0_dp - s1)*gradient_2
        gate_gradient_left_dot = gradient_1_dot*s2 + gradient_1*s2_dot + &
            s1_dot*gradient_2 + s1*gradient_2_dot
        gate_gradient_right_dot = -gradient_1_dot*(1.0_dp - s2) + gradient_1*s2_dot + &
            s1_dot*gradient_2 - (1.0_dp - s1)*gradient_2_dot
    end subroutine change_point_parameter_gate_hvp

    real(dp) function local_periodic_value(self, x1, x2) result(value)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp) :: derivative(4), squared_distance

        squared_distance = sum((x1 - x2)**2)
        call local_periodic_value_derivatives(self%log_parameters, squared_distance, &
            value, derivative)
    end function local_periodic_value

    subroutine local_periodic_value_derivatives(log_parameters, squared_distance, &
            value, derivative)
        real(dp), intent(in) :: log_parameters(:), squared_distance
        real(dp), intent(out) :: value, derivative(4)
        real(dp) :: variance, envelope_scale, periodic_scale, period
        real(dp) :: distance, argument, sine_value, cosine_value, pi
        real(dp) :: a, b
        real(dp) :: log_direction(4)

        variance = exp(log_parameters(1))
        envelope_scale = exp(log_parameters(2))
        periodic_scale = exp(log_parameters(3))
        period = exp(log_parameters(4))
        pi = acos(-1.0_dp)
        distance = sqrt(max(squared_distance, 0.0_dp))
        argument = pi*distance/period
        sine_value = sin(argument)
        cosine_value = cos(argument)
        a = 0.5_dp/(envelope_scale*envelope_scale)
        b = 2.0_dp/(periodic_scale*periodic_scale)
        value = variance*exp(-a*squared_distance - b*sine_value*sine_value)
        log_direction(1) = 1.0_dp
        log_direction(2) = 2.0_dp*a*squared_distance
        log_direction(3) = 2.0_dp*b*sine_value*sine_value
        log_direction(4) = 2.0_dp*b*argument*sine_value*cosine_value
        derivative = value*log_direction
    end subroutine local_periodic_value_derivatives

    subroutine local_periodic_value_hvp(log_parameters, direction, squared_distance, &
            value, value_dot, derivative, derivative_dot)
        real(dp), intent(in) :: log_parameters(:), direction(:), squared_distance
        real(dp), intent(out) :: value, value_dot, derivative(4), derivative_dot(4)
        real(dp) :: variance, envelope_scale, periodic_scale, period
        real(dp) :: distance, argument, sine_value, cosine_value, pi
        real(dp) :: sine_dot, cosine_dot, argument_dot, a, b, b_dot
        real(dp) :: log_derivative(4), log_derivative_dot(4), log_direction
        integer :: i

        variance = exp(log_parameters(1))
        envelope_scale = exp(log_parameters(2))
        periodic_scale = exp(log_parameters(3))
        period = exp(log_parameters(4))
        pi = acos(-1.0_dp)
        distance = sqrt(max(squared_distance, 0.0_dp))
        argument = pi*distance/period
        sine_value = sin(argument)
        cosine_value = cos(argument)
        a = 0.5_dp/(envelope_scale*envelope_scale)
        b = 2.0_dp/(periodic_scale*periodic_scale)
        argument_dot = -argument*direction(4)
        sine_dot = cosine_value*argument_dot
        cosine_dot = -sine_value*argument_dot
        b_dot = -2.0_dp*b*direction(3)
        value = variance*exp(-a*squared_distance - b*sine_value*sine_value)
        log_derivative(1) = 1.0_dp
        log_derivative(2) = 2.0_dp*a*squared_distance
        log_derivative(3) = 2.0_dp*b*sine_value*sine_value
        log_derivative(4) = 2.0_dp*b*argument*sine_value*cosine_value
        log_derivative_dot(1) = 0.0_dp
        log_derivative_dot(2) = -2.0_dp*log_derivative(2)*direction(2)
        log_derivative_dot(3) = 2.0_dp*(b_dot*sine_value*sine_value + &
            2.0_dp*b*sine_value*sine_dot)
        log_derivative_dot(4) = 2.0_dp*(b_dot*argument*sine_value*cosine_value + &
            b*(argument_dot*sine_value*cosine_value + argument*sine_dot*cosine_value + &
            argument*sine_value*cosine_dot))
        log_direction = 0.0_dp
        do i = 1, 4
            log_direction = log_direction + log_derivative(i)*direction(i)
        end do
        value_dot = value*log_direction
        derivative = value*log_derivative
        derivative_dot = value_dot*log_derivative + value*log_derivative_dot
    end subroutine local_periodic_value_hvp

    subroutine local_periodic_matrix(self, x1, x2, matrix)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :)
        real(dp), intent(out) :: matrix(:, :)
        integer :: i, j

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                matrix(i, j) = local_periodic_value(self, x1(i, :), x2(j, :))
            end do
        end do
    end subroutine local_periodic_matrix

    subroutine local_periodic_matrix_jvp(self, x1, x2, direction, matrix, matrix_dot)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), direction(:)
        real(dp), intent(out) :: matrix(:, :), matrix_dot(:, :)
        real(dp) :: derivative(4), derivative_dot(4), squared_distance
        integer :: i, j

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                squared_distance = sum((x1(i, :) - x2(j, :))**2)
                call local_periodic_value_hvp(self%log_parameters, direction, &
                    squared_distance, matrix(i, j), matrix_dot(i, j), derivative, derivative_dot)
            end do
        end do
    end subroutine local_periodic_matrix_jvp

    subroutine local_periodic_input_derivatives(self, x1, x2, value, gradient_x1, &
            gradient_x2, mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: difference(size(x1)), squared_distance, distance
        real(dp) :: parameter_derivative(4)
        real(dp) :: envelope_scale, periodic_scale, period, a, b, c
        real(dp) :: argument, sine_value, cosine_value, pi
        real(dp) :: first_r, second_r, first_t, second_t
        real(dp) :: log_r, log_rr
        integer :: i, j

        difference = x1 - x2
        squared_distance = sum(difference*difference)
        call local_periodic_value_derivatives(self%log_parameters, squared_distance, &
            value, parameter_derivative)
        envelope_scale = exp(self%log_parameters(2))
        periodic_scale = exp(self%log_parameters(3))
        period = exp(self%log_parameters(4))
        a = 0.5_dp/(envelope_scale*envelope_scale)
        b = 2.0_dp/(periodic_scale*periodic_scale)
        pi = acos(-1.0_dp)
        c = pi/period
        distance = sqrt(max(squared_distance, 0.0_dp))
        if (distance == 0.0_dp) then
            first_t = -value*(a + b*c*c)
            second_t = value*((a + b*c*c)**2 + 2.0_dp*b*c**4/3.0_dp)
        else
            argument = c*distance
            sine_value = sin(argument)
            cosine_value = cos(argument)
            log_r = -2.0_dp*a*distance - 2.0_dp*b*c*sine_value*cosine_value
            log_rr = -2.0_dp*a - 2.0_dp*b*c*c*(cosine_value*cosine_value - &
                sine_value*sine_value)
            first_r = value*log_r
            second_r = value*(log_r*log_r + log_rr)
            first_t = first_r/(2.0_dp*distance)
            second_t = (second_r - first_r/distance)/(4.0_dp*squared_distance)
        end if
        gradient_x1 = 2.0_dp*first_t*difference
        gradient_x2 = -gradient_x1
        do i = 1, size(x1)
            do j = 1, size(x2)
                mixed_hessian(i, j) = -2.0_dp*first_t*merge(1.0_dp, 0.0_dp, i == j) - &
                    4.0_dp*second_t*difference(i)*difference(j)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine local_periodic_input_derivatives

    subroutine local_periodic_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, offset)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        integer, intent(in) :: offset
        real(dp) :: derivative(4), value, squared_distance
        integer :: i, j, parameter

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                squared_distance = sum((x1(i, :) - x2(j, :))**2)
                call local_periodic_value_derivatives(self%log_parameters, squared_distance, &
                    value, derivative)
                do parameter = 1, 4
                    parameter_bar(offset + parameter - 1) = parameter_bar(offset + parameter - 1) + &
                        matrix_bar(i, j)*derivative(parameter)
                end do
            end do
        end do
    end subroutine local_periodic_parameter_vjp

    subroutine local_periodic_parameter_hvp(self, x1, x2, matrix_bar, direction, &
            parameter_bar, parameter_bar_dot, offset, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :), direction(:)
        real(dp), intent(inout) :: parameter_bar(:), parameter_bar_dot(:)
        integer, intent(in) :: offset
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: derivative(4), derivative_dot(4), value, value_dot, squared_distance
        integer :: i, j, parameter

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                squared_distance = sum((x1(i, :) - x2(j, :))**2)
                call local_periodic_value_hvp(self%log_parameters, direction, squared_distance, &
                    value, value_dot, derivative, derivative_dot)
                do parameter = 1, 4
                    parameter_bar(offset + parameter - 1) = parameter_bar(offset + parameter - 1) + &
                        matrix_bar(i, j)*derivative(parameter)
                    parameter_bar_dot(offset + parameter - 1) = &
                        parameter_bar_dot(offset + parameter - 1) + &
                        matrix_bar(i, j)*derivative_dot(parameter)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine local_periodic_parameter_hvp

    subroutine periodic_value_derivatives(variance, lengthscale, period, squared_distance, &
            value, derivative_variance, derivative_lengthscale, derivative_period)
        real(dp), intent(in) :: variance, lengthscale, period, squared_distance
        real(dp), intent(out) :: value, derivative_variance, derivative_lengthscale
        real(dp), intent(out) :: derivative_period
        real(dp) :: distance, argument, sine_value, cosine_value, inverse_length_squared

        distance = sqrt(max(squared_distance, 0.0_dp))
        argument = acos(-1.0_dp)*distance/period
        sine_value = sin(argument)
        cosine_value = cos(argument)
        inverse_length_squared = 1.0_dp/(lengthscale*lengthscale)
        value = variance*exp(-2.0_dp*inverse_length_squared*sine_value*sine_value)
        derivative_variance = value
        derivative_lengthscale = value*4.0_dp*inverse_length_squared*sine_value*sine_value
        derivative_period = value*4.0_dp*inverse_length_squared*argument* &
            sine_value*cosine_value
    end subroutine periodic_value_derivatives

    subroutine periodic_value_jvp(variance, lengthscale, period, squared_distance, &
            variance_dot, lengthscale_dot, period_dot, value, value_dot)
        real(dp), intent(in) :: variance, lengthscale, period, squared_distance
        real(dp), intent(in) :: variance_dot, lengthscale_dot, period_dot
        real(dp), intent(out) :: value, value_dot
        real(dp) :: derivative_variance, derivative_lengthscale, derivative_period

        call periodic_value_derivatives(variance, lengthscale, period, squared_distance, &
            value, derivative_variance, derivative_lengthscale, derivative_period)
        value_dot = derivative_variance*variance_dot + derivative_lengthscale*lengthscale_dot + &
            derivative_period*period_dot
    end subroutine periodic_value_jvp

    subroutine periodic_value_hvp(log_variance, log_lengthscale, log_period, &
            variance_dot, lengthscale_dot, period_dot, squared_distance, value, value_dot, &
            derivative_variance, derivative_variance_dot, derivative_lengthscale, &
            derivative_lengthscale_dot, derivative_period, derivative_period_dot)
        real(dp), intent(in) :: log_variance, log_lengthscale, log_period
        real(dp), intent(in) :: variance_dot, lengthscale_dot, period_dot, squared_distance
        real(dp), intent(out) :: value, value_dot, derivative_variance, derivative_variance_dot
        real(dp), intent(out) :: derivative_lengthscale, derivative_lengthscale_dot
        real(dp), intent(out) :: derivative_period, derivative_period_dot
        real(dp) :: variance, lengthscale, period, inverse_length_squared
        real(dp) :: distance, argument, sine_value, cosine_value
        real(dp) :: argument_dot, inverse_length_squared_dot
        real(dp) :: q, q_dot, g_lengthscale, g_period, g_lengthscale_dot, g_period_dot
        real(dp) :: log_direction

        variance = exp(log_variance)
        lengthscale = exp(log_lengthscale)
        period = exp(log_period)
        inverse_length_squared = exp(-2.0_dp*log_lengthscale)
        distance = sqrt(max(squared_distance, 0.0_dp))
        argument = acos(-1.0_dp)*distance/period
        sine_value = sin(argument)
        cosine_value = cos(argument)
        argument_dot = -argument*period_dot
        inverse_length_squared_dot = -2.0_dp*inverse_length_squared*lengthscale_dot
        q = 2.0_dp*inverse_length_squared*sine_value*sine_value
        q_dot = 2.0_dp*(inverse_length_squared_dot*sine_value*sine_value + &
            2.0_dp*inverse_length_squared*sine_value*cosine_value*argument_dot)
        value = variance*exp(-q)
        g_lengthscale = 2.0_dp*q
        g_period = 4.0_dp*inverse_length_squared*argument*sine_value*cosine_value
        g_lengthscale_dot = 2.0_dp*q_dot
        g_period_dot = 4.0_dp*(inverse_length_squared_dot*argument*sine_value*cosine_value + &
            inverse_length_squared*argument_dot*sine_value*cosine_value + &
            inverse_length_squared*argument*cosine_value*cosine_value*argument_dot - &
            inverse_length_squared*argument*sine_value*sine_value*argument_dot)
        log_direction = variance_dot + g_lengthscale*lengthscale_dot + g_period*period_dot
        value_dot = value*log_direction
        derivative_variance = value
        derivative_variance_dot = value_dot
        derivative_lengthscale = value*g_lengthscale
        derivative_lengthscale_dot = value*(log_direction*g_lengthscale + g_lengthscale_dot)
        derivative_period = value*g_period
        derivative_period_dot = value*(log_direction*g_period + g_period_dot)
    end subroutine periodic_value_hvp

    subroutine periodic_input_derivatives(self, x1, x2, value, gradient_x1, gradient_x2, &
            mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: variance, lengthscale, period, squared_distance, distance
        real(dp) :: argument, sine_value, cosine_value, inverse_length_squared, pi_over_period
        real(dp) :: radial_scale, radial_second, difference(size(x1))
        integer :: i, j

        variance = exp(self%log_parameters(1))
        lengthscale = exp(self%log_parameters(2))
        period = exp(self%log_parameters(3))
        difference = x1 - x2
        squared_distance = sum(difference**2)
        distance = sqrt(squared_distance)
        call periodic_value_derivatives(variance, lengthscale, period, squared_distance, &
            value, radial_scale, radial_second, argument)
        inverse_length_squared = 1.0_dp/(lengthscale*lengthscale)
        pi_over_period = acos(-1.0_dp)/period
        argument = pi_over_period*distance
        sine_value = sin(argument)
        cosine_value = cos(argument)
        if (distance == 0.0_dp) then
            radial_scale = -4.0_dp*value*inverse_length_squared*pi_over_period*pi_over_period
            gradient_x1 = 0.0_dp
            gradient_x2 = 0.0_dp
            mixed_hessian = 0.0_dp
            do i = 1, size(x1)
                mixed_hessian(i, i) = -radial_scale
            end do
        else
            radial_scale = -4.0_dp*value*inverse_length_squared*pi_over_period* &
                sine_value*cosine_value/distance
            radial_second = value*(16.0_dp*inverse_length_squared**2*pi_over_period**2* &
                sine_value**2*cosine_value**2 - 4.0_dp*inverse_length_squared* &
                pi_over_period**2*(cosine_value**2 - sine_value**2))
            do i = 1, size(x1)
                gradient_x1(i) = radial_scale*difference(i)
                gradient_x2(i) = -gradient_x1(i)
                do j = 1, size(x2)
                    mixed_hessian(i, j) = -(radial_scale*merge(1.0_dp, 0.0_dp, i == j) + &
                        (radial_second - radial_scale)*difference(i)*difference(j)/squared_distance)
                end do
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine periodic_input_derivatives

    subroutine cosine_value_derivatives(variance, lengthscale, squared_distance, &
            value, derivative_variance, derivative_lengthscale)
        real(dp), intent(in) :: variance, lengthscale, squared_distance
        real(dp), intent(out) :: value, derivative_variance, derivative_lengthscale
        real(dp) :: distance, argument

        distance = sqrt(max(squared_distance, 0.0_dp))
        argument = distance/lengthscale
        value = variance*cos(argument)
        derivative_variance = value
        derivative_lengthscale = variance*argument*sin(argument)
    end subroutine cosine_value_derivatives

    subroutine cosine_value_jvp(variance, lengthscale, squared_distance, &
            variance_dot, lengthscale_dot, value, value_dot)
        real(dp), intent(in) :: variance, lengthscale, squared_distance
        real(dp), intent(in) :: variance_dot, lengthscale_dot
        real(dp), intent(out) :: value, value_dot
        real(dp) :: derivative_variance, derivative_lengthscale

        call cosine_value_derivatives(variance, lengthscale, squared_distance, value, &
            derivative_variance, derivative_lengthscale)
        value_dot = derivative_variance*variance_dot + derivative_lengthscale*lengthscale_dot
    end subroutine cosine_value_jvp

    subroutine cosine_value_hvp(log_variance, log_lengthscale, variance_dot, &
            lengthscale_dot, squared_distance, value, value_dot, derivative_variance, &
            derivative_variance_dot, derivative_lengthscale, derivative_lengthscale_dot)
        real(dp), intent(in) :: log_variance, log_lengthscale, variance_dot, lengthscale_dot
        real(dp), intent(in) :: squared_distance
        real(dp), intent(out) :: value, value_dot, derivative_variance, derivative_variance_dot
        real(dp), intent(out) :: derivative_lengthscale, derivative_lengthscale_dot
        real(dp) :: variance, lengthscale, distance, argument, sine_value, cosine_value
        real(dp) :: length_derivative_direction

        variance = exp(log_variance)
        lengthscale = exp(log_lengthscale)
        distance = sqrt(max(squared_distance, 0.0_dp))
        argument = distance/lengthscale
        sine_value = sin(argument)
        cosine_value = cos(argument)
        value = variance*cosine_value
        derivative_variance = value
        derivative_lengthscale = variance*argument*sine_value
        value_dot = value*variance_dot + variance*argument*sine_value*lengthscale_dot
        derivative_variance_dot = value_dot
        length_derivative_direction = variance_dot*variance*argument*sine_value + &
            lengthscale_dot*variance*(-argument*sine_value - argument*argument*cosine_value)
        derivative_lengthscale_dot = length_derivative_direction
    end subroutine cosine_value_hvp

    subroutine cosine_input_derivatives(self, x1, x2, value, gradient_x1, gradient_x2, &
            mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: variance, lengthscale, squared_distance, distance
        real(dp) :: radial_first, radial_second, radial_scale, radial_coefficient
        real(dp) :: difference(size(x1))
        integer :: i, j

        variance = exp(self%log_parameters(1))
        lengthscale = exp(self%log_parameters(2))
        difference = x1 - x2
        squared_distance = sum(difference*difference)
        distance = sqrt(squared_distance)
        value = variance*cos(distance/lengthscale)
        radial_first = -variance*sin(distance/lengthscale)/lengthscale
        radial_second = -value/(lengthscale*lengthscale)
        if (distance == 0.0_dp) then
            gradient_x1 = 0.0_dp
            gradient_x2 = 0.0_dp
            mixed_hessian = 0.0_dp
            do i = 1, size(x1)
                mixed_hessian(i, i) = -radial_second
            end do
        else
            radial_scale = radial_first/distance
            radial_coefficient = (radial_second - radial_scale)/squared_distance
            do i = 1, size(x1)
                gradient_x1(i) = radial_scale*difference(i)
                gradient_x2(i) = -gradient_x1(i)
                do j = 1, size(x2)
                    mixed_hessian(i, j) = -(radial_scale*merge(1.0_dp, 0.0_dp, i == j) + &
                        radial_coefficient*difference(i)*difference(j))
                end do
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine cosine_input_derivatives

    subroutine polynomial_value_derivatives(variance, scale, offset, degree, inner_product, &
            value, derivative_variance, derivative_scale, derivative_offset, derivative_degree)
        real(dp), intent(in) :: variance, scale, offset, degree, inner_product
        real(dp), intent(out) :: value, derivative_variance, derivative_scale
        real(dp), intent(out) :: derivative_offset, derivative_degree
        real(dp) :: base, log_base

        base = offset + scale*inner_product
        if (base <= 0.0_dp) then
            value = 0.0_dp
            derivative_variance = 0.0_dp
            derivative_scale = 0.0_dp
            derivative_offset = 0.0_dp
            derivative_degree = 0.0_dp
            return
        end if
        log_base = log(base)
        value = variance*exp(degree*log_base)
        derivative_variance = value
        derivative_scale = value*degree*scale*inner_product/base
        derivative_offset = value*degree*offset/base
        derivative_degree = value*degree*log_base
    end subroutine polynomial_value_derivatives

    subroutine polynomial_value_jvp(variance, scale, offset, degree, inner_product, &
            variance_dot, scale_dot, offset_dot, degree_dot, value, value_dot)
        real(dp), intent(in) :: variance, scale, offset, degree, inner_product
        real(dp), intent(in) :: variance_dot, scale_dot, offset_dot, degree_dot
        real(dp), intent(out) :: value, value_dot
        real(dp) :: derivative_variance, derivative_scale, derivative_offset
        real(dp) :: derivative_degree

        call polynomial_value_derivatives(variance, scale, offset, degree, inner_product, &
            value, derivative_variance, derivative_scale, derivative_offset, derivative_degree)
        value_dot = derivative_variance*variance_dot + derivative_scale*scale_dot + &
            derivative_offset*offset_dot + derivative_degree*degree_dot
    end subroutine polynomial_value_jvp

    subroutine polynomial_value_hvp(log_variance, log_scale, log_offset, log_degree, &
            variance_dot, scale_dot, offset_dot, degree_log_dot, inner_product, value, &
            value_dot, derivative_variance, derivative_variance_dot, derivative_scale, &
            derivative_scale_dot, derivative_offset, derivative_offset_dot, derivative_degree, &
            derivative_degree_dot)
        real(dp), intent(in) :: log_variance, log_scale, log_offset, log_degree
        real(dp), intent(in) :: variance_dot, scale_dot, offset_dot, degree_log_dot
        real(dp), intent(in) :: inner_product
        real(dp), intent(out) :: value, value_dot, derivative_variance, derivative_variance_dot
        real(dp), intent(out) :: derivative_scale, derivative_scale_dot
        real(dp), intent(out) :: derivative_offset, derivative_offset_dot
        real(dp), intent(out) :: derivative_degree, derivative_degree_dot
        real(dp) :: variance, scale, offset, degree, base, log_base, base_dot, degree_dot
        real(dp) :: h_scale, h_offset, h_degree, h_scale_dot, h_offset_dot, h_degree_dot
        real(dp) :: log_direction

        variance = exp(log_variance)
        scale = exp(log_scale)
        offset = exp(log_offset)
        degree = exp(log_degree)
        base = offset + scale*inner_product
        if (base <= 0.0_dp) then
            value = 0.0_dp
            value_dot = 0.0_dp
            derivative_variance = 0.0_dp
            derivative_variance_dot = 0.0_dp
            derivative_scale = 0.0_dp
            derivative_scale_dot = 0.0_dp
            derivative_offset = 0.0_dp
            derivative_offset_dot = 0.0_dp
            derivative_degree = 0.0_dp
            derivative_degree_dot = 0.0_dp
            return
        end if
        log_base = log(base)
        degree_dot = degree*degree_log_dot
        base_dot = scale*inner_product*scale_dot + offset*offset_dot
        value = variance*exp(degree*log_base)
        log_direction = variance_dot + degree*base_dot/base + degree_log_dot*degree*log_base
        value_dot = value*log_direction
        derivative_variance = value
        derivative_variance_dot = value_dot
        h_scale = degree*scale*inner_product/base
        h_offset = degree*offset/base
        h_degree = degree*log_base
        h_scale_dot = degree_dot*scale*inner_product/base + degree*scale*inner_product* &
            scale_dot/base - degree*scale*inner_product*base_dot/(base*base)
        h_offset_dot = degree_dot*offset/base + degree*offset*offset_dot/base - &
            degree*offset*base_dot/(base*base)
        h_degree_dot = degree_dot*log_base + degree*base_dot/base
        derivative_scale = value*h_scale
        derivative_scale_dot = value_dot*h_scale + value*h_scale_dot
        derivative_offset = value*h_offset
        derivative_offset_dot = value_dot*h_offset + value*h_offset_dot
        derivative_degree = value*h_degree
        derivative_degree_dot = value_dot*h_degree + value*h_degree_dot
    end subroutine polynomial_value_hvp

    subroutine polynomial_input_derivatives(self, x1, x2, value, gradient_x1, gradient_x2, &
            mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: variance, scale, offset, degree, base, coefficient, curvature
        real(dp) :: inner_product
        integer :: i, j

        variance = exp(self%log_parameters(1))
        scale = exp(self%log_parameters(2))
        offset = exp(self%log_parameters(3))
        degree = exp(self%log_parameters(4))
        inner_product = dot_product(x1, x2)
        base = offset + scale*inner_product
        if (base <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "polynomial kernel input derivatives: base must be positive")
            return
        end if
        value = variance*base**degree
        coefficient = variance*degree*scale*base**(degree - 1.0_dp)
        curvature = variance*degree*(degree - 1.0_dp)*scale*scale*base**(degree - 2.0_dp)
        gradient_x1 = coefficient*x2
        gradient_x2 = coefficient*x1
        do i = 1, size(x1)
            do j = 1, size(x2)
                mixed_hessian(i, j) = coefficient*merge(1.0_dp, 0.0_dp, i == j) + &
                    curvature*x2(i)*x1(j)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine polynomial_input_derivatives

    subroutine rational_quadratic_value_derivatives(variance, lengthscale, alpha, &
            squared_distance, value, derivative_variance, derivative_lengthscale, derivative_alpha)
        real(dp), intent(in) :: variance, lengthscale, alpha, squared_distance
        real(dp), intent(out) :: value, derivative_variance, derivative_lengthscale, derivative_alpha
        real(dp) :: inverse_length_squared, t, denominator, log_denominator

        inverse_length_squared = 1.0_dp/(lengthscale*lengthscale)
        t = 0.5_dp*squared_distance*inverse_length_squared/alpha
        denominator = 1.0_dp + t
        log_denominator = log(denominator)
        value = variance*exp(-alpha*log_denominator)
        derivative_variance = value
        derivative_lengthscale = value*(2.0_dp*alpha*t/denominator)
        derivative_alpha = value*alpha*(t/denominator - log_denominator)
    end subroutine rational_quadratic_value_derivatives

    subroutine rational_quadratic_value_jvp(variance, lengthscale, alpha, squared_distance, &
            variance_dot, lengthscale_dot, alpha_dot, value, value_dot)
        real(dp), intent(in) :: variance, lengthscale, alpha, squared_distance
        real(dp), intent(in) :: variance_dot, lengthscale_dot, alpha_dot
        real(dp), intent(out) :: value, value_dot
        real(dp) :: derivative_variance, derivative_lengthscale, derivative_alpha

        call rational_quadratic_value_derivatives(variance, lengthscale, alpha, squared_distance, &
            value, derivative_variance, derivative_lengthscale, derivative_alpha)
        value_dot = derivative_variance*variance_dot + derivative_lengthscale*lengthscale_dot + &
            derivative_alpha*alpha_dot
    end subroutine rational_quadratic_value_jvp

    subroutine rational_quadratic_value_hvp(log_variance, log_lengthscale, log_alpha, &
            variance_dot, lengthscale_dot, alpha_dot, squared_distance, value, value_dot, &
            derivative_variance, derivative_variance_dot, derivative_lengthscale, &
            derivative_lengthscale_dot, derivative_alpha, derivative_alpha_dot)
        real(dp), intent(in) :: log_variance, log_lengthscale, log_alpha
        real(dp), intent(in) :: variance_dot, lengthscale_dot, alpha_dot, squared_distance
        real(dp), intent(out) :: value, value_dot, derivative_variance, derivative_variance_dot
        real(dp), intent(out) :: derivative_lengthscale, derivative_lengthscale_dot
        real(dp), intent(out) :: derivative_alpha, derivative_alpha_dot
        real(dp) :: variance, lengthscale, alpha, inverse_length_squared
        real(dp) :: t, denominator, log_denominator, alpha_actual_dot, t_dot
        real(dp) :: h, h_dot, g_lengthscale, g_alpha, g_lengthscale_dot, g_alpha_dot
        real(dp) :: log_direction

        variance = exp(log_variance)
        lengthscale = exp(log_lengthscale)
        alpha = exp(log_alpha)
        inverse_length_squared = exp(-2.0_dp*log_lengthscale)
        t = 0.5_dp*squared_distance*inverse_length_squared/alpha
        denominator = 1.0_dp + t
        log_denominator = log(denominator)
        alpha_actual_dot = alpha*alpha_dot
        t_dot = t*(-2.0_dp*lengthscale_dot - alpha_dot)
        h = t/denominator
        h_dot = t_dot/(denominator*denominator)
        value = variance*exp(-alpha*log_denominator)
        g_lengthscale = 2.0_dp*alpha*h
        g_alpha = alpha*(h - log_denominator)
        g_lengthscale_dot = 2.0_dp*(alpha_actual_dot*h + alpha*h_dot)
        g_alpha_dot = alpha_actual_dot*(h - log_denominator) + &
            alpha*(h_dot - t_dot/denominator)
        log_direction = variance_dot + g_lengthscale*lengthscale_dot + g_alpha*alpha_dot
        value_dot = value*log_direction
        derivative_variance = value
        derivative_variance_dot = value_dot
        derivative_lengthscale = value*g_lengthscale
        derivative_lengthscale_dot = value*(log_direction*g_lengthscale + g_lengthscale_dot)
        derivative_alpha = value*g_alpha
        derivative_alpha_dot = value*(log_direction*g_alpha + g_alpha_dot)
    end subroutine rational_quadratic_value_hvp

    subroutine rational_quadratic_input_derivatives(self, x1, x2, value, gradient_x1, &
            gradient_x2, mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:), mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: variance, lengthscale, alpha, squared_distance, inverse_length_squared
        real(dp) :: denominator, radial_first, radial_second_r2, difference(size(x1))
        integer :: i, j

        variance = exp(self%log_parameters(1))
        lengthscale = exp(self%log_parameters(2))
        alpha = exp(self%log_parameters(3))
        inverse_length_squared = 1.0_dp/(lengthscale*lengthscale)
        difference = x1 - x2
        squared_distance = sum(difference**2)
        call rational_quadratic_value_derivatives(variance, lengthscale, alpha, &
            squared_distance, value, radial_first, radial_second_r2, denominator)
        denominator = 1.0_dp + 0.5_dp*squared_distance*inverse_length_squared/alpha
        radial_first = -value/(2.0_dp*lengthscale*lengthscale*denominator)
        radial_second_r2 = value*(1.0_dp + 1.0_dp/alpha)/(4.0_dp*lengthscale**4*denominator**2)
        gradient_x1 = 2.0_dp*radial_first*difference
        gradient_x2 = -gradient_x1
        do i = 1, size(x1)
            do j = 1, size(x2)
                mixed_hessian(i, j) = -2.0_dp*radial_first*merge(1.0_dp, 0.0_dp, i == j) - &
                    4.0_dp*radial_second_r2*difference(i)*difference(j)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rational_quadratic_input_derivatives

    subroutine make_leaf(kernel, kind, input_dim, variance, lengthscale, status)
        type(kernel_t), intent(out) :: kernel
        integer, intent(in) :: kind, input_dim
        real(dp), intent(in) :: variance, lengthscale
        type(fortnum_status_t), intent(out) :: status

        if (input_dim < 1 .or. variance <= 0.0_dp .or. lengthscale <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel constructor: dimensions and scales must be positive")
            return
        end if
        kernel%kind = kind
        kernel%input_dim = input_dim
        if (kind == KERNEL_RBF .or. kind == KERNEL_MATERN12 .or. &
            kind == KERNEL_MATERN32 .or. kind == KERNEL_MATERN52 .or. &
            kind == KERNEL_COSINE) then
            allocate(kernel%log_parameters(2))
            kernel%log_parameters = [log(variance), log(lengthscale)]
        else
            allocate(kernel%log_parameters(1))
            kernel%log_parameters(1) = log(variance)
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine make_leaf

    subroutine make_three_parameter_leaf( &
            kernel, kind, input_dim, variance, lengthscale, third_parameter, status)
        type(kernel_t), intent(out) :: kernel
        integer, intent(in) :: kind, input_dim
        real(dp), intent(in) :: variance, lengthscale, third_parameter
        type(fortnum_status_t), intent(out) :: status

        if (input_dim < 1 .or. variance <= 0.0_dp .or. lengthscale <= 0.0_dp .or. &
            third_parameter <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel constructor: dimensions and scales must be positive")
            return
        end if
        kernel%kind = kind
        kernel%input_dim = input_dim
        allocate(kernel%log_parameters(3))
        kernel%log_parameters = [log(variance), log(lengthscale), log(third_parameter)]
        call status_set(status, FORTNUM_OK, "")
    end subroutine make_three_parameter_leaf

    subroutine make_composite(kernel, kind, left, right, status)
        type(kernel_t), intent(out) :: kernel
        integer, intent(in) :: kind
        type(kernel_t), intent(in) :: left, right
        type(fortnum_status_t), intent(out) :: status

        if (.not. kernel_valid(left) .or. .not. kernel_valid(right)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel composition: both operands must be initialized")
            return
        end if
        if (left%input_dim /= right%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel composition: input dimensions must agree")
            return
        end if
        kernel%kind = kind
        kernel%input_dim = left%input_dim
        allocate(kernel%left, source=left)
        allocate(kernel%right, source=right)
        call status_set(status, FORTNUM_OK, "")
    end subroutine make_composite

    subroutine check_matrix_shapes(self, x1, x2, matrix, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. kernel_valid(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel matrix: kernel is not initialized")
            return
        end if
        if (size(x1, 2) /= self%input_dim .or. size(x2, 2) /= self%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel matrix: input dimension is invalid")
            return
        end if
        if (any(shape(matrix) /= [size(x1, 1), size(x2, 1)])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel matrix: output shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine check_matrix_shapes

    !! `squared(i, j) = |x1(i, :) - x2(j, :)|^2`, for every pair at once.
    !!
    !! Built from `|a|^2 + |b|^2 - 2 a.b` so the pairwise term is a single
    !! matrix product. The expansion is famously less accurate than
    !! differencing when the two points nearly coincide -- the leading terms
    !! cancel -- so the result is floored at zero, which is where that
    !! cancellation shows up and is also exactly what a squared distance must
    !! satisfy. Kernels here are smooth in `r2` at the origin, so a few ulp of
    !! error near coincidence changes nothing that matters; the training Gram
    !! matrix still gets its diagonal from the same expression it always did.
    pure subroutine pairwise_squared_distances(x1, x2, squared)
        real(dp), intent(in) :: x1(:, :), x2(:, :)
        real(dp), allocatable, intent(out) :: squared(:, :)
        real(dp), allocatable :: norm1(:), norm2(:)
        integer :: i, j

        allocate (squared(size(x1, 1), size(x2, 1)))
        allocate (norm1(size(x1, 1)), norm2(size(x2, 1)))
        do i = 1, size(x1, 1)
            norm1(i) = dot_product(x1(i, :), x1(i, :))
        end do
        do j = 1, size(x2, 1)
            norm2(j) = dot_product(x2(j, :), x2(j, :))
        end do

        squared = -2.0_dp*matmul(x1, transpose(x2))
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                squared(i, j) = max(squared(i, j) + norm1(i) + norm2(j), 0.0_dp)
            end do
        end do
    end subroutine pairwise_squared_distances

    recursive subroutine kernel_matrix_impl(self, x1, x2, matrix)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :)
        real(dp), intent(out) :: matrix(:, :)
        real(dp) :: variance, lengthscale, third_parameter, r2, value, dummy
        real(dp) :: parameter_derivative_1, parameter_derivative_2
        real(dp), allocatable :: other(:, :), squared(:, :)
        integer :: i, j

        select case (self%kind)
        case (KERNEL_SUM, KERNEL_PRODUCT)
            call kernel_matrix_impl(self%left, x1, x2, matrix)
            allocate(other, mold=matrix)
            call kernel_matrix_impl(self%right, x1, x2, other)
            if (self%kind == KERNEL_SUM) then
                matrix = matrix + other
            else
                matrix = matrix*other
            end if
        case (KERNEL_RBF_ARD)
            call ard_rbf_matrix(self, x1, x2, matrix)
        case (KERNEL_SPECTRAL_MIXTURE)
            call spectral_matrix(self, x1, x2, matrix)
        case (KERNEL_LOCAL_PERIODIC)
            call local_periodic_matrix(self, x1, x2, matrix)
        case (KERNEL_CHANGE_POINT)
            call change_point_matrix(self, x1, x2, matrix)
        case default
            variance = exp(self%log_parameters(1))
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52 .or. &
                self%kind == KERNEL_PERIODIC .or. self%kind == KERNEL_COSINE .or. &
                self%kind == KERNEL_RATIONAL_QUADRATIC) then
                lengthscale = exp(self%log_parameters(2))
            else
                lengthscale = 1.0_dp
            end if
            third_parameter = 1.0_dp
            if (self%kind == KERNEL_PERIODIC .or. &
                self%kind == KERNEL_RATIONAL_QUADRATIC) then
                third_parameter = exp(self%log_parameters(3))
            end if
            ! Squared distances for every pair at once, through the
            ! expansion |a - b|^2 = |a|^2 + |b|^2 - 2 a.b, so the pairwise
            ! work becomes one matrix product.
            !
            ! The loop this replaces evaluated `sum((x1(i, :) - x2(j, :))**2)`
            ! per entry. Two costs, both invisible in the source: `x1(i, :)`
            ! walks a row of an `(n, d)` array, so consecutive elements sit `n`
            ! apart and every access misses cache; and the expression builds a
            ! temporary array per entry. At Bayesian-optimization sizes --
            ! four thousand candidates against forty training points -- that
            ! measured 82 ns per kernel evaluation, which is an order of
            ! magnitude off what an eight-dimensional squared exponential
            ! costs.
            !
            ! `matmul` is left to the compiler's BLAS rather than hand-blocked;
            ! the point is to stop doing the work entry by entry.
            call pairwise_squared_distances(x1, x2, squared)

            ! Whole-array fast paths for the stationary kernels, which are the
            ! ones every Bayesian-optimization run actually uses. The general
            ! loop below dispatches through a chain of `if`s and a procedure
            ! call *per entry*; at forty training points against four thousand
            ! candidates that is a hundred and sixty thousand calls and
            ! measured 22 ns each, which dominated the posterior. Expressed as
            ! array operations the compiler vectorizes them and the branch is
            ! taken once for the whole matrix instead of once per element.
            !
            ! The formulas are the same ones the scalar path evaluates, kept
            ! literally identical so the two cannot drift: `test_kernels`
            ! checks the matrix against the per-entry `value` for every kind.
            if (self%kind == KERNEL_RBF) then
                matrix = variance*exp(-0.5_dp*squared/(lengthscale*lengthscale))
                return
            else if (self%kind == KERNEL_MATERN12) then
                matrix = variance*exp(-sqrt(squared)/lengthscale)
                return
            else if (self%kind == KERNEL_MATERN32) then
                block
                    real(dp), allocatable :: scaled(:, :)
                    allocate (scaled(size(x1, 1), size(x2, 1)))
                    scaled = sqrt(3.0_dp)*sqrt(squared)/lengthscale
                    matrix = variance*(1.0_dp + scaled)*exp(-scaled)
                end block
                return
            else if (self%kind == KERNEL_MATERN52) then
                block
                    real(dp), allocatable :: scaled(:, :)
                    allocate (scaled(size(x1, 1), size(x2, 1)))
                    scaled = sqrt(5.0_dp)*sqrt(squared)/lengthscale
                    matrix = variance*(1.0_dp + scaled + scaled*scaled/3.0_dp) &
                        *exp(-scaled)
                end block
                return
            end if

            do j = 1, size(x2, 1)
                do i = 1, size(x1, 1)
                    r2 = squared(i, j)
                    value = 0.0_dp
                    if (self%kind == KERNEL_RBF) then
                        call fortml_generated_rbf_leaf_fortran( &
                            variance, r2, lengthscale, value)
                    else if (self%kind == KERNEL_PERIODIC) then
                        call periodic_value_derivatives(variance, lengthscale, &
                            third_parameter, r2, value, parameter_derivative_1, &
                            parameter_derivative_2, dummy)
                    else if (self%kind == KERNEL_RATIONAL_QUADRATIC) then
                        call rational_quadratic_value_derivatives(variance, lengthscale, &
                            third_parameter, r2, value, parameter_derivative_1, &
                            parameter_derivative_2, dummy)
                    else if (self%kind == KERNEL_COSINE) then
                        call cosine_value_derivatives(variance, lengthscale, r2, value, &
                            parameter_derivative_1, parameter_derivative_2)
                    else if (self%kind == KERNEL_POLYNOMIAL) then
                        call polynomial_value_derivatives(variance, exp(self%log_parameters(2)), &
                            exp(self%log_parameters(3)), exp(self%log_parameters(4)), &
                            dot_product(x1(i, :), x2(j, :)), value, parameter_derivative_1, &
                            parameter_derivative_2, dummy, third_parameter)
                    else if (self%kind /= KERNEL_USER) then
                        call leaf_value_and_length_derivative(self%kind, variance, &
                            lengthscale, r2, value, dummy)
                    end if
                    if (self%kind == KERNEL_USER) then
                        matrix(i, j) = variance*self%formula%evaluate( &
                            r2, dot_product(x1(i, :), x2(j, :)))
                    else if (self%kind == KERNEL_LINEAR) then
                        matrix(i, j) = variance*dot_product(x1(i, :), x2(j, :))
                    else if (self%kind == KERNEL_CONSTANT) then
                        matrix(i, j) = variance
                    else if (self%kind == KERNEL_WHITE_NOISE) then
                        matrix(i, j) = variance*merge(1.0_dp, 0.0_dp, &
                            same_row(x1, i, x2, j))
                    else
                        matrix(i, j) = value
                    end if
                end do
            end do
        end select
    end subroutine kernel_matrix_impl

    subroutine change_point_matrix(self, x1, x2, matrix)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :)
        real(dp), intent(out) :: matrix(:, :)
        real(dp) :: gate_left, gate_right
        integer :: i, j

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                call change_point_gate(self, x1(i, :), x2(j, :), gate_left, gate_right)
                matrix(i, j) = gate_left*self%left%value(x1(i, :), x2(j, :)) + &
                    gate_right*self%right%value(x1(i, :), x2(j, :))
            end do
        end do
    end subroutine change_point_matrix

    recursive subroutine kernel_input_derivatives_impl( &
            self, x1, x2, value, gradient_x1, gradient_x2, mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: left_gradient_x1(:), right_gradient_x1(:)
        real(dp), allocatable :: left_gradient_x2(:), right_gradient_x2(:)
        real(dp), allocatable :: left_hessian(:, :), right_hessian(:, :)
        real(dp) :: left_value, right_value
        real(dp) :: variance, lengthscale, inverse_length_squared
        real(dp) :: squared_distance, difference
        integer :: i, j

        !! Composite callers allocate the child output arrays before entering
        !! this routine.  Initialize every output here as well as in the
        !! public wrapper so constant and refused leaves cannot leak
        !! uninitialized values into a product rule.
        value = 0.0_dp
        gradient_x1 = 0.0_dp
        gradient_x2 = 0.0_dp
        mixed_hessian = 0.0_dp

        select case (self%kind)
        case (KERNEL_SUM, KERNEL_PRODUCT)
            allocate(left_gradient_x1(size(x1)), right_gradient_x1(size(x1)))
            allocate(left_gradient_x2(size(x2)), right_gradient_x2(size(x2)))
            allocate(left_hessian(size(x1), size(x2)))
            allocate(right_hessian(size(x1), size(x2)))
            call kernel_input_derivatives_impl( &
                self%left, x1, x2, left_value, left_gradient_x1, &
                left_gradient_x2, left_hessian, status)
            if (status%code /= FORTNUM_OK) return
            call kernel_input_derivatives_impl( &
                self%right, x1, x2, right_value, right_gradient_x1, &
                right_gradient_x2, right_hessian, status)
            if (status%code /= FORTNUM_OK) return
            if (self%kind == KERNEL_SUM) then
                value = left_value + right_value
                gradient_x1 = left_gradient_x1 + right_gradient_x1
                gradient_x2 = left_gradient_x2 + right_gradient_x2
                mixed_hessian = left_hessian + right_hessian
            else
                value = left_value*right_value
                gradient_x1 = left_gradient_x1*right_value + &
                    right_gradient_x1*left_value
                gradient_x2 = left_gradient_x2*right_value + &
                    right_gradient_x2*left_value
                mixed_hessian = left_hessian*right_value + &
                    right_hessian*left_value + &
                    spread(left_gradient_x1, dim=2, ncopies=size(x2))* &
                    spread(right_gradient_x2, dim=1, ncopies=size(x1)) + &
                    spread(right_gradient_x1, dim=2, ncopies=size(x2))* &
                    spread(left_gradient_x2, dim=1, ncopies=size(x1))
            end if
            call status_set(status, FORTNUM_OK, "")
        case (KERNEL_RBF_ARD)
            call ard_rbf_input_derivatives(self, x1, x2, value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
        case (KERNEL_SPECTRAL_MIXTURE)
            call spectral_input_derivatives(self, x1, x2, value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
        case (KERNEL_LOCAL_PERIODIC)
            call local_periodic_input_derivatives(self, x1, x2, value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
        case (KERNEL_CHANGE_POINT)
            call change_point_input_derivatives(self, x1, x2, value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
        case (KERNEL_RBF)
            variance = exp(self%log_parameters(1))
            lengthscale = exp(self%log_parameters(2))
            inverse_length_squared = 1.0_dp/(lengthscale*lengthscale)
            squared_distance = sum((x1 - x2)**2)
            call fortml_generated_rbf_leaf_fortran( &
                variance, squared_distance, lengthscale, value)
            do i = 1, self%input_dim
                difference = x1(i) - x2(i)
                gradient_x1(i) = -value*difference*inverse_length_squared
                gradient_x2(i) = -gradient_x1(i)
                do j = 1, self%input_dim
                    mixed_hessian(i, j) = value*( &
                        merge(inverse_length_squared, 0.0_dp, i == j) - &
                        (x1(i) - x2(i))*(x1(j) - x2(j))* &
                        inverse_length_squared*inverse_length_squared)
                end do
            end do
            call status_set(status, FORTNUM_OK, "")
        case (KERNEL_PERIODIC)
            call periodic_input_derivatives(self, x1, x2, value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
        case (KERNEL_RATIONAL_QUADRATIC)
            call rational_quadratic_input_derivatives(self, x1, x2, value, &
                gradient_x1, gradient_x2, mixed_hessian, status)
        case (KERNEL_COSINE)
            call cosine_input_derivatives(self, x1, x2, value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
        case (KERNEL_POLYNOMIAL)
            call polynomial_input_derivatives(self, x1, x2, value, gradient_x1, &
                gradient_x2, mixed_hessian, status)
        case (KERNEL_LINEAR)
            variance = exp(self%log_parameters(1))
            value = variance*dot_product(x1, x2)
            gradient_x1 = variance*x2
            gradient_x2 = variance*x1
            mixed_hessian = 0.0_dp
            do i = 1, self%input_dim
                mixed_hessian(i, i) = variance
            end do
            call status_set(status, FORTNUM_OK, "")
        case (KERNEL_CONSTANT)
            variance = exp(self%log_parameters(1))
            value = variance
            call status_set(status, FORTNUM_OK, "")
        case (KERNEL_WHITE_NOISE)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel input_derivatives: white-noise kernel is nonsmooth")
        case (KERNEL_MATERN12, KERNEL_MATERN32, KERNEL_MATERN52)
            call matern_input_derivatives( &
                self, x1, x2, value, gradient_x1, gradient_x2, mixed_hessian, status)
        case (KERNEL_USER)
            call user_input_derivatives( &
                self, x1, x2, value, gradient_x1, gradient_x2, mixed_hessian, status)
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel input_derivatives: this kernel has no smooth input rule")
        end select
    end subroutine kernel_input_derivatives_impl

    subroutine change_point_input_derivatives(self, x1, x2, value, gradient_x1, &
            gradient_x2, mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: left_gradient_x1(:), right_gradient_x1(:)
        real(dp), allocatable :: left_gradient_x2(:), right_gradient_x2(:)
        real(dp), allocatable :: left_hessian(:, :), right_hessian(:, :)
        real(dp) :: left_value, right_value, gate_left, gate_right
        real(dp) :: gate_left_x1(size(x1)), gate_left_x2(size(x2))
        real(dp) :: gate_right_x1(size(x1)), gate_right_x2(size(x2))
        real(dp) :: gate_left_hessian(size(x1), size(x2))
        real(dp) :: gate_right_hessian(size(x1), size(x2))
        integer :: i, j

        allocate(left_gradient_x1(size(x1)), right_gradient_x1(size(x1)))
        allocate(left_gradient_x2(size(x2)), right_gradient_x2(size(x2)))
        allocate(left_hessian(size(x1), size(x2)), right_hessian(size(x1), size(x2)))
        call self%left%input_derivatives(x1, x2, left_value, left_gradient_x1, &
            left_gradient_x2, left_hessian, status)
        if (status%code /= FORTNUM_OK) return
        call self%right%input_derivatives(x1, x2, right_value, right_gradient_x1, &
            right_gradient_x2, right_hessian, status)
        if (status%code /= FORTNUM_OK) return
        call change_point_input_gates(self, x1, x2, gate_left, gate_right, &
            gate_left_x1, gate_left_x2, gate_right_x1, gate_right_x2, &
            gate_left_hessian, gate_right_hessian)
        value = gate_left*left_value + gate_right*right_value
        gradient_x1 = gate_left*left_gradient_x1 + gate_right*right_gradient_x1 + &
            gate_left_x1*left_value + gate_right_x1*right_value
        gradient_x2 = gate_left*left_gradient_x2 + gate_right*right_gradient_x2 + &
            gate_left_x2*left_value + gate_right_x2*right_value
        mixed_hessian = gate_left*left_hessian + gate_right*right_hessian
        do j = 1, size(x2)
            do i = 1, size(x1)
                mixed_hessian(i, j) = mixed_hessian(i, j) + &
                    gate_left_x1(i)*left_gradient_x2(j) + &
                    left_gradient_x1(i)*gate_left_x2(j) + &
                    gate_right_x1(i)*right_gradient_x2(j) + &
                    right_gradient_x1(i)*gate_right_x2(j) + &
                    gate_left_hessian(i, j)*left_value + &
                    gate_right_hessian(i, j)*right_value
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine change_point_input_derivatives

    subroutine change_point_input_gates(self, x1, x2, gate_left, gate_right, &
            gate_left_x1, gate_left_x2, gate_right_x1, gate_right_x2, &
            gate_left_hessian, gate_right_hessian)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: gate_left, gate_right
        real(dp), intent(out) :: gate_left_x1(:), gate_left_x2(:)
        real(dp), intent(out) :: gate_right_x1(:), gate_right_x2(:)
        real(dp), intent(out) :: gate_left_hessian(:, :), gate_right_hessian(:, :)
        real(dp) :: width, center, z1, z2, s1, s2, q1, q2
        real(dp) :: sx1, sx2
        integer :: feature

        width = exp(self%log_parameters(1))
        center = self%log_parameters(2)
        feature = self%change_feature
        z1 = (x1(feature) - center)/width
        z2 = (x2(feature) - center)/width
        s1 = 0.5_dp*(1.0_dp + tanh(z1))
        s2 = 0.5_dp*(1.0_dp + tanh(z2))
        q1 = 0.5_dp*(1.0_dp - tanh(z1)**2)
        q2 = 0.5_dp*(1.0_dp - tanh(z2)**2)
        sx1 = q1/width
        sx2 = q2/width
        gate_left = s1*s2
        gate_right = (1.0_dp - s1)*(1.0_dp - s2)
        gate_left_x1 = 0.0_dp
        gate_left_x2 = 0.0_dp
        gate_right_x1 = 0.0_dp
        gate_right_x2 = 0.0_dp
        gate_left_hessian = 0.0_dp
        gate_right_hessian = 0.0_dp
        gate_left_x1(feature) = sx1*s2
        gate_left_x2(feature) = s1*sx2
        gate_right_x1(feature) = -sx1*(1.0_dp - s2)
        gate_right_x2(feature) = -(1.0_dp - s1)*sx2
        gate_left_hessian(feature, feature) = sx1*sx2
        gate_right_hessian(feature, feature) = sx1*sx2
    end subroutine change_point_input_gates

    subroutine user_input_derivatives(self, x1, x2, value, gradient_x1, &
            gradient_x2, mixed_hessian, status)
        !! Forward-mode derivatives of a validated user formula.
        !!
        !! The formula is a postfix expression in squared distance, distance,
        !! and inner product.  Carrying the value, both input gradients, and
        !! the mixed Hessian through the same stack gives derivative-observation
        !! GPs the same contract as built-in kernels without procedure-pointer
        !! callbacks or a finite-difference fallback.
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: values(MAX_FORMULA_STACK)
        real(dp) :: gradients_x1(MAX_FORMULA_STACK, size(x1))
        real(dp) :: gradients_x2(MAX_FORMULA_STACK, size(x2))
        real(dp) :: hessians(MAX_FORMULA_STACK, size(x1), size(x2))
        real(dp) :: difference(size(x1)), squared_distance, distance, inner_product
        real(dp) :: left_value, right_value, left_gradient, right_gradient
        real(dp) :: left_hessian, right_hessian, factor
        integer :: i, j, top, opcode

        value = 0.0_dp
        gradient_x1 = 0.0_dp
        gradient_x2 = 0.0_dp
        mixed_hessian = 0.0_dp
        if (.not. allocated(self%formula) .or. .not. self%formula%static_lowering_eligible()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel user derivatives: formula is not validated")
            return
        end if

        difference = x1 - x2
        squared_distance = sum(difference**2)
        distance = sqrt(squared_distance)
        inner_product = dot_product(x1, x2)
        top = 0
        do i = 1, self%formula%length
            opcode = self%formula%opcode(i)
            select case (opcode)
            case (OPCODE_PUSH_R2)
                top = top + 1
                values(top) = squared_distance
                gradients_x1(top, :) = 2.0_dp*difference
                gradients_x2(top, :) = -2.0_dp*difference
                hessians(top, :, :) = 0.0_dp
                do j = 1, size(x1)
                    hessians(top, j, j) = -2.0_dp
                end do
            case (OPCODE_PUSH_R)
                if (distance <= 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel user derivatives: distance derivative is undefined at coincidence")
                    return
                end if
                top = top + 1
                values(top) = distance
                gradients_x1(top, :) = difference/distance
                gradients_x2(top, :) = -gradients_x1(top, :)
                hessians(top, :, :) = 0.0_dp
                do j = 1, size(x1)
                    hessians(top, j, j) = -1.0_dp/distance
                    hessians(top, j, :) = hessians(top, j, :) + &
                        difference(j)*difference/distance**3
                end do
            case (OPCODE_PUSH_DOT)
                top = top + 1
                values(top) = inner_product
                gradients_x1(top, :) = x2
                gradients_x2(top, :) = x1
                hessians(top, :, :) = 0.0_dp
                do j = 1, size(x1)
                    hessians(top, j, j) = 1.0_dp
                end do
            case (OPCODE_PUSH_CONST)
                top = top + 1
                values(top) = self%formula%operand(i)
                gradients_x1(top, :) = 0.0_dp
                gradients_x2(top, :) = 0.0_dp
                hessians(top, :, :) = 0.0_dp
            case (OPCODE_ADD, OPCODE_SUBTRACT, OPCODE_MULTIPLY)
                left_value = values(top - 1)
                right_value = values(top)
                if (opcode == OPCODE_ADD) then
                    values(top - 1) = left_value + right_value
                    gradients_x1(top - 1, :) = gradients_x1(top - 1, :) + gradients_x1(top, :)
                    gradients_x2(top - 1, :) = gradients_x2(top - 1, :) + gradients_x2(top, :)
                    hessians(top - 1, :, :) = hessians(top - 1, :, :) + hessians(top, :, :)
                else if (opcode == OPCODE_SUBTRACT) then
                    values(top - 1) = left_value - right_value
                    gradients_x1(top - 1, :) = gradients_x1(top - 1, :) - gradients_x1(top, :)
                    gradients_x2(top - 1, :) = gradients_x2(top - 1, :) - gradients_x2(top, :)
                    hessians(top - 1, :, :) = hessians(top - 1, :, :) - hessians(top, :, :)
                else
                    values(top - 1) = left_value*right_value
                    hessians(top - 1, :, :) = hessians(top - 1, :, :)*right_value + &
                        hessians(top, :, :)*left_value + &
                        spread(gradients_x1(top - 1, :), dim=2, ncopies=size(x2))* &
                        spread(gradients_x2(top, :), dim=1, ncopies=size(x1)) + &
                        spread(gradients_x1(top, :), dim=2, ncopies=size(x2))* &
                        spread(gradients_x2(top - 1, :), dim=1, ncopies=size(x1))
                    gradients_x1(top - 1, :) = gradients_x1(top - 1, :)*right_value + &
                        gradients_x1(top, :)*left_value
                    gradients_x2(top - 1, :) = gradients_x2(top - 1, :)*right_value + &
                        gradients_x2(top, :)*left_value
                end if
                top = top - 1
            case (OPCODE_NEGATE)
                values(top) = -values(top)
                gradients_x1(top, :) = -gradients_x1(top, :)
                gradients_x2(top, :) = -gradients_x2(top, :)
                hessians(top, :, :) = -hessians(top, :, :)
            case (OPCODE_EXP)
                factor = exp(values(top))
                hessians(top, :, :) = factor*(hessians(top, :, :) + &
                    spread(gradients_x1(top, :), dim=2, ncopies=size(x2))* &
                    spread(gradients_x2(top, :), dim=1, ncopies=size(x1)))
                gradients_x1(top, :) = factor*gradients_x1(top, :)
                gradients_x2(top, :) = factor*gradients_x2(top, :)
                values(top) = factor
            case (OPCODE_DIVIDE_CONST)
                factor = 1.0_dp/self%formula%operand(i)
                values(top) = values(top)*factor
                gradients_x1(top, :) = gradients_x1(top, :)*factor
                gradients_x2(top, :) = gradients_x2(top, :)*factor
                hessians(top, :, :) = hessians(top, :, :)*factor
            end select
        end do
        value = exp(self%log_parameters(1))*values(top)
        gradient_x1 = exp(self%log_parameters(1))*gradients_x1(top, :)
        gradient_x2 = exp(self%log_parameters(1))*gradients_x2(top, :)
        mixed_hessian = exp(self%log_parameters(1))*hessians(top, :, :)
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient_x1)) .or. &
            any(.not. ieee_is_finite(gradient_x2)) .or. any(.not. ieee_is_finite(mixed_hessian))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "kernel user derivatives: nonfinite result")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine user_input_derivatives

    subroutine matern_input_derivatives( &
            self, x1, x2, value, gradient_x1, gradient_x2, mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: variance, lengthscale, squared_distance, distance
        real(dp) :: radial_first, radial_second, radial_scale
        real(dp) :: value_dot, distance_bar_d
        real(dp) :: variance_bar, variance_bar_d, lengthscale_bar, lengthscale_bar_d
        integer :: i, j

        value = 0.0_dp
        gradient_x1 = 0.0_dp
        gradient_x2 = 0.0_dp
        mixed_hessian = 0.0_dp
        variance = exp(self%log_parameters(1))
        lengthscale = exp(self%log_parameters(2))
        squared_distance = sum((x1 - x2)**2)
        distance = sqrt(squared_distance)
        if (self%kind == KERNEL_MATERN12 .and. distance == 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Matern 1/2 input derivatives are undefined at coincident points")
            return
        end if

        select case (self%kind)
        case (KERNEL_MATERN12)
            call fortml_matern12_hvp( &
                distance, 1.0_dp, self%log_parameters(1), 0.0_dp, &
                self%log_parameters(2), 0.0_dp, value, value_dot, 1.0_dp, &
                radial_first, distance_bar_d, variance_bar, variance_bar_d, &
                lengthscale_bar, lengthscale_bar_d)
            radial_second = distance_bar_d
        case (KERNEL_MATERN32)
            call fortml_matern32_hvp( &
                distance, 1.0_dp, self%log_parameters(1), 0.0_dp, &
                self%log_parameters(2), 0.0_dp, value, value_dot, 1.0_dp, &
                radial_first, distance_bar_d, variance_bar, variance_bar_d, &
                lengthscale_bar, lengthscale_bar_d)
            radial_second = distance_bar_d
        case (KERNEL_MATERN52)
            call fortml_matern52_hvp( &
                distance, 1.0_dp, self%log_parameters(1), 0.0_dp, &
                self%log_parameters(2), 0.0_dp, value, value_dot, 1.0_dp, &
                radial_first, distance_bar_d, variance_bar, variance_bar_d, &
                lengthscale_bar, lengthscale_bar_d)
            radial_second = distance_bar_d
        end select

        if (distance == 0.0_dp) then
            gradient_x1 = 0.0_dp
            gradient_x2 = 0.0_dp
            mixed_hessian = 0.0_dp
            if (self%kind == KERNEL_MATERN32) then
                radial_scale = 3.0_dp*variance/(lengthscale*lengthscale)
            else
                radial_scale = 5.0_dp*variance/(3.0_dp*lengthscale*lengthscale)
            end if
            do i = 1, size(x1)
                mixed_hessian(i, i) = radial_scale
            end do
        else
            radial_scale = radial_first/distance
            do i = 1, size(x1)
                gradient_x1(i) = radial_scale*(x1(i) - x2(i))
                gradient_x2(i) = -gradient_x1(i)
                do j = 1, size(x2)
                    mixed_hessian(i, j) = -(radial_scale*merge(1.0_dp, 0.0_dp, i == j) + &
                        (radial_second - radial_scale)*(x1(i) - x2(i))* &
                        (x1(j) - x2(j))/squared_distance)
                end do
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine matern_input_derivatives

    recursive subroutine kernel_matrix_jvp_impl(self, x1, x2, direction, matrix, &
            matrix_dot)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), direction(:)
        real(dp), intent(out) :: matrix(:, :), matrix_dot(:, :)
        real(dp), allocatable :: other(:, :), other_dot(:, :)
        real(dp) :: variance, lengthscale, third_parameter, r2, value, length_derivative
        real(dp) :: log_variance_dot, log_lengthscale_dot, log_third_parameter_dot
        real(dp) :: dvariance, ddistance, dlengthscale
        real(dp) :: polynomial_scale, polynomial_offset, polynomial_degree
        integer :: i, j, left_count

        select case (self%kind)
        case (KERNEL_SUM, KERNEL_PRODUCT)
            left_count = self%left%parameter_count()
            allocate(other, mold=matrix)
            allocate(other_dot, mold=matrix_dot)
            call kernel_matrix_jvp_impl(self%left, x1, x2, direction(:left_count), &
                matrix, matrix_dot)
            call kernel_matrix_jvp_impl(self%right, x1, x2, direction(left_count + 1:), &
                other, other_dot)
            if (self%kind == KERNEL_SUM) then
                matrix = matrix + other
                matrix_dot = matrix_dot + other_dot
            else
                matrix_dot = matrix_dot*other + matrix*other_dot
                matrix = matrix*other
            end if
        case (KERNEL_RBF_ARD)
            call ard_rbf_matrix_jvp(self, x1, x2, direction, matrix, matrix_dot)
        case (KERNEL_SPECTRAL_MIXTURE)
            call spectral_matrix_jvp(self, x1, x2, direction, matrix, matrix_dot)
        case (KERNEL_LOCAL_PERIODIC)
            call local_periodic_matrix_jvp(self, x1, x2, direction, matrix, matrix_dot)
        case (KERNEL_CHANGE_POINT)
            call change_point_matrix_jvp(self, x1, x2, direction, matrix, matrix_dot)
        case default
            variance = exp(self%log_parameters(1))
            log_variance_dot = direction(1)
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52 .or. &
                self%kind == KERNEL_PERIODIC .or. self%kind == KERNEL_COSINE .or. &
                self%kind == KERNEL_RATIONAL_QUADRATIC) then
                lengthscale = exp(self%log_parameters(2))
                log_lengthscale_dot = direction(2)
            else
                lengthscale = 1.0_dp
                log_lengthscale_dot = 0.0_dp
            end if
            third_parameter = 1.0_dp
            log_third_parameter_dot = 0.0_dp
            if (self%kind == KERNEL_PERIODIC .or. &
                self%kind == KERNEL_RATIONAL_QUADRATIC) then
                third_parameter = exp(self%log_parameters(3))
                log_third_parameter_dot = direction(3)
            end if
            do j = 1, size(x2, 1)
                do i = 1, size(x1, 1)
                    r2 = sum((x1(i, :) - x2(j, :))**2)
                    if (self%kind == KERNEL_RBF) then
                        call fortml_generated_rbf_leaf_derivatives( &
                            variance, r2, lengthscale, value, dvariance, &
                            ddistance, dlengthscale)
                        matrix_dot(i, j) = dvariance*variance* &
                            log_variance_dot + dlengthscale*lengthscale* &
                            log_lengthscale_dot
                    else if (self%kind == KERNEL_PERIODIC) then
                        call periodic_value_jvp(variance, lengthscale, third_parameter, &
                            r2, log_variance_dot, log_lengthscale_dot, &
                            log_third_parameter_dot, value, matrix_dot(i, j))
                    else if (self%kind == KERNEL_RATIONAL_QUADRATIC) then
                        call rational_quadratic_value_jvp(variance, lengthscale, &
                            third_parameter, r2, log_variance_dot, &
                            log_lengthscale_dot, log_third_parameter_dot, value, &
                            matrix_dot(i, j))
                    else if (self%kind == KERNEL_COSINE) then
                        call cosine_value_jvp(variance, lengthscale, r2, log_variance_dot, &
                            log_lengthscale_dot, value, matrix_dot(i, j))
                    else if (self%kind == KERNEL_POLYNOMIAL) then
                        polynomial_scale = exp(self%log_parameters(2))
                        polynomial_offset = exp(self%log_parameters(3))
                        polynomial_degree = exp(self%log_parameters(4))
                        call polynomial_value_jvp(variance, polynomial_scale, polynomial_offset, &
                            polynomial_degree, dot_product(x1(i, :), x2(j, :)), &
                            log_variance_dot, direction(2), direction(3), direction(4), &
                            value, matrix_dot(i, j))
                    else
                        call leaf_value_and_length_derivative(self%kind, variance, &
                            lengthscale, r2, value, length_derivative)
                        if (self%kind == KERNEL_LINEAR) then
                            value = variance*dot_product(x1(i, :), x2(j, :))
                            matrix_dot(i, j) = value*log_variance_dot
                        else if (self%kind == KERNEL_CONSTANT) then
                            value = variance
                            matrix_dot(i, j) = value*log_variance_dot
                        else if (self%kind == KERNEL_WHITE_NOISE) then
                            value = variance*merge(1.0_dp, 0.0_dp, &
                                same_row(x1, i, x2, j))
                            matrix_dot(i, j) = value*log_variance_dot
                        else
                            matrix_dot(i, j) = value*log_variance_dot + &
                                length_derivative*log_lengthscale_dot
                        end if
                    end if
                    matrix(i, j) = value
                end do
            end do
        end select
    end subroutine kernel_matrix_jvp_impl

    subroutine change_point_matrix_jvp(self, x1, x2, direction, matrix, matrix_dot)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), direction(:)
        real(dp), intent(out) :: matrix(:, :), matrix_dot(:, :)
        real(dp), allocatable :: left_matrix(:, :), right_matrix(:, :)
        real(dp), allocatable :: left_matrix_dot(:, :), right_matrix_dot(:, :)
        real(dp) :: gate_left, gate_right, gate_left_dot, gate_right_dot
        real(dp) :: gate_direction(2)
        integer :: left_count, right_count, i, j

        left_count = self%left%parameter_count()
        right_count = self%right%parameter_count()
        allocate(left_matrix(size(x1, 1), size(x2, 1)))
        allocate(right_matrix, mold=left_matrix)
        allocate(left_matrix_dot, mold=left_matrix)
        allocate(right_matrix_dot, mold=left_matrix)
        call kernel_matrix_jvp_impl(self%left, x1, x2, direction(:left_count), &
            left_matrix, left_matrix_dot)
        call kernel_matrix_jvp_impl(self%right, x1, x2, &
            direction(left_count + 1:left_count + right_count), right_matrix, right_matrix_dot)
        gate_direction = direction(left_count + right_count + 1:left_count + right_count + 2)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                call change_point_gate(self, x1(i, :), x2(j, :), gate_left, gate_right, &
                    gate_left_dot, gate_right_dot, gate_direction)
                matrix(i, j) = gate_left*left_matrix(i, j) + gate_right*right_matrix(i, j)
                matrix_dot(i, j) = gate_left_dot*left_matrix(i, j) + &
                    gate_left*left_matrix_dot(i, j) + gate_right_dot*right_matrix(i, j) + &
                    gate_right*right_matrix_dot(i, j)
            end do
        end do
    end subroutine change_point_matrix_jvp

    recursive subroutine kernel_parameter_vjp_impl(self, x1, x2, matrix_bar, &
            parameter_bar, offset)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        integer, intent(in) :: offset
        real(dp), allocatable :: left_matrix(:, :), right_matrix(:, :)
        real(dp) :: variance, lengthscale, third_parameter, r2, value
        real(dp) :: polynomial_scale, polynomial_offset, polynomial_degree, inner_product
        real(dp) :: derivative_1, derivative_2, derivative_3, derivative_4
        integer :: i, j
        integer :: left_count

        select case (self%kind)
        case (KERNEL_SUM)
            left_count = self%left%parameter_count()
            call kernel_parameter_vjp_impl(self%left, x1, x2, matrix_bar, &
                parameter_bar, offset)
            call kernel_parameter_vjp_impl(self%right, x1, x2, matrix_bar, &
                parameter_bar, offset + left_count)
        case (KERNEL_PRODUCT)
            allocate(left_matrix(size(x1, 1), size(x2, 1)))
            allocate(right_matrix(size(x1, 1), size(x2, 1)))
            call kernel_matrix_impl(self%left, x1, x2, left_matrix)
            call kernel_matrix_impl(self%right, x1, x2, right_matrix)
            left_count = self%left%parameter_count()
            call kernel_parameter_vjp_impl(self%left, x1, x2, &
                matrix_bar*right_matrix, parameter_bar, offset)
            call kernel_parameter_vjp_impl(self%right, x1, x2, &
                matrix_bar*left_matrix, parameter_bar, offset + left_count)
        case (KERNEL_SPECTRAL_MIXTURE)
            call spectral_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, offset)
        case (KERNEL_LOCAL_PERIODIC)
            call local_periodic_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, offset)
        case (KERNEL_CHANGE_POINT)
            call change_point_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, offset)
        case default
            variance = exp(self%log_parameters(1))
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52 .or. &
                self%kind == KERNEL_PERIODIC .or. self%kind == KERNEL_COSINE .or. &
                self%kind == KERNEL_RATIONAL_QUADRATIC) then
                lengthscale = exp(self%log_parameters(2))
            else
                lengthscale = 1.0_dp
            end if
            if (self%kind == KERNEL_PERIODIC .or. &
                self%kind == KERNEL_RATIONAL_QUADRATIC) then
                third_parameter = exp(self%log_parameters(3))
            else
                third_parameter = 1.0_dp
            end if
            if (self%kind == KERNEL_RBF) then
                call kernel_rbf_parameter_vjp(self, x1, x2, matrix_bar, &
                    parameter_bar, offset)
            else if (self%kind == KERNEL_COSINE) then
                do j = 1, size(x2, 1)
                    do i = 1, size(x1, 1)
                        r2 = sum((x1(i, :) - x2(j, :))**2)
                        call cosine_value_derivatives(variance, lengthscale, r2, value, &
                            derivative_1, derivative_2)
                        parameter_bar(offset) = parameter_bar(offset) + &
                            matrix_bar(i, j)*derivative_1
                        parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                            matrix_bar(i, j)*derivative_2
                    end do
                end do
            else if (self%kind == KERNEL_PERIODIC .or. &
                self%kind == KERNEL_RATIONAL_QUADRATIC) then
                do j = 1, size(x2, 1)
                    do i = 1, size(x1, 1)
                        r2 = sum((x1(i, :) - x2(j, :))**2)
                        if (self%kind == KERNEL_PERIODIC) then
                            call periodic_value_derivatives(variance, lengthscale, &
                                third_parameter, r2, value, derivative_1, &
                                derivative_2, derivative_3)
                        else
                            call rational_quadratic_value_derivatives(variance, &
                                lengthscale, third_parameter, r2, value, derivative_1, &
                                derivative_2, derivative_3)
                        end if
                        parameter_bar(offset) = parameter_bar(offset) + &
                            matrix_bar(i, j)*derivative_1
                        parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                            matrix_bar(i, j)*derivative_2
                        parameter_bar(offset + 2) = parameter_bar(offset + 2) + &
                            matrix_bar(i, j)*derivative_3
                    end do
                end do
            else if (self%kind == KERNEL_POLYNOMIAL) then
                polynomial_scale = exp(self%log_parameters(2))
                polynomial_offset = exp(self%log_parameters(3))
                polynomial_degree = exp(self%log_parameters(4))
                do j = 1, size(x2, 1)
                    do i = 1, size(x1, 1)
                        inner_product = dot_product(x1(i, :), x2(j, :))
                        call polynomial_value_derivatives(variance, polynomial_scale, &
                            polynomial_offset, polynomial_degree, inner_product, value, &
                            derivative_1, derivative_2, derivative_3, derivative_4)
                        parameter_bar(offset) = parameter_bar(offset) + &
                            matrix_bar(i, j)*derivative_1
                        parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                            matrix_bar(i, j)*derivative_2
                        parameter_bar(offset + 2) = parameter_bar(offset + 2) + &
                            matrix_bar(i, j)*derivative_3
                        parameter_bar(offset + 3) = parameter_bar(offset + 3) + &
                            matrix_bar(i, j)*derivative_4
                    end do
                end do
            else
                parameter_bar(offset) = parameter_bar(offset) + &
                    sum(matrix_bar*leaf_matrix(self, x1, x2, variance, lengthscale, 1))
            end if
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52) then
                if (self%kind /= KERNEL_RBF) then
                    parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                        sum(matrix_bar*leaf_matrix(self, x1, x2, variance, lengthscale, 2))
                end if
            end if
        case (KERNEL_RBF_ARD)
            call ard_rbf_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, offset)
        end select
    end subroutine kernel_parameter_vjp_impl

    subroutine change_point_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, offset)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        integer, intent(in) :: offset
        real(dp), allocatable :: left_matrix(:, :), right_matrix(:, :)
        real(dp), allocatable :: gate_left_matrix(:, :), gate_right_matrix(:, :)
        real(dp) :: gate_left, gate_right, gate_gradient_left(2), gate_gradient_right(2)
        integer :: left_count, right_count, i, j

        left_count = self%left%parameter_count()
        right_count = self%right%parameter_count()
        allocate(left_matrix(size(x1, 1), size(x2, 1)))
        allocate(right_matrix, mold=left_matrix)
        allocate(gate_left_matrix, mold=left_matrix)
        allocate(gate_right_matrix, mold=left_matrix)
        call kernel_matrix_impl(self%left, x1, x2, left_matrix)
        call kernel_matrix_impl(self%right, x1, x2, right_matrix)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                call change_point_parameter_gates(self, x1(i, :), x2(j, :), gate_left, &
                    gate_right, gate_gradient_left, gate_gradient_right)
                gate_left_matrix(i, j) = gate_left
                gate_right_matrix(i, j) = gate_right
                parameter_bar(offset + left_count + right_count) = &
                    parameter_bar(offset + left_count + right_count) + matrix_bar(i, j)* &
                    (gate_gradient_left(1)*left_matrix(i, j) + &
                    gate_gradient_right(1)*right_matrix(i, j))
                parameter_bar(offset + left_count + right_count + 1) = &
                    parameter_bar(offset + left_count + right_count + 1) + matrix_bar(i, j)* &
                    (gate_gradient_left(2)*left_matrix(i, j) + &
                    gate_gradient_right(2)*right_matrix(i, j))
            end do
        end do
        call kernel_parameter_vjp_impl(self%left, x1, x2, matrix_bar*gate_left_matrix, &
            parameter_bar, offset)
        call kernel_parameter_vjp_impl(self%right, x1, x2, matrix_bar*gate_right_matrix, &
            parameter_bar, offset + left_count)
    end subroutine change_point_parameter_vjp

    subroutine change_point_parameter_gates(self, x1, x2, gate_left, gate_right, &
            gate_gradient_left, gate_gradient_right)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: gate_left, gate_right
        real(dp), intent(out) :: gate_gradient_left(2), gate_gradient_right(2)
        real(dp) :: width, center, z1, z2, s1, s2, q1, q2
        real(dp) :: gradient_1(2), gradient_2(2)
        integer :: feature

        width = exp(self%log_parameters(1))
        center = self%log_parameters(2)
        feature = self%change_feature
        z1 = (x1(feature) - center)/width
        z2 = (x2(feature) - center)/width
        s1 = 0.5_dp*(1.0_dp + tanh(z1))
        s2 = 0.5_dp*(1.0_dp + tanh(z2))
        q1 = 0.5_dp*(1.0_dp - tanh(z1)**2)
        q2 = 0.5_dp*(1.0_dp - tanh(z2)**2)
        gradient_1 = q1*[-z1, -1.0_dp/width]
        gradient_2 = q2*[-z2, -1.0_dp/width]
        gate_left = s1*s2
        gate_right = (1.0_dp - s1)*(1.0_dp - s2)
        gate_gradient_left = gradient_1*s2 + s1*gradient_2
        gate_gradient_right = -gradient_1*(1.0_dp - s2) - (1.0_dp - s1)*gradient_2
    end subroutine change_point_parameter_gates

    subroutine kernel_rbf_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, offset)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        integer, intent(in) :: offset
        real(dp) :: variance, lengthscale, r2, value
        real(dp) :: dvariance, ddistance, dlengthscale
        integer :: i, j

        variance = exp(self%log_parameters(1))
        lengthscale = exp(self%log_parameters(2))
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                r2 = sum((x1(i, :) - x2(j, :))**2)
                call fortml_generated_rbf_leaf_derivatives( &
                    variance, r2, lengthscale, value, dvariance, ddistance, &
                    dlengthscale)
                parameter_bar(offset) = parameter_bar(offset) + matrix_bar(i, j)* &
                    dvariance*variance
                parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                    matrix_bar(i, j)*dlengthscale*lengthscale
            end do
        end do
    end subroutine kernel_rbf_parameter_vjp

    real(dp) function ard_rbf_value(self, x1, x2) result(value)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp) :: weighted_squared_distance, difference
        integer :: feature

        weighted_squared_distance = 0.0_dp
        do feature = 1, self%input_dim
            difference = x1(feature) - x2(feature)
            weighted_squared_distance = weighted_squared_distance + difference*difference* &
                exp(-2.0_dp*self%log_parameters(feature + 1))
        end do
        value = exp(self%log_parameters(1) - 0.5_dp*weighted_squared_distance)
    end function ard_rbf_value

    subroutine ard_rbf_matrix(self, x1, x2, matrix)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :)
        real(dp), intent(out) :: matrix(:, :)
        real(dp) :: weighted_squared_distance, difference
        integer :: i, j, feature

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                weighted_squared_distance = 0.0_dp
                do feature = 1, self%input_dim
                    difference = x1(i, feature) - x2(j, feature)
                    weighted_squared_distance = weighted_squared_distance + difference*difference* &
                        exp(-2.0_dp*self%log_parameters(feature + 1))
                end do
                matrix(i, j) = exp(self%log_parameters(1) - 0.5_dp*weighted_squared_distance)
            end do
        end do
    end subroutine ard_rbf_matrix

    subroutine ard_rbf_matrix_jvp(self, x1, x2, direction, matrix, matrix_dot)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), direction(:)
        real(dp), intent(out) :: matrix(:, :), matrix_dot(:, :)
        real(dp) :: weighted_squared_distance, log_direction, q, difference
        real(dp) :: value
        integer :: i, j, feature

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                weighted_squared_distance = 0.0_dp
                log_direction = direction(1)
                do feature = 1, self%input_dim
                    difference = x1(i, feature) - x2(j, feature)
                    q = difference*difference*exp(-2.0_dp*self%log_parameters(feature + 1))
                    weighted_squared_distance = weighted_squared_distance + q
                    log_direction = log_direction + q*direction(feature + 1)
                end do
                value = exp(self%log_parameters(1) - 0.5_dp*weighted_squared_distance)
                matrix(i, j) = value
                matrix_dot(i, j) = value*log_direction
            end do
        end do
    end subroutine ard_rbf_matrix_jvp

    subroutine ard_rbf_input_derivatives(self, x1, x2, value, gradient_x1, &
            gradient_x2, mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: weighted_squared_distance, difference_i, difference_j
        real(dp) :: inverse_length_i_squared, inverse_length_j_squared
        integer :: i, j

        weighted_squared_distance = 0.0_dp
        do i = 1, self%input_dim
            difference_i = x1(i) - x2(i)
            weighted_squared_distance = weighted_squared_distance + difference_i*difference_i* &
                exp(-2.0_dp*self%log_parameters(i + 1))
        end do
        value = exp(self%log_parameters(1) - 0.5_dp*weighted_squared_distance)
        do i = 1, self%input_dim
            difference_i = x1(i) - x2(i)
            inverse_length_i_squared = exp(-2.0_dp*self%log_parameters(i + 1))
            gradient_x1(i) = -value*difference_i*inverse_length_i_squared
            gradient_x2(i) = -gradient_x1(i)
            do j = 1, self%input_dim
                difference_j = x1(j) - x2(j)
                inverse_length_j_squared = exp(-2.0_dp*self%log_parameters(j + 1))
                mixed_hessian(i, j) = value*( &
                    inverse_length_i_squared*merge(1.0_dp, 0.0_dp, i == j) - &
                    difference_i*difference_j*inverse_length_i_squared* &
                    inverse_length_j_squared)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ard_rbf_input_derivatives

    subroutine ard_rbf_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, offset)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        integer, intent(in) :: offset
        real(dp) :: weighted_squared_distance, difference, value, q
        integer :: i, j, feature

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                weighted_squared_distance = 0.0_dp
                do feature = 1, self%input_dim
                    difference = x1(i, feature) - x2(j, feature)
                    weighted_squared_distance = weighted_squared_distance + difference*difference* &
                        exp(-2.0_dp*self%log_parameters(feature + 1))
                end do
                value = exp(self%log_parameters(1) - 0.5_dp*weighted_squared_distance)
                parameter_bar(offset) = parameter_bar(offset) + matrix_bar(i, j)*value
                do feature = 1, self%input_dim
                    difference = x1(i, feature) - x2(j, feature)
                    q = difference*difference*exp(-2.0_dp*self%log_parameters(feature + 1))
                    parameter_bar(offset + feature) = parameter_bar(offset + feature) + &
                        matrix_bar(i, j)*value*q
                end do
            end do
        end do
    end subroutine ard_rbf_parameter_vjp

    subroutine ard_rbf_parameter_hvp(self, x1, x2, matrix_bar, direction, &
            parameter_bar, parameter_bar_dot, offset, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :), direction(:)
        real(dp), intent(inout) :: parameter_bar(:), parameter_bar_dot(:)
        integer, intent(in) :: offset
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: weighted_squared_distance, direction_log, difference, value, q, q_dot
        integer :: i, j, feature

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                weighted_squared_distance = 0.0_dp
                direction_log = direction(1)
                do feature = 1, self%input_dim
                    difference = x1(i, feature) - x2(j, feature)
                    q = difference*difference*exp(-2.0_dp*self%log_parameters(feature + 1))
                    weighted_squared_distance = weighted_squared_distance + q
                    direction_log = direction_log + q*direction(feature + 1)
                end do
                value = exp(self%log_parameters(1) - 0.5_dp*weighted_squared_distance)
                parameter_bar(offset) = parameter_bar(offset) + matrix_bar(i, j)*value
                parameter_bar_dot(offset) = parameter_bar_dot(offset) + &
                    matrix_bar(i, j)*value*direction_log
                do feature = 1, self%input_dim
                    difference = x1(i, feature) - x2(j, feature)
                    q = difference*difference*exp(-2.0_dp*self%log_parameters(feature + 1))
                    q_dot = -2.0_dp*q*direction(feature + 1)
                    parameter_bar(offset + feature) = parameter_bar(offset + feature) + &
                        matrix_bar(i, j)*value*q
                    parameter_bar_dot(offset + feature) = &
                        parameter_bar_dot(offset + feature) + matrix_bar(i, j)*value*( &
                        q*direction_log + q_dot)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ard_rbf_parameter_hvp

    function leaf_matrix(self, x1, x2, variance, lengthscale, derivative_kind) &
            result(matrix)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), variance, lengthscale
        integer, intent(in) :: derivative_kind
        real(dp), allocatable :: matrix(:, :)
        real(dp) :: r2, value, length_derivative
        integer :: i, j

        allocate(matrix(size(x1, 1), size(x2, 1)))
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                r2 = sum((x1(i, :) - x2(j, :))**2)
                if (self%kind == KERNEL_RBF .and. derivative_kind == 1) then
                    call fortml_generated_rbf_leaf_fortran( &
                        variance, r2, lengthscale, value)
                else
                    call leaf_value_and_length_derivative(self%kind, variance, &
                        lengthscale, r2, value, length_derivative)
                end if
                if (self%kind == KERNEL_LINEAR) then
                    value = variance*dot_product(x1(i, :), x2(j, :))
                    matrix(i, j) = value
                else if (self%kind == KERNEL_CONSTANT) then
                    matrix(i, j) = variance
                else if (self%kind == KERNEL_WHITE_NOISE) then
                    matrix(i, j) = variance*merge(1.0_dp, 0.0_dp, &
                        same_row(x1, i, x2, j))
                else if (derivative_kind == 1) then
                    matrix(i, j) = value
                else
                    matrix(i, j) = length_derivative
                end if
            end do
        end do
    end function leaf_matrix

    subroutine leaf_value_and_length_derivative(kind, variance, lengthscale, r2, &
            value, length_derivative)
        integer, intent(in) :: kind
        real(dp), intent(in) :: variance, lengthscale, r2
        real(dp), intent(out) :: value, length_derivative
        real(dp) :: r, exponential, a

        value = variance
        length_derivative = 0.0_dp
        if (kind == KERNEL_RBF) then
            value = variance*exp(-0.5_dp*r2/(lengthscale*lengthscale))
            length_derivative = value*r2/(lengthscale*lengthscale)
        else if (kind == KERNEL_MATERN12 .or. kind == KERNEL_MATERN32 .or. &
                kind == KERNEL_MATERN52) then
            r = sqrt(r2)/lengthscale
            if (kind == KERNEL_MATERN12) then
                a = 1.0_dp
                value = variance*exp(-r)
                length_derivative = value*r
            else if (kind == KERNEL_MATERN32) then
                a = sqrt(3.0_dp)
                exponential = exp(-a*r)
                value = variance*(1.0_dp + a*r)*exponential
                length_derivative = variance*3.0_dp*r*r*exponential
            else
                a = sqrt(5.0_dp)
                exponential = exp(-a*r)
                value = variance*(1.0_dp + a*r + 5.0_dp*r*r/3.0_dp)*exponential
                length_derivative = variance*(5.0_dp/3.0_dp)*r*r*(1.0_dp + a*r) &
                    * exponential
            end if
        end if
    end subroutine leaf_value_and_length_derivative

    logical recursive function kernel_valid(self) result(valid)
        class(kernel_t), intent(in) :: self

        valid = self%input_dim > 0
        if (.not. valid) return
        select case (self%kind)
        case (KERNEL_RBF_ARD)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) == self%input_dim + 1
        case (KERNEL_SPECTRAL_MIXTURE)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) >= 1 + 2*self%input_dim
            if (.not. valid) return
            valid = mod(size(self%log_parameters), 1 + 2*self%input_dim) == 0
        case (KERNEL_RBF, KERNEL_MATERN12, KERNEL_MATERN32, KERNEL_MATERN52, &
                KERNEL_COSINE)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) == 2
        case (KERNEL_PERIODIC, KERNEL_RATIONAL_QUADRATIC)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) == 3
        case (KERNEL_LOCAL_PERIODIC)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) == 4
        case (KERNEL_CHANGE_POINT)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) == 2
            if (.not. valid) return
            valid = associated(self%left) .and. associated(self%right)
            if (.not. valid) return
            valid = self%change_feature >= 1 .and. self%change_feature <= self%input_dim
            if (.not. valid) return
            valid = kernel_valid(self%left) .and. kernel_valid(self%right)
            if (.not. valid) return
            valid = self%left%input_dim == self%input_dim .and. &
                self%right%input_dim == self%input_dim
        case (KERNEL_POLYNOMIAL)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) == 4
        case (KERNEL_LINEAR, KERNEL_CONSTANT, KERNEL_WHITE_NOISE)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) == 1
        case (KERNEL_USER)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) == 1
            if (.not. valid) return
            valid = allocated(self%formula)
            if (.not. valid) return
            valid = self%formula%static_lowering_eligible()
        case (KERNEL_SUM, KERNEL_PRODUCT)
            valid = associated(self%left) .and. associated(self%right)
            if (.not. valid) return
            valid = self%left%input_dim == self%right%input_dim
            if (.not. valid) return
            valid = kernel_valid(self%left)
            if (.not. valid) return
            valid = kernel_valid(self%right)
        case default
            valid = .false.
        end select
    end function kernel_valid

    real(dp) function spectral_value(self, x1, x2) result(value)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp) :: tau(self%input_dim), weight, product_value
        real(dp) :: scales(self%input_dim), means(self%input_dim)
        real(dp) :: factors(self%input_dim), factors_dot(self%input_dim)
        real(dp) :: factors_second(self%input_dim), means_derivative(self%input_dim)
        integer :: q

        value = 0.0_dp
        tau = x1 - x2
        do q = 1, size(self%log_parameters)/(1 + 2*self%input_dim)
            call spectral_component_factors(self, q, tau, weight, scales, means, &
                factors, factors_dot, factors_second, means_derivative)
            product_value = product(factors)
            value = value + weight*product_value
        end do
    end function spectral_value

    subroutine spectral_matrix(self, x1, x2, matrix)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :)
        real(dp), intent(out) :: matrix(:, :)
        integer :: i, j

        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                matrix(i, j) = spectral_value(self, x1(i, :), x2(j, :))
            end do
        end do
    end subroutine spectral_matrix

    subroutine spectral_matrix_jvp(self, x1, x2, direction, matrix, matrix_dot)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), direction(:)
        real(dp), intent(out) :: matrix(:, :), matrix_dot(:, :)
        integer :: i, j, q, d, base, block
        real(dp) :: tau(self%input_dim), weight, product_value, product_dot
        real(dp) :: scales(self%input_dim), means(self%input_dim)
        real(dp) :: factors(self%input_dim), factors_dot(self%input_dim)
        real(dp) :: factors_second(self%input_dim), means_derivative(self%input_dim)
        real(dp) :: exponent_direction, mean_direction, av, c_mu
        real(dp) :: two_pi, scale_log_direction, mean_dot

        block = 1 + 2*self%input_dim
        two_pi = 2.0_dp*acos(-1.0_dp)
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                tau = x1(i, :) - x2(j, :)
                matrix(i, j) = 0.0_dp
                matrix_dot(i, j) = 0.0_dp
                do q = 1, size(self%log_parameters)/block
                    base = (q - 1)*block + 1
                    call spectral_component_factors(self, q, tau, weight, scales, means, &
                        factors, factors_dot, factors_second, means_derivative)
                    product_value = product(factors)
                    product_dot = 0.0_dp
                    do d = 1, self%input_dim
                        scale_log_direction = direction(base + d)
                        mean_direction = direction(base + self%input_dim + d)
                        av = -two_pi*two_pi*tau(d)*tau(d)*scales(d)*scales(d)
                        c_mu = -two_pi*tau(d)*sin(two_pi*tau(d)*means(d))
                        mean_dot = exp(-0.5_dp*two_pi*two_pi*tau(d)*tau(d)*scales(d)*scales(d))* &
                            (av*scale_log_direction*cos(two_pi*tau(d)*means(d)) + &
                            c_mu*mean_direction)
                        factors_dot(d) = mean_dot
                        product_dot = product_dot + mean_dot* &
                            spectral_product_except(factors, d)
                    end do
                    matrix(i, j) = matrix(i, j) + weight*product_value
                    matrix_dot(i, j) = matrix_dot(i, j) + weight*( &
                        direction(base)*product_value + product_dot)
                end do
            end do
        end do
    end subroutine spectral_matrix_jvp

    subroutine spectral_input_derivatives(self, x1, x2, value, gradient_x1, &
            gradient_x2, mixed_hessian, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:), x2(:)
        real(dp), intent(out) :: value, gradient_x1(:), gradient_x2(:)
        real(dp), intent(out) :: mixed_hessian(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: q, i, j
        real(dp) :: tau(self%input_dim), weight, product_value
        real(dp) :: scales(self%input_dim), means(self%input_dim)
        real(dp) :: factors(self%input_dim), factors_dot(self%input_dim)
        real(dp) :: factors_second(self%input_dim), means_derivative(self%input_dim)
        real(dp) :: product_except, second_tau

        value = 0.0_dp
        gradient_x1 = 0.0_dp
        gradient_x2 = 0.0_dp
        mixed_hessian = 0.0_dp
        tau = x1 - x2
        do q = 1, size(self%log_parameters)/(1 + 2*self%input_dim)
            call spectral_component_factors(self, q, tau, weight, scales, means, &
                factors, factors_dot, factors_second, means_derivative)
            product_value = product(factors)
            value = value + weight*product_value
            do i = 1, self%input_dim
                product_except = spectral_product_except(factors, i)
                gradient_x1(i) = gradient_x1(i) + weight*factors_dot(i)*product_except
                gradient_x2(i) = gradient_x2(i) - weight*factors_dot(i)*product_except
                do j = 1, self%input_dim
                    if (i == j) then
                        second_tau = factors_second(i)*product_except
                    else
                        second_tau = factors_dot(i)*factors_dot(j)* &
                            spectral_product_except_two(factors, i, j)
                    end if
                    mixed_hessian(i, j) = mixed_hessian(i, j) - weight*second_tau
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine spectral_input_derivatives

    subroutine spectral_parameter_vjp(self, x1, x2, matrix_bar, parameter_bar, offset)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        integer, intent(in) :: offset
        integer :: i, j, q, d, base, block
        real(dp) :: tau(self%input_dim), weight, product_value, mean_product
        real(dp) :: scales(self%input_dim), means(self%input_dim)
        real(dp) :: factors(self%input_dim), factors_dot(self%input_dim)
        real(dp) :: factors_second(self%input_dim), means_derivative(self%input_dim)
        real(dp) :: av

        block = 1 + 2*self%input_dim
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                tau = x1(i, :) - x2(j, :)
                do q = 1, size(self%log_parameters)/block
                    base = offset + (q - 1)*block
                    call spectral_component_factors(self, q, tau, weight, scales, means, &
                        factors, factors_dot, factors_second, means_derivative)
                    product_value = product(factors)
                    parameter_bar(base) = parameter_bar(base) + matrix_bar(i, j)* &
                        weight*product_value
                    do d = 1, self%input_dim
                        av = -4.0_dp*acos(-1.0_dp)**2*tau(d)*tau(d)*scales(d)*scales(d)
                        parameter_bar(base + d) = parameter_bar(base + d) + matrix_bar(i, j)* &
                            weight*product_value*av
                        mean_product = means_derivative(d)*spectral_product_except(factors, d)
                        parameter_bar(base + self%input_dim + d) = &
                            parameter_bar(base + self%input_dim + d) + matrix_bar(i, j)* &
                            weight*mean_product
                    end do
                end do
            end do
        end do
    end subroutine spectral_parameter_vjp

    subroutine spectral_parameter_hvp(self, x1, x2, matrix_bar, direction, &
            parameter_bar, parameter_bar_dot, offset, status)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :), direction(:)
        real(dp), intent(inout) :: parameter_bar(:), parameter_bar_dot(:)
        integer, intent(in) :: offset
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, q, k, block, base
        real(dp) :: tau(self%input_dim), component, component_dot
        real(dp) :: gradients(1 + 2*self%input_dim), gradients_dot(1 + 2*self%input_dim)

        block = 1 + 2*self%input_dim
        do j = 1, size(x2, 1)
            do i = 1, size(x1, 1)
                tau = x1(i, :) - x2(j, :)
                do q = 1, size(self%log_parameters)/block
                    base = offset + (q - 1)*block
                    call spectral_component_hvp(self, q, tau, direction, base, component, &
                        component_dot, gradients, gradients_dot)
                    do k = 1, block
                        parameter_bar(base + k - 1) = parameter_bar(base + k - 1) + &
                            matrix_bar(i, j)*gradients(k)
                        parameter_bar_dot(base + k - 1) = parameter_bar_dot(base + k - 1) + &
                            matrix_bar(i, j)*gradients_dot(k)
                    end do
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine spectral_parameter_hvp

    subroutine spectral_component_hvp(self, q, tau, direction, base, component, &
            component_dot, gradients, gradients_dot)
        class(kernel_t), intent(in) :: self
        integer, intent(in) :: q, base
        real(dp), intent(in) :: tau(:), direction(:)
        real(dp), intent(out) :: component, component_dot
        real(dp), intent(out) :: gradients(:), gradients_dot(:)
        integer :: d, e, block
        real(dp) :: weight, product_value, product_dot, av, av_dot, dlw
        real(dp) :: scales(self%input_dim), means(self%input_dim)
        real(dp) :: factors(self%input_dim), factors_tau(self%input_dim)
        real(dp) :: factors_second(self%input_dim), means_derivative(self%input_dim)
        real(dp) :: factors_parameter_dot(self%input_dim)
        real(dp) :: mean_second(self%input_dim), mean_product, mean_product_dot
        real(dp) :: scale_direction, mean_direction, two_pi

        block = 1 + 2*self%input_dim
        two_pi = 2.0_dp*acos(-1.0_dp)
        call spectral_component_factors(self, q, tau, weight, scales, means, factors, &
            factors_tau, factors_second, means_derivative)
        product_value = product(factors)
        product_dot = 0.0_dp
        do d = 1, self%input_dim
            scale_direction = direction(base + d)
            mean_direction = direction(base + self%input_dim + d)
            av = -two_pi*two_pi*tau(d)*tau(d)*scales(d)*scales(d)
            factors_parameter_dot(d) = exp(-0.5_dp*two_pi*two_pi*tau(d)*tau(d)* &
                scales(d)*scales(d))* &
                (av*scale_direction*cos(two_pi*tau(d)*means(d)) + &
                (-two_pi*tau(d)*sin(two_pi*tau(d)*means(d)))*mean_direction)
            product_dot = product_dot + factors_parameter_dot(d)* &
                spectral_product_except(factors, d)
        end do
        dlw = direction(base)
        component = weight*product_value
        component_dot = weight*(dlw*product_value + product_dot)
        gradients = 0.0_dp
        gradients_dot = 0.0_dp
        gradients(1) = component
        gradients_dot(1) = component_dot
        do d = 1, self%input_dim
            scale_direction = direction(base + d)
            av = -two_pi*two_pi*tau(d)*tau(d)*scales(d)*scales(d)
            av_dot = 2.0_dp*av*scale_direction
            gradients(1 + d) = component*av
            gradients_dot(1 + d) = component_dot*av + component*av_dot
            mean_product = means_derivative(d)*spectral_product_except(factors, d)
            mean_second(d) = -two_pi*two_pi*tau(d)*tau(d)* &
                cos(two_pi*tau(d)*means(d))* &
                exp(-0.5_dp*two_pi*two_pi*tau(d)*tau(d)*scales(d)*scales(d))
            mean_product_dot = (mean_second(d)*direction(base + self%input_dim + d) + &
                exp(-0.5_dp*two_pi*two_pi*tau(d)*tau(d)*scales(d)*scales(d))*av* &
                scale_direction*(-two_pi*tau(d)*sin(two_pi*tau(d)*means(d))))* &
                spectral_product_except(factors, d)
            do e = 1, self%input_dim
                if (e /= d) then
                    mean_product_dot = mean_product_dot + means_derivative(d)* &
                        factors_parameter_dot(e)*spectral_product_except_two(factors, d, e)
                end if
            end do
            gradients(1 + self%input_dim + d) = weight*mean_product
            gradients_dot(1 + self%input_dim + d) = weight*(dlw*mean_product + &
                mean_product_dot)
        end do
    end subroutine spectral_component_hvp

    subroutine spectral_component_factors(self, q, tau, weight, scales, means, factors, &
            factors_tau, factors_second, means_derivative)
        class(kernel_t), intent(in) :: self
        integer, intent(in) :: q
        real(dp), intent(in) :: tau(:)
        real(dp), intent(out) :: weight, scales(:), means(:), factors(:), factors_tau(:)
        real(dp), intent(out) :: factors_second(:), means_derivative(:)
        integer :: d, base
        real(dp) :: two_pi, argument, exponential, cosine_value, sine_value
        real(dp) :: exp_tau, exp_second, cosine_tau, cosine_second

        two_pi = 2.0_dp*acos(-1.0_dp)
        base = (q - 1)*(1 + 2*self%input_dim)
        weight = exp(self%log_parameters(base + 1))
        do d = 1, self%input_dim
            scales(d) = exp(self%log_parameters(base + 1 + d))
            means(d) = self%log_parameters(base + 1 + self%input_dim + d)
            argument = two_pi*tau(d)*means(d)
            exponential = exp(-0.5_dp*two_pi*two_pi*tau(d)*tau(d)*scales(d)*scales(d))
            cosine_value = cos(argument)
            sine_value = sin(argument)
            exp_tau = -two_pi*two_pi*tau(d)*scales(d)*scales(d)*exponential
            exp_second = exponential*( &
                -two_pi*two_pi*scales(d)*scales(d) + &
                two_pi**4*tau(d)*tau(d)*scales(d)**4)
            cosine_tau = -two_pi*means(d)*sine_value
            cosine_second = -two_pi*two_pi*means(d)*means(d)*cosine_value
            factors(d) = exponential*cosine_value
            factors_tau(d) = exp_tau*cosine_value + exponential*cosine_tau
            factors_second(d) = exp_second*cosine_value + 2.0_dp*exp_tau*cosine_tau + &
                exponential*cosine_second
            means_derivative(d) = exponential*(-two_pi*tau(d)*sine_value)
        end do
    end subroutine spectral_component_factors

    real(dp) function spectral_product_except(values, skip) result(product_value)
        real(dp), intent(in) :: values(:)
        integer, intent(in) :: skip
        integer :: i

        product_value = 1.0_dp
        do i = 1, size(values)
            if (i /= skip) product_value = product_value*values(i)
        end do
    end function spectral_product_except

    real(dp) function spectral_product_except_two(values, skip_one, skip_two) result(product_value)
        real(dp), intent(in) :: values(:)
        integer, intent(in) :: skip_one, skip_two
        integer :: i

        product_value = 1.0_dp
        do i = 1, size(values)
            if (i /= skip_one .and. i /= skip_two) product_value = product_value*values(i)
        end do
    end function spectral_product_except_two

    logical function same_row(x1, i, x2, j) result(same)
        real(dp), intent(in) :: x1(:, :), x2(:, :)
        integer, intent(in) :: i, j
        integer :: k

        same = .true.
        do k = 1, size(x1, 2)
            if (x1(i, k) /= x2(j, k)) then
                same = .false.
                return
            end if
        end do
    end function same_row

end module fortml_kernels
