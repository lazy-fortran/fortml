module fortml_kernels
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortml_generated_matern_products, only: fortml_matern12_hvp, &
        fortml_matern32_hvp, fortml_matern52_hvp
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

    type, public :: kernel_t
        integer :: kind = 0
        integer :: input_dim = 0
        real(dp), allocatable :: log_parameters(:)
        type(kernel_t), pointer :: left => null()
        type(kernel_t), pointer :: right => null()
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
    public :: make_matern12_kernel
    public :: make_matern32_kernel
    public :: make_matern52_kernel
    public :: make_linear_kernel
    public :: make_constant_kernel
    public :: make_white_noise_kernel
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
        case (KERNEL_RBF, KERNEL_MATERN12, KERNEL_MATERN32, KERNEL_MATERN52)
            count = 2
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
        integer :: left_count

        allocate(parameters(self%parameter_count()))
        if (self%kind == KERNEL_SUM .or. self%kind == KERNEL_PRODUCT) then
            left_count = self%left%parameter_count()
            parameters(1:left_count) = self%left%parameters()
            parameters(left_count + 1:) = self%right%parameters()
        else if (size(parameters) > 0) then
            parameters = self%log_parameters
        end if
    end function kernel_parameters

    recursive subroutine kernel_set_parameters(self, parameters, status)
        class(kernel_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: left_count

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
        real(dp) :: variance, lengthscale, r2, length_derivative

        value = 0.0_dp
        if (size(x1) /= self%input_dim) return
        if (size(x2) /= self%input_dim) return
        if (.not. kernel_valid(self)) return
        select case (self%kind)
        case (KERNEL_SUM)
            value = self%left%value(x1, x2) + self%right%value(x1, x2)
        case (KERNEL_PRODUCT)
            value = self%left%value(x1, x2)*self%right%value(x1, x2)
        case default
            variance = exp(self%log_parameters(1))
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52) then
                lengthscale = exp(self%log_parameters(2))
            else
                lengthscale = 1.0_dp
            end if
            r2 = sum((x1 - x2)**2)
            value = 0.0_dp
            if (self%kind == KERNEL_RBF) then
                call fortml_generated_rbf_leaf_fortran( &
                    variance, r2, lengthscale, value)
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
        case default
            continue
        end select

        variance_log = self%log_parameters(1)
        if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
            self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52) then
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
            kind == KERNEL_MATERN32 .or. kind == KERNEL_MATERN52) then
            allocate(kernel%log_parameters(2))
            kernel%log_parameters = [log(variance), log(lengthscale)]
        else
            allocate(kernel%log_parameters(1))
            kernel%log_parameters(1) = log(variance)
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine make_leaf

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

    recursive subroutine kernel_matrix_impl(self, x1, x2, matrix)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :)
        real(dp), intent(out) :: matrix(:, :)
        real(dp) :: variance, lengthscale, r2, value, dummy
        real(dp), allocatable :: other(:, :)
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
        case default
            variance = exp(self%log_parameters(1))
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52) then
                lengthscale = exp(self%log_parameters(2))
            else
                lengthscale = 1.0_dp
            end if
            do j = 1, size(x2, 1)
                do i = 1, size(x1, 1)
                    r2 = sum((x1(i, :) - x2(j, :))**2)
                    value = 0.0_dp
                    if (self%kind == KERNEL_RBF) then
                        call fortml_generated_rbf_leaf_fortran( &
                            variance, r2, lengthscale, value)
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
        real(dp) :: variance, lengthscale, r2, value, length_derivative
        real(dp) :: log_variance_dot, log_lengthscale_dot
        real(dp) :: dvariance, ddistance, dlengthscale
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
        case default
            variance = exp(self%log_parameters(1))
            log_variance_dot = direction(1)
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52) then
                lengthscale = exp(self%log_parameters(2))
                log_lengthscale_dot = direction(2)
            else
                lengthscale = 1.0_dp
                log_lengthscale_dot = 0.0_dp
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

    recursive subroutine kernel_parameter_vjp_impl(self, x1, x2, matrix_bar, &
            parameter_bar, offset)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), matrix_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        integer, intent(in) :: offset
        real(dp), allocatable :: left_matrix(:, :), right_matrix(:, :)
        real(dp) :: variance, lengthscale
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
        case default
            variance = exp(self%log_parameters(1))
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52) then
                lengthscale = exp(self%log_parameters(2))
            else
                lengthscale = 1.0_dp
            end if
            if (self%kind == KERNEL_RBF) then
                call kernel_rbf_parameter_vjp(self, x1, x2, matrix_bar, &
                    parameter_bar, offset)
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
        end select
    end subroutine kernel_parameter_vjp_impl

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
        case (KERNEL_RBF, KERNEL_MATERN12, KERNEL_MATERN32, KERNEL_MATERN52)
            valid = allocated(self%log_parameters)
            if (.not. valid) return
            valid = size(self%log_parameters) == 2
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
