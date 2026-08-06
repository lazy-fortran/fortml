module fortml_kernels
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    interface
        subroutine fortml_generated_rbf_leaf_fortran( &
                variance, distance, lengthscale, output)
            use, intrinsic :: iso_fortran_env, only: real64
            real(real64), intent(in) :: variance, distance, lengthscale
            real(real64), intent(out) :: output
        end subroutine fortml_generated_rbf_leaf_fortran
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

    type, public :: kernel_t
        integer :: kind = 0
        integer :: input_dim = 0
        real(dp), allocatable :: log_parameters(:)
        type(kernel_t), pointer :: left => null()
        type(kernel_t), pointer :: right => null()
    contains
        procedure, public :: parameter_count => kernel_parameter_count
        procedure, public :: parameters => kernel_parameters
        procedure, public :: set_parameters => kernel_set_parameters
        procedure, public :: value => kernel_value
        procedure, public :: input_derivatives => kernel_input_derivatives
        procedure, public :: matrix => kernel_matrix
        procedure, public :: matrix_jvp => kernel_matrix_jvp
        procedure, public :: parameter_vjp => kernel_parameter_vjp
    end type kernel_t

    public :: make_rbf_kernel
    public :: make_matern12_kernel
    public :: make_matern32_kernel
    public :: make_matern52_kernel
    public :: make_linear_kernel
    public :: make_constant_kernel
    public :: make_white_noise_kernel
    public :: kernel_add
    public :: kernel_multiply
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

    recursive integer function kernel_parameter_count(self) result(count)
        class(kernel_t), intent(in) :: self

        count = 0
        select case (self%kind)
        case (KERNEL_RBF, KERNEL_MATERN12, KERNEL_MATERN32, KERNEL_MATERN52)
            count = 2
        case (KERNEL_LINEAR, KERNEL_CONSTANT, KERNEL_WHITE_NOISE)
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
            if (self%kind == KERNEL_RBF) then
                call fortml_generated_rbf_leaf_fortran( &
                    variance, r2, lengthscale, value)
            else
                call leaf_value_and_length_derivative(self%kind, variance, &
                    lengthscale, r2, value, length_derivative)
            end if
            if (self%kind == KERNEL_LINEAR) then
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
                    if (self%kind == KERNEL_RBF) then
                        call fortml_generated_rbf_leaf_fortran( &
                            variance, r2, lengthscale, value)
                    else
                        call leaf_value_and_length_derivative(self%kind, variance, &
                            lengthscale, r2, value, dummy)
                    end if
                    if (self%kind == KERNEL_LINEAR) then
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
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel input_derivatives: this kernel has no smooth input rule")
        end select
    end subroutine kernel_input_derivatives_impl

    recursive subroutine kernel_matrix_jvp_impl(self, x1, x2, direction, matrix, &
            matrix_dot)
        class(kernel_t), intent(in) :: self
        real(dp), intent(in) :: x1(:, :), x2(:, :), direction(:)
        real(dp), intent(out) :: matrix(:, :), matrix_dot(:, :)
        real(dp), allocatable :: other(:, :), other_dot(:, :)
        real(dp) :: variance, lengthscale, r2, value, length_derivative
        real(dp) :: log_variance_dot, log_lengthscale_dot
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
                        call fortml_generated_rbf_leaf_fortran( &
                            variance, r2, lengthscale, value)
                        length_derivative = value*r2/(lengthscale*lengthscale)
                    else
                        call leaf_value_and_length_derivative(self%kind, variance, &
                            lengthscale, r2, value, length_derivative)
                    end if
                    if (self%kind == KERNEL_LINEAR) then
                        matrix(i, j) = variance*dot_product(x1(i, :), x2(j, :))
                        matrix_dot(i, j) = matrix(i, j)*log_variance_dot
                    else if (self%kind == KERNEL_CONSTANT) then
                        matrix(i, j) = variance
                        matrix_dot(i, j) = matrix(i, j)*log_variance_dot
                    else if (self%kind == KERNEL_WHITE_NOISE) then
                        matrix(i, j) = variance*merge(1.0_dp, 0.0_dp, &
                            same_row(x1, i, x2, j))
                        matrix_dot(i, j) = matrix(i, j)*log_variance_dot
                    else
                        matrix(i, j) = value
                        matrix_dot(i, j) = value*log_variance_dot + &
                            length_derivative*log_lengthscale_dot
                    end if
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
            parameter_bar(offset) = parameter_bar(offset) + &
                sum(matrix_bar*leaf_matrix(self, x1, x2, variance, lengthscale, 1))
            if (self%kind == KERNEL_RBF .or. self%kind == KERNEL_MATERN12 .or. &
                self%kind == KERNEL_MATERN32 .or. self%kind == KERNEL_MATERN52) then
                parameter_bar(offset + 1) = parameter_bar(offset + 1) + &
                    sum(matrix_bar*leaf_matrix(self, x1, x2, variance, lengthscale, 2))
            end if
        end select
    end subroutine kernel_parameter_vjp_impl

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
