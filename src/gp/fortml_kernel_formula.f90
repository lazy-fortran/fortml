module fortml_kernel_formula
    !! Safe static lowering contract for user-supplied kernel formulas.
    !!
    !! A user formula reaches an accelerator region as *data*, never as code. A
    !! procedure pointer cannot be lowered: it would have to be called back from
    !! inside the device loop, which is why `fortml_basis` refuses static
    !! lowering for its callback maps. A formula built here is instead a short
    !! postfix program over a fixed opcode set, so the operator can inline it
    !! into the same stack machine that already evaluates built-in kernel trees.
    !!
    !! Every opcode is a total function of its operands, which is what makes the
    !! contract safe rather than merely restrictive:
    !!
    !! * `push_squared_distance`, `push_distance`, `push_inner_product` and
    !!   `push_constant` are the only sources of values. `push_distance` is a
    !!   primitive rather than a general square root, so no operand can reach
    !!   `sqrt` negative.
    !! * `add`, `subtract`, `multiply`, `negate` and `exponential` are defined
    !!   everywhere.
    !! * `divide_by_constant` carries its divisor, and validation rejects a zero
    !!   one. There is no general division, so no operand can reach a divide
    !!   with a value only known at run time.
    !!
    !! Validation additionally proves the program is well formed for the stack
    !! machine: the depth never underflows, never exceeds the fixed stack, and
    !! ends at exactly one value. A formula that fails validation is refused by
    !! both kernel construction and operator lowering, so an ill-formed program
    !! can never reach a device region.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    ! Negative codes so a lowered formula opcode can share the operator's
    ! program array with the positive built-in kernel kinds.
    integer, parameter, public :: OPCODE_PUSH_R2 = -1
    integer, parameter, public :: OPCODE_PUSH_R = -2
    integer, parameter, public :: OPCODE_PUSH_DOT = -3
    integer, parameter, public :: OPCODE_PUSH_CONST = -4
    integer, parameter, public :: OPCODE_ADD = -5
    integer, parameter, public :: OPCODE_SUBTRACT = -6
    integer, parameter, public :: OPCODE_MULTIPLY = -7
    integer, parameter, public :: OPCODE_NEGATE = -8
    integer, parameter, public :: OPCODE_EXP = -9
    integer, parameter, public :: OPCODE_DIVIDE_CONST = -10

    integer, parameter, public :: MAX_FORMULA_LENGTH = 64
    integer, parameter, public :: MAX_FORMULA_STACK = 16

    type, public :: kernel_formula_t
        integer :: length = 0
        logical :: validated = .false.
        integer :: opcode(MAX_FORMULA_LENGTH) = 0
        real(dp) :: operand(MAX_FORMULA_LENGTH) = 0.0_dp
    contains
        procedure, public :: reset => formula_reset
        procedure, public :: push_squared_distance => formula_push_r2
        procedure, public :: push_distance => formula_push_r
        procedure, public :: push_inner_product => formula_push_dot
        procedure, public :: push_constant => formula_push_constant
        procedure, public :: add => formula_add
        procedure, public :: subtract => formula_subtract
        procedure, public :: multiply => formula_multiply
        procedure, public :: negate => formula_negate
        procedure, public :: exponential => formula_exp
        procedure, public :: divide_by_constant => formula_divide_constant
        procedure, public :: validate => formula_validate
        procedure, public :: static_lowering_eligible => formula_eligible
        procedure, public :: evaluate => formula_evaluate
    end type kernel_formula_t

    public :: formula_opcode_is_known

contains

    subroutine formula_reset(self)
        class(kernel_formula_t), intent(inout) :: self

        self%length = 0
        self%validated = .false.
        self%opcode = 0
        self%operand = 0.0_dp
    end subroutine formula_reset

    subroutine append(self, opcode, operand)
        !! Appending invalidates the formula: it must be validated again before
        !! anything may lower it.
        class(kernel_formula_t), intent(inout) :: self
        integer, intent(in) :: opcode
        real(dp), intent(in) :: operand

        self%validated = .false.
        if (self%length >= MAX_FORMULA_LENGTH) then
            self%length = MAX_FORMULA_LENGTH + 1
            return
        end if
        self%length = self%length + 1
        self%opcode(self%length) = opcode
        self%operand(self%length) = operand
    end subroutine append

    subroutine formula_push_r2(self)
        class(kernel_formula_t), intent(inout) :: self

        call append(self, OPCODE_PUSH_R2, 0.0_dp)
    end subroutine formula_push_r2

    subroutine formula_push_r(self)
        class(kernel_formula_t), intent(inout) :: self

        call append(self, OPCODE_PUSH_R, 0.0_dp)
    end subroutine formula_push_r

    subroutine formula_push_dot(self)
        class(kernel_formula_t), intent(inout) :: self

        call append(self, OPCODE_PUSH_DOT, 0.0_dp)
    end subroutine formula_push_dot

    subroutine formula_push_constant(self, value)
        class(kernel_formula_t), intent(inout) :: self
        real(dp), intent(in) :: value

        call append(self, OPCODE_PUSH_CONST, value)
    end subroutine formula_push_constant

    subroutine formula_add(self)
        class(kernel_formula_t), intent(inout) :: self

        call append(self, OPCODE_ADD, 0.0_dp)
    end subroutine formula_add

    subroutine formula_subtract(self)
        class(kernel_formula_t), intent(inout) :: self

        call append(self, OPCODE_SUBTRACT, 0.0_dp)
    end subroutine formula_subtract

    subroutine formula_multiply(self)
        class(kernel_formula_t), intent(inout) :: self

        call append(self, OPCODE_MULTIPLY, 0.0_dp)
    end subroutine formula_multiply

    subroutine formula_negate(self)
        class(kernel_formula_t), intent(inout) :: self

        call append(self, OPCODE_NEGATE, 0.0_dp)
    end subroutine formula_negate

    subroutine formula_exp(self)
        class(kernel_formula_t), intent(inout) :: self

        call append(self, OPCODE_EXP, 0.0_dp)
    end subroutine formula_exp

    subroutine formula_divide_constant(self, value)
        class(kernel_formula_t), intent(inout) :: self
        real(dp), intent(in) :: value

        call append(self, OPCODE_DIVIDE_CONST, value)
    end subroutine formula_divide_constant

    subroutine formula_validate(self, status)
        !! Prove the program is lowerable, or refuse it by name.
        class(kernel_formula_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer :: i, depth

        self%validated = .false.
        if (self%length < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel formula: the program is empty")
            return
        end if
        if (self%length > MAX_FORMULA_LENGTH) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel formula: the program is too long to lower")
            return
        end if

        depth = 0
        do i = 1, self%length
            if (.not. formula_opcode_is_known(self%opcode(i))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel formula: unknown opcode")
                return
            end if
            select case (self%opcode(i))
            case (OPCODE_PUSH_R2, OPCODE_PUSH_R, OPCODE_PUSH_DOT)
                depth = depth + 1
            case (OPCODE_PUSH_CONST)
                if (.not. finite_operand(self%operand(i))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel formula: constant is not finite")
                    return
                end if
                depth = depth + 1
            case (OPCODE_ADD, OPCODE_SUBTRACT, OPCODE_MULTIPLY)
                if (depth < 2) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel formula: a binary operation has too few operands")
                    return
                end if
                depth = depth - 1
            case (OPCODE_NEGATE, OPCODE_EXP)
                if (depth < 1) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel formula: a unary operation has no operand")
                    return
                end if
            case (OPCODE_DIVIDE_CONST)
                if (depth < 1) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel formula: a unary operation has no operand")
                    return
                end if
                if (.not. finite_operand(self%operand(i))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel formula: divisor is not finite")
                    return
                end if
                if (self%operand(i) == 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "kernel formula: division by zero")
                    return
                end if
            end select
            if (depth > MAX_FORMULA_STACK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "kernel formula: the program needs too deep a stack")
                return
            end if
        end do

        if (depth /= 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "kernel formula: the program does not leave one value")
            return
        end if
        self%validated = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine formula_validate

    logical function formula_eligible(self) result(eligible)
        !! The refusal boundary the operator consults before lowering.
        class(kernel_formula_t), intent(in) :: self

        eligible = self%validated
    end function formula_eligible

    real(dp) function formula_evaluate(self, squared_distance, inner_product) &
            result(value)
        !! Host reference evaluation, sharing no code with the device stack
        !! machine in `fortml_kernel_operator`.
        class(kernel_formula_t), intent(in) :: self
        real(dp), intent(in) :: squared_distance, inner_product
        real(dp) :: stack(MAX_FORMULA_STACK)
        integer :: i, top

        value = 0.0_dp
        if (.not. self%validated) return
        top = 0
        do i = 1, self%length
            select case (self%opcode(i))
            case (OPCODE_PUSH_R2)
                top = top + 1
                stack(top) = squared_distance
            case (OPCODE_PUSH_R)
                top = top + 1
                stack(top) = sqrt(squared_distance)
            case (OPCODE_PUSH_DOT)
                top = top + 1
                stack(top) = inner_product
            case (OPCODE_PUSH_CONST)
                top = top + 1
                stack(top) = self%operand(i)
            case (OPCODE_ADD)
                stack(top - 1) = stack(top - 1) + stack(top)
                top = top - 1
            case (OPCODE_SUBTRACT)
                stack(top - 1) = stack(top - 1) - stack(top)
                top = top - 1
            case (OPCODE_MULTIPLY)
                stack(top - 1) = stack(top - 1)*stack(top)
                top = top - 1
            case (OPCODE_NEGATE)
                stack(top) = -stack(top)
            case (OPCODE_EXP)
                stack(top) = exp(stack(top))
            case (OPCODE_DIVIDE_CONST)
                stack(top) = stack(top)/self%operand(i)
            end select
        end do
        value = stack(top)
    end function formula_evaluate

    logical function formula_opcode_is_known(opcode) result(known)
        integer, intent(in) :: opcode

        known = opcode == OPCODE_PUSH_R2 .or. opcode == OPCODE_PUSH_R .or. &
            opcode == OPCODE_PUSH_DOT .or. opcode == OPCODE_PUSH_CONST .or. &
            opcode == OPCODE_ADD .or. opcode == OPCODE_SUBTRACT .or. &
            opcode == OPCODE_MULTIPLY .or. opcode == OPCODE_NEGATE .or. &
            opcode == OPCODE_EXP .or. opcode == OPCODE_DIVIDE_CONST
    end function formula_opcode_is_known

    logical function finite_operand(value) result(finite)
        real(dp), intent(in) :: value

        finite = value == value
        if (.not. finite) return
        finite = abs(value) <= huge(value)
    end function finite_operand

end module fortml_kernel_formula
