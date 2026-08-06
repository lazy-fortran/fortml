program test_kernel_formula
    !! Oracles for the user-supplied kernel formula lowering contract.
    !!
    !! Three independent things are checked. A formula that spells out the RBF
    !! kernel must agree with the built-in RBF leaf, which was itself verified
    !! against generated code. A formula that is not a built-in kernel is
    !! checked against a direct pairwise loop written here, through the operator
    !! products that lower it into the device postfix program. And the refusal
    !! boundary is checked case by case: an ill-formed program must never reach
    !! a kernel or an operator.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_kernel_formula, only: kernel_formula_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel, make_user_kernel, &
        kernel_add, make_constant_kernel
    use fortml_kernel_operator, only: kernel_operator_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_rbf_formula_matches_builtin(failures)
    call test_operator_matches_direct_pairwise(failures)
    call test_composite_with_user_leaf(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " kernel formula test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_rbf_formula(formula, lengthscale, status)
        !! exp(-r2/(2 l^2)) written in the opcode grammar.
        type(kernel_formula_t), intent(out) :: formula
        real(dp), intent(in) :: lengthscale
        type(fortnum_status_t), intent(out) :: status

        call formula%reset()
        call formula%push_squared_distance()
        call formula%divide_by_constant(-2.0_dp*lengthscale*lengthscale)
        call formula%exponential()
        call formula%validate(status)
    end subroutine build_rbf_formula

    subroutine build_damped_formula(formula, status)
        !! (1 + 3 r2) exp(-r), which is no built-in kernel.
        type(kernel_formula_t), intent(out) :: formula
        type(fortnum_status_t), intent(out) :: status

        call formula%reset()
        call formula%push_constant(1.0_dp)
        call formula%push_squared_distance()
        call formula%push_constant(3.0_dp)
        call formula%multiply()
        call formula%add()
        call formula%push_distance()
        call formula%negate()
        call formula%exponential()
        call formula%multiply()
        call formula%validate(status)
    end subroutine build_damped_formula

    real(dp) function damped_reference(x1, x2, variance) result(value)
        !! Direct evaluation of the same kernel, sharing no code with the
        !! formula machinery.
        real(dp), intent(in) :: x1(:), x2(:), variance
        real(dp) :: squared

        squared = sum((x1 - x2)**2)
        value = variance*(1.0_dp + 3.0_dp*squared)*exp(-sqrt(squared))
    end function damped_reference

    subroutine build_points(points)
        real(dp), intent(out) :: points(:, :)
        integer :: i, j

        do j = 1, size(points, 2)
            do i = 1, size(points, 1)
                points(i, j) = 0.31_dp*sin(real(i + 2*j, dp)) &
                    + 0.17_dp*cos(real(3*i - j, dp))
            end do
        end do
    end subroutine build_points

    subroutine test_rbf_formula_matches_builtin(failures)
        integer, intent(inout) :: failures
        type(kernel_formula_t) :: formula
        type(kernel_t) :: builtin, user
        type(fortnum_status_t) :: status
        real(dp) :: x1(3), x2(3), worst
        integer :: i

        call build_rbf_formula(formula, 0.8_dp, status)
        builtin = make_rbf_kernel(3, 1.7_dp, 0.8_dp, status)
        user = make_user_kernel(3, 1.7_dp, formula, status)
        worst = 0.0_dp
        do i = 1, 12
            x1 = [0.2_dp*real(i, dp), -0.1_dp*real(i, dp), 0.05_dp]
            x2 = [0.3_dp, 0.4_dp*cos(real(i, dp)), -0.2_dp*real(i, dp)]
            worst = max(worst, abs(user%value(x1, x2) - builtin%value(x1, x2)))
        end do
        if (.not. status_ok(status) .or. worst > 1.0e-14_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [formula] user RBF disagrees with the built-in leaf ", worst
            failures = failures + 1
        end if
    end subroutine test_rbf_formula_matches_builtin

    subroutine test_operator_matches_direct_pairwise(failures)
        integer, intent(inout) :: failures
        type(kernel_formula_t) :: formula
        type(kernel_t) :: kernel
        type(kernel_operator_t) :: operator
        type(fortnum_status_t) :: status
        integer, parameter :: n = 24, d = 3
        real(dp) :: points(n, d), input(n), output(n), expected(n)
        real(dp) :: matrix_input(n, 2), matrix_output(n, 2)
        real(dp) :: shift, worst
        integer :: i, j

        shift = 0.05_dp
        call build_points(points)
        do i = 1, n
            input(i) = sin(0.7_dp*real(i, dp))
        end do
        matrix_input(:, 1) = input
        matrix_input(:, 2) = cos(0.4_dp*[(real(i, dp), i=1, n)])

        call build_damped_formula(formula, status)
        kernel = make_user_kernel(d, 1.3_dp, formula, status)
        call operator%initialize(points, kernel, shift, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [formula] the operator refused a validated formula"
            failures = failures + 1
            return
        end if
        call operator%matvec(input, output)

        do i = 1, n
            expected(i) = shift*input(i)
            do j = 1, n
                expected(i) = expected(i) &
                    + damped_reference(points(i, :), points(j, :), 1.3_dp)*input(j)
            end do
        end do
        worst = maxval(abs(output - expected))
        if (worst > 1.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [formula] lowered matvec disagrees with the direct sum ", worst
            failures = failures + 1
        end if

        call operator%matmat(matrix_input, matrix_output)
        if (maxval(abs(matrix_output(:, 1) - expected)) > 1.0e-11_dp) then
            write (error_unit, '(a)') &
                "FAIL [formula] lowered matmat disagrees with the vector product"
            failures = failures + 1
        end if
    end subroutine test_operator_matches_direct_pairwise

    subroutine test_composite_with_user_leaf(failures)
        !! A user leaf must compose with built-in kernels in the same postfix
        !! program.
        integer, intent(inout) :: failures
        type(kernel_formula_t) :: formula
        type(kernel_t) :: user, constant, composite
        type(kernel_operator_t) :: operator
        type(fortnum_status_t) :: status
        integer, parameter :: n = 16, d = 2
        real(dp) :: points(n, d), input(n), output(n), expected(n)
        real(dp) :: worst
        integer :: i, j

        call build_points(points)
        input = 1.0_dp
        call build_damped_formula(formula, status)
        user = make_user_kernel(d, 1.3_dp, formula, status)
        constant = make_constant_kernel(d, 0.4_dp, status)
        composite = kernel_add(user, constant, status)
        call operator%initialize(points, composite, 0.0_dp, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [formula] the operator refused a composite user kernel"
            failures = failures + 1
            return
        end if
        call operator%matvec(input, output)

        do i = 1, n
            expected(i) = 0.0_dp
            do j = 1, n
                expected(i) = expected(i) &
                    + damped_reference(points(i, :), points(j, :), 1.3_dp) + 0.4_dp
            end do
        end do
        worst = maxval(abs(output - expected))
        if (worst > 1.0e-11_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [formula] composite user kernel product ", worst
            failures = failures + 1
        end if
    end subroutine test_composite_with_user_leaf

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(kernel_formula_t) :: formula
        type(kernel_t) :: kernel
        type(kernel_operator_t) :: operator
        type(fortnum_status_t) :: status
        real(dp) :: points(4, 2)

        call build_points(points)

        call formula%reset()
        call formula%validate(status)
        call check_refused(status, formula, "empty program", failures)

        call formula%reset()
        call formula%push_squared_distance()
        call formula%add()
        call formula%validate(status)
        call check_refused(status, formula, "stack underflow", failures)

        call formula%reset()
        call formula%push_squared_distance()
        call formula%push_distance()
        call formula%validate(status)
        call check_refused(status, formula, "two values left", failures)

        call formula%reset()
        call formula%push_squared_distance()
        call formula%divide_by_constant(0.0_dp)
        call formula%validate(status)
        call check_refused(status, formula, "division by zero", failures)

        ! A formula that validated but was then extended must be refused until
        ! it validates again.
        call build_rbf_formula(formula, 0.8_dp, status)
        call formula%push_distance()
        kernel = make_user_kernel(2, 1.0_dp, formula, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [refuse] an edited formula was accepted without revalidation"
            failures = failures + 1
        end if

        ! The operator refuses the same way when handed an unlowerable leaf.
        call build_rbf_formula(formula, 0.8_dp, status)
        kernel = make_user_kernel(2, 1.0_dp, formula, status)
        deallocate (kernel%formula)
        call operator%initialize(points, kernel, 0.0_dp, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') &
                "FAIL [refuse] the operator lowered a formula-less user leaf"
            failures = failures + 1
        end if
    end subroutine test_refusals

    subroutine check_refused(status, formula, label, failures)
        type(fortnum_status_t), intent(in) :: status
        type(kernel_formula_t), intent(in) :: formula
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures

        if (status_ok(status) .or. formula%static_lowering_eligible()) then
            write (error_unit, '(a)') "FAIL [refuse] accepted: "//label
            failures = failures + 1
        end if
    end subroutine check_refused

end program test_kernel_formula
