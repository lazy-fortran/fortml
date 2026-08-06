program test_fortad_mlp_products
    !! Compare the explicit MLP product contract with FortAD-generated code.
    !!
    !! The generated routines are compiled as a separate program. Their
    !! values and gradients are checked against a hand-derived scalar MLP,
    !! while the HVP is checked against central differences of that gradient
    !! and against Hessian symmetry.
    use fortad, only: fad_hvp, fad_result_t, fad_vjp
    implicit none

    character(len=1), parameter :: nl = achar(10)
    character(len=*), parameter :: SOURCE = &
        "function mlp_scalar(x1, x2, w1, w2, b1, w3, b2) result(y)"//nl// &
        "    real(8), intent(in) :: x1, x2, w1, w2, b1, w3, b2"//nl// &
        "    real(8) :: h, y"//nl// &
        "    h = tanh(x1*w1 + x2*w2 + b1)"//nl// &
        "    y = h*w3 + b2"//nl// &
        "end function mlp_scalar"//nl
    integer :: failures

    failures = 0
    call check_generated_products(failures)

    if (failures == 0) then
        print *, "test_fortad_mlp_products: all cases passed"
    else
        print *, "test_fortad_mlp_products: ", failures, " case(s) FAILED"
        error stop 1
    end if

contains

    subroutine check_generated_products(failures)
        integer, intent(inout) :: failures
        type(fad_result_t) :: vjp, hvp
        character(len=:), allocatable :: dir
        integer :: stat, unit

        dir = "build/oracle_fortad_mlp"
        call execute_command_line("mkdir -p "//dir, exitstat=stat)

        vjp = fad_vjp(SOURCE, [character(len=2) :: "x1", "x2", "w1", "w2", &
            "b1", "w3", "b2"], name="mlp_vjp")
        hvp = fad_hvp(SOURCE, [character(len=2) :: "x1", "x2", "w1", "w2", &
            "b1", "w3", "b2"], name="mlp_hvp")
        if (.not. (vjp%ok .and. hvp%ok)) then
            print *, "FAIL fortad MLP generation"
            if (.not. vjp%ok) print *, "  vjp: ", vjp%message
            if (.not. hvp%ok) print *, "  hvp: ", hvp%message
            failures = failures + 1
            return
        end if

        open (newunit=unit, file=dir//"/derivs.f90", status="replace", &
            action="write")
        write (unit, '(a)') "module fad_generated"
        write (unit, '(a)') "    implicit none"
        write (unit, '(a)') "contains"
        write (unit, '(a)') vjp%code
        write (unit, '(a)') hvp%code
        write (unit, '(a)') "end module fad_generated"
        close (unit)

        open (newunit=unit, file=dir//"/driver.f90", status="replace", &
            action="write")
        write (unit, '(a)') driver_text()
        close (unit)

        call execute_command_line( &
            "cd "//dir//" && gfortran -O2 -o run derivs.f90 driver.f90 "// &
            "> build.log 2>&1", exitstat=stat)
        if (stat /= 0) then
            print *, "FAIL fortad MLP: generated code did not compile"
            call show_file(dir//"/build.log")
            failures = failures + 1
            return
        end if

        call execute_command_line("cd "//dir//" && ./run > out.txt 2>&1", &
            exitstat=stat)
        if (stat /= 0) then
            print *, "FAIL fortad MLP: generated product oracle mismatch"
            call show_file(dir//"/out.txt")
            failures = failures + 1
            return
        end if
        print *, "pass fortad_generated_mlp_value_vjp_hvp"
    end subroutine check_generated_products

    function driver_text() result(text)
        character(len=:), allocatable :: text

        text = &
            "program driver"//nl// &
            "    use fad_generated, only: mlp_vjp, mlp_hvp"//nl// &
            "    implicit none"//nl// &
            "    real(8) :: q(7), v(7), u(7), qp(7), qm(7)"//nl// &
            "    real(8) :: y, yd, yb, base, h, lhs, rhs"//nl// &
            "    real(8) :: g(7), hv(7), gu(7), hu(7), gfd(7), hfd(7)"//nl// &
            "    real(8) :: gp(7), gm(7), yd_fd"//nl// &
            "    logical :: bad"//nl// &
            "    q = [0.4d0, -0.7d0, 0.6d0, -0.2d0, 0.1d0, 1.3d0, -0.2d0]"//nl// &
            "    v = [0.3d0, -0.4d0, 0.7d0, 0.2d0, -0.5d0, 0.6d0, 0.8d0]"//nl// &
            "    u = [-0.2d0, 0.9d0, 0.1d0, -0.8d0, 0.4d0, 0.5d0, -0.3d0]"//nl// &
            "    bad = .false."//nl// &
            "    yb = 1.0d0"//nl// &
            "    call mlp_vjp(q(1), q(2), q(3), q(4), q(5), q(6), q(7), &"//nl// &
            "        y, yb, g(1), g(2), g(3), g(4), g(5), g(6), g(7))"//nl// &
            "    base = mlp_value(q)"//nl// &
            "    if (abs(y - base) > 1.0d-12) then"//nl// &
            "        print *, 'generated primal mismatch', y, base"//nl// &
            "        bad = .true."//nl// &
            "    end if"//nl// &
            "    call mlp_gradient(q, gfd)"//nl// &
            "    if (maxval(abs(g - gfd)) > 2.0d-7) then"//nl// &
            "        print *, 'VJP finite-difference mismatch', maxval(abs(g-gfd))"//nl// &
            "        bad = .true."//nl// &
            "    end if"//nl// &
            "    yb = 1.0d0"//nl// &
            "    call mlp_hvp(q(1), v(1), q(2), v(2), q(3), v(3), q(4), v(4), &"//nl// &
            "        q(5), v(5), q(6), v(6), q(7), v(7), y, yd, yb, &"//nl// &
            "        g(1), hv(1), g(2), hv(2), g(3), hv(3), g(4), hv(4), &"//nl// &
            "        g(5), hv(5), g(6), hv(6), g(7))"//nl// &
            "    hv(7) = 0.0d0"//nl// &
            "    if (maxval(abs(g - gfd)) > 2.0d-12) then"//nl// &
            "        print *, 'HVP gradient disagrees with VJP', maxval(abs(g-gfd))"//nl// &
            "        bad = .true."//nl// &
            "    end if"//nl// &
            "    h = 1.0d-5"//nl// &
            "    yd_fd = (mlp_value(q + h*v) - mlp_value(q - h*v))/(2.0d0*h)"//nl// &
            "    if (abs(yd - yd_fd) > 2.0d-9) then"//nl// &
            "        print *, 'JVP in HVP mismatch', yd, yd_fd"//nl// &
            "        bad = .true."//nl// &
            "    end if"//nl// &
            "    qp = q + h*v"//nl// &
            "    qm = q - h*v"//nl// &
            "    call mlp_gradient(qp, gp)"//nl// &
            "    call mlp_gradient(qm, gm)"//nl// &
            "    hfd = (gp - gm)/(2.0d0*h)"//nl// &
            "    if (maxval(abs(hv - hfd)) > 2.0d-8) then"//nl// &
            "        print *, 'HVP finite-difference mismatch', maxval(abs(hv-hfd))"//nl// &
            "        bad = .true."//nl// &
            "    end if"//nl// &
            "    call mlp_hvp(q(1), u(1), q(2), u(2), q(3), u(3), q(4), u(4), &"//nl// &
            "        q(5), u(5), q(6), u(6), q(7), u(7), y, yd, yb, &"//nl// &
            "        gu(1), hu(1), gu(2), hu(2), gu(3), hu(3), gu(4), hu(4), &"//nl// &
            "        gu(5), hu(5), gu(6), hu(6), gu(7))"//nl// &
            "    hu(7) = 0.0d0"//nl// &
            "    lhs = dot_product(u, hv)"//nl// &
            "    rhs = dot_product(v, hu)"//nl// &
            "    if (abs(lhs-rhs) > 2.0d-10) then"//nl// &
            "        print *, 'Hessian symmetry mismatch', lhs, rhs"//nl// &
            "        bad = .true."//nl// &
            "    end if"//nl// &
            "    if (bad) error stop 1"//nl// &
            "contains"//nl// &
            driver_helpers()// &
            "end program driver"//nl
    end function driver_text

    function driver_helpers() result(text)
        character(len=:), allocatable :: text

        text = &
            "    function mlp_value(q) result(y)"//nl// &
            "        real(8), intent(in) :: q(:)"//nl// &
            "        real(8) :: y, h"//nl// &
            "        h = tanh(q(1)*q(3) + q(2)*q(4) + q(5))"//nl// &
            "        y = h*q(6) + q(7)"//nl// &
            "    end function mlp_value"//nl// &
            "    subroutine mlp_gradient(q, g)"//nl// &
            "        real(8), intent(in) :: q(:)"//nl// &
            "        real(8), intent(out) :: g(:)"//nl// &
            "        real(8) :: h, s"//nl// &
            "        h = tanh(q(1)*q(3) + q(2)*q(4) + q(5))"//nl// &
            "        s = 1.0d0 - h*h"//nl// &
            "        g(1) = q(6)*s*q(3)"//nl// &
            "        g(2) = q(6)*s*q(4)"//nl// &
            "        g(3) = q(6)*s*q(1)"//nl// &
            "        g(4) = q(6)*s*q(2)"//nl// &
            "        g(5) = q(6)*s"//nl// &
            "        g(6) = h"//nl// &
            "        g(7) = 1.0d0"//nl// &
            "    end subroutine mlp_gradient"//nl
    end function driver_helpers

    subroutine show_file(path)
        character(len=*), intent(in) :: path
        integer :: unit, ios
        character(len=512) :: line

        open (newunit=unit, file=path, status="old", action="read", iostat=ios)
        if (ios /= 0) return
        do
            read (unit, '(a)', iostat=ios) line
            if (ios /= 0) exit
            print *, "    ", trim(line)
        end do
        close (unit)
    end subroutine show_file

end program test_fortad_mlp_products
