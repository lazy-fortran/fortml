program test_fortad_kernel_products
    !! Generate scalar RBF parameter products with FortAD and check them
    !! against an independent hand derivative and finite differences.
    use fortad, only: fad_hvp, fad_jvp, fad_result_t, fad_vjp
    implicit none

    character(len=1), parameter :: nl = achar(10)
    character(len=*), parameter :: SOURCE = &
        "function rbf_scalar(x1, x2, lv, ll) result(y)"//nl// &
        "    real(8), intent(in) :: x1, x2, lv, ll"//nl// &
        "    real(8) :: y"//nl// &
        "    y = exp(lv - 0.5d0*(x1-x2)**2*exp(-2.0d0*ll))"//nl// &
        "end function rbf_scalar"//nl
    integer :: failures

    failures = 0
    call check_generated_products(failures)
    if (failures == 0) then
        write (*, '(a)') "PASS FortAD-generated RBF parameter products"
    else
        write (*, '(a,i0)') "FAIL FortAD-generated RBF cases: ", failures
        error stop 1
    end if

contains

    subroutine check_generated_products(failures)
        integer, intent(inout) :: failures
        type(fad_result_t) :: jvp, vjp, hvp
        character(len=:), allocatable :: directory
        integer :: stat, unit

        directory = "build/oracle_fortad_kernel"
        call execute_command_line("mkdir -p "//directory, exitstat=stat)
        jvp = fad_jvp(SOURCE, [character(len=2) :: "x1", "x2", "lv", "ll"], &
            name="rbf_jvp")
        vjp = fad_vjp(SOURCE, [character(len=2) :: "x1", "x2", "lv", "ll"], &
            name="rbf_vjp")
        hvp = fad_hvp(SOURCE, [character(len=2) :: "x1", "x2", "lv", "ll"], &
            name="rbf_hvp")
        if (.not. (jvp%ok .and. vjp%ok .and. hvp%ok)) then
            write (*, '(a)') "  FAIL FortAD RBF generation"
            if (.not. jvp%ok) write (*, '(a)') "    JVP: "//jvp%message
            if (.not. vjp%ok) write (*, '(a)') "    VJP: "//vjp%message
            if (.not. hvp%ok) write (*, '(a)') "    HVP: "//hvp%message
            failures = failures + 1
            return
        end if

        open (newunit=unit, file=directory//"/derivs.f90", status="replace", &
            action="write")
        write (unit, '(a)') "module fad_generated"
        write (unit, '(a)') "    implicit none"
        write (unit, '(a)') "contains"
        write (unit, '(a)') jvp%code
        write (unit, '(a)') vjp%code
        write (unit, '(a)') hvp%code
        write (unit, '(a)') "end module fad_generated"
        close (unit)

        open (newunit=unit, file=directory//"/driver.f90", status="replace", &
            action="write")
        write (unit, '(a)') driver_text()
        close (unit)

        call execute_command_line( &
            "cd "//directory//" && gfortran -O2 -o run derivs.f90 driver.f90 "// &
            "> build.log 2>&1", exitstat=stat)
        if (stat /= 0) then
            write (*, '(a)') "  FAIL generated RBF code did not compile"
            call show_file(directory//"/build.log")
            failures = failures + 1
            return
        end if
        call execute_command_line("cd "//directory//" && ./run > out.txt 2>&1", &
            exitstat=stat)
        if (stat /= 0) then
            write (*, '(a)') "  FAIL generated RBF independent oracle"
            call show_file(directory//"/out.txt")
            failures = failures + 1
        end if
    end subroutine check_generated_products

    function driver_text() result(text)
        character(len=:), allocatable :: text

        text = &
            "program driver"//nl// &
            "    use fad_generated"//nl// &
            "    implicit none"//nl// &
            "    real(8) :: q(4), v(4), w(4), qp(4), qm(4)"//nl// &
            "    real(8) :: y, yd, yb, h, yd_fd, lhs, rhs"//nl// &
            "    real(8) :: x1_b, x2_b, lv_b, ll_b"//nl// &
            "    real(8) :: x1_b_d, x2_b_d, lv_b_d, ll_b_d"//nl// &
            "    real(8) :: g(4), gp(4), gm(4), hv(4), hw(4)"//nl// &
            "    logical :: bad"//nl// &
            "    q = [0.4d0, -0.7d0, 0.2d0, -0.3d0]"//nl// &
            "    v = [0.3d0, -0.2d0, 0.5d0, -0.4d0]"//nl// &
            "    w = [-0.6d0, 0.7d0, -0.1d0, 0.8d0]"//nl// &
            "    yb = 1.0d0"//nl// &
            "    bad = .false."//nl// &
            "    call rbf_jvp(q(1), v(1), q(2), v(2), q(3), v(3), q(4), v(4), &"//nl// &
            "        y, yd)"//nl// &
            "    h = 1.0d-5"//nl// &
            "    yd_fd = (rbf_value(q+h*v)-rbf_value(q-h*v))/(2.0d0*h)"//nl// &
            "    if (abs(yd-yd_fd) > 2.0d-9) bad = .true."//nl// &
            "    call rbf_vjp(q(1), q(2), q(3), q(4), y, yb, x1_b, x2_b, lv_b, ll_b)"//nl// &
            "    call rbf_gradient(q, g)"//nl// &
            "    if (maxval(abs([x1_b,x2_b,lv_b,ll_b]-g)) > 2.0d-10) bad = .true."//nl// &
            "    lhs = dot_product(g, v)"//nl// &
            "    rhs = yd"//nl// &
            "    if (abs(lhs-rhs) > 2.0d-12) bad = .true."//nl// &
            "    call rbf_hvp(q(1), v(1), q(2), v(2), q(3), v(3), q(4), v(4), &"//nl// &
            "        y, yd, yb, x1_b, x1_b_d, x2_b, x2_b_d, lv_b, lv_b_d, ll_b, ll_b_d)"//nl// &
            "    qp = q+h*v"//nl// &
            "    qm = q-h*v"//nl// &
            "    call rbf_gradient(qp, gp)"//nl// &
            "    call rbf_gradient(qm, gm)"//nl// &
            "    hv = (gp-gm)/(2.0d0*h)"//nl// &
            "    if (maxval(abs(hv-[x1_b_d,x2_b_d,lv_b_d,ll_b_d])) > 3.0d-8) bad = .true."//nl// &
            "    call rbf_hvp(q(1), w(1), q(2), w(2), q(3), w(3), q(4), w(4), &"//nl// &
            "        y, yd, yb, x1_b, x1_b_d, x2_b, x2_b_d, lv_b, lv_b_d, ll_b, ll_b_d)"//nl// &
            "    hw = [x1_b_d,x2_b_d,lv_b_d,ll_b_d]"//nl// &
            "    if (abs(dot_product(w,hv)-dot_product(v,hw)) > 3.0d-9) bad = .true."//nl// &
            "    if (bad) error stop 1"//nl// &
            "contains"//nl// &
            "    function rbf_value(q) result(y)"//nl// &
            "        real(8), intent(in) :: q(:)"//nl// &
            "        real(8) :: y"//nl// &
            "        y = exp(q(3)-0.5d0*(q(1)-q(2))**2*exp(-2.0d0*q(4)))"//nl// &
            "    end function rbf_value"//nl// &
            "    subroutine rbf_gradient(q, g)"//nl// &
            "        real(8), intent(in) :: q(:)"//nl// &
            "        real(8), intent(out) :: g(:)"//nl// &
            "        real(8) :: d, e, y"//nl// &
            "        d = q(1)-q(2)"//nl// &
            "        e = exp(-2.0d0*q(4))"//nl// &
            "        y = rbf_value(q)"//nl// &
            "        g = [-d*e*y, d*e*y, y, d*d*e*y]"//nl// &
            "    end subroutine rbf_gradient"//nl// &
            "end program driver"//nl
    end function driver_text

    subroutine show_file(path)
        character(*), intent(in) :: path
        integer :: unit, ios
        character(len=512) :: line

        open (newunit=unit, file=path, status="old", action="read", iostat=ios)
        if (ios /= 0) return
        do
            read (unit, '(a)', iostat=ios) line
            if (ios /= 0) exit
            write (*, '(a)') "    "//trim(line)
        end do
        close (unit)
    end subroutine show_file

end program test_fortad_kernel_products
