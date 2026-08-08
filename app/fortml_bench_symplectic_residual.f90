program fortml_bench_symplectic_residual
    !! Release executable for the canonical symplectic-form diagnostic.
    use, intrinsic :: iso_fortran_env, only: output_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_symplectic, only: symplectic_form_diagnostic_t
    implicit none

    type(symplectic_form_diagnostic_t) :: diagnostic
    type(fortnum_status_t) :: status
    real(dp) :: jacobian(2, 2), residual(4), value
    real(dp), parameter :: step = 0.07_dp
    logical :: structure_ok

    call diagnostic%initialize(1, status)
    if (.not. status_ok(status)) error stop 1
    jacobian = reshape([1.0_dp - 0.5_dp*step**2, -step + 0.25_dp*step**3, &
        step, 1.0_dp - 0.5_dp*step**2], [2, 2])
    call diagnostic%residual(jacobian, residual, status)
    if (.not. status_ok(status)) error stop 1
    call diagnostic%value(jacobian, value, status)
    if (.not. status_ok(status)) error stop 1
    call diagnostic%is_symplectic(jacobian, 1.0e-13_dp, structure_ok, status)
    if (.not. status_ok(status)) error stop 1
    write (output_unit, '(a)') &
        "workload,step_size,defect_max,value,is_symplectic,device,status"
    write (output_unit, '(a,",",es24.16,",",es24.16,",",l1,",",a)') &
        "verlet", step, maxval(abs(residual)), value, structure_ok, "cpu,pass"
end program fortml_bench_symplectic_residual
