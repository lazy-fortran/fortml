program fortml_bench_power_transformer
    !! Release workload for deterministic Yeo--Johnson and Box--Cox maps.
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_preprocessing, only: power_transformer_t, &
        POWER_METHOD_YEO_JOHNSON, POWER_METHOD_BOX_COX
    implicit none

    type(power_transformer_t) :: yeo, box
    type(fortnum_status_t) :: status
    real(dp) :: yj(256, 1), yj_out(256, 1), yj_back(256, 1)
    real(dp) :: bc(256, 1), bc_out(256, 1), bc_back(256, 1)
    real(dp) :: lambda(1), yj_checksum, bc_checksum
    real(dp) :: yj_error, bc_error, elapsed
    real(dp), allocatable :: yeo_lambdas(:), box_lambdas(:)
    integer(int64) :: tick_start, tick_end, ticks_per_second
    integer :: i

    do i = 1, 256
        yj(i, 1) = -1.25_dp + 2.5_dp*real(i - 1, dp)/255.0_dp
        bc(i, 1) = 0.25_dp + 4.0_dp*real(i - 1, dp)/255.0_dp
    end do
    lambda = 0.0_dp
    call yeo%fit(yj, status, method=POWER_METHOD_YEO_JOHNSON, &
        standardize=.false., lambdas=lambda)
    if (.not. status_ok(status)) error stop "Yeo--Johnson benchmark fit failed"
    lambda = 0.5_dp
    call box%fit(bc, status, method=POWER_METHOD_BOX_COX, &
        standardize=.false., lambdas=lambda)
    if (.not. status_ok(status)) error stop "Box--Cox benchmark fit failed"

    call system_clock(tick_start, ticks_per_second)
    call yeo%transform(yj, yj_out, status)
    call yeo%inverse_transform(yj_out, yj_back, status)
    call box%transform(bc, bc_out, status)
    call box%inverse_transform(bc_out, bc_back, status)
    call system_clock(tick_end)
    if (.not. status_ok(status)) error stop "power benchmark product failed"
    elapsed = real(tick_end - tick_start, dp)/real(ticks_per_second, dp)
    yj_checksum = sum(yj_out)
    bc_checksum = sum(bc_out)
    yj_error = maxval(abs(yj_back - yj))
    bc_error = maxval(abs(bc_back - bc))
    yeo_lambdas = yeo%lambdas()
    box_lambdas = box%lambdas()
    write (*, '(a,",pass,checksum,",es24.16,",",es24.16,",lambda,",es24.16)') &
        "power_transformer,yeo_johnson", yj_checksum, elapsed, yeo_lambdas(1)
    write (*, '(a,",pass,checksum,",es24.16,",",es24.16,",lambda,",es24.16)') &
        "power_transformer,box_cox", bc_checksum, elapsed, box_lambdas(1)
    write (*, '(a,",pass,inverse_max_abs_error,",es24.16,",",es24.16)') &
        "power_transformer,yeo_johnson", yj_error, elapsed
    write (*, '(a,",pass,inverse_max_abs_error,",es24.16,",",es24.16)') &
        "power_transformer,box_cox", bc_error, elapsed
    write (*, '(a,",unavailable,typed_refusal,",i0)') &
        "power_transformer,cuda", 3
end program fortml_bench_power_transformer
