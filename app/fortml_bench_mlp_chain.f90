program fortml_bench_mlp_chain
    !! Release workload for the composable sequential MLP module tree.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_chain, only: mlp_chain_t
    implicit none

    integer, parameter :: n_samples = 64, n_features = 2, n_outputs = 1
    integer, parameter :: n_parameters = 17, repetitions = 2048
    type(mlp_t) :: encoder, head
    type(mlp_chain_t) :: chain
    type(fortnum_status_t) :: status
    real(dp) :: x(n_samples, n_features), dx(n_samples, n_features)
    real(dp) :: y(n_samples, n_outputs), dy(n_samples, n_outputs)
    real(dp) :: u(n_samples, n_outputs), x_bar(n_samples, n_features)
    real(dp) :: theta(n_parameters), dtheta(n_parameters)
    real(dp) :: theta_bar(n_parameters), theta_hvp(n_parameters)
    real(dp) :: x_hvp(n_samples, n_features), elapsed, sink
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, repetition, oracle_unit, environment_status
    character(len=1024) :: oracle_path

    call encoder%initialize([n_features, 4], status, output_activation=MLP_LINEAR)
    call head%initialize([4, n_outputs], status, output_activation=MLP_LINEAR)
    theta = [ &
        0.30_dp, -0.20_dp, 0.10_dp, 0.40_dp, -0.50_dp, 0.60_dp, 0.20_dp, -0.10_dp, &
        0.05_dp, -0.03_dp, 0.07_dp, -0.09_dp, 0.11_dp, -0.13_dp, 0.17_dp, &
        -0.19_dp, 0.23_dp]
    call encoder%set_parameters(theta(:12), status)
    call head%set_parameters(theta(13:), status)
    call chain%initialize(n_features, status)
    call chain%append(encoder, status, name="encoder")
    call chain%append(head, status, name="head")
    if (.not. status_ok(status) .or. chain%parameter_count() /= n_parameters) then
        error stop "MLP chain benchmark: invalid fixture"
    end if
    do i = 1, n_samples
        x(i, :) = [sin(0.17_dp*real(i, dp)), cos(0.11_dp*real(i, dp))]
        dx(i, :) = [0.03_dp*cos(0.17_dp*real(i, dp)), -0.02_dp*sin(0.11_dp*real(i, dp))]
        u(i, 1) = 0.2_dp + 0.01_dp*real(mod(i, 7), dp)
    end do
    dtheta = 0.01_dp*sin([(real(j, dp), j=1,n_parameters)])

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_MLP_CHAIN_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", action="write")
        write (oracle_unit, '(a)') "quantity,index,value"
        call chain%predict(x, y, status)
        call chain%jvp(x, dtheta, dx, y, dy, status)
        call chain%vjp(x, u, theta_bar, x_bar, status)
        call chain%hvp(x, u, dtheta, dx, theta_hvp, x_hvp, status)
        if (.not. status_ok(status)) error stop "MLP chain benchmark: oracle failed"
        do i = 1, n_samples
            write (oracle_unit, '(a,i0,a,es26.17e3)') "prediction,", i, ",", y(i, 1)
            write (oracle_unit, '(a,i0,a,es26.17e3)') "jvp,", i, ",", dy(i, 1)
            write (oracle_unit, '(a,i0,a,es26.17e3)') "x_vjp,", i, ",", x_bar(i, 1)
            write (oracle_unit, '(a,i0,a,es26.17e3)') "x_hvp,", i, ",", x_hvp(i, 1)
        end do
        do i = 1, n_parameters
            write (oracle_unit, '(a,i0,a,es26.17e3)') "parameter_vjp,", i, ",", theta_bar(i)
            write (oracle_unit, '(a,i0,a,es26.17e3)') "parameter_hvp,", i, ",", theta_hvp(i)
        end do
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    sink = 0.0_dp
    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call chain%predict(x, y, status)
        sink = sink + y(1, 1)
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_chain_predict,", elapsed

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call chain%jvp(x, dtheta, dx, y, dy, status)
        sink = sink + dy(1, 1)
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_chain_jvp,", elapsed

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call chain%vjp(x, u, theta_bar, x_bar, status)
        sink = sink + theta_bar(1)
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_chain_vjp,", elapsed

    call system_clock(clock_start, clock_rate)
    do repetition = 1, repetitions
        call chain%hvp(x, u, dtheta, dx, theta_hvp, x_hvp, status)
        sink = sink + theta_hvp(1)
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end-clock_start, dp)/real(clock_rate, dp)/real(repetitions, dp)
    write (*, '(a,es24.16)') "mlp_chain_hvp,", elapsed
    if (.not. ieee_is_finite(sink)) error stop "MLP chain benchmark: nonfinite sink"

contains

    logical function oracle_only_requested()
        character(len=16) :: environment_value
        integer :: environment_code

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", environment_value, &
            status=environment_code)
        oracle_only_requested = environment_code == 0 .and. trim(environment_value) == "1"
    end function oracle_only_requested

end program fortml_bench_mlp_chain
