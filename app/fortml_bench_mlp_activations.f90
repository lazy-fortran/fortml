program fortml_bench_mlp_activations
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_mlp, only: mlp_t, MLP_LINEAR, MLP_TANH, MLP_RELU, MLP_GELU, &
        MLP_SILU, MLP_ELU, MLP_SOFTPLUS, MLP_LEAKY_RELU, MLP_SIGMOID, MLP_MISH
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 2048
    integer, parameter :: n_features = 8
    integer, parameter :: n_hidden = 32
    integer, parameter :: n_outputs = 4
    integer, parameter :: repetitions = 32
    integer, parameter :: n_parameters = n_features*n_hidden + n_hidden + &
        n_hidden*n_outputs + n_outputs
    integer, parameter :: kinds(10) = [MLP_LINEAR, MLP_TANH, MLP_RELU, MLP_GELU, &
        MLP_SILU, MLP_ELU, MLP_SOFTPLUS, MLP_LEAKY_RELU, MLP_SIGMOID, MLP_MISH]
    character(len=16), parameter :: names(10) = [character(len=16) :: &
        "linear", "tanh", "relu", "gelu", "silu", "elu", "softplus", &
        "leaky_relu", "sigmoid", "mish"]
    real(dp) :: x(n_samples, n_features), y(n_samples, n_outputs)
    real(dp) :: theta(n_parameters), checksum, elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    integer :: i, j, k
    type(mlp_t) :: model
    type(fortnum_status_t) :: status

    do j = 1, n_features
        do i = 1, n_samples
            x(i, j) = sin(0.017_dp*real(i, dp) + 0.13_dp*real(j, dp)) + &
                0.15_dp*cos(0.009_dp*real(i*j, dp))
        end do
    end do
    do i = 1, n_parameters
        theta(i) = 0.07_dp*sin(0.37_dp*real(i, dp))
    end do

    write (*, '(a)') "activation,n_samples,n_features,n_hidden,n_outputs,repetitions," // &
        "checksum,seconds_per_operation"
    do k = 1, size(kinds)
        call model%initialize([n_features, n_hidden, n_outputs], status, &
            hidden_activation=kinds(k), initialization_seed=19)
        call model%set_parameters(theta, status)
        call model%predict(x, y, status)
        if (.not. status_ok(status)) error stop "MLP activation benchmark setup failed"
        checksum = sum(y)
        call system_clock(clock_start, clock_rate)
        do j = 1, repetitions
            call model%predict(x, y, status)
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
        if (.not. status_ok(status)) error stop "MLP activation benchmark failed"
        write (*, '(a,a,i0,a,i0,a,i0,a,i0,a,i0,a,es24.16,a,es24.16)') trim(names(k)), &
            ",", n_samples, ",", n_features, ",", n_hidden, ",", n_outputs, &
            ",", repetitions, ",", checksum, ",", elapsed/real(repetitions, dp)
    end do
end program fortml_bench_mlp_activations
