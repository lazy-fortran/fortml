program test_derivative_gp_fit_products
    !! Independent through-fit oracle for mixed value/first-derivative GPs.
    !! The production JVP/VJP is differentiated with respect to y_train;
    !! independent refits provide the finite-difference behavior oracle.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    implicit none

    integer, parameter :: n_train = 4, n_outputs = 2, n_query = 3
    real(dp), parameter :: noise = 0.08_dp, jitter = 1.0e-10_dp
    real(dp), parameter :: finite_difference_step = 2.0e-5_dp
    type(gp_derivative_regression_t) :: model
    type(kernel_t) :: kernel
    type(fortml_device_t) :: cpu, cuda, unselected
    type(fortnum_status_t) :: status
    real(dp) :: x(n_train, 1), y(n_train, n_outputs), y_direction(n_train, n_outputs)
    real(dp) :: query(n_query, 1)
    real(dp) :: mean(n_query, n_outputs), mean_dot(n_query, n_outputs)
    real(dp) :: variance(n_query), variance_dot(n_query)
    real(dp) :: mean_reference(n_query, n_outputs), variance_reference(n_query)
    real(dp) :: mean_plus(n_query, n_outputs), mean_minus(n_query, n_outputs)
    real(dp) :: variance_plus(n_query), variance_minus(n_query)
    real(dp) :: mean_bar(n_query, n_outputs), variance_bar(n_query)
    real(dp) :: observation_bar(n_train, n_outputs)
    real(dp) :: mean_cpu(n_query, n_outputs), mean_dot_cpu(n_query, n_outputs)
    real(dp) :: variance_cpu(n_query), variance_dot_cpu(n_query)
    real(dp) :: observation_bar_cpu(n_train, n_outputs)
    real(dp) :: mean_cuda(n_query, n_outputs), mean_dot_cuda(n_query, n_outputs)
    real(dp) :: variance_cuda(n_query), variance_dot_cuda(n_query)
    real(dp) :: observation_bar_cuda(n_train, n_outputs)
    real(dp) :: finite_mean_dot(n_query, n_outputs), finite_variance_dot(n_query)
    real(dp) :: lhs, rhs, finite_scalar_dot, scalar_plus, scalar_minus
    integer :: failures

    failures = 0
    x(:, 1) = [-0.7_dp, -0.15_dp, 0.55_dp, 1.0_dp]
    y(:, 1) = [0.7_dp, -0.2_dp, 0.95_dp, 0.35_dp]
    y(:, 2) = [-0.4_dp, 0.8_dp, 0.1_dp, -0.65_dp]
    y_direction(:, 1) = [0.17_dp, -0.11_dp, 0.04_dp, -0.09_dp]
    y_direction(:, 2) = [-0.08_dp, 0.13_dp, -0.07_dp, 0.16_dp]
    query(:, 1) = [0.1_dp, 0.72_dp, -0.4_dp]
    mean_bar = reshape([0.21_dp, -0.14_dp, 0.09_dp, 0.17_dp, -0.13_dp, 0.08_dp], &
        shape(mean_bar))
    variance_bar = [-0.12_dp, 0.07_dp, 0.19_dp]

    kernel = make_rbf_kernel(1, 1.4_dp, 0.75_dp, status)
    call check(status_ok(status), "RBF construction", failures)
    call model%fit(x, [0, 1, 0, 1], y, kernel, noise, status, jitter=jitter)
    call check(status_ok(status), "mixed derivative-GP fit", failures)
    if (.not. status_ok(status)) error stop 1

    call model%predict(query, [1, 0, 1], mean_reference, variance_reference, status)
    call check(status_ok(status), "reference prediction", failures)
    call model%predict_observation_jvp(query, [1, 0, 1], y_direction, mean, mean_dot, &
        variance, variance_dot, status)
    call check(status_ok(status), "observation JVP", failures)
    call check(maxval(abs(mean - mean_reference)) < 3.0e-13_dp .and. &
        maxval(abs(variance - variance_reference)) < 3.0e-13_dp, &
        "observation JVP returns the primal prediction", failures)
    call check(maxval(abs(variance_dot)) < 3.0e-14_dp, &
        "observation JVP variance tangent is zero", failures)

    call fit_predict(y + finite_difference_step*y_direction, mean_plus, variance_plus, &
        status)
    call check(status_ok(status), "positive observation refit", failures)
    call fit_predict(y - finite_difference_step*y_direction, mean_minus, variance_minus, &
        status)
    call check(status_ok(status), "negative observation refit", failures)
    finite_mean_dot = (mean_plus - mean_minus)/(2.0_dp*finite_difference_step)
    finite_variance_dot = (variance_plus - variance_minus)/(2.0_dp* &
        finite_difference_step)
    call check(maxval(abs(mean_dot - finite_mean_dot)) < 3.0e-9_dp, &
        "observation JVP independent refit oracle", failures)
    call check(maxval(abs(variance_dot - finite_variance_dot)) < 3.0e-11_dp, &
        "observation JVP variance refit oracle", failures)

    call model%predict_observation_vjp(query, [1, 0, 1], mean_bar, variance_bar, &
        observation_bar, status)
    call check(status_ok(status), "observation VJP", failures)
    lhs = sum(mean_bar*mean_dot) + sum(variance_bar*variance_dot)
    rhs = sum(observation_bar*y_direction)
    call check(abs(lhs - rhs) < 3.0e-12_dp, &
        "observation JVP/VJP adjoint identity", failures)
    scalar_plus = sum(mean_bar*mean_plus) + sum(variance_bar*variance_plus)
    scalar_minus = sum(mean_bar*mean_minus) + sum(variance_bar*variance_minus)
    finite_scalar_dot = (scalar_plus - scalar_minus)/(2.0_dp*finite_difference_step)
    call check(abs(finite_scalar_dot - rhs) < 3.0e-9_dp, &
        "observation VJP independent scalar refit oracle", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call check(status_ok(status), "CPU device selection", failures)
    call model%predict_observation_jvp_device(cpu, query, [1, 0, 1], y_direction, &
        mean_cpu, mean_dot_cpu, variance_cpu, variance_dot_cpu, status)
    call check(status_ok(status) .and. maxval(abs(mean_cpu - mean)) < 3.0e-13_dp .and. &
        maxval(abs(mean_dot_cpu - mean_dot)) < 3.0e-13_dp .and. &
        maxval(abs(variance_cpu - variance)) < 3.0e-13_dp .and. &
        maxval(abs(variance_dot_cpu - variance_dot)) < 3.0e-13_dp, &
        "CPU observation JVP dispatch", failures)
    call model%predict_observation_vjp_device(cpu, query, [1, 0, 1], mean_bar, &
        variance_bar, &
        observation_bar_cpu, status)
    call check(status_ok(status) .and. maxval(abs(observation_bar_cpu - observation_bar)) < &
        3.0e-13_dp, &
        "CPU observation VJP dispatch", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    mean_cuda = 1234.0_dp
    mean_dot_cuda = 2345.0_dp
    variance_cuda = 3456.0_dp
    variance_dot_cuda = 4567.0_dp
    call model%predict_observation_jvp_device(cuda, query, [1, 0, 1], y_direction, &
        mean_cuda, mean_dot_cuda, variance_cuda, variance_dot_cuda, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA observation JVP refusal", failures)
    call check(all(mean_cuda == 1234.0_dp) .and. all(mean_dot_cuda == 2345.0_dp) .and. &
        all(variance_cuda == 3456.0_dp) .and. all(variance_dot_cuda == 4567.0_dp), &
        "CUDA observation JVP refusal preserves outputs", failures)
    observation_bar_cuda = 5678.0_dp
    call model%predict_observation_vjp_device(cuda, query, [1, 0, 1], mean_bar, &
        variance_bar, &
        observation_bar_cuda, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA observation VJP refusal", failures)
    call check(all(observation_bar_cuda == 5678.0_dp), &
        "CUDA observation VJP refusal preserves output", failures)
    call model%predict_observation_jvp_device(unselected, query, [1, 0, 1], &
        y_direction, &
        mean_cuda, mean_dot_cuda, variance_cuda, variance_dot_cuda, status)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "unselected observation-product device refusal", failures)
    call check(model%device_supported(FORTML_DEVICE_CPU) .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), &
        "observation-product device capability metadata", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL derivative GP fit-product cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS derivative GP through-fit finite-difference oracle"

contains

    subroutine fit_predict(observations, prediction_mean, prediction_variance, &
            local_status)
        real(dp), intent(in) :: observations(:, :)
        real(dp), intent(out) :: prediction_mean(:, :), prediction_variance(:)
        type(fortnum_status_t), intent(out) :: local_status
        type(gp_derivative_regression_t) :: local_model

        call local_model%fit(x, [0, 1, 0, 1], observations, kernel, noise, &
            local_status, &
            jitter=jitter)
        if (.not. status_ok(local_status)) return
        call local_model%predict(query, [1, 0, 1], prediction_mean, &
            prediction_variance, &
            local_status)
    end subroutine fit_predict

    subroutine check(condition, description, count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: count

        if (.not. condition) then
            count = count + 1
            write (error_unit, '(a)') &
                "  FAIL [derivative-gp-fit-products] "//description
        end if
    end subroutine check

end program test_derivative_gp_fit_products
