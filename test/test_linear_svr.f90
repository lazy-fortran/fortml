program test_linear_svr
    !! Independent behavior and derivative checks for weighted linear SVR.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortml_linear_svr, only: linear_svr_regression_t, SVR_LOSS_EPSILON, &
        SVR_LOSS_SQUARED_EPSILON
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_DOMAIN_ERROR
    implicit none

    type(linear_svr_regression_t) :: model, hinge_model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(real64) :: x(6, 2), query(4, 2), x_dot(4, 2)
    real(real64) :: targets(6), weights(6), prediction(6), query_prediction(4)
    real(real64) :: prediction_dot(4), prediction_plus(4), prediction_minus(4)
    real(real64) :: targets_bar(4), theta_dot(3), theta_bar(3), x_bar(4, 2)
    real(real64) :: value, value_plus, value_minus, l2_gradient, epsilon_gradient
    real(real64) :: gradient(3), gradient_fd(3), scratch_gradient(3)
    real(real64) :: parameters(3), parameters_plus(3)
    real(real64) :: parameters_minus(3)
    real(real64) :: prediction_oracle(6)
    integer :: failures, i
    real(real64), parameter :: step = 1.0e-6_real64

    failures = 0
    x(1, :) = [-2.0_real64, -1.0_real64]
    x(2, :) = [-1.5_real64, -0.5_real64]
    x(3, :) = [-1.0_real64,  0.5_real64]
    x(4, :) = [ 0.5_real64,  1.0_real64]
    x(5, :) = [ 1.5_real64,  0.25_real64]
    x(6, :) = [ 2.0_real64,  1.5_real64]
    targets = [1.10_real64, 1.85_real64, 2.30_real64, 3.75_real64, &
        5.05_real64, 6.30_real64]
    weights = [1.0_real64, 2.0_real64, 0.5_real64, 1.5_real64, 2.0_real64, 0.75_real64]

    call model%fit(x, targets, status, l2=0.05_real64, epsilon=0.08_real64, &
        loss=SVR_LOSS_SQUARED_EPSILON, sample_weight=weights, max_iterations=1500, &
        tolerance=1.0e-8_real64)
    call check(status_ok(status) .and. model%fitted(), &
        "weighted squared-epsilon fit", failures)
    if (.not. status_ok(status)) error stop 1
    call model%predict(x, prediction, status)
    parameters = model%parameters()
    prediction_oracle = x(:, 1)*parameters(1) + x(:, 2)*parameters(2) + parameters(3)
    call check(status_ok(status) .and. maxval(abs(prediction-prediction_oracle)) < 2.0e-12_real64, &
        "packed affine prediction", failures)
    call check(model%parameter_count() == 3 .and. model%feature_count() == 2 .and. &
        abs(model%epsilon()-0.08_real64) < 1.0e-14_real64, &
        "packed state and hyperparameters", failures)

    call model%objective_value_gradient(x, targets, parameters, value, gradient, status, &
        l2=0.05_real64, epsilon=0.13_real64, loss=SVR_LOSS_SQUARED_EPSILON, &
        sample_weight=weights, l2_gradient=l2_gradient, epsilon_gradient=epsilon_gradient)
    do i = 1, size(parameters)
        parameters_plus = parameters
        parameters_plus(i) = parameters_plus(i) + step
        call model%objective_value_gradient(x, targets, parameters_plus, value_plus, &
            scratch_gradient, status, l2=0.05_real64, epsilon=0.13_real64, &
            loss=SVR_LOSS_SQUARED_EPSILON, sample_weight=weights)
        parameters_minus = parameters
        parameters_minus(i) = parameters_minus(i) - step
        call model%objective_value_gradient(x, targets, parameters_minus, value_minus, &
            scratch_gradient, status, l2=0.05_real64, epsilon=0.13_real64, &
            loss=SVR_LOSS_SQUARED_EPSILON, sample_weight=weights)
        gradient_fd(i) = (value_plus-value_minus)/(2.0_real64*step)
    end do
    call check(status_ok(status) .and. maxval(abs(gradient-gradient_fd)) < 3.0e-6_real64, &
        "squared-epsilon objective gradient oracle", failures)
    call check(abs(l2_gradient-0.5_real64*sum(parameters(:2)**2)) < 2.0e-12_real64, &
        "L2 hyperparameter derivative", failures)
    call check(ieee_is_finite(epsilon_gradient), "epsilon hyperparameter derivative", failures)

    query(1, :) = [-1.25_real64, -0.75_real64]
    query(2, :) = [-0.25_real64, -0.25_real64]
    query(3, :) = [ 0.25_real64,  0.75_real64]
    query(4, :) = [ 1.25_real64,  0.50_real64]
    x_dot(1, :) = [ 0.10_real64, -0.20_real64]
    x_dot(2, :) = [-0.30_real64,  0.40_real64]
    x_dot(3, :) = [ 0.20_real64,  0.30_real64]
    x_dot(4, :) = [-0.15_real64,  0.05_real64]
    theta_dot = [0.07_real64, -0.11_real64, 0.13_real64]
    call model%predict_jvp(query, theta_dot, x_dot, query_prediction, prediction_dot, status)
    parameters_plus = parameters + step*theta_dot
    call model%set_parameters(parameters_plus, status)
    call model%predict(query+step*x_dot, prediction_plus, status)
    parameters_minus = parameters - step*theta_dot
    call model%set_parameters(parameters_minus, status)
    call model%predict(query-step*x_dot, prediction_minus, status)
    call model%set_parameters(parameters, status)
    call check(status_ok(status) .and. maxval(abs(prediction_dot - &
        (prediction_plus-prediction_minus)/(2.0_real64*step))) < 2.0e-8_real64, &
        "affine input/parameter JVP", failures)

    targets_bar = [0.2_real64, -0.1_real64, 0.4_real64, -0.3_real64]
    call model%predict_vjp(query, targets_bar, theta_bar, x_bar, status)
    call check(status_ok(status) .and. abs(sum(targets_bar*prediction_dot) - &
        (sum(theta_bar*theta_dot)+sum(x_bar*x_dot))) < 2.0e-10_real64, &
        "affine input/parameter VJP adjoint", failures)

    block
        real(real64) :: split_x(1, 1), split_targets(1), split_theta(2)
        real(real64) :: split_gradient(2), split_value
        split_x(1, 1) = 1.0_real64
        split_targets(1) = 0.0_real64
        split_theta = [0.10_real64, 0.0_real64]
        call model%objective_value_gradient(split_x, split_targets, split_theta, &
            split_value, split_gradient, status, l2=0.0_real64, epsilon=0.10_real64, &
            fit_intercept=.true., loss=SVR_LOSS_EPSILON)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "ordinary-epsilon exact kink refusal", failures)
    end block

    call hinge_model%fit(x, targets, status, l2=0.05_real64, epsilon=0.08_real64, &
        loss=SVR_LOSS_EPSILON, sample_weight=weights, max_iterations=2000, &
        tolerance=1.0e-7_real64)
    call check(status_ok(status), "ordinary-epsilon fit", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_device(cuda, query, query_prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA prediction refusal", failures)
    call check(model%device_supported(FORTML_DEVICE_CUDA) .eqv. .false., &
        "CUDA capability refusal", failures)

    call model%fit(x, targets, status, sample_weight=0.0_real64*weights)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "zero-weight refusal", failures)
    call model%fit(x, targets, status, epsilon=-1.0_real64)
    call check(status%code == FORTNUM_DOMAIN_ERROR, "negative-epsilon refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL linear SVR cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS linear SVR independent behavioral oracles"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [linear-svr] "//description
        end if
    end subroutine check

end program test_linear_svr
