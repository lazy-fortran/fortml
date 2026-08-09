program test_lightgbm_multiclass_log_proba
    !! Independent behavioral oracle for LightGBM multiclass log-probability
    !! products and fixed-structure leaf-coordinate derivatives.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_lightgbm, only: lightgbm_options_t
    use fortml_lightgbm_multiclass, only: lightgbm_multiclass_t
    implicit none

    type(lightgbm_multiclass_t) :: model
    type(lightgbm_options_t) :: options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 1), query(3, 1), query_dot(3, 1)
    real(dp) :: probabilities(3, 3), log_probabilities(3, 3)
    real(dp) :: probability_dot(3, 3), log_probability_dot(3, 3)
    real(dp) :: log_plus(3, 3), log_minus(3, 3), log_bar(3, 3)
    real(dp) :: probability_bar(3, 3), x_bar(3, 1), sentinel(3, 3)
    real(dp), allocatable :: direction(:), parameter_bar(:), parameters(:)
    real(dp), allocatable :: parameter_probability_dot(:, :)
    real(dp), allocatable :: parameter_log_dot(:, :)
    real(dp) :: lhs, rhs, h
    integer :: labels(9), failures

    x(:, 1) = [-4.0_dp, -3.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, &
        2.0_dp, 3.0_dp, 4.0_dp]
    labels = [-8, -8, -8, 2, 2, 2, 11, 11, 11]
    query(:, 1) = [-2.3_dp, 0.17_dp, 2.4_dp]
    query_dot(:, 1) = [0.17_dp, -0.13_dp, 0.23_dp]
    log_bar = reshape([0.2_dp, -0.1_dp, 0.4_dp, -0.3_dp, 0.1_dp, -0.2_dp, &
        0.5_dp, -0.4_dp, 0.3_dp], shape(log_bar))
    probability_bar = reshape([0.2_dp, -0.1_dp, 0.4_dp, -0.3_dp, 0.1_dp, -0.2_dp, &
        0.5_dp, -0.4_dp, 0.3_dp], shape(probability_bar))
    failures = 0
    options%n_estimators = 4
    options%num_leaves = 2
    options%max_depth = 1
    options%min_data_in_leaf = 1
    options%max_bin = 16
    options%learning_rate = 0.4_dp
    options%l2 = 1.0_dp
    options%seed = 19
    call model%fit(x, labels, status, options)
    call check(status_ok(status) .and. model%fitted(), "fit", failures)
    call check(all(model%classes() == [-8, 2, 11]), "sorted arbitrary labels", failures)

    call model%predict_proba(query, probabilities, status)
    call model%predict_log_proba(query, log_probabilities, status)
    call check(status_ok(status), "probability and log-probability status", failures)
    call check(maxval(abs(exp(log_probabilities) - probabilities)) < 3.0e-14_dp, &
        "log/simplex round trip", failures)
    call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 3.0e-14_dp, &
        "probability simplex", failures)
    call independent_log_oracle(model, query, log_probabilities, failures)

    h = 1.0e-6_dp
    call model%predict_log_proba_jvp(query, query_dot, log_probabilities, &
        log_probability_dot, status)
    call model%predict_log_proba(query + h*query_dot, log_plus, status)
    call model%predict_log_proba(query - h*query_dot, log_minus, status)
    call check(status_ok(status) .and. maxval(abs(log_probability_dot - &
        (log_plus - log_minus)/(2.0_dp*h))) < 4.0e-8_dp, &
        "input log-probability JVP central difference", failures)
    call model%predict_log_proba_vjp(query, log_bar, x_bar, status)
    lhs = sum(x_bar*query_dot)
    rhs = sum(log_bar*log_probability_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 4.0e-12_dp, &
        "input log-probability VJP adjoint", failures)

    allocate(direction(model%parameter_count()), parameter_bar(model%parameter_count()), &
        parameters(model%parameter_count()), parameter_probability_dot(3, 3), &
        parameter_log_dot(3, 3))
    parameters = model%parameters(status)
    call check(status_ok(status) .and. size(parameters) == model%parameter_count() .and. &
        model%parameter_count() > 0, "packed leaf parameter metadata", failures)
    direction = 0.013_dp
    direction(1::2) = -0.021_dp
    call model%predict_proba_parameter_jvp(query, direction, probabilities, &
        parameter_probability_dot, status)
    call model%predict_proba_parameter_vjp(query, probability_bar, parameter_bar, status)
    lhs = dot_product(parameter_bar, direction)
    rhs = sum(probability_bar*parameter_probability_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-12_dp, &
        "probability parameter JVP/VJP adjoint", failures)
    call model%predict_log_proba_parameter_jvp(query, direction, log_probabilities, &
        parameter_log_dot, status)
    call model%predict_log_proba_parameter_vjp(query, log_bar, parameter_bar, status)
    lhs = dot_product(parameter_bar, direction)
    rhs = sum(log_bar*parameter_log_dot)
    call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-12_dp, &
        "log-probability parameter JVP/VJP adjoint", failures)

    cpu%kind = FORTML_DEVICE_CPU
    cpu%selected = .true.
    cpu%available = .true.
    call model%predict_log_proba_device(cpu, query, log_plus, status)
    call check(status_ok(status) .and. maxval(abs(log_plus - log_probabilities)) < 3.0e-14_dp, &
        "CPU log-probability dispatch", failures)
    call model%predict_log_proba_parameter_jvp_device(cpu, query, direction, &
        log_plus, parameter_log_dot, status)
    call check(status_ok(status), "CPU parameter log-probability dispatch", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    sentinel = -91.0_dp
    call model%predict_log_proba_device(cuda, query, sentinel, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        maxval(abs(sentinel + 91.0_dp)) < 1.0e-14_dp, &
        "typed CUDA log-probability refusal is transactional", failures)
    call model%predict_log_proba_parameter_jvp_device(cuda, query, direction, &
        sentinel, parameter_log_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "typed CUDA parameter JVP refusal", failures)

    sentinel = -37.0_dp
    call model%predict_log_proba(query(:2, :), sentinel, status)
    call check(.not. status_ok(status) .and. maxval(abs(sentinel + 37.0_dp)) < 1.0e-14_dp, &
        "malformed query refusal is transactional", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL LightGBM multiclass log-probability cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS LightGBM multiclass log-probability independent oracle"

contains

    subroutine independent_log_oracle(model, query, observed, failures)
        type(lightgbm_multiclass_t), intent(in) :: model
        real(dp), intent(in) :: query(:, :), observed(:, :)
        integer, intent(inout) :: failures
        real(dp) :: margins(3, 3), positive(3, 3), expected(3, 3)
        real(dp) :: total
        integer :: i, j

        call model%decision_function(query, margins, status)
        do i = 1, size(query, 1)
            total = 0.0_dp
            do j = 1, size(margins, 2)
                positive(i, j) = stable_sigmoid(margins(i, j))
                total = total + positive(i, j)
            end do
            do j = 1, size(margins, 2)
                expected(i, j) = log(positive(i, j)/total)
            end do
        end do
        call check(status_ok(status) .and. maxval(abs(observed - expected)) < 3.0e-14_dp, &
            "independent margin/log normalization oracle", failures)
    end subroutine independent_log_oracle

    real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [lgbm-multiclass-log-proba] "//description
        end if
    end subroutine check

end program test_lightgbm_multiclass_log_proba
