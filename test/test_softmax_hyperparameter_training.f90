program test_softmax_hyperparameter_training
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_softmax_regression, only: softmax_regression_t
    use fortml_softmax_training, only: softmax_training_objective_t, &
        softmax_lbfgsb_options_t, softmax_lbfgsb_result_t, &
        softmax_optimize_lbfgsb
    use fortopt_objective, only: objective_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(softmax_regression_t), target :: model
    type(softmax_training_objective_t) :: objective
    type(softmax_lbfgsb_options_t) :: options
    type(softmax_lbfgsb_result_t) :: result
    type(objective_t) :: fortopt_objective
    type(fortnum_status_t) :: status
    real(dp) :: x(5, 2), parameters(10), direction(10)
    real(dp) :: gradient(10), reference_gradient(10), plus_gradient(10)
    real(dp) :: minus_gradient(10), value, reference_value, value_plus, value_minus
    real(dp) :: sample_weight(5), class_weight(3), h
    integer :: labels(5), failures

    failures = 0
    x = reshape([ &
        -1.0_dp, 0.2_dp, 0.7_dp, 1.1_dp, -0.3_dp, -0.8_dp, &
        1.3_dp, -0.4_dp, 0.2_dp, 0.9_dp], shape(x))
    labels = [-3, 4, 9, 4, -3]
    sample_weight = [1.0_dp, 2.0_dp, 0.5_dp, 1.2_dp, 0.7_dp]
    class_weight = [1.0_dp, 2.0_dp, 3.0_dp]

    call model%fit(x, labels, status, l2=0.2_dp, max_iterations=600, &
        tolerance=1.0e-8_dp)
    call check(status_ok(status), "softmax setup fit", failures)
    call objective%initialize(model, x, labels, 0.35_dp, status, &
        optimize_l2=.true., sample_weight=sample_weight, &
        class_weight=class_weight)
    call check(status_ok(status), "objective initialization", failures)
    call check(objective%parameter_count() == 10, &
        "objective packed parameter count", failures)

    parameters = [ &
        0.20_dp, -0.10_dp, -0.30_dp, 0.15_dp, 0.25_dp, -0.20_dp, &
        0.10_dp, -0.05_dp, 0.30_dp, 0.35_dp]
    direction = [ &
        -0.13_dp, 0.08_dp, 0.11_dp, -0.09_dp, 0.07_dp, 0.12_dp, &
        -0.06_dp, 0.15_dp, -0.10_dp, 0.04_dp]
    call objective%value_gradient(parameters, value, gradient, status)
    call check(status_ok(status), "objective value/gradient", failures)
    call reference_value_gradient(x, labels, sample_weight, class_weight, &
        parameters, reference_value, reference_gradient)
    call check(abs(value - reference_value) < 3.0e-13_dp, &
        "independent softmax objective value", failures)
    call check(maxval(abs(gradient - reference_gradient)) < 3.0e-13_dp, &
        "independent softmax objective gradient", failures)

    h = 1.0e-6_dp
    call objective%value_gradient(parameters + h*direction, value_plus, &
        plus_gradient, status)
    call check(status_ok(status), "HVP plus gradient", failures)
    call objective%value_gradient(parameters - h*direction, value_minus, &
        minus_gradient, status)
    call check(status_ok(status), "HVP minus gradient", failures)
    call objective%hvp(parameters, direction, gradient, status)
    call check(status_ok(status), "objective HVP", failures)
    call check(maxval(abs(gradient - (plus_gradient - minus_gradient)/(2.0_dp*h))) &
        < 3.0e-7_dp, "independent softmax HVP finite difference", failures)
    call check(abs(dot_product(reference_gradient, direction) - &
        (value_plus - value_minus)/(2.0_dp*h)) < 3.0e-7_dp, &
        "gradient directional adjoint oracle", failures)
    call check(abs(gradient(10) - dot_product(parameters(:6), direction(:6))) &
        < 3.0e-13_dp, "exact L2 hyperparameter HVP block", failures)

    call objective%fortopt(fortopt_objective, status)
    call check(status_ok(status), "FortOpt callback initialization", failures)
    call fortopt_objective%value_gradient(parameters, value_plus, plus_gradient, status)
    call check(status_ok(status) .and. abs(value_plus - value) < 3.0e-13_dp .and. &
        maxval(abs(plus_gradient - reference_gradient)) < 3.0e-13_dp, &
        "FortOpt callback oracle", failures)

    options%l2 = 0.35_dp
    options%max_iterations = 300
    options%gradient_tolerance = 1.0e-7_dp
    options%lower_bound = -8.0_dp
    options%upper_bound = 8.0_dp
    call softmax_optimize_lbfgsb(model, x, labels, options, result, status, &
        sample_weight=sample_weight, class_weight=class_weight)
    call check(status_ok(status), "FortOpt softmax optimization", failures)
    call check(result%converged, "FortOpt softmax convergence", failures)
    call check(result%l2 == options%l2, "fixed L2 result", failures)
    call check(abs(model%regularization() - options%l2) < 3.0e-13_dp, &
        "model L2 state", failures)
    call check(result%objective < huge(1.0_dp), "finite optimization objective", failures)

    options%optimize_l2 = .true.
    options%l2 = 0.5_dp
    options%l2_lower_bound = 1.0_dp
    options%l2_upper_bound = 0.1_dp
    call softmax_optimize_lbfgsb(model, x, labels, options, result, status)
    call check(.not. status_ok(status), "invalid L2 bounds refusal", failures)

    if (failures /= 0) error stop 1
    write (*, '(a)') "PASS softmax hyperparameter objective independent oracles"

contains

    subroutine reference_value_gradient(x, labels, sample_weight, class_weight, &
            parameters, value, gradient)
        real(dp), intent(in) :: x(:, :), sample_weight(:), class_weight(:), parameters(:)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, gradient(:)
        real(dp) :: logits(3), probabilities(3), effective_weight, weight_sum
        real(dp) :: maximum, normalizer, residual, l2
        integer :: i, j, k, target

        weight_sum = 0.0_dp
        do i = 1, size(labels)
            select case (labels(i))
            case (-3)
                weight_sum = weight_sum + sample_weight(i)*class_weight(1)
            case (4)
                weight_sum = weight_sum + sample_weight(i)*class_weight(2)
            case (9)
                weight_sum = weight_sum + sample_weight(i)*class_weight(3)
            end select
        end do
        value = 0.0_dp
        gradient = 0.0_dp
        do i = 1, size(labels)
            effective_weight = sample_weight(i)
            select case (labels(i))
            case (-3)
                target = 1
            case (4)
                target = 2
                effective_weight = effective_weight*class_weight(2)
            case (9)
                target = 3
                effective_weight = effective_weight*class_weight(3)
            case default
                target = 0
            end select
            if (labels(i) == -3) effective_weight = effective_weight*class_weight(1)
            do j = 1, 3
                logits(j) = dot_product(x(i, :), parameters((j - 1)*2 + 1:j*2)) &
                    + parameters(6 + j)
            end do
            maximum = maxval(logits)
            normalizer = sum(exp(logits - maximum))
            probabilities = exp(logits - maximum)/normalizer
            value = value + effective_weight*(maximum - logits(target) + log(normalizer))
            do j = 1, 3
                residual = effective_weight/weight_sum * &
                    (probabilities(j) - merge(1.0_dp, 0.0_dp, j == target))
                do k = 1, 2
                    gradient((j - 1)*2 + k) = gradient((j - 1)*2 + k) + residual*x(i, k)
                end do
                gradient(6 + j) = gradient(6 + j) + residual
            end do
        end do
        l2 = parameters(10)
        value = value/weight_sum + 0.5_dp*l2*sum(parameters(:6)**2)
        gradient(:6) = gradient(:6) + l2*parameters(:6)
        gradient(10) = 0.5_dp*sum(parameters(:6)**2)
    end subroutine reference_value_gradient

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL "//trim(description)
            error stop 1
        end if
    end subroutine check

end program test_softmax_hyperparameter_training
