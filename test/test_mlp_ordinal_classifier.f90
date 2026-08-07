program test_mlp_ordinal_classifier
    !! Independent cumulative-logit neural classifier oracle.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp_ordinal_classifier, only: mlp_ordinal_classifier_t, &
        mlp_ordinal_classifier_options_t
    implicit none

    type(mlp_ordinal_classifier_t) :: model
    type(mlp_ordinal_classifier_options_t) :: options
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cpu, cuda
    real(real64) :: x(8, 2), x_dot(8, 2), p(8, 3), p_dot(8, 3)
    real(real64) :: p_plus(8, 3), p_minus(8, 3), p_bar(8, 3)
    real(real64), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
    real(real64), allocatable :: theta_bar(:), x_bar(:, :), scores(:, :)
    real(real64) :: expected(8, 3), q1, q2, score, h, lhs, rhs
    real(real64), allocatable :: threshold(:)
    integer :: labels(8), predicted(8), classes(3), failures, i

    x = reshape([ &
        -1.2_real64, 0.3_real64, -0.8_real64, 0.9_real64, &
        -0.4_real64, -0.6_real64, 0.0_real64, 0.2_real64, &
        0.5_real64, -0.1_real64, 0.8_real64, 0.7_real64, &
        1.1_real64, -0.5_real64, 1.4_real64, 0.4_real64], shape(x))
    x_dot = reshape([ &
        0.04_real64, -0.03_real64, 0.02_real64, -0.01_real64, &
        0.05_real64, -0.06_real64, 0.07_real64, -0.08_real64, &
        0.09_real64, -0.10_real64, 0.11_real64, -0.12_real64, &
        0.13_real64, -0.14_real64, 0.15_real64, -0.16_real64], shape(x_dot))
    labels = [30, 10, 20, 30, 10, 20, 30, 10]
    p_bar = reshape([ &
        0.2_real64, -0.3_real64, 0.4_real64, -0.1_real64, 0.5_real64, -0.6_real64, &
        0.7_real64, -0.8_real64, 0.9_real64, -0.2_real64, 0.3_real64, -0.4_real64, &
        0.1_real64, -0.5_real64, 0.6_real64, -0.7_real64, 0.8_real64, -0.9_real64, &
        0.4_real64, -0.2_real64, 0.3_real64, -0.1_real64, 0.6_real64, -0.5_real64], &
        shape(p_bar))
    failures = 0
    h = 1.0e-6_real64
    options%max_iterations = 400
    options%tolerance = 1.0e-6_real64
    options%l2 = 1.0e-2_real64
    call model%fit(x, labels, status, hidden_layer_sizes=[3], options=options)
    call check(status_ok(status) .and. model%fitted(), &
        "deterministic ordinal neural fit", failures)
    call check(model%class_count() == 3 .and. model%feature_count() == 2, &
        "ordinal neural metadata", failures)
    classes = model%classes()
    call check(all(classes == [10, 20, 30]) .and. model%parameter_count() > 3, &
        "sorted ordered labels and packed parameters", failures)

    allocate(theta(model%parameter_count()), theta_dot(model%parameter_count()), &
        theta_plus(model%parameter_count()), theta_minus(model%parameter_count()), &
        theta_bar(model%parameter_count()), x_bar(size(x,1),size(x,2)), scores(size(x,1),1))
    theta = model%parameters()
    threshold = model%thresholds()
    call model%decision_function(x, scores, status)
    call check(status_ok(status), "latent score prediction", failures)
    call model%predict_proba(x, p, status)
    do i = 1, size(x, 1)
        score = scores(i,1)
        q1 = stable_sigmoid(threshold(1)-score)
        q2 = stable_sigmoid(threshold(2)-score)
        expected(i,:) = [q1, q2-q1, 1.0_real64-q2]
    end do
    call check(status_ok(status) .and. maxval(abs(p-expected)) < 3.0e-14_real64, &
        "cumulative-logit neural probability oracle", failures)
    call check(maxval(abs(sum(p, dim=2)-1.0_real64)) < 3.0e-14_real64 .and. &
        all(p >= -1.0e-12_real64), "probability simplex", failures)
    call model%predict(x, predicted, status)
    call check(status_ok(status) .and. all(predicted >= 10) .and. all(predicted <= 30), &
        "ordered neural predictions", failures)

    theta_dot = [(0.002_real64*real(i,real64), i=1,size(theta))]
    call model%predict_proba_parameter_jvp(x, theta_dot, p, p_dot, status)
    theta_plus = theta+h*theta_dot
    theta_minus = theta-h*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(x, p_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(x, p_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(p_dot-(p_plus-p_minus)/(2.0_real64*h))) < &
        4.0e-7_real64, "packed parameter JVP finite difference", failures)
    call model%predict_proba_parameter_vjp(x, p_bar, theta_bar, status, x_bar)
    lhs = sum(p_bar*p_dot)
    rhs = sum(theta_bar*theta_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 5.0e-7_real64, &
        "packed parameter VJP adjoint identity", failures)

    call model%predict_proba_jvp(x, theta_dot, x_dot, p, p_dot, status)
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(x+h*x_dot, p_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(x-h*x_dot, p_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(p_dot-(p_plus-p_minus)/(2.0_real64*h))) < &
        4.0e-7_real64, "joint input and parameter JVP finite difference", failures)
    call model%predict_proba_parameter_vjp(x, p_bar, theta_bar, status, x_bar)
    lhs = sum(p_bar*p_dot)
    rhs = sum(x_bar*x_dot) + sum(theta_bar*theta_dot)
    call check(status_ok(status) .and. abs(lhs-rhs) < 5.0e-7_real64, &
        "joint input and parameter VJP adjoint identity", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, p, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        .not. model%device_supported(FORTML_DEVICE_CUDA), "typed CUDA refusal", failures)
    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_proba_device(cpu, x, p_plus, status)
    call model%predict_proba(x, p_minus, status)
    call check(status_ok(status) .and. maxval(abs(p_plus-p_minus)) < 3.0e-14_real64, &
        "typed CPU dispatch", failures)

    if (failures /= 0) error stop failures
    print '(a)', "test_mlp_ordinal_classifier: PASS"

contains

    pure real(real64) function stable_sigmoid(value) result(probability)
        real(real64), intent(in) :: value
        if (value >= 0.0_real64) then
            probability = 1.0_real64/(1.0_real64+exp(-value))
        else
            probability = exp(value)/(1.0_real64+exp(value))
        end if
    end function stable_sigmoid

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count
        if (.not. condition) then
            failure_count = failure_count+1
            write (*, '(a)') "FAIL: "//trim(description)
        end if
    end subroutine check

end program test_mlp_ordinal_classifier
