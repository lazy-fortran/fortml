program test_mlp_multilabel_classifier
    !! Independent finite-difference and adjoint checks for the multilabel MLP.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_mlp_multilabel_classifier, only: mlp_multilabel_classifier_t, &
        mlp_multilabel_classifier_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer, parameter :: n = 6, p = 2, labels_count = 2
    real(dp), parameter :: eps = 2.0e-6_dp
    real(dp) :: x(n, p), x_dot(n, p), scores(n, labels_count), scores_dot(n, labels_count)
    real(dp) :: scores_tangent(n, labels_count)
    real(dp) :: score_plus(n, labels_count), score_minus(n, labels_count)
    real(dp) :: probabilities(n, labels_count), probabilities_dot(n, labels_count)
    real(dp) :: probability_plus(n, labels_count), probability_minus(n, labels_count)
    real(dp) :: targets(n, labels_count), loss, loss_plus, loss_minus, lhs, rhs
    real(dp), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
    real(dp), allocatable :: gradient(:), gradient_plus(:), gradient_minus(:), hvp(:)
    real(dp), allocatable :: theta_bar(:), x_bar(:, :)
    integer :: predicted(n, labels_count), i, j, failures
    type(mlp_multilabel_classifier_t) :: model
    type(mlp_multilabel_classifier_options_t) :: options
    type(fortml_device_t) :: cuda
    type(fortnum_status_t) :: status

    failures = 0
    x(:, 1) = [-1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.2_dp]
    x(:, 2) = [-1.0_dp, -0.2_dp, 0.0_dp, 0.2_dp, 1.0_dp, 0.8_dp]
    targets(:, 1) = [0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
    targets(:, 2) = [0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp]
    options%max_epochs = 2
    options%batch_size = 0
    options%restore_best = .false.
    options%learning_rate = 0.02_dp
    options%epsilon = 1.0e-7_dp
    options%l2 = 0.01_dp
    options%tolerance = 0.0_dp
    options%initialization_seed = 31
    call model%fit(x, targets, status, options=options)
    call check(status_ok(status), "multilabel fit", failures)
    call check(model%fitted(), "multilabel fitted", failures)
    call check(model%label_count() == labels_count, "label count", failures)
    call check(model%feature_count() == p, "feature count", failures)
    call check(model%parameter_count() == labels_count*(p + 1), &
        "packed parameter count", failures)
    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)), &
        gradient(size(theta)), gradient_plus(size(theta)), gradient_minus(size(theta)), &
        hvp(size(theta)), theta_bar(size(theta)), x_bar(n, p))
    do i = 1, size(theta)
        theta_dot(i) = 0.013_dp*real(i, dp)
    end do
    do i = 1, n
        do j = 1, p
            x_dot(i, j) = 0.017_dp*real(i + 2*j, dp)
        end do
    end do

    call model%predict_proba(x, probabilities, status)
    call check(status_ok(status), "probability prediction", failures)
    call check(all(probabilities > 0.0_dp) .and. all(probabilities < 1.0_dp), &
        "probability range", failures)
    call model%predict(x, predicted, status)
    call check(status_ok(status), "hard prediction", failures)
    call check(all((predicted == 0) .or. (predicted == 1)), &
        "indicator prediction", failures)
    call model%loss_gradient(x, targets, options%l2, loss, gradient, status)
    call check(status_ok(status), "loss gradient", failures)
    call model%loss_hvp(x, targets, options%l2, theta_dot, hvp, status)
    call check(status_ok(status), "loss HVP", failures)

    theta_plus = theta + eps*theta_dot
    theta_minus = theta - eps*theta_dot
    call model%set_parameters(theta_plus, status)
    call check(status_ok(status), "set plus parameters", failures)
    call model%decision_function(x + eps*x_dot, score_plus, status)
    call check(status_ok(status), "score plus", failures)
    call model%predict_proba(x + eps*x_dot, probability_plus, status)
    call check(status_ok(status), "probability plus", failures)
    call model%loss_gradient(x, targets, options%l2, loss_plus, gradient_plus, status)
    call check(status_ok(status), "gradient plus", failures)
    call model%set_parameters(theta_minus, status)
    call check(status_ok(status), "set minus parameters", failures)
    call model%decision_function(x - eps*x_dot, score_minus, status)
    call check(status_ok(status), "score minus", failures)
    call model%predict_proba(x - eps*x_dot, probability_minus, status)
    call check(status_ok(status), "probability minus", failures)
    call model%loss_gradient(x, targets, options%l2, loss_minus, gradient_minus, status)
    call check(status_ok(status), "gradient minus", failures)
    call model%set_parameters(theta, status)
    call check(status_ok(status), "restore parameters", failures)

    call check(maxval(abs(hvp - (gradient_plus - gradient_minus)/(2.0_dp*eps))) &
        < 2.0e-5_dp, "loss HVP finite difference", failures)
    call model%decision_function_jvp(x, theta_dot, x_dot, scores, scores_tangent, status)
    call check(status_ok(status), "score JVP", failures)
    call check(maxval(abs(scores_tangent - (score_plus - score_minus)/(2.0_dp*eps))) &
        < 2.0e-5_dp, "score JVP finite difference", failures)
    call model%predict_proba_jvp(x, theta_dot, x_dot, probabilities, probabilities_dot, status)
    call check(status_ok(status), "probability JVP", failures)
    call check(maxval(abs(probabilities_dot - (probability_plus - probability_minus)/ &
        (2.0_dp*eps))) < 2.0e-5_dp, "probability JVP finite difference", failures)

    do i = 1, n
        do j = 1, labels_count
            scores_dot(i, j) = 0.11_dp*real(i + j, dp)
        end do
    end do
    call model%decision_function_vjp(x, scores_dot, theta_bar, x_bar, status)
    call check(status_ok(status), "score VJP", failures)
    lhs = sum(scores_dot*scores_tangent)
    rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
    call check(abs(lhs - rhs) < 2.0e-10_dp, "score VJP duality", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "multilabel CUDA refusal", failures)

    targets(1, 1) = 0.5_dp
    call model%fit(x, targets, status, options=options)
    call check(status%code == FORTNUM_DOMAIN_ERROR, &
        "non-indicator target refusal", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP multilabel cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP multilabel independent behavioral oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: label
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: "//label
        end if
    end subroutine check

end program test_mlp_multilabel_classifier
