program test_mlp_classifier_parameter_products
    !! Independent finite-difference and adjoint oracle for fixed-input
    !! multiclass MLP probability parameter products.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_mlp_classifier, only: mlp_classifier_t, mlp_classifier_options_t
    implicit none

    integer, parameter :: dp = real64
    type(mlp_classifier_t) :: model, unfitted
    type(mlp_classifier_options_t) :: options
    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    real(dp) :: x(9, 2), probabilities(9, 3), probabilities_dot(9, 3)
    real(dp) :: probabilities_plus(9, 3), probabilities_minus(9, 3)
    real(dp) :: probabilities_bar(9, 3), tangent_left, tangent_right, h
    real(dp), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
    real(dp), allocatable :: theta_bar(:)
    integer :: labels(9), failures, i

    x = reshape([ &
        -2.0_dp, -1.0_dp, -1.0_dp, -2.0_dp, 0.0_dp, 2.0_dp, &
        0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp, 2.0_dp, 1.0_dp, &
        2.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, 2.0_dp, 1.0_dp], shape(x))
    labels = [31, 31, -7, -7, 42, 42, 42, 31, -7]
    failures = 0

    call unfitted%predict_proba_parameter_jvp(x, [0.0_dp], probabilities, &
        probabilities_dot, status)
    call check(.not. status_ok(status), "unfitted parameter JVP refusal", failures)

    options%max_epochs = 30
    options%learning_rate = 0.03_dp
    options%initialization_seed = 29
    options%restore_best = .false.
    call model%fit(x, labels, status, hidden_layer_sizes=[3], options=options)
    call check(status_ok(status), "parameter-product fixture fit", failures)
    theta = model%parameters()
    allocate(theta_dot(size(theta)), theta_plus(size(theta)), theta_minus(size(theta)), &
        theta_bar(size(theta)))
    theta_dot = [(0.013_dp*real(i, dp), i=1, size(theta))]
    probabilities_bar = reshape([(0.017_dp*real(i, dp), i=1, size(probabilities_bar))], &
        shape(probabilities_bar))
    h = 1.0e-6_dp

    call model%predict_proba_parameter_jvp(x, theta_dot, probabilities, &
        probabilities_dot, status)
    theta_plus = theta + h*theta_dot
    theta_minus = theta - h*theta_dot
    call model%set_parameters(theta_plus, status)
    call model%predict_proba(x, probabilities_plus, status)
    call model%set_parameters(theta_minus, status)
    call model%predict_proba(x, probabilities_minus, status)
    call model%set_parameters(theta, status)
    call check(status_ok(status) .and. maxval(abs(probabilities_dot - &
        (probabilities_plus - probabilities_minus)/(2.0_dp*h))) < 4.0e-5_dp, &
        "parameter probability JVP central difference", failures)

    call model%predict_proba_parameter_vjp(x, probabilities_bar, theta_bar, status)
    tangent_left = sum(probabilities_bar*probabilities_dot)
    tangent_right = dot_product(theta_bar, theta_dot)
    call check(status_ok(status) .and. abs(tangent_left - tangent_right) < 3.0e-10_dp, &
        "parameter probability VJP duality", failures)

    call cpu%select(FORTML_DEVICE_CPU, status)
    call model%predict_proba_parameter_jvp_device(cpu, x, theta_dot, probabilities, &
        probabilities_dot, status)
    call check(status_ok(status), "CPU parameter JVP dispatch", failures)
    call model%predict_proba_parameter_vjp_device(cpu, x, probabilities_bar, theta_bar, &
        status)
    call check(status_ok(status), "CPU parameter VJP dispatch", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call model%predict_proba_parameter_jvp_device(cuda, x, theta_dot, probabilities, &
        probabilities_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA parameter JVP refusal", failures)
    call model%predict_proba_parameter_vjp_device(cuda, x, probabilities_bar, theta_bar, &
        status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "CUDA parameter VJP refusal", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP classifier parameter-product cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP classifier parameter-product independent oracle"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_mlp_classifier_parameter_products
