program test_bayesian_ridge
    !! Independent closed-form posterior and fixed-state product oracle.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_bayesian_ridge, only: bayesian_ridge_regression_t
    implicit none
    integer, parameter :: dp = real64
    type(bayesian_ridge_regression_t) :: model
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: x(6, 2), y(6, 2), weights(6), design(6, 3)
    real(dp) :: precision(3, 3), rhs(3, 2), expected(3, 2), prediction(6, 2)
    real(dp) :: x_dot(6, 2), y_dot(6, 2), y_plus(6, 2), y_minus(6, 2)
    real(dp) :: theta_dot(6), theta(6), u(6, 2), theta_bar(6), x_bar(6, 2)
    real(dp) :: h, value, error, old_theta(6)
    integer :: failures, i, j, k

    failures = 0
    x = reshape([ -1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp, -2.0_dp, 0.5_dp, &
        0.5_dp, -1.5_dp, 1.5_dp, 0.25_dp, -0.75_dp, 2.0_dp ], shape(x))
    y(:, 1) = [0.2_dp, 1.1_dp, 2.0_dp, 2.6_dp, -0.8_dp, 1.7_dp]
    y(:, 2) = [-0.3_dp, 0.4_dp, 1.8_dp, 3.1_dp, -1.2_dp, 0.9_dp]
    weights = [0.5_dp, 1.0_dp, 1.4_dp, 0.8_dp, 1.2_dp, 0.7_dp]
    call model%fit(x, y, status, 2.3_dp, 0.7_dp, .true., weights)
    call check(status_ok(status), "weighted posterior fit", failures)
    design(:, 1) = 1.0_dp; design(:, 2:) = x
    precision = 0.7_dp * identity3()
    rhs = 0.0_dp
    do i = 1, size(x, 1)
        precision = precision + 2.3_dp * weights(i) * &
            outer(design(i, :), design(i, :))
        do j = 1, 3
            rhs(j, :) = rhs(j, :) + 2.3_dp * weights(i) * design(i, j) * y(i, :)
        end do
    end do
    call solve3(precision, rhs, expected)
    call model%predict(x, prediction, status)
    call check(status_ok(status), "posterior prediction", failures)
    call check(maxval(abs(model%posterior_mean() - expected)) < 2.0e-11_dp, &
        "closed-form posterior mean", failures)
    call check(maxval(abs(prediction - matmul(design, expected))) < 2.0e-11_dp, &
        "closed-form posterior prediction", failures)
    call check(abs(model%alpha()-2.3_dp) < 1.0e-14_dp .and. &
        abs(model%lambda()-0.7_dp) < 1.0e-14_dp .and. &
        model%log_evidence() == model%log_evidence(), "posterior metadata", failures)

    theta = model%parameters(); theta_dot = [0.03_dp, -0.02_dp, 0.01_dp, &
        0.04_dp, -0.05_dp, 0.02_dp]
    x_dot = 0.01_dp * reshape([(real(i, dp), i=1, size(x_dot))], shape(x_dot))
    call model%predict_jvp(x, theta_dot, x_dot, prediction, y_dot, status)
    h = 2.0e-6_dp; old_theta = theta
    call model%set_parameters(theta+h*theta_dot, status)
    call model%predict(x+h*x_dot, y_plus, status)
    call model%set_parameters(theta-h*theta_dot, status)
    call model%predict(x-h*x_dot, y_minus, status)
    call model%set_parameters(old_theta, status)
    error = maxval(abs(y_dot-(y_plus-y_minus)/(2.0_dp*h)))
    call check(error < 3.0e-8_dp, "prediction JVP finite-difference oracle", failures)
    u = reshape([(0.02_dp*real(i, dp), i=1, size(u))], shape(u))
    call model%predict_vjp(x, u, theta_bar, x_bar, status)
    call check(status_ok(status) .and. abs(dot_product(theta_dot, theta_bar) + &
        sum(x_dot*x_bar) - sum(y_dot*u)) < 2.0e-11_dp, &
        "prediction VJP duality", failures)

    call model%fit(x, y, status, -1.0_dp, 0.7_dp, .true., weights)
    call check(status%code /= 0, "invalid alpha refusal", failures)
    call model%fit(x, y, status, 2.3_dp, 0.7_dp, .true., 0.0_dp*weights)
    call check(status%code /= 0, "all-zero weight refusal", failures)
    cuda%kind = FORTML_DEVICE_CUDA; cuda%selected = .true.; cuda%available = .true.
    call model%predict_device(cuda, x, prediction, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA refusal", failures)

    if (failures > 0) then
        write (*, '(a,i0)') "FAIL Bayesian-ridge cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS weighted Bayesian-ridge independent behavioral oracle"

contains
    function identity3() result(value)
        real(dp) :: value(3,3)
        value = 0.0_dp; value(1,1)=1.0_dp; value(2,2)=1.0_dp; value(3,3)=1.0_dp
    end function identity3
    function outer(a, b) result(value)
        real(dp), intent(in) :: a(:), b(:)
        real(dp) :: value(size(a),size(b))
        integer :: i, j
        do i=1,size(a); do j=1,size(b); value(i,j)=a(i)*b(j); end do; end do
    end function outer
    subroutine solve3(a, b, result)
        real(dp), intent(in) :: a(3,3), b(3,2)
        real(dp), intent(out) :: result(3,2)
        real(dp) :: m(3,5), factor
        integer :: i, j, k
        m(:,1:3)=a; m(:,4:5)=b
        do k=1,2
            do i=k+1,3
                factor=m(i,k)/m(k,k); m(i,k:5)=m(i,k:5)-factor*m(k,k:5)
            end do
        end do
        do j=1,2
            result(3,j)=m(3,3+j)/m(3,3)
            result(2,j)=(m(2,3+j)-m(2,3)*result(3,j))/m(2,2)
            result(1,j)=(m(1,3+j)-m(1,2)*result(2,j)-m(1,3)*result(3,j))/m(1,1)
        end do
    end subroutine solve3
    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures
        if (.not. condition) then; failures=failures+1; write(*,'(a)') "FAIL: "//trim(label); end if
    end subroutine check
end program test_bayesian_ridge
