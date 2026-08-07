program test_gp_mean
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_gp_mean, only: gp_mean_t, make_constant_mean, make_linear_mean
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(gp_regression_t) :: constant_model, linear_model, bad_model
    type(gp_mean_t) :: constant_mean, linear_mean, bad_mean
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    real(dp) :: x(4, 1), y_constant(4, 1), y_linear(4, 2)
    real(dp) :: query(2, 1), mean_value(2, 2), mean_dot(2, 2)
    real(dp) :: constant_mean_value(2, 1), constant_mean_dot(2, 1)
    real(dp) :: variance(2), variance_dot(2), direction(7), hvp(7)
    real(dp) :: mean_bar(2, 2), variance_bar(2), parameter_bar(7)
    real(dp) :: parameter_bar_plus(7), parameter_bar_minus(7)
    real(dp) :: objective_plus, objective_minus, directional_fd
    real(dp) :: lml_plus, lml_minus, epsilon, max_gradient_error
    real(dp) :: theta(7), theta_plus(7), theta_minus(7), fd(7)
    real(dp) :: gradient(7), gradient_plus(7), gradient_minus(7)
    real(dp), allocatable :: parameters(:), mean_parameters(:), theta_constant(:)
    integer :: i, j

    do i = 1, 4
        x(i, 1) = real(i - 2, dp)
        y_constant(i, 1) = 0.3_dp + 0.2_dp*x(i, 1) + 0.05_dp*x(i, 1)**2
        y_linear(i, 1) = y_constant(i, 1)
        y_linear(i, 2) = -0.4_dp + 0.1_dp*x(i, 1) - 0.03_dp*x(i, 1)**2
    end do
    query(:, 1) = [-0.5_dp, 1.5_dp]
    kernel = make_rbf_kernel(1, 0.8_dp, 1.3_dp, status)
    constant_mean = make_constant_mean(1, status, 0.2_dp)
    call constant_model%fit(x, y_constant, kernel, 0.15_dp, status, mean=constant_mean)
    if (.not. status_ok(status) .or. constant_model%mean_parameter_count() /= 1 .or. &
        constant_model%parameter_count() /= 4) then
        write (error_unit, '(a)') "FAIL [constant GP mean fit/packing]"
        error stop 1
    end if

    call check_lml_gradient(constant_model, 4, max_gradient_error)
    if (max_gradient_error > 3.0e-5_dp) then
        write (error_unit, '(a,es14.5)') &
            "FAIL [constant GP mean LML gradient] error=", max_gradient_error
        error stop 1
    end if

    theta_constant = constant_model%parameters()
    theta_constant(4) = 0.35_dp
    call constant_model%set_parameters(theta_constant(:3), status)
    if (status_ok(status)) then
        write (error_unit, '(a)') "FAIL [constant GP mean malformed pack accepted]"
        error stop 1
    end if
    call constant_model%set_parameters(theta_constant, status)
    if (.not. status_ok(status)) then
        write (error_unit, '(a,i4,1x,a)') "FAIL [constant GP mean set_parameters]", &
            status%code, status%msg
        error stop 1
    end if

    direction = [0.03_dp, -0.04_dp, 0.02_dp, 0.07_dp, 0.0_dp, 0.0_dp, 0.0_dp]
    call check_lml_hvp(constant_model, direction(:4), max_gradient_error)
    if (max_gradient_error > 2.0e-4_dp) then
        write (error_unit, '(a,es14.5)') &
            "FAIL [constant GP mean LML HVP] error=", max_gradient_error
        error stop 1
    end if

    call constant_model%predict_jvp(query, direction(:4), constant_mean_value, &
        constant_mean_dot, &
        variance, variance_dot, status)
    if (.not. status_ok(status)) then
        write (error_unit, '(a,i4,1x,a)') &
            "FAIL [constant GP mean prediction JVP status]", status%code, status%msg
        error stop 1
    end if
    call check_prediction_jvp(constant_model, query, direction(:4), constant_mean_dot, &
        variance_dot, max_gradient_error)
    if (max_gradient_error > 3.0e-5_dp) then
        write (error_unit, '(a,es14.5)') &
            "FAIL [constant GP mean prediction JVP] error=", max_gradient_error
        error stop 1
    end if

    linear_mean = make_linear_mean(1, status, [0.1_dp, 0.5_dp])
    call linear_model%fit(x, y_linear, kernel, 0.15_dp, status, mean=linear_mean)
    if (.not. status_ok(status) .or. linear_model%mean_parameter_count() /= 4 .or. &
        linear_model%parameter_count() /= 7) then
        write (error_unit, '(a)') "FAIL [linear GP mean fit/packing]"
        error stop 1
    end if
    mean_parameters = linear_model%mean_parameters()
    if (size(mean_parameters) /= 4 .or. maxval(abs(mean_parameters - &
        [0.1_dp, 0.5_dp, 0.1_dp, 0.5_dp])) > 1.0e-13_dp) then
        write (error_unit, '(a)') "FAIL [linear GP mean coefficient replication]"
        error stop 1
    end if

    call check_lml_gradient(linear_model, 7, max_gradient_error)
    if (max_gradient_error > 3.0e-5_dp) then
        write (error_unit, '(a,es14.5)') &
            "FAIL [linear GP mean LML gradient] error=", max_gradient_error
        error stop 1
    end if

    theta = linear_model%parameters()
    direction = [0.02_dp, -0.03_dp, 0.01_dp, 0.04_dp, -0.02_dp, 0.03_dp, -0.05_dp]
    call linear_model%predict_jvp(query, direction, mean_value, mean_dot, &
        variance, variance_dot, status)
    if (.not. status_ok(status)) then
        write (error_unit, '(a)') "FAIL [linear GP mean prediction JVP status]"
        error stop 1
    end if
    call check_prediction_jvp(linear_model, query, direction, mean_dot, &
        variance_dot, max_gradient_error)
    if (max_gradient_error > 3.0e-5_dp) then
        write (error_unit, '(a,es14.5)') &
            "FAIL [linear GP mean prediction JVP] error=", max_gradient_error
        error stop 1
    end if

    mean_bar = reshape([0.7_dp, -0.2_dp, 0.4_dp, 0.3_dp], shape(mean_bar))
    variance_bar = [0.15_dp, -0.25_dp]
    call linear_model%predict_vjp(query, mean_bar, variance_bar, parameter_bar, status)
    if (.not. status_ok(status)) then
        write (error_unit, '(a)') "FAIL [linear GP mean prediction VJP status]"
        error stop 1
    end if
    theta = linear_model%parameters()
    theta_plus = theta + 1.0e-5_dp*direction
    theta_minus = theta - 1.0e-5_dp*direction
    call linear_model%set_parameters(theta_plus, status)
    call linear_model%predict(query, mean_value, variance, status)
    objective_plus = sum(mean_value*mean_bar) + dot_product(variance, variance_bar)
    call linear_model%set_parameters(theta_minus, status)
    call linear_model%predict(query, mean_value, variance, status)
    objective_minus = sum(mean_value*mean_bar) + dot_product(variance, variance_bar)
    call linear_model%set_parameters(theta, status)
    directional_fd = (objective_plus - objective_minus)/(2.0e-5_dp)
    if (abs(directional_fd - dot_product(parameter_bar, direction)) > 4.0e-5_dp) then
        write (error_unit, '(a,es14.5)') &
            "FAIL [linear GP mean prediction VJP dual] error=", &
            abs(directional_fd - dot_product(parameter_bar, direction))
        error stop 1
    end if

    call linear_model%predict_hvp(query, mean_bar, direction, hvp, status)
    call linear_model%set_parameters(theta_plus, status)
    call linear_model%predict_vjp(query, mean_bar, variance_bar, parameter_bar_plus, status)
    call linear_model%set_parameters(theta_minus, status)
    call linear_model%predict_vjp(query, mean_bar, variance_bar, parameter_bar_minus, status)
    call linear_model%set_parameters(theta, status)
    if (.not. status_ok(status) .or. maxval(abs(hvp - &
        (parameter_bar_plus - parameter_bar_minus)/(2.0e-5_dp))) > 5.0e-4_dp) then
        write (error_unit, '(a)') "FAIL [linear GP mean prediction HVP]"
        error stop 1
    end if

    bad_mean = make_constant_mean(2, status)
    call bad_model%fit(x, y_constant, kernel, 0.15_dp, status, mean=bad_mean)
    if (status_ok(status)) then
        write (error_unit, '(a)') "FAIL [GP mean dimension refusal]"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine check_lml_gradient(model, count, error)
        type(gp_regression_t), intent(inout) :: model
        integer, intent(in) :: count
        real(dp), intent(out) :: error
        real(dp) :: base(count), plus(count), minus(count), gradient_local(count)
        real(dp) :: plus_value, minus_value, step
        type(fortnum_status_t) :: local_status
        integer :: k

        base = model%parameters()
        call model%hyperparameter_gradient(gradient_local, local_status)
        error = huge(1.0_dp)
        if (.not. status_ok(local_status)) return
        step = 1.0e-5_dp
        do k = 1, count
            plus = base
            minus = base
            plus(k) = plus(k) + step
            minus(k) = minus(k) - step
            call model%set_parameters(plus, local_status)
            call model%log_marginal_likelihood(plus_value, local_status)
            call model%set_parameters(minus, local_status)
            call model%log_marginal_likelihood(minus_value, local_status)
            if (.not. status_ok(local_status)) return
            if (abs(plus_value) > huge(1.0_dp)/4.0_dp .or. &
                abs(minus_value) > huge(1.0_dp)/4.0_dp) return
            base(k) = base(k)
            gradient_local(k) = gradient_local(k)
            plus_value = (plus_value - minus_value)/(2.0_dp*step)
            gradient_local(k) = gradient_local(k) - plus_value
        end do
        call model%set_parameters(base, local_status)
        error = maxval(abs(gradient_local))
    end subroutine check_lml_gradient

    subroutine check_lml_hvp(model, direction_local, error)
        type(gp_regression_t), intent(inout) :: model
        real(dp), intent(in) :: direction_local(:)
        real(dp), intent(out) :: error
        real(dp), allocatable :: base(:), plus(:), minus(:), hessian_vector(:)
        real(dp), allocatable :: gradient_plus_local(:), gradient_minus_local(:)
        real(dp) :: step
        type(fortnum_status_t) :: local_status

        base = model%parameters()
        allocate(plus(size(base)), minus(size(base)))
        allocate(gradient_plus_local(size(base)), gradient_minus_local(size(base)))
        allocate(hessian_vector(size(base)))
        call model%hyperparameter_hvp(direction_local, hessian_vector, local_status)
        step = 1.0e-5_dp
        plus = base + step*direction_local
        minus = base - step*direction_local
        call model%set_parameters(plus, local_status)
        call model%hyperparameter_gradient(gradient_plus_local, local_status)
        call model%set_parameters(minus, local_status)
        call model%hyperparameter_gradient(gradient_minus_local, local_status)
        call model%set_parameters(base, local_status)
        error = maxval(abs(hessian_vector - &
            (gradient_plus_local - gradient_minus_local)/(2.0_dp*step)))
    end subroutine check_lml_hvp

    subroutine check_prediction_jvp(model, query_local, direction_local, mean_dot_local, &
            variance_dot_local, error)
        type(gp_regression_t), intent(inout) :: model
        real(dp), intent(in) :: query_local(:, :), direction_local(:)
        real(dp), intent(in) :: mean_dot_local(:, :), variance_dot_local(:)
        real(dp), intent(out) :: error
        real(dp), allocatable :: base(:), plus(:), minus(:)
        real(dp), allocatable :: mean_plus(:, :), mean_minus(:, :)
        real(dp), allocatable :: variance_plus(:), variance_minus(:)
        real(dp) :: step
        type(fortnum_status_t) :: local_status

        base = model%parameters()
        allocate(plus(size(base)), minus(size(base)))
        allocate(mean_plus(size(query_local, 1), size(mean_dot_local, 2)))
        allocate(mean_minus, mold=mean_plus)
        allocate(variance_plus(size(query_local, 1)), variance_minus(size(query_local, 1)))
        step = 1.0e-5_dp
        plus = base + step*direction_local
        minus = base - step*direction_local
        call model%set_parameters(plus, local_status)
        call model%predict(query_local, mean_plus, variance_plus, local_status)
        call model%set_parameters(minus, local_status)
        call model%predict(query_local, mean_minus, variance_minus, local_status)
        call model%set_parameters(base, local_status)
        error = max(maxval(abs(mean_dot_local - (mean_plus - mean_minus)/(2.0_dp*step))), &
            maxval(abs(variance_dot_local - (variance_plus - variance_minus)/(2.0_dp*step))))
    end subroutine check_prediction_jvp

end program test_gp_mean
