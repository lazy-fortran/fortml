program test_derivative_gp_local_periodic
    !! Independent analytic and finite-difference checks for local-periodic
    !! mixed value/derivative observations and parameter/query products.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_derivative_gaussian_process, only: gp_derivative_regression_t
    use fortml_kernels, only: kernel_t, make_local_periodic_kernel
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_local_periodic_products(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " local-periodic derivative GP test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS local-periodic derivative GP analytic oracle"

contains

    subroutine test_local_periodic_products(failures)
        integer, intent(inout) :: failures
        type(gp_derivative_regression_t) :: model
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x_train(5, 2), y_train(5, 1), x_query(3, 2)
        real(dp) :: mean(3, 1), variance(3), mean_ref(3, 1), variance_ref(3)
        real(dp) :: theta(5), gradient(5), fd_gradient(5), direction(5)
        real(dp) :: value, value_dot, expected, h, plus, minus
        real(dp) :: mean_dot(3, 1), variance_dot(3), x_direction(3, 2)
        real(dp) :: mean_plus(3, 1), mean_minus(3, 1)
        real(dp) :: variance_plus(3), variance_minus(3), query_error
        real(dp) :: query_step
        integer :: components(5), query_components(3), i

        x_train = reshape([ &
            -0.8_dp, 0.1_dp, 0.35_dp, -0.55_dp, 0.9_dp, 0.72_dp, &
            -0.15_dp, 1.05_dp, 0.6_dp, -0.95_dp], shape(x_train))
        y_train(:, 1) = [0.7_dp, -0.2_dp, 0.95_dp, 0.3_dp, -0.65_dp]
        components = [0, 1, 2, 1, 0]
        x_query = reshape([0.15_dp, -0.3_dp, 0.72_dp, 0.44_dp, -0.48_dp, 0.93_dp], &
            shape(x_query))
        query_components = [2, 0, 1]
        ! The first query coincides with a value-observation training row.
        ! This exercises the removable local-periodic radial limits while the
        ! remaining rows cover value, first-feature, and second-feature query
        ! components in the same call.
        x_query(1, :) = x_train(1, :)
        kernel = make_local_periodic_kernel(2, 1.3_dp, 0.85_dp, 0.62_dp, 1.7_dp, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [local-periodic derivative GP] constructor"
            failures = failures + 1
            return
        end if
        call model%fit(x_train, components, y_train, kernel, 0.045_dp, status, &
            jitter=1.0e-10_dp)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [local-periodic derivative GP] fit"
            failures = failures + 1
            return
        end if

        theta = model%parameters()
        call model%log_marginal_likelihood(value, status)
        expected = oracle_lml(theta, x_train, components, y_train, 1.0e-10_dp)
        if (.not. status_ok(status) .or. abs(value - expected) > 2.0e-10_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [local-periodic derivative GP] independent likelihood oracle ", &
                abs(value - expected)
            failures = failures + 1
        end if

        call model%predict(x_query, query_components, mean, variance, status)
        call oracle_predict(theta, x_train, components, y_train, x_query, query_components, &
            1.0e-10_dp, mean_ref, variance_ref)
        if (.not. status_ok(status) .or. maxval(abs(mean - mean_ref)) > 4.0e-10_dp .or. &
            maxval(abs(variance - variance_ref)) > 4.0e-10_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [local-periodic derivative GP] independent prediction oracle ", &
                maxval(abs(mean - mean_ref)), maxval(abs(variance - variance_ref))
            failures = failures + 1
        end if

        call model%hyperparameter_gradient(gradient, status)
        h = 2.0e-5_dp
        do i = 1, size(theta)
            plus = oracle_lml(theta + h*unit_vector(size(theta), i), x_train, components, &
                y_train, 1.0e-10_dp)
            minus = oracle_lml(theta - h*unit_vector(size(theta), i), x_train, components, &
                y_train, 1.0e-10_dp)
            fd_gradient(i) = (plus - minus)/(2.0_dp*h)
        end do
        if (.not. status_ok(status) .or. maxval(abs(gradient - fd_gradient)) > 3.0e-6_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [local-periodic derivative GP] parameter JVP oracle ", &
                maxval(abs(gradient - fd_gradient))
            failures = failures + 1
        end if

        direction = [0.11_dp, -0.08_dp, 0.14_dp, -0.06_dp, 0.17_dp]
        call model%log_marginal_likelihood_jvp(direction, value_dot, status)
        if (.not. status_ok(status) .or. &
            abs(value_dot - dot_product(gradient, direction)) > 2.0e-9_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [local-periodic derivative GP] likelihood JVP ", &
                abs(value_dot - dot_product(gradient, direction))
            failures = failures + 1
        end if

        x_direction = reshape([0.07_dp, -0.03_dp, 0.11_dp, -0.05_dp, 0.09_dp, -0.08_dp], &
            shape(x_direction))
        call model%predict_input_jvp(x_query, query_components, x_direction, &
            mean, mean_dot, variance, variance_dot, status)
        query_step = 2.0e-5_dp
        call oracle_predict(theta, x_train, components, y_train, x_query + query_step*x_direction, &
            query_components, 1.0e-10_dp, mean_plus, variance_plus)
        call oracle_predict(theta, x_train, components, y_train, x_query - query_step*x_direction, &
            query_components, 1.0e-10_dp, mean_minus, variance_minus)
        query_error = max(maxval(abs(mean_dot - (mean_plus - mean_minus)/(2.0_dp*query_step))), &
            maxval(abs(variance_dot - (variance_plus - variance_minus)/(2.0_dp*query_step))))
        if (.not. status_ok(status) .or. query_error > 3.0e-7_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [local-periodic derivative GP] query-input JVP oracle ", query_error
            failures = failures + 1
        end if
    end subroutine test_local_periodic_products

    real(dp) function oracle_lml(theta, x, components, y, jitter) result(value)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), jitter
        integer, intent(in) :: components(:)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: status
        real(dp), allocatable :: covariance(:, :), alpha(:, :)
        real(dp) :: logdet
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)))
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = oracle_covariance(theta, x(i, :), components(i), &
                    x(j, :), components(j))
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(5)) + jitter
        end do
        call factor%factorize(covariance, status)
        allocate(alpha, source=y)
        call factor%solve(alpha, status)
        call factor%log_determinant(logdet, status)
        value = -0.5_dp*sum(y*alpha) - 0.5_dp*logdet - &
            0.5_dp*real(size(x, 1), dp)*log(2.0_dp*acos(-1.0_dp))
    end function oracle_lml

    subroutine oracle_predict(theta, x, components, y, x_query, query_components, jitter, &
            mean, variance)
        real(dp), intent(in) :: theta(:), x(:, :), y(:, :), x_query(:, :), jitter
        integer, intent(in) :: components(:), query_components(:)
        real(dp), intent(out) :: mean(:, :), variance(:)
        type(cholesky_factorization_t) :: factor
        type(fortnum_status_t) :: status
        real(dp), allocatable :: covariance(:, :), cross(:, :), alpha(:, :), work(:, :)
        integer :: i, j

        allocate(covariance(size(x, 1), size(x, 1)), cross(size(x, 1), size(x_query, 1)))
        allocate(alpha, source=y)
        do j = 1, size(x, 1)
            do i = 1, size(x, 1)
                covariance(i, j) = oracle_covariance(theta, x(i, :), components(i), &
                    x(j, :), components(j))
            end do
        end do
        do i = 1, size(x, 1)
            covariance(i, i) = covariance(i, i) + exp(theta(5)) + jitter
        end do
        call factor%factorize(covariance, status)
        call factor%solve(alpha, status)
        do j = 1, size(x_query, 1)
            do i = 1, size(x, 1)
                cross(i, j) = oracle_covariance(theta, x(i, :), components(i), &
                    x_query(j, :), query_components(j))
            end do
        end do
        mean = matmul(transpose(cross), alpha)
        allocate(work, source=cross)
        call factor%solve(work, status)
        do j = 1, size(x_query, 1)
            variance(j) = oracle_covariance(theta, x_query(j, :), query_components(j), &
                x_query(j, :), query_components(j)) - dot_product(cross(:, j), work(:, j))
        end do
    end subroutine oracle_predict

    real(dp) function oracle_covariance(theta, x1, component1, x2, component2) result(value)
        real(dp), intent(in) :: theta(:), x1(:), x2(:)
        integer, intent(in) :: component1, component2
        real(dp) :: difference(size(x1)), squared_distance, distance
        real(dp) :: variance, envelope_scale, periodic_scale, period, a, b, c
        real(dp) :: argument, sine_value, cosine_value, log_r, log_rr
        real(dp) :: first_r, second_r, first_t, second_t

        difference = x1 - x2
        squared_distance = sum(difference*difference)
        distance = sqrt(squared_distance)
        variance = exp(theta(1))
        envelope_scale = exp(theta(2))
        periodic_scale = exp(theta(3))
        period = exp(theta(4))
        a = 0.5_dp/(envelope_scale*envelope_scale)
        b = 2.0_dp/(periodic_scale*periodic_scale)
        c = acos(-1.0_dp)/period
        if (distance == 0.0_dp) then
            value = variance
            first_t = -value*(a + b*c*c)
            second_t = value*((a + b*c*c)**2 + 2.0_dp*b*c**4/3.0_dp)
        else
            argument = c*distance
            sine_value = sin(argument)
            cosine_value = cos(argument)
            value = variance*exp(-a*squared_distance - b*sine_value*sine_value)
            log_r = -2.0_dp*a*distance - 2.0_dp*b*c*sine_value*cosine_value
            log_rr = -2.0_dp*a - 2.0_dp*b*c*c*(cosine_value*cosine_value - &
                sine_value*sine_value)
            first_r = value*log_r
            second_r = value*(log_r*log_r + log_rr)
            first_t = first_r/(2.0_dp*distance)
            second_t = (second_r - first_r/distance)/(4.0_dp*squared_distance)
        end if
        if (component1 == 0 .and. component2 == 0) return
        if (component1 > 0 .and. component2 == 0) then
            value = 2.0_dp*first_t*difference(component1)
        else if (component1 == 0 .and. component2 > 0) then
            value = -2.0_dp*first_t*difference(component2)
        else
            value = -2.0_dp*first_t*merge(1.0_dp, 0.0_dp, component1 == component2) - &
                4.0_dp*second_t*difference(component1)*difference(component2)
        end if
    end function oracle_covariance

    pure function unit_vector(n, index) result(vector)
        integer, intent(in) :: n, index
        real(dp) :: vector(n)

        vector = 0.0_dp
        vector(index) = 1.0_dp
    end function unit_vector

end program test_derivative_gp_local_periodic
