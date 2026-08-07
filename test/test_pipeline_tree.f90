program test_pipeline_tree
    use, intrinsic :: ieee_arithmetic, only: ieee_quiet_nan, ieee_value, &
        ieee_positive_inf
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_polynomial_basis, make_fourier_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline, &
        sequential_basis_pipeline_t, make_sequential_basis_pipeline
    use fortml_tree, only: decision_stump_t, gradient_boosting_regressor_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_pipeline_products(failures)
    call test_sequential_pipeline_products(failures)
    call test_pipeline_refusals(failures)
    call test_stump_oracle(failures)
    call test_boosting(failures)
    call test_tree_nonfinite_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " pipeline/tree test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_pipeline_products(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial, fourier
        type(basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), x_dot(4, 2)
        real(dp) :: x_bar(4, 2), lhs, rhs, h
        real(dp), allocatable :: phi(:, :), phi_dot(:, :), phi_plus(:, :)
        real(dp), allocatable :: phi_minus(:, :), u(:, :)
        real(dp), allocatable :: theta(:), theta_dot(:), theta_plus(:), theta_minus(:)
        real(dp), allocatable :: theta_bar(:)
        integer :: n_features

        x = reshape([0.1_dp, -0.2_dp, 0.4_dp, 0.3_dp, &
            -0.5_dp, 0.8_dp, 0.7_dp, -0.6_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, &
            0.5_dp, 0.6_dp, -0.2_dp, 0.3_dp], shape(x_dot))
        polynomial = make_polynomial_basis(2, 2, status, include_intercept=.true.)
        fourier = make_fourier_basis(2, reshape([1.2_dp, 0.7_dp], [1, 2]), &
            status, include_intercept=.true.)
        pipeline = make_basis_pipeline(2, status)
        call pipeline%append(polynomial, status)
        call pipeline%append(fourier, status)
        call pipeline%fit(x, status)
        n_features = pipeline%feature_count()
        allocate(phi(4, n_features), phi_dot(4, n_features))
        allocate(phi_plus(4, n_features), phi_minus(4, n_features), &
            u(4, n_features))
        allocate(theta( pipeline%parameter_count()))
        allocate(theta_dot(size(theta)), theta_bar(size(theta)))
        theta = pipeline%parameters()
        theta_dot = [0.17_dp, -0.23_dp]
        call pipeline%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        h = 1.0e-6_dp
        theta_plus = theta + h*theta_dot
        theta_minus = theta - h*theta_dot
        call pipeline%set_parameters(theta_plus, status)
        call pipeline%transform(x + h*x_dot, phi_plus, status)
        call pipeline%set_parameters(theta_minus, status)
        call pipeline%transform(x - h*x_dot, phi_minus, status)
        call pipeline%set_parameters(theta, status)
        if (.not. status_ok(status) .or. pipeline%stage_count() /= 2 .or. &
            n_features /= polynomial%feature_count() + fourier%feature_count() .or. &
            .not. pipeline%is_fitted() .or. &
            maxval(abs(phi_dot - (phi_plus - phi_minus)/(2.0_dp*h))) > 3.0e-9_dp) then
            write (error_unit, '(a)') "FAIL [pipeline] value/JVP or shape"
            failures = failures + 1
        end if

        call fill_cotangent(u)
        call pipeline%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*phi_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 3.0e-11_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [pipeline] VJP identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_pipeline_products

    subroutine test_sequential_pipeline_products(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial, fourier
        type(sequential_basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 2), x_dot(4, 2), x_bar(4, 2)
        real(dp) :: lhs, rhs, h
        real(dp), allocatable :: y(:, :), y_dot(:, :), y_plus(:, :), y_minus(:, :)
        real(dp), allocatable :: u(:, :), theta(:), theta_dot(:)
        real(dp), allocatable :: theta_plus(:), theta_minus(:), theta_bar(:)
        integer :: n_features, n_parameters

        x = reshape([0.1_dp, -0.2_dp, 0.4_dp, 0.3_dp, &
            -0.5_dp, 0.8_dp, 0.7_dp, -0.6_dp], shape(x))
        x_dot = reshape([-0.3_dp, 0.2_dp, 0.1_dp, -0.4_dp, &
            0.5_dp, 0.6_dp, -0.2_dp, 0.3_dp], shape(x_dot))
        polynomial = make_polynomial_basis(2, 2, status, include_intercept=.true.)
        fourier = make_fourier_basis(5, reshape([1.1_dp, 0.8_dp, 0.6_dp, &
            1.4_dp, 0.9_dp], [1, 5]), status)
        pipeline = make_sequential_basis_pipeline(2, status)
        call pipeline%append(polynomial, status)
        call pipeline%append(fourier, status)
        call pipeline%fit(x, status)
        n_features = pipeline%feature_count()
        allocate(y(4, n_features), y_dot(4, n_features), y_plus(4, n_features), &
            y_minus(4, n_features), u(4, n_features))
        n_parameters = pipeline%parameter_count()
        allocate(theta(n_parameters))
        allocate(theta_dot(n_parameters), theta_plus(n_parameters), &
            theta_minus(n_parameters), theta_bar(n_parameters))
        theta = pipeline%parameters()
        theta_dot = 0.0_dp
        theta_dot(1) = 0.13_dp
        theta_dot(3) = -0.21_dp
        call pipeline%jvp(x, theta_dot, x_dot, y, y_dot, status)
        h = 1.0e-6_dp
        theta_plus = theta + h*theta_dot
        theta_minus = theta - h*theta_dot
        call pipeline%set_parameters(theta_plus, status)
        call pipeline%transform(x + h*x_dot, y_plus, status)
        call pipeline%set_parameters(theta_minus, status)
        call pipeline%transform(x - h*x_dot, y_minus, status)
        call pipeline%set_parameters(theta, status)
        if (.not. status_ok(status) .or. pipeline%stage_count() /= 2 .or. &
            .not. pipeline%is_fitted() .or. maxval(abs(y_dot - &
            (y_plus - y_minus)/(2.0_dp*h))) > 4.0e-9_dp) then
            write (error_unit, '(a)') &
                "FAIL [sequential pipeline] value/JVP or shape"
            failures = failures + 1
        end if

        call fill_cotangent(u)
        call pipeline%vjp(x, u, theta_bar, x_bar, status)
        lhs = sum(u*y_dot)
        rhs = sum(theta_bar*theta_dot) + sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 5.0e-10_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [sequential pipeline] VJP identity=", abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_sequential_pipeline_products

    subroutine test_pipeline_refusals(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial
        type(basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 2), phi(2, 3)

        x = reshape([0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp], shape(x))
        polynomial = make_polynomial_basis(2, 2, status)
        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(polynomial, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. pipeline%stage_count() /= 0) then
            write (error_unit, '(a)') "FAIL [pipeline] mismatched input refusal"
            failures = failures + 1
        end if
        pipeline = make_basis_pipeline(2, status)
        call pipeline%transform(x, phi, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [pipeline] empty transform refusal"
            failures = failures + 1
        end if
    end subroutine test_pipeline_refusals

    subroutine test_stump_oracle(failures)
        integer, intent(inout) :: failures
        type(decision_stump_t) :: stump
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 1), y(6), query(4, 1), prediction(4), x_dot(4, 1)
        real(dp) :: prediction_dot(4)

        x(:, 1) = [0.0_dp, 0.5_dp, 1.0_dp, 2.0_dp, 2.5_dp, 3.0_dp]
        y = [0.0_dp, 0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp, 10.0_dp]
        call stump%fit(x, y, status, min_samples_leaf=1)
        query(:, 1) = [-0.5_dp, 0.75_dp, 1.75_dp, 2.75_dp]
        call stump%predict(query, prediction, status)
        x_dot(:, 1) = [0.2_dp, -0.1_dp, 0.3_dp, -0.2_dp]
        call stump%jvp(query, x_dot, prediction, prediction_dot, status)
        if (.not. status_ok(status) .or. maxval(abs(prediction - &
            [0.0_dp, 0.0_dp, 10.0_dp, 10.0_dp])) > 1.0e-12_dp .or. &
            maxval(abs(prediction_dot)) > 1.0e-14_dp .or. &
            stump%split_feature() /= 1 .or. stump%split_threshold() <= 1.0_dp .or. &
            stump%split_threshold() >= 2.0_dp) then
            write (error_unit, '(a)') "FAIL [stump] independent split oracle"
            failures = failures + 1
        end if
    end subroutine test_stump_oracle

    subroutine test_boosting(failures)
        integer, intent(inout) :: failures
        type(gradient_boosting_regressor_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(8, 1), y(8), prediction(8), baseline, mse_model, mse_base

        x(:, 1) = [0.0_dp, 0.2_dp, 0.4_dp, 0.6_dp, 1.4_dp, 1.6_dp, 1.8_dp, 2.0_dp]
        y = [0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 4.0_dp, 4.0_dp, 4.0_dp, 4.0_dp]
        baseline = sum(y)/real(size(y), dp)
        mse_base = sum((y - baseline)**2)/real(size(y), dp)
        call model%fit(x, y, status, n_estimators=4, learning_rate=0.5_dp)
        call model%predict(x, prediction, status)
        mse_model = sum((y - prediction)**2)/real(size(y), dp)
        if (.not. status_ok(status) .or. model%estimator_count() /= 4 .or. &
            model%input_count() /= 1 .or. mse_model >= mse_base) then
            write (error_unit, '(a)') "FAIL [boosting] residual reduction oracle"
            failures = failures + 1
        end if
    end subroutine test_boosting

    subroutine test_tree_nonfinite_refusals(failures)
        integer, intent(inout) :: failures
        type(decision_stump_t) :: stump
        type(gradient_boosting_regressor_t) :: booster
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), y(4), query(2, 1), prediction(2), prediction_dot(2)
        real(dp) :: x_dot(2, 1)
        real(dp) :: nan_value, inf_value

        nan_value = ieee_value(0.0_dp, ieee_quiet_nan)
        inf_value = ieee_value(0.0_dp, ieee_positive_inf)
        x(:, 1) = [0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp]
        y = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
        query(:, 1) = [0.25_dp, 2.25_dp]
        x_dot(:, 1) = [0.1_dp, -0.2_dp]

        x(2, 1) = nan_value
        call stump%fit(x, y, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [tree refusal] NaN fit"
            failures = failures + 1
        end if
        x(2, 1) = 1.0_dp
        y(3) = inf_value
        call stump%fit(x, y, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [tree refusal] Inf target fit"
            failures = failures + 1
        end if

        y = [0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp]
        call stump%fit(x, y, status)
        query(1, 1) = nan_value
        call stump%predict(query, prediction, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [tree refusal] NaN prediction"
            failures = failures + 1
        end if
        query(1, 1) = 0.25_dp
        x_dot(2, 1) = inf_value
        call stump%jvp(query, x_dot, prediction, prediction_dot, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [tree refusal] Inf tangent"
            failures = failures + 1
        end if

        x(3, 1) = nan_value
        call booster%fit(x, y, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [boosting refusal] NaN fit"
            failures = failures + 1
        end if
        x(3, 1) = 2.0_dp
        call booster%fit(x, y, status, n_estimators=2)
        query(1, 1) = inf_value
        call booster%predict(query, prediction, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') "FAIL [boosting refusal] Inf prediction"
            failures = failures + 1
        end if
    end subroutine test_tree_nonfinite_refusals

    subroutine fill_cotangent(u)
        real(dp), intent(out) :: u(:, :)
        integer :: i, j

        do j = 1, size(u, 2)
            do i = 1, size(u, 1)
                u(i, j) = 0.17_dp*real(i, dp) - 0.11_dp*real(j, dp)
            end do
        end do
    end subroutine fill_cotangent

end program test_pipeline_tree
