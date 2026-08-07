program test_xgboost_ranking
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_xgboost, only: xgboost_t, xgboost_options_t, &
        xgb_pairwise_loss, xgb_pairwise_derivatives
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_pairwise_finite_difference(failures)
    call test_group_isolation(failures)
    call test_two_item_ordering(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " xgboost ranking test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_pairwise_finite_difference(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: margin(3), target(3), plus(3), minus(3), gradient(3), hessian(3)
        integer :: group(3), i
        real(dp) :: loss_plus, loss_minus, numerical, error
        real(dp), parameter :: epsilon = 1.0e-6_dp

        margin = [0.2_dp, -0.1_dp, 0.3_dp]
        target = [1.0_dp, 0.0_dp, 0.0_dp]
        group = [1, 1, 2]
        call xgb_pairwise_derivatives(margin, target, group, gradient, hessian, status)
        if (status%code /= FORTNUM_OK) then
            write (error_unit, '(a)') "FAIL [xgb ranking] derivative status"
            failures = failures + 1
            return
        end if
        error = 0.0_dp
        do i = 1, 3
            plus = margin
            minus = margin
            plus(i) = plus(i) + epsilon
            minus(i) = minus(i) - epsilon
            call xgb_pairwise_loss(plus, target, group, loss_plus, status)
            call xgb_pairwise_loss(minus, target, group, loss_minus, status)
            numerical = (loss_plus - loss_minus)/(2.0_dp*epsilon)
            error = max(error, abs(numerical - gradient(i)))
        end do
        if (error > 2.0e-8_dp .or. abs(hessian(1) - hessian(2)) > 1.0e-14_dp .or. &
            hessian(3) /= 0.0_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb ranking] finite-difference gradient error ", error
            failures = failures + 1
        end if
    end subroutine test_pairwise_finite_difference

    subroutine test_group_isolation(failures)
        integer, intent(inout) :: failures
        type(fortnum_status_t) :: status
        real(dp) :: margin(4), target(4), gradient(4), hessian(4), loss
        integer :: group(4)
        real(dp) :: expected_gradient(4), expected_hessian(4)

        margin = 0.0_dp
        target = [1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp]
        group = [1, 1, 2, 2]
        expected_gradient = [-0.5_dp, 0.5_dp, 0.5_dp, -0.5_dp]
        expected_hessian = 0.25_dp
        call xgb_pairwise_derivatives(margin, target, group, gradient, hessian, status)
        call xgb_pairwise_loss(margin, target, group, loss, status)
        if (status%code /= FORTNUM_OK .or. maxval(abs(gradient - expected_gradient)) > &
            2.0e-14_dp .or. maxval(abs(hessian - expected_hessian)) > 2.0e-14_dp .or. &
            abs(loss - log(2.0_dp)) > 2.0e-14_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [xgb ranking] query isolation oracle ", &
                maxval(abs(gradient - expected_gradient))
            failures = failures + 1
        end if
    end subroutine test_group_isolation

    subroutine test_two_item_ordering(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2), prediction(2)
        integer :: group(2)

        x(:, 1) = [0.0_dp, 1.0_dp]
        target = [0.0_dp, 1.0_dp]
        group = [7, 7]
        options%n_estimators = 1
        options%max_depth = 1
        options%learning_rate = 1.0_dp
        options%l2 = 0.0_dp
        options%min_child_weight = 0.0_dp
        call model%fit_ranking(x, target, group, status, options)
        call model%predict(x, prediction, status)
        if (status%code /= FORTNUM_OK .or. trim(model%objective_name()) /= &
            "rank:pairwise" .or. prediction(2) <= prediction(1) .or. &
            abs(prediction(1) + 2.0_dp) > 2.0e-12_dp .or. &
            abs(prediction(2) - 2.0_dp) > 2.0e-12_dp) then
            write (error_unit, '(a,2es12.4)') &
                "FAIL [xgb ranking] two-item ordering oracle ", prediction
            failures = failures + 1
        end if
    end subroutine test_two_item_ordering

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), target(2)
        integer :: group(2)

        x(:, 1) = [0.0_dp, 1.0_dp]
        target = [0.0_dp, 1.0_dp]
        group = [1, 2]
        options%n_estimators = 1
        call model%fit_ranking(x, target, group, status, options)
        if (status%code /= FORTNUM_DOMAIN_ERROR) then
            write (error_unit, '(a)') &
                "FAIL [xgb ranking] singleton queries must be refused"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_xgboost_ranking
