program test_logistic_regression
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use fortml_logistic_regression, only: logistic_regression_t
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_intercept_only_oracle(failures)
    call test_symmetric_coefficient_oracle(failures)
    call test_weighted_intercept_oracle(failures)
    call test_refusals(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL logistic regression cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS logistic regression independent behavioral oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

    subroutine test_intercept_only_oracle(failures)
        integer, intent(inout) :: failures
        type(logistic_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), scores(4), probabilities(4, 2)
        real(dp) :: expected_score
        real(dp), allocatable :: coefficients(:)
        integer :: labels(4), prediction(4), classes(2)

        x = 0.0_dp
        labels = [-7, -7, -7, 42]
        expected_score = log(1.0_dp/3.0_dp)
        call model%fit(x, labels, status, l2=0.0_dp, tolerance=1.0e-10_dp)
        call check(status_ok(status), "intercept-only fit", failures)
        if (.not. status_ok(status)) return

        call model%decision_function(x, scores, status)
        call model%predict_proba(x, probabilities, status)
        call model%predict(x, prediction, status)
        coefficients = model%coefficients()
        classes = model%classes()
        call check(status_ok(status), "intercept-only predictions", failures)
        call check(model%fitted(), "fitted flag", failures)
        call check(model%feature_count() == 1, "feature count", failures)
        call check(size(coefficients) == 1, "coefficient shape", failures)
        call check(abs(coefficients(1)) < 1.0e-11_dp, &
            "zero coefficient for a zero feature", failures)
        call check(abs(model%intercept_value() - expected_score) < 2.0e-8_dp, &
            "analytic log-odds intercept", failures)
        call check(maxval(abs(scores - expected_score)) < 2.0e-8_dp, &
            "analytic decision score", failures)
        call check(maxval(abs(probabilities(:, 2) - 0.25_dp)) < 5.0e-9_dp, &
            "empirical class probability", failures)
        call check(maxval(abs(sum(probabilities, dim=2) - 1.0_dp)) < 1.0e-15_dp, &
            "probability normalization", failures)
        call check(all(classes == [-7, 42]), "deterministic class order", failures)
        call check(all(prediction == -7), "classification threshold", failures)
    end subroutine test_intercept_only_oracle

    subroutine test_symmetric_coefficient_oracle(failures)
        integer, intent(inout) :: failures
        type(logistic_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp), parameter :: penalty = 0.2_dp
        real(dp) :: x(4, 1), probabilities(4, 2), expected_coefficient
        real(dp), allocatable :: coefficients(:)
        integer :: labels(4), prediction(4)

        x(:, 1) = [-2.0_dp, -1.0_dp, 1.0_dp, 2.0_dp]
        labels = [3, 3, 11, 11]
        expected_coefficient = solve_score_equation(x(:, 1), penalty)

        call model%fit(x, labels, status, l2=penalty, tolerance=1.0e-8_dp)
        call check(status_ok(status), "symmetric fit", failures)
        if (.not. status_ok(status)) return
        call model%predict_proba(x, probabilities, status)
        call model%predict(x, prediction, status)
        coefficients = model%coefficients()

        call check(status_ok(status), "symmetric predictions", failures)
        call check(abs(coefficients(1) - expected_coefficient) < 5.0e-8_dp, &
            "independent score-equation coefficient", failures)
        call check(abs(model%intercept_value()) < 2.0e-10_dp, &
            "symmetric zero intercept", failures)
        call check(maxval(abs(probabilities(:, 2) + &
            probabilities(4:1:-1, 2) - 1.0_dp)) < 2.0e-10_dp, &
            "symmetric probabilities", failures)
        call check(all(prediction == labels), "training classification", failures)
    end subroutine test_symmetric_coefficient_oracle

    subroutine test_weighted_intercept_oracle(failures)
        integer, intent(inout) :: failures
        type(logistic_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(4, 1), probabilities(4, 2)
        real(dp), parameter :: weights(4) = [1.0_dp, 1.0_dp, 1.0_dp, 3.0_dp]
        integer :: labels(4)

        x = 0.0_dp
        labels = [-7, -7, -7, 42]
        call model%fit(x, labels, status, l2=0.0_dp, sample_weight=weights, &
            tolerance=1.0e-10_dp)
        call check(status_ok(status), "weighted intercept fit", failures)
        if (.not. status_ok(status)) return
        call model%predict_proba(x, probabilities, status)
        call check(status_ok(status), "weighted intercept prediction", failures)
        call check(abs(model%intercept_value()) < 2.0e-8_dp, &
            "weighted analytic intercept", failures)
        call check(maxval(abs(probabilities - 0.5_dp)) < 2.0e-8_dp, &
            "weighted empirical probability", failures)
    end subroutine test_weighted_intercept_oracle

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(logistic_regression_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), scores(3), bad_x(3, 1)
        integer :: one_class(3), three_classes(3), prediction(3)

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        one_class = 4
        three_classes = [1, 2, 3]
        call model%predict(x, prediction, status)
        call check(.not. status_ok(status), "predict before fit refusal", failures)
        call model%fit(x, one_class, status)
        call check(.not. status_ok(status), "one-class refusal", failures)
        call model%fit(x, three_classes, status)
        call check(.not. status_ok(status), "three-class refusal", failures)
        bad_x = x
        bad_x(2, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
        call model%fit(bad_x, [0, 0, 1], status)
        call check(.not. status_ok(status), "nonfinite input refusal", failures)
        call model%fit(x, [0, 0, 1], status, l2=-1.0_dp)
        call check(.not. status_ok(status), "negative penalty refusal", failures)
        call model%fit(x, [0, 0, 1], status, sample_weight=[0.0_dp, 0.0_dp, 0.0_dp])
        call check(.not. status_ok(status), "zero sample-weight refusal", failures)
        call model%fit(x, [0, 0, 1], status, fit_intercept=.false.)
        call check(status_ok(status), "no-intercept fit", failures)
        if (.not. status_ok(status)) return
        call model%decision_function(reshape([0.0_dp, 0.0_dp], [1, 2]), &
            scores(:1), status)
        call check(.not. status_ok(status), "feature-shape refusal", failures)
    end subroutine test_refusals

    real(dp) function solve_score_equation(x, penalty) result(root)
        real(dp), intent(in) :: x(:), penalty
        real(dp) :: lower, upper, middle, derivative
        real(dp) :: encoded(size(x))
        integer :: i, iteration

        encoded = 0.0_dp
        encoded(size(x)/2 + 1:) = 1.0_dp
        lower = 0.0_dp
        upper = 20.0_dp
        do iteration = 1, 200
            middle = 0.5_dp*(lower + upper)
            derivative = penalty*middle
            do i = 1, size(x)
                derivative = derivative + &
                    x(i)*(reference_sigmoid(middle*x(i)) - encoded(i))/real(size(x), dp)
            end do
            if (derivative > 0.0_dp) then
                upper = middle
            else
                lower = middle
            end if
        end do
        root = 0.5_dp*(lower + upper)
    end function solve_score_equation

    pure real(dp) function reference_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        probability = 0.5_dp*(1.0_dp + tanh(0.5_dp*value))
    end function reference_sigmoid

end program test_logistic_regression
