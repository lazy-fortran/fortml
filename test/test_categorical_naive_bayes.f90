program test_categorical_naive_bayes
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortml_categorical_naive_bayes, only: categorical_naive_bayes_t
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures
    failures = 0
    call test_oracle(failures)
    call test_unknown_and_derivative_refusal(failures)
    if (failures /= 0) error stop "categorical naive Bayes tests failed"
    write (*, '(a)') "categorical naive Bayes tests passed"

contains

    subroutine test_oracle(failures)
        integer, intent(inout) :: failures
        type(categorical_naive_bayes_t) :: model
        type(fortnum_status_t) :: status
        integer :: x(6, 2), labels(6), query(2, 2), predicted(2)
        real(dp) :: probabilities(2, 2), expected(2, 2), priors(2)

        x(:, 1) = [1, 1, 2, 2, 1, 2]
        x(:, 2) = [10, 20, 10, 20, 20, 10]
        labels = [0, 0, 0, 1, 1, 1]
        call model%fit(x, labels, status, alpha=1.0_dp)
        call check(status_ok(status), "fit", failures)
        call check(model%category_count(1) == 2 .and. model%category_count(2) == 2, &
            "category counts", failures)
        query(1, :) = [1, 10]
        query(2, :) = [2, 20]
        call model%predict_proba(query, probabilities, status)
        expected(1, :) = [9.0_dp/13.0_dp, 4.0_dp/13.0_dp]
        expected(2, :) = [4.0_dp/13.0_dp, 9.0_dp/13.0_dp]
        call check(status_ok(status), "prediction status", failures)
        call check(maxval(abs(probabilities - expected)) < 2.0e-13_dp, &
            "independent categorical probability oracle", failures)
        call model%predict(query, predicted, status)
        call check(status_ok(status) .and. all(predicted == [0, 1]), &
            "predict labels", failures)
        priors = model%class_prior()
        call check(maxval(abs(priors - [0.5_dp, 0.5_dp])) < 2.0e-14_dp, &
            "class prior", failures)
    end subroutine test_oracle

    subroutine test_unknown_and_derivative_refusal(failures)
        integer, intent(inout) :: failures
        type(categorical_naive_bayes_t) :: rejecting, ignoring
        type(fortnum_status_t) :: status
        integer :: x(4, 1), labels(4), query(1, 1)
        real(dp) :: probabilities(1, 2), probabilities_dot(1, 2), x_dot(1, 1)

        x(:, 1) = [1, 1, 2, 2]
        labels = [0, 0, 1, 1]
        query(1, 1) = 99
        call rejecting%fit(x, labels, status)
        call rejecting%predict_proba(query, probabilities, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, "unknown category refusal", failures)
        call ignoring%fit(x, labels, status, handle_unknown=.true.)
        call ignoring%predict_proba(query, probabilities, status)
        call check(status_ok(status), "unknown category ignore", failures)
        call check(maxval(abs(probabilities(1, :) - [0.5_dp, 0.5_dp])) < 2.0e-14_dp, &
            "unknown category neutral likelihood", failures)
        x_dot = 0.0_dp
        call ignoring%predict_proba_jvp(query, x_dot, probabilities, probabilities_dot, &
            status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, "discrete JVP refusal", failures)
    end subroutine test_unknown_and_derivative_refusal

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [CategoricalNB] "//description
            failures = failures + 1
        end if
    end subroutine check

end program test_categorical_naive_bayes
