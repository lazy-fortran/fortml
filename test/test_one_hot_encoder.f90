program test_one_hot_encoder
    use fortml_one_hot_encoder, only: one_hot_encoder_t, &
        ONE_HOT_UNKNOWN_IGNORE, ONE_HOT_MISSING_IGNORE
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_sorted_encoding(failures)
    call test_missing_and_drop(failures)
    call test_refusals_and_derivative_boundary(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL one-hot encoder cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS one-hot encoder independent behavioral oracles"

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

    subroutine test_sorted_encoding(failures)
        integer, intent(inout) :: failures
        type(one_hot_encoder_t) :: encoder
        type(fortnum_status_t) :: status
        integer :: x(4, 2), categories(5), category_offsets(3), output_offsets(3)
        real(dp) :: transformed(4, 5), expected(4, 5)

        x(:, 1) = [2, 1, 2, 3]
        x(:, 2) = [20, 10, 20, 10]
        call encoder%fit(x, status, handle_unknown="ignore")
        call check(status_ok(status), "sorted fit", failures)
        call check(encoder%fitted(), "fitted flag", failures)
        categories = encoder%categories()
        category_offsets = encoder%category_offsets()
        output_offsets = encoder%output_offsets()
        call check(all(categories == [1, 2, 3, 10, 20]), &
            "ascending category metadata", failures)
        call check(all(category_offsets == [1, 4, 6]), &
            "packed category offsets", failures)
        call check(all(output_offsets == [1, 4, 6]), &
            "packed output offsets", failures)
        call check(encoder%feature_category_count(1) == 3 .and. &
            encoder%feature_category_count(2) == 2, "category counts", failures)
        call check(encoder%output_count() == 5, "output count", failures)
        call check(encoder%unknown_policy() == ONE_HOT_UNKNOWN_IGNORE, &
            "unknown policy metadata", failures)

        call encoder%transform(x, transformed, status)
        call check(status_ok(status), "transform", failures)
        expected = 0.0_dp
        expected(1, 2) = 1.0_dp
        expected(1, 5) = 1.0_dp
        expected(2, 1) = 1.0_dp
        expected(2, 4) = 1.0_dp
        expected(3, 2) = 1.0_dp
        expected(3, 5) = 1.0_dp
        expected(4, 3) = 1.0_dp
        expected(4, 4) = 1.0_dp
        call check(maxval(abs(transformed - expected)) < 1.0e-14_dp, &
            "one-hot value oracle", failures)
    end subroutine test_sorted_encoding

    subroutine test_missing_and_drop(failures)
        integer, intent(inout) :: failures
        type(one_hot_encoder_t) :: ignored, categorical, dropped
        type(fortnum_status_t) :: status
        integer :: x(4, 1), query(3, 1), categories(3), x_no_missing(3, 1)
        integer :: missing_query(1, 1)
        real(dp) :: transformed(3, 2), expected(3, 2)
        real(dp) :: categorical_output(4, 3), categorical_expected(4, 3)
        real(dp) :: dropped_output(4, 2), absent_missing_output(1, 3)
        real(dp) :: dropped_expected(4, 2)

        x(:, 1) = [3, -99, 1, 3]
        call ignored%fit(x, status, missing_value=-99, &
            handle_missing="ignore", handle_unknown="ignore")
        call check(status_ok(status), "missing-ignore fit", failures)
        call check(ignored%missing_policy() == ONE_HOT_MISSING_IGNORE, &
            "missing-ignore policy metadata", failures)
        query(:, 1) = [-99, 7, 1]
        call ignored%transform(query, transformed, status)
        call check(status_ok(status), "missing/unknown ignore transform", failures)
        expected = 0.0_dp
        expected(3, 1) = 1.0_dp
        call check(maxval(abs(transformed - expected)) < 1.0e-14_dp, &
            "missing/unknown all-zero oracle", failures)

        call categorical%fit(x, status, missing_value=-99, &
            handle_missing="category", handle_unknown="error")
        call check(status_ok(status), "missing-category fit", failures)
        categories = categorical%categories()
        call check(all(categories == [-99, 1, 3]), &
            "missing category sorted metadata", failures)
        call categorical%transform(x, categorical_output, status)
        call check(status_ok(status), "missing-category transform", failures)
        categorical_expected = 0.0_dp
        categorical_expected(1, 3) = 1.0_dp
        categorical_expected(2, 1) = 1.0_dp
        categorical_expected(3, 2) = 1.0_dp
        categorical_expected(4, 3) = 1.0_dp
        call check(maxval(abs(categorical_output - categorical_expected)) < &
            1.0e-14_dp, &
            "missing category value oracle", failures)

        call dropped%fit(x, status, missing_value=-99, &
            handle_missing="category", drop_first=.true.)
        call check(status_ok(status), "drop-first fit", failures)
        call check(dropped%output_count() == 2, "drop-first output count", failures)
        call check(dropped%feature_output_count(1) == 2, &
            "drop-first feature output count", failures)
        call dropped%transform(x, dropped_output, status)
        call check(status_ok(status), "drop-first transform", failures)
        dropped_expected = 0.0_dp
        dropped_expected(1, 2) = 1.0_dp
        dropped_expected(3, 1) = 1.0_dp
        dropped_expected(4, 2) = 1.0_dp
        call check(maxval(abs(dropped_output - dropped_expected)) < 1.0e-14_dp, &
            "drop-first value oracle", failures)

        x_no_missing(:, 1) = [1, 3, 1]
        call categorical%fit(x_no_missing, status, missing_value=-99, &
            handle_missing="category")
        call check(status_ok(status), "absent-missing category fit", failures)
        call check(all(categorical%categories() == [-99, 1, 3]), &
            "absent-missing category metadata", failures)
        missing_query(1, 1) = -99
        call categorical%transform(missing_query, absent_missing_output, status)
        call check(status_ok(status), "absent-missing category transform", failures)
        call check(abs(absent_missing_output(1, 1) - 1.0_dp) < 1.0e-14_dp .and. &
            maxval(abs(absent_missing_output(1, 2:3))) < 1.0e-14_dp, &
            "absent-missing category value oracle", failures)
    end subroutine test_missing_and_drop

    subroutine test_refusals_and_derivative_boundary(failures)
        integer, intent(inout) :: failures
        type(one_hot_encoder_t) :: encoder, error_encoder
        type(fortnum_status_t) :: status
        integer :: x(3, 1), query(1, 1), bad_x(3, 2)
        real(dp) :: transformed(3, 2), output_bar(3, 2), input_bar(3, 1)
        real(dp) :: x_dot(3, 1), transformed_dot(3, 2)

        x(:, 1) = [1, 2, 1]
        call error_encoder%fit(reshape([1, -9, 1], [3, 1]), status, &
            missing_value=-9)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "missing error fit refusal", failures)
        call encoder%fit(x, status)
        call check(status_ok(status), "refusal fixture fit", failures)
        query(1, 1) = 9
        call encoder%transform(query, transformed(1:1, :), status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "unknown error transform refusal", failures)
        x_dot = 1.0_dp
        call encoder%transform_jvp(x, x_dot, transformed_dot, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "categorical JVP refusal", failures)
        output_bar = 1.0_dp
        call encoder%transform_vjp(x, output_bar, input_bar, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "categorical VJP refusal", failures)
        bad_x = 1
        call encoder%transform_jvp(bad_x(:, 1:2), x_dot, transformed_dot, status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "JVP shape refusal", failures)
    end subroutine test_refusals_and_derivative_boundary

end program test_one_hot_encoder
