program test_linear_sgd
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_linear_sgd, only: linear_sgd_options_t, linear_sgd_regression_t, &
        linear_sgd_classifier_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_regression_recurrence(failures)
    call test_partial_fit_and_average(failures)
    call test_classifier_recurrence(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " linear-SGD test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_regression_recurrence(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(3, 1), y(3), prediction(3), expected(2)
        type(linear_sgd_options_t) :: options
        type(linear_sgd_regression_t) :: model
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        y = [1.0_dp, 1.0_dp, 3.0_dp]
        options%epochs = 1
        options%batch_size = 2
        options%learning_rate = 0.1_dp
        options%fit_intercept = .true.
        call model%fit(x, y, status, options)
        call check(status_ok(status), "regression fit", failures)

        ! Independent mean-gradient recurrence, first batch [-1,0], then [1].
        expected = [0.395_dp, 0.245_dp]
        call model%predict(x, prediction, status)
        call check(status_ok(status), "regression prediction", failures)
        call check(maxval(abs(model%parameters() - expected)) < 2.0e-14_dp, &
            "regression independent recurrence", failures)
        call check(maxval(abs(prediction - [1.0_dp, 1.0_dp, 3.0_dp])) > 0.0_dp, &
            "regression moved from zero", failures)
    end subroutine test_regression_recurrence

    subroutine test_partial_fit_and_average(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(4, 1), y(4)
        real(dp), allocatable :: left_parameters(:), right_parameters(:)
        type(linear_sgd_options_t) :: options
        type(linear_sgd_regression_t) :: direct, streamed, averaged
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp, 2.0_dp]
        y = [1.0_dp, 1.0_dp, 3.0_dp, 5.0_dp]
        options%epochs = 2
        options%batch_size = 2
        options%learning_rate = 0.05_dp
        options%shuffle = .true.
        options%shuffle_seed = 29
        call direct%fit(x, y, status, options)
        call check(status_ok(status), "direct fit", failures)
        call streamed%partial_fit(x, y, status, options)
        call check(status_ok(status), "first partial_fit", failures)
        call streamed%partial_fit(x, y, status, options)
        call check(status_ok(status), "second partial_fit", failures)
        left_parameters = direct%parameters()
        right_parameters = streamed%parameters()
        call check(maxval(abs(left_parameters-right_parameters)) < 2.0e-14_dp, &
            "partial_fit continuation matches fit", failures)
        call check(streamed%update_count() == 4, "partial_fit update count", failures)

        options%average = .true.
        options%shuffle = .false.
        options%epochs = 1
        call averaged%fit(x, y, status, options)
        call check(status_ok(status), "averaged fit", failures)
        call check(maxval(abs(averaged%parameters())) < 1.0e3_dp, &
            "averaged parameters finite", failures)
    end subroutine test_partial_fit_and_average

    subroutine test_classifier_recurrence(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 1), probabilities(2, 2), expected_score
        integer :: labels(2), prediction(2)
        type(linear_sgd_options_t) :: options
        type(linear_sgd_classifier_t) :: model
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 1.0_dp]
        labels = [-2, 4]
        options%epochs = 1
        options%batch_size = 2
        options%learning_rate = 0.2_dp
        call model%fit(x, labels, status, options)
        call check(status_ok(status), "classifier fit", failures)
        ! At zero parameters, the batch gradient is [-0.5, 0.0], so w=0.1.
        expected_score = 0.1_dp
        call model%decision_function(x, probabilities(:, 1), status)
        call check(status_ok(status), "classifier decision", failures)
        call check(maxval(abs(probabilities(:, 1)-[-expected_score, expected_score])) < 2.0e-14_dp, &
            "classifier independent recurrence", failures)
        call model%predict_proba(x, probabilities, status)
        call check(status_ok(status), "classifier probabilities", failures)
        call check(abs(sum(probabilities(1, :))-1.0_dp) < 2.0e-15_dp, &
            "classifier probability normalization", failures)
        call model%predict(x, prediction, status)
        call check(status_ok(status) .and. all(prediction == labels), &
            "classifier labels", failures)
    end subroutine test_classifier_recurrence

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 1), y(2), output(2, 1)
        integer :: labels(2)
        type(linear_sgd_options_t) :: options
        type(linear_sgd_regression_t) :: regression
        type(linear_sgd_classifier_t) :: classifier
        type(fortml_device_t) :: device
        type(fortnum_status_t) :: status

        x(:, 1) = [0.0_dp, 1.0_dp]
        y = [0.0_dp, 1.0_dp]
        labels = [0, 1]
        options%epochs = 1
        options%batch_size = 1
        call regression%fit(x, y, status, options)
        call check(status_ok(status), "refusal regression setup", failures)
        device%kind = FORTML_DEVICE_CUDA
        call regression%predict_device(device, x, output, status)
        call check(.not. status_ok(status), "CUDA regression refusal", failures)
        call classifier%fit(x, labels, status, options)
        call check(status_ok(status), "refusal classifier setup", failures)
        call classifier%partial_fit(x, [0, 2], status, options, classes=[0, 1])
        call check(.not. status_ok(status), "unknown partial_fit label refusal", failures)
    end subroutine test_refusals

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            write (error_unit, '(a)') "FAIL ["//trim(description)//"]"
            failures = failures + 1
        end if
    end subroutine check

end program test_linear_sgd
