program test_xgboost_interaction_constraints
    !! Independent oracle for path-local XGBoost interaction constraints.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_path_constraint(failures)
    call test_invalid_metadata(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " xgboost interaction-constraint test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_path_constraint(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: unconstrained, constrained, restored
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(8, 3), y(8), unconstrained_prediction(8)
        real(real64) :: constrained_prediction(8), restored_prediction(8)
        character(*), parameter :: path = "test_xgboost_interaction_constraints.txt"

        ! Feature one separates the two large groups. Feature two only
        ! separates rows within each group. An unconstrained depth-two tree
        ! therefore has four leaves and reproduces y exactly. The independent
        ! interaction partition {feature 1}, {feature 2} forbids that second
        ! split, leaving the group means as the only admissible leaves.
        x(:, 1) = [0.0_real64, 0.0_real64, 0.0_real64, 0.0_real64, &
            1.0_real64, 1.0_real64, 1.0_real64, 1.0_real64]
        x(:, 2) = [0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64, &
            0.0_real64, 1.0_real64, 2.0_real64, 3.0_real64]
        x(:, 3) = 0.0_real64
        y = [0.0_real64, 0.0_real64, 4.0_real64, 4.0_real64, &
            10.0_real64, 10.0_real64, 14.0_real64, 14.0_real64]
        options = xgboost_options_t()
        options%n_estimators = 1
        options%max_depth = 2
        options%learning_rate = 1.0_real64
        options%l2 = 0.0_real64
        options%min_child_weight = 0.0_real64
        call unconstrained%fit_regression(x, y, status, options)
        call check(status_ok(status), "unconstrained fit", failures)
        call unconstrained%predict(x, unconstrained_prediction, status)
        call check(status_ok(status) .and. unconstrained%tree_node_count(1) == 7 .and. &
            maxval(abs(unconstrained_prediction - y)) < 2.0e-13_real64, &
            "unconstrained four-leaf oracle", failures)

        options%interaction_groups = [1, 2, 0]
        call constrained%fit_regression(x, y, status, options)
        call constrained%predict(x, constrained_prediction, status)
        call check(status_ok(status) .and. constrained%tree_node_count(1) == 3 .and. &
            constrained%interaction_group(1) == 1 .and. &
            constrained%interaction_group(2) == 2 .and. &
            constrained%interaction_group(3) == 0 .and. &
            maxval(abs(constrained_prediction - &
                [2.0_real64, 2.0_real64, 2.0_real64, 2.0_real64, &
                 12.0_real64, 12.0_real64, 12.0_real64, 12.0_real64])) < &
            2.0e-13_real64, "path-local interaction oracle", failures)

        call constrained%save_text(path, status)
        call check(status_ok(status), "interaction metadata save", failures)
        call restored%load_text(path, status)
        call restored%predict(x, restored_prediction, status)
        call check(status_ok(status) .and. restored%interaction_group(1) == 1 .and. &
            restored%interaction_group(2) == 2 .and. &
            maxval(abs(restored_prediction - constrained_prediction)) < 2.0e-13_real64, &
            "interaction metadata round trip", failures)
        call delete_file(path)
    end subroutine test_path_constraint

    subroutine test_invalid_metadata(failures)
        integer, intent(inout) :: failures
        type(xgboost_t) :: model
        type(xgboost_options_t) :: options
        type(fortnum_status_t) :: status
        real(real64) :: x(4, 2), y(4)

        x = reshape([0.0_real64, 0.0_real64, 1.0_real64, 1.0_real64, &
            0.0_real64, 1.0_real64, 0.0_real64, 1.0_real64], shape(x))
        y = [0.0_real64, 1.0_real64, 1.0_real64, 2.0_real64]
        options = xgboost_options_t()
        options%n_estimators = 1
        options%interaction_groups = [1]
        call model%fit_regression(x, y, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "wrong interaction-group length refusal", failures)
        options%interaction_groups = [1, -1]
        call model%fit_regression(x, y, status, options)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "negative interaction-group refusal", failures)
    end subroutine test_invalid_metadata

    subroutine delete_file(path)
        character(*), intent(in) :: path
        integer :: unit, ios

        open(newunit=unit, file=path, status="old", action="read", &
            iostat=ios)
        if (ios == 0) close(unit, status="delete")
    end subroutine delete_file

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            write (error_unit, '(a)') "FAIL [xgb interaction] "//label
            failures = failures + 1
        end if
    end subroutine check

end program test_xgboost_interaction_constraints
