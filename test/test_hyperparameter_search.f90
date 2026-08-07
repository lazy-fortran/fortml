program test_hyperparameter_search
    !! Independent quadratic oracle for grid and FortOpt L-BFGS-B search.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_hyperparameter_search, only: &
        hyperparameter_search_result_t, hyperparameter_grid_search, &
        hyperparameter_lbfgsb_search, hyperparameter_random_search
    implicit none

    type(objective_t) :: objective
    type(hyperparameter_search_result_t) :: grid_result, optimizer_result, random_result
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: lower(2), upper(2), initial(2)
    integer :: failures

    failures = 0
    lower = [-3.0_dp, -3.0_dp]
    upper = [3.0_dp, 3.0_dp]
    initial = [0.0_dp, 0.0_dp]
    call objective%initialize(2, quadratic_objective, status)
    call check(status_ok(status), "objective initialization", failures)

    call hyperparameter_grid_search(objective, lower, upper, [7, 7], &
        grid_result, status)
    call check(status_ok(status) .and. grid_result%converged .and. &
        grid_result%evaluations == 49 .and. &
        maxval(abs(grid_result%best_parameters - [2.0_dp, -1.0_dp])) < 1.0e-13_dp .and. &
        abs(grid_result%best_value) < 1.0e-13_dp, &
        "Cartesian grid finds exact quadratic minimum", failures)

    call hyperparameter_lbfgsb_search(objective, initial, lower, upper, &
        optimizer_result, status)
    call check(status_ok(status) .and. optimizer_result%converged .and. &
        maxval(abs(optimizer_result%best_parameters - [2.0_dp, -1.0_dp])) < 2.0e-7_dp .and. &
        optimizer_result%best_value < 1.0e-12_dp, &
        "L-BFGS-B reaches analytic quadratic minimum", failures)

    call hyperparameter_random_search(objective, lower, upper, 32, 1234_int64, &
        random_result, status)
    call check(status_ok(status) .and. random_result%converged .and. &
        random_result%evaluations == 32_int64 .and. &
        all(random_result%best_parameters >= lower) .and. &
        all(random_result%best_parameters <= upper), &
        "seeded random search stays within bounds", failures)
    call hyperparameter_random_search(objective, lower, upper, 32, 1234_int64, &
        optimizer_result, status)
    call check(status_ok(status) .and. &
        maxval(abs(optimizer_result%best_parameters - random_result%best_parameters)) < &
        1.0e-14_dp .and. abs(optimizer_result%best_value - random_result%best_value) < &
        1.0e-14_dp, "seeded random search is reproducible", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call hyperparameter_grid_search(objective, lower, upper, [3, 3], grid_result, &
        status, device=cuda)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "generic search refuses absent resident CUDA objective", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL hyperparameter search cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS hyperparameter grid/L-BFGS-B independent oracle"

contains

    subroutine quadratic_objective(x, value, gradient, status)
        real(dp), intent(in) :: x(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        value = sum((x - [2.0_dp, -1.0_dp])**2)
        gradient = 2.0_dp*(x - [2.0_dp, -1.0_dp])
        call status_set(status, FORTNUM_OK, "")
    end subroutine quadratic_objective

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [hyperparameter-search] "//description
        end if
    end subroutine check

end program test_hyperparameter_search
