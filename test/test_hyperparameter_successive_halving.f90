program test_hyperparameter_successive_halving
    !! Independent behavioral oracle for deterministic multi-fidelity HPO.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_hyperparameter_search, only: &
        hyperparameter_resource_objective_t, hyperparameter_search_result_t, &
        hyperparameter_successive_halving_search, &
        hyperparameter_lbfgsb_resource_search
    implicit none

    type :: quadratic_resource_fixture_t
        real(dp) :: target(2) = [0.45_dp, -0.35_dp]
    end type quadratic_resource_fixture_t

    type(quadratic_resource_fixture_t), target :: fixture
    type(hyperparameter_resource_objective_t) :: objective
    type(hyperparameter_search_result_t) :: halving, refined
    type(fortnum_status_t) :: status
    type(fortml_device_t) :: cuda
    real(dp) :: lower(2), upper(2)
    integer :: failures

    failures = 0
    lower = [-1.0_dp, -1.0_dp]
    upper = [1.0_dp, 1.0_dp]
    call objective%initialize_context(2, fixture, quadratic_resource_callback, status)
    call check(status_ok(status), "resource objective initialization", failures)

    call hyperparameter_successive_halving_search(objective, lower, upper, 32, 1, 8, &
        2, 20260809_int64, halving, status)
    call check(status_ok(status), "successive halving status", failures)
    if (status_ok(status)) then
        call check(halving%converged, "successive halving converged", failures)
        call check(halving%rung_count == 4, "resource schedule has four rungs", failures)
        call check(halving%evaluations == 60_int64, &
            "successive halving evaluates each retained rung", failures)
        call check(halving%best_resource == 8, "best candidate reaches max resource", failures)
        call check(halving%best_value < 0.60_dp, &
            "seeded survivor beats the independent quadratic bound", failures)
        call check(all(halving%best_parameters >= lower) .and. &
            all(halving%best_parameters <= upper), &
            "survivor remains in the closed parameter box", failures)
    end if

    call hyperparameter_lbfgsb_resource_search(objective, halving%best_parameters, lower, &
        upper, 8, refined, status)
    call check(status_ok(status), "fixed-resource L-BFGS-B status", failures)
    if (status_ok(status)) then
        call check(refined%converged, "fixed-resource L-BFGS-B converged", failures)
        call check(maxval(abs(refined%best_parameters - fixture%target)) < 2.0e-7_dp, &
            "FortOpt refinement reaches the analytic resource optimum", failures)
        call check(abs(refined%best_value - 0.125_dp) < 2.0e-12_dp, &
            "resource objective value matches the analytic optimum", failures)
        call check(refined%best_resource == 8, "refinement records its resource", failures)
    end if

    call hyperparameter_successive_halving_search(objective, lower, upper, 4, 1, 8, 1, &
        1_int64, halving, status)
    call check(.not. status_ok(status), "reduction factor one is refused", failures)

    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.
    call hyperparameter_successive_halving_search(objective, lower, upper, 4, 1, 8, 2, &
        1_int64, halving, status, device=cuda)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "resident CUDA search is a typed refusal", failures)

    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL successive-halving cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS successive-halving and resource L-BFGS-B independent oracle"

contains

    subroutine quadratic_resource_callback(context, parameters, resource, value, gradient, &
            status)
        class(*), intent(inout) :: context
        real(dp), intent(in) :: parameters(:)
        integer, intent(in) :: resource
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select type (fixture => context)
            type is (quadratic_resource_fixture_t)
                value = sum((parameters - fixture%target)**2) + &
                    1.0_dp/real(resource, dp)
                gradient = 2.0_dp*(parameters - fixture%target)
                call status_set(status, FORTNUM_OK, "")
            class default
                value = huge(1.0_dp)
                gradient = 0.0_dp
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "quadratic fixture: context has wrong type")
        end select
    end subroutine quadratic_resource_callback

    subroutine check(condition, description, failure_count)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failure_count

        if (.not. condition) then
            failure_count = failure_count + 1
            write (error_unit, '(a)') "  FAIL [successive-halving] "//description
        end if
    end subroutine check

end program test_hyperparameter_successive_halving
