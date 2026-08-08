program test_pipeline_persistence
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis, make_polynomial_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortml_pipeline_persistence, only: basis_pipeline_state_t, &
        capture_basis_pipeline_state, restore_basis_pipeline_state, &
        save_basis_pipeline_text, load_basis_pipeline_text, &
        save_basis_pipeline_device
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_round_trip(failures)
    call test_invalid_input_is_transactional(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " pipeline persistence test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS pipeline persistence independent oracle"

contains

    subroutine make_fixture(pipeline, x, status)
        type(basis_pipeline_t), intent(out) :: pipeline
        real(dp), intent(out) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(basis_map_t) :: polynomial, fourier
        real(dp) :: frequencies(1, 1)
        character(len=16) :: names(1)

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp], shape(x))
        frequencies = reshape([0.8_dp], shape(frequencies))
        names = [character(len=16) :: "time"]
        polynomial = make_polynomial_basis(1, 2, status)
        if (.not. status_ok(status)) return
        fourier = make_fourier_basis(1, frequencies, status)
        if (.not. status_ok(status)) return
        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(polynomial, status, name="trend")
        if (.not. status_ok(status)) return
        call pipeline%append(fourier, status, name="seasonal")
        if (.not. status_ok(status)) return
        call pipeline%set_input_schema(names, status)
        if (.not. status_ok(status)) return
        call pipeline%fit(x, status)
    end subroutine make_fixture

    subroutine test_round_trip(failures)
        integer, intent(inout) :: failures
        type(basis_pipeline_t) :: original, restored
        type(basis_pipeline_state_t) :: state
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), x_dot(3, 1), u(3, 4)
        real(dp) :: phi(3, 4), phi_dot(3, 4), phi_loaded(3, 4)
        real(dp) :: phi_dot_loaded(3, 4), x_bar(3, 1), x_bar_loaded(3, 1)
        real(dp) :: x_hvp(3, 1), x_hvp_loaded(3, 1)
        real(dp) :: theta_dot(1), theta_hvp(1), theta_hvp_loaded(1)
        real(dp), allocatable :: theta(:), theta_loaded(:)
        character(len=*), parameter :: path = &
            "/mnt/storage/worktrees/fortml-basis-pipeline-persistence/pipeline_state.txt"
        real(dp), parameter :: tolerance = 5.0e-13_dp
        integer :: unit, ios, i

        call make_fixture(original, x, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [round trip] fixture setup"
            failures = failures + 1
            return
        end if
        call original%transform(x, phi, status)
        x_dot = reshape([0.1_dp, -0.2_dp, 0.3_dp], shape(x_dot))
        u = reshape([(real(i, dp)/10.0_dp, i=1,12)], shape(u))
        theta_dot = [0.15_dp]
        allocate(theta(1), theta_loaded(1))
        call original%jvp(x, theta_dot, x_dot, phi, phi_dot, status)
        call original%vjp(x, u, theta, x_bar, status)
        call original%hvp(x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [round trip] derivative setup"
            failures = failures + 1
            return
        end if
        call capture_basis_pipeline_state(original, state, status)
        if (.not. status_ok(status) .or. .not. state%valid() .or. &
            state%version /= 1 .or. state%n_features /= 4 .or. &
            state%n_parameters /= 1 .or. state%stage_names(2) /= "seasonal" .or. &
            state%feature_offsets(2) /= 3 .or. &
            state%parameter_offsets(2) /= 1) then
            write (error_unit, '(a)') "FAIL [round trip] state dictionary metadata"
            failures = failures + 1
            return
        end if
        call save_basis_pipeline_text(original, path, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [round trip] text save"
            failures = failures + 1
            return
        end if

        call make_fixture(restored, x, status)
        call restored%set_parameters([log(2.1_dp)], status)
        call load_basis_pipeline_text(restored, path, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [round trip] text load"
            failures = failures + 1
            return
        end if
        call restored%transform(x, phi_loaded, status)
        call restored%jvp(x, theta_dot, x_dot, phi_loaded, phi_dot_loaded, status)
        call restored%vjp(x, u, theta_loaded, x_bar_loaded, status)
        call restored%hvp(x, u, theta_dot, x_dot, theta_hvp_loaded, x_hvp_loaded, status)
        if (.not. status_ok(status) .or. maxval(abs(phi_loaded - phi)) > tolerance .or. &
            maxval(abs(phi_dot_loaded - phi_dot)) > tolerance .or. &
            maxval(abs(theta_loaded - theta)) > tolerance .or. &
            maxval(abs(x_bar_loaded - x_bar)) > tolerance .or. &
            maxval(abs(theta_hvp_loaded - theta_hvp)) > tolerance .or. &
            maxval(abs(x_hvp_loaded - x_hvp)) > tolerance .or. &
            restored%input_schema_name(1) /= "time" .or. &
            restored%feature_name(4) /= original%feature_name(4) .or. &
            restored%parameter_name(1) /= original%parameter_name(1)) then
            write (error_unit, '(a)') "FAIL [round trip] value/JVP/VJP/HVP equivalence"
            failures = failures + 1
        end if
        call save_basis_pipeline_device(original, path, FORTML_DEVICE_CUDA, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
            write (error_unit, '(a)') "FAIL [round trip] typed CUDA refusal"
            failures = failures + 1
        end if
        open (newunit=unit, file=path, status="old", iostat=ios)
        if (ios == 0) close (unit, status="delete")
    end subroutine test_round_trip

    subroutine test_invalid_input_is_transactional(failures)
        integer, intent(inout) :: failures
        type(basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), before(1), after(1)
        character(len=*), parameter :: path = &
            "/mnt/storage/worktrees/fortml-basis-pipeline-persistence/pipeline_state_invalid.txt"
        integer :: unit, ios

        call make_fixture(pipeline, x, status)
        before = pipeline%parameters()
        open (newunit=unit, file=path, status="replace", action="write", iostat=ios)
        if (ios == 0) then
            write (unit, '(a)') "FORTML_BASIS_PIPELINE_STATE 99"
            close (unit)
        end if
        call load_basis_pipeline_text(pipeline, path, status)
        after = pipeline%parameters()
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. maxval(abs(after - before)) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [invalid input] load transaction"
            failures = failures + 1
        end if
        open (newunit=unit, file=path, status="old", iostat=ios)
        if (ios == 0) close (unit, status="delete")
    end subroutine test_invalid_input_is_transactional

end program test_pipeline_persistence
