program test_basis_pipeline_clone
    !! Independent behavioral oracle for transactional pipeline cloning.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_polynomial_basis, make_radial_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call check_clone_and_independence(failures)
    call check_transactional_failure(failures)
    call check_device_boundary(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " pipeline clone test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS basis pipeline clone independent behavioral oracle"

contains

    subroutine make_fixture(pipeline, status)
        type(basis_pipeline_t), intent(out) :: pipeline
        type(fortnum_status_t), intent(out) :: status
        type(basis_map_t) :: polynomial, radial
        real(dp) :: centers(2, 1), scales(2, 1)

        pipeline = make_basis_pipeline(2, status)
        polynomial = make_polynomial_basis(2, 2, status, include_intercept=.true.)
        centers = reshape([0.25_dp, -0.50_dp], shape(centers))
        scales = reshape([0.70_dp, 1.10_dp], shape(scales))
        radial = make_radial_basis(2, centers, scales, status)
        call pipeline%append(polynomial, status, "polynomial")
        call pipeline%append(radial, status, "radial")
    end subroutine make_fixture

    subroutine check_clone_and_independence(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(4, 2), source_phi(4, 6), clone_phi(4, 6)
        real(dp), allocatable :: source_parameters(:), changed(:)
        character(len=16) :: names(2)
        type(basis_pipeline_t) :: source, clone
        type(fortnum_status_t) :: status

        x = reshape([0.2_dp, -0.3_dp, 0.7_dp, 0.1_dp, -0.4_dp, &
            0.8_dp, 0.5_dp, -0.9_dp], shape(x))
        names = [character(len=16) :: "position", "velocity"]
        call make_fixture(source, status)
        call source%set_input_schema(names, status)
        call source%fit(x, status)
        call source%transform(x, source_phi, status)
        source_parameters = source%parameters()
        call source%clone(clone, status)
        call clone%transform(x, clone_phi, status)
        if (.not. status_ok(status) .or. .not. clone%valid() .or. &
                .not. clone%is_fitted() .or. clone%stage_count() /= source%stage_count() .or. &
                clone%feature_count() /= source%feature_count() .or. &
                clone%parameter_count() /= source%parameter_count() .or. &
                maxval(abs(source_phi - clone_phi)) > 1.0e-14_dp .or. &
                maxval(abs(source_parameters - clone%parameters())) > 1.0e-14_dp .or. &
                clone%input_schema_name(1) /= "position" .or. &
                clone%input_schema_name(2) /= "velocity") then
            write (error_unit, '(a)') "FAIL [clone] copied state/metadata oracle"
            failures = failures + 1
            return
        end if

        changed = source_parameters
        changed(1) = changed(1) + 0.35_dp
        call clone%set_parameters(changed, status)
        call clone%transform(x, clone_phi, status)
        call source%transform(x, source_phi, status)
        if (.not. status_ok(status) .or. maxval(abs(source%parameters() - &
                source_parameters)) > 1.0e-14_dp .or. &
                maxval(abs(source_phi - clone_phi)) < 1.0e-8_dp) then
            write (error_unit, '(a)') "FAIL [clone] stage state is not independent"
            failures = failures + 1
        end if
    end subroutine check_clone_and_independence

    subroutine check_transactional_failure(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 2), before(2, 6), after(2, 6)
        type(basis_pipeline_t) :: source, destination
        type(fortnum_status_t) :: status
        integer :: clone_code

        x = reshape([0.2_dp, -0.3_dp, 0.7_dp, 0.1_dp], shape(x))
        call make_fixture(source, status)
        call make_fixture(destination, status)
        call destination%transform(x, before, status)
        call source%initialize(0, status)
        call source%clone(destination, status)
        clone_code = status%code
        call destination%transform(x, after, status)
        if (clone_code == 0 .or. maxval(abs(before - after)) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [clone] invalid source was not transactional"
            failures = failures + 1
        end if
    end subroutine check_transactional_failure

    subroutine check_device_boundary(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 2), before(2, 6), after(2, 6)
        type(basis_pipeline_t) :: source, destination
        type(fortml_device_t) :: device
        type(fortnum_status_t) :: status
        integer :: clone_code

        x = reshape([0.2_dp, -0.3_dp, 0.7_dp, 0.1_dp], shape(x))
        call make_fixture(source, status)
        call make_fixture(destination, status)
        call destination%transform(x, before, status)
        device%kind = FORTML_DEVICE_CUDA
        device%selected = .true.
        device%available = .true.
        call source%clone_device(device, destination, status)
        clone_code = status%code
        call destination%transform(x, after, status)
        if (clone_code /= FORTNUM_NOT_IMPLEMENTED .or. &
                maxval(abs(before - after)) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [clone] CUDA refusal/destination oracle"
            failures = failures + 1
        end if
        device%kind = FORTML_DEVICE_CPU
        call source%clone_device(device, destination, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [clone] CPU device clone"
            failures = failures + 1
        end if
    end subroutine check_device_boundary

end program test_basis_pipeline_clone
