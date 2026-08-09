program test_sequential_pipeline_clone
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis, &
        make_polynomial_basis
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    use fortml_pipeline, only: sequential_basis_pipeline_t, &
        make_sequential_basis_pipeline
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures

    failures = 0
    call test_copy_and_mutation(failures)
    call test_invalid_source_is_transactional(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " sequential pipeline clone test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS sequential pipeline clone independent behavioral oracle"

contains

    subroutine make_fixture(pipeline, x, status)
        type(sequential_basis_pipeline_t), intent(out) :: pipeline
        real(dp), intent(out) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(basis_map_t) :: polynomial, fourier

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp], shape(x))
        polynomial = make_polynomial_basis(1, 2, status)
        if (.not. status_ok(status)) return
        fourier = make_fourier_basis(2, reshape([0.7_dp, 1.1_dp], [1, 2]), &
            status)
        if (.not. status_ok(status)) return
        pipeline = make_sequential_basis_pipeline(1, status)
        call pipeline%append(polynomial, status, name="powers")
        if (.not. status_ok(status)) return
        call pipeline%append(fourier, status, name="harmonics")
        if (.not. status_ok(status)) return
        call pipeline%fit(x, status)
    end subroutine make_fixture

    subroutine test_copy_and_mutation(failures)
        integer, intent(inout) :: failures
        type(sequential_basis_pipeline_t) :: source, clone
        type(fortnum_status_t) :: status
        type(fortml_device_t) :: device
        real(dp) :: x(3, 1), source_y(3, 4), clone_y(3, 4), changed_y(3, 4)
        real(dp), allocatable :: source_theta(:), clone_theta(:), source_after(:)
        real(dp) :: source_parameter

        call make_fixture(source, x, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [sequential clone] fixture setup"
            failures = failures + 1
            return
        end if
        call source%clone(clone, status)
        call source%transform(x, source_y, status)
        call clone%transform(x, clone_y, status)
        if (.not. status_ok(status) .or. maxval(abs(source_y - clone_y)) > 1.0e-14_dp .or. &
                clone%stage_count() /= source%stage_count() .or. &
                clone%feature_count() /= source%feature_count() .or. &
                clone%stage_name(2) /= "harmonics") then
            write (error_unit, '(a)') "FAIL [sequential clone] copied output/metadata"
            failures = failures + 1
            return
        end if

        source_theta = source%parameters()
        clone_theta = source_theta
        if (size(clone_theta) < 1) then
            write (error_unit, '(a)') "FAIL [sequential clone] parameter fixture"
            failures = failures + 1
            return
        end if
        source_parameter = source_theta(1)
        allocate(source_after(size(source_theta)))
        clone_theta(1) = clone_theta(1) + 0.25_dp
        call clone%set_parameters(clone_theta, status)
        call clone%transform(x, changed_y, status)
        source_after = source%parameters()
        if (.not. status_ok(status) .or. abs(source_after(1) - source_parameter) > &
                0.0_dp .or. maxval(abs(changed_y - source_y)) < 1.0e-10_dp) then
            write (error_unit, '(a)') "FAIL [sequential clone] mutation isolation"
            failures = failures + 1
        end if

        device%kind = FORTML_DEVICE_CUDA
        device%selected = .true.
        device%available = .true.
        call source%clone_device(device, clone, status)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED) then
            write (error_unit, '(a)') "FAIL [sequential clone] typed CUDA boundary"
            failures = failures + 1
        end if
    end subroutine test_copy_and_mutation

    subroutine test_invalid_source_is_transactional(failures)
        integer, intent(inout) :: failures
        type(sequential_basis_pipeline_t) :: source, destination
        type(fortnum_status_t) :: status, invalid_status
        real(dp) :: x(3, 1), before(3, 4), after(3, 4)

        call make_fixture(destination, x, status)
        call destination%transform(x, before, status)
        call source%clone(destination, invalid_status)
        call destination%transform(x, after, status)
        if (invalid_status%code /= FORTNUM_DOMAIN_ERROR .or. &
                .not. status_ok(status) .or. &
                maxval(abs(after - before)) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [sequential clone] invalid source transaction"
            failures = failures + 1
        end if
    end subroutine test_invalid_source_is_transactional

end program test_sequential_pipeline_clone
