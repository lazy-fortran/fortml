program test_pipeline_metadata
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis, &
        make_polynomial_basis
    use fortml_pipeline, only: basis_input_schema_t, basis_pipeline_t, &
        make_basis_pipeline, make_sequential_basis_pipeline, &
        sequential_basis_pipeline_t
    use fortml_column_pipeline, only: column_basis_pipeline_t, &
        make_column_basis_pipeline
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_horizontal_metadata(failures)
    call test_sequential_metadata(failures)
    call test_input_schema_contract(failures)
    call test_column_metadata(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " pipeline metadata test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS pipeline metadata independent naming oracle"

contains

    subroutine test_horizontal_metadata(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial, fourier
        type(basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), phi(2, 4)
        character(len=129) :: long_name

        x(:, 1) = [0.25_dp, -0.75_dp]
        polynomial = make_polynomial_basis(1, 2, status)
        fourier = make_fourier_basis(1, reshape([0.8_dp], [1, 1]), status)
        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(polynomial, status, name="trend")
        call pipeline%append(fourier, status, name="seasonal")
        call pipeline%fit(x, status)
        call pipeline%transform(x, phi, status)
        if (.not. status_ok(status) .or. pipeline%stage_count() /= 2 .or. &
            pipeline%stage_name(1) /= "trend" .or. &
            pipeline%stage_name(2) /= "seasonal" .or. &
            pipeline%feature_name(1) /= "trend.feature_1" .or. &
            pipeline%feature_name(2) /= "trend.feature_2" .or. &
            pipeline%feature_name(3) /= "seasonal.feature_1" .or. &
            pipeline%feature_name(4) /= "seasonal.feature_2" .or. &
            pipeline%parameter_name(1) /= "seasonal.parameter_1" .or. &
            pipeline%stage_feature_offset(1) /= 1 .or. &
            pipeline%stage_feature_offset(2) /= 3 .or. &
            pipeline%stage_parameter_offset(1) /= 1 .or. &
            pipeline%stage_parameter_offset(2) /= 1 .or. &
            pipeline%feature_name(0) /= "" .or. &
            pipeline%parameter_name(0) /= "" .or. &
            maxval(abs(phi(:, 1:2) - reshape([x(1, 1), x(2, 1), &
            x(1, 1)**2, x(2, 1)**2], [2, 2]))) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [horizontal pipeline] metadata oracle"
            failures = failures + 1
        end if

        call pipeline%append(polynomial, status, name="trend")
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
            pipeline%stage_count() /= 2) then
            write (error_unit, '(a)') &
                "FAIL [horizontal pipeline] duplicate name refusal"
            failures = failures + 1
        end if
        call pipeline%append(polynomial, status, name=" ")
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
            pipeline%stage_count() /= 2) then
            write (error_unit, '(a)') &
                "FAIL [horizontal pipeline] empty name refusal"
            failures = failures + 1
        end if

        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(polynomial, status)
        long_name = repeat("x", len(long_name))
        call pipeline%append(fourier, status, name=long_name)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
            pipeline%stage_name(1) /= "stage_1" .or. &
            pipeline%stage_count() /= 1) then
            write (error_unit, '(a)') &
                "FAIL [horizontal pipeline] generated/overlong name contract"
            failures = failures + 1
        end if
    end subroutine test_horizontal_metadata

    subroutine test_sequential_metadata(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial, fourier
        type(sequential_basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 1), y(2, 4)

        x(:, 1) = [0.2_dp, -0.4_dp]
        polynomial = make_polynomial_basis(1, 2, status)
        fourier = make_fourier_basis(2, reshape([0.7_dp, 1.1_dp], [1, 2]), &
            status)
        pipeline = make_sequential_basis_pipeline(1, status)
        call pipeline%append(polynomial, status, name="powers")
        call pipeline%append(fourier, status, name="harmonics")
        call pipeline%fit(x, status)
        call pipeline%transform(x, y, status)
        if (.not. status_ok(status) .or. pipeline%feature_count() /= 4 .or. &
            pipeline%feature_name(1) /= "harmonics.feature_1" .or. &
            pipeline%feature_name(4) /= "harmonics.feature_4" .or. &
            pipeline%parameter_name(1) /= "harmonics.parameter_1" .or. &
            pipeline%parameter_name(2) /= "harmonics.parameter_2" .or. &
            pipeline%stage_feature_offset(1) /= 1 .or. &
            pipeline%stage_feature_offset(2) /= 3 .or. &
            pipeline%stage_parameter_offset(1) /= 1 .or. &
            pipeline%stage_parameter_offset(2) /= 1) then
            write (error_unit, '(a)') "FAIL [sequential pipeline] metadata oracle"
            failures = failures + 1
        end if
    end subroutine test_sequential_metadata

    subroutine test_input_schema_contract(failures)
        integer, intent(inout) :: failures
        type(basis_input_schema_t) :: schema
        type(basis_map_t) :: polynomial
        type(basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        character(len=16) :: names(2), same_names(2), bad_names(2)

        names = [character(len=16) :: "time", "position"]
        same_names = names
        bad_names = [character(len=16) :: "time", "velocity"]
        call schema%initialize(2, status, names)
        if (.not. status_ok(status) .or. .not. schema%valid() .or. &
                schema%count() /= 2 .or. schema%name(1) /= "time" .or. &
                schema%name(2) /= "position") then
            write (error_unit, '(a)') "FAIL [input schema] initialization oracle"
            failures = failures + 1
        end if
        call schema%validate_names(same_names, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [input schema] matching validation"
            failures = failures + 1
        end if
        call schema%validate_names(bad_names, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
                schema%name(2) /= "position") then
            write (error_unit, '(a)') "FAIL [input schema] mismatch/refusal transaction"
            failures = failures + 1
        end if
        call schema%set_names([character(len=16) :: "time", "time"], status)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
                schema%name(2) /= "position") then
            write (error_unit, '(a)') "FAIL [input schema] duplicate refusal transaction"
            failures = failures + 1
        end if

        polynomial = make_polynomial_basis(2, 1, status)
        pipeline = make_basis_pipeline(2, status)
        call pipeline%append(polynomial, status, name="identity")
        call pipeline%set_input_schema(names, status)
        call pipeline%validate_input_schema(same_names, status)
        if (.not. status_ok(status) .or. pipeline%input_schema_name(1) /= "time" .or. &
                pipeline%input_schema_name(2) /= "position") then
            write (error_unit, '(a)') "FAIL [horizontal pipeline] schema routing"
            failures = failures + 1
        end if
        call pipeline%validate_input_schema(bad_names, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
                pipeline%input_schema_name(2) /= "position") then
            write (error_unit, '(a)') "FAIL [horizontal pipeline] schema refusal transaction"
            failures = failures + 1
        end if
    end subroutine test_input_schema_contract

    subroutine test_column_metadata(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: fourier, polynomial
        type(column_basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        real(dp) :: x(2, 3), phi(2, 4)
        integer, allocatable :: columns(:)

        x = reshape([0.2_dp, -0.4_dp, 1.0_dp, 0.5_dp, -0.7_dp, 0.3_dp], &
            shape(x))
        fourier = make_fourier_basis(1, reshape([0.9_dp], [1, 1]), status)
        polynomial = make_polynomial_basis(1, 2, status)
        pipeline = make_column_basis_pipeline(3, status)
        call pipeline%append(fourier, [3], status, name="seasonal")
        call pipeline%append(polynomial, [1], status, name="trend")
        call pipeline%fit(x, status)
        call pipeline%transform(x, phi, status)
        columns = pipeline%stage_columns(1)
        if (.not. status_ok(status) .or. pipeline%stage_name(1) /= "seasonal" .or. &
            pipeline%stage_name(2) /= "trend" .or. &
            pipeline%feature_name(1) /= "seasonal.feature_1" .or. &
            pipeline%feature_name(4) /= "trend.feature_2" .or. &
            pipeline%parameter_name(1) /= "seasonal.parameter_1" .or. &
            pipeline%stage_feature_offset(2) /= 3 .or. &
            pipeline%stage_parameter_offset(2) /= 2 .or. &
            size(columns) /= 1 .or. columns(1) /= 3) then
            write (error_unit, '(a)') "FAIL [column pipeline] metadata oracle"
            failures = failures + 1
        end if
    end subroutine test_column_metadata

end program test_pipeline_metadata
