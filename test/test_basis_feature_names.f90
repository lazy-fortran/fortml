program test_basis_feature_names
    !! Independent behavioral oracle for semantic basis feature names.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis, &
        make_polynomial_basis
    use fortml_column_pipeline, only: column_basis_pipeline_t, &
        make_column_basis_pipeline
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline, &
        make_sequential_basis_pipeline, sequential_basis_pipeline_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call check_map_names(failures)
    call check_horizontal_pipeline_names(failures)
    call check_sequential_pipeline_names(failures)
    call check_column_pipeline_names(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " semantic feature-name test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS semantic basis feature-name independent oracle"

contains

    subroutine check_map_names(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial
        type(fortnum_status_t) :: status
        character(*), parameter :: names(5) = [character(16) :: "bias", "x1", &
            "x1_sq", "x2", "x2_sq"]
        character(*), parameter :: duplicate_names(5) = [character(16) :: &
            "bias", "x1", "x1", "x2", "x2_sq"]
        real(dp) :: x(2, 2), phi(2, 5), expected(2, 5)

        polynomial = make_polynomial_basis(2, 2, status, include_intercept=.true.)
        call polynomial%set_feature_names(names, status)
        if (.not. status_ok(status) .or. polynomial%feature_name(1) /= "bias" .or. &
            polynomial%feature_name(3) /= "x1_sq") then
            write (error_unit, '(a)') "FAIL [map names] install/access"
            failures = failures + 1
        end if

        call polynomial%set_feature_names(duplicate_names, status)
        if (status%code /= FORTNUM_DOMAIN_ERROR .or. &
            polynomial%feature_name(3) /= "x1_sq") then
            write (error_unit, '(a)') "FAIL [map names] transactional duplicate refusal"
            failures = failures + 1
        end if

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 0.1_dp], shape(x))
        call polynomial%evaluate(x, phi, status)
        expected = 0.0_dp
        expected(:, 1) = 1.0_dp
        expected(:, 2) = x(:, 1)
        expected(:, 3) = x(:, 1)**2
        expected(:, 4) = x(:, 2)
        expected(:, 5) = x(:, 2)**2
        if (.not. status_ok(status) .or. maxval(abs(phi - expected)) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [map names] values changed by metadata"
            failures = failures + 1
        end if
    end subroutine check_map_names

    subroutine check_horizontal_pipeline_names(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial, fourier
        type(basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        character(*), parameter :: polynomial_names(2) = [character(16) :: &
            "x1", "x1_sq"]
        character(*), parameter :: fourier_names(4) = [character(16) :: &
            "sin_x1", "cos_x1", "sin_x2", "cos_x2"]

        polynomial = make_polynomial_basis(2, 1, status)
        call polynomial%set_feature_names(polynomial_names, status)
        fourier = make_fourier_basis(2, reshape([0.8_dp, 1.1_dp], [1, 2]), status)
        call fourier%set_feature_names(fourier_names, status)
        pipeline = make_basis_pipeline(2, status)
        call pipeline%append(polynomial, status, name="poly")
        call pipeline%append(fourier, status, name="harmonics")
        if (.not. status_ok(status) .or. pipeline%feature_name(1) /= "poly.x1" .or. &
            pipeline%feature_name(3) /= "harmonics.sin_x1" .or. &
            pipeline%feature_name(6) /= "harmonics.cos_x2") then
            write (error_unit, '(a)') "FAIL [horizontal pipeline names] routing"
            failures = failures + 1
        end if
    end subroutine check_horizontal_pipeline_names

    subroutine check_sequential_pipeline_names(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial, fourier
        type(sequential_basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        character(*), parameter :: names(4) = [character(16) :: "f1", "f2", &
            "f3", "f4"]

        pipeline = make_sequential_basis_pipeline(2, status)
        polynomial = make_polynomial_basis(2, 1, status)
        call pipeline%append(polynomial, status, name="linear")
        fourier = make_fourier_basis(2, reshape([0.8_dp, 1.1_dp], [1, 2]), status)
        call fourier%set_feature_names(names, status)
        call pipeline%append(fourier, status, name="fourier")
        if (.not. status_ok(status) .or. pipeline%feature_name(1) /= "fourier.f1" .or. &
            pipeline%feature_name(4) /= "fourier.f4") then
            write (error_unit, '(a)') "FAIL [sequential pipeline names] routing"
            failures = failures + 1
        end if
    end subroutine check_sequential_pipeline_names

    subroutine check_column_pipeline_names(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: polynomial, fourier
        type(column_basis_pipeline_t) :: pipeline
        type(fortnum_status_t) :: status
        character(*), parameter :: polynomial_names(2) = [character(16) :: &
            "value", "square"]
        character(*), parameter :: fourier_names(2) = [character(16) :: &
            "sin", "cos"]

        polynomial = make_polynomial_basis(1, 2, status)
        call polynomial%set_feature_names(polynomial_names, status)
        fourier = make_fourier_basis(1, reshape([0.8_dp], [1, 1]), status)
        call fourier%set_feature_names(fourier_names, status)
        pipeline = make_column_basis_pipeline(2, status)
        call pipeline%append(polynomial, [1], status, name="left")
        call pipeline%append(fourier, [2], status, name="right")
        if (.not. status_ok(status) .or. pipeline%feature_name(1) /= "left.value" .or. &
            pipeline%feature_name(4) /= "right.cos") then
            write (error_unit, '(a)') "FAIL [column pipeline names] routing"
            failures = failures + 1
        end if
    end subroutine check_column_pipeline_names

end program test_basis_feature_names
