program test_pca
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_pca, only: pca_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_fit_against_closed_form_oracle(failures)
    call test_whitening_and_inverse(failures)
    call test_input_products(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " PCA test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_fit_against_closed_form_oracle(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(6, 2), centered(6, 2), covariance(2, 2)
        real(dp) :: mean_ref(2), expected(2), components(2, 2)
        real(dp) :: lambda, norm_value
        real(dp), allocatable :: model_mean(:), model_components(:)
        real(dp), allocatable :: model_matrix(:, :), variance(:), ratio(:)
        type(pca_t) :: model
        type(fortnum_status_t) :: status

        call fixture(x)
        mean_ref = sum(x, dim=1) / real(size(x, 1), dp)
        centered = x - spread(mean_ref, dim=1, ncopies=size(x, 1))
        covariance = matmul(transpose(centered), centered) / &
            real(size(x, 1) - 1, dp)
        lambda = 0.5_dp * (covariance(1, 1) + covariance(2, 2) + sqrt( &
            (covariance(1, 1) - covariance(2, 2))**2 + &
            4.0_dp*covariance(1, 2)**2))
        expected = [covariance(1, 2), lambda - covariance(1, 1)]
        norm_value = sqrt(sum(expected**2))
        expected = expected / norm_value
        if (abs(expected(1)) >= abs(expected(2))) then
            if (expected(1) < 0.0_dp) expected = -expected
        else
            if (expected(2) < 0.0_dp) expected = -expected
        end if

        call model%fit(x, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a,a)') "FAIL [fit] ", trim(status%msg)
            failures = failures + 1
            return
        end if
        model_mean = model%mean()
        model_matrix = model%components()
        variance = model%explained_variance()
        ratio = model%explained_variance_ratio()
        components = model_matrix
        if (maxval(abs(model_mean - mean_ref)) > 1.0e-13_dp) then
            write (error_unit, '(a)') "FAIL [mean] centered mean oracle"
            failures = failures + 1
        end if
        if (maxval(abs(components(1, :) - expected)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [components] closed-form eigenvector"
            failures = failures + 1
        end if
        if (maxval(abs(variance - [4.0_dp, 1.5_dp])) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [variance] covariance eigenvalues"
            failures = failures + 1
        end if
        if (maxval(abs(ratio - [8.0_dp/11.0_dp, 3.0_dp/11.0_dp])) > &
            1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [ratio] explained variance ratio"
            failures = failures + 1
        end if
        if (model%n_components() /= 2 .or. model%feature_count() /= 2 .or. &
            model%sample_count() /= 6 .or. .not. model%fitted()) then
            write (error_unit, '(a)') "FAIL [metadata] fitted PCA state"
            failures = failures + 1
        end if
    end subroutine test_fit_against_closed_form_oracle

    subroutine test_whitening_and_inverse(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(6, 2), centered(6, 2), transformed(6, 1), recovered(6, 2)
        real(dp) :: expected(6, 2), components(1, 2), mean_value(2), variance
        type(pca_t) :: model
        type(fortnum_status_t) :: status

        call fixture(x)
        call model%fit(x, status, n_components=1, whiten=.true.)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [whiten fit] status"
            failures = failures + 1
            return
        end if
        call model%transform(x, transformed, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [whiten transform] status"
            failures = failures + 1
            return
        end if
        variance = sum(transformed(:, 1)**2) / real(size(x, 1) - 1, dp)
        if (abs(variance - 1.0_dp) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [whiten] sample variance=", &
                variance
            failures = failures + 1
        end if
        components = model%components()
        mean_value = model%mean()
        centered = x - spread(mean_value, dim=1, ncopies=size(x, 1))
        expected = matmul(matmul(centered, transpose(components)), components)
        expected = expected + spread(mean_value, dim=1, ncopies=size(x, 1))
        call model%inverse_transform(transformed, recovered, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [inverse] status"
            failures = failures + 1
            return
        end if
        if (maxval(abs(recovered - expected)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [inverse] projection oracle"
            failures = failures + 1
        end if
        if (.not. model%whiten()) then
            write (error_unit, '(a)') "FAIL [metadata] whitening flag"
            failures = failures + 1
        end if
    end subroutine test_whitening_and_inverse

    subroutine test_input_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(6, 2), x_dot(4, 2), transformed_dot(4, 2)
        real(dp) :: transformed_bar(4, 2), x_bar(4, 2)
        real(dp) :: components(2, 2), expected(4, 2), lhs, rhs
        type(pca_t) :: model
        type(fortnum_status_t) :: status

        call fixture(x)
        call model%fit(x, status)
        if (.not. status_ok(status)) then
            write (error_unit, '(a)') "FAIL [products fit] status"
            failures = failures + 1
            return
        end if
        components = model%components()
        x_dot = reshape([ &
            0.2_dp, -0.1_dp, &
            0.4_dp, 0.3_dp, &
            -0.5_dp, 0.6_dp, &
            0.1_dp, -0.7_dp], shape(x_dot))
        call model%transform_jvp(x_dot, transformed_dot, status)
        expected = matmul(x_dot, transpose(components))
        if (.not. status_ok(status) .or. maxval(abs(transformed_dot - expected)) > &
            1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [JVP] fixed-state transform"
            failures = failures + 1
        end if
        transformed_bar = reshape([ &
            0.3_dp, -0.2_dp, &
            0.1_dp, 0.4_dp, &
            -0.6_dp, 0.5_dp, &
            0.2_dp, -0.1_dp], shape(transformed_bar))
        call model%transform_vjp(transformed_bar, x_bar, status)
        lhs = sum(transformed_bar*transformed_dot)
        rhs = sum(x_bar*x_dot)
        if (.not. status_ok(status) .or. abs(lhs - rhs) > 1.0e-12_dp) then
            write (error_unit, '(a,es12.4)') "FAIL [VJP] adjoint identity=", &
                abs(lhs - rhs)
            failures = failures + 1
        end if
    end subroutine test_input_products

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(6, 2), transformed(6, 2), tiny_x(1, 2)
        type(pca_t) :: model, empty
        type(fortnum_status_t) :: status

        call fixture(x)
        call empty%transform(x, transformed, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [refusal] unfitted transform"
            failures = failures + 1
        end if
        call model%fit(x, status, n_components=3)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [refusal] invalid n_components"
            failures = failures + 1
        end if
        tiny_x = 0.0_dp
        call model%fit(tiny_x, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [refusal] one-row fit"
            failures = failures + 1
        end if
    end subroutine test_refusals

    subroutine fixture(x)
        real(dp), intent(out) :: x(:, :)

        x(1, :) = [1.0_dp, 2.0_dp]
        x(2, :) = [2.0_dp, 4.0_dp]
        x(3, :) = [3.0_dp, 1.0_dp]
        x(4, :) = [4.0_dp, 3.0_dp]
        x(5, :) = [5.0_dp, 0.0_dp]
        x(6, :) = [6.0_dp, 2.0_dp]
    end subroutine fixture

end program test_pca
