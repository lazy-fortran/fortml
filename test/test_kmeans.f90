program test_kmeans
    !! Independent behavioral oracle for dense seeded k-means.
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, only: error_unit
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_kmeans, only: kmeans_t
    implicit none

    integer :: failures

    failures = 0
    call test_fit_predict_transform(failures)
    call test_fixed_center_products(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " k-means test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS K-means independent behavioral oracle"

contains

    subroutine test_fit_predict_transform(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(6, 2), transformed(6, 2), expected_centers(2, 2)
        integer :: expected_labels(6)
        real(dp), allocatable :: centers(:, :)
        integer, allocatable :: fitted_labels(:)
        type(kmeans_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: expected_inertia

        call fixture(x)
        call model%fit(x, status, n_clusters=2, max_iter=50, tolerance=1.0e-12_dp, &
            initialization_seed=3)
        call check(status_ok(status), "fit converged", failures)
        expected_centers = reshape([ &
            (-0.2_dp+0.0_dp+0.1_dp)/3.0_dp, (10.0_dp+10.1_dp+9.8_dp)/3.0_dp, &
            (0.1_dp-0.1_dp+0.0_dp)/3.0_dp, (10.0_dp+9.9_dp+10.2_dp)/3.0_dp], &
            shape(expected_centers))
        centers = model%cluster_centers()
        call check(maxval(abs(centers-expected_centers)) < 1.0e-12_dp, &
            "centroid oracle", failures)
        expected_labels = [1, 1, 1, 2, 2, 2]
        fitted_labels = model%labels()
        call check(all(fitted_labels == expected_labels), "stable labels", failures)
        expected_inertia = sum((x(1:3, :)-spread(expected_centers(1, :), 1, 3))**2) + &
            sum((x(4:6, :)-spread(expected_centers(2, :), 1, 3))**2)
        call check(abs(model%inertia()-expected_inertia) < 1.0e-12_dp, &
            "inertia oracle", failures)
        call model%transform(x, transformed, status)
        call check(status_ok(status), "transform status", failures)
        call check(abs(transformed(1, 1)-sqrt(sum((x(1, :)-expected_centers(1, :))**2))) < 1.0e-12_dp, &
            "distance transform oracle", failures)
        call model%predict(x, fitted_labels, status)
        call check(status_ok(status) .and. all(fitted_labels == expected_labels), &
            "predict labels", failures)
        call check(model%fitted() .and. model%n_clusters() == 2 .and. &
            model%feature_count() == 2 .and. model%sample_count() == 6 .and. &
            model%initialization_seed() == 3, "fit metadata", failures)
    end subroutine test_fit_predict_transform

    subroutine test_fixed_center_products(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 2), x_dot(2, 2), distances_dot(2, 2), distances_bar(2, 2), x_bar(2, 2)
        real(dp) :: expected(2, 2), lhs, rhs, centers(2, 2)
        type(kmeans_t) :: model
        type(fortnum_status_t) :: status

        call fixture(x)
        call model%fit(x, status, n_clusters=2, max_iter=50, tolerance=1.0e-12_dp, &
            initialization_seed=3)
        call check(status_ok(status), "product fixture fit", failures)
        x = reshape([1.0_dp, 1.5_dp, 8.0_dp, 8.5_dp], shape(x))
        x_dot = reshape([0.2_dp, -0.1_dp, 0.3_dp, 0.4_dp], shape(x_dot))
        distances_bar = reshape([0.3_dp, -0.2_dp, 0.1_dp, 0.4_dp], shape(distances_bar))
        centers = model%cluster_centers()
        call model%transform_jvp(x, x_dot, distances_dot, status)
        call check(status_ok(status), "transform JVP status", failures)
        expected(1, 1) = dot_product(x(1, :)-centers(1, :), x_dot(1, :)) / &
            sqrt(sum((x(1, :)-centers(1, :))**2))
        call check(abs(distances_dot(1, 1)-expected(1, 1)) < 1.0e-12_dp, &
            "transform JVP oracle", failures)
        call model%transform_vjp(x, distances_bar, x_bar, status)
        lhs = sum(distances_bar*distances_dot)
        rhs = sum(x_bar*x_dot)
        call check(status_ok(status) .and. abs(lhs-rhs) < 1.0e-12_dp, &
            "transform VJP adjoint", failures)
    end subroutine test_fixed_center_products

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(6, 2), duplicate(4, 2), bad(2, 1)
        type(kmeans_t) :: model
        type(fortnum_status_t) :: status

        call fixture(x)
        call model%fit(x, status, n_clusters=2, initialization_seed=3, &
            device_kind=FORTML_DEVICE_CUDA)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, "CUDA fit refusal", failures)
        duplicate = 0.0_dp
        call model%fit(duplicate, status, n_clusters=2, initialization_seed=1)
        call check(status%code == FORTNUM_CONVERGENCE_ERROR, &
            "empty cluster refusal", failures)
        bad = 0.0_dp
        bad(1, 1) = ieee_value(0.0_dp, ieee_quiet_nan)
        call model%fit(bad, status, n_clusters=1)
        call check(.not. status_ok(status), "nonfinite input refusal", failures)
    end subroutine test_refusals

    subroutine fixture(x)
        real(dp), intent(out) :: x(:, :)

        x(1, :) = [0.0_dp, 0.0_dp]
        x(2, :) = [0.1_dp, -0.1_dp]
        x(3, :) = [-0.2_dp, 0.1_dp]
        x(4, :) = [10.0_dp, 10.0_dp]
        x(5, :) = [10.1_dp, 9.9_dp]
        x(6, :) = [9.8_dp, 10.2_dp]
    end subroutine fixture

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [kmeans] "//trim(description)
        end if
    end subroutine check

end program test_kmeans
