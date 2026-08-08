program test_mlp_pca_initializer
    !! Independent PCA reconstruction oracle for the MLP initializer.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_pca, only: pca_t
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    implicit none

    integer :: failures

    failures = 0
    call test_reconstruction(.false., failures)
    call test_reconstruction(.true., failures)
    call test_refusal_transaction(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " MLP PCA-initializer test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine test_reconstruction(whiten, failures)
        logical, intent(in) :: whiten
        integer, intent(inout) :: failures
        type(pca_t) :: pca
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(6, 3), latent(6, 2), expected(6, 3), actual(6, 3)

        x = reshape([ &
            -2.0_dp, -1.0_dp, 0.5_dp, &
            -1.0_dp,  0.0_dp, 1.5_dp, &
            0.0_dp,  1.0_dp, 2.5_dp, &
            1.0_dp,  2.0_dp, 3.5_dp, &
            2.0_dp,  3.0_dp, 4.5_dp, &
            3.0_dp,  4.0_dp, 5.5_dp], shape(x))
        call pca%fit(x, status, n_components=2, whiten=whiten)
        call check(status_ok(status), "PCA fit", failures)
        call model%initialize_from_pca(pca, status)
        call check(status_ok(status), "PCA MLP initialization", failures)
        call check(model%hidden_activation == MLP_LINEAR .and. &
            model%output_activation == MLP_LINEAR, &
            "PCA MLP activations", failures)

        call pca%transform(x, latent, status)
        call pca%inverse_transform(latent, expected, status)
        call model%predict(x, actual, status)
        call check(status_ok(status) .and. maxval(abs(actual - expected)) < 3.0e-12_dp, &
            "PCA reconstruction oracle", failures)
    end subroutine test_reconstruction

    subroutine test_refusal_transaction(failures)
        integer, intent(inout) :: failures
        type(pca_t) :: unfitted
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: weight1(3, 2), bias1(2), weight2(2, 3), bias2(3)
        real(dp) :: before(17), after(17)

        weight1 = 0.2_dp
        bias1 = [-0.1_dp, 0.3_dp]
        weight2 = -0.4_dp
        bias2 = [0.5_dp, -0.6_dp, 0.7_dp]
        call model%initialize_linear(weight1, bias1, weight2, bias2, status)
        before = model%parameters()
        call model%initialize_from_pca(unfitted, status)
        after = model%parameters()
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "unfitted PCA refusal", failures)
        call check(maxval(abs(after - before)) == 0.0_dp, &
            "unfitted PCA transaction", failures)
    end subroutine test_refusal_transaction

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [" // trim(label) // "]"
        end if
    end subroutine check

end program test_mlp_pca_initializer
