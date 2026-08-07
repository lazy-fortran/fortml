program test_linear_autoencoder
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_linear_autoencoder, only: linear_autoencoder_t
    use fortml_pca, only: pca_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(pca_t) :: pca
    type(linear_autoencoder_t) :: autoencoder
    type(fortnum_status_t) :: status
    real(dp) :: x(5, 3), latent(5, 2), reconstruction(5, 3)
    real(dp) :: expected_latent(5, 2), expected_reconstruction(5, 3)
    real(dp) :: x_dot(5, 3), reconstruction_dot(5, 3), expected_dot(5, 3)
    real(dp), allocatable :: components(:, :), center(:)
    integer :: failures

    failures = 0
    x = reshape([ &
        -2.0_dp, -1.0_dp, 0.0_dp, &
        -1.0_dp,  0.0_dp, 1.0_dp, &
         0.0_dp,  1.0_dp, 2.0_dp, &
         1.0_dp,  2.0_dp, 3.0_dp, &
         2.0_dp,  3.0_dp, 4.0_dp], shape(x))
    x_dot = reshape([ &
         0.1_dp, -0.2_dp,  0.3_dp, &
        -0.4_dp,  0.5_dp, -0.6_dp, &
         0.7_dp, -0.8_dp,  0.9_dp, &
        -1.0_dp,  1.1_dp, -1.2_dp, &
         1.3_dp, -1.4_dp,  1.5_dp], shape(x_dot))

    call pca%fit(x, status, n_components=2)
    if (.not. status_ok(status)) then
        write (error_unit, '(a)') "FAIL [linear-autoencoder] PCA fit"
        error stop 1
    end if
    call autoencoder%initialize_from_pca(pca, status)
    if (.not. status_ok(status) .or. .not. autoencoder%fitted()) then
        write (error_unit, '(a)') "FAIL [linear-autoencoder] initialization"
        error stop 1
    end if

    components = pca%components()
    center = pca%mean()
    expected_latent = matmul(x - spread(center, 1, size(x, 1)), transpose(components))
    call autoencoder%encode(x, latent, status)
    if (.not. status_ok(status) .or. maxval(abs(latent - expected_latent)) > 2.0e-12_dp) then
        write (error_unit, '(a,es12.4)') "FAIL [linear-autoencoder] encode oracle=", &
            maxval(abs(latent - expected_latent))
        failures = failures + 1
    end if

    expected_reconstruction = matmul(expected_latent, components) + &
        spread(center, 1, size(x, 1))
    call autoencoder%decode(latent, reconstruction, status)
    if (.not. status_ok(status) .or. &
            maxval(abs(reconstruction - expected_reconstruction)) > 2.0e-12_dp) then
        write (error_unit, '(a,es12.4)') "FAIL [linear-autoencoder] decode oracle=", &
            maxval(abs(reconstruction - expected_reconstruction))
        failures = failures + 1
    end if
    call autoencoder%reconstruct(x, reconstruction, status)
    if (.not. status_ok(status) .or. &
            maxval(abs(reconstruction - expected_reconstruction)) > 2.0e-12_dp) then
        write (error_unit, '(a,es12.4)') "FAIL [linear-autoencoder] reconstruct oracle=", &
            maxval(abs(reconstruction - expected_reconstruction))
        failures = failures + 1
    end if

    expected_dot = matmul(matmul(x_dot, transpose(components)), components)
    call autoencoder%reconstruct_jvp(x_dot, reconstruction_dot, status)
    if (.not. status_ok(status) .or. &
            maxval(abs(reconstruction_dot - expected_dot)) > 2.0e-12_dp) then
        write (error_unit, '(a,es12.4)') "FAIL [linear-autoencoder] JVP oracle=", &
            maxval(abs(reconstruction_dot - expected_dot))
        failures = failures + 1
    end if

    if (.not. autoencoder%device_supported(FORTML_DEVICE_CPU) .or. &
            autoencoder%device_supported(FORTML_DEVICE_CUDA)) then
        write (error_unit, '(a)') "FAIL [linear-autoencoder] device contract"
        failures = failures + 1
    end if
    if (failures /= 0) error stop 1
    write (*, '(a)') "PASS"
end program test_linear_autoencoder
