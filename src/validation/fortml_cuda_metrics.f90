module fortml_cuda_metrics
    !! Native CUDA reductions with explicit transfer-inclusive semantics.
    !!
    !! The CUDA MSE entry point accepts host arrays, copies them to a resident
    !! temporary allocation, evaluates the weighted elementwise squared error
    !! and block reduction on CUDA, and copies one scalar back.  This is an
    !! explicit transfer path; it never falls back to the Fortran CPU metric.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_loc, c_null_ptr, &
        c_ptr
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CUDA
    implicit none
    private

    public :: fortml_cuda_mse_available
    public :: fortml_cuda_mean_squared_error
    public :: cuda_mean_squared_error

    interface
        function fortml_cuda_mse_available() bind(C, &
                name="fortml_cuda_mse_available") result(available)
            import :: c_int
            integer(c_int) :: available
        end function fortml_cuda_mse_available

        function fortml_cuda_mean_squared_error( &
                target, prediction, sample_weight, value, n_samples, &
                n_outputs, device_index) bind(C, &
                name="fortml_cuda_mean_squared_error") result(code)
            import :: c_double, c_int, c_ptr
            type(c_ptr), value :: target, prediction, sample_weight, value
            integer(c_int), value :: n_samples, n_outputs, device_index
            integer(c_int) :: code
        end function fortml_cuda_mean_squared_error
    end interface

contains

    subroutine cuda_mean_squared_error(device, target, prediction, value, status, &
            sample_weight)
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in), target, contiguous :: target(:, :), prediction(:, :)
        real(dp), intent(out), target :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), target, contiguous, optional :: sample_weight(:)

        type(c_ptr) :: weight_ptr
        integer(c_int) :: code
        real(dp) :: weight_mass

        value = 0.0_dp
        if (device%kind /= FORTML_DEVICE_CUDA .or. .not. device%selected .or. &
                .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA MSE: a selected, available CUDA device is required")
            return
        end if
        if (size(target, 1) < 1 .or. size(target, 2) < 1 .or. &
                any(shape(target) /= shape(prediction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA MSE: target and prediction shapes must match")
            return
        end if
        if (any(.not. ieee_is_finite(target)) .or. &
                any(.not. ieee_is_finite(prediction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "CUDA MSE: target and prediction must be finite")
            return
        end if
        weight_ptr = c_null_ptr
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(target, 1) .or. &
                    any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CUDA MSE: weights must be finite and nonnegative")
                return
            end if
            weight_mass = sum(sample_weight)
            if (weight_mass <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "CUDA MSE: weight mass must be positive")
                return
            end if
            weight_ptr = c_loc(sample_weight)
        end if
        if (fortml_cuda_mse_available() == 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MSE: native CUDA reduction is unavailable")
            return
        end if
        code = fortml_cuda_mean_squared_error(c_loc(target), c_loc(prediction), &
            weight_ptr, c_loc(value), int(size(target, 1), c_int), &
            int(size(target, 2), c_int), int(device%device_index, c_int))
        if (code /= 0_c_int) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "CUDA MSE: native reduction failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine cuda_mean_squared_error

end module fortml_cuda_metrics
