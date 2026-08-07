function fortml_cuda_mse_available() bind(C, name="fortml_cuda_mse_available") result(available)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    integer(c_int) :: available

    available = 0_c_int
end function fortml_cuda_mse_available

function fortml_cuda_mean_squared_error( &
        target, prediction, sample_weight, value, n_samples, n_outputs, &
        device_index) bind(C, name="fortml_cuda_mean_squared_error") result(status)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    implicit none
    type(c_ptr), value :: target, prediction, sample_weight, value
    integer(c_int), value :: n_samples, n_outputs, device_index
    integer(c_int) :: status

    status = 3_c_int
end function fortml_cuda_mean_squared_error
