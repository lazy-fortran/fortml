function fortml_cuda_rbf_available() bind(C, &
        name="fortml_cuda_rbf_available") result(available)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none

    integer(c_int) :: available

    available = 0_c_int
end function fortml_cuda_rbf_available

function fortml_cuda_rbf_matvec( &
        points, input, output, n_samples, variance, inverse_scale, &
        diagonal_shift) bind(C, name="fortml_cuda_rbf_matvec") result(status)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    implicit none

    type(c_ptr), value :: points, input, output
    integer(c_int), value :: n_samples
    real(c_double), value :: variance, inverse_scale, diagonal_shift
    integer(c_int) :: status

    status = 1_c_int
end function fortml_cuda_rbf_matvec
