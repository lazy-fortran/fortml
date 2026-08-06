function fortml_cuda_kernel_available() bind(C, &
        name="fortml_cuda_kernel_available") result(available)
    use, intrinsic :: iso_c_binding, only: c_int
    integer(c_int) :: available
    available = 0_c_int
end function fortml_cuda_kernel_available

function fortml_cuda_kernel_plan_create( &
        points, program_kind, program_variance, program_lengthscale, &
        n_samples, n_features, program_size, status) bind(C, &
        name="fortml_cuda_kernel_plan_create") result(plan)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr, c_null_ptr
    type(c_ptr), value :: points, program_kind, program_variance
    type(c_ptr), value :: program_lengthscale
    integer(c_int), value :: n_samples, n_features, program_size
    integer(c_int) :: status
    type(c_ptr) :: plan
    plan = c_null_ptr
    status = 1_c_int
end function fortml_cuda_kernel_plan_create

function fortml_cuda_kernel_plan_destroy(plan) bind(C, &
        name="fortml_cuda_kernel_plan_destroy") result(status)
    use, intrinsic :: iso_c_binding, only: c_int, c_ptr
    type(c_ptr), value :: plan
    integer(c_int) :: status
    status = 1_c_int
end function fortml_cuda_kernel_plan_destroy

function fortml_cuda_kernel_plan_matvec( &
        plan, input, output, diagonal_shift) bind(C, &
        name="fortml_cuda_kernel_plan_matvec") result(status)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    type(c_ptr), value :: plan, input, output
    real(c_double), value :: diagonal_shift
    integer(c_int) :: status
    status = 1_c_int
end function fortml_cuda_kernel_plan_matvec

function fortml_cuda_kernel_plan_matmat( &
        plan, input, output, n_rhs, diagonal_shift) bind(C, &
        name="fortml_cuda_kernel_plan_matmat") result(status)
    use, intrinsic :: iso_c_binding, only: c_double, c_int, c_ptr
    type(c_ptr), value :: plan, input, output
    integer(c_int), value :: n_rhs
    real(c_double), value :: diagonal_shift
    integer(c_int) :: status
    status = 1_c_int
end function fortml_cuda_kernel_plan_matmat
