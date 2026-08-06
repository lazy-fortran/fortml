program test_parameter_registry
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_parameter_registry, only: parameter_block_t, &
        parameter_block_from_kernel, parameter_block_from_mlp, &
        parameter_registry_t
    use fortml_mlp, only: mlp_t
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(mlp_t), target :: model
    type(kernel_t), target :: kernel
    type(parameter_block_t) :: model_block, kernel_block, duplicate_block
    type(parameter_registry_t) :: registry
    type(fortnum_status_t) :: status
    real(dp) :: model_parameters(7), packed(9)
    real(dp) :: updated_model(7), updated_kernel(2)
    integer :: first, last
    logical :: found

    model_parameters = [0.2_dp, -0.1_dp, 0.4_dp, 0.3_dp, -0.5_dp, 0.6_dp, 0.1_dp]
    call model%initialize([1, 2, 1], status)
    call model%set_parameters(model_parameters, status)
    kernel = make_rbf_kernel(1, 1.5_dp, 0.9_dp, status)

    call parameter_block_from_mlp(model_block, "network.weights", model, status)
    call parameter_block_from_kernel(kernel_block, "gp.rbf", kernel, status)
    call registry%add(model_block, status)
    call registry%add(kernel_block, status)
    call registry%pack(packed, status)
    if (.not. status_ok(status)) error stop 1
    if (maxval(abs(packed(:7) - model_parameters)) > 1.0e-14_dp) error stop 2
    if (maxval(abs(packed(8:) - kernel%parameters())) > 1.0e-14_dp) error stop 3

    call registry%range("gp.rbf", first, last, found)
    if (.not. found .or. first /= 8 .or. last /= 9) error stop 4

    updated_model = packed(:7) + 0.05_dp
    updated_kernel = packed(8:) - [0.1_dp, 0.2_dp]
    packed(:7) = updated_model
    packed(8:) = updated_kernel
    call registry%unpack(packed, status)
    if (.not. status_ok(status)) error stop 5
    if (maxval(abs(model%parameters() - updated_model)) > 1.0e-14_dp) error stop 6
    if (maxval(abs(kernel%parameters() - updated_kernel)) > 1.0e-14_dp) error stop 7

    call parameter_block_from_mlp(duplicate_block, "network.weights", model, status)
    call registry%add(duplicate_block, status)
    if (status_ok(status)) error stop 8
    write (*, '(a)') "PASS parameter registry independent packing oracle"
end program test_parameter_registry
