program test_basis_pipeline_training
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_basis, only: basis_map_t, make_fourier_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    use fortml_basis_pipeline_training, only: &
        basis_pipeline_training_objective_t
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    implicit none

    integer :: failures
    failures = 0
    call check_products(failures)
    call check_cuda_refusal(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, &
            " basis pipeline training test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine check_products(failures)
        integer, intent(inout) :: failures
        integer, parameter :: n = 7
        real(dp) :: x(n, 1), y(n, 1), theta(4), direction(4), gradient(4)
        real(dp) :: gradient_plus(4), gradient_minus(4), hvp(4)
        real(dp) :: theta_plus(4), theta_minus(4)
        real(dp) :: value, value_plus, value_minus, tangent
        real(dp) :: h, log_frequency, frequency
        type(basis_map_t) :: fourier
        type(basis_pipeline_t) :: pipeline
        type(basis_pipeline_training_objective_t) :: objective
        type(fortnum_status_t) :: status
        integer :: i

        x(:, 1) = [-1.2_dp, -0.8_dp, -0.35_dp, 0.1_dp, 0.45_dp, 0.9_dp, 1.3_dp]
        log_frequency = log(0.73_dp)
        frequency = exp(log_frequency)
        do i = 1, n
            y(i, 1) = 0.4_dp + 1.2_dp*sin(frequency*x(i, 1)) - &
                0.8_dp*cos(frequency*x(i, 1))
        end do
        fourier = make_fourier_basis(1, reshape([frequency], [1, 1]), status)
        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(fourier, status, name="fourier")
        call objective%initialize(pipeline, x, y, status, ridge=0.03_dp)
        if (.not. status_ok(status) .or. .not. objective%initialized() .or. &
            objective%parameter_count() /= 4) then
            write (error_unit, '(a)') "FAIL [basis training] initialization oracle"
            failures = failures + 1
            return
        end if
        theta = [log_frequency, 0.4_dp, 1.2_dp, -0.8_dp]
        direction = [0.17_dp, -0.11_dp, 0.07_dp, -0.05_dp]
        call objective%value_gradient(theta, value, gradient, status)
        call objective%jvp(theta, direction, value_plus, tangent, status)
        call objective%hvp(theta, direction, hvp, status)
        if (.not. status_ok(status) .or. value > 5.0e-2_dp .or. &
            abs(tangent - dot_product(gradient, direction)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [basis training] value/JVP oracle"
            failures = failures + 1
            return
        end if
        h = 2.0e-5_dp
        theta_plus = theta + h*direction
        theta_minus = theta - h*direction
        call objective%value_gradient(theta_plus, value_plus, gradient_plus, status)
        call objective%value_gradient(theta_minus, value_minus, gradient_minus, status)
        if (.not. status_ok(status) .or. &
            abs((value_plus - value_minus)/(2.0_dp*h) - tangent) > 2.0e-7_dp .or. &
            maxval(abs((gradient_plus-gradient_minus)/(2.0_dp*h)-hvp)) > 2.0e-6_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [basis training] directional HVP oracle max error=", &
                maxval(abs((gradient_plus-gradient_minus)/(2.0_dp*h)-hvp))
            failures = failures + 1
        end if
        call objective%value_gradient(theta, value_plus, gradient_plus, status)
        if (.not. status_ok(status)) failures = failures + 1
    end subroutine check_products

    subroutine check_cuda_refusal(failures)
        integer, intent(inout) :: failures
        real(dp) :: x(2, 1), y(2, 1)
        type(basis_map_t) :: fourier
        type(basis_pipeline_t) :: pipeline
        type(basis_pipeline_training_objective_t) :: objective
        type(fortnum_status_t) :: status

        x(:, 1) = [-1.0_dp, 1.0_dp]
        y(:, 1) = [0.0_dp, 1.0_dp]
        fourier = make_fourier_basis(1, reshape([1.0_dp], [1, 1]), status)
        pipeline = make_basis_pipeline(1, status)
        call pipeline%append(fourier, status)
        call objective%initialize(pipeline, x, y, status, &
            device_kind=FORTML_DEVICE_CUDA)
        if (status%code /= FORTNUM_NOT_IMPLEMENTED .or. &
            objective%initialized()) then
            write (error_unit, '(a)') "FAIL [basis training] CUDA typed refusal"
            failures = failures + 1
        end if
    end subroutine check_cuda_refusal

end program test_basis_pipeline_training
