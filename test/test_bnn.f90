program test_bnn
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_bnn, only: bnn_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_kl_against_analytic_formula(failures)
    call test_seeded_determinism(failures)
    call test_elbo_gradient_against_finite_difference(failures)
    call test_elbo_jvp_against_finite_difference(failures)
    call test_elbo_hvp_against_finite_difference(failures)
    call test_invalid_shapes_refuse(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " BNN test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_model(model, status)
        type(bnn_t), intent(out) :: model
        type(fortnum_status_t), intent(out) :: status

        call model%initialize([2, 3, 1], 4, 20260806, status, &
            prior_variance=4.0_dp, noise_variance=0.25_dp)
    end subroutine build_model

    subroutine build_data(x, y)
        real(dp), intent(out) :: x(:, :), y(:, :)

        x = reshape([ &
            0.2_dp, -0.4_dp, 0.7_dp, &
            0.1_dp, -0.5_dp, 0.8_dp], shape(x))
        y = reshape([0.3_dp, -0.2_dp, 0.9_dp], shape(y))
    end subroutine build_data

    function packed_state(n) result(lambda)
        integer, intent(in) :: n
        real(dp), allocatable :: lambda(:)
        integer :: i

        allocate(lambda(2*n))
        do i = 1, n
            lambda(i) = 0.35_dp*sin(real(i, dp)) - 0.15_dp
            lambda(n + i) = -0.4_dp + 0.12_dp*cos(real(2*i, dp))
        end do
    end function packed_state

    subroutine test_kl_against_analytic_formula(failures)
        !! Compare the module KL with the textbook Gaussian formula and with a
        !! hand-evaluated constant case: mu = 1/2, sigma = 2, prior variance 4
        !! gives log(2/2) + (4 + 1/4)/8 - 1/2 = 1/32 per parameter.
        integer, intent(inout) :: failures
        type(bnn_t) :: model
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lambda(:), gradient(:), reference(:)
        real(dp) :: value, expected, plus, minus, h
        integer :: n, i

        call build_model(model, status)
        n = model%net%parameter_count()
        allocate(lambda(2*n), gradient(2*n), reference(2*n))
        lambda(1:n) = 0.5_dp
        lambda(n + 1:2*n) = log(2.0_dp)
        call model%set_parameters(lambda, status)
        call model%kl(value, status)
        expected = real(n, dp)/32.0_dp
        if (.not. status_ok(status) .or. abs(value - expected) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [kl] hand-derived Gaussian KL value"
            failures = failures + 1
        end if

        lambda = packed_state(n)
        call model%set_parameters(lambda, status)
        call model%kl(value, status)
        expected = 0.0_dp
        do i = 1, n
            expected = expected + log(2.0_dp/exp(lambda(n + i))) &
                + (exp(2.0_dp*lambda(n + i)) + lambda(i)**2)/8.0_dp - 0.5_dp
        end do
        if (abs(value - expected) > 1.0e-13_dp) then
            write (error_unit, '(a)') "FAIL [kl] independent KL formula"
            failures = failures + 1
        end if

        call model%kl_gradient(gradient, status)
        h = 1.0e-6_dp
        do i = 1, 2*n
            call perturbed_kl(model, lambda, i, h, plus)
            call perturbed_kl(model, lambda, i, -h, minus)
            reference(i) = (plus - minus)/(2.0_dp*h)
        end do
        if (maxval(abs(gradient - reference)) > 1.0e-7_dp) then
            write (error_unit, '(a)') "FAIL [kl] KL gradient finite difference"
            failures = failures + 1
        end if
    end subroutine test_kl_against_analytic_formula

    subroutine perturbed_kl(model, lambda, index, step, value)
        type(bnn_t), intent(inout) :: model
        real(dp), intent(in) :: lambda(:)
        integer, intent(in) :: index
        real(dp), intent(in) :: step
        real(dp), intent(out) :: value
        type(fortnum_status_t) :: status
        real(dp), allocatable :: shifted(:)

        allocate(shifted, source=lambda)
        shifted(index) = shifted(index) + step
        call model%set_parameters(shifted, status)
        call model%kl(value, status)
        call model%set_parameters(lambda, status)
    end subroutine perturbed_kl

    subroutine test_seeded_determinism(failures)
        integer, intent(inout) :: failures
        type(bnn_t) :: first, second, other
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lambda(:)
        real(dp) :: x(3, 2), y(3, 1)
        real(dp) :: value_first, value_second, value_other
        integer :: n

        call build_data(x, y)
        call build_model(first, status)
        call build_model(second, status)
        n = first%net%parameter_count()
        lambda = packed_state(n)
        call first%set_parameters(lambda, status)
        call second%set_parameters(lambda, status)
        call first%elbo(x, y, value_first, status)
        call second%elbo(x, y, value_second, status)

        call other%initialize([2, 3, 1], 4, 987654321, status, &
            prior_variance=4.0_dp, noise_variance=0.25_dp)
        call other%set_parameters(lambda, status)
        call other%elbo(x, y, value_other, status)

        if (abs(value_first - value_second) > 0.0_dp .or. &
            abs(value_first - value_other) < 1.0e-8_dp) then
            write (error_unit, '(a)') &
                "FAIL [mc] seeded Monte Carlo determinism or seed sensitivity"
            failures = failures + 1
        end if
    end subroutine test_seeded_determinism

    subroutine test_elbo_gradient_against_finite_difference(failures)
        integer, intent(inout) :: failures
        type(bnn_t) :: model
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lambda(:), gradient(:), reference(:)
        real(dp) :: x(3, 2), y(3, 1)
        real(dp) :: plus, minus, h
        integer :: n, i

        call build_data(x, y)
        call build_model(model, status)
        n = model%net%parameter_count()
        lambda = packed_state(n)
        call model%set_parameters(lambda, status)
        allocate(gradient(2*n), reference(2*n))
        call model%elbo_vjp(x, y, 1.0_dp, gradient, status)

        h = 1.0e-6_dp
        do i = 1, 2*n
            call perturbed_elbo(model, lambda, x, y, i, h, plus)
            call perturbed_elbo(model, lambda, x, y, i, -h, minus)
            reference(i) = (plus - minus)/(2.0_dp*h)
        end do
        if (.not. status_ok(status) .or. &
            maxval(abs(gradient - reference)) > 1.0e-7_dp) then
            write (error_unit, '(a)') &
                "FAIL [vjp] ELBO gradient against central finite differences"
            failures = failures + 1
        end if

        call model%elbo_vjp(x, y, -2.5_dp, reference, status)
        if (maxval(abs(reference + 2.5_dp*gradient)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [vjp] scalar cotangent scaling"
            failures = failures + 1
        end if
    end subroutine test_elbo_gradient_against_finite_difference

    subroutine perturbed_elbo(model, lambda, x, y, index, step, value)
        type(bnn_t), intent(inout) :: model
        real(dp), intent(in) :: lambda(:), x(:, :), y(:, :)
        integer, intent(in) :: index
        real(dp), intent(in) :: step
        real(dp), intent(out) :: value
        type(fortnum_status_t) :: status
        real(dp), allocatable :: shifted(:)

        allocate(shifted, source=lambda)
        shifted(index) = shifted(index) + step
        call model%set_parameters(shifted, status)
        call model%elbo(x, y, value, status)
        call model%set_parameters(lambda, status)
    end subroutine perturbed_elbo

    subroutine test_elbo_jvp_against_finite_difference(failures)
        integer, intent(inout) :: failures
        type(bnn_t) :: model
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lambda(:), direction(:), shifted(:), gradient(:)
        real(dp) :: x(3, 2), y(3, 1)
        real(dp) :: value, tangent, plus, minus, h, reference
        integer :: n, i

        call build_data(x, y)
        call build_model(model, status)
        n = model%net%parameter_count()
        lambda = packed_state(n)
        allocate(direction(2*n), shifted(2*n), gradient(2*n))
        do i = 1, 2*n
            direction(i) = 0.3_dp*cos(real(3*i, dp)) - 0.1_dp
        end do
        call model%set_parameters(lambda, status)
        call model%elbo_jvp(x, y, direction, value, tangent, status)

        h = 1.0e-6_dp
        shifted = lambda + h*direction
        call model%set_parameters(shifted, status)
        call model%elbo(x, y, plus, status)
        shifted = lambda - h*direction
        call model%set_parameters(shifted, status)
        call model%elbo(x, y, minus, status)
        reference = (plus - minus)/(2.0_dp*h)
        call model%set_parameters(lambda, status)
        call model%elbo_vjp(x, y, 1.0_dp, gradient, status)

        if (.not. status_ok(status) .or. abs(tangent - reference) > 1.0e-6_dp &
            .or. abs(tangent - sum(gradient*direction)) > 1.0e-11_dp) then
            write (error_unit, '(a)') &
                "FAIL [jvp] ELBO directional derivative or adjoint identity"
            failures = failures + 1
        end if
    end subroutine test_elbo_jvp_against_finite_difference

    subroutine test_elbo_hvp_against_finite_difference(failures)
        integer, intent(inout) :: failures
        type(bnn_t) :: model
        type(fortnum_status_t) :: status
        real(dp), allocatable :: lambda(:), direction(:), shifted(:)
        real(dp), allocatable :: product(:), plus(:), minus(:), reference(:)
        real(dp), allocatable :: unit_i(:), unit_j(:), hvp_i(:), hvp_j(:)
        real(dp) :: x(3, 2), y(3, 1), h
        integer :: n, i

        call build_data(x, y)
        call build_model(model, status)
        n = model%net%parameter_count()
        lambda = packed_state(n)
        allocate(direction(2*n), shifted(2*n), product(2*n))
        allocate(plus(2*n), minus(2*n), reference(2*n))
        allocate(unit_i(2*n), unit_j(2*n), hvp_i(2*n), hvp_j(2*n))
        do i = 1, 2*n
            direction(i) = 0.25_dp*sin(real(2*i + 1, dp)) + 0.05_dp
        end do
        call model%set_parameters(lambda, status)
        call model%elbo_hvp(x, y, direction, product, status)

        h = 1.0e-6_dp
        shifted = lambda + h*direction
        call model%set_parameters(shifted, status)
        call model%elbo_vjp(x, y, 1.0_dp, plus, status)
        shifted = lambda - h*direction
        call model%set_parameters(shifted, status)
        call model%elbo_vjp(x, y, 1.0_dp, minus, status)
        reference = (plus - minus)/(2.0_dp*h)
        call model%set_parameters(lambda, status)
        if (.not. status_ok(status) .or. &
            maxval(abs(product - reference)) > 1.0e-6_dp) then
            write (error_unit, '(a)') &
                "FAIL [hvp] ELBO Hessian product against finite differences"
            failures = failures + 1
        end if

        unit_i = 0.0_dp
        unit_j = 0.0_dp
        unit_i(2) = 1.0_dp
        unit_j(n + 3) = 1.0_dp
        call model%elbo_hvp(x, y, unit_i, hvp_i, status)
        call model%elbo_hvp(x, y, unit_j, hvp_j, status)
        if (abs(hvp_i(n + 3) - hvp_j(2)) > 1.0e-12_dp) then
            write (error_unit, '(a)') "FAIL [hvp] ELBO Hessian symmetry"
            failures = failures + 1
        end if
    end subroutine test_elbo_hvp_against_finite_difference

    subroutine test_invalid_shapes_refuse(failures)
        integer, intent(inout) :: failures
        type(bnn_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), y(3, 1), bad_gradient(3), value
        real(dp), allocatable :: lambda(:)
        integer :: n

        call build_data(x, y)
        call build_model(model, status)
        n = model%net%parameter_count()
        lambda = packed_state(n)
        call model%set_parameters(lambda, status)
        call model%elbo_vjp(x, y, 1.0_dp, bad_gradient, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] wrong gradient size accepted"
            failures = failures + 1
        end if
        call model%initialize([2, 3, 1], 0, 7, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] zero Monte Carlo count accepted"
            failures = failures + 1
        end if
        call model%elbo(x, y, value, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] uninitialized model accepted"
            failures = failures + 1
        end if
    end subroutine test_invalid_shapes_refuse

end program test_bnn
