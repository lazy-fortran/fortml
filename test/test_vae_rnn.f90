program test_vae_rnn
    !! Oracles for the variational autoencoder and the recurrent network.
    !!
    !! The VAE checks are: the KL term against its analytic formula computed
    !! independently from the encoder output, the complete ELBO gradient
    !! against central finite differences, and determinism of the seeded draw.
    !! The RNN checks are: a hand-rolled two-step forward reference, and the
    !! backpropagation-through-time gradient against central finite
    !! differences, which is an independent oracle for the reverse scan.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortml_vae, only: vae_t
    use fortml_rnn, only: rnn_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer :: failures

    failures = 0
    call test_vae_kl_and_determinism(failures)
    call test_vae_gradient(failures)
    call test_rnn_forward_reference(failures)
    call test_rnn_gradient(failures)
    call test_refusals(failures)
    if (failures /= 0) then
        write (error_unit, '(i0,a)') failures, " VAE/RNN test(s) failed"
        error stop 1
    end if
    write (*, '(a)') "PASS"

contains

    subroutine build_vae(model, status)
        type(vae_t), intent(out) :: model
        type(fortnum_status_t), intent(out) :: status

        call model%initialize(3, 4, 2, 5, 20260806, status, &
            likelihood_variance=0.4_dp)
    end subroutine build_vae

    subroutine build_inputs(x)
        real(dp), intent(out) :: x(:, :)
        integer :: i, j

        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                x(i, j) = 0.4_dp*sin(real(i + 2*j, dp)) - 0.1_dp
            end do
        end do
    end subroutine build_inputs

    function vae_state(n) result(theta)
        integer, intent(in) :: n
        real(dp), allocatable :: theta(:)
        integer :: i

        allocate(theta(n))
        do i = 1, n
            theta(i) = 0.3_dp*sin(0.7_dp*real(i, dp)) - 0.05_dp
        end do
    end function vae_state

    subroutine test_vae_kl_and_determinism(failures)
        integer, intent(inout) :: failures
        type(vae_t) :: model, twin
        type(fortnum_status_t) :: status
        real(dp), allocatable :: theta(:)
        real(dp) :: x(5, 3), value, twin_value, likelihood, kl_value
        real(dp) :: code(5, 4), expected
        integer :: i, j

        call build_vae(model, status)
        call build_vae(twin, status)
        theta = vae_state(model%parameter_count())
        call model%set_parameters(theta, status)
        call twin%set_parameters(theta, status)
        call build_inputs(x)

        call model%elbo(x, value, status, expected_log_likelihood=likelihood, &
            kl_value=kl_value)
        call twin%elbo(x, twin_value, status)
        if (.not. status_ok(status) .or. abs(value - twin_value) > 0.0_dp) then
            write (error_unit, '(a)') "FAIL [vae] the same seed is not reproduced"
            failures = failures + 1
        end if
        if (abs(value - (likelihood - kl_value)) > 1.0e-13_dp) then
            write (error_unit, '(a)') "FAIL [vae] ELBO decomposition"
            failures = failures + 1
        end if

        ! The analytic KL of a diagonal Gaussian against the unit normal,
        ! evaluated here from the encoder output.
        call model%encoder%predict(x, code, status)
        expected = 0.0_dp
        do j = 1, 2
            do i = 1, 5
                expected = expected + 0.5_dp*(exp(2.0_dp*code(i, 2 + j)) &
                    + code(i, j)**2 - 1.0_dp - 2.0_dp*code(i, 2 + j))
            end do
        end do
        if (abs(kl_value - expected) > 1.0e-12_dp .or. kl_value < 0.0_dp) then
            write (error_unit, '(a,2es14.6)') &
                "FAIL [vae] KL against the analytic formula ", kl_value, expected
            failures = failures + 1
        end if
    end subroutine test_vae_kl_and_determinism

    subroutine test_vae_gradient(failures)
        integer, intent(inout) :: failures
        type(vae_t) :: model
        type(fortnum_status_t) :: status
        real(dp), allocatable :: theta(:), gradient(:), reference(:), shifted(:)
        real(dp) :: x(5, 3), value, plus, minus, h
        integer :: i, n

        call build_vae(model, status)
        n = model%parameter_count()
        theta = vae_state(n)
        call model%set_parameters(theta, status)
        call build_inputs(x)
        allocate(gradient(n), reference(n), shifted(n))
        call model%elbo_gradient(x, value, gradient, status)

        h = 1.0e-6_dp
        do i = 1, n
            shifted = theta
            shifted(i) = theta(i) + h
            call model%set_parameters(shifted, status)
            call model%elbo(x, plus, status)
            shifted(i) = theta(i) - h
            call model%set_parameters(shifted, status)
            call model%elbo(x, minus, status)
            reference(i) = (plus - minus)/(2.0_dp*h)
        end do
        call model%set_parameters(theta, status)

        if (.not. status_ok(status) .or. &
            maxval(abs(gradient - reference)) > 2.0e-7_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [vae] ELBO gradient against finite differences ", &
                maxval(abs(gradient - reference))
            failures = failures + 1
        end if
    end subroutine test_vae_gradient

    subroutine test_rnn_forward_reference(failures)
        !! Two steps, one sequence, hand-rolled from the recurrence.
        integer, intent(inout) :: failures
        type(rnn_t) :: model
        type(fortnum_status_t) :: status
        real(dp) :: inputs(2, 1, 1), outputs(2, 1, 1), hidden(2, 1, 1)
        real(dp) :: theta(5), h1, h2, y1, y2

        call model%initialize(1, 1, 1, status)
        ! [W_xh, W_hh, b_h, W_hy, b_y]
        theta = [0.7_dp, -0.4_dp, 0.2_dp, 1.3_dp, -0.1_dp]
        call model%set_parameters(theta, status)
        inputs(1, 1, 1) = 0.5_dp
        inputs(2, 1, 1) = -0.8_dp
        call model%forward(inputs, outputs, hidden, status)

        h1 = tanh(0.5_dp*0.7_dp + 0.0_dp + 0.2_dp)
        y1 = h1*1.3_dp - 0.1_dp
        h2 = tanh(-0.8_dp*0.7_dp + h1*(-0.4_dp) + 0.2_dp)
        y2 = h2*1.3_dp - 0.1_dp

        if (.not. status_ok(status) .or. &
            abs(hidden(1, 1, 1) - h1) > 1.0e-14_dp .or. &
            abs(hidden(2, 1, 1) - h2) > 1.0e-14_dp .or. &
            abs(outputs(1, 1, 1) - y1) > 1.0e-14_dp .or. &
            abs(outputs(2, 1, 1) - y2) > 1.0e-14_dp) then
            write (error_unit, '(a)') "FAIL [rnn] forward scan reference"
            failures = failures + 1
        end if
    end subroutine test_rnn_forward_reference

    subroutine test_rnn_gradient(failures)
        integer, intent(inout) :: failures
        type(rnn_t) :: model
        type(fortnum_status_t) :: status
        real(dp), allocatable :: theta(:), gradient(:), reference(:), shifted(:)
        real(dp) :: inputs(4, 3, 2), targets(4, 3, 2)
        real(dp) :: value, plus, minus, h
        integer :: i, j, k, n

        call model%initialize(2, 3, 2, status)
        n = model%parameter_count()
        allocate(theta(n), gradient(n), reference(n), shifted(n))
        do i = 1, n
            theta(i) = 0.35_dp*sin(0.9_dp*real(i, dp)) - 0.08_dp
        end do
        call model%set_parameters(theta, status)
        do k = 1, 2
            do j = 1, 3
                do i = 1, 4
                    inputs(i, j, k) = 0.3_dp*sin(real(i + 2*j + 3*k, dp))
                    targets(i, j, k) = 0.2_dp*cos(real(2*i + j - k, dp))
                end do
            end do
        end do

        call model%loss_gradient(inputs, targets, value, gradient, status)
        h = 1.0e-6_dp
        do i = 1, n
            shifted = theta
            shifted(i) = theta(i) + h
            call model%set_parameters(shifted, status)
            call model%loss(inputs, targets, plus, status)
            shifted(i) = theta(i) - h
            call model%set_parameters(shifted, status)
            call model%loss(inputs, targets, minus, status)
            reference(i) = (plus - minus)/(2.0_dp*h)
        end do
        call model%set_parameters(theta, status)

        if (.not. status_ok(status) .or. &
            maxval(abs(gradient - reference)) > 2.0e-7_dp) then
            write (error_unit, '(a,es12.4)') &
                "FAIL [rnn] BPTT gradient against finite differences ", &
                maxval(abs(gradient - reference))
            failures = failures + 1
        end if
    end subroutine test_rnn_gradient

    subroutine test_refusals(failures)
        integer, intent(inout) :: failures
        type(vae_t) :: model
        type(rnn_t) :: network
        type(fortnum_status_t) :: status
        real(dp) :: x(5, 3), wrong(4, 3), value
        real(dp) :: inputs(2, 1, 1), targets(2, 1, 1), gradient(3)

        call model%initialize(3, 4, 2, 5, 7, status, likelihood_variance=-1.0_dp)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] negative likelihood variance"
            failures = failures + 1
        end if
        call build_vae(model, status)
        call build_inputs(x)
        wrong = 0.0_dp
        call model%elbo(wrong, value, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a wrong batch size accepted"
            failures = failures + 1
        end if

        call network%initialize(1, 0, 1, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a zero hidden width accepted"
            failures = failures + 1
        end if
        call network%initialize(1, 2, 1, status)
        inputs = 0.0_dp
        targets = 0.0_dp
        call network%loss_gradient(inputs, targets, value, gradient, status)
        if (status_ok(status)) then
            write (error_unit, '(a)') "FAIL [guard] a short gradient accepted"
            failures = failures + 1
        end if
    end subroutine test_refusals

end program test_vae_rnn
