program test_mlp_trainable_state
    !! Independent behavioral oracle for named MLP freeze state.
    !!
    !! The oracle compares the model's products before and after freezing a
    !! block.  It deliberately does not inspect implementation storage: a
    !! frozen coordinate must disappear from JVP/VJP/HVP products while the
    !! packed deployment state and all other gradients remain unchanged.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_DOMAIN_ERROR
    use fortml_mlp, only: mlp_t, mlp_parameter_block_t, MLP_TANH, MLP_LINEAR
    implicit none

    integer :: failures

    failures = 0
    call test_freeze_products(failures)
    call test_unknown_path_is_transactional(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP trainable-state cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP trainable-state behavioral oracle"

contains

    subroutine test_freeze_products(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_parameter_block_t), allocatable :: layout(:)
        type(fortnum_status_t) :: status
        real(dp), parameter :: x(2, 1) = reshape([0.25_dp, -0.75_dp], [2, 1])
        real(dp), parameter :: theta(7) = [0.4_dp, -0.2_dp, 0.1_dp, -0.3_dp, &
            0.7_dp, -0.6_dp, 0.05_dp]
        real(dp), allocatable :: y(:, :), y_frozen(:, :), dy(:, :), dtheta(:)
        real(dp), allocatable :: u(:, :), dx(:, :)
        real(dp), allocatable :: gradient(:), frozen_gradient(:), x_bar(:, :)
        logical, allocatable :: mask(:)
        integer :: first, last
        logical :: found

        call model%initialize([1, 2, 1], status, hidden_activation=MLP_TANH, &
            output_activation=MLP_LINEAR)
        call check(status_ok(status), "initialize model", failures)
        call model%set_parameters(theta, status)
        call check(status_ok(status), "load deterministic parameters", failures)

        allocate(y(2, 1), y_frozen(2, 1), dy(2, 1), dtheta(7), gradient(7), &
            frozen_gradient(7), x_bar(2, 1), u(2, 1), dx(2, 1))
        u = 1.0_dp
        dx = 0.0_dp
        call model%predict(x, y, status)
        call check(status_ok(status), "baseline prediction", failures)
        call model%vjp(x, u, gradient, x_bar, status)
        call check(status_ok(status), "baseline VJP", failures)
        call check(maxval(abs(gradient)) > 1.0e-12_dp, &
            "baseline VJP has a live coordinate", failures)

        call model%set_trainable("layer_1.weight", .false., status)
        call check(status_ok(status), "freeze named weight block", failures)
        layout = model%parameter_layout()
        call check(.not. layout(1)%trainable .and. layout(2)%trainable, &
            "layout exposes block freeze", failures)
        mask = model%trainable_mask()
        call check(size(mask) == 7 .and. .not. any(mask(1:2)) .and. &
            all(mask(3:7)), "coordinate mask follows named block", failures)
        call check(model%trainable_parameter_count() == 5, &
            "trainable coordinate count", failures)

        call model%predict(x, y_frozen, status)
        call check(status_ok(status) .and. maxval(abs(y_frozen - y)) < 1.0e-14_dp, &
            "freezing does not mutate deployment values", failures)
        call model%vjp(x, u, frozen_gradient, x_bar, status)
        call check(status_ok(status), "frozen VJP", failures)
        call check(maxval(abs(frozen_gradient(1:2))) == 0.0_dp .and. &
            maxval(abs(frozen_gradient(3:7) - gradient(3:7))) < 1.0e-14_dp, &
            "frozen VJP masks only selected coordinates", failures)

        dtheta = 0.0_dp
        dtheta(1) = 1.0_dp
        call model%jvp(x, dtheta, dx, y_frozen, dy, status)
        call check(status_ok(status) .and. maxval(abs(dy)) == 0.0_dp, &
            "frozen JVP ignores selected direction", failures)
        call model%set_trainable("layer_1.weight", .true., status)
        call check(status_ok(status), "unfreeze named weight block", failures)
        call model%jvp(x, dtheta, dx, y_frozen, dy, status)
        call check(status_ok(status) .and. maxval(abs(dy)) > 1.0e-12_dp, &
            "unfrozen JVP restores selected direction", failures)

        call model%parameter_range("layer_1.weight", first, last, found)
        call check(found .and. first == 1 .and. last == 2, &
            "freeze uses stable parameter path range", failures)
    end subroutine test_freeze_products

    subroutine test_unknown_path_is_transactional(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(fortnum_status_t) :: status
        logical, allocatable :: mask(:)

        call model%initialize([1, 1], status)
        mask = model%trainable_mask()
        call model%set_trainable("does.not.exist", .false., status)
        call check(status%code == FORTNUM_DOMAIN_ERROR, &
            "unknown path returns typed refusal", failures)
        call check(all(mask .eqv. model%trainable_mask()), &
            "unknown path leaves trainability unchanged", failures)
    end subroutine test_unknown_path_is_transactional

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_mlp_trainable_state
