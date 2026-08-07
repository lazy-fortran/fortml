program test_mlp_parameter_layout
    !! Independent behavioral oracle for the named MLP parameter tree.
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp, only: mlp_t, mlp_parameter_block_t, MLP_LINEAR, MLP_TANH
    implicit none

    integer :: failures

    failures = 0
    call test_layout_and_ranges(failures)
    call test_uninitialized_layout(failures)
    if (failures > 0) then
        write (*, '(a,i0)') "FAIL MLP parameter-layout cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP parameter-layout behavioral oracle"

contains

    subroutine test_layout_and_ranges(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_parameter_block_t), allocatable :: layout(:)
        type(fortnum_status_t) :: status
        integer :: first, last
        logical :: found

        call model%initialize([2, 3, 1], status, hidden_activation=MLP_TANH, &
            output_activation=MLP_LINEAR)
        layout = model%parameter_layout()
        call check(status_ok(status), "model initialization", failures)
        call check(model%parameter_block_count() == 4 .and. size(layout) == 4, &
            "four dense parameter blocks", failures)
        if (size(layout) /= 4) return

        call check(trim(layout(1)%name) == "layer_1.weight" .and. &
            trim(layout(2)%name) == "layer_1.bias" .and. &
            trim(layout(3)%name) == "layer_2.weight" .and. &
            trim(layout(4)%name) == "layer_2.bias", &
            "stable layer paths", failures)
        call check(all(layout%trainable) .and. .not. any(layout%is_buffer), &
            "parameter versus buffer roles", failures)
        call check(layout(1)%first == 1 .and. layout(1)%last == 6 .and. &
            layout(1)%rows == 2 .and. layout(1)%columns == 3, &
            "first weight range and shape", failures)
        call check(layout(2)%first == 7 .and. layout(2)%last == 9 .and. &
            layout(2)%rows == 3 .and. layout(2)%columns == 1, &
            "first bias range and shape", failures)
        call check(layout(3)%first == 10 .and. layout(3)%last == 12 .and. &
            layout(3)%rows == 3 .and. layout(3)%columns == 1, &
            "second weight range and shape", failures)
        call check(layout(4)%first == 13 .and. layout(4)%last == 13 .and. &
            layout(4)%rows == 1 .and. layout(4)%columns == 1, &
            "second bias range and shape", failures)

        call model%parameter_range("layer_2.weight", first, last, found)
        call check(found .and. first == 10 .and. last == 12, &
            "named range lookup", failures)
        call model%parameter_range("missing", first, last, found)
        call check(.not. found .and. first == 0 .and. last == -1, &
            "unknown range refusal", failures)
    end subroutine test_layout_and_ranges

    subroutine test_uninitialized_layout(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: model
        type(mlp_parameter_block_t), allocatable :: layout(:)
        integer :: first, last
        logical :: found

        layout = model%parameter_layout()
        call check(model%parameter_block_count() == 0 .and. size(layout) == 0, &
            "uninitialized tree is empty", failures)
        call model%parameter_range("layer_1.weight", first, last, found)
        call check(.not. found .and. first == 0 .and. last == -1, &
            "uninitialized range refusal", failures)
    end subroutine test_uninitialized_layout

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "FAIL: " // trim(label)
        end if
    end subroutine check

end program test_mlp_parameter_layout
