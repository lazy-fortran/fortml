program test_cuda_boosted_tree_api
    !! Ordinary-build contract test for resident additive-tree dispatch.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, only: real64, int64, error_unit
    use fortnum_status, only: fortnum_status_t, FORTNUM_DOMAIN_ERROR, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_cuda_boosted_tree_api, only: cuda_boosted_tree_plan_t
    implicit none

    type(cuda_boosted_tree_plan_t) :: plan
    type(fortnum_status_t) :: status
    integer :: tree_offset(3), node_feature(6), node_left(6), node_right(6), &
        node_missing_left(6), failures
    real(real64) :: node_threshold(6), node_weight(6), tree_scale(2), &
        query_x(4, 2), query_dot(4, 2), margin(4), margin_dot(4), expected(4)
    real(real64) :: nan_value
    integer(int64) :: host_to_device_bytes, device_to_host_bytes, resident_bytes

    failures = 0
    tree_offset = [0, 3, 6]
    node_feature = [0, -1, -1, 1, -1, -1]
    node_left = [1, -1, -1, 4, -1, -1]
    node_right = [2, -1, -1, 5, -1, -1]
    node_threshold = [0.0_real64, 0.0_real64, 0.0_real64, 1.0_real64, &
        0.0_real64, 0.0_real64]
    node_weight = [0.0_real64, -1.0_real64, 2.0_real64, 0.0_real64, &
        0.5_real64, -0.5_real64]
    node_missing_left = [1, 0, 0, 0, 0, 0]
    tree_scale = [1.0_real64, 0.5_real64]
    query_x = reshape([ &
        -1.0_real64, 0.0_real64, 0.5_real64, 2.0_real64, &
         0.0_real64, 1.0_real64, 1.5_real64, 2.0_real64], shape(query_x))
    query_dot = 0.25_real64
    margin = -31.0_real64
    margin_dot = -37.0_real64
    expected = cpu_oracle(query_x, tree_offset, node_feature, node_left, &
        node_right, node_missing_left, node_weight, tree_scale, 0.2_real64, &
        0.7_real64)

    call plan%create(tree_offset, node_feature, node_left, node_right, &
        node_threshold, node_weight, node_missing_left, tree_scale, 0.2_real64, &
        0.7_real64, -1, status, n_inputs=2)
    call check(status%code == FORTNUM_DOMAIN_ERROR .and. .not. plan%fitted(), &
        "negative device index is rejected", failures)
    call plan%create(tree_offset, node_feature, node_left, node_right, &
        node_threshold, node_weight, node_missing_left, tree_scale, 0.2_real64, &
        0.7_real64, 0, status, n_inputs=2)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. .not. plan%fitted(), &
        "ordinary build keeps typed CUDA refusal", failures)
    call plan%predict(query_x, margin, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(margin == -31.0_real64), &
        "prediction refusal preserves output", failures)
    call plan%predict_jvp(query_x, query_dot, margin, margin_dot, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. all(margin == -31.0_real64) .and. &
        all(margin_dot == -37.0_real64), "JVP refusal preserves outputs", failures)
    call plan%transfer_stats(host_to_device_bytes, device_to_host_bytes, &
        resident_bytes, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED .and. &
        host_to_device_bytes == 0_int64 .and. device_to_host_bytes == 0_int64 .and. &
        resident_bytes == 0_int64, "transfer statistics refusal is typed", failures)

    ! The independent scalar oracle documents the intended resident route and
    ! catches accidental changes to scale/base/learning-rate semantics even
    ! when the ordinary build has no native CUDA implementation.
    call check(maxval(abs(expected - [ -0.325_real64, 1.425_real64, 1.425_real64, &
        1.425_real64 ])) < 1.0e-14_real64, &
        "independent additive-tree oracle", failures)

    nan_value = ieee_value(0.0_real64, ieee_quiet_nan)
    query_x(1, 1) = nan_value
    call plan%create(tree_offset, node_feature, node_left, node_right, &
        node_threshold, node_weight, node_missing_left, tree_scale, 0.2_real64, &
        0.7_real64, 0, status, n_inputs=2)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "missing-value model creation remains typed refusal", failures)

    call plan%destroy(status)
    call check(.not. plan%fitted(), "destroy clears wrapper state", failures)
    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL CUDA boosted tree API cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS CUDA boosted tree Fortran API refusal/oracle"

contains

    function cpu_oracle(x, offsets, features, left, right, missing_left, &
            weights, scales, base, rate) result(values)
        real(real64), intent(in) :: x(:, :), weights(:), scales(:), base, rate
        integer, intent(in) :: offsets(:), features(:), left(:), right(:), &
            missing_left(:)
        real(real64) :: values(size(x, 1))
        integer :: i, tree, node, feature
        real(real64) :: value

        values = base
        do i = 1, size(x, 1)
            value = base
            do tree = 1, size(scales)
                node = offsets(tree) + 1
                do while (features(node) >= 0)
                    feature = features(node) + 1
                    if (ieee_is_nan(x(i, feature))) then
                        if (missing_left(node) /= 0) then
                            node = left(node) + 1
                        else
                            node = right(node) + 1
                        end if
                    else if (x(i, feature) < node_threshold(node)) then
                        node = left(node) + 1
                    else
                        node = right(node) + 1
                    end if
                end do
                value = value + rate*scales(tree)*weights(node)
            end do
            values(i) = value
        end do
    end function cpu_oracle

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures
        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL [CUDA boosted tree API] "//description
        end if
    end subroutine check

end program test_cuda_boosted_tree_api
