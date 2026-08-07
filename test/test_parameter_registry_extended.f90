program test_parameter_registry_extended
    !! Independent oracle for registry adapters beyond the original MLP/GP set.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_parameter_registry, only: parameter_block_t, &
        parameter_block_from_ridge, parameter_block_from_elastic_net, &
        parameter_registry_t
    use fortml_ridge_regression, only: ridge_regression_t
    use fortml_elastic_net_regression, only: elastic_net_regression_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    type(ridge_regression_t), target :: ridge
    type(elastic_net_regression_t), target :: elastic
    type(parameter_block_t) :: ridge_block, elastic_block
    type(parameter_registry_t) :: registry
    type(fortnum_status_t) :: status
    real(dp) :: x(6, 2), y(6), ridge_values(3), elastic_values(3)
    real(dp) :: packed(6), updated(6)

    x = reshape([0.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, &
        1.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 1.0_dp, 0.0_dp], shape(x))
    y = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp]

    call ridge%fit(x, y, status, alpha=0.1_dp, fit_intercept=.true.)
    if (.not. status_ok(status)) error stop 1
    call elastic%fit(x, y, status, alpha=0.1_dp, l1_ratio=0.25_dp, &
        fit_intercept=.true.)
    if (.not. status_ok(status)) error stop 2
    if (ridge%parameter_count() /= 3 .or. elastic%parameter_count() /= 3) error stop 3

    call parameter_block_from_ridge(ridge_block, "ridge.coefficients", ridge, status)
    if (.not. status_ok(status)) error stop 4
    call parameter_block_from_elastic_net(elastic_block, "elastic.coefficients", &
        elastic, status)
    if (.not. status_ok(status)) error stop 5
    call registry%add(ridge_block, status)
    if (.not. status_ok(status)) error stop 6
    call registry%add(elastic_block, status)
    if (.not. status_ok(status)) error stop 7
    call registry%pack(packed, status)
    if (.not. status_ok(status)) error stop 8
    ridge_values = ridge%parameters()
    elastic_values = elastic%parameters()
    if (maxval(abs(packed(:3)-ridge_values)) > 1.0e-13_dp .or. &
        maxval(abs(packed(4:)-elastic_values)) > 1.0e-13_dp) error stop 9

    updated = packed + [0.1_dp, -0.2_dp, 0.05_dp, -0.1_dp, 0.15_dp, -0.04_dp]
    call registry%unpack(updated, status)
    if (.not. status_ok(status)) error stop 10
    if (maxval(abs(ridge%parameters()-updated(:3))) > 1.0e-13_dp .or. &
        maxval(abs(elastic%parameters()-updated(4:))) > 1.0e-13_dp) error stop 11
    write (*, '(a)') "PASS extended parameter registry independent packing oracle"
end program test_parameter_registry_extended
