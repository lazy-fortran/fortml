program fortml_bench_softmax_training
    !! Machine-readable release protocol for the weighted softmax objective.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortml_softmax_regression, only: softmax_regression_t
    use fortml_softmax_training, only: softmax_training_objective_t, &
        softmax_lbfgsb_options_t, softmax_lbfgsb_result_t, &
        softmax_optimize_lbfgsb
    use fortnum_status, only: fortnum_status_t, status_ok, &
        FORTNUM_NOT_IMPLEMENTED
    implicit none

    type(softmax_regression_t), target :: model
    type(softmax_training_objective_t) :: objective
    type(softmax_lbfgsb_options_t) :: options
    type(softmax_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    real(dp) :: x(5, 2), parameters(10), direction(10)
    real(dp) :: gradient(10), product(10), value
    real(dp) :: sample_weight(5), class_weight(3)
    integer :: labels(5), i

    x = reshape([ &
        -1.0_dp, 0.2_dp, 0.7_dp, 1.1_dp, -0.3_dp, -0.8_dp, &
        1.3_dp, -0.4_dp, 0.2_dp, 0.9_dp], shape(x))
    labels = [-3, 4, 9, 4, -3]
    sample_weight = [1.0_dp, 2.0_dp, 0.5_dp, 1.2_dp, 0.7_dp]
    class_weight = [1.0_dp, 2.0_dp, 3.0_dp]
    parameters = [ &
        0.20_dp, -0.10_dp, -0.30_dp, 0.15_dp, 0.25_dp, -0.20_dp, &
        0.10_dp, -0.05_dp, 0.30_dp, 0.35_dp]
    direction = [ &
        -0.13_dp, 0.08_dp, 0.11_dp, -0.09_dp, 0.07_dp, 0.12_dp, &
        -0.06_dp, 0.15_dp, -0.10_dp, 0.04_dp]

    call model%fit(x, labels, status, l2=0.2_dp, max_iterations=600, &
        tolerance=1.0e-8_dp)
    if (.not. status_ok(status)) error stop 1
    call objective%initialize(model, x, labels, 0.35_dp, status, &
        optimize_l2=.true., sample_weight=sample_weight, &
        class_weight=class_weight)
    if (.not. status_ok(status)) error stop 1
    call objective%value_gradient(parameters, value, gradient, status)
    if (.not. status_ok(status)) error stop 1
    write (*, '(a,",",es24.16,",",es24.16,10(",",es24.16))') &
        "softmax_objective", value, sqrt(sum(gradient**2)), gradient
    call objective%hvp(parameters, direction, product, status)
    if (.not. status_ok(status)) error stop 1
    write (*, '(a,10(",",es24.16))') "softmax_hvp", product

    options%l2 = 0.35_dp
    options%max_iterations = 300
    options%gradient_tolerance = 1.0e-7_dp
    options%lower_bound = -8.0_dp
    options%upper_bound = 8.0_dp
    call softmax_optimize_lbfgsb(model, x, labels, options, result, status, &
        sample_weight=sample_weight, class_weight=class_weight)
    if (.not. status_ok(status)) error stop 1
    write (*, '(a,",",l1,",",i0,",",es24.16,",",es24.16)') &
        "softmax_fit", result%converged, result%iterations, &
        result%objective, result%gradient_norm

    ! There is no resident softmax-training CUDA dispatch in this release.
    ! Keep the typed capability code in the protocol; do not imply host work.
    i = FORTNUM_NOT_IMPLEMENTED
    write (*, '(a,",",a,",",i0)') "softmax_cuda", "unavailable", i
end program fortml_bench_softmax_training
