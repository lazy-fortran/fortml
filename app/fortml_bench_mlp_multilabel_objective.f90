program fortml_bench_mlp_multilabel_objective
    !! Release probe for weighted multilabel MLP objective products.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_mlp_multilabel_classifier, only: &
        mlp_multilabel_classifier_t, mlp_multilabel_classifier_options_t, &
        mlp_multilabel_training_objective_t, mlp_multilabel_lbfgsb_options_t, &
        mlp_multilabel_lbfgsb_result_t, mlp_multilabel_optimize_lbfgsb
    implicit none

    integer, parameter :: n = 6, p = 2, labels_count = 2
    real(dp) :: x(n, p), sample_weight(n), class_weight(2, labels_count)
    integer :: indicators(n, labels_count), i
    real(dp), allocatable :: theta(:), direction(:), gradient(:), hvp(:)
    real(dp) :: value, tangent
    type(mlp_multilabel_classifier_t), target :: model
    type(mlp_multilabel_classifier_options_t) :: fit_options
    type(mlp_multilabel_training_objective_t) :: objective
    type(mlp_multilabel_lbfgsb_options_t) :: optimizer_options
    type(mlp_multilabel_lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status

    x(:, 1) = [-1.0_dp, -0.5_dp, 0.0_dp, 0.5_dp, 1.0_dp, 1.2_dp]
    x(:, 2) = [-1.0_dp, -0.2_dp, 0.0_dp, 0.2_dp, 1.0_dp, 0.8_dp]
    indicators(:, 1) = [0, 0, 0, 1, 1, 1]
    indicators(:, 2) = [0, 1, 0, 1, 0, 1]
    sample_weight = [0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp, 0.75_dp, 1.25_dp]
    class_weight = reshape([1.1_dp, 0.8_dp, 0.9_dp, 1.4_dp], shape(class_weight))
    fit_options%max_epochs = 1
    fit_options%learning_rate = 0.03_dp
    fit_options%beta1 = 0.8_dp
    fit_options%beta2 = 0.95_dp
    fit_options%epsilon = 1.0e-7_dp
    fit_options%l2 = 0.02_dp
    fit_options%tolerance = 0.0_dp
    fit_options%restore_best = .false.
    fit_options%initialization_seed = 29
    call model%fit(x, indicators, status, options=fit_options)
    if (.not. status_ok(status)) error stop "multilabel objective app fit failed"
    call objective%initialize(model, x, indicators, 0.02_dp, status, &
        optimize_log_l2=.true., sample_weight=sample_weight, class_weight=class_weight)
    if (.not. status_ok(status)) error stop "multilabel objective app initialization failed"
    theta = objective%parameters()
    allocate(direction(size(theta)), gradient(size(theta)), hvp(size(theta)))
    direction = [(0.01_dp*real(i, dp), i=1, size(theta))]
    call objective%value_gradient(theta, value, gradient, status)
    if (.not. status_ok(status)) error stop "multilabel objective app gradient failed"
    call objective%jvp(theta, direction, value, tangent, status)
    if (.not. status_ok(status)) error stop "multilabel objective app JVP failed"
    call objective%hvp(theta, direction, hvp, status)
    if (.not. status_ok(status)) error stop "multilabel objective app HVP failed"
    write (*, '(a,es24.16)') "mlp_multilabel_objective_value,", value
    write (*, '(a,es24.16)') "mlp_multilabel_objective_jvp,", tangent
    write (*, '(a,es24.16)') "mlp_multilabel_objective_gradient_norm,", &
        sqrt(sum(gradient*gradient))
    write (*, '(a,es24.16)') "mlp_multilabel_objective_hvp_norm,", &
        sqrt(sum(hvp*hvp))
    optimizer_options%max_iterations = 120
    optimizer_options%max_line_search = 60
    optimizer_options%gradient_tolerance = 1.0e-2_dp
    optimizer_options%l2 = 0.02_dp
    optimizer_options%optimize_log_l2 = .true.
    optimizer_options%log_l2_lower_bound = -8.0_dp
    optimizer_options%log_l2_upper_bound = 1.0_dp
    call mlp_multilabel_optimize_lbfgsb(model, x, indicators, optimizer_options, &
        result, status, sample_weight=sample_weight, class_weight=class_weight)
    write (*, '(a,i0)') "mlp_multilabel_objective_optimizer_status,", status%code
    write (*, '(a,l1)') "mlp_multilabel_objective_optimizer_converged,", result%converged
    write (*, '(a,es24.16)') "mlp_multilabel_objective_optimizer_l2,", result%l2
end program fortml_bench_mlp_multilabel_objective
