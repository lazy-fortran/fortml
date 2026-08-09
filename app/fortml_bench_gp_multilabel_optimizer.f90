program fortml_bench_gp_multilabel_optimizer
    !! Release probe for shared multilabel GP fixed-state kernel HPO.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gp_multilabel_classification, only: &
        gp_multilabel_classification_t, gp_multilabel_classification_options_t, &
        gp_multilabel_classification_state_t, &
        gp_multilabel_training_objective_t, gp_multilabel_lbfgsb_options_t, &
        gp_multilabel_lbfgsb_result_t, gp_multilabel_optimize_lbfgsb
    implicit none

    type(gp_multilabel_classification_t), target :: model
    type(gp_multilabel_classification_options_t) :: fit_options
    type(gp_multilabel_classification_state_t) :: fit_state
    type(gp_multilabel_training_objective_t) :: objective
    type(gp_multilabel_lbfgsb_options_t) :: optimizer_options
    type(gp_multilabel_lbfgsb_result_t) :: optimizer_result
    type(kernel_t) :: kernel
    type(fortnum_status_t) :: status
    real(dp) :: x(10, 1), parameters(2), direction(2), gradient(2)
    real(dp) :: objective_value, objective_value_plus, objective_value_minus
    real(dp) :: fd_plus, fd_minus
    real(dp) :: objective_tangent, objective_vjp(2), final_parameters(2), h
    integer :: indicators(10, 2), i, status_code

    x(:, 1) = [-2.0_dp, -1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, &
        0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp]
    indicators(:, 1) = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    indicators(:, 2) = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    kernel = make_rbf_kernel(1, 1.3_dp, 0.75_dp, status)
    fit_options%max_iterations = 100
    fit_options%tolerance = 1.0e-9_dp
    fit_options%jitter = 1.0e-7_dp
    call model%fit(x, indicators, kernel, status, fit_options, fit_state, &
        sample_weight=[1.0_dp, 0.9_dp, 1.1_dp, 1.0_dp, 0.8_dp, 1.2_dp, 1.0_dp, 1.1_dp, 0.9_dp, 1.0_dp])
    if (.not. status_ok(status)) error stop 1
    parameters = model%shared_parameters()
    direction = [0.17_dp, -0.11_dp]
    call model%fixed_state_value_gradient(parameters, objective_value, gradient, status)
    if (.not. status_ok(status)) error stop 2
    h = 1.0e-5_dp
    call model%fixed_state_value_gradient(parameters + h*direction, objective_value_plus, &
        objective_vjp, status)
    if (.not. status_ok(status)) error stop 3
    fd_plus = objective_value_plus
    call model%fixed_state_value_gradient(parameters - h*direction, objective_value_minus, &
        objective_vjp, status)
    if (.not. status_ok(status)) error stop 4
    fd_minus = objective_value_minus
    call model%set_shared_parameters(parameters, status)
    if (.not. status_ok(status)) error stop 5
    call objective%initialize(model, status)
    call objective%jvp(parameters, direction, objective_value_plus, objective_tangent, status)
    if (.not. status_ok(status)) error stop 6
    call objective%vjp(parameters, 1.0_dp, objective_vjp, status)
    if (.not. status_ok(status)) error stop 7

    optimizer_options%max_iterations = 200
    optimizer_options%max_line_search = 40
    optimizer_options%lower_bound = -1.0_dp
    optimizer_options%upper_bound = 1.0_dp
    optimizer_options%gradient_tolerance = 1.0e-3_dp
    optimizer_options%objective_tolerance = 1.0e-5_dp
    optimizer_options%step_tolerance = 1.0e-8_dp
    call gp_multilabel_optimize_lbfgsb(model, optimizer_options, optimizer_result, status)
    status_code = status%code
    final_parameters = model%shared_parameters()

    write (*, '(a,es24.16,a,es24.16)') 'gp_multilabel_optimizer_parameters,', &
        parameters(1), ',', parameters(2)
    write (*, '(a,es24.16)') 'gp_multilabel_optimizer_objective,', objective_value
    write (*, '(a,es24.16)') 'gp_multilabel_optimizer_fit_log_posterior,', fit_state%log_posterior
    write (*, '(a,es24.16,a,es24.16)') 'gp_multilabel_optimizer_gradient,', &
        gradient(1), ',', gradient(2)
    write (*, '(a,es24.16)') 'gp_multilabel_optimizer_jvp,', objective_tangent
    write (*, '(a,es24.16,a,es24.16)') 'gp_multilabel_optimizer_vjp,', &
        objective_vjp(1), ',', objective_vjp(2)
    write (*, '(a,es24.16,a,es24.16)') 'gp_multilabel_optimizer_fd_values,', &
        fd_plus, ',', fd_minus
    write (*, '(a,i0,a,l1,a,i0,a,es24.16,a,es24.16)') &
        'gp_multilabel_optimizer_result,', status_code, ',', optimizer_result%converged, &
        ',', optimizer_result%iterations, ',', optimizer_result%objective, ',', &
        optimizer_result%gradient_norm
    write (*, '(a,i0)') 'gp_multilabel_optimizer_labels,', size(indicators, 2)
    do i = 1, size(parameters)
        write (*, '(a,i0,a,es24.16)') 'gp_multilabel_optimizer_final_parameter,', i, ',', &
            final_parameters(i)
    end do
    write (*, '(a,i0)') 'gp_multilabel_optimizer_cuda,', &
        merge(1, 0, model%device_supported(FORTML_DEVICE_CUDA))
end program fortml_bench_gp_multilabel_optimizer
