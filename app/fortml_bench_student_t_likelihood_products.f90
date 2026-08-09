program fortml_bench_student_t_likelihood_products
    !! Release probe for fixed-latent Student-t likelihood products.
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_CONVERGENCE_ERROR
    use fortml_student_t_likelihood, only: student_t_likelihood_t, &
        student_t_log_likelihood_value, student_t_log_likelihood_gradient, &
        student_t_log_likelihood_jvp, student_t_log_likelihood_vjp, &
        student_t_log_likelihood_hvp
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none

    type(student_t_likelihood_t), target :: model
    type(objective_t) :: objective
    type(lbfgsb_t) :: optimizer
    type(lbfgsb_options_t) :: options
    type(lbfgsb_result_t) :: result
    type(fortnum_status_t) :: status
    real(dp) :: observations(7), locations(7), parameters(2), direction(2)
    real(dp) :: value, tangent, gradient(2), parameter_bar(2), product(2)
    real(dp) :: lower(2), upper(2), initial_value, optimized_value
    integer :: i

    do i = 1, 7
        locations(i) = -1.2_dp + 0.4_dp*real(i - 1, dp)
        observations(i) = locations(i) + 0.25_dp*sin(1.7_dp*locations(i))
    end do
    parameters = [log(0.75_dp), log(4.3_dp)]
    direction = [0.35_dp, -0.6_dp]
    call model%initialize(observations, locations, status, parameters)
    if (.not. status_ok(status)) error stop "Student-t likelihood initialization failed"
    call student_t_log_likelihood_value(observations, locations, parameters, value, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood value failed"
    call student_t_log_likelihood_gradient(observations, locations, parameters, gradient, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood gradient failed"
    call student_t_log_likelihood_jvp(observations, locations, parameters, direction, &
        value, tangent, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood JVP failed"
    call student_t_log_likelihood_vjp(observations, locations, parameters, 1.0_dp, &
        parameter_bar, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood VJP failed"
    call student_t_log_likelihood_hvp(observations, locations, parameters, direction, &
        product, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood HVP failed"
    write (*, '(a,es24.16)') "student_t_observation_value,", value
    write (*, '(a,es24.16)') "student_t_observation_gradient_scale,", gradient(1)
    write (*, '(a,es24.16)') "student_t_observation_gradient_nu,", gradient(2)
    write (*, '(a,es24.16)') "student_t_observation_jvp,", tangent
    write (*, '(a,es24.16)') "student_t_observation_vjp_scale,", parameter_bar(1)
    write (*, '(a,es24.16)') "student_t_observation_vjp_nu,", parameter_bar(2)
    write (*, '(a,es24.16)') "student_t_observation_hvp_scale,", product(1)
    write (*, '(a,es24.16)') "student_t_observation_hvp_nu,", product(2)

    call model%fortopt(objective, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood FortOpt context failed"
    call objective%value_gradient(parameters, initial_value, gradient, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood objective failed"
    lower = [-4.0_dp, -4.0_dp]
    upper = [4.0_dp, 6.0_dp]
    options%max_iterations = 40
    options%gradient_tolerance = 1.0e-8_dp
    options%objective_tolerance = 1.0e-12_dp
    call optimizer%minimize(objective, parameters, lower, upper, options, result, status)
    if (status%code /= 0 .and. status%code /= FORTNUM_CONVERGENCE_ERROR) then
        error stop "Student-t likelihood FortOpt optimization failed"
    end if
    call objective%value_gradient(parameters, optimized_value, gradient, status)
    if (.not. status_ok(status)) error stop "Student-t likelihood optimized objective failed"
    write (*, '(a,es24.16)') "student_t_observation_initial_objective,", initial_value
    write (*, '(a,es24.16)') "student_t_observation_optimized_objective,", optimized_value
    write (*, '(a,i0)') "student_t_observation_optimizer_iterations,", result%state%iteration
    write (*, '(a,i0)') "student_t_observation_optimizer_converged,", &
        merge(1, 0, result%state%converged)
end program fortml_bench_student_t_likelihood_products
