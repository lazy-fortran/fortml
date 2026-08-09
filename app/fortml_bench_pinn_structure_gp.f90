program fortml_bench_pinn_structure_gp
    !! Release protocol for the finite-feature PINN structure-GP initializer.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_ok, status_set, &
        FORTNUM_OK, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_TANH, MLP_LINEAR
    use fortml_physics_objective, only: physics_constraint_t, physics_objective_t
    use fortml_pinn_structure_gp, only: pinn_structure_gp_initializer_t, &
        pinn_structure_gp_metadata_t
    implicit none

    integer, parameter :: dp = real64
    type(mlp_t) :: model
    type(pinn_structure_gp_initializer_t) :: initializer
    type(pinn_structure_gp_metadata_t) :: metadata
    type(physics_constraint_t) :: data, residual, boundary, conservation
    type(physics_objective_t) :: objective
    type(fortnum_status_t) :: status
    real(dp) :: x(8, 1), target(8, 1), prediction(8, 1), terms(4)
    real(dp), allocatable :: before(:), after(:), hidden_before(:), hidden_after(:)
    integer :: i, hidden_count

    do i = 1, size(x, 1)
        x(i, 1) = -1.0_dp + 2.0_dp*real(i - 1, dp)/real(size(x, 1) - 1, dp)
        target(i, 1) = sin(acos(-1.0_dp)*x(i, 1)) + 0.2_dp*x(i, 1)**2
    end do
    call model%initialize([1, 3, 1], status, hidden_activation=MLP_TANH, &
        output_activation=MLP_LINEAR, initialization_seed=41)
    call require(status, "model initialization")
    call data%initialize(model%parameter_count(), 1, 1.0_dp, &
        residual_proc=probe_residual, jvp_proc=probe_jvp, vjp_proc=probe_vjp, status=status)
    call require(status, "data constraint")
    call residual%initialize(model%parameter_count(), 1, 2.0_dp, &
        residual_proc=probe_residual, jvp_proc=probe_jvp, vjp_proc=probe_vjp, status=status)
    call require(status, "residual constraint")
    call boundary%initialize(model%parameter_count(), 1, 0.5_dp, &
        residual_proc=probe_residual, jvp_proc=probe_jvp, vjp_proc=probe_vjp, status=status)
    call require(status, "boundary constraint")
    call conservation%initialize(model%parameter_count(), 1, 1.5_dp, &
        residual_proc=probe_residual, jvp_proc=probe_jvp, vjp_proc=probe_vjp, status=status)
    call require(status, "conservation constraint")
    call objective%initialize(model%parameter_count(), data, residual, boundary, conservation, status)
    call require(status, "objective initialization")

    before = model%parameters()
    hidden_count = size(before) - 4
    allocate(hidden_before(hidden_count))
    hidden_before = before(1:hidden_count)
    call initializer%fit(model, x, target, objective, status, regularization=0.07_dp)
    call require(status, "PINN structure-GP fit")
    metadata = initializer%metadata()
    call initializer%apply(model, status)
    call require(status, "PINN structure-GP apply")
    after = model%parameters()
    allocate(hidden_after(hidden_count))
    hidden_after = after(1:hidden_count)
    call model%predict(x, prediction, status)
    call require(status, "prediction")
    call initializer%objective_terms(objective, model, terms, status)
    call require(status, "named objective terms")

    write (*, '(a,",",es24.16)') "pinn_structure_gp_prediction_mean", sum(prediction)/real(size(prediction), dp)
    write (*, '(a,",",es24.16)') "pinn_structure_gp_objective_before", metadata%objective_before
    write (*, '(a,",",es24.16)') "pinn_structure_gp_objective_after", metadata%objective_after
    write (*, '(a,",",es24.16)') "pinn_structure_gp_structure_defect", metadata%structure_defect
    write (*, '(a,",",es24.16)') "pinn_structure_gp_hidden_delta", maxval(abs(hidden_after-hidden_before))
    write (*, '(a,",",es24.16)') "pinn_structure_gp_residual_term", terms(2)
    call initializer%apply_cuda(model, status)
    if (status%code /= FORTNUM_NOT_IMPLEMENTED) error stop "PINN CUDA contract changed"
    write (*, '(a)') "pinn_structure_gp_cuda,unavailable"

contains

    subroutine require(value, label)
        type(fortnum_status_t), intent(in) :: value
        character(*), intent(in) :: label
        if (.not. status_ok(value)) error stop trim(label)
    end subroutine require

    subroutine probe_residual(context, theta, residual_value, value_status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:)
        real(dp), intent(out) :: residual_value(:)
        type(fortnum_status_t), intent(out) :: value_status
        associate (unused_context => context)
        end associate
        residual_value(1) = theta(size(theta)) - 0.10_dp
        call status_set(value_status, FORTNUM_OK, "")
    end subroutine probe_residual

    subroutine probe_jvp(context, theta, theta_dot, residual_value, residual_dot, value_status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), theta_dot(:)
        real(dp), intent(out) :: residual_value(:), residual_dot(:)
        type(fortnum_status_t), intent(out) :: value_status
        call probe_residual(context, theta, residual_value, value_status)
        residual_dot(1) = theta_dot(size(theta_dot))
    end subroutine probe_jvp

    subroutine probe_vjp(context, theta, residual_bar, theta_bar, value_status)
        class(*), pointer, intent(in) :: context
        real(dp), intent(in) :: theta(:), residual_bar(:)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: value_status
        associate (unused_context => context)
        end associate
        associate (unused_theta => theta)
        end associate
        theta_bar = 0.0_dp
        theta_bar(size(theta_bar)) = residual_bar(1)
        call status_set(value_status, FORTNUM_OK, "")
    end subroutine probe_vjp

end program fortml_bench_pinn_structure_gp
