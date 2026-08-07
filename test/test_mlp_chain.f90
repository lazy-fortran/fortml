program test_mlp_chain
    !! Independent chain-rule, adjoint, HVP, optimizer, and device contracts.
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CUDA
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    use fortml_mlp_chain, only: mlp_chain_t, mlp_chain_objective_t, &
        mlp_chain_lbfgsb_options_t, mlp_chain_lbfgsb_result_t, &
        mlp_chain_optimize_lbfgsb
    use fortml_parameter_registry, only: parameter_block_t, parameter_registry_t, &
        parameter_block_from_mlp_chain
    use fortml_parameter_products, only: parameter_products_t, &
        parameter_products_from_mlp_chain
    implicit none

    integer :: failures

    failures = 0
    call test_composed_products(failures)
    call test_objective_and_lbfgsb(failures)
    call test_cuda_refusal(failures)
    if (failures /= 0) then
        write (error_unit, '(a,i0)') "FAIL MLP chain cases: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS MLP chain independent behavioral oracles"

contains

    subroutine test_composed_products(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: first, second
        type(mlp_chain_t), target :: chain
        type(parameter_block_t) :: chain_block
        type(parameter_registry_t) :: registry
        type(parameter_products_t) :: products
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2), dx(3, 2), u(3, 1)
        real(dp) :: y(3, 1), dy(3, 1), y_plus(3, 1), y_minus(3, 1)
        real(dp) :: y_products(3, 1), dy_products(3, 1), dy_fixed(3, 1)
        real(dp) :: zero_dx(3, 2), x_hvp_fixed(3, 2)
        real(dp), allocatable :: theta(:), theta_plus(:), theta_minus(:), dtheta(:)
        real(dp), allocatable :: parameter_bar(:), parameter_hvp(:), parameter_hvp_fixed(:)
        real(dp), allocatable :: plus_bar(:), minus_bar(:), products_bar(:), products_hvp(:)
        real(dp) :: x_bar(3, 2), x_hvp(3, 2), plus_x_bar(3, 2), minus_x_bar(3, 2)
        real(dp) :: epsilon, lhs, rhs
        integer :: first_range, last_range
        logical :: found

        call first%initialize([2, 2], status, output_activation=MLP_LINEAR)
        call second%initialize([2, 1], status, output_activation=MLP_LINEAR)
        call first%set_parameters([1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 0.5_dp, -0.25_dp], status)
        call second%set_parameters([1.5_dp, -0.75_dp, 0.2_dp], status)
        call chain%initialize(2, status)
        call chain%append(first, status, name="encoder")
        call chain%append(second, status, name="head")
        call check(status_ok(status) .and. chain%stage_count() == 2 .and. &
            chain%input_count() == 2 .and. chain%output_count() == 1, &
            "chain topology and dimensions", failures)
        call chain%parameter_range("head", first_range, last_range, found)
        call check(found .and. first_range == 7 .and. last_range == 9, &
            "deterministic named parameter range", failures)
        call parameter_block_from_mlp_chain(chain_block, "network", chain, status)
        call registry%add(chain_block, status)
        allocate(theta(chain%parameter_count()))
        call registry%pack(theta, status)
        call check(status_ok(status) .and. maxval(abs(theta - chain%parameters())) < 1.0e-14_dp, &
            "parameter registry routes the composed tree", failures)
        call parameter_products_from_mlp_chain(products, "network", chain, status)
        call check(status_ok(status) .and. products%initialized() .and. &
            products%has_hvp() .and. products%parameter_count() == chain%parameter_count(), &
            "parameter products expose the composed tree", failures)

        x = reshape([0.2_dp, -0.4_dp, 0.7_dp, 1.1_dp, -0.3_dp, 0.9_dp], shape(x))
        dx = reshape([-0.1_dp, 0.3_dp, 0.4_dp, -0.2_dp, 0.6_dp, 0.5_dp], shape(dx))
        zero_dx = 0.0_dp
        u(:, 1) = [0.7_dp, -0.4_dp, 0.9_dp]
        theta = chain%parameters()
        dtheta = [0.03_dp, -0.05_dp, 0.02_dp, 0.04_dp, -0.01_dp, 0.06_dp, &
            -0.02_dp, 0.07_dp, -0.03_dp]
        allocate(parameter_bar(size(theta)), parameter_hvp(size(theta)))
        allocate(plus_bar(size(theta)), minus_bar(size(theta)))
        call chain%predict(x, y, status)
        call chain%jvp(x, dtheta, dx, y, dy, status)
        allocate(products_bar(size(theta)), products_hvp(size(theta)))
        call products%value(x, y_products, status)
        call products%jvp(x, dtheta, y_products, dy_products, status)
        call products%vjp(x, u, products_bar, status)
        call products%hvp(x, u, dtheta, products_hvp, status)
        call chain%jvp(x, dtheta, zero_dx, y, dy_fixed, status)
        allocate(parameter_hvp_fixed(size(theta)))
        call chain%hvp(x, u, dtheta, zero_dx, parameter_hvp_fixed, x_hvp_fixed, status)
        call check(status_ok(status), "composed value/JVP status", failures)
        call check(maxval(abs(y_products - y)) < 1.0e-14_dp .and. &
            maxval(abs(dy_products - dy_fixed)) < 1.0e-14_dp, &
            "parameter products agree with chain products", failures)
        epsilon = 1.0e-6_dp
        theta_plus = theta + epsilon*dtheta
        theta_minus = theta - epsilon*dtheta
        call chain%set_parameters(theta_plus, status)
        call chain%predict(x + epsilon*dx, y_plus, status)
        call chain%set_parameters(theta_minus, status)
        call chain%predict(x - epsilon*dx, y_minus, status)
        call chain%set_parameters(theta, status)
        call check(maxval(abs(dy - (y_plus - y_minus)/(2.0_dp*epsilon))) < 2.0e-8_dp, &
            "chain JVP agrees with central oracle", failures)

        call chain%vjp(x, u, parameter_bar, x_bar, status)
        lhs = sum(u*dy)
        rhs = dot_product(parameter_bar, dtheta) + sum(x_bar*dx)
        call check(status_ok(status) .and. abs(lhs - rhs) < 2.0e-10_dp, &
            "chain VJP adjoint identity", failures)
        call check(maxval(abs(products_bar - parameter_bar)) < 1.0e-14_dp, &
            "parameter-products VJP agrees with chain", failures)

        call chain%hvp(x, u, dtheta, dx, parameter_hvp, x_hvp, status)
        call chain%set_parameters(theta_plus, status)
        call chain%vjp(x + epsilon*dx, u, plus_bar, plus_x_bar, status)
        call chain%set_parameters(theta_minus, status)
        call chain%vjp(x - epsilon*dx, u, minus_bar, minus_x_bar, status)
        call chain%set_parameters(theta, status)
        call check(status_ok(status) .and. &
            maxval(abs(parameter_hvp - (plus_bar - minus_bar)/(2.0_dp*epsilon))) < 2.0e-6_dp .and. &
            maxval(abs(x_hvp - (plus_x_bar - minus_x_bar)/(2.0_dp*epsilon))) < 2.0e-6_dp, &
            "chain HVP agrees with differentiated VJP oracle", failures)
        call check(maxval(abs(products_hvp - parameter_hvp_fixed)) < 1.0e-14_dp, &
            "parameter-products HVP agrees with chain", failures)
    end subroutine test_composed_products

    subroutine test_objective_and_lbfgsb(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: stage
        type(mlp_chain_t) :: chain
        type(mlp_chain_objective_t) :: objective
        type(mlp_chain_lbfgsb_options_t) :: options
        type(mlp_chain_lbfgsb_result_t) :: result
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 1), target(3, 1)
        real(dp), allocatable :: theta(:), gradient(:), direction(:), product(:)
        real(dp) :: value, tangent, value_plus, value_minus, epsilon

        call stage%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call stage%set_parameters([0.0_dp, 0.0_dp], status)
        call chain%initialize(1, status)
        call chain%append(stage, status)
        x(:, 1) = [-1.0_dp, 0.0_dp, 1.0_dp]
        target(:, 1) = x(:, 1)
        call objective%initialize(chain, x, target, 0.1_dp, status, optimize_l2=.true.)
        theta = objective%parameters()
        allocate(gradient(size(theta)), direction(size(theta)), product(size(theta)))
        direction = [0.07_dp, -0.04_dp, 0.13_dp]
        call objective%value_gradient(theta, value, gradient, status)
        call objective%jvp(theta, direction, value_plus, tangent, status)
        call check(status_ok(status) .and. abs(tangent - dot_product(gradient, direction)) < 1.0e-12_dp, &
            "objective scalar JVP uses analytic gradient", failures)
        epsilon = 1.0e-6_dp
        call objective%value_gradient(theta + epsilon*direction, value_plus, gradient, status)
        call objective%value_gradient(theta - epsilon*direction, value_minus, gradient, status)
        call check(abs(tangent - (value_plus - value_minus)/(2.0_dp*epsilon)) < 2.0e-8_dp, &
            "objective JVP central oracle", failures)
        call objective%hvp(theta, direction, product, status)
        call check(status_ok(status) .and. all(product == product), &
            "objective HVP finite", failures)

        call stage%set_parameters([0.0_dp, 0.0_dp], status)
        options%l2 = 0.1_dp
        options%optimize_l2 = .true.
        options%max_iterations = 200
        options%gradient_tolerance = 1.0e-8_dp
        options%step_tolerance = 1.0e-14_dp
        options%objective_tolerance = 1.0e-14_dp
        call mlp_chain_optimize_lbfgsb(chain, x, target, options, result, status)
        theta = chain%parameters()
        call check(status_ok(status) .and. result%converged .and. result%l2 <= 1.0e-8_dp .and. &
            abs(theta(1) - 1.0_dp) < 2.0e-6_dp .and. abs(theta(2)) < 2.0e-7_dp, &
            "chain FortOpt L-BFGS-B reaches analytic linear optimum", failures)
    end subroutine test_objective_and_lbfgsb

    subroutine test_cuda_refusal(failures)
        integer, intent(inout) :: failures
        type(mlp_t) :: stage
        type(mlp_chain_t) :: chain
        type(mlp_chain_objective_t) :: objective
        type(mlp_chain_lbfgsb_options_t) :: options
        type(mlp_chain_lbfgsb_result_t) :: result
        type(fortnum_status_t) :: status
        real(dp) :: x(1, 1), target(1, 1)

        call stage%initialize([1, 1], status, output_activation=MLP_LINEAR)
        call chain%initialize(1, status)
        call chain%append(stage, status)
        x = 0.0_dp
        target = 0.0_dp
        call check(.not. chain%device_supported(FORTML_DEVICE_CUDA), &
            "chain CUDA capability is explicitly absent", failures)
        call objective%initialize(chain, x, target, 0.0_dp, status, &
            device_kind=FORTML_DEVICE_CUDA)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "chain objective CUDA refusal", failures)
        options%device_kind = FORTML_DEVICE_CUDA
        call mlp_chain_optimize_lbfgsb(chain, x, target, options, result, status)
        call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
            "chain L-BFGS-B CUDA refusal", failures)
    end subroutine test_cuda_refusal

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "  FAIL [MLP chain] "//description
        end if
    end subroutine check

end program test_mlp_chain
