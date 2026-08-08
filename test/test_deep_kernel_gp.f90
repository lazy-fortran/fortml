program test_deep_kernel_gp
    !! Deep kernel learning (arXiv:1511.02222), checked against oracles that do
    !! not come from the same derivation as the code.
    !!
    !! The load-bearing check is the gradient one. `weight_gradient` implements
    !! the paper's chain rule -- equation (7) contracted with dK/dg and then
    !! backpropagated -- and the only way to know that factorization is right is
    !! to compare it against the quantity it claims to be: a central difference
    !! of the log marginal likelihood, taken by actually perturbing a weight,
    !! re-mapping the inputs and refitting. That oracle knows nothing about how
    !! the analytic gradient is assembled, so it catches a wrong contraction, a
    !! transposed adjoint, or a missing factor of two, none of which a
    !! self-consistency check would notice.
    !!
    !! The second oracle is a reduction: with an identity feature map, a deep
    !! kernel GP *is* an ordinary GP on the inputs, so its posterior must equal
    !! one. That pins the composition -- it catches the base kernel being
    !! applied to inputs instead of features, which is the mistake the
    !! constructor's width check exists to prevent and which would otherwise
    !! produce a model that runs and means nothing.

    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK, FORTNUM_DOMAIN_ERROR
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_deep_kernel_gp, only: deep_kernel_gp_t
    use fortml_mlp, only: MLP_TANH
    implicit none

    integer :: failures

    failures = 0
    call check_identity_features_reduce_to_a_plain_gp(failures)
    call check_weight_gradient_matches_finite_differences(failures)
    call check_features_are_learned_not_frozen(failures)
    call check_refusals(failures)

    if (failures == 0) then
        print *, "test_deep_kernel_gp: PASS"
    else
        print *, "test_deep_kernel_gp: FAIL", failures
        error stop 1
    end if

contains

    subroutine training_set(x, y)
        real(dp), intent(out) :: x(:, :), y(:, :)
        integer :: k

        do k = 1, size(x, 1)
            x(k, 1) = -1.5_dp + 0.31_dp*real(k, dp)
            x(k, 2) = 0.4_dp*cos(0.7_dp*real(k, dp))
            y(k, 1) = sin(1.3_dp*x(k, 1)) + 0.5_dp*x(k, 2)
        end do
    end subroutine training_set

    !! A deep kernel GP whose feature map is the identity must agree exactly
    !! with a plain GP on the same inputs. If it does not, the composition is
    !! wrong somewhere -- most likely the base kernel is seeing the wrong space.
    subroutine check_identity_features_reduce_to_a_plain_gp(failures)
        integer, intent(inout) :: failures
        type(deep_kernel_gp_t) :: deep
        type(gp_regression_t) :: plain
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(12, 2), y(12, 1), query(5, 2)
        real(dp) :: deep_mean(5, 1), deep_variance(5)
        real(dp) :: plain_mean(5, 1), plain_variance(5)
        real(dp) :: identity(6)
        integer :: k

        call training_set(x, y)
        do k = 1, 5
            query(k, 1) = -1.0_dp + 0.5_dp*real(k, dp)
            query(k, 2) = 0.2_dp*real(k, dp) - 0.3_dp
        end do

        kernel = make_rbf_kernel(2, 1.1_dp, 0.8_dp, status)
        call expect(status%code == FORTNUM_OK, "the base kernel builds", failures)

        ! A single linear layer, set to the identity: weights are the 2x2
        ! identity and the biases are zero.
        call deep%initialize([2, 2], kernel, status)
        call expect(status%code == FORTNUM_OK, &
            "a deep kernel GP with a linear feature map initializes", failures)
        if (status%code /= FORTNUM_OK) return

        identity = [1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 0.0_dp]
        call deep%features%set_parameters(identity, status)
        call expect(status%code == FORTNUM_OK, "the identity map is settable", &
            failures)

        call deep%fit(x, y, 1.0e-6_dp, status, jitter=0.0_dp)
        call expect(status%code == FORTNUM_OK, "the deep kernel GP fits", failures)
        if (status%code /= FORTNUM_OK) return
        call deep%predict(query, deep_mean, deep_variance, status)
        call expect(status%code == FORTNUM_OK, "the deep kernel GP predicts", &
            failures)

        call plain%fit(x, y, kernel, 1.0e-6_dp, status, jitter=0.0_dp)
        call plain%predict(query, plain_mean, plain_variance, status)
        call expect(status%code == FORTNUM_OK, "the plain GP predicts", failures)

        call expect(maxval(abs(deep_mean(:, 1) - plain_mean(:, 1))) < 1.0e-12_dp, &
            "an identity feature map reproduces the plain GP mean", failures)
        call expect(maxval(abs(deep_variance - plain_variance)) < 1.0e-12_dp, &
            "an identity feature map reproduces the plain GP variance", failures)
    end subroutine check_identity_features_reduce_to_a_plain_gp

    !! The paper's chain rule against a central difference of the quantity it
    !! differentiates. This is the check that the factorization is right.
    subroutine check_weight_gradient_matches_finite_differences(failures)
        integer, intent(inout) :: failures
        type(deep_kernel_gp_t) :: deep
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(10, 2), y(10, 1)
        real(dp), allocatable :: analytic(:), weights(:), perturbed(:)
        real(dp) :: step, plus, minus, numeric, worst, scale
        integer :: n_parameters, k, checked

        call training_set(x, y)
        kernel = make_rbf_kernel(3, 1.0_dp, 0.9_dp, status)
        call deep%initialize([2, 4, 3], kernel, status, &
            hidden_activation=MLP_TANH, initialization_seed=7)
        call expect(status%code == FORTNUM_OK, &
            "a two-layer feature map initializes", failures)
        if (status%code /= FORTNUM_OK) return

        call deep%fit(x, y, 1.0e-4_dp, status, jitter=0.0_dp)
        call expect(status%code == FORTNUM_OK, "the model fits", failures)
        if (status%code /= FORTNUM_OK) return

        n_parameters = deep%features%parameter_count()
        allocate (analytic(n_parameters), weights(n_parameters))
        allocate (perturbed(n_parameters))

        call deep%weight_gradient(x, y, analytic, status)
        call expect(status%code == FORTNUM_OK, "the weight gradient computes", &
            failures)
        if (status%code /= FORTNUM_OK) return

        weights = deep%features%parameters()

        ! The step is chosen, not guessed. A central difference carries
        ! truncation error growing as h^2 and cancellation error growing as
        ! 1/h, and the likelihood here is evaluated through a Cholesky of a
        ! matrix with 1e-4 noise, so the cancellation term dominates early. At
        ! h = 1e-6 the disagreement was 3e-5 and at 1e-5 and 1e-4 it vanishes
        ! -- the 1/h signature of roundoff rather than of a wrong gradient,
        ! which is how it was told apart from the genuine factor-of-two error
        ! this check found first.
        !
        ! Every weight, not a sample of them. A gradient that is right for the
        ! first layer and wrong for the second is a realistic failure, and
        ! spot-checking one coordinate would miss it.
        step = 1.0e-5_dp
        worst = 0.0_dp
        checked = 0
        do k = 1, n_parameters
            perturbed = weights
            perturbed(k) = weights(k) + step
            call refit_and_score(deep, perturbed, x, y, plus, status)
            if (status%code /= FORTNUM_OK) cycle
            perturbed(k) = weights(k) - step
            call refit_and_score(deep, perturbed, x, y, minus, status)
            if (status%code /= FORTNUM_OK) cycle

            numeric = (plus - minus)/(2.0_dp*step)
            ! Relative to the gradient's own scale, so a large component and a
            ! near-zero one are held to comparable standards.
            scale = max(1.0_dp, abs(numeric))
            worst = max(worst, abs(numeric - analytic(k))/scale)
            checked = checked + 1
        end do

        ! Restore, so a later check does not inherit a perturbed network.
        call deep%features%set_parameters(weights, status)

        call expect(checked == n_parameters, &
            "every weight was differenced, not a sample of them", failures)
        call expect(worst < 1.0e-5_dp, &
            "the analytic weight gradient matches central differences", failures)
        if (worst >= 1.0e-5_dp) print *, "    worst relative error:", worst
    end subroutine check_weight_gradient_matches_finite_differences

    !! Set the weights, remap, refit, and report the log marginal likelihood.
    subroutine refit_and_score(deep, weights, x, y, value, status)
        type(deep_kernel_gp_t), intent(inout) :: deep
        real(dp), intent(in) :: weights(:), x(:, :), y(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        call deep%features%set_parameters(weights, status)
        if (status%code /= FORTNUM_OK) return
        call deep%fit(x, y, 1.0e-4_dp, status, jitter=0.0_dp)
        if (status%code /= FORTNUM_OK) return
        call deep%log_marginal_likelihood(value, status)
    end subroutine refit_and_score

    !! The features must actually depend on the weights, and the gradient must
    !! be non-trivial. A feature map that collapsed to a constant would pass a
    !! gradient check vacuously -- both sides would be zero.
    subroutine check_features_are_learned_not_frozen(failures)
        integer, intent(inout) :: failures
        type(deep_kernel_gp_t) :: deep
        type(kernel_t) :: kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(10, 2), y(10, 1)
        real(dp) :: features(10, 3), moved(10, 3)
        real(dp), allocatable :: weights(:), gradient(:)
        integer :: n_parameters

        call training_set(x, y)
        kernel = make_rbf_kernel(3, 1.0_dp, 0.9_dp, status)
        call deep%initialize([2, 4, 3], kernel, status, initialization_seed=11)
        call deep%fit(x, y, 1.0e-4_dp, status, jitter=0.0_dp)
        call expect(status%code == FORTNUM_OK, "the model fits", failures)
        if (status%code /= FORTNUM_OK) return

        call deep%transform(x, features, status)
        call expect(status%code == FORTNUM_OK, "the feature map is exposed", &
            failures)

        n_parameters = deep%features%parameter_count()
        allocate (weights(n_parameters), gradient(n_parameters))
        call deep%weight_gradient(x, y, gradient, status)
        call expect(maxval(abs(gradient)) > 1.0e-8_dp, &
            "the weight gradient is not identically zero", failures)

        weights = deep%features%parameters()
        weights = weights + 0.25_dp
        call deep%features%set_parameters(weights, status)
        call deep%transform(x, moved, status)
        call expect(maxval(abs(moved - features)) > 1.0e-6_dp, &
            "the features move when the weights move", failures)
    end subroutine check_features_are_learned_not_frozen

    subroutine check_refusals(failures)
        integer, intent(inout) :: failures
        type(deep_kernel_gp_t) :: deep, unfitted
        type(kernel_t) :: kernel, wrong_kernel
        type(fortnum_status_t) :: status
        real(dp) :: x(10, 2), y(10, 1), mean(2, 1), variance(2), value
        real(dp) :: gradient(4)

        call training_set(x, y)
        kernel = make_rbf_kernel(3, 1.0_dp, 0.9_dp, status)

        ! The base kernel must take the *feature* width. A kernel sized to the
        ! inputs would compose without complaint and model the wrong space.
        wrong_kernel = make_rbf_kernel(2, 1.0_dp, 0.9_dp, status)
        call deep%initialize([2, 4, 3], wrong_kernel, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a base kernel sized to the inputs is refused", failures)

        call deep%initialize([2], kernel, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a feature map with no output layer is refused", failures)

        call deep%initialize([2, 0, 3], kernel, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a layer with no units is refused", failures)

        call unfitted%initialize([2, 4, 3], kernel, status)
        call unfitted%predict(x(1:2, :), mean, variance, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "predicting before fitting is refused", failures)
        call unfitted%log_marginal_likelihood(value, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "scoring before fitting is refused", failures)
        call unfitted%weight_gradient(x, y, gradient, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "differentiating before fitting is refused", failures)

        call unfitted%fit(x, y, -1.0_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "a non-positive noise variance is refused", failures)

        call unfitted%fit(x(:, 1:1), y, 1.0e-4_dp, status)
        call expect(status%code == FORTNUM_DOMAIN_ERROR, &
            "training inputs of the wrong width are refused", failures)
    end subroutine check_refusals

    subroutine expect(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            print *, "  FAIL: ", description
        end if
    end subroutine expect

end program test_deep_kernel_gp
