program fortml_bench_scalable_gp
    !! Matched benchmark of every scalable-GP method in this repository, on the
    !! fixture of Liu, Ong, Shen and Cai (IEEE TNNLS 31(11):4405-4423, 2020).
    !!
    !! One row per method: wall time for training, wall time for prediction,
    !! peak resident memory of the process, and the standardized mean squared
    !! error and mean negative log predictive density against a held-out set.
    !! Accuracy is measured against the truth, so an approximation that is fast
    !! and wrong is visible as such.
    !!
    !! Usage: fortml_bench_scalable_gp <method> <n> <m> <M> <d> <repetitions>
    !! where `m` is the inducing/subset size (the maximum total SKI grid
    !! budget in `d > 1`) and `M` the expert count.
    !! The method names are the paper's: full, sod, sor, dtc, fitc, pitc, vfe,
    !! ski, nle, poe, gpoe, bcm, rbcm, grbcm, moe, keops. Every local method
    !! also has a `_clustered` variant, for example `poe_clustered`.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64, output_unit
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortml_gaussian_process, only: gp_regression_t
    use fortml_review_toy, only: review_toy_data, review_toy_grid, &
        review_toy_truth, REVIEW_TOY_NOISE_VARIANCE
    use fortml_sparse_prior_gp, only: sparse_prior_gp_t, SPARSE_SOR, &
        SPARSE_DTC, SPARSE_FITC, SPARSE_PITC
    use fortml_sparse_gp, only: sparse_gp_t
    use fortml_ski_gp, only: ski_operator_t, subset_of_data_indices
    use fortml_local_experts, only: local_expert_gp_t, AGGREGATE_NLE, &
        AGGREGATE_POE, AGGREGATE_GPOE, AGGREGATE_BCM, AGGREGATE_RBCM, &
        AGGREGATE_GRBCM, AGGREGATE_MOE
    use fortml_kernel_operator, only: kernel_operator_t
    use fortml_lanczos, only: lanczos_predictive_variance
    use fortnum_krylov, only: KRYLOV_OK
    use fortnum_status, only: fortnum_status_t, status_ok
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    implicit none

    real(dp), parameter :: SIGNAL_VARIANCE = 1.0_dp
    real(dp), parameter :: LENGTHSCALE = 1.0_dp
    !! Lanczos steps for the LOVE predictive variance of the matrix-free lanes.
    integer, parameter :: LANCZOS_STEPS = 20
    !! Largest dense covariance this benchmark will try to allocate.
    real(dp), parameter :: DENSE_BUDGET_GB = 8.0_dp
    !! Above this sample count the matrix-free lanes report a mean only: the
    !! LOVE variance costs one Lanczos sweep per test point, which is a
    !! separate measurement rather than part of the prediction cost.
    integer, parameter :: LOVE_SAMPLE_LIMIT = 2048

    character(len=32) :: method, argument
    integer :: n_samples, n_inducing, n_experts, n_features, repetitions
    real(dp), allocatable :: x(:, :), y(:), test_points(:, :), truth(:)
    real(dp), allocatable :: mean(:), variance(:)
    real(dp) :: train_seconds, predict_seconds, smse, mnlpd, peak_kb
    integer :: n_test, repetition
    integer(int64) :: clock_start, clock_end, clock_rate
    type(fortnum_status_t) :: status

    call get_command_argument(1, method)
    if (len_trim(method) == 0) then
        write (output_unit, '(a)') &
            "usage: fortml_bench_scalable_gp <method> <n> <m> <M> <d> <reps>"
        stop 1
    end if
    n_samples = integer_argument(2, 1024)
    n_inducing = integer_argument(3, 64)
    n_experts = integer_argument(4, 8)
    n_features = integer_argument(5, 1)
    repetitions = integer_argument(6, 3)
    n_test = 256

    call refuse_infeasible()
    call build_problem()
    allocate(mean(n_test), variance(n_test))

    train_seconds = huge(1.0_dp)
    predict_seconds = huge(1.0_dp)
    do repetition = 1, repetitions
        call run_once()
    end do
    call accuracy(mean, variance, smse, mnlpd)
    peak_kb = peak_resident_kilobytes()

    ! method,n,m,M,d,train_s,predict_s,peak_kb,smse,mnlpd
    write (output_unit, '(a,a,i0,a,i0,a,i0,a,i0,a,es16.8,a,es16.8,a,f14.1,a,es16.8,a,es16.8)') &
        trim(method), ",", n_samples, ",", n_inducing, ",", n_experts, ",", &
        n_features, ",", train_seconds, ",", predict_seconds, ",", peak_kb, &
        ",", smse, ",", mnlpd

contains

    subroutine refuse_infeasible()
        !! Refuse known-infeasible dense storage, tensor-grid budgets, and
        !! expert counts before allocating or entering a timed region.
        real(dp) :: gigabytes

        gigabytes = 8.0_dp*real(n_samples, dp)*real(n_samples, dp)/1.0e9_dp
        select case (trim(method))
        case ("full")
            if (gigabytes > DENSE_BUDGET_GB) then
                call report_refusal()
            end if
        case ("ski")
            if (.not. ski_budget_is_feasible(n_inducing, &
                n_features)) call report_refusal()
        case ("grbcm", "grbcm_clustered")
            if (min(n_experts, n_samples) < 2) call report_refusal()
        end select
    end subroutine refuse_infeasible

    subroutine report_refusal()
        !! Report a full row so a sweep records the refusal in place rather
        !! than failing to parse it. NaN marks "did not run".
        write (output_unit, &
            '(a,a,i0,a,i0,a,i0,a,i0,a,a,a,a,a,a,a,a,a,a)') &
            trim(method), ",", n_samples, ",", n_inducing, ",", n_experts, &
            ",", n_features, ",", "NaN", ",", "NaN", ",", "NaN", ",", &
            "NaN", ",", "NaN"
        stop 0
    end subroutine report_refusal

    logical function ski_budget_is_feasible(grid_budget, dimensions) &
            result(feasible)
        integer, intent(in) :: grid_budget, dimensions
        integer :: dimension, minimum_points

        feasible = .false.
        if (grid_budget < 1 .or. dimensions < 1) return
        minimum_points = 1
        do dimension = 1, dimensions
            if (minimum_points > grid_budget/2) return
            minimum_points = 2*minimum_points
        end do
        feasible = .true.
    end function ski_budget_is_feasible

    integer function integer_argument(position, fallback) result(value)
        integer, intent(in) :: position, fallback
        character(len=32) :: text
        integer :: io_status

        value = fallback
        call get_command_argument(position, text)
        if (len_trim(text) == 0) return
        read (text, *, iostat=io_status) value
        if (io_status /= 0 .or. value < 1) value = fallback
    end function integer_argument

    subroutine build_problem()
        !! The paper's one-dimensional fixture, extended to `d` dimensions by
        !! adding uninformative coordinates so the input-dimension scaling can
        !! be measured on the same target function.
        real(dp), allocatable :: base_x(:, :), base_test(:, :)
        integer :: i, j

        call review_toy_data(n_samples, 20260806, base_x, y, status)
        if (.not. status_ok(status)) error stop "benchmark: fixture failed"
        call review_toy_grid(n_test, -6.5_dp, 6.5_dp, base_test, status)
        if (.not. status_ok(status)) error stop "benchmark: test grid failed"

        allocate(x(n_samples, n_features), test_points(n_test, n_features))
        allocate(truth(n_test))
        x = 0.0_dp
        test_points = 0.0_dp
        x(:, 1) = base_x(:, 1)
        test_points(:, 1) = base_test(:, 1)
        do j = 2, n_features
            do i = 1, n_samples
                x(i, j) = 0.05_dp*sin(real(i*j, dp))
            end do
            do i = 1, n_test
                test_points(i, j) = 0.05_dp*cos(real(i*j, dp))
            end do
        end do
        do i = 1, n_test
            truth(i) = review_toy_truth(test_points(i, 1))
        end do
    end subroutine build_problem

    subroutine run_once()
        real(dp) :: train_time, predict_time

        select case (trim(method))
        case ("full")
            call run_exact(n_samples, train_time, predict_time)
        case ("sod")
            call run_subset(train_time, predict_time)
        case ("sor")
            call run_prior(SPARSE_SOR, train_time, predict_time)
        case ("dtc")
            call run_prior(SPARSE_DTC, train_time, predict_time)
        case ("fitc")
            call run_prior(SPARSE_FITC, train_time, predict_time)
        case ("pitc")
            call run_prior(SPARSE_PITC, train_time, predict_time)
        case ("vfe")
            call run_vfe(train_time, predict_time)
        case ("ski")
            call run_ski(train_time, predict_time)
        case ("nle")
            call run_local(AGGREGATE_NLE, train_time, predict_time)
        case ("poe")
            call run_local(AGGREGATE_POE, train_time, predict_time)
        case ("gpoe")
            call run_local(AGGREGATE_GPOE, train_time, predict_time)
        case ("bcm")
            call run_local(AGGREGATE_BCM, train_time, predict_time)
        case ("rbcm")
            call run_local(AGGREGATE_RBCM, train_time, predict_time)
        case ("grbcm")
            call run_local(AGGREGATE_GRBCM, train_time, predict_time)
        case ("moe")
            call run_local(AGGREGATE_MOE, train_time, predict_time)
        case ("nle_clustered")
            call run_local(AGGREGATE_NLE, train_time, predict_time, .true.)
        case ("poe_clustered")
            call run_local(AGGREGATE_POE, train_time, predict_time, .true.)
        case ("gpoe_clustered")
            call run_local(AGGREGATE_GPOE, train_time, predict_time, .true.)
        case ("bcm_clustered")
            call run_local(AGGREGATE_BCM, train_time, predict_time, .true.)
        case ("rbcm_clustered")
            call run_local(AGGREGATE_RBCM, train_time, predict_time, .true.)
        case ("grbcm_clustered")
            call run_local(AGGREGATE_GRBCM, train_time, predict_time, .true.)
        case ("moe_clustered")
            call run_local(AGGREGATE_MOE, train_time, predict_time, .true.)
        case ("keops")
            call run_keops_style(train_time, predict_time, .false.)
        case ("keops_gpu")
            call run_keops_style(train_time, predict_time, .true.)
        case ("keops_matvec")
            call run_matvec_only(train_time, predict_time)
        case default
            error stop "benchmark: unknown method"
        end select
        train_seconds = min(train_seconds, train_time)
        predict_seconds = min(predict_seconds, predict_time)
    end subroutine run_once

    subroutine run_exact(subset, train_time, predict_time)
        integer, intent(in) :: subset
        real(dp), intent(out) :: train_time, predict_time
        type(gp_regression_t) :: model
        type(kernel_t) :: kernel
        real(dp), allocatable :: targets(:, :), prediction(:, :)

        allocate(targets(subset, 1), prediction(n_test, 1))
        targets(:, 1) = y(1:subset)
        kernel = make_rbf_kernel(n_features, SIGNAL_VARIANCE, LENGTHSCALE, status)
        call tic()
        call model%fit(x(1:subset, :), targets, kernel, &
            REVIEW_TOY_NOISE_VARIANCE, status)
        call toc(train_time)
        if (.not. status_ok(status)) error stop "benchmark: exact fit failed"
        call tic()
        call model%predict(test_points, prediction, variance, status)
        call toc(predict_time)
        if (.not. status_ok(status)) error stop "benchmark: exact predict failed"
        mean = prediction(:, 1)
    end subroutine run_exact

    subroutine run_subset(train_time, predict_time)
        real(dp), intent(out) :: train_time, predict_time
        type(gp_regression_t) :: model
        type(kernel_t) :: kernel
        integer, allocatable :: indices(:)
        real(dp), allocatable :: subset_x(:, :), targets(:, :), prediction(:, :)
        integer :: i, subset

        subset = min(n_inducing, n_samples)
        call subset_of_data_indices(n_samples, subset, indices, status)
        if (.not. status_ok(status)) error stop "benchmark: subset failed"
        allocate(subset_x(subset, n_features), targets(subset, 1))
        allocate(prediction(n_test, 1))
        do i = 1, subset
            subset_x(i, :) = x(indices(i), :)
            targets(i, 1) = y(indices(i))
        end do
        kernel = make_rbf_kernel(n_features, SIGNAL_VARIANCE, LENGTHSCALE, status)
        call tic()
        call model%fit(subset_x, targets, kernel, REVIEW_TOY_NOISE_VARIANCE, status)
        call toc(train_time)
        call tic()
        call model%predict(test_points, prediction, variance, status)
        call toc(predict_time)
        if (.not. status_ok(status)) error stop "benchmark: SoD failed"
        mean = prediction(:, 1)
    end subroutine run_subset

    subroutine run_prior(kind, train_time, predict_time)
        integer, intent(in) :: kind
        real(dp), intent(out) :: train_time, predict_time
        type(sparse_prior_gp_t) :: model
        type(kernel_t) :: kernel
        real(dp), allocatable :: inducing(:, :)

        call inducing_set(inducing)
        kernel = make_rbf_kernel(n_features, SIGNAL_VARIANCE, LENGTHSCALE, status)
        if (kind == SPARSE_PITC) then
            call model%initialize(inducing, kernel, REVIEW_TOY_NOISE_VARIANCE, &
                kind, status, block_size=max(n_samples/max(n_experts, 1), 1))
        else
            call model%initialize(inducing, kernel, REVIEW_TOY_NOISE_VARIANCE, &
                kind, status)
        end if
        if (.not. status_ok(status)) error stop "benchmark: sparse init failed"
        call tic()
        call model%fit(x, y, status)
        call toc(train_time)
        if (.not. status_ok(status)) error stop "benchmark: sparse fit failed"
        call tic()
        call model%predict(test_points, mean, variance, status)
        call toc(predict_time)
        if (.not. status_ok(status)) error stop "benchmark: sparse predict failed"
    end subroutine run_prior

    subroutine run_vfe(train_time, predict_time)
        !! The collapsed VFE bound with the optimal `q(u)` formed in closed
        !! form, so the timing measures inference rather than an optimizer.
        real(dp), intent(out) :: train_time, predict_time
        type(sparse_gp_t) :: model
        type(kernel_t) :: kernel
        real(dp), allocatable :: inducing(:, :), k_uu(:, :), k_uf(:, :)
        real(dp), allocatable :: middle(:, :), q_mean(:), factor(:, :)
        real(dp), allocatable :: rhs(:), solution(:), work(:, :), covariance(:, :)
        integer :: i, j, m

        call inducing_set(inducing)
        m = size(inducing, 1)
        kernel = make_rbf_kernel(n_features, SIGNAL_VARIANCE, LENGTHSCALE, status)
        call model%initialize(inducing, kernel, REVIEW_TOY_NOISE_VARIANCE, status)
        if (.not. status_ok(status)) error stop "benchmark: VFE init failed"

        allocate(k_uu(m, m), k_uf(m, n_samples), middle(m, m))
        allocate(q_mean(m), factor(m, m), rhs(m), solution(m))
        allocate(work(m, m), covariance(m, m))
        call tic()
        call kernel%matrix(inducing, inducing, k_uu, status)
        call kernel%matrix(inducing, x, k_uf, status)
        middle = k_uu + matmul(k_uf, transpose(k_uf))/REVIEW_TOY_NOISE_VARIANCE
        do i = 1, m
            middle(i, i) = middle(i, i) + 1.0e-8_dp
        end do
        rhs = matmul(k_uf, y)/REVIEW_TOY_NOISE_VARIANCE
        call gaussian_solve(middle, rhs, solution)
        q_mean = matmul(k_uu, solution)
        do j = 1, m
            call gaussian_solve(middle, k_uu(:, j), work(:, j))
        end do
        covariance = matmul(k_uu, work)
        do i = 1, m
            covariance(i, i) = covariance(i, i) + 1.0e-9_dp
        end do
        call cholesky_factor(covariance, factor)
        call model%set_variational(q_mean, factor, status)
        call toc(train_time)
        if (.not. status_ok(status)) error stop "benchmark: VFE fit failed"
        call tic()
        call model%predict(test_points, mean, variance, status)
        call toc(predict_time)
        if (.not. status_ok(status)) error stop "benchmark: VFE predict failed"
    end subroutine run_vfe

    subroutine run_ski(train_time, predict_time)
        !! SKI trains by one matrix-free CG solve against the interpolated
        !! kernel, then predicts with the interpolated cross-covariance.
        real(dp), intent(out) :: train_time, predict_time
        type(ski_operator_t) :: operator
        type(kernel_t) :: kernel
        real(dp), allocatable :: weights(:)
        real(dp) :: residual_norm
        integer :: info, iterations

        kernel = make_rbf_kernel(n_features, SIGNAL_VARIANCE, LENGTHSCALE, status)
        allocate(weights(n_samples))
        call tic()
        call operator%initialize(x, kernel, n_inducing, &
            REVIEW_TOY_NOISE_VARIANCE, status)
        if (.not. status_ok(status)) error stop "benchmark: SKI init failed"
        weights = 0.0_dp
        call operator%solve_cg(y, weights, 1.0e-8_dp, 500, info, iterations, &
            residual_norm)
        call toc(train_time)
        if (info /= KRYLOV_OK) error stop "benchmark: SKI solve did not converge"
        ! The interpolated kernel W K_uu W^T is rank deficient by construction,
        ! so the LOVE quadrature on it is not a usable variance: it overshoots
        ! and clips. SKI therefore reports a mean only, and its predictive
        ! density is left undefined rather than fabricated.
        call tic()
        call operator%cross_matvec(test_points, weights, mean, status)
        variance = -1.0_dp
        call toc(predict_time)
        if (.not. status_ok(status)) then
            error stop "benchmark: SKI cross-covariance product failed"
        end if
    end subroutine run_ski

    subroutine run_local(aggregation, train_time, predict_time, clustered)
        integer, intent(in) :: aggregation
        real(dp), intent(out) :: train_time, predict_time
        logical, intent(in), optional :: clustered
        type(local_expert_gp_t) :: model
        type(kernel_t) :: kernel
        logical :: use_clusters

        use_clusters = .false.
        if (present(clustered)) use_clusters = clustered
        kernel = make_rbf_kernel(n_features, SIGNAL_VARIANCE, LENGTHSCALE, status)
        call model%initialize(kernel, REVIEW_TOY_NOISE_VARIANCE, aggregation, &
            status)
        if (.not. status_ok(status)) error stop "benchmark: local init failed"
        call tic()
        if (use_clusters) then
            call model%fit_clustered(x, y, min(n_experts, n_samples), status)
        else
            call model%fit(x, y, min(n_experts, n_samples), status)
        end if
        call toc(train_time)
        if (.not. status_ok(status)) error stop "benchmark: local fit failed"
        call tic()
        call model%predict(test_points, mean, variance, status)
        call toc(predict_time)
        if (.not. status_ok(status)) error stop "benchmark: local predict failed"
    end subroutine run_local

    subroutine run_matvec_only(train_time, predict_time)
        !! One matrix-free kernel product. This is the unit every matrix-free
        !! method is built from, and the only cost that can be measured cleanly
        !! at a sample count where a full solve would not finish.
        real(dp), intent(out) :: train_time, predict_time
        type(kernel_operator_t) :: operator
        type(kernel_t) :: kernel
        real(dp), allocatable :: input(:), output(:)
        integer :: i

        kernel = make_rbf_kernel(n_features, SIGNAL_VARIANCE, LENGTHSCALE, status)
        allocate(input(n_samples), output(n_samples))
        do i = 1, n_samples
            input(i) = 1.0_dp/real(i, dp)
        end do
        call operator%initialize(x, kernel, REVIEW_TOY_NOISE_VARIANCE, status)
        if (.not. status_ok(status)) error stop "benchmark: matvec init failed"
        call tic()
        call operator%matvec(input, output)
        call toc(train_time)
        predict_time = 0.0_dp
        mean = 0.0_dp
        variance = -1.0_dp
        ! Keep the compiler from removing the product.
        if (output(1) /= output(1)) error stop "benchmark: matvec produced NaN"
    end subroutine run_matvec_only

    subroutine run_keops_style(train_time, predict_time, on_device)
        !! The matrix-free exact lane: no approximation at all, one CG solve
        !! against the full kernel with the covariance never formed. With
        !! `on_device` the sample points stay resident and the Krylov products
        !! run through OpenACC.
        real(dp), intent(out) :: train_time, predict_time
        logical, intent(in) :: on_device
        type(kernel_operator_t) :: operator
        type(kernel_t) :: kernel
        real(dp), allocatable :: weights(:)
        real(dp) :: residual_norm
        integer :: info, iterations, i, j

        kernel = make_rbf_kernel(n_features, SIGNAL_VARIANCE, LENGTHSCALE, status)
        allocate(weights(n_samples))
        call tic()
        call operator%initialize(x, kernel, REVIEW_TOY_NOISE_VARIANCE, status)
        if (.not. status_ok(status)) error stop "benchmark: KeOps-style init failed"
        weights = 0.0_dp
        if (on_device) then
            call operator%enter_data(status)
            if (.not. status_ok(status)) error stop "benchmark: residency failed"
            call operator%solve_cg_device(y, weights, 1.0e-8_dp, 2000, info, &
                iterations, residual_norm)
            call operator%exit_data(status)
        else
            call operator%solve_cg(y, weights, 1.0e-8_dp, 2000, info, iterations, &
                residual_norm)
        end if
        call toc(train_time)
        if (info /= KRYLOV_OK) error stop "benchmark: KeOps-style solve failed"
        call tic()
        call predict_with_love(operator, kernel, weights, predict_time)
        call toc(predict_time)
    end subroutine run_keops_style

    subroutine predict_with_love(operator, kernel, weights, predict_time)
        !! Matrix-free prediction: the mean from the solved weights, and the
        !! variance from the LOVE Lanczos estimate, so the matrix-free lanes
        !! report a real predictive variance rather than the prior.
        class(*), intent(inout) :: operator
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: weights(:)
        real(dp), intent(out) :: predict_time
        real(dp), allocatable :: cross(:)
        integer :: i, j

        predict_time = 0.0_dp
        allocate(cross(n_samples))
        do i = 1, n_test
            do j = 1, n_samples
                cross(j) = kernel%value(test_points(i, :), x(j, :))
            end do
            mean(i) = sum(cross*weights)
            if (n_samples > LOVE_SAMPLE_LIMIT) then
                variance(i) = -1.0_dp
                cycle
            end if
            select type (operator)
                type is (kernel_operator_t)
                call lanczos_predictive_variance(operator, cross, &
                    SIGNAL_VARIANCE, LANCZOS_STEPS, variance(i), status)
                type is (ski_operator_t)
                call lanczos_predictive_variance(operator, cross, &
                    SIGNAL_VARIANCE, LANCZOS_STEPS, variance(i), status)
            end select
            if (.not. status_ok(status)) error stop "benchmark: LOVE failed"
            variance(i) = max(variance(i), 1.0e-10_dp)
        end do
    end subroutine predict_with_love

    subroutine inducing_set(inducing)
        real(dp), allocatable, intent(out) :: inducing(:, :)
        real(dp), allocatable :: base(:, :)
        integer :: i, j, m

        m = max(min(n_inducing, n_samples), 2)
        call review_toy_grid(m, -6.0_dp, 6.0_dp, base, status)
        if (.not. status_ok(status)) error stop "benchmark: inducing grid failed"
        allocate(inducing(m, n_features))
        inducing = 0.0_dp
        inducing(:, 1) = base(:, 1)
        do j = 2, n_features
            do i = 1, m
                inducing(i, j) = 0.05_dp*sin(real(i*j, dp))
            end do
        end do
    end subroutine inducing_set

    subroutine accuracy(prediction, prediction_variance, smse_out, mnlpd_out)
        !! Standardized mean squared error against the target variance, and
        !! the mean negative log predictive density, both against the truth.
        real(dp), intent(in) :: prediction(:), prediction_variance(:)
        real(dp), intent(out) :: smse_out, mnlpd_out
        real(dp) :: target_mean, target_variance, total, density
        integer :: i

        target_mean = sum(truth)/real(n_test, dp)
        target_variance = sum((truth - target_mean)**2)/real(n_test, dp)
        total = sum((prediction - truth)**2)/real(n_test, dp)
        smse_out = total/max(target_variance, tiny(1.0_dp))
        if (any(prediction_variance < 0.0_dp)) then
            ! A method that reports no predictive variance gets no density.
            mnlpd_out = ieee_value(1.0_dp, ieee_quiet_nan)
            return
        end if
        density = 0.0_dp
        do i = 1, n_test
            density = density + 0.5_dp*log(8.0_dp*atan(1.0_dp)* &
                max(prediction_variance(i), 1.0e-12_dp)) &
                + 0.5_dp*(prediction(i) - truth(i))**2/ &
                max(prediction_variance(i), 1.0e-12_dp)
        end do
        mnlpd_out = density/real(n_test, dp)
    end subroutine accuracy

    subroutine gaussian_solve(matrix, rhs, solution)
        real(dp), intent(in) :: matrix(:, :), rhs(:)
        real(dp), intent(out) :: solution(:)
        real(dp), allocatable :: a(:, :), b(:)
        real(dp) :: multiplier, swap
        integer :: i, j, k, pivot, m

        m = size(rhs)
        allocate(a, source=matrix)
        allocate(b, source=rhs)
        do k = 1, m - 1
            pivot = k
            do i = k + 1, m
                if (abs(a(i, k)) > abs(a(pivot, k))) pivot = i
            end do
            if (pivot /= k) then
                do j = 1, m
                    swap = a(k, j)
                    a(k, j) = a(pivot, j)
                    a(pivot, j) = swap
                end do
                swap = b(k)
                b(k) = b(pivot)
                b(pivot) = swap
            end if
            do i = k + 1, m
                multiplier = a(i, k)/a(k, k)
                a(i, k:) = a(i, k:) - multiplier*a(k, k:)
                b(i) = b(i) - multiplier*b(k)
            end do
        end do
        do i = m, 1, -1
            solution(i) = (b(i) - sum(a(i, i + 1:)*solution(i + 1:)))/a(i, i)
        end do
    end subroutine gaussian_solve

    subroutine cholesky_factor(matrix, factor)
        real(dp), intent(in) :: matrix(:, :)
        real(dp), intent(out) :: factor(:, :)
        real(dp) :: total
        integer :: i, j, k

        factor = 0.0_dp
        do i = 1, size(matrix, 1)
            do j = 1, i
                total = matrix(i, j)
                do k = 1, j - 1
                    total = total - factor(i, k)*factor(j, k)
                end do
                if (i == j) then
                    factor(i, i) = sqrt(max(total, 1.0e-14_dp))
                else
                    factor(i, j) = total/factor(j, j)
                end if
            end do
        end do
    end subroutine cholesky_factor

    real(dp) function peak_resident_kilobytes() result(value)
        !! `VmHWM` is the kernel's own high-water mark for this process, so it
        !! covers every allocation the run made, not only the ones this program
        !! can see.
        character(len=256) :: line
        integer :: unit, io_status

        value = 0.0_dp
        open (newunit=unit, file="/proc/self/status", status="old", &
            action="read", iostat=io_status)
        if (io_status /= 0) return
        do
            read (unit, '(a)', iostat=io_status) line
            if (io_status /= 0) exit
            if (index(line, "VmHWM:") == 1) then
                read (line(7:), *, iostat=io_status) value
                exit
            end if
        end do
        close (unit)
    end function peak_resident_kilobytes

    subroutine tic()
        call system_clock(clock_start, clock_rate)
    end subroutine tic

    subroutine toc(seconds)
        real(dp), intent(out) :: seconds

        call system_clock(clock_end)
        seconds = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    end subroutine toc

end program fortml_bench_scalable_gp
