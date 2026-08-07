program fortml_bench_neural_losses
    !! Release benchmark app for the shared differentiable neural losses.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_losses, only: binary_cross_entropy_with_logits_hvp, &
        softmax_cross_entropy_hvp, weighted_mse_loss_hvp, huber_loss_hvp, &
        mae_loss_jvp, focal_binary_cross_entropy_with_logits_jvp, &
        gaussian_nll_hvp, poisson_nll_hvp
    use fortml_mlp, only: mlp_t
    use fortml_mlp_training, only: mlp_loss_value_gradient
    implicit none

    integer, parameter :: n = 64, k = 3, repetitions = 2048
    real(dp) :: logits(n, k), targets(n, k), direction(n, k), product(n, k)
    real(dp) :: prediction(n, 1), target(n, 1), weights(n), value, l2_gradient
    real(dp) :: log_variance(n, k), count_targets(n, k), variance_direction(n, k)
    real(dp) :: variance_product(n, k)
    real(dp) :: value_dot, loss_value
    real(dp), allocatable :: gradient(:)
    integer :: labels(n), i, repetition
    real(dp) :: start, finish
    type(fortnum_status_t) :: status
    type(mlp_t) :: model

    do i = 1, n
        logits(i, :) = [sin(0.13_dp*real(i, dp)), cos(0.07_dp*real(i, dp)), &
            0.2_dp*sin(0.19_dp*real(i, dp))]
        targets(i, :) = [merge(1.0_dp, 0.0_dp, mod(i, 3) == 0), &
            merge(1.0_dp, 0.0_dp, mod(i, 3) == 1), &
            merge(1.0_dp, 0.0_dp, mod(i, 3) == 2)]
        direction(i, :) = [0.01_dp*sin(0.11_dp*real(i, dp)), &
            -0.02_dp*cos(0.17_dp*real(i, dp)), 0.03_dp]
        log_variance(i, :) = [0.2_dp*sin(0.03_dp*real(i, dp)), &
            -0.1_dp*cos(0.05_dp*real(i, dp)), 0.15_dp*sin(0.07_dp*real(i, dp))]
        count_targets(i, :) = [real(mod(i, 5), dp), real(mod(i + 1, 5), dp), &
            0.5_dp + real(mod(i + 2, 4), dp)]
        variance_direction(i, :) = [0.02_dp*cos(0.09_dp*real(i, dp)), &
            -0.01_dp*sin(0.08_dp*real(i, dp)), 0.015_dp]
        labels(i) = 1 + mod(i - 1, k)
        prediction(i, 1) = logits(i, 1)
        target(i, 1) = 0.4_dp*sin(0.05_dp*real(i, dp))
        weights(i) = 0.5_dp + real(mod(i, 7), dp)/7.0_dp
    end do

    call binary_cross_entropy_with_logits_hvp(logits, targets, direction, product, status)
    if (.not. status_ok(status)) error stop "BCE warmup failed"
    call cpu_time(start)
    do repetition = 1, repetitions
        call binary_cross_entropy_with_logits_hvp(logits, targets, direction, product, status)
    end do
    call cpu_time(finish)
    call emit("bce_hvp", finish - start, sum(product))

    call softmax_cross_entropy_hvp(logits, labels, direction, product, status)
    if (.not. status_ok(status)) error stop "softmax warmup failed"
    call cpu_time(start)
    do repetition = 1, repetitions
        call softmax_cross_entropy_hvp(logits, labels, direction, product, status)
    end do
    call cpu_time(finish)
    call emit("softmax_cross_entropy_hvp", finish - start, sum(product))

    call weighted_mse_loss_hvp(prediction, target, weights, direction(:, 1:1), &
        product(:, 1:1), status)
    if (.not. status_ok(status)) error stop "weighted MSE warmup failed"
    call cpu_time(start)
    do repetition = 1, repetitions
        call weighted_mse_loss_hvp(prediction, target, weights, direction(:, 1:1), &
            product(:, 1:1), status)
    end do
    call cpu_time(finish)
    call emit("weighted_mse_hvp", finish - start, sum(product(:, 1:1)))

    call huber_loss_hvp(prediction, target, 0.75_dp, direction(:, 1:1), &
        product(:, 1:1), status)
    if (.not. status_ok(status)) error stop "Huber warmup failed"
    call cpu_time(start)
    do repetition = 1, repetitions
        call huber_loss_hvp(prediction, target, 0.75_dp, direction(:, 1:1), &
            product(:, 1:1), status)
    end do
    call cpu_time(finish)
    call emit("huber_hvp", finish - start, sum(product(:, 1:1)))

    call mae_loss_jvp(prediction, target, direction(:, 1:1), loss_value, value_dot, &
        status, weights)
    if (.not. status_ok(status)) error stop "MAE warmup failed"
    call cpu_time(start)
    do repetition = 1, repetitions
        call mae_loss_jvp(prediction, target, direction(:, 1:1), loss_value, &
            value_dot, status, weights)
    end do
    call cpu_time(finish)
    call emit("mae_jvp", finish - start, value_dot)

    call focal_binary_cross_entropy_with_logits_jvp(logits, targets, 0.25_dp, &
        2.0_dp, direction, loss_value, value_dot, status, weights)
    if (.not. status_ok(status)) error stop "focal BCE warmup failed"
    call cpu_time(start)
    do repetition = 1, repetitions
        call focal_binary_cross_entropy_with_logits_jvp(logits, targets, 0.25_dp, &
            2.0_dp, direction, loss_value, value_dot, status, weights)
    end do
    call cpu_time(finish)
    call emit("focal_bce_jvp", finish - start, value_dot)

    call gaussian_nll_hvp(logits, targets, log_variance, direction, &
        variance_direction, product, variance_product, status, weights)
    if (.not. status_ok(status)) error stop "Gaussian NLL warmup failed"
    call cpu_time(start)
    do repetition = 1, repetitions
        call gaussian_nll_hvp(logits, targets, log_variance, direction, &
            variance_direction, product, variance_product, status, weights)
    end do
    call cpu_time(finish)
    call emit("gaussian_nll_hvp", finish - start, sum(product) + sum(variance_product))

    call poisson_nll_hvp(logits, count_targets, direction, product, status, weights)
    if (.not. status_ok(status)) error stop "Poisson NLL warmup failed"
    call cpu_time(start)
    do repetition = 1, repetitions
        call poisson_nll_hvp(logits, count_targets, direction, product, status, weights)
    end do
    call cpu_time(finish)
    call emit("poisson_nll_hvp", finish - start, sum(product))

    call model%initialize([1, 4, 1], status, initialization_seed=29)
    if (.not. status_ok(status)) error stop "MLP initialization failed"
    allocate(gradient(model%parameter_count()))
    call mlp_loss_value_gradient(model, prediction, target, 0.0_dp, value, gradient, &
        l2_gradient, status, sample_weight=weights)
    if (.not. status_ok(status)) error stop "MLP weighted objective warmup failed"
    call cpu_time(start)
    do repetition = 1, repetitions
        call mlp_loss_value_gradient(model, prediction, target, 0.0_dp, value, &
            gradient, l2_gradient, status, sample_weight=weights)
    end do
    call cpu_time(finish)
    call emit("mlp_weighted_objective", finish - start, value + sum(gradient))

contains

    subroutine emit(name, elapsed, checksum)
        character(len=*), intent(in) :: name
        real(dp), intent(in) :: elapsed, checksum

        write (*, '(a,",",es24.16,",",es24.16)') trim(name), &
            elapsed/real(repetitions, dp), checksum
    end subroutine emit

end program fortml_bench_neural_losses
