program fortml_bench_gp_classification_training
    !! Correctness-gated timing for the bounded GP-classification adapters.
    !!
    !! The objective reported here is the negative converged Laplace
    !! mode-log-posterior.  It is not a full Laplace evidence benchmark.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t
    use fortml_gp_multiclass_classification, only: &
        gp_multiclass_classification_t
    use fortml_gp_classification_training, only: &
        gp_classification_hyperparameter_options_t, &
        gp_classification_hyperparameter_result_t, &
        gp_multiclass_hyperparameter_options_t, &
        gp_multiclass_hyperparameter_result_t, &
        gp_classification_optimize_hyperparameters, &
        gp_multiclass_optimize_hyperparameters
    use fortml_kernels, only: kernel_t, make_rbf_kernel
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 8, n_features = 1, n_classes = 3
    integer, parameter :: repetitions = 3
    real(dp), parameter :: jitter = 1.0e-7_dp
    real(dp) :: x(n_samples, n_features)
    integer :: binary_labels(n_samples), multiclass_labels(n_samples)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: seconds
    type(fortnum_status_t) :: status

    x(:, 1) = [-1.5_dp, -1.0_dp, -0.5_dp, -0.1_dp, &
        0.1_dp, 0.5_dp, 1.0_dp, 1.5_dp]
    binary_labels = [-7, -7, -7, -7, 11, 11, 11, 11]
    multiclass_labels = [42, 42, 42, -7, -7, 11, 11, 11]

    call benchmark_binary(x, binary_labels, seconds, status)
    if (.not. status_ok(status)) error stop "GP classification binary benchmark failed"
    call benchmark_multiclass(x, multiclass_labels, seconds, status)
    if (.not. status_ok(status)) error stop "GP classification multiclass benchmark failed"

contains

    subroutine benchmark_binary(input, labels, elapsed, final_status)
        real(dp), intent(in) :: input(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: elapsed
        type(fortnum_status_t), intent(out) :: final_status
        type(kernel_t) :: kernel, timed_kernel
        type(gp_classification_t) :: model, timed_model
        type(gp_classification_hyperparameter_options_t) :: options, invalid_options
        type(gp_classification_hyperparameter_result_t) :: result, invalid_result
        type(gp_classification_options_t) :: fit_options
        real(dp), allocatable :: parameters(:), gradient(:)
        integer :: repetition, i
        type(fortnum_status_t) :: status

        fit_options%max_iterations = 100
        fit_options%tolerance = 1.0e-9_dp
        fit_options%jitter = jitter
        options%fit = fit_options
        options%max_iterations = 100
        options%gradient_tolerance = 2.0e-3_dp
        options%lower_bound = -5.0_dp
        options%upper_bound = 5.0_dp
        kernel = make_rbf_kernel(n_features, 1.2_dp, 0.8_dp, status)
        call gp_classification_optimize_hyperparameters(model, input, labels, kernel, &
            options, result, status)
        if (.not. status_ok(status)) then
            final_status = status
            return
        end if
        parameters = kernel%parameters()
        allocate(gradient(size(parameters)))
        call model%hyperparameter_gradient(gradient, status)
        if (.not. status_ok(status)) then
            final_status = status
            return
        end if

        call system_clock(clock_start, clock_rate)
        do repetition = 1, repetitions
            timed_kernel = make_rbf_kernel(n_features, 1.2_dp, 0.8_dp, status)
            call gp_classification_optimize_hyperparameters(timed_model, input, labels, &
                timed_kernel, options, invalid_result, status)
            if (.not. status_ok(status)) then
                final_status = status
                return
            end if
        end do
        call system_clock(clock_end)
        elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/ &
            real(repetitions, dp)

        write (*, '(a,i0,a,i0,a,i0,a,es24.16,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
            "gp_classification_binary,", size(input, 1), ",", size(input, 2), ",", &
            repetitions, ",", elapsed, ",", result%iterations, ",", &
            result%negative_log_posterior, ",", result%gradient_norm, ",", &
            parameters(1), ",", parameters(2), ",", gradient(1), ",", gradient(2)

        invalid_options = options
        invalid_options%lower_bound = 2.0_dp
        invalid_options%upper_bound = -2.0_dp
        call gp_classification_optimize_hyperparameters(model, input, labels, kernel, &
            invalid_options, invalid_result, status)
        i = merge(1, 0, .not. status_ok(status))
        write (*, '(a,i0)') "gp_classification_binary_invalid_bounds,", i
        final_status = status
        ! The invalid-bound refusal is expected; restore a successful status.
        call status_set_ok(final_status)
    end subroutine benchmark_binary

    subroutine benchmark_multiclass(input, labels, elapsed, final_status)
        real(dp), intent(in) :: input(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: elapsed
        type(fortnum_status_t), intent(out) :: final_status
        type(kernel_t) :: kernel, timed_kernel
        type(gp_multiclass_classification_t) :: model, timed_model
        type(gp_multiclass_hyperparameter_options_t) :: options
        type(gp_multiclass_hyperparameter_result_t) :: result, timed_result
        real(dp), allocatable :: parameters(:), packed_gradient(:), gradient(:)
        integer :: repetition, class_index, block, first, last
        type(fortnum_status_t) :: status

        options%fit%max_iterations = 100
        options%fit%tolerance = 1.0e-9_dp
        options%fit%jitter = jitter
        options%max_iterations = 100
        options%gradient_tolerance = 5.0e-3_dp
        options%lower_bound = -5.0_dp
        options%upper_bound = 5.0_dp
        kernel = make_rbf_kernel(n_features, 1.2_dp, 0.8_dp, status)
        call gp_multiclass_optimize_hyperparameters(model, input, labels, kernel, &
            options, result, status)
        if (.not. status_ok(status)) then
            final_status = status
            return
        end if
        parameters = kernel%parameters()
        block = size(parameters)
            allocate(packed_gradient(block*n_classes), gradient(block))
            call model%hyperparameter_gradient(packed_gradient, status)
            if (.not. status_ok(status)) then
                final_status = status
                return
            end if
            gradient = 0.0_dp
            first = 1
            do class_index = 1, n_classes
                last = first + block - 1
                gradient = gradient + packed_gradient(first:last)
                first = last + 1
            end do

            call system_clock(clock_start, clock_rate)
            do repetition = 1, repetitions
                timed_kernel = make_rbf_kernel(n_features, 1.2_dp, 0.8_dp, status)
                call gp_multiclass_optimize_hyperparameters(timed_model, input, labels, &
                    timed_kernel, options, timed_result, status)
                if (.not. status_ok(status)) then
                    final_status = status
                    return
                end if
            end do
            call system_clock(clock_end)
            elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)/ &
                real(repetitions, dp)
            write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16,a,i0,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16,a,es24.16)') &
                "gp_classification_multiclass,", size(input, 1), ",", size(input, 2), ",", &
                n_classes, ",", repetitions, ",", elapsed, ",", result%iterations, ",", &
                result%negative_log_posterior, ",", result%gradient_norm, ",", &
                parameters(1), ",", parameters(2), ",", gradient(1), ",", gradient(2)
            final_status = status
        end subroutine benchmark_multiclass

        subroutine status_set_ok(output)
            type(fortnum_status_t), intent(out) :: output

            output%code = 0
            output%msg = ""
        end subroutine status_set_ok

    end program fortml_bench_gp_classification_training
