module fortml_mlp_training
    !! Deterministic, optimizer-facing training products for dense MLPs.
    !!
    !! `mlp_train` deliberately keeps the data contract explicit: rows are
    !! samples, columns are features (or outputs), and the objective is the
    !! mean squared error with an optional L2 penalty.  The same loss product
    !! is used for full-batch diagnostics and every mini-batch update, so the
    !! reported gradient is the derivative of the reported objective.  The
    !! scalar L2 derivative is returned as a first-class hyperparameter
    !! product, which lets an outer optimizer differentiate this objective
    !! without finite-difference plumbing.
    use, intrinsic :: iso_fortran_env, only: int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_mlp, only: mlp_t
    use fortopt_adam, only: adam_t
    implicit none
    private

    abstract interface
        subroutine mlp_epoch_callback_proc(epoch, loss, gradient_norm, stop)
            import :: dp
            integer, intent(in) :: epoch
            real(dp), intent(in) :: loss, gradient_norm
            logical, intent(out) :: stop
        end subroutine mlp_epoch_callback_proc
    end interface

    type, public :: mlp_training_options_t
        integer :: max_epochs = 1000
        integer :: batch_size = 0
        integer :: patience = 0
        integer :: shuffle_seed = 17
        logical :: shuffle = .false.
        logical :: restore_best = .true.
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: min_delta = 0.0_dp
        procedure(mlp_epoch_callback_proc), pointer, nopass :: callback => null()
    end type mlp_training_options_t

    type, public :: mlp_training_state_t
        integer :: epochs = 0
        integer :: updates = 0
        integer :: best_epoch = 0
        logical :: converged = .false.
        logical :: early_stopped = .false.
        real(dp) :: initial_loss = huge(1.0_dp)
        real(dp) :: final_loss = huge(1.0_dp)
        real(dp) :: best_loss = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp), allocatable :: loss_history(:)
    end type mlp_training_state_t

    public :: mlp_epoch_callback_proc
    public :: mlp_loss_value_gradient
    public :: mlp_train

contains

    subroutine mlp_loss_value_gradient(model, x, target, l2, value, gradient, &
            l2_gradient, status)
        !! MSE value, network-parameter gradient, and derivative with respect
        !! to the scalar L2 coefficient.
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :), l2
        real(dp), intent(out) :: value, gradient(:), l2_gradient
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: prediction(:, :), residual(:, :), x_bar(:, :)
        real(dp), allocatable :: theta(:)
        integer :: n_samples

        value = 0.0_dp
        l2_gradient = 0.0_dp
        if (.not. valid_data(model, x, target) .or. l2 < 0.0_dp .or. &
                size(gradient) /= model%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP loss: model, data, penalty, or gradient shape is invalid")
            return
        end if
        n_samples = size(x, 1)
        allocate(prediction(size(target, 1), size(target, 2)))
        allocate(residual, mold=prediction)
        allocate(x_bar, mold=x)
        call model%predict(x, prediction, status)
        if (status%code /= FORTNUM_OK) return
        residual = prediction - target
        value = 0.5_dp*sum(residual*residual)/real(n_samples, dp)
        call model%vjp(x, residual/real(n_samples, dp), gradient, x_bar, status)
        if (status%code /= FORTNUM_OK) return
        theta = model%parameters()
        value = value + 0.5_dp*l2*sum(theta*theta)
        l2_gradient = 0.5_dp*sum(theta*theta)
        gradient = gradient + l2*theta
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_loss_value_gradient

    subroutine mlp_train(model, x, target, status, options, state)
        !! Train `model` with deterministic Adam updates.
        !!
        !! A zero batch size selects full-batch updates.  When shuffling is
        !! enabled, an explicit Park--Miller stream seeded by `shuffle_seed`
        !! drives Fisher--Yates permutations; no process-global RNG state is
        !! touched.  Callback execution occurs once per completed epoch.
        class(mlp_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(mlp_training_options_t), intent(in), optional :: options
        type(mlp_training_state_t), intent(out), optional :: state
        type(mlp_training_options_t) :: config
        type(mlp_training_state_t) :: result
        type(adam_t) :: optimizer
        real(dp), allocatable :: theta(:), best_theta(:), gradient(:)
        real(dp), allocatable :: x_batch(:, :), target_batch(:, :)
        integer, allocatable :: order(:)
        real(dp) :: loss, l2_gradient, gradient_norm, improvement
        real(dp) :: best_loss
        integer :: n_samples, n_outputs, n_parameters
        integer :: batch, first, last, epoch
        integer :: stale_epochs
        integer(int64) :: shuffle_state
        logical :: stop_now

        if (present(options)) config = options
        if (.not. valid_options(config) .or. &
                .not. valid_data(model, x, target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP train: invalid model, data, or options")
            if (present(state)) state = result
            return
        end if

        n_samples = size(x, 1)
        n_outputs = size(target, 2)
        n_parameters = model%parameter_count()
        batch = config%batch_size
        if (batch == 0) batch = n_samples
        batch = min(batch, n_samples)

        theta = model%parameters()
        allocate(best_theta, source=theta)
        allocate(gradient(n_parameters))
        allocate(order(n_samples))
        allocate(result%loss_history(config%max_epochs))
        call mlp_loss_value_gradient(model, x, target, config%l2, loss, &
            gradient, l2_gradient, status)
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if
        result%initial_loss = loss
        best_loss = loss
        result%best_loss = loss
        stale_epochs = 0
        shuffle_state = int(config%shuffle_seed, int64)
        if (shuffle_state <= 0_int64) shuffle_state = 1_int64
        call optimizer%initialize(n_parameters, status, &
            learning_rate=config%learning_rate, beta1=config%beta1, &
            beta2=config%beta2, epsilon=config%epsilon)
        if (status%code /= FORTNUM_OK) then
            if (present(state)) state = result
            return
        end if

        do epoch = 1, config%max_epochs
            order = [(first, first=1, n_samples)]
            if (config%shuffle) call shuffle_order(order, shuffle_state)
            do first = 1, n_samples, batch
                last = min(first + batch - 1, n_samples)
                allocate(x_batch(last - first + 1, size(x, 2)))
                allocate(target_batch(last - first + 1, n_outputs))
                x_batch = x(order(first:last), :)
                target_batch = target(order(first:last), :)
                call mlp_loss_value_gradient(model, x_batch, target_batch, &
                    config%l2, loss, gradient, l2_gradient, status)
                deallocate(x_batch, target_batch)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                call optimizer%step(theta, gradient, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                call model%set_parameters(theta, status)
                if (status%code /= FORTNUM_OK) then
                    if (present(state)) state = result
                    return
                end if
                result%updates = result%updates + 1
            end do

            call mlp_loss_value_gradient(model, x, target, config%l2, loss, &
                gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            gradient_norm = sqrt(sum(gradient*gradient))
            result%epochs = epoch
            result%loss_history(epoch) = loss
            result%gradient_norm = gradient_norm
            improvement = best_loss - loss
            if (improvement > config%min_delta) then
                best_loss = loss
                result%best_loss = loss
                result%best_epoch = epoch
                best_theta = theta
                stale_epochs = 0
            else
                stale_epochs = stale_epochs + 1
            end if
            stop_now = .false.
            if (associated(config%callback)) then
                call config%callback(epoch, loss, gradient_norm, stop_now)
            end if
            if (gradient_norm <= config%tolerance) then
                result%converged = .true.
                exit
            end if
            if (stop_now) then
                result%early_stopped = .true.
                exit
            end if
            if (config%patience > 0 .and. stale_epochs >= config%patience) then
                result%early_stopped = .true.
                exit
            end if
        end do

        call shrink_history(result%loss_history, result%epochs)
        if (config%restore_best .and. result%best_epoch > 0 .and. &
                result%best_epoch < result%epochs) then
            theta = best_theta
            call model%set_parameters(theta, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            call mlp_loss_value_gradient(model, x, target, config%l2, loss, &
                gradient, l2_gradient, status)
            if (status%code /= FORTNUM_OK) then
                if (present(state)) state = result
                return
            end if
            result%gradient_norm = sqrt(sum(gradient*gradient))
        end if
        result%final_loss = loss
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_train

    logical function valid_options(options) result(valid)
        type(mlp_training_options_t), intent(in) :: options

        valid = options%max_epochs >= 1 .and. options%batch_size >= 0 .and. &
            options%patience >= 0 .and. options%learning_rate > 0.0_dp .and. &
            options%beta1 >= 0.0_dp .and. options%beta1 < 1.0_dp .and. &
            options%beta2 >= 0.0_dp .and. options%beta2 < 1.0_dp .and. &
            options%epsilon > 0.0_dp .and. options%l2 >= 0.0_dp .and. &
            options%tolerance >= 0.0_dp .and. options%min_delta >= 0.0_dp
        if (options%shuffle) valid = valid .and. options%shuffle_seed > 0
    end function valid_options

    logical function valid_data(model, x, target) result(valid)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)

        valid = size(x, 1) > 0 .and. size(target, 1) == size(x, 1)
        if (.not. valid) return
        valid = size(x, 2) > 0 .and. size(target, 2) > 0
        if (.not. valid) return
        valid = all(ieee_is_finite(x)) .and. all(ieee_is_finite(target))
        if (.not. valid) return
        ! The model itself performs the authoritative feature/output shape
        ! check; this call avoids exposing its private allocation predicate.
        valid = model%parameter_count() > 0
    end function valid_data

    subroutine shuffle_order(order, generator)
        integer, intent(inout) :: order(:)
        integer(int64), intent(inout) :: generator
        integer :: i, j, temporary

        do i = size(order), 2, -1
            generator = mod(48271_int64*generator, 2147483647_int64)
            j = 1 + int(mod(generator, int(i, int64)))
            temporary = order(i)
            order(i) = order(j)
            order(j) = temporary
        end do
    end subroutine shuffle_order

    subroutine shrink_history(history, length)
        real(dp), allocatable, intent(inout) :: history(:)
        integer, intent(in) :: length
        real(dp), allocatable :: compact(:)

        if (size(history) == length) return
        allocate(compact(max(0, length)))
        if (length > 0) compact = history(:length)
        call move_alloc(compact, history)
    end subroutine shrink_history

end module fortml_mlp_training
