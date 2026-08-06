module fortml_rnn
    !! Sequence-batched vanilla recurrent network with backpropagation through
    !! time.
    !!
    !! Time is the leading axis: `inputs(t, i, :)` is step `t` of sequence `i`.
    !! The recurrence is
    !!
    !!     h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)
    !!     y_t = h_t W_hy + b_y
    !!
    !! and the loss is one half of the squared error against the targets at
    !! every step. The reverse pass walks the same steps backwards, so the
    !! gradient is exact rather than truncated; `deep` sequences are handled by
    !! keeping every hidden state, which is what a checkpointing scheme would
    !! later trade away.
    !!
    !! Parameters are packed as `[W_xh, W_hh, b_h, W_hy, b_y]`, each matrix in
    !! column-major order, matching the flat layout `mlp_t` uses.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    type, public :: rnn_t
        real(dp), allocatable :: input_weight(:, :)
        real(dp), allocatable :: hidden_weight(:, :)
        real(dp), allocatable :: hidden_bias(:)
        real(dp), allocatable :: output_weight(:, :)
        real(dp), allocatable :: output_bias(:)
        integer :: input_dim = 0
        integer :: hidden_dim = 0
        integer :: output_dim = 0
    contains
        procedure, public :: initialize => rnn_initialize
        procedure, public :: parameter_count => rnn_parameter_count
        procedure, public :: parameters => rnn_parameters
        procedure, public :: set_parameters => rnn_set_parameters
        procedure, public :: forward => rnn_forward
        procedure, public :: loss => rnn_loss
        procedure, public :: loss_gradient => rnn_loss_gradient
    end type rnn_t

contains

    subroutine rnn_initialize(self, input_dim, hidden_dim, output_dim, status)
        class(rnn_t), intent(out) :: self
        integer, intent(in) :: input_dim, hidden_dim, output_dim
        type(fortnum_status_t), intent(out) :: status

        if (input_dim < 1 .or. hidden_dim < 1 .or. output_dim < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RNN: layer shape is invalid")
            return
        end if
        self%input_dim = input_dim
        self%hidden_dim = hidden_dim
        self%output_dim = output_dim
        allocate(self%input_weight(input_dim, hidden_dim))
        allocate(self%hidden_weight(hidden_dim, hidden_dim))
        allocate(self%hidden_bias(hidden_dim))
        allocate(self%output_weight(hidden_dim, output_dim))
        allocate(self%output_bias(output_dim))
        self%input_weight = 0.0_dp
        self%hidden_weight = 0.0_dp
        self%hidden_bias = 0.0_dp
        self%output_weight = 0.0_dp
        self%output_bias = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine rnn_initialize

    integer function rnn_parameter_count(self) result(count)
        class(rnn_t), intent(in) :: self

        count = 0
        if (self%hidden_dim < 1) return
        count = self%input_dim*self%hidden_dim + self%hidden_dim*self%hidden_dim &
            + self%hidden_dim + self%hidden_dim*self%output_dim + self%output_dim
    end function rnn_parameter_count

    function rnn_parameters(self) result(theta)
        class(rnn_t), intent(in) :: self
        real(dp), allocatable :: theta(:)
        integer :: cursor, block_size

        allocate(theta(self%parameter_count()))
        cursor = 1
        block_size = size(self%input_weight)
        theta(cursor:cursor + block_size - 1) = &
            reshape(self%input_weight, [block_size])
        cursor = cursor + block_size
        block_size = size(self%hidden_weight)
        theta(cursor:cursor + block_size - 1) = &
            reshape(self%hidden_weight, [block_size])
        cursor = cursor + block_size
        block_size = size(self%hidden_bias)
        theta(cursor:cursor + block_size - 1) = self%hidden_bias
        cursor = cursor + block_size
        block_size = size(self%output_weight)
        theta(cursor:cursor + block_size - 1) = &
            reshape(self%output_weight, [block_size])
        cursor = cursor + block_size
        theta(cursor:) = self%output_bias
    end function rnn_parameters

    subroutine rnn_set_parameters(self, theta, status)
        class(rnn_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: cursor, block_size

        if (self%hidden_dim < 1 .or. size(theta) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RNN: parameter shape is invalid")
            return
        end if
        cursor = 1
        block_size = size(self%input_weight)
        self%input_weight = reshape(theta(cursor:cursor + block_size - 1), &
            shape(self%input_weight))
        cursor = cursor + block_size
        block_size = size(self%hidden_weight)
        self%hidden_weight = reshape(theta(cursor:cursor + block_size - 1), &
            shape(self%hidden_weight))
        cursor = cursor + block_size
        block_size = size(self%hidden_bias)
        self%hidden_bias = theta(cursor:cursor + block_size - 1)
        cursor = cursor + block_size
        block_size = size(self%output_weight)
        self%output_weight = reshape(theta(cursor:cursor + block_size - 1), &
            shape(self%output_weight))
        cursor = cursor + block_size
        self%output_bias = theta(cursor:)
        call status_set(status, FORTNUM_OK, "")
    end subroutine rnn_set_parameters

    subroutine rnn_forward(self, inputs, outputs, hidden, status)
        !! Run the scan. `hidden(t, i, :)` is the state after step `t`.
        class(rnn_t), intent(in) :: self
        real(dp), intent(in) :: inputs(:, :, :)
        real(dp), intent(out) :: outputs(:, :, :)
        real(dp), intent(out) :: hidden(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: state(:, :), preactivation(:, :)
        integer :: t, n_steps, n_batch

        if (.not. valid_sequence(self, inputs, outputs)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RNN: sequence shape is invalid")
            return
        end if
        n_steps = size(inputs, 1)
        n_batch = size(inputs, 2)
        if (size(hidden, 1) /= n_steps .or. size(hidden, 2) /= n_batch .or. &
            size(hidden, 3) /= self%hidden_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RNN: hidden state shape is invalid")
            return
        end if

        allocate(state(n_batch, self%hidden_dim))
        allocate(preactivation(n_batch, self%hidden_dim))
        state = 0.0_dp
        do t = 1, n_steps
            preactivation = matmul(inputs(t, :, :), self%input_weight) &
                + matmul(state, self%hidden_weight) &
                + spread(self%hidden_bias, dim=1, ncopies=n_batch)
            state = tanh(preactivation)
            hidden(t, :, :) = state
            outputs(t, :, :) = matmul(state, self%output_weight) &
                + spread(self%output_bias, dim=1, ncopies=n_batch)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rnn_forward

    subroutine rnn_loss(self, inputs, targets, value, status)
        class(rnn_t), intent(in) :: self
        real(dp), intent(in) :: inputs(:, :, :), targets(:, :, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: outputs(:, :, :), hidden(:, :, :)

        value = 0.0_dp
        allocate(outputs(size(targets, 1), size(targets, 2), size(targets, 3)))
        allocate(hidden(size(inputs, 1), size(inputs, 2), self%hidden_dim))
        call rnn_forward(self, inputs, outputs, hidden, status)
        if (status%code /= FORTNUM_OK) return
        value = 0.5_dp*sum((outputs - targets)**2)
    end subroutine rnn_loss

    subroutine rnn_loss_gradient(self, inputs, targets, value, gradient, status)
        !! Backpropagation through time over the packed parameter vector.
        class(rnn_t), intent(in) :: self
        real(dp), intent(in) :: inputs(:, :, :), targets(:, :, :)
        real(dp), intent(out) :: value
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: outputs(:, :, :), hidden(:, :, :)
        real(dp), allocatable :: state_bar(:, :), preactivation_bar(:, :)
        real(dp), allocatable :: previous_state(:, :), output_bar(:, :)
        real(dp), allocatable :: input_weight_bar(:, :), hidden_weight_bar(:, :)
        real(dp), allocatable :: output_weight_bar(:, :)
        real(dp), allocatable :: hidden_bias_bar(:), output_bias_bar(:)
        integer :: t, n_steps, n_batch, cursor, block_size

        value = 0.0_dp
        if (.not. valid_sequence(self, inputs, targets) .or. &
            size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RNN: gradient or sequence shape is invalid")
            return
        end if
        n_steps = size(inputs, 1)
        n_batch = size(inputs, 2)
        allocate(outputs(n_steps, n_batch, self%output_dim))
        allocate(hidden(n_steps, n_batch, self%hidden_dim))
        call rnn_forward(self, inputs, outputs, hidden, status)
        if (status%code /= FORTNUM_OK) return
        value = 0.5_dp*sum((outputs - targets)**2)

        allocate(state_bar(n_batch, self%hidden_dim))
        allocate(preactivation_bar(n_batch, self%hidden_dim))
        allocate(previous_state(n_batch, self%hidden_dim))
        allocate(output_bar(n_batch, self%output_dim))
        allocate(input_weight_bar, mold=self%input_weight)
        allocate(hidden_weight_bar, mold=self%hidden_weight)
        allocate(output_weight_bar, mold=self%output_weight)
        allocate(hidden_bias_bar, mold=self%hidden_bias)
        allocate(output_bias_bar, mold=self%output_bias)
        input_weight_bar = 0.0_dp
        hidden_weight_bar = 0.0_dp
        output_weight_bar = 0.0_dp
        hidden_bias_bar = 0.0_dp
        output_bias_bar = 0.0_dp
        state_bar = 0.0_dp

        do t = n_steps, 1, -1
            output_bar = outputs(t, :, :) - targets(t, :, :)
            output_weight_bar = output_weight_bar + &
                matmul(transpose(hidden(t, :, :)), output_bar)
            output_bias_bar = output_bias_bar + sum(output_bar, dim=1)
            state_bar = state_bar + matmul(output_bar, &
                transpose(self%output_weight))

            preactivation_bar = state_bar*(1.0_dp - hidden(t, :, :)**2)
            if (t > 1) then
                previous_state = hidden(t - 1, :, :)
            else
                previous_state = 0.0_dp
            end if
            input_weight_bar = input_weight_bar + &
                matmul(transpose(inputs(t, :, :)), preactivation_bar)
            hidden_weight_bar = hidden_weight_bar + &
                matmul(transpose(previous_state), preactivation_bar)
            hidden_bias_bar = hidden_bias_bar + sum(preactivation_bar, dim=1)
            state_bar = matmul(preactivation_bar, transpose(self%hidden_weight))
        end do

        cursor = 1
        block_size = size(input_weight_bar)
        gradient(cursor:cursor + block_size - 1) = &
            reshape(input_weight_bar, [block_size])
        cursor = cursor + block_size
        block_size = size(hidden_weight_bar)
        gradient(cursor:cursor + block_size - 1) = &
            reshape(hidden_weight_bar, [block_size])
        cursor = cursor + block_size
        block_size = size(hidden_bias_bar)
        gradient(cursor:cursor + block_size - 1) = hidden_bias_bar
        cursor = cursor + block_size
        block_size = size(output_weight_bar)
        gradient(cursor:cursor + block_size - 1) = &
            reshape(output_weight_bar, [block_size])
        cursor = cursor + block_size
        gradient(cursor:) = output_bias_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine rnn_loss_gradient

    logical function valid_sequence(self, inputs, outputs) result(valid)
        class(rnn_t), intent(in) :: self
        real(dp), intent(in) :: inputs(:, :, :), outputs(:, :, :)

        valid = self%hidden_dim > 0 .and. allocated(self%input_weight)
        if (.not. valid) return
        valid = size(inputs, 1) > 0 .and. size(inputs, 2) > 0 .and. &
            size(inputs, 3) == self%input_dim
        if (.not. valid) return
        valid = size(outputs, 1) == size(inputs, 1) .and. &
            size(outputs, 2) == size(inputs, 2) .and. &
            size(outputs, 3) == self%output_dim
    end function valid_sequence

end module fortml_rnn
