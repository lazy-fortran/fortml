module fortml_adafactor_factored
    !! Layout-aware Adafactor with matrix-factorized second-moment state.
    !!
    !! Matrix blocks use the Adafactor row/column estimator
    !! ``v = outer(row, column) / mean(row)``.  One-dimensional blocks use
    !! the exact unfactored vector recurrence.  The packed parameter vector is
    !! deliberately explicit through `adafactor_block_spec_t`; this avoids
    !! guessing a matrix shape from a flat optimizer state.  The recurrence is
    !! deterministic and CPU-resident.  `step_device` returns a typed refusal
    !! for CUDA until a resident row/column kernel and transfer contract exist.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    type, public :: adafactor_block_spec_t
        !! One contiguous block in a packed parameter vector.
        integer :: first = 0
        integer :: last = -1
        integer :: rows = 0
        integer :: columns = 0
        logical :: factored = .false.
    contains
        procedure, public :: valid => adafactor_block_spec_valid
    end type adafactor_block_spec_t

    type, public :: adafactor_factored_block_state_t
        real(dp), allocatable :: row_moment(:)
        real(dp), allocatable :: column_moment(:)
        real(dp), allocatable :: second_moment(:)
    end type adafactor_factored_block_state_t

    type, public :: adafactor_factored_t
        !! Deterministic layout-aware Adafactor recurrence.
        type(adafactor_block_spec_t), allocatable :: blocks(:)
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: decay = 0.999_dp
        real(dp) :: epsilon = 1.0e-30_dp
        real(dp) :: clip_threshold = 1.0_dp
        logical :: relative_step = .false.
        logical :: scale_parameter = .false.
        integer :: n_parameters = 0
        integer :: step_count = 0
        type(adafactor_factored_block_state_t), allocatable :: state(:)
    contains
        procedure, public :: initialize => adafactor_factored_initialize
        procedure, public :: step => adafactor_factored_step
        procedure, public :: step_device => adafactor_factored_step_device
        procedure, public :: device_supported => adafactor_factored_device_supported
        procedure, public :: dense_second_moment => adafactor_factored_dense_second_moment
        procedure, public :: initialized => adafactor_factored_initialized
    end type adafactor_factored_t

    public :: adafactor_factored_initialize
    public :: adafactor_factored_step

contains

    logical function adafactor_block_spec_valid(self) result(valid)
        class(adafactor_block_spec_t), intent(in) :: self
        integer :: count

        count = self%last - self%first + 1
        valid = self%first >= 1 .and. self%last >= self%first .and. &
            self%rows >= 1 .and. self%columns >= 1 .and. &
            count == self%rows*self%columns
        if (self%factored) valid = valid .and. self%rows > 1 .and. self%columns > 1
    end function adafactor_block_spec_valid

    subroutine adafactor_factored_initialize(self, n_parameters, blocks, status, &
            learning_rate, decay, epsilon, clip_threshold, relative_step, &
            scale_parameter)
        class(adafactor_factored_t), intent(out) :: self
        integer, intent(in) :: n_parameters
        type(adafactor_block_spec_t), intent(in) :: blocks(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: learning_rate, decay, epsilon, clip_threshold
        logical, intent(in), optional :: relative_step, scale_parameter
        integer :: i, expected

        self%learning_rate = 1.0e-3_dp
        self%decay = 0.999_dp
        self%epsilon = 1.0e-30_dp
        self%clip_threshold = 1.0_dp
        self%relative_step = .false.
        self%scale_parameter = .false.
        if (present(learning_rate)) self%learning_rate = learning_rate
        if (present(decay)) self%decay = decay
        if (present(epsilon)) self%epsilon = epsilon
        if (present(clip_threshold)) self%clip_threshold = clip_threshold
        if (present(relative_step)) self%relative_step = relative_step
        if (present(scale_parameter)) self%scale_parameter = scale_parameter

        if (n_parameters < 1 .or. size(blocks) < 1 .or. &
                .not. ieee_is_finite(self%learning_rate) .or. &
                self%learning_rate <= 0.0_dp .or. .not. ieee_is_finite(self%decay) .or. &
                self%decay < 0.0_dp .or. self%decay >= 1.0_dp .or. &
                .not. ieee_is_finite(self%epsilon) .or. self%epsilon <= 0.0_dp .or. &
                .not. ieee_is_finite(self%clip_threshold) .or. self%clip_threshold <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: invalid dimension or hyperparameter")
            return
        end if
        expected = 1
        do i = 1, size(blocks)
            if (.not. blocks(i)%valid() .or. blocks(i)%first /= expected .or. &
                    blocks(i)%last > n_parameters) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "factored Adafactor: blocks must cover the packed vector")
                return
            end if
            expected = blocks(i)%last + 1
        end do
        if (expected /= n_parameters + 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: blocks do not cover all parameters")
            return
        end if

        allocate(self%blocks(size(blocks)), self%state(size(blocks)))
        self%blocks = blocks
        do i = 1, size(blocks)
            if (blocks(i)%factored) then
                allocate(self%state(i)%row_moment(blocks(i)%rows), &
                    self%state(i)%column_moment(blocks(i)%columns))
                self%state(i)%row_moment = 0.0_dp
                self%state(i)%column_moment = 0.0_dp
            else
                allocate(self%state(i)%second_moment(blocks(i)%last - blocks(i)%first + 1))
                self%state(i)%second_moment = 0.0_dp
            end if
        end do
        self%n_parameters = n_parameters
        self%step_count = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine adafactor_factored_initialize

    logical function adafactor_factored_initialized(self) result(yes)
        class(adafactor_factored_t), intent(in) :: self
        yes = self%n_parameters > 0 .and. allocated(self%blocks) .and. &
            allocated(self%state) .and. size(self%blocks) == size(self%state)
    end function adafactor_factored_initialized

    logical function adafactor_factored_device_supported(self, device_kind) result(supported)
        class(adafactor_factored_t), intent(in) :: self
        integer, intent(in) :: device_kind
        supported = .false.
        if (.not. self%initialized()) return
        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = .true.
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        end select
    end function adafactor_factored_device_supported

    subroutine adafactor_factored_step_device(self, device, x, gradient, status)
        class(adafactor_factored_t), intent(inout) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(inout) :: x(:)
        real(dp), intent(in) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%step(x, gradient, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "factored Adafactor: resident CUDA row/column state is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: invalid device kind")
        end select
    end subroutine adafactor_factored_step_device

    subroutine adafactor_factored_step(self, x, gradient, status)
        class(adafactor_factored_t), intent(inout) :: self
        real(dp), intent(inout) :: x(:)
        real(dp), intent(in) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: dense(:)
        real(dp) :: update_rms, clip_scale, rate, parameter_rms
        integer :: i, j, k, index, count

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: optimizer is not initialized")
            return
        end if
        if (size(x) /= self%n_parameters .or. size(gradient) /= self%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: state and gradient dimensions do not match")
            return
        end if
        if (.not. ieee_is_finite(self%learning_rate) .or. self%learning_rate <= 0.0_dp .or. &
                .not. ieee_is_finite(self%decay) .or. self%decay < 0.0_dp .or. &
                self%decay >= 1.0_dp .or. .not. ieee_is_finite(self%epsilon) .or. &
                self%epsilon <= 0.0_dp .or. .not. ieee_is_finite(self%clip_threshold) .or. &
                self%clip_threshold <= 0.0_dp .or. any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: invalid state or non-finite parameter/gradient")
            return
        end if
        if (self%step_count == huge(self%step_count)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: step count overflow")
            return
        end if

        allocate(dense(self%n_parameters))
        dense = 0.0_dp
        do i = 1, size(self%blocks)
            count = self%blocks(i)%last - self%blocks(i)%first + 1
            if (self%blocks(i)%factored) then
                self%state(i)%row_moment = self%decay*self%state(i)%row_moment
                self%state(i)%column_moment = self%decay*self%state(i)%column_moment
                do j = 1, self%blocks(i)%columns
                    do k = 1, self%blocks(i)%rows
                        index = self%blocks(i)%first + (j - 1)*self%blocks(i)%rows + k - 1
                        self%state(i)%row_moment(k) = self%state(i)%row_moment(k) + &
                            (1.0_dp - self%decay)*gradient(index)**2 / &
                            real(self%blocks(i)%columns, dp)
                        self%state(i)%column_moment(j) = self%state(i)%column_moment(j) + &
                            (1.0_dp - self%decay)*gradient(index)**2 / &
                            real(self%blocks(i)%rows, dp)
                    end do
                end do
                if (any(.not. ieee_is_finite(self%state(i)%row_moment)) .or. &
                        any(.not. ieee_is_finite(self%state(i)%column_moment))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "factored Adafactor: factor state overflow")
                    return
                end if
                do j = 1, self%blocks(i)%columns
                    do k = 1, self%blocks(i)%rows
                        index = self%blocks(i)%first + (j - 1)*self%blocks(i)%rows + k - 1
                        dense(index) = self%state(i)%row_moment(k)*self%state(i)%column_moment(j) / &
                            max(sum(self%state(i)%row_moment) / &
                            real(self%blocks(i)%rows, dp), self%epsilon)
                    end do
                end do
            else
                self%state(i)%second_moment = self%decay*self%state(i)%second_moment + &
                    (1.0_dp - self%decay)*gradient(self%blocks(i)%first:self%blocks(i)%last)**2
                if (any(.not. ieee_is_finite(self%state(i)%second_moment))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "factored Adafactor: vector state overflow")
                    return
                end if
                dense(self%blocks(i)%first:self%blocks(i)%last) = &
                    self%state(i)%second_moment
            end if
        end do

        update_rms = sqrt(sum(dense)/real(self%n_parameters, dp))
        clip_scale = max(1.0_dp, update_rms/self%clip_threshold)
        rate = self%learning_rate
        if (self%relative_step) rate = min(rate, 1.0_dp/sqrt(real(self%step_count + 1, dp)))
        if (self%scale_parameter) then
            parameter_rms = sqrt(sum(x**2)/real(self%n_parameters, dp))
            rate = rate*max(parameter_rms, self%epsilon)
        end if
        do i = 1, self%n_parameters
            x(i) = x(i) - rate*gradient(i)/clip_scale/(sqrt(dense(i)) + self%epsilon)
        end do
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: parameter update is non-finite")
            return
        end if
        self%step_count = self%step_count + 1
        call status_set(status, FORTNUM_OK, "")
    end subroutine adafactor_factored_step

    subroutine adafactor_factored_dense_second_moment(self, values, status)
        class(adafactor_factored_t), intent(in) :: self
        real(dp), allocatable, intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k, index

        if (.not. self%initialized()) then
            allocate(values(0))
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: optimizer is not initialized")
            return
        end if
        allocate(values(self%n_parameters))
        values = 0.0_dp
        do i = 1, size(self%blocks)
            if (self%blocks(i)%factored) then
                do j = 1, self%blocks(i)%columns
                    do k = 1, self%blocks(i)%rows
                        index = self%blocks(i)%first + (j - 1)*self%blocks(i)%rows + k - 1
                        values(index) = self%state(i)%row_moment(k)*self%state(i)%column_moment(j) / &
                            max(sum(self%state(i)%row_moment) / &
                            real(self%blocks(i)%rows, dp), self%epsilon)
                    end do
                end do
            else
                values(self%blocks(i)%first:self%blocks(i)%last) = self%state(i)%second_moment
            end if
        end do
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "factored Adafactor: dense state is non-finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine adafactor_factored_dense_second_moment

end module fortml_adafactor_factored
