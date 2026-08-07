module fortml_trainer
    !! Objective-driven full-batch training with explicit resumable state.
    !!
    !! The trainer is deliberately independent of a model family.  Any
    !! `fortopt_objective::objective_t` can therefore use the same optimizer,
    !! clipping, projection, EMA, convergence, and history contract: MLP
    !! module trees, basis pipelines, linear estimators, GP objectives, and
    !! future physics objectives all share one state machine.  Mini-batch
    !! data loading remains the responsibility of the objective adapter; a
    !! hidden callback or host/device copy is never introduced here.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortopt_objective, only: objective_t
    use fortopt_adam, only: adam_t
    use fortopt_adamw, only: adamw_t
    use fortopt_adagrad, only: adagrad_t
    use fortopt_rmsprop, only: rmsprop_t
    use fortopt_sgd, only: sgd_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    integer, parameter, public :: FORTML_TRAIN_SGD = 1
    integer, parameter, public :: FORTML_TRAIN_ADAM = 2
    integer, parameter, public :: FORTML_TRAIN_ADAMW = 3
    integer, parameter, public :: FORTML_TRAIN_ADAGRAD = 4
    integer, parameter, public :: FORTML_TRAIN_RMSPROP = 5
    integer, parameter, public :: FORTML_TRAIN_LBFGSB = 6

    abstract interface
        subroutine trainer_step_callback_proc(step, value, gradient_norm, stop, status)
            import :: dp, fortnum_status_t
            integer, intent(in) :: step
            real(dp), intent(in) :: value, gradient_norm
            logical, intent(out) :: stop
            type(fortnum_status_t), intent(out) :: status
        end subroutine trainer_step_callback_proc
    end interface

    type, public :: trainer_options_t
        integer :: optimizer = FORTML_TRAIN_ADAM
        integer :: max_steps = 1000
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: rmsprop_decay = 0.99_dp
        real(dp) :: rmsprop_momentum = 0.0_dp
        logical :: rmsprop_centered = .false.
        real(dp) :: momentum = 0.0_dp
        logical :: nesterov = .false.
        real(dp) :: weight_decay = 0.0_dp
        real(dp) :: gradient_clip_norm = 0.0_dp
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: ema_decay = 0.0_dp
        logical :: use_bounds = .false.
        real(dp), allocatable :: lower(:), upper(:)
        type(lbfgsb_options_t) :: lbfgsb
        procedure(trainer_step_callback_proc), pointer, nopass :: callback => null()
    end type trainer_options_t

    type, public :: trainer_state_t
        integer :: n_parameters = 0
        integer :: steps = 0
        integer :: history_length = 0
        logical :: initialized = .false.
        logical :: converged = .false.
        logical :: stopped_by_callback = .false.
        integer :: clipped_steps = 0
        real(dp) :: initial_value = huge(1.0_dp)
        real(dp) :: final_value = huge(1.0_dp)
        real(dp) :: best_value = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: last_step_norm = huge(1.0_dp)
        real(dp), allocatable :: parameters(:)
        real(dp), allocatable :: ema_parameters(:)
        real(dp), allocatable :: value_history(:)
        real(dp), allocatable :: gradient_norm_history(:)
    contains
        procedure, public :: clear => trainer_state_clear
    end type trainer_state_t

    type, public :: trainer_t
        private
        type(objective_t) :: objective
        type(trainer_options_t) :: options
        type(trainer_state_t) :: state
        type(sgd_t) :: sgd
        type(adam_t) :: adam
        type(adamw_t) :: adamw
        type(adagrad_t) :: adagrad
        type(rmsprop_t) :: rmsprop
        type(lbfgsb_t) :: lbfgsb
        logical :: ready = .false.
    contains
        procedure, public :: initialize => trainer_initialize
        procedure, public :: step => trainer_step
        procedure, public :: fit => trainer_fit
        procedure, public :: parameters => trainer_parameters
        procedure, public :: state_copy => trainer_state_copy
        procedure, public :: clone => trainer_clone
        procedure, public :: value_gradient => trainer_value_gradient
        procedure, public :: initialized => trainer_initialized
    end type trainer_t

    public :: trainer_step_callback_proc

contains

    subroutine trainer_initialize(self, objective, initial, status, options)
        class(trainer_t), intent(out) :: self
        type(objective_t), intent(in) :: objective
        real(dp), intent(in) :: initial(:)
        type(fortnum_status_t), intent(out) :: status
        type(trainer_options_t), intent(in), optional :: options

        type(trainer_options_t) :: settings
        integer :: n
        real(dp), allocatable :: initial_gradient(:)
        real(dp) :: initial_value

        self%ready = .false.
        call trainer_state_clear(self%state)
        settings = trainer_options_t()
        if (present(options)) settings = options
        call validate_options(settings, objective%n_parameters, status)
        if (status%code /= FORTNUM_OK) return
        n = size(initial)
        if (n < 1 .or. n /= objective%n_parameters .or. &
            any(.not. ieee_is_finite(initial))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: objective and initial parameter shapes are invalid")
            return
        end if

        self%objective = objective
        self%options = settings
        self%state%n_parameters = n
        self%state%initialized = .true.
        allocate(self%state%parameters(n), self%state%ema_parameters(n), &
            self%state%value_history(settings%max_steps + 1), &
            self%state%gradient_norm_history(settings%max_steps + 1))
        self%state%parameters = initial
        self%state%ema_parameters = initial
        self%state%value_history = huge(1.0_dp)
        self%state%gradient_norm_history = huge(1.0_dp)

        allocate(initial_gradient(n))
        call objective%value_gradient(initial, initial_value, initial_gradient, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. ieee_is_finite(initial_value) .or. &
            any(.not. ieee_is_finite(initial_gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: initial objective returned a non-finite product")
            return
        end if
        self%state%initial_value = initial_value
        self%state%final_value = initial_value
        self%state%best_value = initial_value
        self%state%gradient_norm = sqrt(max(0.0_dp, &
            dot_product(initial_gradient, initial_gradient)))
        self%state%value_history(1) = initial_value
        self%state%gradient_norm_history(1) = self%state%gradient_norm
        self%state%history_length = 1

        select case (settings%optimizer)
        case (FORTML_TRAIN_SGD)
            call self%sgd%initialize(n, status, settings%learning_rate, &
                settings%momentum, settings%nesterov)
        case (FORTML_TRAIN_ADAM)
            call self%adam%initialize(n, status, settings%learning_rate, &
                settings%beta1, settings%beta2, settings%epsilon)
        case (FORTML_TRAIN_ADAMW)
            call self%adamw%initialize(n, status, settings%learning_rate, &
                settings%beta1, settings%beta2, settings%epsilon, &
                settings%weight_decay)
        case (FORTML_TRAIN_ADAGRAD)
            call self%adagrad%initialize(n, status, settings%learning_rate, &
                settings%epsilon)
        case (FORTML_TRAIN_RMSPROP)
            call self%rmsprop%initialize(n, status, settings%learning_rate, &
                settings%rmsprop_decay, settings%epsilon, &
                settings%rmsprop_momentum, settings%rmsprop_centered)
        case (FORTML_TRAIN_LBFGSB)
            ! L-BFGS-B initializes its own history during fit.
            call status_set(status, FORTNUM_OK, "")
        end select
        if (status%code /= FORTNUM_OK) return

        self%ready = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine trainer_initialize

    subroutine trainer_step(self, status)
        class(trainer_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: gradient(:), before(:)
        real(dp) :: value, norm, scale, step_norm, new_value
        logical :: stop

        if (.not. self%ready .or. .not. self%state%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: state is not initialized")
            return
        end if
        if (self%options%optimizer == FORTML_TRAIN_LBFGSB) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: L-BFGS-B is a fit-level optimizer")
            return
        end if
        allocate(gradient(self%state%n_parameters), before(self%state%n_parameters))
        call self%objective%value_gradient(self%state%parameters, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: objective returned a non-finite value or gradient")
            return
        end if

        norm = sqrt(max(0.0_dp, dot_product(gradient, gradient)))
        if (norm <= self%options%tolerance) then
            self%state%converged = .true.
            self%state%gradient_norm = norm
            self%state%final_value = value
            call record_history(self, value, norm)
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        before = self%state%parameters
        if (self%options%gradient_clip_norm > 0.0_dp .and. &
            norm > self%options%gradient_clip_norm) then
            scale = self%options%gradient_clip_norm/norm
            gradient = scale*gradient
            norm = self%options%gradient_clip_norm
            self%state%clipped_steps = self%state%clipped_steps + 1
        end if

        select case (self%options%optimizer)
        case (FORTML_TRAIN_SGD)
            call self%sgd%step(self%state%parameters, gradient, status)
        case (FORTML_TRAIN_ADAM)
            call self%adam%step(self%state%parameters, gradient, status)
        case (FORTML_TRAIN_ADAMW)
            call self%adamw%step(self%state%parameters, gradient, status)
        case (FORTML_TRAIN_ADAGRAD)
            call self%adagrad%step(self%state%parameters, gradient, status)
        case (FORTML_TRAIN_RMSPROP)
            call self%rmsprop%step(self%state%parameters, gradient, status)
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: unsupported optimizer kind")
            return
        end select
        if (status%code /= FORTNUM_OK) then
            self%state%parameters = before
            return
        end if

        if (self%options%use_bounds) then
            self%state%parameters = min(self%options%upper, &
                max(self%options%lower, self%state%parameters))
        end if
        if (any(.not. ieee_is_finite(self%state%parameters))) then
            self%state%parameters = before
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: optimizer produced non-finite parameters")
            return
        end if
        step_norm = sqrt(max(0.0_dp, dot_product( &
            self%state%parameters - before, self%state%parameters - before)))
        self%state%last_step_norm = step_norm
        self%state%steps = self%state%steps + 1
        self%state%gradient_norm = norm
        if (self%options%ema_decay > 0.0_dp) then
            self%state%ema_parameters = self%options%ema_decay* &
                self%state%ema_parameters + (1.0_dp - self%options%ema_decay)* &
                self%state%parameters
        end if

        call self%objective%value_gradient(self%state%parameters, new_value, &
            gradient, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. ieee_is_finite(new_value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: post-update objective is non-finite")
            return
        end if
        self%state%final_value = new_value
        if (self%state%steps == 1) self%state%initial_value = value
        self%state%best_value = min(self%state%best_value, new_value)
        call record_history(self, new_value, norm)
        if (step_norm <= self%options%step_tolerance .or. &
            abs(value - new_value) <= self%options%objective_tolerance) then
            self%state%converged = .true.
        end if
        stop = .false.
        if (associated(self%options%callback)) then
            call self%options%callback(self%state%steps, new_value, norm, stop, status)
            if (status%code /= FORTNUM_OK) return
            if (stop) self%state%stopped_by_callback = .true.
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine trainer_step

    subroutine trainer_fit(self, status)
        class(trainer_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status

        type(lbfgsb_result_t) :: result
        real(dp), allocatable :: lower(:), upper(:), gradient(:)
        real(dp) :: value
        integer :: i

        if (.not. self%ready) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: state is not initialized")
            return
        end if
        if (self%options%optimizer == FORTML_TRAIN_LBFGSB) then
            allocate(lower(self%state%n_parameters), upper(self%state%n_parameters))
            if (self%options%use_bounds) then
                lower = self%options%lower
                upper = self%options%upper
            else
                lower = -huge(1.0_dp)
                upper = huge(1.0_dp)
            end if
            call self%lbfgsb%minimize(self%objective, self%state%parameters, lower, &
                upper, self%options%lbfgsb, result, status)
            if (status%code /= FORTNUM_OK) return
            self%state%steps = result%state%iteration
            self%state%converged = result%state%converged
            allocate(gradient(self%state%n_parameters))
            call self%objective%value_gradient(self%state%parameters, value, gradient, status)
            if (status%code /= FORTNUM_OK) return
            self%state%initial_value = result%state%previous_value
            self%state%final_value = value
            self%state%best_value = value
            self%state%gradient_norm = sqrt(max(0.0_dp, dot_product(gradient, gradient)))
            call record_history(self, value, self%state%gradient_norm)
            if (self%options%ema_decay > 0.0_dp) self%state%ema_parameters = &
                self%state%parameters
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        do i = self%state%steps + 1, self%options%max_steps
            call self%step(status)
            if (status%code /= FORTNUM_OK) return
            if (self%state%converged .or. self%state%stopped_by_callback) exit
        end do
        if (.not. self%state%converged .and. .not. self%state%stopped_by_callback .and. &
            self%state%steps >= self%options%max_steps) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "trainer: maximum steps reached before convergence")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine trainer_fit

    function trainer_parameters(self) result(parameters)
        class(trainer_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (allocated(self%state%parameters)) then
            parameters = self%state%parameters
        else
            allocate(parameters(0))
        end if
    end function trainer_parameters

    function trainer_state_copy(self) result(state)
        class(trainer_t), intent(in) :: self
        type(trainer_state_t) :: state

        state = self%state
    end function trainer_state_copy

    subroutine trainer_clone(self, copy, status)
        class(trainer_t), intent(in) :: self
        type(trainer_t), intent(out) :: copy
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%ready) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: cannot clone an uninitialized state")
            return
        end if
        copy = self
        call status_set(status, FORTNUM_OK, "")
    end subroutine trainer_clone

    subroutine trainer_value_gradient(self, value, gradient, status)
        class(trainer_t), intent(in) :: self
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%ready .or. size(gradient) /= self%state%n_parameters) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: value-gradient shape or state is invalid")
            return
        end if
        call self%objective%value_gradient(self%state%parameters, value, gradient, status)
    end subroutine trainer_value_gradient

    logical function trainer_initialized(self)
        class(trainer_t), intent(in) :: self
        trainer_initialized = self%ready .and. self%state%initialized
    end function trainer_initialized

    subroutine trainer_state_clear(self)
        class(trainer_state_t), intent(inout) :: self
        self%n_parameters = 0
        self%steps = 0
        self%history_length = 0
        self%initialized = .false.
        self%converged = .false.
        self%stopped_by_callback = .false.
        self%clipped_steps = 0
        self%initial_value = huge(1.0_dp)
        self%final_value = huge(1.0_dp)
        self%best_value = huge(1.0_dp)
        self%gradient_norm = huge(1.0_dp)
        self%last_step_norm = huge(1.0_dp)
        if (allocated(self%parameters)) deallocate(self%parameters)
        if (allocated(self%ema_parameters)) deallocate(self%ema_parameters)
        if (allocated(self%value_history)) deallocate(self%value_history)
        if (allocated(self%gradient_norm_history)) deallocate(self%gradient_norm_history)
    end subroutine trainer_state_clear

    subroutine record_history(self, value, gradient_norm)
        class(trainer_t), intent(inout) :: self
        real(dp), intent(in) :: value, gradient_norm
        integer :: index

        index = min(self%state%steps + 1, size(self%state%value_history))
        if (self%state%history_length == 0) then
            self%state%value_history(1) = value
            self%state%gradient_norm_history(1) = gradient_norm
            self%state%history_length = 1
        else if (index > self%state%history_length) then
            self%state%value_history(index) = value
            self%state%gradient_norm_history(index) = gradient_norm
            self%state%history_length = index
        else
            self%state%value_history(index) = value
            self%state%gradient_norm_history(index) = gradient_norm
        end if
    end subroutine record_history

    subroutine validate_options(options, n_parameters, status)
        type(trainer_options_t), intent(in) :: options
        integer, intent(in) :: n_parameters
        type(fortnum_status_t), intent(out) :: status

        if (n_parameters < 1 .or. options%max_steps < 1 .or. &
            options%optimizer < FORTML_TRAIN_SGD .or. &
            options%optimizer > FORTML_TRAIN_LBFGSB .or. &
            .not. ieee_is_finite(options%learning_rate) .or. &
            options%learning_rate <= 0.0_dp .or. &
            .not. ieee_is_finite(options%beta1) .or. options%beta1 < 0.0_dp .or. &
            options%beta1 >= 1.0_dp .or. .not. ieee_is_finite(options%beta2) .or. &
            options%beta2 < 0.0_dp .or. options%beta2 >= 1.0_dp .or. &
            .not. ieee_is_finite(options%epsilon) .or. options%epsilon <= 0.0_dp .or. &
            .not. ieee_is_finite(options%rmsprop_decay) .or. &
            options%rmsprop_decay < 0.0_dp .or. options%rmsprop_decay >= 1.0_dp .or. &
            .not. ieee_is_finite(options%rmsprop_momentum) .or. &
            options%rmsprop_momentum < 0.0_dp .or. options%rmsprop_momentum >= 1.0_dp .or. &
            .not. ieee_is_finite(options%momentum) .or. options%momentum < 0.0_dp .or. &
            options%momentum >= 1.0_dp .or. (options%nesterov .and. options%momentum <= 0.0_dp) .or. &
            .not. ieee_is_finite(options%weight_decay) .or. options%weight_decay < 0.0_dp .or. &
            .not. ieee_is_finite(options%gradient_clip_norm) .or. options%gradient_clip_norm < 0.0_dp .or. &
            .not. ieee_is_finite(options%tolerance) .or. options%tolerance < 0.0_dp .or. &
            .not. ieee_is_finite(options%step_tolerance) .or. options%step_tolerance < 0.0_dp .or. &
            .not. ieee_is_finite(options%objective_tolerance) .or. options%objective_tolerance < 0.0_dp .or. &
            .not. ieee_is_finite(options%ema_decay) .or. options%ema_decay < 0.0_dp .or. &
            options%ema_decay >= 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: invalid optimizer or convergence option")
            return
        end if
        if (options%use_bounds) then
            if (.not. allocated(options%lower) .or. .not. allocated(options%upper) .or. &
                size(options%lower) /= n_parameters .or. &
                size(options%upper) /= n_parameters .or. &
                any(.not. ieee_is_finite(options%lower)) .or. &
                any(.not. ieee_is_finite(options%upper)) .or. &
                any(options%lower > options%upper)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "trainer: bounds are missing, nonfinite, or inconsistent")
                return
            end if
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_options

end module fortml_trainer
