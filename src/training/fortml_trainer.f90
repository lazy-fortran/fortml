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
    use, intrinsic :: iso_fortran_env, only: iostat_end
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortopt_objective, only: objective_t
    use fortopt_adam, only: adam_t
    use fortopt_adamw, only: adamw_t
    use fortopt_adagrad, only: adagrad_t
    use fortopt_rmsprop, only: rmsprop_t
    use fortopt_sgd, only: sgd_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    use fortml_adafactor, only: adafactor_t
    use fortml_lion, only: lion_t
    use fortml_mlp_schedules, only: mlp_learning_rate_schedule_t, &
        MLP_SCHEDULE_PLATEAU
    implicit none
    private

    integer, parameter, public :: FORTML_TRAIN_SGD = 1
    integer, parameter, public :: FORTML_TRAIN_ADAM = 2
    integer, parameter, public :: FORTML_TRAIN_ADAMW = 3
    integer, parameter, public :: FORTML_TRAIN_ADAGRAD = 4
    integer, parameter, public :: FORTML_TRAIN_RMSPROP = 5
    integer, parameter, public :: FORTML_TRAIN_LBFGSB = 6
    integer, parameter, public :: FORTML_TRAIN_ADAFACTOR = 7
    integer, parameter, public :: FORTML_TRAIN_LION = 8
    character(*), parameter, public :: FORTML_TRAINER_CHECKPOINT_MAGIC = &
        "FORTML_TRAINER_CHECKPOINT_TEXT"
    integer, parameter, public :: FORTML_TRAINER_CHECKPOINT_SCHEMA_VERSION = 6

    abstract interface
        subroutine trainer_step_callback_proc(step, value, gradient_norm, stop, status)
            import :: dp, fortnum_status_t
            integer, intent(in) :: step
            real(dp), intent(in) :: value, gradient_norm
            logical, intent(out) :: stop
            type(fortnum_status_t), intent(out) :: status
        end subroutine trainer_step_callback_proc

        subroutine trainer_validation_callback_proc(step, parameters, validation_value, status)
            import :: dp, fortnum_status_t
            integer, intent(in) :: step
            real(dp), intent(in) :: parameters(:)
            real(dp), intent(out) :: validation_value
            type(fortnum_status_t), intent(out) :: status
        end subroutine trainer_validation_callback_proc
    end interface

    type, public :: trainer_options_t
        integer :: optimizer = FORTML_TRAIN_ADAM
        integer :: max_steps = 1000
        real(dp) :: learning_rate = 1.0e-3_dp
        !! Optional stateless schedule evaluated at every streaming update.
        !! The base `learning_rate` remains the first-class hyperparameter;
        !! the schedule only supplies a positive multiplicative trajectory.
        logical :: use_learning_rate_schedule = .false.
        type(mlp_learning_rate_schedule_t) :: learning_rate_schedule
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: adafactor_decay = 0.999_dp
        real(dp) :: adafactor_clip_threshold = 1.0_dp
        logical :: adafactor_relative_step = .false.
        logical :: adafactor_scale_parameter = .false.
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
        integer :: validation_patience = 0
        real(dp) :: validation_min_delta = 0.0_dp
        logical :: validation_higher_is_better = .false.
        logical :: validation_restore_best = .false.
        type(lbfgsb_options_t) :: lbfgsb
        procedure(trainer_step_callback_proc), pointer, nopass :: callback => null()
        procedure(trainer_validation_callback_proc), pointer, nopass :: validation_callback => null()
    end type trainer_options_t

    type, public :: trainer_state_t
        integer :: n_parameters = 0
        integer :: steps = 0
        integer :: history_length = 0
        logical :: initialized = .false.
        logical :: converged = .false.
        logical :: stopped_by_callback = .false.
        logical :: stopped_by_validation = .false.
        integer :: clipped_steps = 0
        integer :: validation_history_length = 0
        integer :: validation_bad_steps = 0
        integer :: validation_best_step = 0
        real(dp) :: initial_value = huge(1.0_dp)
        real(dp) :: final_value = huge(1.0_dp)
        real(dp) :: best_value = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
        real(dp) :: last_step_norm = huge(1.0_dp)
        real(dp) :: last_learning_rate = 0.0_dp
        real(dp) :: validation_value = huge(1.0_dp)
        real(dp) :: best_validation_value = huge(1.0_dp)
        real(dp), allocatable :: parameters(:)
        real(dp), allocatable :: ema_parameters(:)
        real(dp), allocatable :: value_history(:)
        real(dp), allocatable :: gradient_norm_history(:)
        real(dp), allocatable :: validation_history(:)
        real(dp), allocatable :: validation_best_parameters(:)
        real(dp), allocatable :: learning_rate_history(:)
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
        type(adafactor_t) :: adafactor
        type(lion_t) :: lion
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
        procedure, public :: save_checkpoint => trainer_save_checkpoint
        procedure, public :: load_checkpoint => trainer_load_checkpoint
    end type trainer_t

    public :: trainer_step_callback_proc
    public :: trainer_validation_callback_proc

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
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(trainer_options_t) :: trainer_options_t_default

        self%ready = .false.
        call trainer_state_clear(self%state)
        settings = trainer_options_t_default
        if (present(options)) settings = options
        call validate_options(settings, objective%n_parameters, status)
        if (status%code /= FORTNUM_OK) return
        if (settings%optimizer == FORTML_TRAIN_LBFGSB .and. &
            associated(settings%validation_callback)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: validation callback is unavailable for L-BFGS-B")
            return
        end if
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
            self%state%gradient_norm_history(settings%max_steps + 1), &
            self%state%validation_history(settings%max_steps + 1), &
            self%state%validation_best_parameters(n), &
            self%state%learning_rate_history(settings%max_steps + 1))
        self%state%parameters = initial
        self%state%ema_parameters = initial
        self%state%value_history = huge(1.0_dp)
        self%state%gradient_norm_history = huge(1.0_dp)
        self%state%validation_history = huge(1.0_dp)
        self%state%validation_best_parameters = initial
        self%state%learning_rate_history = 0.0_dp
        self%state%last_learning_rate = settings%learning_rate
        self%state%learning_rate_history(1) = settings%learning_rate

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
        if (associated(settings%validation_callback)) then
            call settings%validation_callback(0, initial, self%state%validation_value, status)
            if (status%code /= FORTNUM_OK) return
            if (.not. ieee_is_finite(self%state%validation_value)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "trainer: validation callback returned a non-finite value")
                return
            end if
            self%state%best_validation_value = self%state%validation_value
            self%state%validation_history(1) = self%state%validation_value
            self%state%validation_history_length = 1
            self%state%validation_best_step = 0
        end if

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
        case (FORTML_TRAIN_ADAFACTOR)
            call self%adafactor%initialize(n, status, settings%learning_rate, &
                settings%adafactor_decay, settings%epsilon, &
                settings%adafactor_clip_threshold, settings%adafactor_relative_step, &
                settings%adafactor_scale_parameter)
        case (FORTML_TRAIN_LION)
            call self%lion%initialize(n, status, settings%learning_rate, &
                settings%beta1, settings%beta2, settings%weight_decay)
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
        real(dp) :: learning_rate, previous_learning_rate
        logical :: stop, validation_stop

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
            call record_validation(self, validation_stop, status)
            if (status%code /= FORTNUM_OK) return
            if (validation_stop) self%state%stopped_by_validation = .true.
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

        previous_learning_rate = self%state%last_learning_rate
        learning_rate = self%options%learning_rate
        if (self%options%use_learning_rate_schedule) then
            call self%options%learning_rate_schedule%rate(self%state%steps + 1, &
                self%options%learning_rate, learning_rate, status)
            if (status%code /= FORTNUM_OK) return
        end if
        call set_optimizer_learning_rate(self, learning_rate)

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
        case (FORTML_TRAIN_ADAFACTOR)
            call self%adafactor%step(self%state%parameters, gradient, status)
        case (FORTML_TRAIN_LION)
            call self%lion%step(self%state%parameters, gradient, status)
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: unsupported optimizer kind")
            return
        end select
        if (status%code /= FORTNUM_OK) then
            self%state%parameters = before
            call set_optimizer_learning_rate(self, previous_learning_rate)
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
        self%state%last_learning_rate = learning_rate
        self%state%learning_rate_history(self%state%steps + 1) = learning_rate
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
        call record_validation(self, validation_stop, status)
        if (status%code /= FORTNUM_OK) return
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
        if (validation_stop) self%state%stopped_by_validation = .true.
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
            if (self%state%converged .or. self%state%stopped_by_callback .or. &
                self%state%stopped_by_validation) exit
        end do
        if (.not. self%state%converged .and. .not. self%state%stopped_by_callback .and. &
            .not. self%state%stopped_by_validation .and. &
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

    subroutine trainer_save_checkpoint(self, path, status)
        !! Save a complete, portable trainer snapshot.
        !!
        !! The objective procedure is intentionally not serialized: a loaded
        !! snapshot must be attached to the same objective (and therefore is
        !! loaded into an already initialized trainer).  Optimizer recurrence
        !! state is serialized for the streaming optimizers.  L-BFGS-B is a
        !! fit-level optimizer in this API and has no resumable state here.
        class(trainer_t), intent(in) :: self
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        integer :: unit, ios, close_ios

        call validate_checkpoint(self, status)
        if (status%code /= FORTNUM_OK) return
        open (newunit=unit, file=path, status="replace", action="write", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint save: cannot open destination")
            return
        end if

        write (unit, "(A)", iostat=ios) FORTML_TRAINER_CHECKPOINT_MAGIC
        if (ios == 0) call write_i(unit, "schema_version", &
            FORTML_TRAINER_CHECKPOINT_SCHEMA_VERSION, ios)
        if (ios == 0) call write_i(unit, "optimizer", self%options%optimizer, ios)
        if (ios == 0) call write_i(unit, "max_steps", self%options%max_steps, ios)
        if (ios == 0) call write_r(unit, "learning_rate", self%options%learning_rate, ios)
        if (ios == 0) call write_l(unit, "use_learning_rate_schedule", &
            self%options%use_learning_rate_schedule, ios)
        if (ios == 0) call write_i(unit, "schedule_kind", &
            self%options%learning_rate_schedule%kind, ios)
        if (ios == 0) call write_i(unit, "schedule_warmup_updates", &
            self%options%learning_rate_schedule%warmup_updates, ios)
        if (ios == 0) call write_i(unit, "schedule_total_updates", &
            self%options%learning_rate_schedule%total_updates, ios)
        if (ios == 0) call write_r(unit, "schedule_min_rate_fraction", &
            self%options%learning_rate_schedule%min_rate_fraction, ios)
        if (ios == 0) call write_r(unit, "schedule_decay_factor", &
            self%options%learning_rate_schedule%decay_factor, ios)
        if (ios == 0) call write_r(unit, "schedule_peak_rate_fraction", &
            self%options%learning_rate_schedule%peak_rate_fraction, ios)
        if (ios == 0) call write_r(unit, "schedule_final_rate_fraction", &
            self%options%learning_rate_schedule%final_rate_fraction, ios)
        if (ios == 0) call write_i(unit, "schedule_metric_mode", &
            self%options%learning_rate_schedule%metric_mode, ios)
        if (ios == 0) call write_i(unit, "schedule_patience_updates", &
            self%options%learning_rate_schedule%patience_updates, ios)
        if (ios == 0) call write_r(unit, "schedule_min_delta", &
            self%options%learning_rate_schedule%min_delta, ios)
        if (ios == 0) call write_r(unit, "schedule_plateau_factor", &
            self%options%learning_rate_schedule%plateau_factor, ios)
        if (ios == 0) call write_r(unit, "beta1", self%options%beta1, ios)
        if (ios == 0) call write_r(unit, "beta2", self%options%beta2, ios)
        if (ios == 0) call write_r(unit, "epsilon", self%options%epsilon, ios)
        if (ios == 0) call write_r(unit, "adafactor_decay", self%options%adafactor_decay, ios)
        if (ios == 0) call write_r(unit, "adafactor_clip_threshold", self%options%adafactor_clip_threshold, ios)
        if (ios == 0) call write_l(unit, "adafactor_relative_step", self%options%adafactor_relative_step, ios)
        if (ios == 0) call write_l(unit, "adafactor_scale_parameter", self%options%adafactor_scale_parameter, ios)
        if (ios == 0) call write_r(unit, "rmsprop_decay", self%options%rmsprop_decay, ios)
        if (ios == 0) call write_r(unit, "rmsprop_momentum", self%options%rmsprop_momentum, ios)
        if (ios == 0) call write_l(unit, "rmsprop_centered", self%options%rmsprop_centered, ios)
        if (ios == 0) call write_r(unit, "momentum", self%options%momentum, ios)
        if (ios == 0) call write_l(unit, "nesterov", self%options%nesterov, ios)
        if (ios == 0) call write_r(unit, "weight_decay", self%options%weight_decay, ios)
        if (ios == 0) call write_r(unit, "gradient_clip_norm", self%options%gradient_clip_norm, ios)
        if (ios == 0) call write_r(unit, "tolerance", self%options%tolerance, ios)
        if (ios == 0) call write_r(unit, "step_tolerance", self%options%step_tolerance, ios)
        if (ios == 0) call write_r(unit, "objective_tolerance", self%options%objective_tolerance, ios)
        if (ios == 0) call write_r(unit, "ema_decay", self%options%ema_decay, ios)
        if (ios == 0) call write_l(unit, "use_bounds", self%options%use_bounds, ios)
        if (ios == 0) call write_i(unit, "validation_patience", &
            self%options%validation_patience, ios)
        if (ios == 0) call write_r(unit, "validation_min_delta", &
            self%options%validation_min_delta, ios)
        if (ios == 0) call write_l(unit, "validation_higher_is_better", &
            self%options%validation_higher_is_better, ios)
        if (ios == 0) call write_l(unit, "validation_restore_best", &
            self%options%validation_restore_best, ios)
        if (ios == 0) call write_l(unit, "validation_callback_present", &
            associated(self%options%validation_callback), ios)
        if (ios == 0) call write_i(unit, "lbfgsb_memory", self%options%lbfgsb%memory, ios)
        if (ios == 0) call write_i(unit, "lbfgsb_max_iterations", self%options%lbfgsb%max_iterations, ios)
        if (ios == 0) call write_i(unit, "lbfgsb_max_line_search", self%options%lbfgsb%max_line_search, ios)
        if (ios == 0) call write_r(unit, "lbfgsb_gradient_tolerance", self%options%lbfgsb%gradient_tolerance, ios)
        if (ios == 0) call write_r(unit, "lbfgsb_step_tolerance", self%options%lbfgsb%step_tolerance, ios)
        if (ios == 0) call write_r(unit, "lbfgsb_objective_tolerance", self%options%lbfgsb%objective_tolerance, ios)
        if (ios == 0) call write_r(unit, "lbfgsb_armijo_constant", self%options%lbfgsb%armijo_constant, ios)
        if (ios == 0) call write_r(unit, "lbfgsb_minimum_step", self%options%lbfgsb%minimum_step, ios)
        if (ios == 0) call write_r(unit, "lbfgsb_curvature_tolerance", self%options%lbfgsb%curvature_tolerance, ios)
        if (ios == 0) call write_l(unit, "callback_present", associated(self%options%callback), ios)
        if (ios == 0 .and. self%options%use_bounds) then
            call write_r_array(unit, "lower", self%options%lower, ios)
            if (ios == 0) call write_r_array(unit, "upper", self%options%upper, ios)
        end if

        if (ios == 0) call write_i(unit, "n_parameters", self%state%n_parameters, ios)
        if (ios == 0) call write_i(unit, "steps", self%state%steps, ios)
        if (ios == 0) call write_i(unit, "history_length", self%state%history_length, ios)
        if (ios == 0) call write_l(unit, "initialized", self%state%initialized, ios)
        if (ios == 0) call write_l(unit, "converged", self%state%converged, ios)
        if (ios == 0) call write_l(unit, "stopped_by_callback", self%state%stopped_by_callback, ios)
        if (ios == 0) call write_l(unit, "stopped_by_validation", self%state%stopped_by_validation, ios)
        if (ios == 0) call write_i(unit, "clipped_steps", self%state%clipped_steps, ios)
        if (ios == 0) call write_i(unit, "validation_history_length", &
            self%state%validation_history_length, ios)
        if (ios == 0) call write_i(unit, "validation_bad_steps", &
            self%state%validation_bad_steps, ios)
        if (ios == 0) call write_i(unit, "validation_best_step", &
            self%state%validation_best_step, ios)
        if (ios == 0) call write_r(unit, "initial_value", self%state%initial_value, ios)
        if (ios == 0) call write_r(unit, "final_value", self%state%final_value, ios)
        if (ios == 0) call write_r(unit, "best_value", self%state%best_value, ios)
        if (ios == 0) call write_r(unit, "gradient_norm", self%state%gradient_norm, ios)
        if (ios == 0) call write_r(unit, "last_step_norm", self%state%last_step_norm, ios)
        if (ios == 0) call write_r(unit, "last_learning_rate", &
            self%state%last_learning_rate, ios)
        if (ios == 0) call write_r(unit, "validation_value", &
            self%state%validation_value, ios)
        if (ios == 0) call write_r(unit, "best_validation_value", &
            self%state%best_validation_value, ios)
        if (ios == 0) call write_r_array(unit, "parameters", self%state%parameters, ios)
        if (ios == 0) call write_r_array(unit, "ema_parameters", self%state%ema_parameters, ios)
        if (ios == 0) call write_r_array(unit, "value_history", &
            self%state%value_history(:self%state%history_length), ios)
        if (ios == 0) call write_r_array(unit, "gradient_norm_history", &
            self%state%gradient_norm_history(:self%state%history_length), ios)
        if (ios == 0) call write_r_array(unit, "validation_history", &
            self%state%validation_history(:self%state%validation_history_length), ios)
        if (ios == 0) call write_r_array(unit, "validation_best_parameters", &
            self%state%validation_best_parameters, ios)
        if (ios == 0) call write_r_array(unit, "learning_rate_history", &
            self%state%learning_rate_history(:self%state%history_length), ios)

        if (ios == 0) then
            select case (self%options%optimizer)
            case (FORTML_TRAIN_SGD)
                call write_i(unit, "optimizer_step_count", self%sgd%step_count, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_velocity", self%sgd%velocity, ios)
            case (FORTML_TRAIN_ADAM)
                call write_i(unit, "optimizer_step_count", self%adam%step_count, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_first_moment", self%adam%first_moment, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_second_moment", self%adam%second_moment, ios)
            case (FORTML_TRAIN_ADAMW)
                call write_i(unit, "optimizer_step_count", self%adamw%step_count, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_first_moment", self%adamw%first_moment, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_second_moment", self%adamw%second_moment, ios)
            case (FORTML_TRAIN_ADAGRAD)
                call write_i(unit, "optimizer_step_count", self%adagrad%step_count, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_accumulated_square", self%adagrad%accumulated_square, ios)
            case (FORTML_TRAIN_RMSPROP)
                call write_i(unit, "optimizer_step_count", self%rmsprop%step_count, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_square_average", self%rmsprop%square_average, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_momentum_buffer", self%rmsprop%momentum_buffer, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_gradient_average", self%rmsprop%gradient_average, ios)
            case (FORTML_TRAIN_ADAFACTOR)
                call write_i(unit, "optimizer_step_count", self%adafactor%step_count, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_second_moment", self%adafactor%second_moment, ios)
            case (FORTML_TRAIN_LION)
                call write_i(unit, "optimizer_step_count", self%lion%step_count, ios)
                if (ios == 0) call write_r_array(unit, "optimizer_momentum", self%lion%momentum, ios)
            end select
        end if
        close_ios = 0
        close (unit, iostat=close_ios)
        if (ios /= 0 .or. close_ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint save: formatted write failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine trainer_save_checkpoint

    subroutine trainer_load_checkpoint(self, path, status)
        !! Load a snapshot into an initialized trainer with the same objective.
        !! Parsing is transactional: malformed, truncated, or extra records
        !! leave the destination untouched.
        class(trainer_t), intent(inout) :: self
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        type(trainer_options_t) :: options
        type(trainer_state_t) :: state
        character(len=256) :: line
        integer :: unit, ios, close_ios, schema, n, history_length, optimizer_step
        integer :: validation_history_length
        logical :: callback_present, validation_callback_present
        real(dp), allocatable :: vector(:), payload1(:), payload2(:), payload3(:)

        if (.not. self%ready .or. .not. self%state%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint load: destination is not initialized")
            return
        end if
        open (newunit=unit, file=path, status="old", action="read", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint load: cannot open source")
            return
        end if
        read (unit, "(A)", iostat=ios) line
        if (ios /= 0 .or. trim(line) /= FORTML_TRAINER_CHECKPOINT_MAGIC) goto 900
        call read_i(unit, "schema_version", schema, ios)
        if (ios /= 0 .or. schema /= FORTML_TRAINER_CHECKPOINT_SCHEMA_VERSION) goto 900
        call read_i(unit, "optimizer", options%optimizer, ios)
        if (ios == 0) call read_i(unit, "max_steps", options%max_steps, ios)
        if (ios == 0) call read_r(unit, "learning_rate", options%learning_rate, ios)
        if (ios == 0) call read_l(unit, "use_learning_rate_schedule", &
            options%use_learning_rate_schedule, ios)
        if (ios == 0) call read_i(unit, "schedule_kind", &
            options%learning_rate_schedule%kind, ios)
        if (ios == 0) call read_i(unit, "schedule_warmup_updates", &
            options%learning_rate_schedule%warmup_updates, ios)
        if (ios == 0) call read_i(unit, "schedule_total_updates", &
            options%learning_rate_schedule%total_updates, ios)
        if (ios == 0) call read_r(unit, "schedule_min_rate_fraction", &
            options%learning_rate_schedule%min_rate_fraction, ios)
        if (ios == 0) call read_r(unit, "schedule_decay_factor", &
            options%learning_rate_schedule%decay_factor, ios)
        if (ios == 0) call read_r(unit, "schedule_peak_rate_fraction", &
            options%learning_rate_schedule%peak_rate_fraction, ios)
        if (ios == 0) call read_r(unit, "schedule_final_rate_fraction", &
            options%learning_rate_schedule%final_rate_fraction, ios)
        if (ios == 0) call read_i(unit, "schedule_metric_mode", &
            options%learning_rate_schedule%metric_mode, ios)
        if (ios == 0) call read_i(unit, "schedule_patience_updates", &
            options%learning_rate_schedule%patience_updates, ios)
        if (ios == 0) call read_r(unit, "schedule_min_delta", &
            options%learning_rate_schedule%min_delta, ios)
        if (ios == 0) call read_r(unit, "schedule_plateau_factor", &
            options%learning_rate_schedule%plateau_factor, ios)
        if (ios == 0) call read_r(unit, "beta1", options%beta1, ios)
        if (ios == 0) call read_r(unit, "beta2", options%beta2, ios)
        if (ios == 0) call read_r(unit, "epsilon", options%epsilon, ios)
        if (ios == 0) call read_r(unit, "adafactor_decay", options%adafactor_decay, ios)
        if (ios == 0) call read_r(unit, "adafactor_clip_threshold", options%adafactor_clip_threshold, ios)
        if (ios == 0) call read_l(unit, "adafactor_relative_step", options%adafactor_relative_step, ios)
        if (ios == 0) call read_l(unit, "adafactor_scale_parameter", options%adafactor_scale_parameter, ios)
        if (ios == 0) call read_r(unit, "rmsprop_decay", options%rmsprop_decay, ios)
        if (ios == 0) call read_r(unit, "rmsprop_momentum", options%rmsprop_momentum, ios)
        if (ios == 0) call read_l(unit, "rmsprop_centered", options%rmsprop_centered, ios)
        if (ios == 0) call read_r(unit, "momentum", options%momentum, ios)
        if (ios == 0) call read_l(unit, "nesterov", options%nesterov, ios)
        if (ios == 0) call read_r(unit, "weight_decay", options%weight_decay, ios)
        if (ios == 0) call read_r(unit, "gradient_clip_norm", options%gradient_clip_norm, ios)
        if (ios == 0) call read_r(unit, "tolerance", options%tolerance, ios)
        if (ios == 0) call read_r(unit, "step_tolerance", options%step_tolerance, ios)
        if (ios == 0) call read_r(unit, "objective_tolerance", options%objective_tolerance, ios)
        if (ios == 0) call read_r(unit, "ema_decay", options%ema_decay, ios)
        if (ios == 0) call read_l(unit, "use_bounds", options%use_bounds, ios)
        if (ios == 0) call read_i(unit, "validation_patience", &
            options%validation_patience, ios)
        if (ios == 0) call read_r(unit, "validation_min_delta", &
            options%validation_min_delta, ios)
        if (ios == 0) call read_l(unit, "validation_higher_is_better", &
            options%validation_higher_is_better, ios)
        if (ios == 0) call read_l(unit, "validation_restore_best", &
            options%validation_restore_best, ios)
        if (ios == 0) call read_l(unit, "validation_callback_present", &
            validation_callback_present, ios)
        if (ios == 0) call read_i(unit, "lbfgsb_memory", options%lbfgsb%memory, ios)
        if (ios == 0) call read_i(unit, "lbfgsb_max_iterations", options%lbfgsb%max_iterations, ios)
        if (ios == 0) call read_i(unit, "lbfgsb_max_line_search", options%lbfgsb%max_line_search, ios)
        if (ios == 0) call read_r(unit, "lbfgsb_gradient_tolerance", options%lbfgsb%gradient_tolerance, ios)
        if (ios == 0) call read_r(unit, "lbfgsb_step_tolerance", options%lbfgsb%step_tolerance, ios)
        if (ios == 0) call read_r(unit, "lbfgsb_objective_tolerance", options%lbfgsb%objective_tolerance, ios)
        if (ios == 0) call read_r(unit, "lbfgsb_armijo_constant", options%lbfgsb%armijo_constant, ios)
        if (ios == 0) call read_r(unit, "lbfgsb_minimum_step", options%lbfgsb%minimum_step, ios)
        if (ios == 0) call read_r(unit, "lbfgsb_curvature_tolerance", options%lbfgsb%curvature_tolerance, ios)
        if (ios == 0) call read_l(unit, "callback_present", callback_present, ios)
        if (ios == 0 .and. options%use_bounds) then
            call read_r_array(unit, "lower_count", "lower_item", self%state%n_parameters, options%lower, ios)
            if (ios == 0) call read_r_array(unit, "upper_count", "upper_item", self%state%n_parameters, options%upper, ios)
        end if
        if (ios /= 0) goto 900

        call read_i(unit, "n_parameters", n, ios)
        if (ios == 0 .and. n /= self%state%n_parameters) ios = 1
        call read_i(unit, "steps", state%steps, ios)
        if (ios == 0) call read_i(unit, "history_length", history_length, ios)
        if (ios == 0) state%history_length = history_length
        if (ios == 0) call read_l(unit, "initialized", state%initialized, ios)
        if (ios == 0) call read_l(unit, "converged", state%converged, ios)
        if (ios == 0) call read_l(unit, "stopped_by_callback", state%stopped_by_callback, ios)
        if (ios == 0) call read_l(unit, "stopped_by_validation", &
            state%stopped_by_validation, ios)
        if (ios == 0) call read_i(unit, "clipped_steps", state%clipped_steps, ios)
        if (ios == 0) call read_i(unit, "validation_history_length", &
            validation_history_length, ios)
        if (ios == 0) state%validation_history_length = validation_history_length
        if (ios == 0) call read_i(unit, "validation_bad_steps", &
            state%validation_bad_steps, ios)
        if (ios == 0) call read_i(unit, "validation_best_step", &
            state%validation_best_step, ios)
        if (ios == 0) call read_r(unit, "initial_value", state%initial_value, ios)
        if (ios == 0) call read_r(unit, "final_value", state%final_value, ios)
        if (ios == 0) call read_r(unit, "best_value", state%best_value, ios)
        if (ios == 0) call read_r(unit, "gradient_norm", state%gradient_norm, ios)
        if (ios == 0) call read_r(unit, "last_step_norm", state%last_step_norm, ios)
        if (ios == 0) call read_r(unit, "last_learning_rate", &
            state%last_learning_rate, ios)
        if (ios == 0) call read_r(unit, "validation_value", state%validation_value, ios)
        if (ios == 0) call read_r(unit, "best_validation_value", &
            state%best_validation_value, ios)
        if (ios /= 0 .or. .not. state%initialized .or. history_length < 1 .or. &
            history_length > options%max_steps + 1 .or. state%steps < 0 .or. &
            validation_history_length < 0 .or. &
            validation_history_length > options%max_steps + 1 .or. &
            state%validation_bad_steps < 0 .or. state%validation_best_step < 0) goto 900
        state%n_parameters = n
        call read_r_array(unit, "parameters_count", "parameters_item", n, state%parameters, ios)
        if (ios == 0) call read_r_array(unit, "ema_parameters_count", "ema_parameters_item", n, state%ema_parameters, ios)
        if (ios == 0) call read_r_array(unit, "value_history_count", "value_history_item", history_length, vector, ios)
        if (ios == 0) then
            allocate(state%value_history(options%max_steps + 1))
            state%value_history = huge(1.0_dp)
            state%value_history(:history_length) = vector
            deallocate(vector)
        end if
        if (ios == 0) call read_r_array(unit, "gradient_norm_history_count", "gradient_norm_history_item", history_length, vector, ios)
        if (ios == 0) then
            allocate(state%gradient_norm_history(options%max_steps + 1))
            state%gradient_norm_history = huge(1.0_dp)
            state%gradient_norm_history(:history_length) = vector
            deallocate(vector)
        end if
        if (ios == 0) call read_r_array(unit, "validation_history_count", &
            "validation_history_item", validation_history_length, vector, ios)
        if (ios == 0) then
            allocate(state%validation_history(options%max_steps + 1))
            state%validation_history = huge(1.0_dp)
            if (validation_history_length > 0) then
                state%validation_history(:validation_history_length) = vector
            end if
            deallocate(vector)
        end if
        if (ios == 0) call read_r_array(unit, "validation_best_parameters_count", &
            "validation_best_parameters_item", n, state%validation_best_parameters, ios)
        if (ios == 0) call read_r_array(unit, "learning_rate_history_count", &
            "learning_rate_history_item", history_length, vector, ios)
        if (ios == 0) then
            allocate(state%learning_rate_history(options%max_steps + 1))
            state%learning_rate_history = 0.0_dp
            state%learning_rate_history(:history_length) = vector
            deallocate(vector)
        end if
        if (ios /= 0) goto 900
        select case (options%optimizer)
        case (FORTML_TRAIN_SGD)
            call read_i(unit, "optimizer_step_count", optimizer_step, ios)
            call read_r_array(unit, "optimizer_velocity_count", "optimizer_velocity_item", n, payload1, ios)
        case (FORTML_TRAIN_ADAM, FORTML_TRAIN_ADAMW)
            call read_i(unit, "optimizer_step_count", optimizer_step, ios)
            call read_r_array(unit, "optimizer_first_moment_count", "optimizer_first_moment_item", n, payload1, ios)
            call read_r_array(unit, "optimizer_second_moment_count", "optimizer_second_moment_item", n, payload2, ios)
        case (FORTML_TRAIN_ADAGRAD)
            call read_i(unit, "optimizer_step_count", optimizer_step, ios)
            call read_r_array(unit, "optimizer_accumulated_square_count", "optimizer_accumulated_square_item", n, payload1, ios)
        case (FORTML_TRAIN_RMSPROP)
            call read_i(unit, "optimizer_step_count", optimizer_step, ios)
            call read_r_array(unit, "optimizer_square_average_count", "optimizer_square_average_item", n, payload1, ios)
            call read_r_array(unit, "optimizer_momentum_buffer_count", "optimizer_momentum_buffer_item", n, payload2, ios)
            call read_r_array(unit, "optimizer_gradient_average_count", "optimizer_gradient_average_item", n, payload3, ios)
        case (FORTML_TRAIN_ADAFACTOR)
            call read_i(unit, "optimizer_step_count", optimizer_step, ios)
            call read_r_array(unit, "optimizer_second_moment_count", "optimizer_second_moment_item", n, payload1, ios)
        case (FORTML_TRAIN_LION)
            call read_i(unit, "optimizer_step_count", optimizer_step, ios)
            call read_r_array(unit, "optimizer_momentum_count", "optimizer_momentum_item", n, payload1, ios)
        case default
            ios = 1
        end select
        if (ios /= 0) goto 900
        read (unit, "(A)", iostat=ios) line
        if (ios == 0 .or. ios /= iostat_end) goto 900
        close_ios = 0
        close (unit, iostat=close_ios)
        if (close_ios /= 0) goto 900
        if (options%optimizer == FORTML_TRAIN_LBFGSB) goto 900
        if (callback_present .and. .not. associated(self%options%callback)) then
            ! A callback is process-local and cannot be reconstructed from text.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint load: callback must be attached by caller")
            return
        end if
        if (validation_callback_present .and. &
            .not. associated(self%options%validation_callback)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint load: validation callback must be attached by caller")
            return
        end if
        if (.not. validation_callback_present .and. &
            associated(self%options%validation_callback)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint load: validation callback presence changed")
            return
        end if
        if (associated(self%options%callback)) options%callback => self%options%callback
        if (associated(self%options%validation_callback)) then
            options%validation_callback => self%options%validation_callback
        end if
        call validate_options(options, n, status)
        if (status%code /= FORTNUM_OK) return
        call restore_optimizer(self, options, state, optimizer_step, payload1, payload2, payload3, status)
        if (status%code /= FORTNUM_OK) return
        self%options = options
        self%state = state
        call set_optimizer_learning_rate(self, self%state%last_learning_rate)
        self%ready = .true.
        call status_set(status, FORTNUM_OK, "")
        return

        900   continue
        close_ios = 0
        close (unit, iostat=close_ios)
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            "trainer checkpoint load: malformed, truncated, or extra record")
    end subroutine trainer_load_checkpoint

    subroutine validate_checkpoint(self, status)
        class(trainer_t), intent(in) :: self
        type(fortnum_status_t), intent(out) :: status
        integer :: n, h

        if (.not. self%ready .or. .not. self%state%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint save: trainer is not initialized")
            return
        end if
        n = self%state%n_parameters
        h = self%state%history_length
        if (self%options%optimizer == FORTML_TRAIN_LBFGSB) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint save: L-BFGS-B has no resumable state")
            return
        end if
        call validate_options(self%options, n, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. allocated(self%state%parameters) .or. &
            .not. allocated(self%state%ema_parameters) .or. &
            .not. allocated(self%state%value_history) .or. &
            .not. allocated(self%state%gradient_norm_history) .or. &
            .not. allocated(self%state%validation_history) .or. &
            .not. allocated(self%state%validation_best_parameters) .or. &
            .not. allocated(self%state%learning_rate_history)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint save: state is malformed or non-finite")
            return
        end if
        if (n < 1 .or. self%state%steps < 0 .or. h < 1 .or. &
            h > size(self%state%value_history) .or. h > self%options%max_steps + 1 .or. &
            self%state%validation_history_length < 0 .or. &
            self%state%validation_history_length > self%options%max_steps + 1 .or. &
            self%state%validation_bad_steps < 0 .or. self%state%validation_best_step < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint save: state is malformed or non-finite")
            return
        end if
        if (size(self%state%parameters) /= n .or. size(self%state%ema_parameters) /= n .or. &
            size(self%state%value_history) /= self%options%max_steps + 1 .or. &
            size(self%state%gradient_norm_history) /= self%options%max_steps + 1 .or. &
            size(self%state%validation_history) /= self%options%max_steps + 1 .or. &
            size(self%state%learning_rate_history) /= self%options%max_steps + 1 .or. &
            size(self%state%validation_best_parameters) /= n .or. &
            self%state%validation_history_length > size(self%state%validation_history) .or. &
            self%state%validation_best_step > self%state%steps) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint save: state shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(self%state%parameters)) .or. &
            any(.not. ieee_is_finite(self%state%ema_parameters)) .or. &
            any(.not. ieee_is_finite(self%state%value_history(:h))) .or. &
            any(.not. ieee_is_finite(self%state%gradient_norm_history(:h))) .or. &
            any(.not. ieee_is_finite(self%state%validation_best_parameters)) .or. &
            any(.not. ieee_is_finite(self%state%learning_rate_history(:h)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint save: state contains non-finite values")
            return
        end if
        if (self%state%validation_history_length > 0) then
            if (any(.not. ieee_is_finite(self%state%validation_history(: &
                self%state%validation_history_length)))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "trainer checkpoint save: validation history is non-finite")
                return
            end if
        end if
        if (.not. ieee_is_finite(self%state%initial_value) .or. &
            .not. ieee_is_finite(self%state%final_value) .or. &
            .not. ieee_is_finite(self%state%best_value) .or. &
            .not. ieee_is_finite(self%state%gradient_norm) .or. &
            .not. ieee_is_finite(self%state%last_step_norm) .or. &
            .not. ieee_is_finite(self%state%last_learning_rate) .or. &
            .not. ieee_is_finite(self%state%validation_value) .or. &
            .not. ieee_is_finite(self%state%best_validation_value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint save: scalar state is non-finite")
            return
        end if
        select case (self%options%optimizer)
        case (FORTML_TRAIN_SGD)
            if (.not. allocated(self%sgd%velocity)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: SGD state is invalid")
                return
            end if
            if (size(self%sgd%velocity) /= n) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: SGD state is invalid")
                return
            end if
        case (FORTML_TRAIN_ADAM)
            if (.not. allocated(self%adam%first_moment) .or. .not. allocated(self%adam%second_moment)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: Adam state is invalid")
                return
            end if
            if (size(self%adam%first_moment) /= n .or. size(self%adam%second_moment) /= n) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: Adam state is invalid")
                return
            end if
        case (FORTML_TRAIN_ADAMW)
            if (.not. allocated(self%adamw%first_moment) .or. .not. allocated(self%adamw%second_moment)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: AdamW state is invalid")
                return
            end if
            if (size(self%adamw%first_moment) /= n .or. size(self%adamw%second_moment) /= n) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: AdamW state is invalid")
                return
            end if
        case (FORTML_TRAIN_ADAGRAD)
            if (.not. allocated(self%adagrad%accumulated_square)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: Adagrad state is invalid")
                return
            end if
            if (size(self%adagrad%accumulated_square) /= n) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: Adagrad state is invalid")
                return
            end if
        case (FORTML_TRAIN_RMSPROP)
            if (.not. allocated(self%rmsprop%square_average) .or. &
                .not. allocated(self%rmsprop%momentum_buffer) .or. &
                .not. allocated(self%rmsprop%gradient_average) .or. &
                .false.) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: RMSprop state is invalid")
                return
            end if
            if (size(self%rmsprop%square_average) /= n .or. &
                size(self%rmsprop%momentum_buffer) /= n .or. &
                size(self%rmsprop%gradient_average) /= n) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: RMSprop state is invalid")
                return
            end if
        case (FORTML_TRAIN_ADAFACTOR)
            if (.not. allocated(self%adafactor%second_moment)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: Adafactor state is invalid")
                return
            end if
            if (size(self%adafactor%second_moment) /= n .or. &
                any(.not. ieee_is_finite(self%adafactor%second_moment)) .or. &
                any(self%adafactor%second_moment < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: Adafactor state is invalid")
                return
            end if
        case (FORTML_TRAIN_LION)
            if (.not. allocated(self%lion%momentum) .or. size(self%lion%momentum) /= n .or. &
                any(.not. ieee_is_finite(self%lion%momentum)) .or. self%lion%step_count < 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint save: Lion state is invalid")
                return
            end if
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_checkpoint

    subroutine restore_optimizer(self, options, state, step_count, payload1, payload2, &
            payload3, status)
        class(trainer_t), intent(inout) :: self
        type(trainer_options_t), intent(in) :: options
        type(trainer_state_t), intent(in) :: state
        integer, intent(in) :: step_count
        real(dp), allocatable, intent(in) :: payload1(:), payload2(:), payload3(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n

        n = state%n_parameters
        if (step_count < 0 .or. step_count > options%max_steps .or. &
            step_count /= state%steps .or. &
            .not. allocated(payload1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint load: optimizer state is invalid")
            return
        end if
        if (size(payload1) /= n .or. any(.not. ieee_is_finite(payload1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint load: optimizer state is invalid")
            return
        end if
        select case (options%optimizer)
        case (FORTML_TRAIN_SGD)
            call self%sgd%initialize(n, status, options%learning_rate, options%momentum, options%nesterov)
            if (status%code /= FORTNUM_OK) return
            self%sgd%velocity = payload1
            self%sgd%step_count = step_count
        case (FORTML_TRAIN_ADAM)
            if (.not. allocated(payload2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint load: Adam moments are invalid")
                return
            end if
            if (size(payload2) /= n .or. any(.not. ieee_is_finite(payload2))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint load: Adam moments are invalid")
                return
            end if
            call self%adam%initialize(n, status, options%learning_rate, options%beta1, options%beta2, options%epsilon)
            if (status%code /= FORTNUM_OK) return
            self%adam%first_moment = payload1
            self%adam%second_moment = payload2
            self%adam%step_count = step_count
        case (FORTML_TRAIN_ADAMW)
            if (.not. allocated(payload2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint load: AdamW moments are invalid")
                return
            end if
            if (size(payload2) /= n .or. any(.not. ieee_is_finite(payload2))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint load: AdamW moments are invalid")
                return
            end if
            call self%adamw%initialize(n, status, options%learning_rate, options%beta1, options%beta2, &
                options%epsilon, options%weight_decay)
            if (status%code /= FORTNUM_OK) return
            self%adamw%first_moment = payload1
            self%adamw%second_moment = payload2
            self%adamw%step_count = step_count
        case (FORTML_TRAIN_ADAGRAD)
            call self%adagrad%initialize(n, status, options%learning_rate, options%epsilon)
            if (status%code /= FORTNUM_OK) return
            if (any(payload1 < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint load: Adagrad accumulator is negative")
                return
            end if
            self%adagrad%accumulated_square = payload1
            self%adagrad%step_count = step_count
        case (FORTML_TRAIN_RMSPROP)
            if (.not. allocated(payload2) .or. .not. allocated(payload3)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint load: RMSprop state is invalid")
                return
            end if
            if (size(payload2) /= n .or. size(payload3) /= n .or. &
                any(.not. ieee_is_finite(payload2)) .or. any(.not. ieee_is_finite(payload3))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, "trainer checkpoint load: RMSprop state is invalid")
                return
            end if
            call self%rmsprop%initialize(n, status, options%learning_rate, options%rmsprop_decay, &
                options%epsilon, options%rmsprop_momentum, options%rmsprop_centered)
            if (status%code /= FORTNUM_OK) return
            self%rmsprop%square_average = payload1
            self%rmsprop%momentum_buffer = payload2
            self%rmsprop%gradient_average = payload3
            self%rmsprop%step_count = step_count
        case (FORTML_TRAIN_ADAFACTOR)
            if (any(payload1 < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "trainer checkpoint load: Adafactor second moment is negative")
                return
            end if
            call self%adafactor%initialize(n, status, options%learning_rate, &
                options%adafactor_decay, options%epsilon, options%adafactor_clip_threshold, &
                options%adafactor_relative_step, options%adafactor_scale_parameter)
            if (status%code /= FORTNUM_OK) return
            self%adafactor%second_moment = payload1
            self%adafactor%step_count = step_count
        case (FORTML_TRAIN_LION)
            call self%lion%initialize(n, status, options%learning_rate, options%beta1, &
                options%beta2, options%weight_decay)
            if (status%code /= FORTNUM_OK) return
            self%lion%momentum = payload1
            self%lion%step_count = step_count
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer checkpoint load: unsupported optimizer")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine restore_optimizer

    subroutine set_optimizer_learning_rate(self, learning_rate)
        !! Update only the scalar step scale; moments and counters remain
        !! owned by the optimizer and are never reinitialized by a schedule.
        class(trainer_t), intent(inout) :: self
        real(dp), intent(in) :: learning_rate

        select case (self%options%optimizer)
        case (FORTML_TRAIN_SGD)
            self%sgd%learning_rate = learning_rate
        case (FORTML_TRAIN_ADAM)
            self%adam%learning_rate = learning_rate
        case (FORTML_TRAIN_ADAMW)
            self%adamw%learning_rate = learning_rate
        case (FORTML_TRAIN_ADAGRAD)
            self%adagrad%learning_rate = learning_rate
        case (FORTML_TRAIN_RMSPROP)
            self%rmsprop%learning_rate = learning_rate
        case (FORTML_TRAIN_ADAFACTOR)
            self%adafactor%learning_rate = learning_rate
        case (FORTML_TRAIN_LION)
            self%lion%learning_rate = learning_rate
        end select
    end subroutine set_optimizer_learning_rate

    subroutine trainer_state_clear(self)
        class(trainer_state_t), intent(inout) :: self
        self%n_parameters = 0
        self%steps = 0
        self%history_length = 0
        self%initialized = .false.
        self%converged = .false.
        self%stopped_by_callback = .false.
        self%stopped_by_validation = .false.
        self%clipped_steps = 0
        self%validation_history_length = 0
        self%validation_bad_steps = 0
        self%validation_best_step = 0
        self%initial_value = huge(1.0_dp)
        self%final_value = huge(1.0_dp)
        self%best_value = huge(1.0_dp)
        self%gradient_norm = huge(1.0_dp)
        self%last_step_norm = huge(1.0_dp)
        self%last_learning_rate = 0.0_dp
        self%validation_value = huge(1.0_dp)
        self%best_validation_value = huge(1.0_dp)
        if (allocated(self%parameters)) deallocate(self%parameters)
        if (allocated(self%ema_parameters)) deallocate(self%ema_parameters)
        if (allocated(self%value_history)) deallocate(self%value_history)
        if (allocated(self%gradient_norm_history)) deallocate(self%gradient_norm_history)
        if (allocated(self%validation_history)) deallocate(self%validation_history)
        if (allocated(self%validation_best_parameters)) deallocate(self%validation_best_parameters)
        if (allocated(self%learning_rate_history)) deallocate(self%learning_rate_history)
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

    subroutine record_validation(self, stop, status)
        !! Evaluate the process-local validation metric and update its state.
        !! The callback receives the current parameter vector and owns the
        !! validation data.  A patience boundary is transactional: when it
        !! fires, optional best-parameter restoration happens before the
        !! trainer reports the stop.
        class(trainer_t), intent(inout) :: self
        logical, intent(out) :: stop
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value
        integer :: index

        stop = .false.
        if (.not. associated(self%options%validation_callback)) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        call self%options%validation_callback(self%state%steps, &
            self%state%parameters, value, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: validation callback returned a non-finite value")
            return
        end if
        self%state%validation_value = value
        index = min(self%state%steps + 1, size(self%state%validation_history))
        if (self%state%validation_history_length == 0) then
            self%state%validation_history(1) = value
            self%state%validation_history_length = 1
        else if (index > self%state%validation_history_length) then
            self%state%validation_history(index) = value
            self%state%validation_history_length = index
        else
            self%state%validation_history(index) = value
        end if

        if ((self%options%validation_higher_is_better .and. &
            value > self%state%best_validation_value + self%options%validation_min_delta) .or. &
            (.not. self%options%validation_higher_is_better .and. &
            value < self%state%best_validation_value - self%options%validation_min_delta)) then
            self%state%best_validation_value = value
            self%state%validation_best_step = self%state%steps
            self%state%validation_bad_steps = 0
            self%state%validation_best_parameters = self%state%parameters
        else
            self%state%validation_bad_steps = self%state%validation_bad_steps + 1
        end if
        if (self%options%validation_patience > 0) then
            if (self%state%validation_bad_steps >= self%options%validation_patience) then
                stop = .true.
                if (self%options%validation_restore_best) then
                    self%state%parameters = self%state%validation_best_parameters
                end if
            end if
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine record_validation

    subroutine validate_options(options, n_parameters, status)
        type(trainer_options_t), intent(in) :: options
        integer, intent(in) :: n_parameters
        type(fortnum_status_t), intent(out) :: status

        if (n_parameters < 1 .or. options%max_steps < 1 .or. &
            options%optimizer < FORTML_TRAIN_SGD .or. &
            options%optimizer > FORTML_TRAIN_LION .or. &
            .not. ieee_is_finite(options%learning_rate) .or. &
            options%learning_rate <= 0.0_dp .or. &
            .not. ieee_is_finite(options%beta1) .or. options%beta1 < 0.0_dp .or. &
            options%beta1 >= 1.0_dp .or. .not. ieee_is_finite(options%beta2) .or. &
            options%beta2 < 0.0_dp .or. options%beta2 >= 1.0_dp .or. &
            .not. ieee_is_finite(options%epsilon) .or. options%epsilon <= 0.0_dp .or. &
            .not. ieee_is_finite(options%adafactor_decay) .or. &
            options%adafactor_decay < 0.0_dp .or. options%adafactor_decay >= 1.0_dp .or. &
            .not. ieee_is_finite(options%adafactor_clip_threshold) .or. &
            options%adafactor_clip_threshold <= 0.0_dp .or. &
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
        if (options%validation_patience < 0 .or. &
            .not. ieee_is_finite(options%validation_min_delta) .or. &
            options%validation_min_delta < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: validation patience or minimum delta is invalid")
            return
        end if
        if (options%validation_restore_best .and. &
            .not. associated(options%validation_callback)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: validation restore-best requires a validation callback")
            return
        end if
        if (options%validation_patience > 0 .and. &
            .not. associated(options%validation_callback)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "trainer: validation patience requires a validation callback")
            return
        end if
        if (options%use_learning_rate_schedule) then
            if (.not. options%learning_rate_schedule%valid()) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "trainer: learning-rate schedule is invalid")
                return
            end if
            if (options%learning_rate_schedule%kind == MLP_SCHEDULE_PLATEAU) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "trainer: metric-aware plateau schedule requires a validation-aware adapter")
                return
            end if
            if (options%optimizer == FORTML_TRAIN_LBFGSB) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "trainer: learning-rate schedules are for streaming optimizers")
                return
            end if
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_options

    subroutine write_i(unit, key, value, ios)
        integer, intent(in) :: unit, value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write (unit, "(A,1X,I0)", iostat=ios) trim(key), value
    end subroutine write_i

    subroutine write_l(unit, key, value, ios)
        integer, intent(in) :: unit
        logical, intent(in) :: value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write (unit, "(A,1X,I0)", iostat=ios) trim(key), merge(1, 0, value)
    end subroutine write_l

    subroutine write_r(unit, key, value, ios)
        integer, intent(in) :: unit
        real(dp), intent(in) :: value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write (unit, "(A,1X,ES26.17E3)", iostat=ios) trim(key), value
    end subroutine write_r

    subroutine write_r_array(unit, key, values, ios)
        integer, intent(in) :: unit
        real(dp), intent(in) :: values(:)
        character(*), intent(in) :: key
        integer, intent(out) :: ios
        integer :: i
        character(len=96) :: count_key, item_key

        write (count_key, '(A,"_count")') trim(key)
        write (item_key, '(A,"_item")') trim(key)
        call write_i(unit, trim(count_key), size(values), ios)
        do i = 1, size(values)
            if (ios /= 0) return
            call write_r(unit, trim(item_key), values(i), ios)
        end do
    end subroutine write_r_array

    subroutine read_i(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        integer, intent(out) :: value
        integer, intent(out) :: ios
        character(len=96) :: key

        read (unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine read_i

    subroutine read_l(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        logical, intent(out) :: value
        integer, intent(out) :: ios
        character(len=96) :: key
        integer :: encoded

        encoded = 0
        read (unit, *, iostat=ios) key, encoded
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
        if (ios == 0 .and. encoded /= 0 .and. encoded /= 1) ios = 1
        value = encoded == 1
    end subroutine read_l

    subroutine read_r(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        real(dp), intent(out) :: value
        integer, intent(out) :: ios
        character(len=96) :: key

        read (unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine read_r

    subroutine read_r_array(unit, count_key, item_key, expected_count, values, ios)
        integer, intent(in) :: unit, expected_count
        character(*), intent(in) :: count_key, item_key
        real(dp), allocatable, intent(out) :: values(:)
        integer, intent(out) :: ios
        integer :: count, i, alloc_status

        if (allocated(values)) deallocate(values)
        call read_i(unit, count_key, count, ios)
        if (ios /= 0 .or. count < 0 .or. count /= expected_count) then
            ios = 1
            return
        end if
        allocate (values(count), stat=alloc_status)
        if (alloc_status /= 0) then
            ios = 1
            return
        end if
        do i = 1, count
            call read_r(unit, item_key, values(i), ios)
            if (ios /= 0) return
        end do
    end subroutine read_r_array

end module fortml_trainer
