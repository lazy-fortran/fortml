module fortml_hyperparameter_search
    !! Deterministic hyperparameter search over FortOpt objectives.
    !!
    !! The search layer deliberately consumes `fortopt_objective::objective_t`.
    !! Every candidate therefore supplies the same value/gradient product used
    !! by L-BFGS-B and by model-specific hypergradient adapters. Grid search
    !! ignores the gradient only after asking the objective for the complete
    !! product, which keeps malformed or partial objectives visible. The
    !! generic orchestrator is CPU-only; CUDA requests are refused explicitly
    !! until a resident objective/search state contract exists.
    use, intrinsic :: iso_fortran_env, only: int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_rng, only: rng_t, rng_seed, rng_uniform
    use fortml_device, only: FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA, &
        fortml_device_t
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    type, public :: hyperparameter_search_result_t
        real(dp), allocatable :: best_parameters(:)
        real(dp) :: best_value = huge(1.0_dp)
        integer(int64) :: evaluations = 0_int64
        integer :: method = 0
        integer :: start_count = 0
        integer :: successful_starts = 0
        logical :: converged = .false.
    end type hyperparameter_search_result_t

    type, public :: hyperparameter_search_options_t
        integer :: max_evaluations = 1000000
        type(lbfgsb_options_t) :: lbfgsb
    end type hyperparameter_search_options_t

    integer, parameter, public :: HYPERPARAMETER_SEARCH_GRID = 1
    integer, parameter, public :: HYPERPARAMETER_SEARCH_LBFGSB = 2
    integer, parameter, public :: HYPERPARAMETER_SEARCH_RANDOM = 3

    public :: hyperparameter_grid_search
    public :: hyperparameter_lbfgsb_search
    public :: hyperparameter_lbfgsb_multistart_search
    public :: hyperparameter_random_search
    public :: hyperparameter_search_device_supported

contains

    subroutine hyperparameter_grid_search(objective, lower, upper, points, &
            result, status, options, device)
        type(objective_t), intent(in) :: objective
        real(dp), intent(in) :: lower(:), upper(:)
        integer, intent(in) :: points(:)
        type(hyperparameter_search_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(hyperparameter_search_options_t), intent(in), optional :: options
        type(fortml_device_t), intent(in), optional :: device

        real(dp), allocatable :: candidate(:), gradient(:)
        real(dp) :: value
        integer(int64) :: total, linear, remaining
        integer :: i, n, count
        type(hyperparameter_search_options_t) :: settings
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(hyperparameter_search_options_t) :: hyperparameter_search_options_t_default
        type(hyperparameter_search_result_t) :: hyperparameter_search_result_t_default

        result = hyperparameter_search_result_t_default
        result%method = HYPERPARAMETER_SEARCH_GRID
        settings = hyperparameter_search_options_t_default
        if (present(options)) settings = options
        call validate_search_inputs(objective, lower, upper, status, device)
        if (status%code /= FORTNUM_OK) return
        n = size(lower)
        if (size(points) /= n .or. any(points < 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter grid: points must match bounds and be positive")
            return
        end if
        if (settings%max_evaluations < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter grid: max_evaluations must be positive")
            return
        end if
        total = 1_int64
        do i = 1, n
            if (total > int(settings%max_evaluations, int64)/int(points(i), int64)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter grid: Cartesian product exceeds max_evaluations")
                return
            end if
            total = total*int(points(i), int64)
        end do
        allocate(candidate(n), gradient(n), result%best_parameters(n))
        result%best_parameters = lower
        do linear = 0_int64, total - 1_int64
            remaining = linear
            do i = 1, n
                count = int(mod(remaining, int(points(i), int64)))
                remaining = remaining/int(points(i), int64)
                if (points(i) == 1) then
                    candidate(i) = 0.5_dp*(lower(i) + upper(i))
                else
                    candidate(i) = lower(i) + (upper(i) - lower(i))* &
                        real(count, dp)/real(points(i) - 1, dp)
                end if
            end do
            call objective%value_gradient(candidate, value, gradient, status)
            result%evaluations = result%evaluations + 1_int64
            if (status%code /= FORTNUM_OK) return
            if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter grid: objective returned a nonfinite product")
                return
            end if
            if (result%evaluations == 1_int64 .or. value < result%best_value) then
                result%best_value = value
                result%best_parameters = candidate
            end if
        end do
        result%converged = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine hyperparameter_grid_search

    subroutine hyperparameter_random_search(objective, lower, upper, samples, &
            seed, result, status, options, device)
        type(objective_t), intent(in) :: objective
        real(dp), intent(in) :: lower(:), upper(:)
        integer, intent(in) :: samples
        integer(int64), intent(in) :: seed
        type(hyperparameter_search_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(hyperparameter_search_options_t), intent(in), optional :: options
        type(fortml_device_t), intent(in), optional :: device

        type(hyperparameter_search_options_t) :: settings
        type(rng_t) :: generator
        real(dp), allocatable :: candidate(:), gradient(:)
        real(dp) :: value, uniform
        integer :: i, j
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(hyperparameter_search_options_t) :: hyperparameter_search_options_t_default
        type(hyperparameter_search_result_t) :: hyperparameter_search_result_t_default

        result = hyperparameter_search_result_t_default
        result%method = HYPERPARAMETER_SEARCH_RANDOM
        settings = hyperparameter_search_options_t_default
        if (present(options)) settings = options
        call validate_search_inputs(objective, lower, upper, status, device)
        if (status%code /= FORTNUM_OK) return
        if (samples < 1 .or. samples > settings%max_evaluations) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter random: samples exceed the evaluation budget")
            return
        end if
        allocate(candidate(size(lower)), gradient(size(lower)), &
            result%best_parameters(size(lower)))
        call rng_seed(generator, seed, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, samples
            do j = 1, size(lower)
                call rng_uniform(generator, uniform)
                candidate(j) = lower(j) + (upper(j) - lower(j))*uniform
            end do
            call objective%value_gradient(candidate, value, gradient, status)
            result%evaluations = result%evaluations + 1_int64
            if (status%code /= FORTNUM_OK) return
            if (.not. ieee_is_finite(value) .or. &
                any(.not. ieee_is_finite(gradient))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter random: objective returned a nonfinite product")
                return
            end if
            if (i == 1 .or. value < result%best_value) then
                result%best_value = value
                result%best_parameters = candidate
            end if
        end do
        result%converged = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine hyperparameter_random_search

    subroutine hyperparameter_lbfgsb_search(objective, initial, lower, upper, &
            result, status, options, device)
        type(objective_t), intent(in) :: objective
        real(dp), intent(in) :: initial(:), lower(:), upper(:)
        type(hyperparameter_search_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(hyperparameter_search_options_t), intent(in), optional :: options
        type(fortml_device_t), intent(in), optional :: device

        type(lbfgsb_t) :: optimizer
        type(lbfgsb_result_t) :: optimizer_result
        type(hyperparameter_search_options_t) :: settings
        real(dp), allocatable :: x(:), gradient(:)
        real(dp) :: value
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(hyperparameter_search_options_t) :: hyperparameter_search_options_t_default
        type(hyperparameter_search_result_t) :: hyperparameter_search_result_t_default

        result = hyperparameter_search_result_t_default
        result%method = HYPERPARAMETER_SEARCH_LBFGSB
        settings = hyperparameter_search_options_t_default
        if (present(options)) settings = options
        call validate_search_inputs(objective, lower, upper, status, device)
        if (status%code /= FORTNUM_OK) return
        if (size(initial) /= size(lower) .or. any(.not. ieee_is_finite(initial))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter L-BFGS-B: initial point shape or values are invalid")
            return
        end if
        if (settings%max_evaluations < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter L-BFGS-B: max_evaluations must be positive")
            return
        end if
        allocate(x(size(initial)), result%best_parameters(size(initial)), &
            gradient(size(initial)))
        x = initial
        call optimizer%minimize(objective, x, lower, upper, settings%lbfgsb, &
            optimizer_result, status)
        if (status%code /= FORTNUM_OK) return
        call objective%value_gradient(x, value, gradient, status)
        if (status%code /= FORTNUM_OK) return
        if (.not. ieee_is_finite(value) .or. any(.not. ieee_is_finite(gradient))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter L-BFGS-B: final objective product is nonfinite")
            return
        end if
        result%best_parameters = x
        result%best_value = value
        result%evaluations = int(optimizer_result%line_search_evaluations + 1, int64)
        result%converged = optimizer_result%state%converged
        call status_set(status, FORTNUM_OK, "")
    end subroutine hyperparameter_lbfgsb_search

    subroutine hyperparameter_lbfgsb_multistart_search(objective, lower, upper, &
            starts, seed, result, status, options, device)
        !! Run deterministic bounded L-BFGS-B from seeded starts.
        !!
        !! Every start is generated in the closed box and is passed through
        !! the same value/gradient callback as the single-start entry point.
        !! The result keeps the best converged run, sums objective
        !! evaluations, and exposes start accounting so a failed or
        !! non-converged run cannot be mistaken for a successful search.
        type(objective_t), intent(in) :: objective
        real(dp), intent(in) :: lower(:), upper(:)
        integer, intent(in) :: starts
        integer(int64), intent(in) :: seed
        type(hyperparameter_search_result_t), intent(out) :: result
        type(fortnum_status_t), intent(out) :: status
        type(hyperparameter_search_options_t), intent(in), optional :: options
        type(fortml_device_t), intent(in), optional :: device

        type(hyperparameter_search_options_t) :: settings
        type(rng_t) :: generator
        type(hyperparameter_search_result_t) :: local_result
        real(dp), allocatable :: initial(:)
        real(dp) :: uniform
        integer :: i, j
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(hyperparameter_search_options_t) :: hyperparameter_search_options_t_default
        type(hyperparameter_search_result_t) :: hyperparameter_search_result_t_default

        result = hyperparameter_search_result_t_default
        result%method = HYPERPARAMETER_SEARCH_LBFGSB
        result%start_count = starts
        settings = hyperparameter_search_options_t_default
        if (present(options)) settings = options
        call validate_search_inputs(objective, lower, upper, status, device)
        if (status%code /= FORTNUM_OK) return
        if (starts < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter multistart: starts must be positive")
            return
        end if
        if (settings%max_evaluations < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter multistart: max_evaluations must be positive")
            return
        end if
        allocate(initial(size(lower)))
        call rng_seed(generator, seed, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, starts
            do j = 1, size(lower)
                call rng_uniform(generator, uniform)
                initial(j) = lower(j) + (upper(j) - lower(j))*uniform
            end do
            call hyperparameter_lbfgsb_search(objective, initial, lower, upper, &
                local_result, status, settings, device)
            if (status%code /= FORTNUM_OK) return
            result%evaluations = result%evaluations + local_result%evaluations
            if (local_result%converged) then
                result%successful_starts = result%successful_starts + 1
                if (result%successful_starts == 1 .or. &
                    local_result%best_value < result%best_value) then
                    result%best_value = local_result%best_value
                    if (allocated(result%best_parameters)) deallocate(result%best_parameters)
                    allocate(result%best_parameters, source=local_result%best_parameters)
                end if
            end if
        end do
        if (result%successful_starts < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter multistart: no start converged")
            return
        end if
        result%converged = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine hyperparameter_lbfgsb_multistart_search

    logical function hyperparameter_search_device_supported(device_kind) result(supported)
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function hyperparameter_search_device_supported

    subroutine validate_search_inputs(objective, lower, upper, status, device)
        type(objective_t), intent(in) :: objective
        real(dp), intent(in) :: lower(:), upper(:)
        type(fortnum_status_t), intent(out) :: status
        type(fortml_device_t), intent(in), optional :: device

        if (present(device)) then
            if (.not. device%selected .or. .not. device%available) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter search: selected device is unavailable")
                return
            end if
            if (device%kind == FORTML_DEVICE_CUDA) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "hyperparameter search: no resident CUDA objective/search state")
                return
            end if
            if (device%kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "hyperparameter search: device kind is invalid")
                return
            end if
        end if
        if (objective%n_parameters < 1 .or. size(lower) /= objective%n_parameters .or. &
            size(upper) /= size(lower) .or. any(.not. ieee_is_finite(lower)) .or. &
            any(.not. ieee_is_finite(upper)) .or. any(lower > upper)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "hyperparameter search: objective or bounds are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_search_inputs

end module fortml_hyperparameter_search
