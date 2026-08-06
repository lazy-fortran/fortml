module fortml_inference_policy
    !! Structured inference policy: which solver a GP problem should use.
    !!
    !! The operator contracts below this module each cover a different
    !! structure - a dense Cholesky, a tensor-product grid, compact-support
    !! sparsity, a banded Markov precision, matrix-free Krylov, or an
    !! inducing-point bound. Choosing among them by hand at every call site is
    !! how the wrong one gets used on a problem it cannot serve, so the choice
    !! is made here from declared structure and size alone, and is reported
    !! with the reason.
    !!
    !! The policy never guesses at structure it was not told about: a caller
    !! declares what holds, and a declaration that does not fit the problem is
    !! refused rather than silently ignored.
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: INFERENCE_EXACT_CHOLESKY = 1
    integer, parameter, public :: INFERENCE_STRUCTURED_GRID = 2
    integer, parameter, public :: INFERENCE_SPARSE_COMPACT = 3
    integer, parameter, public :: INFERENCE_BANDED_PRECISION = 4
    integer, parameter, public :: INFERENCE_MATRIX_FREE_KRYLOV = 5
    integer, parameter, public :: INFERENCE_INDUCING_POINT = 6

    !! Above this sample count a dense Cholesky is no longer the default: its
    !! cost is cubic and its memory quadratic.
    integer, parameter, public :: DENSE_SAMPLE_LIMIT = 4096

    type, public :: inference_problem_t
        integer :: n_samples = 0
        integer :: n_features = 0
        integer :: n_outputs = 1
        !! Declared structure. Only one of the structural flags may hold.
        logical :: tensor_grid = .false.
        logical :: compact_support = .false.
        logical :: markov_precision = .false.
        !! Grid extents when `tensor_grid` holds; band width when
        !! `markov_precision` holds.
        integer, allocatable :: grid_dimensions(:)
        integer :: bandwidth = 0
        !! A caller that cannot afford an exact solve declares its budget.
        integer :: inducing_budget = 0
    end type inference_problem_t

    type, public :: inference_choice_t
        integer :: policy = 0
        character(len=:), allocatable :: reason
    end type inference_choice_t

    public :: select_inference_policy
    public :: inference_policy_name

contains

    subroutine select_inference_policy(problem, choice, status)
        type(inference_problem_t), intent(in) :: problem
        type(inference_choice_t), intent(out) :: choice
        type(fortnum_status_t), intent(out) :: status
        integer :: declared, total

        choice%policy = 0
        choice%reason = ""
        if (problem%n_samples < 1 .or. problem%n_features < 1 .or. &
            problem%n_outputs < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "inference policy: problem size is invalid")
            return
        end if

        declared = 0
        if (problem%tensor_grid) declared = declared + 1
        if (problem%compact_support) declared = declared + 1
        if (problem%markov_precision) declared = declared + 1
        if (declared > 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "inference policy: at most one structure may be declared")
            return
        end if

        if (problem%tensor_grid) then
            if (.not. allocated(problem%grid_dimensions)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "inference policy: a tensor grid needs its extents")
                return
            end if
            if (size(problem%grid_dimensions) < 1 .or. &
                any(problem%grid_dimensions < 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "inference policy: grid extents must be positive")
                return
            end if
            total = product(problem%grid_dimensions)
            if (total /= problem%n_samples) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "inference policy: grid extents do not multiply to the &
                    &sample count")
                return
            end if
            choice%policy = INFERENCE_STRUCTURED_GRID
            choice%reason = "separable tensor-grid covariance"
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        if (problem%markov_precision) then
            if (problem%bandwidth < 1 .or. &
                problem%bandwidth >= problem%n_samples) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "inference policy: the band must be positive and narrower &
                    &than the matrix")
                return
            end if
            choice%policy = INFERENCE_BANDED_PRECISION
            choice%reason = "banded Markov precision"
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        if (problem%compact_support) then
            choice%policy = INFERENCE_SPARSE_COMPACT
            choice%reason = "compact-support covariance"
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        ! No exploitable structure. The remaining choice is about cost.
        if (problem%inducing_budget > 0) then
            if (problem%inducing_budget >= problem%n_samples) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "inference policy: an inducing budget at or above the &
                    &sample count buys nothing")
                return
            end if
            choice%policy = INFERENCE_INDUCING_POINT
            choice%reason = "declared inducing budget below the sample count"
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        if (problem%n_samples*problem%n_outputs <= DENSE_SAMPLE_LIMIT) then
            choice%policy = INFERENCE_EXACT_CHOLESKY
            choice%reason = "dense solve fits the sample limit"
        else
            choice%policy = INFERENCE_MATRIX_FREE_KRYLOV
            choice%reason = "too large for a dense factorization"
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine select_inference_policy

    function inference_policy_name(policy) result(name)
        integer, intent(in) :: policy
        character(len=:), allocatable :: name

        select case (policy)
        case (INFERENCE_EXACT_CHOLESKY)
            name = "exact-cholesky"
        case (INFERENCE_STRUCTURED_GRID)
            name = "structured-grid"
        case (INFERENCE_SPARSE_COMPACT)
            name = "sparse-compact"
        case (INFERENCE_BANDED_PRECISION)
            name = "banded-precision"
        case (INFERENCE_MATRIX_FREE_KRYLOV)
            name = "matrix-free-krylov"
        case (INFERENCE_INDUCING_POINT)
            name = "inducing-point"
        case default
            name = "none"
        end select
    end function inference_policy_name

end module fortml_inference_policy
