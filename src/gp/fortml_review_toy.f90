module fortml_review_toy
    !! The one-dimensional fixture of the scalable-GP review.
    !!
    !! Liu, Ong, Shen and Cai, "When Gaussian Process Meets Big Data"
    !! (IEEE TNNLS 31(11):4405-4423, 2020) illustrate every approximation on
    !! one problem, described in the captions of their Figs. 4 and 5:
    !!
    !!     y(x) = sinc(x) + eps,   eps ~ N(0, 0.04),   120 training points
    !!
    !! over the plotted range `[-7, 7]`, with the inducing points initialized
    !! on a uniform grid. This module owns that fixture so every method in the
    !! repository is compared on exactly the paper's problem rather than on a
    !! near miss, and so the benchmark repository and the test suite share one
    !! definition.
    !!
    !! The noise draw is seeded through `fortnum_rng`, so the fixture is a
    !! reproducible function of its seed.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_rng, only: rng_t, rng_seed, rng_normal
    use, intrinsic :: iso_fortran_env, only: int64
    implicit none
    private

    integer, parameter, public :: REVIEW_TOY_SAMPLES = 120
    real(dp), parameter, public :: REVIEW_TOY_NOISE_VARIANCE = 0.04_dp
    real(dp), parameter, public :: REVIEW_TOY_LOWER = -7.0_dp
    real(dp), parameter, public :: REVIEW_TOY_UPPER = 7.0_dp

    public :: review_toy_data
    public :: review_toy_grid
    public :: review_toy_truth

contains

    real(dp) elemental function review_toy_truth(x) result(value)
        !! The normalized sinc the paper plots, `sin(pi x)/(pi x)`, with the
        !! removable singularity taken at its limit.
        real(dp), intent(in) :: x
        real(dp) :: argument
        real(dp), parameter :: PI = 3.141592653589793238462643_dp

        argument = PI*x
        if (abs(argument) < 1.0e-12_dp) then
            value = 1.0_dp
        else
            value = sin(argument)/argument
        end if
    end function review_toy_truth

    subroutine review_toy_data(n_samples, seed, inputs, targets, status)
        !! Training inputs spread uniformly over the plotted range, with
        !! seeded Gaussian noise of the paper's variance.
        integer, intent(in) :: n_samples, seed
        real(dp), allocatable, intent(out) :: inputs(:, :), targets(:)
        type(fortnum_status_t), intent(out) :: status
        type(rng_t) :: generator
        real(dp) :: draw, spacing
        integer :: i

        if (n_samples < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "review toy: at least two training points are needed")
            return
        end if
        call rng_seed(generator, int(seed, int64), status)
        if (status%code /= FORTNUM_OK) return

        allocate(inputs(n_samples, 1), targets(n_samples))
        spacing = (REVIEW_TOY_UPPER - REVIEW_TOY_LOWER)/real(n_samples - 1, dp)
        do i = 1, n_samples
            inputs(i, 1) = REVIEW_TOY_LOWER + real(i - 1, dp)*spacing
            call rng_normal(generator, draw)
            targets(i) = review_toy_truth(inputs(i, 1)) &
                + sqrt(REVIEW_TOY_NOISE_VARIANCE)*draw
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine review_toy_data

    subroutine review_toy_grid(n_points, lower, upper, grid, status)
        !! A uniform grid, used both for the initial inducing locations the
        !! paper marks with triangles and for the test locations it plots.
        integer, intent(in) :: n_points
        real(dp), intent(in) :: lower, upper
        real(dp), allocatable, intent(out) :: grid(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: spacing
        integer :: i

        if (n_points < 2 .or. upper <= lower) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "review toy: the grid needs two points and a positive range")
            return
        end if
        allocate(grid(n_points, 1))
        spacing = (upper - lower)/real(n_points - 1, dp)
        do i = 1, n_points
            grid(i, 1) = lower + real(i - 1, dp)*spacing
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine review_toy_grid

end module fortml_review_toy
