module fortml_multilevel_grid
    !! Multilevel embedding of tensor-product grids.
    !!
    !! A structured GP lives on a grid whose points are a tensor product of
    !! one-dimensional grids. Coarsening every dimension by two gives a
    !! hierarchy in which each level embeds in the next: a coarse grid value is
    !! carried to the fine grid by linear interpolation, and a fine grid
    !! residual is carried back by the transpose of that interpolation. Keeping
    !! the two as exact transposes is what makes a two-level correction a
    !! symmetric operator, so it can precondition a conjugate-gradient solve
    !! without breaking the recurrence.
    !!
    !! The prolongation is separable: interpolating along one dimension at a
    !! time costs `sum_i n_i` passes rather than forming the full matrix. Level
    !! 1 is the finest.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    type :: level_shape_t
        integer, allocatable :: dimensions(:)
        integer :: total_size = 0
    end type level_shape_t

    type, public :: multilevel_grid_t
        type(level_shape_t), allocatable :: levels(:)
        integer :: n_levels = 0
        integer :: n_dimensions = 0
    contains
        procedure, public :: initialize => multilevel_initialize
        procedure, public :: level_count => multilevel_level_count
        procedure, public :: level_size => multilevel_level_size
        procedure, public :: level_dimensions => multilevel_level_dimensions
        procedure, public :: prolong => multilevel_prolong
        procedure, public :: restrict => multilevel_restrict
    end type multilevel_grid_t

contains

    subroutine multilevel_initialize(self, dimensions, n_levels, status)
        !! Build the hierarchy by halving each dimension, stopping early if a
        !! further level would leave fewer than two points in any dimension.
        class(multilevel_grid_t), intent(out) :: self
        integer, intent(in) :: dimensions(:)
        integer, intent(in) :: n_levels
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: current(:)
        integer :: level, mode, usable
        integer :: total

        if (size(dimensions) < 1 .or. n_levels < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilevel grid: dimension or level count is invalid")
            return
        end if
        if (any(dimensions < 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilevel grid: every dimension needs at least two points")
            return
        end if

        self%n_dimensions = size(dimensions)
        allocate(current(size(dimensions)))
        current = dimensions
        usable = 1
        do level = 2, n_levels
            if (any(coarse_size(current) < 2)) exit
            current = coarse_size(current)
            usable = usable + 1
        end do

        allocate(self%levels(usable))
        self%n_levels = usable
        current = dimensions
        do level = 1, usable
            allocate(self%levels(level)%dimensions(size(dimensions)))
            self%levels(level)%dimensions = current
            total = 1
            do mode = 1, size(current)
                total = total*current(mode)
            end do
            self%levels(level)%total_size = total
            if (level < usable) current = coarse_size(current)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilevel_initialize

    pure function coarse_size(dimensions) result(coarse)
        !! A grid of `n` points coarsens to `(n + 1)/2`, so the two endpoints
        !! stay grid points at every level.
        integer, intent(in) :: dimensions(:)
        integer :: coarse(size(dimensions))

        coarse = (dimensions + 1)/2
    end function coarse_size

    integer function multilevel_level_count(self) result(count)
        class(multilevel_grid_t), intent(in) :: self

        count = self%n_levels
    end function multilevel_level_count

    integer function multilevel_level_size(self, level) result(count)
        class(multilevel_grid_t), intent(in) :: self
        integer, intent(in) :: level

        count = 0
        if (level < 1 .or. level > self%n_levels) return
        count = self%levels(level)%total_size
    end function multilevel_level_size

    function multilevel_level_dimensions(self, level) result(dimensions)
        class(multilevel_grid_t), intent(in) :: self
        integer, intent(in) :: level
        integer, allocatable :: dimensions(:)

        if (level < 1 .or. level > self%n_levels) then
            allocate(dimensions(0))
            return
        end if
        allocate(dimensions, source=self%levels(level)%dimensions)
    end function multilevel_level_dimensions

    subroutine multilevel_prolong(self, level, coarse, fine, status)
        !! Carry a value from `level + 1` to `level` by separable linear
        !! interpolation.
        class(multilevel_grid_t), intent(in) :: self
        integer, intent(in) :: level
        real(dp), intent(in) :: coarse(:)
        real(dp), intent(out) :: fine(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. valid_transfer(self, level, coarse, fine, status)) return
        call transfer_all_modes(self, level, coarse, fine, .true.)
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilevel_prolong

    subroutine multilevel_restrict(self, level, fine, coarse, status)
        !! The exact transpose of `prolong`, so that
        !! `<prolong(c), f> = <c, restrict(f)>` holds to round-off.
        class(multilevel_grid_t), intent(in) :: self
        integer, intent(in) :: level
        real(dp), intent(in) :: fine(:)
        real(dp), intent(out) :: coarse(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. valid_transfer(self, level, coarse, fine, status)) return
        call transfer_all_modes(self, level, fine, coarse, .false.)
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilevel_restrict

    logical function valid_transfer(self, level, coarse, fine, status) &
            result(valid)
        class(multilevel_grid_t), intent(in) :: self
        integer, intent(in) :: level
        real(dp), intent(in) :: coarse(:), fine(:)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (level < 1 .or. level >= self%n_levels) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilevel grid: no coarser level below this one")
            return
        end if
        if (size(fine) /= self%levels(level)%total_size .or. &
            size(coarse) /= self%levels(level + 1)%total_size) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilevel grid: transfer shape does not match the levels")
            return
        end if
        valid = .true.
    end function valid_transfer

    subroutine transfer_all_modes(self, level, input, output, prolonging)
        !! Apply the one-dimensional transfer along each mode in turn. Mode 1
        !! is the fastest varying index, matching `fortnum_tensor_product`.
        class(multilevel_grid_t), intent(in) :: self
        integer, intent(in) :: level
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)
        logical, intent(in) :: prolonging
        real(dp), allocatable :: current(:), next(:)
        integer, allocatable :: shape_now(:)
        integer :: mode, inner, outer, source_size, target_size, total

        allocate(current, source=input)
        allocate(shape_now(self%n_dimensions))
        if (prolonging) then
            shape_now = self%levels(level + 1)%dimensions
        else
            shape_now = self%levels(level)%dimensions
        end if

        do mode = 1, self%n_dimensions
            source_size = shape_now(mode)
            if (prolonging) then
                target_size = self%levels(level)%dimensions(mode)
            else
                target_size = self%levels(level + 1)%dimensions(mode)
            end if
            inner = 1
            do outer = 1, mode - 1
                inner = inner*shape_now(outer)
            end do
            outer = 1
            do total = mode + 1, self%n_dimensions
                outer = outer*shape_now(total)
            end do
            allocate(next(inner*target_size*outer))
            next = 0.0_dp
            call transfer_one_mode(current, next, inner, source_size, &
                target_size, outer, prolonging)
            deallocate(current)
            call move_alloc(next, current)
            shape_now(mode) = target_size
        end do
        output = current
    end subroutine transfer_all_modes

    subroutine transfer_one_mode(input, output, inner, source_size, &
            target_size, outer, prolonging)
        !! Linear interpolation from `source_size` to `target_size` points
        !! along one mode, or its transpose. A coarse point `j` sits on fine
        !! point `2j - 1`; the fine points between two coarse ones take the
        !! average of their neighbours.
        real(dp), intent(in) :: input(:)
        real(dp), intent(inout) :: output(:)
        integer, intent(in) :: inner, source_size, target_size, outer
        logical, intent(in) :: prolonging
        integer :: i, k, fine_index, coarse_index, left, right
        integer :: fine_size, coarse_size_local

        if (prolonging) then
            fine_size = target_size
            coarse_size_local = source_size
        else
            fine_size = source_size
            coarse_size_local = target_size
        end if

        do k = 1, outer
            do i = 1, inner
                do fine_index = 1, fine_size
                    if (mod(fine_index, 2) == 1) then
                        coarse_index = (fine_index + 1)/2
                        if (prolonging) then
                            output(flat(i, fine_index, k, inner, target_size)) = &
                                output(flat(i, fine_index, k, inner, target_size)) &
                                + input(flat(i, coarse_index, k, inner, &
                                source_size))
                        else
                            output(flat(i, coarse_index, k, inner, target_size)) = &
                                output(flat(i, coarse_index, k, inner, &
                                target_size)) &
                                + input(flat(i, fine_index, k, inner, source_size))
                        end if
                    else
                        left = fine_index/2
                        right = min(left + 1, coarse_size_local)
                        if (prolonging) then
                            output(flat(i, fine_index, k, inner, target_size)) = &
                                output(flat(i, fine_index, k, inner, target_size)) &
                                + 0.5_dp*(input(flat(i, left, k, inner, &
                                source_size)) + input(flat(i, right, k, inner, &
                                source_size)))
                        else
                            output(flat(i, left, k, inner, target_size)) = &
                                output(flat(i, left, k, inner, target_size)) &
                                + 0.5_dp*input(flat(i, fine_index, k, inner, &
                                source_size))
                            output(flat(i, right, k, inner, target_size)) = &
                                output(flat(i, right, k, inner, target_size)) &
                                + 0.5_dp*input(flat(i, fine_index, k, inner, &
                                source_size))
                        end if
                    end if
                end do
            end do
        end do
    end subroutine transfer_one_mode

    pure integer function flat(inner_index, mode_index, outer_index, inner, &
            mode_size) result(index)
        integer, intent(in) :: inner_index, mode_index, outer_index, inner
        integer, intent(in) :: mode_size

        index = inner_index + inner*(mode_index - 1) &
            + inner*mode_size*(outer_index - 1)
    end function flat

end module fortml_multilevel_grid
