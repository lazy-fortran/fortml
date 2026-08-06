module fortml_banded_precision
    !! Gaussian Markov random field path: a GP defined by a banded precision
    !! matrix rather than by a dense covariance.
    !!
    !! A Markov process on a grid has a sparse precision even when its
    !! covariance is dense - an Ornstein-Uhlenbeck process has a tridiagonal
    !! one - so working in the precision costs `O(n b^2)` for a factorization
    !! and `O(n b)` for a solve, where `b` is the bandwidth. The operator this
    !! type exposes to Krylov code is the precision itself; the covariance is
    !! reached through `solve`, and the log determinant of the covariance is
    !! the negated log determinant of the precision.
    !!
    !! Storage is the lower band: `band(k, i)` holds the entry `k` rows below
    !! the diagonal in column `i`, the LAPACK `dpbtrf` layout without the
    !! LAPACK dependency, since the factorization here is a short explicit
    !! recurrence.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    use fortml_linear_operator, only: linear_operator_t
    implicit none
    private

    type, extends(linear_operator_t), public :: banded_precision_operator_t
        !! `band(0:bandwidth, n)`, lower triangle by diagonal offset.
        real(dp), allocatable :: band(:, :)
        real(dp), allocatable :: factor(:, :)
        integer :: n = 0
        integer :: bandwidth = 0
        logical :: factorized = .false.
    contains
        procedure, public :: initialize => banded_initialize
        procedure, public :: matvec => banded_matvec
        procedure, public :: matmat => banded_matmat
        procedure, public :: diagonal => banded_diagonal
        procedure, public :: sample_count => banded_sample_count
        procedure, public :: factorize => banded_factorize
        procedure, public :: solve => banded_solve
        procedure, public :: log_determinant => banded_log_determinant
        procedure, public :: covariance_log_determinant => &
            banded_covariance_log_determinant
    end type banded_precision_operator_t

    public :: make_ornstein_uhlenbeck_precision

contains

    subroutine banded_initialize(self, band, status)
        class(banded_precision_operator_t), intent(out) :: self
        real(dp), intent(in) :: band(0:, :)
        type(fortnum_status_t), intent(out) :: status

        if (size(band, 2) < 1 .or. size(band, 1) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "banded precision: band shape is invalid")
            return
        end if
        if (size(band, 1) - 1 >= size(band, 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "banded precision: bandwidth exceeds the matrix size")
            return
        end if
        self%n = size(band, 2)
        self%bandwidth = size(band, 1) - 1
        allocate(self%band(0:self%bandwidth, self%n))
        self%band = band
        self%factorized = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine banded_initialize

    subroutine banded_matvec(self, input, output)
        class(banded_precision_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)
        integer :: i, k

        output = 0.0_dp
        if (.not. allocated(self%band)) return
        if (size(input) /= self%n .or. size(output) /= self%n) return
        do i = 1, self%n
            output(i) = output(i) + self%band(0, i)*input(i)
            do k = 1, min(self%bandwidth, self%n - i)
                output(i + k) = output(i + k) + self%band(k, i)*input(i)
                output(i) = output(i) + self%band(k, i)*input(i + k)
            end do
        end do
    end subroutine banded_matvec

    subroutine banded_matmat(self, input, output)
        class(banded_precision_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        integer :: column

        output = 0.0_dp
        if (size(input, 2) /= size(output, 2)) return
        do column = 1, size(input, 2)
            call banded_matvec(self, input(:, column), output(:, column))
        end do
    end subroutine banded_matmat

    function banded_diagonal(self) result(values)
        class(banded_precision_operator_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        allocate(values(max(self%n, 0)))
        if (self%n < 1) return
        values = self%band(0, :)
    end function banded_diagonal

    integer function banded_sample_count(self) result(count)
        class(banded_precision_operator_t), intent(in) :: self

        count = self%n
    end function banded_sample_count

    subroutine banded_factorize(self, status)
        !! Banded Cholesky `Q = L L^T`, storing `L` in the same band layout.
        class(banded_precision_operator_t), intent(inout) :: self
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k, offset
        real(dp) :: total

        if (.not. allocated(self%band)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "banded precision: factorize before initialize")
            return
        end if
        if (allocated(self%factor)) deallocate(self%factor)
        allocate(self%factor(0:self%bandwidth, self%n))
        self%factor = 0.0_dp
        self%factorized = .false.

        do j = 1, self%n
            total = self%band(0, j)
            do k = 1, min(self%bandwidth, j - 1)
                total = total - self%factor(k, j - k)*self%factor(k, j - k)
            end do
            if (total <= 0.0_dp) then
                deallocate(self%factor)
                call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                    "banded precision: the precision is not positive definite")
                return
            end if
            self%factor(0, j) = sqrt(total)
            do offset = 1, min(self%bandwidth, self%n - j)
                i = j + offset
                total = self%band(offset, j)
                do k = 1, min(self%bandwidth, j - 1)
                    if (i - (j - k) > self%bandwidth) cycle
                    total = total - self%factor(i - (j - k), j - k)* &
                        self%factor(k, j - k)
                end do
                self%factor(offset, j) = total/self%factor(0, j)
            end do
        end do
        self%factorized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine banded_factorize

    subroutine banded_solve(self, right_hand_side, solution, status)
        !! Apply the covariance: solve `Q x = b`.
        class(banded_precision_operator_t), intent(inout) :: self
        real(dp), intent(in) :: right_hand_side(:)
        real(dp), intent(out) :: solution(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, k
        real(dp) :: total

        solution = 0.0_dp
        if (size(right_hand_side) /= self%n .or. size(solution) /= self%n) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "banded precision: solve shape is invalid")
            return
        end if
        if (.not. self%factorized) then
            call self%factorize(status)
            if (status%code /= FORTNUM_OK) return
        end if

        do i = 1, self%n
            total = right_hand_side(i)
            do k = 1, min(self%bandwidth, i - 1)
                total = total - self%factor(k, i - k)*solution(i - k)
            end do
            solution(i) = total/self%factor(0, i)
        end do
        do i = self%n, 1, -1
            total = solution(i)
            do k = 1, min(self%bandwidth, self%n - i)
                total = total - self%factor(k, i)*solution(i + k)
            end do
            solution(i) = total/self%factor(0, i)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine banded_solve

    subroutine banded_log_determinant(self, value, status)
        !! `log det Q`, from the banded Cholesky diagonal.
        class(banded_precision_operator_t), intent(inout) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        value = 0.0_dp
        if (.not. self%factorized) then
            call self%factorize(status)
            if (status%code /= FORTNUM_OK) return
        end if
        do i = 1, self%n
            value = value + 2.0_dp*log(self%factor(0, i))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine banded_log_determinant

    subroutine banded_covariance_log_determinant(self, value, status)
        !! `log det Q^{-1}`, which is what a marginal likelihood needs.
        class(banded_precision_operator_t), intent(inout) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        call banded_log_determinant(self, value, status)
        if (status%code /= FORTNUM_OK) return
        value = -value
    end subroutine banded_covariance_log_determinant

    subroutine make_ornstein_uhlenbeck_precision( &
            n, spacing, variance, lengthscale, band, status)
        !! Tridiagonal precision of an Ornstein-Uhlenbeck process on a uniform
        !! grid, the exact inverse of the Matern-1/2 covariance
        !! `variance*exp(-|s - t|/lengthscale)`.
        integer, intent(in) :: n
        real(dp), intent(in) :: spacing, variance, lengthscale
        real(dp), allocatable, intent(out) :: band(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: rho, scale
        integer :: i

        if (n < 2 .or. spacing <= 0.0_dp .or. variance <= 0.0_dp .or. &
            lengthscale <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Ornstein-Uhlenbeck precision: parameters must be positive")
            return
        end if
        rho = exp(-spacing/lengthscale)
        scale = 1.0_dp/(variance*(1.0_dp - rho*rho))
        allocate(band(0:1, n))
        band = 0.0_dp
        band(0, 1) = scale
        do i = 2, n - 1
            band(0, i) = scale*(1.0_dp + rho*rho)
        end do
        band(0, n) = scale
        do i = 1, n - 1
            band(1, i) = -scale*rho
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine make_ornstein_uhlenbeck_precision

end module fortml_banded_precision
