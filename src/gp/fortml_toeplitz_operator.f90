module fortml_toeplitz_operator
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t
    use fortnum_toeplitz, only: fortnum_toeplitz_operator_t => &
        toeplitz_operator_t
    use fortml_linear_operator, only: linear_operator_t
    implicit none
    private

    type, extends(linear_operator_t), public :: toeplitz_gp_operator_t
        type(fortnum_toeplitz_operator_t) :: product_operator
    contains
        procedure, public :: initialize => toeplitz_gp_operator_initialize
        procedure, public :: matvec => toeplitz_gp_operator_matvec
        procedure, public :: matmat => toeplitz_gp_operator_matmat
        procedure, public :: diagonal => toeplitz_gp_operator_diagonal
        procedure, public :: sample_count => toeplitz_gp_operator_sample_count
    end type toeplitz_gp_operator_t

contains

    subroutine toeplitz_gp_operator_initialize(self, column, status, row)
        class(toeplitz_gp_operator_t), intent(out) :: self
        real(dp), intent(in) :: column(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: row(:)

        if (present(row)) then
            call self%product_operator%initialize(column, status, row)
        else
            call self%product_operator%initialize(column, status)
        end if
    end subroutine toeplitz_gp_operator_initialize

    subroutine toeplitz_gp_operator_matvec(self, input, output)
        class(toeplitz_gp_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)

        call self%product_operator%matvec(input, output)
    end subroutine toeplitz_gp_operator_matvec

    subroutine toeplitz_gp_operator_matmat(self, input, output)
        class(toeplitz_gp_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)

        call self%product_operator%matmat(input, output)
    end subroutine toeplitz_gp_operator_matmat

    function toeplitz_gp_operator_diagonal(self) result(values)
        class(toeplitz_gp_operator_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        values = self%product_operator%diagonal()
    end function toeplitz_gp_operator_diagonal

    integer function toeplitz_gp_operator_sample_count(self) result(count)
        class(toeplitz_gp_operator_t), intent(in) :: self

        count = self%product_operator%element_count()
    end function toeplitz_gp_operator_sample_count

end module fortml_toeplitz_operator
