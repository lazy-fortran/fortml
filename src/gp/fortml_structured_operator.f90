module fortml_structured_operator
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, FORTNUM_OK
    use fortnum_tensor_product, only: tensor_factor_t, &
        fortnum_tensor_operator_t => tensor_product_operator_t
    use fortml_linear_operator, only: linear_operator_t
    implicit none
    private

    type, extends(linear_operator_t), public :: structured_gp_operator_t
        type(fortnum_tensor_operator_t) :: product_operator
    contains
        procedure, public :: initialize => structured_gp_operator_initialize
        procedure, public :: matvec => structured_gp_operator_matvec
        procedure, public :: matmat => structured_gp_operator_matmat
        procedure, public :: diagonal => structured_gp_operator_diagonal
        procedure, public :: sample_count => structured_gp_operator_sample_count
    end type structured_gp_operator_t

    public :: tensor_factor_t

contains

    subroutine structured_gp_operator_initialize(self, factors, status)
        class(structured_gp_operator_t), intent(out) :: self
        type(tensor_factor_t), intent(in) :: factors(:)
        type(fortnum_status_t), intent(out) :: status

        call self%product_operator%initialize(factors, status)
    end subroutine structured_gp_operator_initialize

    subroutine structured_gp_operator_matvec(self, input, output)
        class(structured_gp_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)
        type(fortnum_status_t) :: status

        call self%product_operator%matvec(input, output, status)
        if (status%code /= FORTNUM_OK) then
            error stop "structured GP operator: invalid matrix-vector shape"
        end if
    end subroutine structured_gp_operator_matvec

    subroutine structured_gp_operator_matmat(self, input, output)
        class(structured_gp_operator_t), intent(in) :: self
        real(dp), intent(in) :: input(:, :)
        real(dp), intent(out) :: output(:, :)
        type(fortnum_status_t) :: status

        call self%product_operator%matmat(input, output, status)
        if (status%code /= FORTNUM_OK) then
            error stop "structured GP operator: invalid matrix-matrix shape"
        end if
    end subroutine structured_gp_operator_matmat

    function structured_gp_operator_diagonal(self) result(values)
        class(structured_gp_operator_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        type(fortnum_status_t) :: status

        allocate(values(self%sample_count()))
        call self%product_operator%diagonal(values, status)
        if (status%code /= FORTNUM_OK) then
            error stop "structured GP operator: diagonal is unavailable"
        end if
    end function structured_gp_operator_diagonal

    integer function structured_gp_operator_sample_count(self) result(count)
        class(structured_gp_operator_t), intent(in) :: self

        count = self%product_operator%element_count()
    end function structured_gp_operator_sample_count

end module fortml_structured_operator
