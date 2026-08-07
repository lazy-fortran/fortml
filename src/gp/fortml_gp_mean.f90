module fortml_gp_mean
    !! Small, explicit mean-function adapters for exact Gaussian processes.
    !!
    !! The adapter is a template: a fitted GP copies its coefficient vector
    !! once per output column into the model's packed parameter vector.  This
    !! keeps the mean independent from the covariance while making constant
    !! and linear trends trainable by the same analytic L-BFGS-B products as
    !! kernel and noise parameters.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: GP_MEAN_ZERO = 0
    integer, parameter, public :: GP_MEAN_CONSTANT = 1
    integer, parameter, public :: GP_MEAN_LINEAR = 2

    type, public :: gp_mean_t
        integer :: kind = GP_MEAN_ZERO
        integer :: input_dim = 0
        !! Template coefficients.  Constant means store one intercept;
        !! linear means store an intercept followed by one slope per feature.
        real(dp), allocatable :: parameters(:)
    contains
        procedure, public :: parameter_count => gp_mean_parameter_count
        procedure, public :: basis => gp_mean_basis
        procedure, public :: validate => gp_mean_validate
    end type gp_mean_t

    public :: make_zero_mean
    public :: make_constant_mean
    public :: make_linear_mean

contains

    function make_zero_mean(input_dim, status) result(mean)
        integer, intent(in) :: input_dim
        type(fortnum_status_t), intent(out) :: status
        type(gp_mean_t) :: mean

        mean%kind = GP_MEAN_ZERO
        mean%input_dim = input_dim
        if (input_dim < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "zero GP mean: input dimension must be positive")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function make_zero_mean

    function make_constant_mean(input_dim, status, value) result(mean)
        integer, intent(in) :: input_dim
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: value
        type(gp_mean_t) :: mean

        mean%kind = GP_MEAN_CONSTANT
        mean%input_dim = input_dim
        if (input_dim < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "constant GP mean: input dimension must be positive")
            return
        end if
        allocate(mean%parameters(1))
        mean%parameters = 0.0_dp
        if (present(value)) mean%parameters(1) = value
        if (.not. ieee_is_finite(mean%parameters(1))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "constant GP mean: coefficient must be finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function make_constant_mean

    function make_linear_mean(input_dim, status, coefficients) result(mean)
        integer, intent(in) :: input_dim
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: coefficients(:)
        type(gp_mean_t) :: mean

        mean%kind = GP_MEAN_LINEAR
        mean%input_dim = input_dim
        if (input_dim < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear GP mean: input dimension must be positive")
            return
        end if
        allocate(mean%parameters(input_dim + 1))
        mean%parameters = 0.0_dp
        if (present(coefficients)) then
            if (size(coefficients) /= input_dim + 1 .or. &
                any(.not. ieee_is_finite(coefficients))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear GP mean: coefficient shape or values are invalid")
                return
            end if
            mean%parameters = coefficients
        end if
        call status_set(status, FORTNUM_OK, "")
    end function make_linear_mean

    integer function gp_mean_parameter_count(self) result(count)
        class(gp_mean_t), intent(in) :: self

        select case (self%kind)
        case (GP_MEAN_ZERO)
            count = 0
        case (GP_MEAN_CONSTANT)
            count = 1
        case (GP_MEAN_LINEAR)
            count = self%input_dim + 1
        case default
            count = -1
        end select
    end function gp_mean_parameter_count

    subroutine gp_mean_validate(self, input_dim, status)
        class(gp_mean_t), intent(in) :: self
        integer, intent(in) :: input_dim
        type(fortnum_status_t), intent(out) :: status
        integer :: count

        count = self%parameter_count()
        if (input_dim < 1 .or. self%input_dim /= input_dim .or. count < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP mean: input dimension or mean kind is invalid")
            return
        end if
        if (count == 0) then
            if (allocated(self%parameters)) then
                if (size(self%parameters) /= 0) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "GP mean: zero mean cannot carry coefficients")
                    return
                end if
            end if
        else
            if (.not. allocated(self%parameters)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP mean: coefficients are not initialized")
                return
            end if
            if (size(self%parameters) /= count .or. &
                any(.not. ieee_is_finite(self%parameters))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP mean: coefficient shape or values are invalid")
                return
            end if
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_mean_validate

    subroutine gp_mean_basis(self, x, basis, status)
        class(gp_mean_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: basis(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: n, p

        n = size(x, 1)
        p = self%parameter_count()
        if (size(x, 2) /= self%input_dim .or. any(shape(basis) /= [n, p])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP mean basis: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP mean basis: inputs must be finite")
            return
        end if
        basis = 0.0_dp
        select case (self%kind)
        case (GP_MEAN_CONSTANT)
            basis(:, 1) = 1.0_dp
        case (GP_MEAN_LINEAR)
            basis(:, 1) = 1.0_dp
            basis(:, 2:) = x
        case (GP_MEAN_ZERO)
            continue
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP mean basis: mean kind is invalid")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_mean_basis

end module fortml_gp_mean
