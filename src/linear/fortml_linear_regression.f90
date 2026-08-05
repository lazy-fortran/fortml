module fortml_linear_regression
    use fortnum_kinds, only: dp
    use fortnum_linalg, only: dense_solve
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    implicit none
    private

    type, public :: linear_regression_t
        real(dp), allocatable :: coef(:, :)
        integer :: n_features = 0
        integer :: n_outputs = 0
        logical :: fit_intercept = .true.
        real(dp) :: ridge = 0.0_dp
    contains
        procedure, public :: fit_matrix => linear_fit_matrix
        procedure, public :: fit_vector => linear_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: predict_matrix => linear_predict_matrix
        procedure, public :: predict_vector => linear_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
    end type linear_regression_t

    public :: linear_predict_jvp
    public :: linear_predict_vjp

contains

    subroutine linear_fit_matrix(self, x, y, status, ridge, fit_intercept)
        class(linear_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: ridge
        logical, intent(in), optional :: fit_intercept
        real(dp), allocatable :: design(:, :), gram(:, :), rhs(:, :)
        integer :: n_samples, n_features, n_outputs, info
        real(dp) :: lambda
        logical :: intercept

        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_outputs = size(y, 2)
        lambda = 0.0_dp
        if (present(ridge)) lambda = ridge
        intercept = .true.
        if (present(fit_intercept)) intercept = fit_intercept

        if (n_samples < 1 .or. n_features < 1 .or. n_outputs < 1 .or. &
            size(y, 1) /= n_samples .or. lambda < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear fit: invalid dimensions or ridge parameter")
            return
        end if

        allocate(design(n_samples, n_features + 1))
        call make_design(x, intercept, design)
        gram = matmul(transpose(design), design)
        rhs = matmul(transpose(design), y)
        if (.not. intercept) then
            ! Keep the public parameter layout stable while pinning the unused
            ! intercept coefficient to zero instead of solving a singular row.
            gram(1, :) = 0.0_dp
            gram(:, 1) = 0.0_dp
            gram(1, 1) = 1.0_dp
            rhs(1, :) = 0.0_dp
        end if
        if (lambda > 0.0_dp) gram(2:, 2:) = gram(2:, 2:) + &
            lambda*identity_matrix(n_features)

        allocate(self%coef(n_features + 1, n_outputs))
        call dense_solve(gram, rhs, self%coef, info)
        if (info /= 0) then
            deallocate(self%coef)
            self%n_features = 0
            self%n_outputs = 0
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "linear fit: design matrix is singular")
            return
        end if

        self%n_features = n_features
        self%n_outputs = n_outputs
        self%fit_intercept = intercept
        self%ridge = lambda
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_fit_matrix

    subroutine linear_fit_vector(self, x, y, status, ridge, fit_intercept)
        class(linear_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: ridge
        logical, intent(in), optional :: fit_intercept
        real(dp), allocatable :: ym(:, :)

        allocate(ym(size(y), 1))
        ym(:, 1) = y
        call linear_fit_matrix(self, x, ym, status, ridge, fit_intercept)
    end subroutine linear_fit_vector

    subroutine linear_predict_matrix(self, x, y, status)
        class(linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :)

        if (.not. allocated(self%coef) .or. size(x, 2) /= self%n_features .or. &
            any(shape(y) /= [size(x, 1), self%n_outputs])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear predict: model or output shape is invalid")
            return
        end if
        allocate(design(size(x, 1), self%n_features + 1))
        call make_design(x, self%fit_intercept, design)
        y = matmul(design, self%coef)
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_predict_matrix

    subroutine linear_predict_vector(self, x, y, status)
        class(linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: ym(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear predict: output shape is invalid")
            return
        end if
        allocate(ym(size(y), 1))
        call linear_predict_matrix(self, x, ym, status)
        if (status%code == FORTNUM_OK) y = ym(:, 1)
    end subroutine linear_predict_vector

    subroutine linear_predict_jvp(coef, x, dcoef, dx, y, dy, fit_intercept)
        real(dp), intent(in) :: coef(:, :), x(:, :), dcoef(:, :), dx(:, :)
        real(dp), intent(out) :: y(:, :), dy(:, :)
        logical, intent(in), optional :: fit_intercept
        real(dp), allocatable :: design(:, :), ddesign(:, :)
        logical :: intercept

        intercept = .true.
        if (present(fit_intercept)) intercept = fit_intercept
        allocate(design(size(x, 1), size(x, 2) + 1))
        allocate(ddesign(size(x, 1), size(x, 2) + 1))
        call make_design(x, intercept, design)
        call make_design(dx, .false., ddesign)
        y = matmul(design, coef)
        dy = matmul(design, dcoef) + matmul(ddesign, coef)
    end subroutine linear_predict_jvp

    subroutine linear_predict_vjp(coef, x, u, coef_bar, x_bar, fit_intercept)
        real(dp), intent(in) :: coef(:, :), x(:, :), u(:, :)
        real(dp), intent(out) :: coef_bar(:, :), x_bar(:, :)
        logical, intent(in), optional :: fit_intercept
        real(dp), allocatable :: design(:, :)
        logical :: intercept
        integer :: j

        intercept = .true.
        if (present(fit_intercept)) intercept = fit_intercept
        allocate(design(size(x, 1), size(x, 2) + 1))
        call make_design(x, intercept, design)
        coef_bar = matmul(transpose(design), u)
        x_bar = 0.0_dp
        do j = 1, size(x, 2)
            x_bar(:, j) = matmul(u, coef(j + 1, :))
        end do
    end subroutine linear_predict_vjp

    subroutine make_design(x, intercept, design)
        real(dp), intent(in) :: x(:, :)
        logical, intent(in) :: intercept
        real(dp), intent(out) :: design(:, :)

        design = 0.0_dp
        if (intercept) design(:, 1) = 1.0_dp
        design(:, 2:) = x
    end subroutine make_design

    pure function identity_matrix(n) result(identity)
        integer, intent(in) :: n
        real(dp) :: identity(n, n)
        integer :: i
        identity = 0.0_dp
        do i = 1, n
            identity(i, i) = 1.0_dp
        end do
    end function identity_matrix

end module fortml_linear_regression
