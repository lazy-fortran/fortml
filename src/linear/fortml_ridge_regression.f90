module fortml_ridge_regression
    !! Weighted dense ridge regression with an explicit packed prediction API.
    !!
    !! The fitted state stores coefficients in Fortran column-major order.  An
    !! intercept, when requested, is the first coefficient row and is not
    !! regularized.  Fitting uses a weighted SVD least-squares solve, with
    !! square-root ridge rows appended for the feature coefficients.  The
    !! prediction products hold this fitted state fixed; the SVD solve and its
    !! rank decisions are intentionally not differentiated.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    implicit none
    private

    interface
        subroutine dgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, &
                work, lwork, info)
            import :: dp
            integer, intent(in) :: m, n, nrhs, lda, ldb, lwork
            real(dp), intent(inout) :: a(lda, *), b(ldb, *)
            real(dp), intent(out) :: s(*)
            real(dp), intent(in) :: rcond
            integer, intent(out) :: rank, info
            real(dp), intent(inout) :: work(*)
        end subroutine dgelss
    end interface

    !> Weighted multi-output ridge estimator.
    type, public :: ridge_regression_t
        private
        real(dp), allocatable :: coefficient(:, :)
        integer :: feature_count_value = 0
        integer :: output_count_value = 0
        real(dp) :: alpha_value = 1.0_dp
        logical :: fit_intercept_value = .true.
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit_matrix => ridge_fit_matrix
        procedure, public :: fit_vector => ridge_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: predict_matrix => ridge_predict_matrix
        procedure, public :: predict_vector => ridge_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_jvp => ridge_predict_jvp
        procedure, public :: predict_vjp => ridge_predict_vjp
        procedure, public :: jvp => ridge_predict_jvp
        procedure, public :: vjp => ridge_predict_vjp
        procedure, public :: coefficients => ridge_coefficients
        procedure, public :: parameters => ridge_parameters
        procedure, public :: set_parameters => ridge_set_parameters
        procedure, public :: parameter_count => ridge_parameter_count
        procedure, public :: feature_count => ridge_feature_count
        procedure, public :: output_count => ridge_output_count
        procedure, public :: regularization => ridge_regularization
        procedure, public :: fit_intercept => ridge_fit_intercept
        procedure, public :: fitted => ridge_fitted
    end type ridge_regression_t

contains

    !> Fit weighted ridge regression for a matrix target.
    subroutine ridge_fit_matrix(self, x, y, status, alpha, fit_intercept, &
            sample_weight)
        class(ridge_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha
        logical, intent(in), optional :: fit_intercept
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: weights(:), design(:, :), rhs(:, :)
        real(dp), allocatable :: singular_values(:), work(:)
        real(dp) :: work_query(1), penalty, weight_mass
        integer :: n_samples, n_features, n_outputs, n_parameters, n_rows
        integer :: lwork, rank, info, i, j, offset
        logical :: include_intercept

        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_outputs = size(y, 2)
        if (n_samples < 1 .or. n_features < 1 .or. n_outputs < 1 .or. &
            size(y, 1) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge fit: input and target dimensions must be positive and match")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge fit: inputs and targets must be finite")
            return
        end if

        penalty = 1.0_dp
        if (present(alpha)) penalty = alpha
        if (.not. ieee_is_finite(penalty) .or. penalty < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge fit: alpha must be finite and nonnegative")
            return
        end if

        include_intercept = .true.
        if (present(fit_intercept)) include_intercept = fit_intercept

        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ridge fit: sample weights must match sample count")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ridge fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        weight_mass = sum(weights)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge fit: sample weights must have positive mass")
            return
        end if

        n_parameters = n_features
        offset = 0
        if (include_intercept) then
            n_parameters = n_parameters + 1
            offset = 1
        end if
        n_rows = n_samples
        if (penalty > 0.0_dp) n_rows = n_rows + n_features

        allocate(design(n_rows, n_parameters))
        allocate(rhs(max(n_rows, n_parameters), n_outputs))
        allocate(singular_values(min(n_rows, n_parameters)))
        design = 0.0_dp
        rhs = 0.0_dp
        if (include_intercept) then
            design(:n_samples, 1) = sqrt(weights)
            do j = 1, n_features
                design(:n_samples, j + offset) = sqrt(weights)*x(:, j)
            end do
        else
            do j = 1, n_features
                design(:n_samples, j) = sqrt(weights)*x(:, j)
            end do
        end if
        do i = 1, n_samples
            rhs(i, :) = sqrt(weights(i))*y(i, :)
        end do
        if (penalty > 0.0_dp) then
            do j = 1, n_features
                design(n_samples + j, j + offset) = sqrt(penalty)
            end do
        end if

        lwork = -1
        call dgelss(n_rows, n_parameters, n_outputs, design, n_rows, rhs, &
            max(n_rows, n_parameters), singular_values, -1.0_dp, rank, &
            work_query, lwork, info)
        if (info /= 0 .or. .not. ieee_is_finite(work_query(1))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ridge fit: SVD workspace query failed")
            return
        end if
        lwork = max(1, int(work_query(1)))
        allocate(work(lwork))
        call dgelss(n_rows, n_parameters, n_outputs, design, n_rows, rhs, &
            max(n_rows, n_parameters), singular_values, -1.0_dp, rank, work, &
            lwork, info)
        if (info /= 0) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ridge fit: SVD least-squares solve failed")
            return
        end if
        if (any(.not. ieee_is_finite(rhs(:n_parameters, :)))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ridge fit: SVD returned nonfinite coefficients")
            return
        end if

        allocate(self%coefficient(n_parameters, n_outputs))
        self%coefficient = rhs(:n_parameters, :)
        self%feature_count_value = n_features
        self%output_count_value = n_outputs
        self%alpha_value = penalty
        self%fit_intercept_value = include_intercept
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine ridge_fit_matrix

    !> Fit weighted ridge regression for a vector target.
    subroutine ridge_fit_vector(self, x, y, status, alpha, fit_intercept, &
            sample_weight)
        class(ridge_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha
        logical, intent(in), optional :: fit_intercept
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: target(:, :)

        allocate(target(size(y), 1))
        target(:, 1) = y
        call ridge_fit_matrix(self, x, target, status, alpha, fit_intercept, &
            sample_weight)
    end subroutine ridge_fit_vector

    !> Predict one or more outputs for a row-major sample matrix.
    subroutine ridge_predict_matrix(self, x, y, status)
        class(ridge_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :)

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge predict: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge predict: inputs must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        call make_design(x, self%fit_intercept_value, design)
        y = matmul(design, self%coefficient)
        call status_set(status, FORTNUM_OK, "")
    end subroutine ridge_predict_matrix

    !> Predict a single output column for a row-major sample matrix.
    subroutine ridge_predict_vector(self, x, y, status)
        class(ridge_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge predict: output shape is invalid")
            return
        end if
        allocate(values(size(y), 1))
        call ridge_predict_matrix(self, x, values, status)
        if (status%code == FORTNUM_OK) y = values(:, 1)
    end subroutine ridge_predict_vector

    !> Prediction JVP over packed coefficients and continuous input rows.
    subroutine ridge_predict_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(ridge_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), design_dot(:, :)
        real(dp), allocatable :: coefficient_dot(:, :)

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(y) /= [size(x, 1), self%output_count_value]) .or. &
            any(shape(y_dot) /= shape(y)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge JVP: model, tangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(theta_dot)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge JVP: inputs and tangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(design_dot, mold=design)
        allocate(coefficient_dot, mold=self%coefficient)
        call make_design(x, self%fit_intercept_value, design)
        call make_tangent_design(x_dot, self%fit_intercept_value, design_dot)
        coefficient_dot = reshape(theta_dot, shape(coefficient_dot))
        y = matmul(design, self%coefficient)
        y_dot = matmul(design_dot, self%coefficient) + &
            matmul(design, coefficient_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine ridge_predict_jvp

    !> Prediction VJP over packed coefficients and continuous input rows.
    subroutine ridge_predict_vjp(self, x, u, theta_bar, x_bar, status)
        class(ridge_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: design(:, :), coefficient_bar(:, :)
        integer :: j, offset

        if (.not. self%fitted_value .or. size(x, 1) < 1 .or. &
            size(x, 2) /= self%feature_count_value .or. &
            any(shape(u) /= [size(x, 1), self%output_count_value]) .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge VJP: model, cotangent, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(u))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(design(size(x, 1), size(self%coefficient, 1)))
        allocate(coefficient_bar, mold=self%coefficient)
        call make_design(x, self%fit_intercept_value, design)
        coefficient_bar = matmul(transpose(design), u)
        theta_bar = reshape(coefficient_bar, [size(theta_bar)])
        x_bar = 0.0_dp
        offset = 0
        if (self%fit_intercept_value) offset = 1
        do j = 1, self%feature_count_value
            x_bar(:, j) = matmul(u, self%coefficient(offset + j, :))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ridge_predict_vjp

    function ridge_coefficients(self) result(values)
        class(ridge_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%coefficient)) then
            allocate(values, mold=self%coefficient)
            values = self%coefficient
        else
            allocate(values(0, 0))
        end if
    end function ridge_coefficients

    function ridge_parameters(self) result(values)
        class(ridge_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        allocate(values(self%parameter_count()))
        if (self%parameter_count() > 0) then
            values = reshape(self%coefficient, [self%parameter_count()])
        end if
    end function ridge_parameters

    subroutine ridge_set_parameters(self, values, status)
        class(ridge_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted_value .or. size(values) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge set_parameters: model or packed shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ridge set_parameters: packed values must be finite")
            return
        end if
        self%coefficient = reshape(values, shape(self%coefficient))
        call status_set(status, FORTNUM_OK, "")
    end subroutine ridge_set_parameters

    integer function ridge_parameter_count(self) result(count)
        class(ridge_regression_t), intent(in) :: self

        if (allocated(self%coefficient)) then
            count = size(self%coefficient)
        else
            count = 0
        end if
    end function ridge_parameter_count

    integer function ridge_feature_count(self) result(count)
        class(ridge_regression_t), intent(in) :: self

        count = self%feature_count_value
    end function ridge_feature_count

    integer function ridge_output_count(self) result(count)
        class(ridge_regression_t), intent(in) :: self

        count = self%output_count_value
    end function ridge_output_count

    real(dp) function ridge_regularization(self) result(value)
        class(ridge_regression_t), intent(in) :: self

        value = self%alpha_value
    end function ridge_regularization

    logical function ridge_fit_intercept(self) result(value)
        class(ridge_regression_t), intent(in) :: self

        value = self%fit_intercept_value
    end function ridge_fit_intercept

    logical function ridge_fitted(self) result(value)
        class(ridge_regression_t), intent(in) :: self

        value = self%fitted_value .and. allocated(self%coefficient)
    end function ridge_fitted

    subroutine make_design(x, intercept, design)
        real(dp), intent(in) :: x(:, :)
        logical, intent(in) :: intercept
        real(dp), intent(out) :: design(:, :)

        design = 0.0_dp
        if (intercept) then
            design(:, 1) = 1.0_dp
            design(:, 2:) = x
        else
            design = x
        end if
    end subroutine make_design

    subroutine make_tangent_design(x_dot, intercept, design_dot)
        real(dp), intent(in) :: x_dot(:, :)
        logical, intent(in) :: intercept
        real(dp), intent(out) :: design_dot(:, :)

        design_dot = 0.0_dp
        if (intercept) then
            design_dot(:, 2:) = x_dot
        else
            design_dot = x_dot
        end if
    end subroutine make_tangent_design

end module fortml_ridge_regression
