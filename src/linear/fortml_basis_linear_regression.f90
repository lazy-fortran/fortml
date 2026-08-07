module fortml_basis_linear_regression
    !! A fitted basis pipeline coupled to a multi-output linear regressor.
    !!
    !! The packed parameter vector stores all differentiable basis parameters
    !! followed by the Fortran-column-major regression coefficients.  The
    !! composition keeps the pipeline and estimator contracts visible while
    !! exposing exact chained input and parameter products.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_pipeline, only: basis_pipeline_t
    use fortml_linear_regression, only: linear_regression_t, &
        linear_predict_jvp, linear_predict_vjp
    implicit none
    private

    type, public :: basis_linear_regression_t
        private
        type(basis_pipeline_t) :: pipeline
        type(linear_regression_t) :: regressor
        integer :: n_outputs = 0
        logical :: fitted = .false.
    contains
        procedure, public :: fit => basis_linear_fit
        procedure, public :: transform => basis_linear_transform
        procedure, public :: predict => basis_linear_predict
        procedure, public :: predict_jvp => basis_linear_predict_jvp
        procedure, public :: predict_vjp => basis_linear_predict_vjp
        procedure, public :: input_count => basis_linear_input_count
        procedure, public :: feature_count => basis_linear_feature_count
        procedure, public :: output_count => basis_linear_output_count
        procedure, public :: parameter_count => basis_linear_parameter_count
        procedure, public :: parameters => basis_linear_parameters
        procedure, public :: set_parameters => basis_linear_set_parameters
        procedure, public :: static_lowering_eligible => &
            basis_linear_static_lowering_eligible
        procedure, public :: is_fitted => basis_linear_is_fitted
    end type basis_linear_regression_t

contains

    subroutine basis_linear_fit(self, pipeline, x, y, status, ridge, fit_intercept)
        class(basis_linear_regression_t), intent(out) :: self
        type(basis_pipeline_t), intent(in) :: pipeline
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: ridge
        logical, intent(in), optional :: fit_intercept
        real(dp), allocatable :: features(:, :)

        if (.not. pipeline%valid() .or. size(x, 1) < 1 .or. &
            size(x, 2) /= pipeline%input_count() .or. size(y, 1) /= size(x, 1) .or. &
            size(y, 2) < 1 .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis linear fit: pipeline or data is invalid")
            return
        end if
        self%pipeline = pipeline
        call self%pipeline%fit(x, status)
        if (status%code /= FORTNUM_OK) return
        allocate(features(size(x, 1), self%pipeline%feature_count()))
        call self%pipeline%transform(x, features, status)
        if (status%code /= FORTNUM_OK) return
        if (present(ridge)) then
            if (present(fit_intercept)) then
                call self%regressor%fit(features, y, status, ridge, fit_intercept)
            else
                call self%regressor%fit(features, y, status, ridge)
            end if
        else if (present(fit_intercept)) then
            call self%regressor%fit(features, y, status, fit_intercept=fit_intercept)
        else
            call self%regressor%fit(features, y, status)
        end if
        if (status%code /= FORTNUM_OK) return
        self%n_outputs = size(y, 2)
        self%fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_linear_fit

    subroutine basis_linear_transform(self, x, features, status)
        class(basis_linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: features(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted .or. size(x, 2) /= self%input_count() .or. &
            size(features, 1) /= size(x, 1) .or. &
            size(features, 2) /= self%feature_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis linear transform: model or shape is invalid")
            return
        end if
        call self%pipeline%transform(x, features, status)
    end subroutine basis_linear_transform

    subroutine basis_linear_predict(self, x, y, status)
        class(basis_linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: features(:, :)

        if (.not. self%fitted .or. size(x, 2) /= self%input_count() .or. &
            size(y, 1) /= size(x, 1) .or. size(y, 2) /= self%n_outputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis linear predict: model or shape is invalid")
            return
        end if
        allocate(features(size(x, 1), self%feature_count()))
        call self%pipeline%transform(x, features, status)
        if (status%code /= FORTNUM_OK) return
        call self%regressor%predict(features, y, status)
    end subroutine basis_linear_predict

    subroutine basis_linear_predict_jvp(self, x, theta_dot, x_dot, y, y_dot, status)
        class(basis_linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: features(:, :), features_dot(:, :)
        real(dp), allocatable :: coefficient_dot(:, :)
        integer :: n_pipeline_parameters, coefficient_count

        if (.not. self%fitted .or. size(x, 2) /= self%input_count() .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y, 1) /= size(x, 1) .or. &
            size(y, 2) /= self%n_outputs .or. any(shape(y_dot) /= shape(y)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis linear JVP: model, tangent, or shape is invalid")
            return
        end if
        n_pipeline_parameters = self%pipeline%parameter_count()
        coefficient_count = size(self%regressor%coef)
        allocate(features(size(x, 1), self%feature_count()))
        allocate(features_dot, mold=features)
        allocate(coefficient_dot, mold=self%regressor%coef)
        call self%pipeline%jvp(x, theta_dot(:n_pipeline_parameters), x_dot, &
            features, features_dot, status)
        if (status%code /= FORTNUM_OK) return
        coefficient_dot = reshape(theta_dot(n_pipeline_parameters + 1:), &
            shape(coefficient_dot))
        call linear_predict_jvp(self%regressor%coef, features, coefficient_dot, &
            features_dot, y, y_dot, self%regressor%fit_intercept)
        if (size(coefficient_dot) /= coefficient_count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis linear JVP: coefficient packing is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_linear_predict_jvp

    subroutine basis_linear_predict_vjp(self, x, u, theta_bar, x_bar, status)
        class(basis_linear_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: features(:, :), features_bar(:, :)
        real(dp), allocatable :: coefficient_bar(:, :), local_theta_bar(:)
        integer :: n_pipeline_parameters

        if (.not. self%fitted .or. size(x, 2) /= self%input_count() .or. &
            size(u, 1) /= size(x, 1) .or. size(u, 2) /= self%n_outputs .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis linear VJP: model, cotangent, or shape is invalid")
            return
        end if
        n_pipeline_parameters = self%pipeline%parameter_count()
        allocate(features(size(x, 1), self%feature_count()))
        allocate(features_bar, mold=features)
        allocate(coefficient_bar, mold=self%regressor%coef)
        allocate(local_theta_bar(n_pipeline_parameters))
        call self%pipeline%transform(x, features, status)
        if (status%code /= FORTNUM_OK) return
        call linear_predict_vjp(self%regressor%coef, features, u, coefficient_bar, &
            features_bar, self%regressor%fit_intercept)
        call self%pipeline%vjp(x, features_bar, local_theta_bar, x_bar, status)
        if (status%code /= FORTNUM_OK) return
        if (n_pipeline_parameters > 0) theta_bar(:n_pipeline_parameters) = &
            local_theta_bar
        theta_bar(n_pipeline_parameters + 1:) = reshape(coefficient_bar, [size(&
            coefficient_bar)])
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_linear_predict_vjp

    integer function basis_linear_input_count(self) result(count)
        class(basis_linear_regression_t), intent(in) :: self
        count = self%pipeline%input_count()
    end function basis_linear_input_count

    integer function basis_linear_feature_count(self) result(count)
        class(basis_linear_regression_t), intent(in) :: self
        count = self%pipeline%feature_count()
    end function basis_linear_feature_count

    integer function basis_linear_output_count(self) result(count)
        class(basis_linear_regression_t), intent(in) :: self
        count = self%n_outputs
    end function basis_linear_output_count

    integer function basis_linear_parameter_count(self) result(count)
        class(basis_linear_regression_t), intent(in) :: self
        count = self%pipeline%parameter_count()
        if (allocated(self%regressor%coef)) count = count + size(self%regressor%coef)
    end function basis_linear_parameter_count

    function basis_linear_parameters(self) result(theta)
        class(basis_linear_regression_t), intent(in) :: self
        real(dp), allocatable :: theta(:), pipeline_parameters(:)
        integer :: n_pipeline_parameters

        allocate(theta(self%parameter_count()))
        theta = 0.0_dp
        n_pipeline_parameters = self%pipeline%parameter_count()
        if (n_pipeline_parameters > 0) then
            pipeline_parameters = self%pipeline%parameters()
            theta(:n_pipeline_parameters) = pipeline_parameters
        end if
        if (allocated(self%regressor%coef)) then
            theta(n_pipeline_parameters + 1:) = reshape(self%regressor%coef, &
                [size(self%regressor%coef)])
        end if
    end function basis_linear_parameters

    subroutine basis_linear_set_parameters(self, theta, status)
        class(basis_linear_regression_t), intent(inout) :: self
        real(dp), intent(in) :: theta(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: n_pipeline_parameters, coefficient_count

        if (.not. self%fitted .or. size(theta) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(theta))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis linear set_parameters: model or vector is invalid")
            return
        end if
        n_pipeline_parameters = self%pipeline%parameter_count()
        coefficient_count = size(self%regressor%coef)
        if (n_pipeline_parameters > 0) then
            call self%pipeline%set_parameters(theta(:n_pipeline_parameters), status)
            if (status%code /= FORTNUM_OK) return
        end if
        self%regressor%coef = reshape(theta(n_pipeline_parameters + 1:), &
            shape(self%regressor%coef))
        if (size(theta(n_pipeline_parameters + 1:)) /= coefficient_count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "basis linear set_parameters: coefficient packing is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine basis_linear_set_parameters

    logical function basis_linear_is_fitted(self) result(fitted)
        class(basis_linear_regression_t), intent(in) :: self
        fitted = self%fitted .and. allocated(self%regressor%coef)
    end function basis_linear_is_fitted

    logical function basis_linear_static_lowering_eligible(self) result(eligible)
        !! Report whether the fitted feature transform is statically lowerable.
        !!
        !! This is deliberately a transform capability, not an end-to-end GPU
        !! claim: the dense regression solve and prediction still execute on
        !! the host until a resident linear-regression kernel is linked.  A
        !! callback stage therefore makes the result false rather than hiding a
        !! host callback behind an accelerator dispatch.
        class(basis_linear_regression_t), intent(in) :: self

        eligible = self%fitted .and. self%pipeline%static_lowering_eligible()
    end function basis_linear_static_lowering_eligible

end module fortml_basis_linear_regression
