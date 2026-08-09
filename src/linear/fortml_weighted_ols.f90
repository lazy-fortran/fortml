module fortml_weighted_ols
    !! Weighted dense ordinary least-squares regression.
    !!
    !! This estimator is the unregularized, fixed-fit companion to the
    !! weighted ridge estimator.  It delegates the deterministic weighted
    !! SVD solve and fixed-state linear products to the shared ridge kernel
    !! with zero penalty; no derivative is taken through fitting.
    use, intrinsic :: iso_fortran_env, only: real64
    use fortnum_status, only: fortnum_status_t, status_set, &
        FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU
    use fortml_ridge_regression, only: ridge_regression_t
    implicit none
    private

    integer, parameter :: dp = real64

    type, public :: weighted_ols_regression_t
        private
        type(ridge_regression_t) :: ridge
    contains
        procedure, public :: fit_matrix => weighted_ols_fit_matrix
        procedure, public :: fit_vector => weighted_ols_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: predict_matrix => weighted_ols_predict_matrix
        procedure, public :: predict_vector => weighted_ols_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_device => weighted_ols_predict_device
        procedure, public :: device_supported => weighted_ols_device_supported
        procedure, public :: predict_jvp => weighted_ols_predict_jvp
        procedure, public :: predict_vjp => weighted_ols_predict_vjp
        procedure, public :: jvp => weighted_ols_predict_jvp
        procedure, public :: vjp => weighted_ols_predict_vjp
        procedure, public :: coefficients => weighted_ols_coefficients
        procedure, public :: parameters => weighted_ols_parameters
        procedure, public :: set_parameters => weighted_ols_set_parameters
        procedure, public :: parameter_count => weighted_ols_parameter_count
        procedure, public :: feature_count => weighted_ols_feature_count
        procedure, public :: output_count => weighted_ols_output_count
        procedure, public :: fit_intercept => weighted_ols_fit_intercept
        procedure, public :: fitted => weighted_ols_fitted
    end type weighted_ols_regression_t

contains

    subroutine weighted_ols_fit_matrix(self, x, y, status, fit_intercept, &
            sample_weight)
        class(weighted_ols_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fit_intercept
        real(dp), intent(in), optional :: sample_weight(:)

        call self%ridge%fit(x, y, status, alpha=0.0_dp, &
            fit_intercept=fit_intercept, sample_weight=sample_weight)
    end subroutine weighted_ols_fit_matrix

    subroutine weighted_ols_fit_vector(self, x, y, status, fit_intercept, &
            sample_weight)
        class(weighted_ols_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fit_intercept
        real(dp), intent(in), optional :: sample_weight(:)

        call self%ridge%fit(x, y, status, alpha=0.0_dp, &
            fit_intercept=fit_intercept, sample_weight=sample_weight)
    end subroutine weighted_ols_fit_vector

    subroutine weighted_ols_predict_matrix(self, x, y, status)
        class(weighted_ols_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        call self%ridge%predict(x, y, status)
    end subroutine weighted_ols_predict_matrix

    subroutine weighted_ols_predict_vector(self, x, y, status)
        class(weighted_ols_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status

        call self%ridge%predict(x, y, status)
    end subroutine weighted_ols_predict_vector

    subroutine weighted_ols_predict_device(self, device, x, y, status)
        class(weighted_ols_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (device%kind == FORTML_DEVICE_CPU) then
            call self%predict_matrix(x, y, status)
        else
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "weighted OLS prediction: CUDA kernel is not resident")
        end if
    end subroutine weighted_ols_predict_device

    logical function weighted_ols_device_supported(self, device_kind) &
            result(supported)
        class(weighted_ols_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function weighted_ols_device_supported

    subroutine weighted_ols_predict_jvp(self, x, theta_dot, x_dot, y, y_dot, &
            status)
        class(weighted_ols_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: y(:, :), y_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        call self%ridge%predict_jvp(x, theta_dot, x_dot, y, y_dot, status)
    end subroutine weighted_ols_predict_jvp

    subroutine weighted_ols_predict_vjp(self, x, u, theta_bar, x_bar, status)
        class(weighted_ols_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), u(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        call self%ridge%predict_vjp(x, u, theta_bar, x_bar, status)
    end subroutine weighted_ols_predict_vjp

    function weighted_ols_coefficients(self) result(values)
        class(weighted_ols_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        values = self%ridge%coefficients()
    end function weighted_ols_coefficients

    function weighted_ols_parameters(self) result(values)
        class(weighted_ols_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        values = self%ridge%parameters()
    end function weighted_ols_parameters

    subroutine weighted_ols_set_parameters(self, values, status)
        class(weighted_ols_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        call self%ridge%set_parameters(values, status)
    end subroutine weighted_ols_set_parameters

    integer function weighted_ols_parameter_count(self) result(count)
        class(weighted_ols_regression_t), intent(in) :: self

        count = self%ridge%parameter_count()
    end function weighted_ols_parameter_count

    integer function weighted_ols_feature_count(self) result(count)
        class(weighted_ols_regression_t), intent(in) :: self

        count = self%ridge%feature_count()
    end function weighted_ols_feature_count

    integer function weighted_ols_output_count(self) result(count)
        class(weighted_ols_regression_t), intent(in) :: self

        count = self%ridge%output_count()
    end function weighted_ols_output_count

    logical function weighted_ols_fit_intercept(self) result(value)
        class(weighted_ols_regression_t), intent(in) :: self

        value = self%ridge%fit_intercept()
    end function weighted_ols_fit_intercept

    logical function weighted_ols_fitted(self) result(value)
        class(weighted_ols_regression_t), intent(in) :: self

        value = self%ridge%fitted()
    end function weighted_ols_fitted

end module fortml_weighted_ols
