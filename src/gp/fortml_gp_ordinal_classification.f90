module fortml_gp_ordinal_classification
    !! Ordered Gaussian-process classification through a latent Gaussian GP.
    !!
    !! This bounded ordinal contract uses one zero-mean GP for the ordered
    !! class score.  Integer classes are mapped to ranks, the GP is fit to
    !! those ranks with a Gaussian observation model, and predictive class
    !! probabilities are adjacent normal-CDF differences at fixed mid-rank
    !! cut points.  This gives a stable ordered baseline with the same kernel,
    !! parameter, and input-product contracts as GP regression.  It is
    !! intentionally explicit about being a latent-Gaussian surrogate rather
    !! than a Laplace approximation to a cumulative likelihood.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED, &
        FORTNUM_CONVERGENCE_ERROR
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t
    use fortml_gaussian_process, only: gp_regression_t
    implicit none
    private

    real(dp), parameter :: SQRT_TWO = 1.4142135623730950488016887242097_dp
    real(dp), parameter :: SQRT_TWO_PI = 2.506628274631000502415765284811_dp
    real(dp), parameter :: MIN_SCALE = 1.0e-12_dp

    integer, parameter, public :: GP_ORDINAL_LIKELIHOOD_LOGISTIC = 1
    integer, parameter, public :: GP_ORDINAL_LIKELIHOOD_PROBIT = 2

    public :: gp_ordinal_log_likelihood_value
    public :: gp_ordinal_log_likelihood_value_device
    public :: gp_ordinal_log_likelihood_jvp
    public :: gp_ordinal_log_likelihood_jvp_device
    public :: gp_ordinal_log_likelihood_vjp
    public :: gp_ordinal_log_likelihood_vjp_device
    public :: gp_ordinal_log_likelihood_hvp
    public :: gp_ordinal_log_likelihood_hvp_device
    public :: gp_ordinal_likelihood_device_supported
    public :: gp_ordinal_predict_log_proba
    public :: gp_ordinal_predict_log_proba_device
    public :: gp_ordinal_predict_log_proba_parameter_jvp
    public :: gp_ordinal_predict_log_proba_parameter_vjp
    public :: gp_ordinal_predict_log_proba_input_jvp
    public :: gp_ordinal_predict_log_proba_input_vjp

    type, public :: gp_ordinal_classification_options_t
        !! Controls the latent Gaussian fit and predictive uncertainty.
        real(dp) :: noise_variance = 0.05_dp
        real(dp) :: jitter = 1.0e-8_dp
    end type gp_ordinal_classification_options_t

    type, public :: gp_ordinal_classification_state_t
        integer :: class_count = 0
        integer :: iterations = 1
        logical :: converged = .false.
        real(dp) :: noise_variance = 0.0_dp
    end type gp_ordinal_classification_state_t

    type, public :: gp_ordinal_classification_t
        private
        type(gp_regression_t) :: latent
        integer, allocatable :: class_label(:)
        real(dp), allocatable :: cut_points(:)
        integer :: n_classes = 0
        integer :: n_features = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => gp_ordinal_fit
        procedure, public :: predict_latent => gp_ordinal_predict_latent
        procedure, public :: predict_proba => gp_ordinal_predict_proba
        procedure, public :: predict_log_proba => gp_ordinal_predict_log_proba
        procedure, public :: predict => gp_ordinal_predict
        procedure, public :: predict_latent_parameter_jvp => &
            gp_ordinal_predict_latent_parameter_jvp
        procedure, public :: predict_latent_parameter_vjp => &
            gp_ordinal_predict_latent_parameter_vjp
        procedure, public :: predict_proba_parameter_jvp => &
            gp_ordinal_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            gp_ordinal_predict_proba_parameter_vjp
        procedure, public :: predict_log_proba_parameter_jvp => &
            gp_ordinal_predict_log_proba_parameter_jvp
        procedure, public :: predict_log_proba_parameter_vjp => &
            gp_ordinal_predict_log_proba_parameter_vjp
        procedure, public :: predict_latent_input_jvp => &
            gp_ordinal_predict_latent_input_jvp
        procedure, public :: predict_latent_input_vjp => &
            gp_ordinal_predict_latent_input_vjp
        procedure, public :: predict_proba_input_jvp => &
            gp_ordinal_predict_proba_input_jvp
        procedure, public :: predict_proba_input_vjp => &
            gp_ordinal_predict_proba_input_vjp
        procedure, public :: predict_log_proba_input_jvp => &
            gp_ordinal_predict_log_proba_input_jvp
        procedure, public :: predict_log_proba_input_vjp => &
            gp_ordinal_predict_log_proba_input_vjp
        procedure, public :: predict_proba_threshold_jvp => &
            gp_ordinal_predict_proba_threshold_jvp
        procedure, public :: predict_proba_threshold_vjp => &
            gp_ordinal_predict_proba_threshold_vjp
        procedure, public :: predict_log_proba_threshold_jvp => &
            gp_ordinal_predict_log_proba_threshold_jvp
        procedure, public :: predict_log_proba_threshold_vjp => &
            gp_ordinal_predict_log_proba_threshold_vjp
        procedure, public :: predict_proba_device => &
            gp_ordinal_predict_proba_device
        procedure, public :: predict_log_proba_device => &
            gp_ordinal_predict_log_proba_device
        procedure, public :: predict_proba_parameter_vjp_device => &
            gp_ordinal_predict_proba_parameter_vjp_device
        procedure, public :: predict_proba_input_vjp_device => &
            gp_ordinal_predict_proba_input_vjp_device
        procedure, public :: predict_proba_threshold_jvp_device => &
            gp_ordinal_predict_proba_threshold_jvp_device
        procedure, public :: predict_proba_threshold_vjp_device => &
            gp_ordinal_predict_proba_threshold_vjp_device
        procedure, public :: classes => gp_ordinal_classes
        procedure, public :: thresholds => gp_ordinal_thresholds
        procedure, public :: set_thresholds => gp_ordinal_set_thresholds
        procedure, public :: class_count => gp_ordinal_class_count
        procedure, public :: feature_count => gp_ordinal_feature_count
        procedure, public :: parameter_count => gp_ordinal_parameter_count
        procedure, public :: hyperparameter_count => gp_ordinal_parameter_count
        procedure, public :: parameters => gp_ordinal_parameters
        procedure, public :: hyperparameters => gp_ordinal_parameters
        procedure, public :: set_parameters => gp_ordinal_set_parameters
        procedure, public :: set_hyperparameters => gp_ordinal_set_parameters
        procedure, public :: log_marginal_likelihood => &
            gp_ordinal_log_marginal_likelihood
        procedure, public :: log_marginal_likelihood_jvp => &
            gp_ordinal_log_marginal_likelihood_jvp
        procedure, public :: hyperparameter_gradient => &
            gp_ordinal_hyperparameter_gradient
        procedure, public :: hyperparameter_hvp => gp_ordinal_hyperparameter_hvp
        procedure, public :: hyperparameter_gradient_device => &
            gp_ordinal_hyperparameter_gradient_device
        procedure, public :: hyperparameter_hvp_device => &
            gp_ordinal_hyperparameter_hvp_device
        procedure, public :: fitted => gp_ordinal_fitted
        procedure, public :: device_supported => gp_ordinal_device_supported
    end type gp_ordinal_classification_t

contains

    !> Sum the ordered-logit or ordered-probit log likelihood.
    !!
    !! `eta` contains latent scores, `labels` are one-based ranks in
    !! `1:size(thresholds)+1`, and `thresholds` are strictly increasing
    !! finite cut points.  This primitive is independent of the latent GP
    !! approximation used by `gp_ordinal_classification_t`; it is the
    !! likelihood contract consumed by future native ordinal inference.
    !! The CPU implementation is analytic and does not silently execute for
    !! CUDA callers; use `gp_ordinal_likelihood_device_supported` to query the
    !! explicit backend boundary.
    subroutine gp_ordinal_log_likelihood_value(eta, labels, thresholds, likelihood, &
            value, status)
        real(dp), intent(in) :: eta(:), thresholds(:)
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: row_value
        integer :: i

        value = 0.0_dp
        call validate_ordinal_likelihood_inputs(eta, labels, thresholds, likelihood, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(eta)
            call ordinal_row_terms(eta(i), labels(i), thresholds, likelihood, row_value, &
                status=status)
            if (status%code /= FORTNUM_OK) return
            value = value + row_value
        end do
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP likelihood: value is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_log_likelihood_value

    !> Forward directional product of the ordered likelihood.
    subroutine gp_ordinal_log_likelihood_jvp(eta, labels, thresholds, likelihood, &
            eta_dot, thresholds_dot, value, value_dot, status)
        real(dp), intent(in) :: eta(:), thresholds(:), eta_dot(:), thresholds_dot(:)
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: row_value, eta_gradient, thresholds_gradient(size(thresholds))
        integer :: i

        value = 0.0_dp
        value_dot = 0.0_dp
        call validate_ordinal_likelihood_inputs(eta, labels, thresholds, likelihood, status)
        if (status%code /= FORTNUM_OK) return
        if (size(eta_dot) /= size(eta) .or. size(thresholds_dot) /= size(thresholds)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(eta_dot)) .or. &
            any(.not. ieee_is_finite(thresholds_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood JVP: tangent is not finite")
            return
        end if
        do i = 1, size(eta)
            call ordinal_row_terms(eta(i), labels(i), thresholds, likelihood, row_value, &
                eta_gradient, thresholds_gradient, status)
            if (status%code /= FORTNUM_OK) return
            value = value + row_value
            value_dot = value_dot + eta_gradient*eta_dot(i) + &
                dot_product(thresholds_gradient, thresholds_dot)
        end do
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(value_dot)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP likelihood JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_log_likelihood_jvp

    !> Reverse product of the ordered likelihood.
    subroutine gp_ordinal_log_likelihood_vjp(eta, labels, thresholds, likelihood, &
            value_bar, eta_bar, thresholds_bar, status)
        real(dp), intent(in) :: eta(:), thresholds(:), value_bar
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: eta_bar(:), thresholds_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: row_value, eta_gradient, local_threshold_bar(size(thresholds))
        integer :: i

        eta_bar = 0.0_dp
        thresholds_bar = 0.0_dp
        call validate_ordinal_likelihood_inputs(eta, labels, thresholds, likelihood, status)
        if (status%code /= FORTNUM_OK) return
        if (size(eta_bar) /= size(eta) .or. size(thresholds_bar) /= size(thresholds) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood VJP: cotangent shape is invalid")
            return
        end if
        do i = 1, size(eta)
            call ordinal_row_terms(eta(i), labels(i), thresholds, likelihood, row_value, &
                eta_gradient, local_threshold_bar, status)
            if (status%code /= FORTNUM_OK) return
            eta_bar(i) = eta_bar(i) + value_bar*eta_gradient
            thresholds_bar = thresholds_bar + value_bar*local_threshold_bar
        end do
        if (any(.not. ieee_is_finite(eta_bar)) .or. &
            any(.not. ieee_is_finite(thresholds_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP likelihood VJP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_log_likelihood_vjp

    !> Directional Hessian product of the ordered likelihood gradient.
    !!
    !! The output is `value_bar * H*[eta_dot,thresholds_dot]`, where `H` is
    !! the exact Hessian of the summed log likelihood.  Logistic and probit
    !! density derivatives are analytic, so this product is suitable for
    !! second-order hyperparameter objectives without finite differences.
    subroutine gp_ordinal_log_likelihood_hvp(eta, labels, thresholds, likelihood, &
            value_bar, eta_dot, thresholds_dot, eta_hvp, thresholds_hvp, status)
        real(dp), intent(in) :: eta(:), thresholds(:), value_bar
        real(dp), intent(in) :: eta_dot(:), thresholds_dot(:)
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: eta_hvp(:), thresholds_hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: row_value, eta_gradient, eta_hessian_dot
        real(dp) :: threshold_gradient(size(thresholds))
        real(dp) :: threshold_hessian_dot(size(thresholds))
        integer :: i

        eta_hvp = 0.0_dp
        thresholds_hvp = 0.0_dp
        call validate_ordinal_likelihood_inputs(eta, labels, thresholds, likelihood, status)
        if (status%code /= FORTNUM_OK) return
        if (size(eta_hvp) /= size(eta) .or. size(thresholds_hvp) /= size(thresholds) .or. &
            size(eta_dot) /= size(eta) .or. size(thresholds_dot) /= size(thresholds) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood HVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(eta_dot)) .or. &
            any(.not. ieee_is_finite(thresholds_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood HVP: tangent is not finite")
            return
        end if
        do i = 1, size(eta)
            call ordinal_row_hessian_direction(eta(i), labels(i), thresholds, likelihood, &
                eta_dot(i), thresholds_dot, row_value, eta_gradient, threshold_gradient, &
                eta_hessian_dot, threshold_hessian_dot, status)
            if (status%code /= FORTNUM_OK) return
            eta_hvp(i) = value_bar*eta_hessian_dot
            thresholds_hvp = thresholds_hvp + value_bar*threshold_hessian_dot
        end do
        if (any(.not. ieee_is_finite(eta_hvp)) .or. &
            any(.not. ieee_is_finite(thresholds_hvp))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP likelihood HVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_log_likelihood_hvp

    logical function gp_ordinal_likelihood_device_supported(device_kind) result(supported)
        !! Report the explicit device contract for the scalar primitive.
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function gp_ordinal_likelihood_device_supported

    subroutine gp_ordinal_log_likelihood_value_device(device, eta, labels, thresholds, &
            likelihood, value, status)
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: eta(:), thresholds(:)
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        call ordinal_device_dispatch(device, status)
        if (status%code /= FORTNUM_OK) return
        call gp_ordinal_log_likelihood_value(eta, labels, thresholds, likelihood, value, status)
    end subroutine gp_ordinal_log_likelihood_value_device

    subroutine gp_ordinal_log_likelihood_jvp_device(device, eta, labels, thresholds, likelihood, &
            eta_dot, thresholds_dot, value, value_dot, status)
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: eta(:), thresholds(:), eta_dot(:), thresholds_dot(:)
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        value_dot = 0.0_dp
        call ordinal_device_dispatch(device, status)
        if (status%code /= FORTNUM_OK) return
        call gp_ordinal_log_likelihood_jvp(eta, labels, thresholds, likelihood, eta_dot, &
            thresholds_dot, value, value_dot, status)
    end subroutine gp_ordinal_log_likelihood_jvp_device

    subroutine gp_ordinal_log_likelihood_vjp_device(device, eta, labels, thresholds, likelihood, &
            value_bar, eta_bar, thresholds_bar, status)
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: eta(:), thresholds(:), value_bar
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: eta_bar(:), thresholds_bar(:)
        type(fortnum_status_t), intent(out) :: status

        eta_bar = 0.0_dp
        thresholds_bar = 0.0_dp
        call ordinal_device_dispatch(device, status)
        if (status%code /= FORTNUM_OK) return
        call gp_ordinal_log_likelihood_vjp(eta, labels, thresholds, likelihood, value_bar, &
            eta_bar, thresholds_bar, status)
    end subroutine gp_ordinal_log_likelihood_vjp_device

    subroutine gp_ordinal_log_likelihood_hvp_device(device, eta, labels, thresholds, likelihood, &
            value_bar, eta_dot, thresholds_dot, eta_hvp, thresholds_hvp, status)
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: eta(:), thresholds(:), value_bar
        real(dp), intent(in) :: eta_dot(:), thresholds_dot(:)
        integer, intent(in) :: labels(:), likelihood
        real(dp), intent(out) :: eta_hvp(:), thresholds_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        eta_hvp = 0.0_dp
        thresholds_hvp = 0.0_dp
        call ordinal_device_dispatch(device, status)
        if (status%code /= FORTNUM_OK) return
        call gp_ordinal_log_likelihood_hvp(eta, labels, thresholds, likelihood, value_bar, &
            eta_dot, thresholds_dot, eta_hvp, thresholds_hvp, status)
    end subroutine gp_ordinal_log_likelihood_hvp_device

    subroutine ordinal_device_dispatch(device, status)
        type(fortml_device_t), intent(in) :: device
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call status_set(status, FORTNUM_OK, "")
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP likelihood device: resident CUDA reduction is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood device: device kind is invalid")
        end select
    end subroutine ordinal_device_dispatch

    subroutine validate_ordinal_likelihood_inputs(eta, labels, thresholds, likelihood, status)
        real(dp), intent(in) :: eta(:), thresholds(:)
        integer, intent(in) :: labels(:), likelihood
        type(fortnum_status_t), intent(out) :: status
        integer :: i, class_count

        if (size(eta) < 1 .or. size(labels) /= size(eta) .or. size(thresholds) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood: input dimensions are invalid")
            return
        end if
        if (likelihood /= GP_ORDINAL_LIKELIHOOD_LOGISTIC .and. &
            likelihood /= GP_ORDINAL_LIKELIHOOD_PROBIT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood: likelihood kind is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(eta)) .or. any(.not. ieee_is_finite(thresholds))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood: inputs must be finite")
            return
        end if
        do i = 2, size(thresholds)
            if (thresholds(i) <= thresholds(i - 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP likelihood: thresholds must be strictly increasing")
                return
            end if
        end do
        class_count = size(thresholds) + 1
        do i = 1, size(labels)
            if (labels(i) < 1 .or. labels(i) > class_count) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP likelihood: labels must be one-based ranks")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_ordinal_likelihood_inputs

    subroutine ordinal_row_terms(eta, label, thresholds, likelihood, row_value, eta_gradient, &
            thresholds_gradient, status)
        real(dp), intent(in) :: eta, thresholds(:)
        integer, intent(in) :: label, likelihood
        real(dp), intent(out) :: row_value
        real(dp), intent(out), optional :: eta_gradient, thresholds_gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: probability, upper_cdf, lower_cdf, upper_pdf, lower_pdf
        real(dp) :: upper_z, lower_z
        logical :: has_upper, has_lower

        row_value = 0.0_dp
        if (present(eta_gradient)) eta_gradient = 0.0_dp
        if (present(thresholds_gradient)) thresholds_gradient = 0.0_dp
        has_lower = label > 1
        has_upper = label <= size(thresholds)
        if (has_upper) then
            upper_z = thresholds(label) - eta
            call ordinal_cdf_pdf(upper_z, likelihood, upper_cdf, upper_pdf)
        else
            upper_cdf = 1.0_dp
            upper_pdf = 0.0_dp
        end if
        if (has_lower) then
            lower_z = thresholds(label - 1) - eta
            call ordinal_cdf_pdf(lower_z, likelihood, lower_cdf, lower_pdf)
        else
            lower_cdf = 0.0_dp
            lower_pdf = 0.0_dp
        end if
        probability = upper_cdf - lower_cdf
        if (.not. ieee_is_finite(probability) .or. probability <= tiny(1.0_dp)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP likelihood: category probability underflowed")
            return
        end if
        row_value = log(probability)
        if (present(eta_gradient)) eta_gradient = (lower_pdf - upper_pdf)/probability
        if (present(thresholds_gradient)) then
            if (size(thresholds_gradient) /= size(thresholds)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP likelihood: threshold gradient shape is invalid")
                return
            end if
            if (has_upper) thresholds_gradient(label) = upper_pdf/probability
            if (has_lower) thresholds_gradient(label - 1) = -lower_pdf/probability
        end if
        if (.not. ieee_is_finite(row_value)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP likelihood: log probability is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_row_terms

    subroutine ordinal_row_hessian_direction(eta, label, thresholds, likelihood, eta_dot, &
            thresholds_dot, row_value, eta_gradient, thresholds_gradient, eta_hessian_dot, &
            thresholds_hessian_dot, status)
        real(dp), intent(in) :: eta, thresholds(:), eta_dot, thresholds_dot(:)
        integer, intent(in) :: label, likelihood
        real(dp), intent(out) :: row_value, eta_gradient, thresholds_gradient(:)
        real(dp), intent(out) :: eta_hessian_dot, thresholds_hessian_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: probability, upper_cdf, lower_cdf, upper_pdf, lower_pdf
        real(dp) :: upper_pdf_prime, lower_pdf_prime, probability_dot
        real(dp) :: eta_first, upper_first, lower_first
        real(dp) :: eta_first_dot, upper_first_dot, lower_first_dot
        real(dp) :: upper_z_dot, lower_z_dot
        logical :: has_upper, has_lower

        row_value = 0.0_dp
        eta_gradient = 0.0_dp
        thresholds_gradient = 0.0_dp
        eta_hessian_dot = 0.0_dp
        thresholds_hessian_dot = 0.0_dp
        if (size(thresholds_dot) /= size(thresholds)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP likelihood HVP: threshold tangent shape is invalid")
            return
        end if
        has_lower = label > 1
        has_upper = label <= size(thresholds)
        if (has_upper) then
            call ordinal_cdf_pdf_derivative(thresholds(label) - eta, likelihood, upper_cdf, &
                upper_pdf, upper_pdf_prime)
            upper_z_dot = thresholds_dot(label) - eta_dot
        else
            upper_cdf = 1.0_dp
            upper_pdf = 0.0_dp
            upper_pdf_prime = 0.0_dp
            upper_z_dot = 0.0_dp
        end if
        if (has_lower) then
            call ordinal_cdf_pdf_derivative(thresholds(label - 1) - eta, likelihood, lower_cdf, &
                lower_pdf, lower_pdf_prime)
            lower_z_dot = thresholds_dot(label - 1) - eta_dot
        else
            lower_cdf = 0.0_dp
            lower_pdf = 0.0_dp
            lower_pdf_prime = 0.0_dp
            lower_z_dot = 0.0_dp
        end if
        probability = upper_cdf - lower_cdf
        if (.not. ieee_is_finite(probability) .or. probability <= tiny(1.0_dp)) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP likelihood HVP: category probability underflowed")
            return
        end if
        row_value = log(probability)
        probability_dot = upper_pdf*upper_z_dot - lower_pdf*lower_z_dot
        eta_first = (lower_pdf - upper_pdf)/probability
        upper_first = upper_pdf/probability
        lower_first = -lower_pdf/probability
        eta_first_dot = (lower_pdf_prime*lower_z_dot - upper_pdf_prime*upper_z_dot)/probability - &
            (lower_pdf - upper_pdf)*probability_dot/(probability*probability)
        upper_first_dot = upper_pdf_prime*upper_z_dot/probability - &
            upper_pdf*probability_dot/(probability*probability)
        lower_first_dot = -lower_pdf_prime*lower_z_dot/probability + &
            lower_pdf*probability_dot/(probability*probability)
        eta_gradient = eta_first
        if (has_upper) thresholds_gradient(label) = upper_first
        if (has_lower) thresholds_gradient(label - 1) = lower_first
        eta_hessian_dot = eta_first_dot
        if (has_upper) thresholds_hessian_dot(label) = upper_first_dot
        if (has_lower) thresholds_hessian_dot(label - 1) = lower_first_dot
        if (any(.not. ieee_is_finite([row_value, eta_gradient, eta_hessian_dot]))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP likelihood HVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_row_hessian_direction

    subroutine ordinal_cdf_pdf(z, likelihood, cdf, pdf)
        real(dp), intent(in) :: z
        integer, intent(in) :: likelihood
        real(dp), intent(out) :: cdf, pdf

        if (likelihood == GP_ORDINAL_LIKELIHOOD_LOGISTIC) then
            cdf = stable_ordinal_sigmoid(z)
            pdf = cdf*(1.0_dp - cdf)
        else
            cdf = 0.5_dp*erfc(-z/SQRT_TWO)
            cdf = min(1.0_dp, max(0.0_dp, cdf))
            pdf = exp(-0.5_dp*z*z)/SQRT_TWO_PI
        end if
    end subroutine ordinal_cdf_pdf

    subroutine ordinal_cdf_pdf_derivative(z, likelihood, cdf, pdf, pdf_prime)
        real(dp), intent(in) :: z
        integer, intent(in) :: likelihood
        real(dp), intent(out) :: cdf, pdf, pdf_prime

        call ordinal_cdf_pdf(z, likelihood, cdf, pdf)
        if (likelihood == GP_ORDINAL_LIKELIHOOD_LOGISTIC) then
            pdf_prime = pdf*(1.0_dp - 2.0_dp*cdf)
        else
            pdf_prime = -z*pdf
        end if
    end subroutine ordinal_cdf_pdf_derivative

    real(dp) function stable_ordinal_sigmoid(value) result(cdf)
        real(dp), intent(in) :: value
        real(dp) :: exponential

        if (value >= 0.0_dp) then
            cdf = 1.0_dp/(1.0_dp + exp(-value))
        else
            exponential = exp(value)
            cdf = exponential/(1.0_dp + exponential)
        end if
    end function stable_ordinal_sigmoid

    subroutine gp_ordinal_fit(self, x, labels, kernel, status, options, state)
        class(gp_ordinal_classification_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), intent(in) :: kernel
        type(fortnum_status_t), intent(out) :: status
        type(gp_ordinal_classification_options_t), intent(in), optional :: options
        type(gp_ordinal_classification_state_t), intent(out), optional :: state
        type(gp_ordinal_classification_options_t) :: requested
        type(gp_ordinal_classification_state_t) :: result
        real(dp), allocatable :: targets(:, :)
        integer, allocatable :: unique_labels(:)
        integer :: i, j
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(gp_ordinal_classification_options_t) :: gp_ordinal_classification_options_t_default
        type(gp_ordinal_classification_state_t) :: gp_ordinal_classification_state_t_default

        result = gp_ordinal_classification_state_t_default
        if (present(state)) state = result
        requested = gp_ordinal_classification_options_t_default
        if (present(options)) requested = options
        if (.not. valid_options(requested)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: options are invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: inputs must be finite")
            return
        end if
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: kernel dimension is invalid")
            return
        end if
        call sorted_unique(labels, unique_labels)
        if (size(unique_labels) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP fit: at least two ordered classes are required")
            return
        end if

        self%n_classes = size(unique_labels)
        self%n_features = size(x, 2)
        allocate(self%class_label(self%n_classes), self%cut_points(self%n_classes - 1))
        self%class_label = unique_labels
        do j = 1, self%n_classes - 1
            self%cut_points(j) = real(j, dp) + 0.5_dp
        end do
        allocate(targets(size(x, 1), 1))
        do i = 1, size(x, 1)
            j = index_of(unique_labels, labels(i))
            if (j < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP fit: class label mapping failed")
                return
            end if
            targets(i, 1) = real(j, dp)
        end do
        call self%latent%fit(x, targets, kernel, requested%noise_variance, status, &
            jitter=requested%jitter)
        if (status%code /= FORTNUM_OK) then
            self%is_fitted = .false.
            return
        end if
        self%is_fitted = .true.
        result%class_count = self%n_classes
        result%iterations = 1
        result%converged = .true.
        result%noise_variance = requested%noise_variance
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_fit

    subroutine gp_ordinal_predict_latent(self, x, mean, variance, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_mean(:, :)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP prediction: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(mean) /= size(x, 1) .or. &
            size(variance) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP prediction: output shape is invalid")
            return
        end if
        allocate(latent_mean(size(x, 1), 1))
        call self%latent%predict(x, latent_mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        mean = latent_mean(:, 1)
    end subroutine gp_ordinal_predict_latent

    subroutine gp_ordinal_predict_proba(self, x, probabilities, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability prediction: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability prediction: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), variance(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probabilities(mean, variance, self%cut_points, probabilities, status)
    end subroutine gp_ordinal_predict_proba

    !> Return stable natural logarithms of the ordered predictive probabilities.
    !! A finite floor protects the tails of the normal-CDF difference while
    !! preserving the probability output contract.  The floor is treated as
    !! a constant by the product routines below, so clipped tails have a zero
    !! derivative instead of an artificial `1/tiny` amplification.
    subroutine gp_ordinal_predict_log_proba(self, x, log_probabilities, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        log_probabilities = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability prediction: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(log_probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability prediction: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        log_probabilities = log(max(probabilities, tiny(1.0_dp)))
        if (any(.not. ieee_is_finite(log_probabilities))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP log probability prediction: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_log_proba

    subroutine gp_ordinal_predict(self, x, labels, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. self%is_fitted .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP label prediction: model or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict

    subroutine gp_ordinal_predict_latent_parameter_jvp(self, x, direction, mean, &
            mean_dot, variance, variance_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_mean(:, :), latent_mean_dot(:, :)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent parameter JVP: model is not fitted")
            return
        end if
        if (size(mean) /= size(x, 1) .or. size(mean_dot) /= size(x, 1) .or. &
            size(variance) /= size(x, 1) .or. size(variance_dot) /= size(x, 1) .or. &
            size(direction) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent parameter JVP: shape is invalid")
            return
        end if
        allocate(latent_mean(size(x, 1), 1), latent_mean_dot(size(x, 1), 1))
        call self%latent%predict_jvp(x, direction, latent_mean, latent_mean_dot, &
            variance, variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        mean = latent_mean(:, 1)
        mean_dot = latent_mean_dot(:, 1)
    end subroutine gp_ordinal_predict_latent_parameter_jvp

    subroutine gp_ordinal_predict_latent_parameter_vjp(self, x, mean_bar, variance_bar, &
            parameter_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean_bar_matrix(:, :)

        parameter_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent parameter VJP: model is not fitted")
            return
        end if
        if (size(mean_bar) /= size(x, 1) .or. size(variance_bar) /= size(x, 1) .or. &
            size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent parameter VJP: shape is invalid")
            return
        end if
        allocate(mean_bar_matrix(size(x, 1), 1))
        mean_bar_matrix(:, 1) = mean_bar
        call self%latent%predict_vjp(x, mean_bar_matrix, variance_bar, parameter_bar, status)
    end subroutine gp_ordinal_predict_latent_parameter_vjp

    subroutine gp_ordinal_predict_proba_parameter_jvp(self, x, direction, probabilities, &
            probabilities_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)

        if (any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability parameter JVP: output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_parameter_jvp(x, direction, mean, mean_dot, variance, &
            variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probabilities_jvp(mean, mean_dot, variance, variance_dot, &
            self%cut_points, probabilities, probabilities_dot, status)
    end subroutine gp_ordinal_predict_proba_parameter_jvp

    subroutine gp_ordinal_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            parameter_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)

        parameter_bar = 0.0_dp
        if (any(shape(probabilities_bar) /= [size(x, 1), self%n_classes]) .or. &
            size(parameter_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability parameter VJP: shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), variance(size(x, 1)), mean_bar(size(x, 1)), &
            variance_bar(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probability_vjp(mean, variance, self%cut_points, probabilities_bar, &
            mean_bar, variance_bar, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_latent_parameter_vjp(x, mean_bar, variance_bar, parameter_bar, status)
    end subroutine gp_ordinal_predict_proba_parameter_vjp

    !> Forward fixed-state parameter product of ordinal log probabilities.
    subroutine gp_ordinal_predict_log_proba_parameter_jvp(self, x, direction, &
            log_probabilities, log_probabilities_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)
        integer :: i, j

        log_probabilities = 0.0_dp
        log_probabilities_dot = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability parameter JVP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction)) .or. &
            any(shape(log_probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability parameter JVP: direction or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            probabilities_dot(size(x, 1), self%n_classes))
        call self%predict_proba_parameter_jvp(x, direction, probabilities, &
            probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                log_probabilities(i, j) = log(max(probabilities(i, j), tiny(1.0_dp)))
                if (probabilities(i, j) > tiny(1.0_dp)) then
                    log_probabilities_dot(i, j) = probabilities_dot(i, j) / probabilities(i, j)
                else
                    log_probabilities_dot(i, j) = 0.0_dp
                end if
            end do
        end do
        if (any(.not. ieee_is_finite(log_probabilities)) .or. &
            any(.not. ieee_is_finite(log_probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP log probability parameter JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_log_proba_parameter_jvp

    !> Reverse fixed-state parameter product of ordinal log probabilities.
    subroutine gp_ordinal_predict_log_proba_parameter_vjp(self, x, &
            log_probabilities_bar, parameter_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probability_bar(:, :)
        integer :: i, j

        parameter_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability parameter VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(parameter_bar) /= self%parameter_count() .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes]) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability parameter VJP: input or cotangent is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            probability_bar(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probability_bar = 0.0_dp
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                if (probabilities(i, j) > tiny(1.0_dp)) then
                    probability_bar(i, j) = log_probabilities_bar(i, j) / probabilities(i, j)
                end if
            end do
        end do
        call self%predict_proba_parameter_vjp(x, probability_bar, parameter_bar, status)
    end subroutine gp_ordinal_predict_log_proba_parameter_vjp

    subroutine gp_ordinal_predict_latent_input_jvp(self, x, x_dot, mean, mean_dot, &
            variance, variance_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: mean(:), mean_dot(:), variance(:), variance_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_mean(:, :), cross(:, :), cross_dot(:, :), work(:, :)
        real(dp), allocatable :: work_dot(:, :)
        real(dp) :: value, grad_x1(self%n_features), grad_x2(self%n_features)
        real(dp), allocatable :: hessian(:, :)
        real(dp) :: prior_dot
        integer :: i, j

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input JVP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. any(shape(x_dot) /= shape(x)) .or. &
            size(mean) /= size(x, 1) .or. size(mean_dot) /= size(x, 1) .or. &
            size(variance) /= size(x, 1) .or. size(variance_dot) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input JVP: shape is invalid")
            return
        end if
        allocate(latent_mean(size(x, 1), 1), cross(self%latent%n_samples, size(x, 1)), &
            cross_dot(self%latent%n_samples, size(x, 1)), work(self%latent%n_samples, size(x, 1)), &
            work_dot(self%latent%n_samples, size(x, 1)), hessian(self%n_features, self%n_features))
        call self%latent%predict(x, latent_mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call self%latent%kernel%matrix(self%latent%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        work = cross
        call self%latent%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        cross_dot = 0.0_dp
        variance_dot = 0.0_dp
        mean_dot = 0.0_dp
        do j = 1, size(x, 1)
            do i = 1, self%latent%n_samples
                call self%latent%kernel%input_derivatives(x(j, :), self%latent%x_train(i, :), &
                    value, grad_x1, grad_x2, hessian, status)
                if (status%code /= FORTNUM_OK) return
                cross_dot(i, j) = dot_product(grad_x1, x_dot(j, :))
                mean_dot(j) = mean_dot(j) + cross_dot(i, j)*self%latent%alpha(i, 1)
            end do
            call self%latent%kernel%input_derivatives(x(j, :), x(j, :), value, grad_x1, &
                grad_x2, hessian, status)
            if (status%code /= FORTNUM_OK) return
            prior_dot = dot_product(grad_x1 + grad_x2, x_dot(j, :))
            variance_dot(j) = prior_dot - sum(cross_dot(:, j)*work(:, j))
        end do
        work_dot = cross_dot
        call self%latent%factorization%solve(work_dot, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(x, 1)
            variance_dot(j) = variance_dot(j) - sum(cross(:, j)*work_dot(:, j))
        end do
        mean = latent_mean(:, 1)
        if (any(.not. ieee_is_finite(mean_dot)) .or. any(.not. ieee_is_finite(variance_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input JVP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_latent_input_jvp

    subroutine gp_ordinal_predict_latent_input_vjp(self, x, mean_bar, variance_bar, x_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), mean_bar(:), variance_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_mean(:, :), variance(:)
        real(dp), allocatable :: cross(:, :), work(:, :)
        real(dp) :: value, grad_x1(self%n_features), grad_x2(self%n_features)
        real(dp), allocatable :: hessian(:, :)
        real(dp) :: prior_bar
        integer :: i, j, k

        x_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(mean_bar) /= size(x, 1) .or. &
            size(variance_bar) /= size(x, 1) .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input VJP: shape is invalid")
            return
        end if
        allocate(latent_mean(size(x, 1), 1), variance(size(x, 1)), &
            cross(self%latent%n_samples, size(x, 1)), &
            work(self%latent%n_samples, size(x, 1)), hessian(self%n_features, self%n_features))
        call self%latent%predict(x, latent_mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call self%latent%kernel%matrix(self%latent%x_train, x, cross, status)
        if (status%code /= FORTNUM_OK) return
        work = cross
        call self%latent%factorization%solve(work, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(x, 1)
            do k = 1, self%n_features
                x_bar(j, k) = 0.0_dp
                do i = 1, self%latent%n_samples
                    call self%latent%kernel%input_derivatives(x(j, :), self%latent%x_train(i, :), &
                        value, grad_x1, grad_x2, hessian, status)
                    if (status%code /= FORTNUM_OK) return
                    x_bar(j, k) = x_bar(j, k) + (mean_bar(j)*self%latent%alpha(i, 1) - &
                        2.0_dp*variance_bar(j)*work(i, j))*grad_x1(k)
                end do
            end do
            call self%latent%kernel%input_derivatives(x(j, :), x(j, :), value, grad_x1, &
                grad_x2, hessian, status)
            if (status%code /= FORTNUM_OK) return
            prior_bar = variance_bar(j)
            do k = 1, self%n_features
                x_bar(j, k) = x_bar(j, k) + prior_bar*(grad_x1(k) + grad_x2(k))
            end do
        end do
        if (any(.not. ieee_is_finite(x_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP latent input VJP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_latent_input_vjp

    subroutine gp_ordinal_predict_proba_input_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), mean_dot(:), variance(:), variance_dot(:)

        allocate(mean(size(x, 1)), mean_dot(size(x, 1)), variance(size(x, 1)), &
            variance_dot(size(x, 1)))
        call self%predict_latent_input_jvp(x, x_dot, mean, mean_dot, variance, variance_dot, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probabilities_jvp(mean, mean_dot, variance, variance_dot, &
            self%cut_points, probabilities, probabilities_dot, status)
    end subroutine gp_ordinal_predict_proba_input_jvp

    subroutine gp_ordinal_predict_proba_input_vjp(self, x, probabilities_bar, x_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:), mean_bar(:), variance_bar(:)

        x_bar = 0.0_dp
        allocate(mean(size(x, 1)), variance(size(x, 1)), mean_bar(size(x, 1)), &
            variance_bar(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probability_vjp(mean, variance, self%cut_points, probabilities_bar, &
            mean_bar, variance_bar, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_latent_input_vjp(x, mean_bar, variance_bar, x_bar, status)
    end subroutine gp_ordinal_predict_proba_input_vjp

    !> Forward query-input product of ordinal log probabilities.
    subroutine gp_ordinal_predict_log_proba_input_jvp(self, x, x_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)
        integer :: i, j

        log_probabilities = 0.0_dp
        log_probabilities_dot = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability input JVP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. any(shape(x_dot) /= shape(x)) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. &
            any(shape(log_probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability input JVP: input or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            probabilities_dot(size(x, 1), self%n_classes))
        call self%predict_proba_input_jvp(x, x_dot, probabilities, probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                log_probabilities(i, j) = log(max(probabilities(i, j), tiny(1.0_dp)))
                if (probabilities(i, j) > tiny(1.0_dp)) then
                    log_probabilities_dot(i, j) = probabilities_dot(i, j) / probabilities(i, j)
                else
                    log_probabilities_dot(i, j) = 0.0_dp
                end if
            end do
        end do
        if (any(.not. ieee_is_finite(log_probabilities)) .or. &
            any(.not. ieee_is_finite(log_probabilities_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "ordinal GP log probability input JVP: result is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_log_proba_input_jvp

    !> Reverse query-input product of ordinal log probabilities.
    subroutine gp_ordinal_predict_log_proba_input_vjp(self, x, log_probabilities_bar, &
            x_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probability_bar(:, :)
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability input VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. any(shape(x_bar) /= shape(x)) .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes]) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability input VJP: input or cotangent is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            probability_bar(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probability_bar = 0.0_dp
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                if (probabilities(i, j) > tiny(1.0_dp)) then
                    probability_bar(i, j) = log_probabilities_bar(i, j) / probabilities(i, j)
                end if
            end do
        end do
        call self%predict_proba_input_vjp(x, probability_bar, x_bar, status)
    end subroutine gp_ordinal_predict_log_proba_input_vjp

    subroutine gp_ordinal_predict_proba_threshold_jvp(self, x, thresholds_dot, &
            probabilities, probabilities_dot, status)
        !! Differentiate predictive probabilities through the ordered cut points.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), thresholds_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. valid_threshold_product(self, x, thresholds_dot, probabilities, &
            probabilities_dot, status, "ordinal GP threshold JVP")) return
        allocate(mean(size(x, 1)), variance(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_probabilities(mean, variance, self%cut_points, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_threshold_jvp(mean, variance, self%cut_points, thresholds_dot, &
            probabilities_dot)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_proba_threshold_jvp

    subroutine gp_ordinal_predict_proba_threshold_vjp(self, x, probabilities_bar, &
            thresholds_bar, status)
        !! Accumulate the predictive-probability cotangent into cut points.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: thresholds_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean(:), variance(:)

        thresholds_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP threshold VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(probabilities_bar) /= [size(x, 1), self%n_classes]) .or. &
            size(thresholds_bar) /= self%n_classes - 1 .or. &
            any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP threshold VJP: input, cotangent, or output shape is invalid")
            return
        end if
        allocate(mean(size(x, 1)), variance(size(x, 1)))
        call self%predict_latent(x, mean, variance, status)
        if (status%code /= FORTNUM_OK) return
        call ordinal_threshold_vjp(mean, variance, self%cut_points, probabilities_bar, &
            thresholds_bar)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_proba_threshold_vjp

    subroutine gp_ordinal_predict_log_proba_threshold_jvp(self, x, thresholds_dot, &
            log_probabilities, log_probabilities_dot, status)
        !! Differentiate stable log probabilities through the cut points.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), thresholds_dot(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_dot(:, :)

        log_probabilities = 0.0_dp
        log_probabilities_dot = 0.0_dp
        allocate(probabilities(size(log_probabilities, 1), size(log_probabilities, 2)), &
            probabilities_dot(size(log_probabilities_dot, 1), &
            size(log_probabilities_dot, 2)))
        call self%predict_proba_threshold_jvp(x, thresholds_dot, probabilities, &
            probabilities_dot, status)
        if (status%code /= FORTNUM_OK) return
        log_probabilities = log(max(probabilities, tiny(1.0_dp)))
        where (probabilities > tiny(1.0_dp))
            log_probabilities_dot = probabilities_dot/probabilities
        elsewhere
            log_probabilities_dot = 0.0_dp
        end where
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_predict_log_proba_threshold_jvp

    subroutine gp_ordinal_predict_log_proba_threshold_vjp(self, x, &
            log_probabilities_bar, thresholds_bar, status)
        !! Accumulate a log-probability cotangent into cut points.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: thresholds_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), probabilities_bar(:, :)

        thresholds_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes]) .or. &
            size(thresholds_bar) /= self%n_classes - 1 .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log threshold VJP: input, cotangent, or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes), &
            probabilities_bar(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probabilities_bar = 0.0_dp
        where (probabilities > tiny(1.0_dp))
            probabilities_bar = log_probabilities_bar/probabilities
        end where
        call self%predict_proba_threshold_vjp(x, probabilities_bar, thresholds_bar, status)
    end subroutine gp_ordinal_predict_log_proba_threshold_vjp

    subroutine gp_ordinal_predict_proba_device(self, device, x, probabilities, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP device prediction: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP device prediction: device kind is invalid")
        end select
    end subroutine gp_ordinal_predict_proba_device

    !> Device-dispatched ordinal log-probability prediction.  CUDA remains a
    !! typed refusal until the covariance and normal-CDF graph is resident.
    subroutine gp_ordinal_predict_log_proba_device(self, device, x, log_probabilities, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        log_probabilities = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_log_proba(x, log_probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP log probability device: resident ordinal graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log probability device: device kind is invalid")
        end select
    end subroutine gp_ordinal_predict_log_proba_device

    subroutine gp_ordinal_predict_proba_parameter_vjp_device(self, device, x, &
            probabilities_bar, parameter_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP parameter VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_vjp(x, probabilities_bar, parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP parameter VJP device: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP parameter VJP device: device kind is invalid")
        end select
    end subroutine gp_ordinal_predict_proba_parameter_vjp_device

    subroutine gp_ordinal_predict_proba_input_vjp_device(self, device, x, &
            probabilities_bar, x_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        x_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP input VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_input_vjp(x, probabilities_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP input VJP device: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP input VJP device: device kind is invalid")
        end select
    end subroutine gp_ordinal_predict_proba_input_vjp_device

    subroutine gp_ordinal_predict_proba_threshold_jvp_device(self, device, x, &
            thresholds_dot, probabilities, probabilities_dot, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), thresholds_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        call ordinal_prediction_device_dispatch(device, status, &
            "ordinal GP threshold JVP device")
        if (status%code /= FORTNUM_OK) return
        call self%predict_proba_threshold_jvp(x, thresholds_dot, probabilities, &
            probabilities_dot, status)
    end subroutine gp_ordinal_predict_proba_threshold_jvp_device

    subroutine gp_ordinal_predict_proba_threshold_vjp_device(self, device, x, &
            probabilities_bar, thresholds_bar, status)
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: thresholds_bar(:)
        type(fortnum_status_t), intent(out) :: status

        thresholds_bar = 0.0_dp
        call ordinal_prediction_device_dispatch(device, status, &
            "ordinal GP threshold VJP device")
        if (status%code /= FORTNUM_OK) return
        call self%predict_proba_threshold_vjp(x, probabilities_bar, thresholds_bar, status)
    end subroutine gp_ordinal_predict_proba_threshold_vjp_device

    function gp_ordinal_classes(self) result(labels)
        class(gp_ordinal_classification_t), intent(in) :: self
        integer, allocatable :: labels(:)

        allocate(labels(self%n_classes))
        if (self%n_classes > 0) labels = self%class_label
    end function gp_ordinal_classes

    function gp_ordinal_thresholds(self) result(thresholds)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), allocatable :: thresholds(:)

        allocate(thresholds(max(0, self%n_classes - 1)))
        if (self%n_classes > 1) thresholds = self%cut_points
    end function gp_ordinal_thresholds

    subroutine gp_ordinal_set_thresholds(self, thresholds, status)
        !! Replace fitted cut points only after validating the complete vector.
        class(gp_ordinal_classification_t), intent(inout) :: self
        real(dp), intent(in) :: thresholds(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP set_thresholds: model is not fitted")
            return
        end if
        if (size(thresholds) /= self%n_classes - 1 .or. &
            any(.not. ieee_is_finite(thresholds))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP set_thresholds: threshold shape or values are invalid")
            return
        end if
        if (size(thresholds) > 1) then
            if (any(thresholds(2:) <= thresholds(:size(thresholds) - 1))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP set_thresholds: thresholds must be strictly increasing")
                return
            end if
        end if
        self%cut_points = thresholds
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_ordinal_set_thresholds

    integer function gp_ordinal_class_count(self) result(count)
        class(gp_ordinal_classification_t), intent(in) :: self

        count = self%n_classes
    end function gp_ordinal_class_count

    integer function gp_ordinal_feature_count(self) result(count)
        class(gp_ordinal_classification_t), intent(in) :: self

        count = self%n_features
    end function gp_ordinal_feature_count

    integer function gp_ordinal_parameter_count(self) result(count)
        class(gp_ordinal_classification_t), intent(in) :: self

        count = 0
        if (self%is_fitted) count = self%latent%parameter_count()
    end function gp_ordinal_parameter_count

    function gp_ordinal_parameters(self) result(parameters)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)

        if (self%is_fitted) then
            parameters = self%latent%parameters()
        else
            allocate(parameters(0))
        end if
    end function gp_ordinal_parameters

    subroutine gp_ordinal_set_parameters(self, parameters, status)
        class(gp_ordinal_classification_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP set_parameters: model is not fitted")
            return
        end if
        call self%latent%set_parameters(parameters, status)
    end subroutine gp_ordinal_set_parameters

    subroutine gp_ordinal_log_marginal_likelihood(self, value, status)
        !! Return the exact latent-Gaussian GP log marginal likelihood.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log marginal likelihood: model is not fitted")
            return
        end if
        call self%latent%log_marginal_likelihood(value, status)
    end subroutine gp_ordinal_log_marginal_likelihood

    subroutine gp_ordinal_log_marginal_likelihood_jvp(self, direction, value_dot, status)
        !! Directional product of the exact latent-Gaussian GP evidence.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: value_dot
        type(fortnum_status_t), intent(out) :: status

        value_dot = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP log marginal likelihood JVP: model is not fitted")
            return
        end if
        call self%latent%log_marginal_likelihood_jvp(direction, value_dot, status)
    end subroutine gp_ordinal_log_marginal_likelihood_jvp

    subroutine gp_ordinal_hyperparameter_gradient(self, gradient, status)
        !! Exact gradient over packed kernel and log-noise hyperparameters.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%hyperparameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter gradient: output shape is invalid")
            return
        end if
        call self%latent%hyperparameter_gradient(gradient, status)
    end subroutine gp_ordinal_hyperparameter_gradient

    subroutine gp_ordinal_hyperparameter_hvp(self, direction, parameter_hvp, status)
        !! Exact directional Hessian product over kernel and log-noise values.
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_hvp = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter HVP: model is not fitted")
            return
        end if
        if (size(direction) /= self%hyperparameter_count() .or. &
            size(parameter_hvp) /= self%hyperparameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter HVP: parameter shape is invalid")
            return
        end if
        call self%latent%hyperparameter_hvp(direction, parameter_hvp, status)
    end subroutine gp_ordinal_hyperparameter_hvp

    subroutine gp_ordinal_hyperparameter_gradient_device(self, device, gradient, status)
        !! Device-dispatch wrapper; CUDA remains an explicit typed refusal.
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter gradient device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hyperparameter_gradient(gradient, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP hyperparameter gradient device: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter gradient device: device kind is invalid")
        end select
    end subroutine gp_ordinal_hyperparameter_gradient_device

    subroutine gp_ordinal_hyperparameter_hvp_device(self, device, direction, parameter_hvp, status)
        !! Device-dispatch wrapper; CUDA remains an explicit typed refusal.
        class(gp_ordinal_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: parameter_hvp(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_hvp = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter HVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%hyperparameter_hvp(direction, parameter_hvp, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ordinal GP hyperparameter HVP device: resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP hyperparameter HVP device: device kind is invalid")
        end select
    end subroutine gp_ordinal_hyperparameter_hvp_device

    logical function gp_ordinal_fitted(self) result(value)
        class(gp_ordinal_classification_t), intent(in) :: self

        value = self%is_fitted
    end function gp_ordinal_fitted

    logical function gp_ordinal_device_supported(self, device_kind) result(value)
        class(gp_ordinal_classification_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            value = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            value = .false.
        case default
            value = .false.
        end select
    end function gp_ordinal_device_supported

    subroutine ordinal_probabilities(mean, variance, cut_points, probabilities, status)
        real(dp), intent(in) :: mean(:), variance(:), cut_points(:)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: scale, upper, lower
        integer :: i, j, n, k

        n = size(mean)
        k = size(cut_points) + 1
        if (size(variance) /= n .or. any(shape(probabilities) /= [n, k])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probabilities: shape is invalid")
            return
        end if
        do i = 1, n
            if (.not. ieee_is_finite(mean(i)) .or. .not. ieee_is_finite(variance(i)) .or. &
                variance(i) < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP probabilities: latent state is invalid")
                return
            end if
            scale = sqrt(max(1.0_dp + variance(i), MIN_SCALE))
            lower = 0.0_dp
            do j = 1, k
                if (j < k) then
                    upper = normal_cdf((cut_points(j) - mean(i))/scale)
                else
                    upper = 1.0_dp
                end if
                probabilities(i, j) = max(0.0_dp, upper - lower)
                lower = upper
            end do
            if (abs(sum(probabilities(i, :)) - 1.0_dp) > 2.0e-14_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal GP probabilities: normalization failed")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_probabilities

    subroutine ordinal_probabilities_jvp(mean, mean_dot, variance, variance_dot, cut_points, &
            probabilities, probabilities_dot, status)
        real(dp), intent(in) :: mean(:), mean_dot(:), variance(:), variance_dot(:), cut_points(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: scale, z, cdf, cdf_dot, lower, lower_dot
        integer :: i, j, n, k

        n = size(mean)
        k = size(cut_points) + 1
        call ordinal_probabilities(mean, variance, cut_points, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        if (size(mean_dot) /= n .or. size(variance_dot) /= n .or. &
            any(shape(probabilities_dot) /= [n, k])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability JVP: shape is invalid")
            return
        end if
        probabilities_dot = 0.0_dp
        do i = 1, n
            scale = sqrt(max(1.0_dp + variance(i), MIN_SCALE))
            lower = 0.0_dp
            lower_dot = 0.0_dp
            do j = 1, k
                if (j < k) then
                    z = (cut_points(j) - mean(i))/scale
                    cdf = normal_cdf(z)
                    cdf_dot = normal_pdf(z)*(-mean_dot(i)/scale - &
                        0.5_dp*(cut_points(j) - mean(i))*variance_dot(i)/(scale**3))
                else
                    cdf = 1.0_dp
                    cdf_dot = 0.0_dp
                end if
                probabilities_dot(i, j) = cdf_dot - lower_dot
                lower = cdf
                lower_dot = cdf_dot
            end do
        end do
        if (any(.not. ieee_is_finite(probabilities_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability JVP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_probabilities_jvp

    subroutine ordinal_probability_vjp(mean, variance, cut_points, probabilities_bar, &
            mean_bar, variance_bar, status)
        real(dp), intent(in) :: mean(:), variance(:), cut_points(:), probabilities_bar(:, :)
        real(dp), intent(out) :: mean_bar(:), variance_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: scale, z, density, boundary_bar
        integer :: i, j, n, k

        n = size(mean)
        k = size(cut_points) + 1
        mean_bar = 0.0_dp
        variance_bar = 0.0_dp
        if (size(variance) /= n .or. size(mean_bar) /= n .or. size(variance_bar) /= n .or. &
            any(shape(probabilities_bar) /= [n, k])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability VJP: shape is invalid")
            return
        end if
        do i = 1, n
            scale = sqrt(max(1.0_dp + variance(i), MIN_SCALE))
            do j = 1, k - 1
                z = (cut_points(j) - mean(i))/scale
                density = normal_pdf(z)
                boundary_bar = probabilities_bar(i, j) - probabilities_bar(i, j + 1)
                mean_bar(i) = mean_bar(i) - boundary_bar*density/scale
                variance_bar(i) = variance_bar(i) - boundary_bar*density* &
                    (cut_points(j) - mean(i))/(2.0_dp*scale**3)
            end do
        end do
        if (any(.not. ieee_is_finite(mean_bar)) .or. any(.not. ieee_is_finite(variance_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal GP probability VJP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_probability_vjp

    subroutine ordinal_threshold_jvp(mean, variance, cut_points, cut_points_dot, &
            probabilities_dot)
        real(dp), intent(in) :: mean(:), variance(:), cut_points(:), cut_points_dot(:)
        real(dp), intent(out) :: probabilities_dot(:, :)
        real(dp) :: boundary_dot, scale, z
        integer :: i, j

        probabilities_dot = 0.0_dp
        do i = 1, size(mean)
            scale = sqrt(max(1.0_dp + variance(i), MIN_SCALE))
            do j = 1, size(cut_points)
                z = (cut_points(j) - mean(i))/scale
                boundary_dot = normal_pdf(z)*cut_points_dot(j)/scale
                probabilities_dot(i, j) = probabilities_dot(i, j) + boundary_dot
                probabilities_dot(i, j + 1) = &
                    probabilities_dot(i, j + 1) - boundary_dot
            end do
        end do
    end subroutine ordinal_threshold_jvp

    subroutine ordinal_threshold_vjp(mean, variance, cut_points, probabilities_bar, &
            cut_points_bar)
        real(dp), intent(in) :: mean(:), variance(:), cut_points(:)
        real(dp), intent(in) :: probabilities_bar(:, :)
        real(dp), intent(out) :: cut_points_bar(:)
        real(dp) :: scale, z
        integer :: i, j

        cut_points_bar = 0.0_dp
        do i = 1, size(mean)
            scale = sqrt(max(1.0_dp + variance(i), MIN_SCALE))
            do j = 1, size(cut_points)
                z = (cut_points(j) - mean(i))/scale
                cut_points_bar(j) = cut_points_bar(j) + normal_pdf(z)* &
                    (probabilities_bar(i, j) - probabilities_bar(i, j + 1))/scale
            end do
        end do
    end subroutine ordinal_threshold_vjp

    logical function valid_threshold_product(self, x, thresholds_dot, probabilities, &
            probabilities_dot, status, operation) result(valid)
        class(gp_ordinal_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), thresholds_dot(:)
        real(dp), intent(in) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        valid = .false.
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            size(thresholds_dot) /= self%n_classes - 1 .or. &
            any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(thresholds_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": input, tangent, or output shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_threshold_product

    subroutine ordinal_prediction_device_dispatch(device, status, operation)
        type(fortml_device_t), intent(in) :: device
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call status_set(status, FORTNUM_OK, "")
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, trim(operation)// &
                ": resident ordinal kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": device kind is invalid")
        end select
    end subroutine ordinal_prediction_device_dispatch

    real(dp) function normal_cdf(value) result(output)
        real(dp), intent(in) :: value

        output = 0.5_dp*erfc(-value/SQRT_TWO)
    end function normal_cdf

    real(dp) function normal_pdf(value) result(output)
        real(dp), intent(in) :: value

        output = exp(-0.5_dp*value*value)/SQRT_TWO_PI
    end function normal_pdf

    logical function valid_options(options) result(value)
        type(gp_ordinal_classification_options_t), intent(in) :: options

        value = options%noise_variance > 0.0_dp .and. options%jitter > 0.0_dp
        if (value) value = ieee_is_finite(options%noise_variance) .and. &
            ieee_is_finite(options%jitter)
    end function valid_options

    subroutine sorted_unique(labels, unique_labels)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: unique_labels(:)
        integer, allocatable :: work(:)
        integer :: i, j, count, value

        allocate(work(size(labels)))
        work = labels
        do i = 2, size(work)
            value = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= value) exit
                work(j + 1) = work(j)
                j = j - 1
            end do
            work(j + 1) = value
        end do
        count = 0
        do i = 1, size(work)
            if (i == 1 .or. work(i) /= work(i - 1)) count = count + 1
        end do
        allocate(unique_labels(count))
        count = 0
        do i = 1, size(work)
            if (i == 1 .or. work(i) /= work(i - 1)) then
                count = count + 1
                unique_labels(count) = work(i)
            end if
        end do
    end subroutine sorted_unique

    integer function index_of(labels, value) result(index)
        integer, intent(in) :: labels(:), value
        integer :: i

        index = 0
        do i = 1, size(labels)
            if (labels(i) == value) then
                index = i
                return
            end if
        end do
    end function index_of

end module fortml_gp_ordinal_classification
