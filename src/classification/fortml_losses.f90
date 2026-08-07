module fortml_losses
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: stable_sigmoid
    public :: sigmoid_value, sigmoid_jvp, sigmoid_vjp
    public :: binary_cross_entropy_with_logits_value
    public :: binary_cross_entropy_with_logits_jvp
    public :: binary_cross_entropy_with_logits_vjp
    public :: binary_cross_entropy_with_logits_hvp
    public :: multilabel_binary_cross_entropy_with_logits_value
    public :: multilabel_binary_cross_entropy_with_logits_jvp
    public :: multilabel_binary_cross_entropy_with_logits_vjp
    public :: multilabel_binary_cross_entropy_with_logits_hvp
    public :: softmax_value, softmax_jvp, softmax_vjp
    public :: log_softmax_value, log_softmax_jvp, log_softmax_vjp
    public :: softmax_cross_entropy_value
    public :: softmax_cross_entropy_jvp
    public :: softmax_cross_entropy_vjp
    public :: softmax_cross_entropy_hvp
    public :: weighted_mse_loss_value, weighted_mse_loss_jvp
    public :: weighted_mse_loss_vjp, weighted_mse_loss_hvp
    public :: mae_loss_value, mae_loss_jvp, mae_loss_vjp
    public :: focal_binary_cross_entropy_with_logits_value
    public :: focal_binary_cross_entropy_with_logits_jvp
    public :: focal_binary_cross_entropy_with_logits_vjp
    public :: mean_absolute_error_loss_value, mean_absolute_error_loss_jvp
    public :: mean_absolute_error_loss_vjp
    public :: binary_focal_cross_entropy_with_logits_value
    public :: binary_focal_cross_entropy_with_logits_jvp
    public :: binary_focal_cross_entropy_with_logits_vjp
    public :: gaussian_nll_value, gaussian_nll_jvp, gaussian_nll_vjp
    public :: gaussian_nll_hvp
    public :: poisson_nll_value, poisson_nll_jvp, poisson_nll_vjp
    public :: poisson_nll_hvp
    public :: poisson_count_nll_value, poisson_count_nll_jvp
    public :: poisson_count_nll_vjp, poisson_count_nll_hvp
    public :: huber_loss_value, huber_loss_jvp, huber_loss_vjp
    public :: huber_loss_hvp
    public :: quantile_loss_value, quantile_loss_jvp, quantile_loss_vjp
    public :: ordinal_cumulative_logit_loss_value
    public :: ordinal_cumulative_logit_loss_jvp
    public :: ordinal_cumulative_logit_loss_vjp
    public :: ordinal_cumulative_logit_loss_hvp

    integer, parameter, public :: LOSS_REDUCTION_MEAN = 1
    integer, parameter, public :: LOSS_REDUCTION_SUM = 2

    interface mean_absolute_error_loss_value
        module procedure mae_loss_value
    end interface mean_absolute_error_loss_value

    interface mean_absolute_error_loss_jvp
        module procedure mae_loss_jvp
    end interface mean_absolute_error_loss_jvp

    interface mean_absolute_error_loss_vjp
        module procedure mae_loss_vjp
    end interface mean_absolute_error_loss_vjp

    interface binary_focal_cross_entropy_with_logits_value
        module procedure focal_binary_cross_entropy_with_logits_value
    end interface binary_focal_cross_entropy_with_logits_value

    interface binary_focal_cross_entropy_with_logits_jvp
        module procedure focal_binary_cross_entropy_with_logits_jvp
    end interface binary_focal_cross_entropy_with_logits_jvp

    interface binary_focal_cross_entropy_with_logits_vjp
        module procedure focal_binary_cross_entropy_with_logits_vjp
    end interface binary_focal_cross_entropy_with_logits_vjp

    interface poisson_count_nll_value
        module procedure poisson_nll_value
    end interface poisson_count_nll_value

    interface poisson_count_nll_jvp
        module procedure poisson_nll_jvp
    end interface poisson_count_nll_jvp

    interface poisson_count_nll_vjp
        module procedure poisson_nll_vjp
    end interface poisson_count_nll_vjp

    interface poisson_count_nll_hvp
        module procedure poisson_nll_hvp
    end interface poisson_count_nll_hvp

contains

    elemental pure real(dp) function stable_sigmoid(logit) result(probability)
        real(dp), intent(in) :: logit
        real(dp) :: exponential

        if (logit >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-logit))
        else
            exponential = exp(logit)
            probability = exponential/(1.0_dp + exponential)
        end if
    end function stable_sigmoid

    subroutine sigmoid_value(logits, probabilities, status)
        real(dp), intent(in) :: logits(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        call validate_elementwise(logits, probabilities, "sigmoid", status)
        if (status%code /= FORTNUM_OK) return
        probabilities = stable_sigmoid(logits)
        call status_set(status, FORTNUM_OK, "")
    end subroutine sigmoid_value

    subroutine sigmoid_jvp(logits, logits_dot, probabilities, &
            probabilities_dot, status)
        real(dp), intent(in) :: logits(:, :), logits_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(shape(probabilities_dot) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sigmoid JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sigmoid JVP: tangent must be finite")
            return
        end if
        call sigmoid_value(logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probabilities_dot = probabilities*(1.0_dp - probabilities)*logits_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine sigmoid_jvp

    subroutine sigmoid_vjp(logits, probabilities_bar, logits_bar, status)
        real(dp), intent(in) :: logits(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: logits_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        logits_bar = 0.0_dp
        if (any(shape(probabilities_bar) /= shape(logits)) .or. &
            any(shape(logits_bar) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sigmoid VJP: cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sigmoid VJP: cotangent must be finite")
            return
        end if
        allocate(probabilities(size(logits, 1), size(logits, 2)))
        call sigmoid_value(logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        logits_bar = probabilities_bar*probabilities*(1.0_dp - probabilities)
        call status_set(status, FORTNUM_OK, "")
    end subroutine sigmoid_vjp

    subroutine binary_cross_entropy_with_logits_value(logits, targets, value, &
            status)
        real(dp), intent(in) :: logits(:, :), targets(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        value = 0.0_dp
        call validate_binary_inputs(logits, targets, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                value = value + binary_loss(logits(i, j), targets(i, j))
            end do
        end do
        value = value/real(size(logits), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine binary_cross_entropy_with_logits_value

    subroutine binary_cross_entropy_with_logits_jvp(logits, targets, logits_dot, &
            value, value_dot, status)
        real(dp), intent(in) :: logits(:, :), targets(:, :), logits_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy JVP: tangent must be finite")
            return
        end if
        call binary_cross_entropy_with_logits_value(logits, targets, value, status)
        if (status%code /= FORTNUM_OK) return
        value_dot = sum((stable_sigmoid(logits) - targets)*logits_dot)/ &
            real(size(logits), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine binary_cross_entropy_with_logits_jvp

    subroutine binary_cross_entropy_with_logits_vjp(logits, targets, value_bar, &
            logits_bar, status)
        real(dp), intent(in) :: logits(:, :), targets(:, :), value_bar
        real(dp), intent(out) :: logits_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        logits_bar = 0.0_dp
        if (any(shape(logits_bar) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy VJP: output shape is invalid")
            return
        end if
        if (.not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy VJP: cotangent must be finite")
            return
        end if
        call validate_binary_inputs(logits, targets, status)
        if (status%code /= FORTNUM_OK) return
        logits_bar = value_bar*(stable_sigmoid(logits) - targets)/ &
            real(size(logits), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine binary_cross_entropy_with_logits_vjp

    subroutine binary_cross_entropy_with_logits_hvp(logits, targets, logits_dot, &
            logits_hvp, status)
        !! Hessian-vector product of the mean BCE-with-logits objective.
        !!
        !! The target is constant and the product is with respect to logits:
        !! `diag(sigmoid(logit)*(1-sigmoid(logit)))/N * logits_dot`.
        real(dp), intent(in) :: logits(:, :), targets(:, :), logits_dot(:, :)
        real(dp), intent(out) :: logits_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        logits_hvp = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(shape(logits_hvp) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy HVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy HVP: tangent must be finite")
            return
        end if
        allocate(probabilities(size(logits, 1), size(logits, 2)))
        call sigmoid_value(logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call validate_binary_inputs(logits, targets, status)
        if (status%code /= FORTNUM_OK) return
        logits_hvp = probabilities*(1.0_dp - probabilities)*logits_dot / &
            real(size(logits), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine binary_cross_entropy_with_logits_hvp

    subroutine multilabel_binary_cross_entropy_with_logits_value(logits, targets, &
            value, status, sample_weight, reduction)
        !! Weighted multilabel BCE-with-logits with row-wise reductions.
        !!
        !! Each column is an independent relaxed indicator target.  Mean
        !! reduction divides by positive row-weight mass, while sum reduction
        !! leaves the weighted sum unnormalised.  This convention lets a
        !! multilabel head and a multi-output regression head share one batch
        !! weighting contract without silently changing the number of labels.
        real(dp), intent(in) :: logits(:, :), targets(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization
        integer :: i, j

        value = 0.0_dp
        call validate_binary_inputs(logits, targets, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                value = value + weights(i)*binary_loss(logits(i, j), targets(i, j))
            end do
        end do
        value = value/normalization
        if (.not. ieee_is_finite(value)) then
            value = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel BCE: reduction is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_binary_cross_entropy_with_logits_value

    subroutine multilabel_binary_cross_entropy_with_logits_jvp(logits, targets, &
            logits_dot, value, value_dot, status, sample_weight, reduction)
        !! Value and exact logits JVP of weighted multilabel BCE.
        real(dp), intent(in) :: logits(:, :), targets(:, :), logits_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, derivative
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel BCE JVP: tangent shape or values are invalid")
            return
        end if
        call validate_binary_inputs(logits, targets, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                derivative = stable_sigmoid(logits(i, j)) - targets(i, j)
                value = value + weights(i)*binary_loss(logits(i, j), targets(i, j))
                value_dot = value_dot + weights(i)*derivative*logits_dot(i, j)
            end do
        end do
        value = value/normalization
        value_dot = value_dot/normalization
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(value_dot)) then
            value = 0.0_dp
            value_dot = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel BCE JVP: reduction is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_binary_cross_entropy_with_logits_jvp

    subroutine multilabel_binary_cross_entropy_with_logits_vjp(logits, targets, &
            value_bar, logits_bar, status, sample_weight, reduction)
        !! Exact reverse product of weighted multilabel BCE with respect to logits.
        real(dp), intent(in) :: logits(:, :), targets(:, :), value_bar
        real(dp), intent(out) :: logits_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization
        integer :: i, j

        logits_bar = 0.0_dp
        if (any(shape(logits_bar) /= shape(logits)) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel BCE VJP: cotangent or output shape is invalid")
            return
        end if
        call validate_binary_inputs(logits, targets, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                logits_bar(i, j) = value_bar*weights(i)* &
                    (stable_sigmoid(logits(i, j)) - targets(i, j))/normalization
            end do
        end do
        if (any(.not. ieee_is_finite(logits_bar))) then
            logits_bar = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel BCE VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_binary_cross_entropy_with_logits_vjp

    subroutine multilabel_binary_cross_entropy_with_logits_hvp(logits, targets, &
            logits_dot, logits_hvp, status, sample_weight, reduction)
        !! Exact logits Hessian-vector product of weighted multilabel BCE.
        real(dp), intent(in) :: logits(:, :), targets(:, :), logits_dot(:, :)
        real(dp), intent(out) :: logits_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, probability
        integer :: i, j

        logits_hvp = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(shape(logits_hvp) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel BCE HVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel BCE HVP: tangent must be finite")
            return
        end if
        call validate_binary_inputs(logits, targets, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                probability = stable_sigmoid(logits(i, j))
                logits_hvp(i, j) = weights(i)*probability*(1.0_dp - probability)* &
                    logits_dot(i, j)/normalization
            end do
        end do
        if (any(.not. ieee_is_finite(logits_hvp))) then
            logits_hvp = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multilabel BCE HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multilabel_binary_cross_entropy_with_logits_hvp

    subroutine ordinal_cumulative_logit_loss_value(logits, labels, value, status, &
            sample_weight, reduction)
        !! Weighted negative log likelihood for a cumulative-logit ordinal head.
        !!
        !! `logits(i,k)` are ordered cumulative logits and encode
        !! `P(Y<=k)=sigmoid(logits(i,k))`; labels are one-based class indices
        !! in `1:size(logits,2)+1`.  Every row must have strictly increasing
        !! cumulative logits so all class probabilities are positive.
        real(dp), intent(in) :: logits(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:), dpdz(:), d2pdz2(:)
        real(dp) :: normalization, probability
        integer :: i

        value = 0.0_dp
        call validate_ordinal_inputs(logits, labels, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        allocate(dpdz(size(logits, 2)), d2pdz2(size(logits, 2)))
        do i = 1, size(logits, 1)
            call ordinal_probability_terms(logits(i, :), labels(i), probability, &
                dpdz, d2pdz2, status)
            if (status%code /= FORTNUM_OK) then
                value = 0.0_dp
                return
            end if
            value = value - weights(i)*log(probability)
        end do
        value = value/normalization
        if (.not. ieee_is_finite(value)) then
            value = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit loss: reduction is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_cumulative_logit_loss_value

    subroutine ordinal_cumulative_logit_loss_jvp(logits, labels, logits_dot, value, &
            value_dot, status, sample_weight, reduction)
        !! Value and exact cumulative-logit JVP of the ordinal loss.
        real(dp), intent(in) :: logits(:, :), logits_dot(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:), dpdz(:), d2pdz2(:)
        real(dp) :: normalization, probability, probability_dot
        integer :: i

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit JVP: tangent shape or values are invalid")
            return
        end if
        call validate_ordinal_inputs(logits, labels, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        allocate(dpdz(size(logits, 2)), d2pdz2(size(logits, 2)))
        do i = 1, size(logits, 1)
            call ordinal_probability_terms(logits(i, :), labels(i), probability, &
                dpdz, d2pdz2, status)
            if (status%code /= FORTNUM_OK) then
                value = 0.0_dp
                value_dot = 0.0_dp
                return
            end if
            probability_dot = dot_product(dpdz, logits_dot(i, :))
            value = value - weights(i)*log(probability)
            value_dot = value_dot - weights(i)*probability_dot/probability
        end do
        value = value/normalization
        value_dot = value_dot/normalization
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(value_dot)) then
            value = 0.0_dp
            value_dot = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit JVP: reduction is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_cumulative_logit_loss_jvp

    subroutine ordinal_cumulative_logit_loss_vjp(logits, labels, value_bar, logits_bar, &
            status, sample_weight, reduction)
        !! Exact reverse product of the weighted ordinal cumulative-logit loss.
        real(dp), intent(in) :: logits(:, :), value_bar
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: logits_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:), dpdz(:), d2pdz2(:)
        real(dp) :: normalization, probability
        integer :: i

        logits_bar = 0.0_dp
        if (any(shape(logits_bar) /= shape(logits)) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit VJP: cotangent or output shape is invalid")
            return
        end if
        call validate_ordinal_inputs(logits, labels, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        allocate(dpdz(size(logits, 2)), d2pdz2(size(logits, 2)))
        do i = 1, size(logits, 1)
            call ordinal_probability_terms(logits(i, :), labels(i), probability, &
                dpdz, d2pdz2, status)
            if (status%code /= FORTNUM_OK) then
                logits_bar = 0.0_dp
                return
            end if
            logits_bar(i, :) = -value_bar*weights(i)*dpdz/ &
                (normalization*probability)
        end do
        if (any(.not. ieee_is_finite(logits_bar))) then
            logits_bar = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_cumulative_logit_loss_vjp

    subroutine ordinal_cumulative_logit_loss_hvp(logits, labels, logits_dot, &
            logits_hvp, status, sample_weight, reduction)
        !! Exact cumulative-logit Hessian-vector product of ordinal loss.
        !!
        !! The class probability is a difference of at most two independent
        !! sigmoid terms.  This makes the diagonal second derivative of the
        !! probability sufficient for an exact Hessian-vector product; no
        !! finite-difference approximation is used.
        real(dp), intent(in) :: logits(:, :), logits_dot(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: logits_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:), dpdz(:), d2pdz2(:)
        real(dp) :: normalization, probability, probability_dot
        integer :: i

        logits_hvp = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(shape(logits_hvp) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit HVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit HVP: tangent must be finite")
            return
        end if
        call validate_ordinal_inputs(logits, labels, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        allocate(dpdz(size(logits, 2)), d2pdz2(size(logits, 2)))
        do i = 1, size(logits, 1)
            call ordinal_probability_terms(logits(i, :), labels(i), probability, &
                dpdz, d2pdz2, status)
            if (status%code /= FORTNUM_OK) then
                logits_hvp = 0.0_dp
                return
            end if
            probability_dot = dot_product(dpdz, logits_dot(i, :))
            logits_hvp(i, :) = -weights(i)/normalization * &
                (d2pdz2*logits_dot(i, :)/probability - &
                dpdz*probability_dot/(probability*probability))
        end do
        if (any(.not. ieee_is_finite(logits_hvp))) then
            logits_hvp = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_cumulative_logit_loss_hvp

    subroutine softmax_value(logits, probabilities, status)
        real(dp), intent(in) :: logits(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: maximum, normalizer
        integer :: i, j

        probabilities = 0.0_dp
        call validate_elementwise(logits, probabilities, "softmax", status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            maximum = maxval(logits(i, :))
            normalizer = 0.0_dp
            do j = 1, size(logits, 2)
                probabilities(i, j) = exp(logits(i, j) - maximum)
                normalizer = normalizer + probabilities(i, j)
            end do
            probabilities(i, :) = probabilities(i, :)/normalizer
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_value

    subroutine softmax_jvp(logits, logits_dot, probabilities, probabilities_dot, &
            status)
        real(dp), intent(in) :: logits(:, :), logits_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: row_dot
        integer :: i

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(shape(probabilities_dot) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax JVP: tangent must be finite")
            return
        end if
        call softmax_value(logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            row_dot = dot_product(probabilities(i, :), logits_dot(i, :))
            probabilities_dot(i, :) = probabilities(i, :)* &
                (logits_dot(i, :) - row_dot)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_jvp

    subroutine softmax_vjp(logits, probabilities_bar, logits_bar, status)
        real(dp), intent(in) :: logits(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: logits_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        real(dp) :: row_bar
        integer :: i

        logits_bar = 0.0_dp
        if (any(shape(probabilities_bar) /= shape(logits)) .or. &
            any(shape(logits_bar) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax VJP: cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax VJP: cotangent must be finite")
            return
        end if
        allocate(probabilities(size(logits, 1), size(logits, 2)))
        call softmax_value(logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            row_bar = dot_product(probabilities(i, :), probabilities_bar(i, :))
            logits_bar(i, :) = probabilities(i, :)* &
                (probabilities_bar(i, :) - row_bar)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_vjp

    subroutine log_softmax_value(logits, log_probabilities, status)
        real(dp), intent(in) :: logits(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_normalizer, maximum, normalizer
        integer :: i, j

        log_probabilities = 0.0_dp
        call validate_elementwise(logits, log_probabilities, "log softmax", status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            maximum = maxval(logits(i, :))
            normalizer = 0.0_dp
            do j = 1, size(logits, 2)
                normalizer = normalizer + exp(logits(i, j) - maximum)
            end do
            log_normalizer = log(normalizer)
            do j = 1, size(logits, 2)
                log_probabilities(i, j) = logits(i, j) - maximum - log_normalizer
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine log_softmax_value

    subroutine log_softmax_jvp(logits, logits_dot, log_probabilities, &
            log_probabilities_dot, status)
        real(dp), intent(in) :: logits(:, :), logits_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        real(dp), intent(out) :: log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: row_dot
        integer :: i

        log_probabilities = 0.0_dp
        log_probabilities_dot = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(shape(log_probabilities_dot) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "log softmax JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "log softmax JVP: tangent must be finite")
            return
        end if
        call log_softmax_value(logits, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            row_dot = dot_product(exp(log_probabilities(i, :)), logits_dot(i, :))
            log_probabilities_dot(i, :) = logits_dot(i, :) - row_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine log_softmax_jvp

    subroutine log_softmax_vjp(logits, log_probabilities_bar, logits_bar, status)
        real(dp), intent(in) :: logits(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: logits_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :)
        real(dp) :: row_bar
        integer :: i

        logits_bar = 0.0_dp
        if (any(shape(log_probabilities_bar) /= shape(logits)) .or. &
            any(shape(logits_bar) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "log softmax VJP: cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "log softmax VJP: cotangent must be finite")
            return
        end if
        allocate(log_probabilities(size(logits, 1), size(logits, 2)))
        call log_softmax_value(logits, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            row_bar = sum(log_probabilities_bar(i, :))
            logits_bar(i, :) = log_probabilities_bar(i, :) - &
                exp(log_probabilities(i, :))*row_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine log_softmax_vjp

    subroutine softmax_cross_entropy_value(logits, labels, value, status)
        real(dp), intent(in) :: logits(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: maximum, normalizer
        integer :: i, j

        value = 0.0_dp
        call validate_categorical_inputs(logits, labels, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            maximum = maxval(logits(i, :))
            normalizer = 0.0_dp
            do j = 1, size(logits, 2)
                normalizer = normalizer + exp(logits(i, j) - maximum)
            end do
            value = value + maximum - logits(i, labels(i)) + log(normalizer)
        end do
        value = value/real(size(logits, 1), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_cross_entropy_value

    subroutine softmax_cross_entropy_jvp(logits, labels, logits_dot, value, &
            value_dot, status)
        real(dp), intent(in) :: logits(:, :), logits_dot(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: maximum, normalizer, weighted_dot
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy JVP: tangent must be finite")
            return
        end if
        call softmax_cross_entropy_value(logits, labels, value, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            maximum = maxval(logits(i, :))
            normalizer = 0.0_dp
            weighted_dot = 0.0_dp
            do j = 1, size(logits, 2)
                normalizer = normalizer + exp(logits(i, j) - maximum)
                weighted_dot = weighted_dot + &
                    exp(logits(i, j) - maximum)*logits_dot(i, j)
            end do
            value_dot = value_dot + weighted_dot/normalizer - &
                logits_dot(i, labels(i))
        end do
        value_dot = value_dot/real(size(logits, 1), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_cross_entropy_jvp

    subroutine softmax_cross_entropy_vjp(logits, labels, value_bar, logits_bar, &
            status)
        real(dp), intent(in) :: logits(:, :), value_bar
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: logits_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: maximum, normalizer
        integer :: i, j

        logits_bar = 0.0_dp
        if (any(shape(logits_bar) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy VJP: output shape is invalid")
            return
        end if
        if (.not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy VJP: cotangent must be finite")
            return
        end if
        call validate_categorical_inputs(logits, labels, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            maximum = maxval(logits(i, :))
            normalizer = 0.0_dp
            do j = 1, size(logits, 2)
                logits_bar(i, j) = exp(logits(i, j) - maximum)
                normalizer = normalizer + logits_bar(i, j)
            end do
            logits_bar(i, :) = logits_bar(i, :)/normalizer
            logits_bar(i, labels(i)) = logits_bar(i, labels(i)) - 1.0_dp
        end do
        logits_bar = value_bar*logits_bar/real(size(logits, 1), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_cross_entropy_vjp

    subroutine softmax_cross_entropy_hvp(logits, labels, logits_dot, logits_hvp, &
            status)
        !! Hessian-vector product of the mean softmax cross-entropy objective.
        !!
        !! For each sample the product is `J_softmax * logits_dot`; the
        !! one-hot label contributes no second derivative.  This is analytic
        !! and remains stable because the probabilities are formed after
        !! subtracting the row maximum.
        real(dp), intent(in) :: logits(:, :), logits_dot(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: logits_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        real(dp) :: row_dot
        integer :: i

        logits_hvp = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(shape(logits_hvp) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy HVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy HVP: tangent must be finite")
            return
        end if
        call validate_categorical_inputs(logits, labels, status)
        if (status%code /= FORTNUM_OK) return
        allocate(probabilities(size(logits, 1), size(logits, 2)))
        call softmax_value(logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            row_dot = dot_product(probabilities(i, :), logits_dot(i, :))
            logits_hvp(i, :) = probabilities(i, :)* &
                (logits_dot(i, :) - row_dot)
        end do
        logits_hvp = logits_hvp/real(size(logits, 1), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_cross_entropy_hvp

    subroutine weighted_mse_loss_value(prediction, targets, sample_weight, value, &
            status, reduction)
        !! Weighted mean/sum squared loss with a reusable reduction contract.
        !!
        !! The per-element loss is `0.5*(prediction-targets)**2`.  Mean
        !! reduction divides by the positive sample-weight mass (not by the
        !! number of elements); sum reduction leaves the weighted sum
        !! unnormalised.  A zero-support weight vector is always refused.
        real(dp), intent(in) :: prediction(:, :), targets(:, :)
        real(dp), intent(in) :: sample_weight(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: reduction
        integer :: reduction_kind, i
        real(dp) :: normalization

        value = 0.0_dp
        call validate_weighted_mse_inputs(prediction, targets, sample_weight, &
            reduction, status, reduction_kind)
        if (status%code /= FORTNUM_OK) return
        normalization = weighted_mse_normalization(sample_weight, reduction_kind)
        do i = 1, size(prediction, 1)
            value = value + 0.5_dp*sample_weight(i)*sum( &
                (prediction(i, :) - targets(i, :))**2)
        end do
        value = value/normalization
        call status_set(status, FORTNUM_OK, "")
    end subroutine weighted_mse_loss_value

    subroutine weighted_mse_loss_jvp(prediction, targets, sample_weight, &
            prediction_dot, value, value_dot, status, reduction)
        !! Scalar value and JVP of `weighted_mse_loss_value`.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), sample_weight(:)
        real(dp), intent(in) :: prediction_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: reduction
        integer :: reduction_kind, i
        real(dp) :: normalization

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(prediction_dot) /= shape(prediction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "weighted MSE JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(prediction_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "weighted MSE JVP: tangent must be finite")
            return
        end if
        call validate_weighted_mse_inputs(prediction, targets, sample_weight, &
            reduction, status, reduction_kind)
        if (status%code /= FORTNUM_OK) return
        normalization = weighted_mse_normalization(sample_weight, reduction_kind)
        do i = 1, size(prediction, 1)
            value = value + 0.5_dp*sample_weight(i)*sum( &
                (prediction(i, :) - targets(i, :))**2)
            value_dot = value_dot + sample_weight(i)*dot_product( &
                prediction(i, :) - targets(i, :), prediction_dot(i, :))
        end do
        value = value/normalization
        value_dot = value_dot/normalization
        call status_set(status, FORTNUM_OK, "")
    end subroutine weighted_mse_loss_jvp

    subroutine weighted_mse_loss_vjp(prediction, targets, sample_weight, &
            value_bar, prediction_bar, status, reduction)
        !! VJP of `weighted_mse_loss_value` with respect to predictions.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), sample_weight(:)
        real(dp), intent(in) :: value_bar
        real(dp), intent(out) :: prediction_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: reduction
        integer :: reduction_kind, i
        real(dp) :: normalization

        prediction_bar = 0.0_dp
        if (any(shape(prediction_bar) /= shape(prediction)) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "weighted MSE VJP: cotangent or output shape is invalid")
            return
        end if
        call validate_weighted_mse_inputs(prediction, targets, sample_weight, &
            reduction, status, reduction_kind)
        if (status%code /= FORTNUM_OK) return
        normalization = weighted_mse_normalization(sample_weight, reduction_kind)
        do i = 1, size(prediction, 1)
            prediction_bar(i, :) = value_bar*sample_weight(i)* &
                (prediction(i, :) - targets(i, :))/normalization
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine weighted_mse_loss_vjp

    subroutine weighted_mse_loss_hvp(prediction, targets, sample_weight, &
            prediction_dot, prediction_hvp, status, reduction)
        !! Hessian-vector product of the weighted squared loss.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), sample_weight(:)
        real(dp), intent(in) :: prediction_dot(:, :)
        real(dp), intent(out) :: prediction_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: reduction
        integer :: reduction_kind, i
        real(dp) :: normalization

        prediction_hvp = 0.0_dp
        if (any(shape(prediction_dot) /= shape(prediction)) .or. &
            any(shape(prediction_hvp) /= shape(prediction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "weighted MSE HVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(prediction_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "weighted MSE HVP: tangent must be finite")
            return
        end if
        call validate_weighted_mse_inputs(prediction, targets, sample_weight, &
            reduction, status, reduction_kind)
        if (status%code /= FORTNUM_OK) return
        normalization = weighted_mse_normalization(sample_weight, reduction_kind)
        do i = 1, size(prediction, 1)
            prediction_hvp(i, :) = sample_weight(i)*prediction_dot(i, :)/ &
                normalization
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine weighted_mse_loss_hvp

    subroutine mae_loss_value(prediction, targets, value, status, sample_weight, &
            reduction)
        !! Weighted mean/sum mean-absolute-error value.
        !!
        !! The optional row weights follow the same positive-mass reduction
        !! contract as weighted MSE.  Targets are constants; derivative
        !! products deliberately refuse an exact zero residual.
        real(dp), intent(in) :: prediction(:, :), targets(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization
        integer :: i

        value = 0.0_dp
        call validate_loss_matrix_inputs(prediction, targets, "MAE", status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(prediction, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(prediction, 1)
            value = value + weights(i)*sum(abs(prediction(i, :) - targets(i, :)))
        end do
        value = value/normalization
        call status_set(status, FORTNUM_OK, "")
    end subroutine mae_loss_value

    subroutine mae_loss_jvp(prediction, targets, prediction_dot, value, value_dot, &
            status, sample_weight, reduction)
        !! Value/JVP of weighted MAE.  An exact zero residual is a true kink,
        !! so the routine returns a domain refusal rather than choosing a
        !! subgradient.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), prediction_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, residual
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(prediction_dot) /= shape(prediction)) .or. &
            any(.not. ieee_is_finite(prediction_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MAE JVP: tangent shape or values are invalid")
            return
        end if
        call validate_loss_matrix_inputs(prediction, targets, "MAE", status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(prediction, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                if (residual == 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "MAE JVP: derivative is undefined at zero residual")
                    value = 0.0_dp
                    value_dot = 0.0_dp
                    return
                end if
                value = value + weights(i)*abs(residual)
                value_dot = value_dot + weights(i)*sign(1.0_dp, residual)* &
                    prediction_dot(i, j)
            end do
        end do
        value = value/normalization
        value_dot = value_dot/normalization
        call status_set(status, FORTNUM_OK, "")
    end subroutine mae_loss_jvp

    subroutine mae_loss_vjp(prediction, targets, value_bar, prediction_bar, status, &
            sample_weight, reduction)
        !! VJP of weighted MAE with respect to predictions.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), value_bar
        real(dp), intent(out) :: prediction_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, residual
        integer :: i, j

        prediction_bar = 0.0_dp
        if (any(shape(prediction_bar) /= shape(prediction)) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MAE VJP: cotangent or output shape is invalid")
            return
        end if
        call validate_loss_matrix_inputs(prediction, targets, "MAE", status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(prediction, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                if (residual == 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "MAE VJP: derivative is undefined at zero residual")
                    prediction_bar = 0.0_dp
                    return
                end if
                prediction_bar(i, j) = value_bar*weights(i)* &
                    sign(1.0_dp, residual)/normalization
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mae_loss_vjp

    subroutine focal_binary_cross_entropy_with_logits_value(logits, targets, &
            alpha, gamma, value, status, sample_weight, reduction)
        !! Stable focal binary cross-entropy value for logits.
        !!
        !! `alpha` is the positive-class weight in `[0,1]` and `gamma` is the
        !! nonnegative focusing exponent.  The target may be a relaxed value
        !! in `[0,1]`; binary targets reduce to the usual `alpha_t *
        !! (1-p_t)**gamma * BCE` expression.  Log-probabilities are formed
        !! with stable softplus/log-sum-exp identities, so large finite logits
        !! do not create `log(0)` or `exp(overflow)` intermediates.
        real(dp), intent(in) :: logits(:, :), targets(:, :), alpha, gamma
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, loss, derivative
        integer :: i, j

        value = 0.0_dp
        call validate_focal_inputs(logits, targets, alpha, gamma, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                call focal_terms(logits(i, j), targets(i, j), alpha, gamma, &
                    loss, derivative)
                value = value + weights(i)*loss
            end do
        end do
        value = value/normalization
        call status_set(status, FORTNUM_OK, "")
    end subroutine focal_binary_cross_entropy_with_logits_value

    subroutine focal_binary_cross_entropy_with_logits_jvp(logits, targets, &
            alpha, gamma, logits_dot, value, value_dot, status, sample_weight, &
            reduction)
        !! Value/JVP of focal binary cross-entropy with respect to logits.
        real(dp), intent(in) :: logits(:, :), targets(:, :), alpha, gamma
        real(dp), intent(in) :: logits_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, loss, derivative
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(logits_dot) /= shape(logits)) .or. &
            any(.not. ieee_is_finite(logits_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "focal BCE JVP: tangent shape or values are invalid")
            return
        end if
        call validate_focal_inputs(logits, targets, alpha, gamma, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                call focal_terms(logits(i, j), targets(i, j), alpha, gamma, &
                    loss, derivative)
                value = value + weights(i)*loss
                value_dot = value_dot + weights(i)*derivative*logits_dot(i, j)
            end do
        end do
        value = value/normalization
        value_dot = value_dot/normalization
        call status_set(status, FORTNUM_OK, "")
    end subroutine focal_binary_cross_entropy_with_logits_jvp

    subroutine focal_binary_cross_entropy_with_logits_vjp(logits, targets, alpha, &
            gamma, value_bar, logits_bar, status, sample_weight, reduction)
        !! VJP of focal binary cross-entropy with respect to logits.
        real(dp), intent(in) :: logits(:, :), targets(:, :), alpha, gamma, value_bar
        real(dp), intent(out) :: logits_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, loss, derivative
        integer :: i, j

        logits_bar = 0.0_dp
        if (any(shape(logits_bar) /= shape(logits)) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "focal BCE VJP: cotangent or output shape is invalid")
            return
        end if
        call validate_focal_inputs(logits, targets, alpha, gamma, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(logits, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(logits, 2)
            do i = 1, size(logits, 1)
                call focal_terms(logits(i, j), targets(i, j), alpha, gamma, &
                    loss, derivative)
                logits_bar(i, j) = value_bar*weights(i)*derivative/normalization
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine focal_binary_cross_entropy_with_logits_vjp

    subroutine gaussian_nll_value(prediction, targets, log_variance, value, &
            status, sample_weight, reduction)
        !! Gaussian negative log likelihood with a learned log variance.
        !!
        !! The per-element value is `0.5*(log_variance + residual**2 /
        !! variance + log(2*pi))`.  Optional row weights and mean/sum
        !! reductions follow the shared positive-weight-mass contract.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), log_variance(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, residual, inverse_variance, term
        integer :: i, j

        value = 0.0_dp
        call validate_gaussian_nll_inputs(prediction, targets, log_variance, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(prediction, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                inverse_variance = exp(-log_variance(i, j))
                term = 0.5_dp*(log_variance(i, j) + residual*residual* &
                    inverse_variance + log(2.0_dp*acos(-1.0_dp)))
                if (.not. ieee_is_finite(term)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "Gaussian NLL: value is not finite")
                    value = 0.0_dp
                    return
                end if
                value = value + weights(i)*term
            end do
        end do
        value = value/normalization
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL: reduction is not finite")
            value = 0.0_dp
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nll_value

    subroutine gaussian_nll_jvp(prediction, targets, log_variance, prediction_dot, &
            log_variance_dot, value, value_dot, status, sample_weight, reduction)
        !! Value/JVP of Gaussian NLL with respect to mean and log variance.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), log_variance(:, :)
        real(dp), intent(in) :: prediction_dot(:, :), log_variance_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, residual, inverse_variance, term
        real(dp) :: mean_gradient, variance_gradient
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(prediction_dot) /= shape(prediction)) .or. &
            any(shape(log_variance_dot) /= shape(prediction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(prediction_dot)) .or. &
            any(.not. ieee_is_finite(log_variance_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL JVP: tangents must be finite")
            return
        end if
        call validate_gaussian_nll_inputs(prediction, targets, log_variance, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(prediction, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                inverse_variance = exp(-log_variance(i, j))
                term = 0.5_dp*(log_variance(i, j) + residual*residual* &
                    inverse_variance + log(2.0_dp*acos(-1.0_dp)))
                mean_gradient = residual*inverse_variance
                variance_gradient = 0.5_dp*(1.0_dp - residual*residual* &
                    inverse_variance)
                if (.not. ieee_is_finite(term) .or. &
                    .not. ieee_is_finite(mean_gradient) .or. &
                    .not. ieee_is_finite(variance_gradient)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "Gaussian NLL JVP: product is not finite")
                    value = 0.0_dp
                    value_dot = 0.0_dp
                    return
                end if
                value = value + weights(i)*term
                value_dot = value_dot + weights(i)*(mean_gradient* &
                    prediction_dot(i, j) + variance_gradient*log_variance_dot(i, j))
            end do
        end do
        value = value/normalization
        value_dot = value_dot/normalization
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(value_dot)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL JVP: reduction is not finite")
            value = 0.0_dp
            value_dot = 0.0_dp
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nll_jvp

    subroutine gaussian_nll_vjp(prediction, targets, log_variance, value_bar, &
            prediction_bar, log_variance_bar, status, sample_weight, reduction)
        !! VJP of Gaussian NLL with respect to mean and log variance.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), log_variance(:, :)
        real(dp), intent(in) :: value_bar
        real(dp), intent(out) :: prediction_bar(:, :), log_variance_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, residual, inverse_variance
        integer :: i, j

        prediction_bar = 0.0_dp
        log_variance_bar = 0.0_dp
        if (any(shape(prediction_bar) /= shape(prediction)) .or. &
            any(shape(log_variance_bar) /= shape(prediction)) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL VJP: cotangent or output shape is invalid")
            return
        end if
        call validate_gaussian_nll_inputs(prediction, targets, log_variance, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(prediction, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                inverse_variance = exp(-log_variance(i, j))
                prediction_bar(i, j) = value_bar*weights(i)*residual* &
                    inverse_variance/normalization
                log_variance_bar(i, j) = value_bar*weights(i)*0.5_dp*(1.0_dp - &
                    residual*residual*inverse_variance)/normalization
            end do
        end do
        if (any(.not. ieee_is_finite(prediction_bar)) .or. &
            any(.not. ieee_is_finite(log_variance_bar))) then
            prediction_bar = 0.0_dp
            log_variance_bar = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nll_vjp

    subroutine gaussian_nll_hvp(prediction, targets, log_variance, prediction_dot, &
            log_variance_dot, prediction_hvp, log_variance_hvp, status, &
            sample_weight, reduction)
        !! Hessian-vector product of Gaussian NLL with respect to mean/log variance.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), log_variance(:, :)
        real(dp), intent(in) :: prediction_dot(:, :), log_variance_dot(:, :)
        real(dp), intent(out) :: prediction_hvp(:, :), log_variance_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, residual, inverse_variance, scale
        integer :: i, j

        prediction_hvp = 0.0_dp
        log_variance_hvp = 0.0_dp
        if (any(shape(prediction_dot) /= shape(prediction)) .or. &
            any(shape(log_variance_dot) /= shape(prediction)) .or. &
            any(shape(prediction_hvp) /= shape(prediction)) .or. &
            any(shape(log_variance_hvp) /= shape(prediction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL HVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(prediction_dot)) .or. &
            any(.not. ieee_is_finite(log_variance_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL HVP: tangents must be finite")
            return
        end if
        call validate_gaussian_nll_inputs(prediction, targets, log_variance, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(prediction, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                inverse_variance = exp(-log_variance(i, j))
                scale = weights(i)*inverse_variance/normalization
                prediction_hvp(i, j) = scale*(prediction_dot(i, j) - residual* &
                    log_variance_dot(i, j))
                log_variance_hvp(i, j) = scale*(-residual*prediction_dot(i, j) + &
                    0.5_dp*residual*residual*log_variance_dot(i, j))
            end do
        end do
        if (any(.not. ieee_is_finite(prediction_hvp)) .or. &
            any(.not. ieee_is_finite(log_variance_hvp))) then
            prediction_hvp = 0.0_dp
            log_variance_hvp = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nll_hvp

    subroutine poisson_nll_value(log_rate, targets, value, status, sample_weight, &
            reduction)
        !! Poisson/count negative log likelihood in log-rate coordinates.
        !!
        !! The per-element value is `exp(log_rate) - target*log_rate +
        !! log_gamma(target+1)`.  Targets are finite nonnegative counts (real
        !! values are accepted for relaxed/count-weighted objectives).
        real(dp), intent(in) :: log_rate(:, :), targets(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, rate, term
        integer :: i, j

        value = 0.0_dp
        call validate_poisson_nll_inputs(log_rate, targets, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(log_rate, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(log_rate, 2)
            do i = 1, size(log_rate, 1)
                rate = exp(log_rate(i, j))
                term = rate - targets(i, j)*log_rate(i, j) + &
                    log_gamma(targets(i, j) + 1.0_dp)
                if (.not. ieee_is_finite(term)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "Poisson NLL: value is not finite")
                    value = 0.0_dp
                    return
                end if
                value = value + weights(i)*term
            end do
        end do
        value = value/normalization
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL: reduction is not finite")
            value = 0.0_dp
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_nll_value

    subroutine poisson_nll_jvp(log_rate, targets, log_rate_dot, value, value_dot, &
            status, sample_weight, reduction)
        !! Value/JVP of Poisson NLL with respect to log rates.
        real(dp), intent(in) :: log_rate(:, :), targets(:, :), log_rate_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, rate, term
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(log_rate_dot) /= shape(log_rate))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(log_rate_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL JVP: tangent must be finite")
            return
        end if
        call validate_poisson_nll_inputs(log_rate, targets, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(log_rate, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(log_rate, 2)
            do i = 1, size(log_rate, 1)
                rate = exp(log_rate(i, j))
                term = rate - targets(i, j)*log_rate(i, j) + &
                    log_gamma(targets(i, j) + 1.0_dp)
                if (.not. ieee_is_finite(term) .or. &
                    .not. ieee_is_finite(rate - targets(i, j))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "Poisson NLL JVP: product is not finite")
                    value = 0.0_dp
                    value_dot = 0.0_dp
                    return
                end if
                value = value + weights(i)*term
                value_dot = value_dot + weights(i)*(rate - targets(i, j))* &
                    log_rate_dot(i, j)
            end do
        end do
        value = value/normalization
        value_dot = value_dot/normalization
        if (.not. ieee_is_finite(value) .or. .not. ieee_is_finite(value_dot)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL JVP: reduction is not finite")
            value = 0.0_dp
            value_dot = 0.0_dp
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_nll_jvp

    subroutine poisson_nll_vjp(log_rate, targets, value_bar, log_rate_bar, status, &
            sample_weight, reduction)
        !! VJP of Poisson NLL with respect to log rates.
        real(dp), intent(in) :: log_rate(:, :), targets(:, :), value_bar
        real(dp), intent(out) :: log_rate_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, rate
        integer :: i, j

        log_rate_bar = 0.0_dp
        if (any(shape(log_rate_bar) /= shape(log_rate)) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL VJP: cotangent or output shape is invalid")
            return
        end if
        call validate_poisson_nll_inputs(log_rate, targets, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(log_rate, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(log_rate, 2)
            do i = 1, size(log_rate, 1)
                rate = exp(log_rate(i, j))
                log_rate_bar(i, j) = value_bar*weights(i)*(rate - targets(i, j))/ &
                    normalization
            end do
        end do
        if (any(.not. ieee_is_finite(log_rate_bar))) then
            log_rate_bar = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL VJP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_nll_vjp

    subroutine poisson_nll_hvp(log_rate, targets, log_rate_dot, log_rate_hvp, &
            status, sample_weight, reduction)
        !! Hessian-vector product of Poisson NLL in log-rate coordinates.
        real(dp), intent(in) :: log_rate(:, :), targets(:, :), log_rate_dot(:, :)
        real(dp), intent(out) :: log_rate_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable :: weights(:)
        real(dp) :: normalization, rate
        integer :: i, j

        log_rate_hvp = 0.0_dp
        if (any(shape(log_rate_dot) /= shape(log_rate)) .or. &
            any(shape(log_rate_hvp) /= shape(log_rate))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL HVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(log_rate_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL HVP: tangent must be finite")
            return
        end if
        call validate_poisson_nll_inputs(log_rate, targets, status)
        if (status%code /= FORTNUM_OK) return
        call prepare_loss_weights(size(log_rate, 1), sample_weight, reduction, &
            weights, normalization, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(log_rate, 2)
            do i = 1, size(log_rate, 1)
                rate = exp(log_rate(i, j))
                log_rate_hvp(i, j) = weights(i)*rate*log_rate_dot(i, j)/ &
                    normalization
            end do
        end do
        if (any(.not. ieee_is_finite(log_rate_hvp))) then
            log_rate_hvp = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL HVP: product is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine poisson_nll_hvp

    !> Mean Huber loss for real-valued predictions.  The derivative is with
    !> respect to `prediction`; targets are treated as constants.
    subroutine huber_loss_value(prediction, targets, delta, value, status)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), delta
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: residual
        integer :: i, j

        value = 0.0_dp
        call validate_regression_inputs(prediction, targets, delta, status, &
            "Huber loss")
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                if (abs(residual) <= delta) then
                    value = value + 0.5_dp*residual*residual
                else
                    value = value + delta*(abs(residual) - 0.5_dp*delta)
                end if
            end do
        end do
        value = value/real(size(prediction), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_loss_value

    subroutine huber_loss_jvp(prediction, targets, delta, prediction_dot, value, &
            value_dot, status)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), delta
        real(dp), intent(in) :: prediction_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: residual, derivative
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(prediction_dot) /= shape(prediction)) .or. &
            any(.not. ieee_is_finite(prediction_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber loss JVP: tangent shape or values are invalid")
            return
        end if
        call huber_loss_value(prediction, targets, delta, value, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                if (abs(residual) <= delta) then
                    derivative = residual
                else
                    derivative = delta*sign(1.0_dp, residual)
                end if
                value_dot = value_dot + derivative*prediction_dot(i, j)
            end do
        end do
        value_dot = value_dot/real(size(prediction), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_loss_jvp

    subroutine huber_loss_vjp(prediction, targets, delta, value_bar, prediction_bar, &
            status)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), delta, value_bar
        real(dp), intent(out) :: prediction_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value, residual, derivative
        integer :: i, j

        prediction_bar = 0.0_dp
        if (any(shape(prediction_bar) /= shape(prediction)) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber loss VJP: cotangent or output shape is invalid")
            return
        end if
        call huber_loss_value(prediction, targets, delta, value, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                if (abs(residual) <= delta) then
                    derivative = residual
                else
                    derivative = delta*sign(1.0_dp, residual)
                end if
                prediction_bar(i, j) = value_bar*derivative/real(size(prediction), dp)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_loss_vjp

    subroutine huber_loss_hvp(prediction, targets, delta, prediction_dot, &
            prediction_hvp, status)
        !! Hessian-vector product of the mean Huber loss.
        !!
        !! The Huber second derivative is one in the quadratic region and zero
        !! in the linear region.  At either transition (`abs(residual)==delta`)
        !! it is not defined, so this routine returns a typed domain refusal
        !! rather than selecting an arbitrary subgradient.
        real(dp), intent(in) :: prediction(:, :), targets(:, :), delta
        real(dp), intent(in) :: prediction_dot(:, :)
        real(dp), intent(out) :: prediction_hvp(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: residual
        integer :: i, j

        prediction_hvp = 0.0_dp
        if (any(shape(prediction_dot) /= shape(prediction)) .or. &
            any(shape(prediction_hvp) /= shape(prediction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber loss HVP: tangent or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(prediction_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Huber loss HVP: tangent must be finite")
            return
        end if
        call validate_regression_inputs(prediction, targets, delta, status, &
            "Huber loss")
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = prediction(i, j) - targets(i, j)
                if (abs(residual) == delta) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "Huber loss HVP: second derivative is undefined at kink")
                    prediction_hvp = 0.0_dp
                    return
                else if (abs(residual) < delta) then
                    prediction_hvp(i, j) = prediction_dot(i, j)
                end if
            end do
        end do
        prediction_hvp = prediction_hvp/real(size(prediction), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine huber_loss_hvp

    !> Mean pinball/quantile loss.  The derivative products refuse residual
    !> zero, where the loss has a genuine kink.
    subroutine quantile_loss_value(prediction, targets, quantile, value, status)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), quantile
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: residual
        integer :: i, j

        value = 0.0_dp
        call validate_quantile_inputs(prediction, targets, quantile, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = targets(i, j) - prediction(i, j)
                if (residual >= 0.0_dp) then
                    value = value + quantile*residual
                else
                    value = value + (quantile - 1.0_dp)*residual
                end if
            end do
        end do
        value = value/real(size(prediction), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_loss_value

    subroutine quantile_loss_jvp(prediction, targets, quantile, prediction_dot, &
            value, value_dot, status)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), quantile
        real(dp), intent(in) :: prediction_dot(:, :)
        real(dp), intent(out) :: value, value_dot
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: residual, derivative
        integer :: i, j

        value = 0.0_dp
        value_dot = 0.0_dp
        if (any(shape(prediction_dot) /= shape(prediction)) .or. &
            any(.not. ieee_is_finite(prediction_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Quantile loss JVP: tangent shape or values are invalid")
            return
        end if
        call quantile_loss_value(prediction, targets, quantile, value, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = targets(i, j) - prediction(i, j)
                if (residual == 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "Quantile loss JVP: derivative is undefined at zero residual")
                    return
                else if (residual > 0.0_dp) then
                    derivative = -quantile
                else
                    derivative = 1.0_dp - quantile
                end if
                value_dot = value_dot + derivative*prediction_dot(i, j)
            end do
        end do
        value_dot = value_dot/real(size(prediction), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_loss_jvp

    subroutine quantile_loss_vjp(prediction, targets, quantile, value_bar, &
            prediction_bar, status)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), quantile, value_bar
        real(dp), intent(out) :: prediction_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: value, residual, derivative
        integer :: i, j

        prediction_bar = 0.0_dp
        if (any(shape(prediction_bar) /= shape(prediction)) .or. &
            .not. ieee_is_finite(value_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Quantile loss VJP: cotangent or output shape is invalid")
            return
        end if
        call quantile_loss_value(prediction, targets, quantile, value, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(prediction, 2)
            do i = 1, size(prediction, 1)
                residual = targets(i, j) - prediction(i, j)
                if (residual == 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "Quantile loss VJP: derivative is undefined at zero residual")
                    return
                else if (residual > 0.0_dp) then
                    derivative = -quantile
                else
                    derivative = 1.0_dp - quantile
                end if
                prediction_bar(i, j) = value_bar*derivative/real(size(prediction), dp)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine quantile_loss_vjp

    pure real(dp) function binary_loss(logit, target) result(value)
        real(dp), intent(in) :: logit, target

        value = log(1.0_dp + exp(-abs(logit)))
        if (logit >= 0.0_dp) then
            value = value + (1.0_dp - target)*logit
        else
            value = value - target*logit
        end if
    end function binary_loss

    subroutine validate_weighted_mse_inputs(prediction, targets, sample_weight, &
            reduction, status, reduction_kind)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), sample_weight(:)
        integer, intent(in), optional :: reduction
        type(fortnum_status_t), intent(out) :: status
        integer, intent(out) :: reduction_kind
        integer :: requested_reduction
        real(dp) :: weight_mass

        reduction_kind = LOSS_REDUCTION_MEAN
        requested_reduction = LOSS_REDUCTION_MEAN
        if (present(reduction)) requested_reduction = reduction
        if (requested_reduction /= LOSS_REDUCTION_MEAN .and. &
            requested_reduction /= LOSS_REDUCTION_SUM) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "weighted MSE: reduction must be mean or sum")
            return
        end if
        if (size(prediction, 1) < 1 .or. size(prediction, 2) < 1 .or. &
            any(shape(targets) /= shape(prediction)) .or. &
            size(sample_weight) /= size(prediction, 1) .or. &
            any(.not. ieee_is_finite(prediction)) .or. &
            any(.not. ieee_is_finite(targets)) .or. &
            any(.not. ieee_is_finite(sample_weight)) .or. &
            any(sample_weight < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "weighted MSE: inputs or sample weights are invalid")
            return
        end if
        weight_mass = sum(sample_weight)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "weighted MSE: sample weights have zero support")
            return
        end if
        reduction_kind = requested_reduction
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_weighted_mse_inputs

    pure real(dp) function weighted_mse_normalization(sample_weight, &
            reduction_kind) result(normalization)
        real(dp), intent(in) :: sample_weight(:)
        integer, intent(in) :: reduction_kind

        normalization = 1.0_dp
        if (reduction_kind == LOSS_REDUCTION_MEAN) normalization = sum(sample_weight)
    end function weighted_mse_normalization

    subroutine validate_loss_matrix_inputs(prediction, targets, name, status)
        real(dp), intent(in) :: prediction(:, :), targets(:, :)
        character(len=*), intent(in) :: name
        type(fortnum_status_t), intent(out) :: status

        if (size(prediction, 1) < 1 .or. size(prediction, 2) < 1 .or. &
            any(shape(targets) /= shape(prediction)) .or. &
            any(.not. ieee_is_finite(prediction)) .or. &
            any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(name)//": inputs must be finite, nonempty, and shape-compatible")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_loss_matrix_inputs

    subroutine prepare_loss_weights(n_samples, sample_weight, reduction, weights, &
            normalization, status)
        integer, intent(in) :: n_samples
        real(dp), intent(in), optional :: sample_weight(:)
        integer, intent(in), optional :: reduction
        real(dp), allocatable, intent(out) :: weights(:)
        real(dp), intent(out) :: normalization
        type(fortnum_status_t), intent(out) :: status
        integer :: reduction_kind
        real(dp) :: weight_mass

        normalization = 0.0_dp
        reduction_kind = LOSS_REDUCTION_MEAN
        if (present(reduction)) reduction_kind = reduction
        if (reduction_kind /= LOSS_REDUCTION_MEAN .and. &
            reduction_kind /= LOSS_REDUCTION_SUM) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "loss: reduction must be mean or sum")
            return
        end if
        if (n_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "loss: sample count must be positive")
            return
        end if
        allocate(weights(n_samples))
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "loss: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        else
            weights = 1.0_dp
        end if
        weight_mass = sum(weights)
        if (.not. ieee_is_finite(weight_mass) .or. weight_mass <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "loss: sample weights have zero support")
            return
        end if
        normalization = 1.0_dp
        if (reduction_kind == LOSS_REDUCTION_MEAN) normalization = weight_mass
        call status_set(status, FORTNUM_OK, "")
    end subroutine prepare_loss_weights

    subroutine validate_focal_inputs(logits, targets, alpha, gamma, status)
        real(dp), intent(in) :: logits(:, :), targets(:, :), alpha, gamma
        type(fortnum_status_t), intent(out) :: status

        if (size(logits, 1) < 1 .or. size(logits, 2) < 1 .or. &
            any(shape(targets) /= shape(logits)) .or. &
            any(.not. ieee_is_finite(logits)) .or. &
            any(.not. ieee_is_finite(targets)) .or. &
            any(targets < 0.0_dp) .or. any(targets > 1.0_dp) .or. &
            .not. ieee_is_finite(alpha) .or. alpha < 0.0_dp .or. alpha > 1.0_dp .or. &
            .not. ieee_is_finite(gamma) .or. gamma < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "focal BCE: logits, targets, alpha, or gamma are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_focal_inputs

    subroutine validate_gaussian_nll_inputs(prediction, targets, log_variance, status)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), log_variance(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (size(prediction, 1) < 1 .or. size(prediction, 2) < 1 .or. &
            any(shape(targets) /= shape(prediction)) .or. &
            any(shape(log_variance) /= shape(prediction)) .or. &
            any(.not. ieee_is_finite(prediction)) .or. &
            any(.not. ieee_is_finite(targets)) .or. &
            any(.not. ieee_is_finite(log_variance)) .or. &
            any(log_variance < log(tiny(1.0_dp)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Gaussian NLL: inputs or log variance are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_gaussian_nll_inputs

    subroutine validate_poisson_nll_inputs(log_rate, targets, status)
        real(dp), intent(in) :: log_rate(:, :), targets(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (size(log_rate, 1) < 1 .or. size(log_rate, 2) < 1 .or. &
            any(shape(targets) /= shape(log_rate)) .or. &
            any(.not. ieee_is_finite(log_rate)) .or. &
            any(.not. ieee_is_finite(targets)) .or. any(targets < 0.0_dp) .or. &
            any(log_rate > log(huge(1.0_dp)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Poisson NLL: log rates and targets are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_poisson_nll_inputs

    real(dp) function stable_softplus(value) result(output)
        real(dp), intent(in) :: value

        if (value > 0.0_dp) then
            output = value + log(1.0_dp + exp(-value))
        else
            output = log(1.0_dp + exp(value))
        end if
    end function stable_softplus

    real(dp) function stable_log_weighted_sum(first_weight, first_log, &
            second_weight, second_log) result(output)
        real(dp), intent(in) :: first_weight, first_log, second_weight, second_log
        real(dp) :: largest, smallest

        if (first_weight <= 0.0_dp) then
            output = second_log
        else if (second_weight <= 0.0_dp) then
            output = first_log
        else
            largest = max(first_log + log(first_weight), &
                second_log + log(second_weight))
            smallest = min(first_log + log(first_weight), &
                second_log + log(second_weight))
            output = largest + log(1.0_dp + exp(smallest - largest))
        end if
    end function stable_log_weighted_sum

    subroutine focal_terms(logit, target, alpha, gamma, loss, derivative)
        real(dp), intent(in) :: logit, target, alpha, gamma
        real(dp), intent(out) :: loss, derivative
        real(dp) :: log_probability, log_one_probability, log_one_minus_pt
        real(dp) :: probability, residual, alpha_t, focal_factor
        real(dp) :: focal_derivative, ratio

        log_probability = -stable_softplus(-logit)
        log_one_probability = -stable_softplus(logit)
        residual = stable_sigmoid(logit) - target
        probability = stable_sigmoid(logit)
        alpha_t = alpha*target + (1.0_dp - alpha)*(1.0_dp - target)
        log_one_minus_pt = stable_log_weighted_sum(target, log_one_probability, &
            1.0_dp - target, log_probability)
        if (gamma == 0.0_dp) then
            focal_factor = 1.0_dp
            focal_derivative = 0.0_dp
        else if (gamma*log_one_minus_pt <= log(tiny(1.0_dp))) then
            ! A finite logit can round a probability to exactly one.  The
            ! limiting focal value and derivative are both zero in this case.
            focal_factor = 0.0_dp
            focal_derivative = 0.0_dp
        else
            focal_factor = exp(gamma*log_one_minus_pt)
            ! Combine the focusing factor and its `1/(1-p_t)` term in
            ! log-space.  This avoids underflowing the denominator first.
            ratio = (gamma - 1.0_dp)*log_one_minus_pt
            focal_derivative = -gamma*(2.0_dp*target - 1.0_dp)*probability* &
                (1.0_dp - probability)*exp(ratio)
        end if
        loss = alpha_t*focal_factor*( &
            -target*log_probability - (1.0_dp - target)*log_one_probability)
        derivative = alpha_t*(focal_derivative*( &
            -target*log_probability - (1.0_dp - target)*log_one_probability) + &
            focal_factor*residual)
    end subroutine focal_terms

    subroutine validate_elementwise(logits, output, operation, status)
        real(dp), intent(in) :: logits(:, :), output(:, :)
        character(len=*), intent(in) :: operation
        type(fortnum_status_t), intent(out) :: status

        if (size(logits, 1) < 1 .or. size(logits, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(operation)//": logits must be a nonempty matrix")
            return
        end if
        if (any(shape(output) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(operation)//": output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(operation)//": logits must be finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_elementwise

    subroutine validate_binary_inputs(logits, targets, status)
        real(dp), intent(in) :: logits(:, :), targets(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (size(logits, 1) < 1 .or. size(logits, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy: logits must be a nonempty matrix")
            return
        end if
        if (any(shape(targets) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy: target shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits)) .or. &
            any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy: inputs must be finite")
            return
        end if
        if (any(targets < 0.0_dp) .or. any(targets > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "binary cross entropy: targets must lie in [0,1]")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_binary_inputs

    subroutine validate_ordinal_inputs(logits, labels, status)
        real(dp), intent(in) :: logits(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(logits, 1) < 1 .or. size(logits, 2) < 1 .or. &
            size(labels) /= size(logits, 1) .or. &
            any(.not. ieee_is_finite(logits)) .or. &
            any(labels < 1) .or. any(labels > size(logits, 2) + 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit loss: logits or labels are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_ordinal_inputs

    subroutine ordinal_probability_terms(logits, label, probability, dpdz, &
            d2pdz2, status)
        !! Probability and diagonal probability derivatives for one ordinal row.
        real(dp), intent(in) :: logits(:)
        integer, intent(in) :: label
        real(dp), intent(out) :: probability, dpdz(:), d2pdz2(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: previous_probability, current_probability
        real(dp) :: lower_probability, upper_probability
        real(dp) :: lower_derivative, upper_derivative
        integer :: j

        probability = 0.0_dp
        dpdz = 0.0_dp
        d2pdz2 = 0.0_dp
        if (size(logits) < 1 .or. size(dpdz) /= size(logits) .or. &
            size(d2pdz2) /= size(logits) .or. label < 1 .or. &
            label > size(logits) + 1 .or. any(.not. ieee_is_finite(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit loss: row shape or label is invalid")
            return
        end if
        previous_probability = stable_sigmoid(logits(1))
        do j = 2, size(logits)
            current_probability = stable_sigmoid(logits(j))
            if (current_probability <= previous_probability) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ordinal cumulative-logit loss: cumulative logits must be strictly ordered")
                return
            end if
            previous_probability = current_probability
        end do

        if (label == 1) then
            upper_probability = stable_sigmoid(logits(1))
            upper_derivative = upper_probability*(1.0_dp - upper_probability)
            probability = upper_probability
            dpdz(1) = upper_derivative
            d2pdz2(1) = upper_derivative*(1.0_dp - 2.0_dp*upper_probability)
        else if (label == size(logits) + 1) then
            lower_probability = stable_sigmoid(logits(size(logits)))
            lower_derivative = lower_probability*(1.0_dp - lower_probability)
            probability = 1.0_dp - lower_probability
            dpdz(size(logits)) = -lower_derivative
            d2pdz2(size(logits)) = -lower_derivative*(1.0_dp - &
                2.0_dp*lower_probability)
        else
            lower_probability = stable_sigmoid(logits(label - 1))
            upper_probability = stable_sigmoid(logits(label))
            lower_derivative = lower_probability*(1.0_dp - lower_probability)
            upper_derivative = upper_probability*(1.0_dp - upper_probability)
            probability = upper_probability - lower_probability
            dpdz(label - 1) = -lower_derivative
            dpdz(label) = upper_derivative
            d2pdz2(label - 1) = -lower_derivative*(1.0_dp - &
                2.0_dp*lower_probability)
            d2pdz2(label) = upper_derivative*(1.0_dp - 2.0_dp*upper_probability)
        end if
        if (.not. ieee_is_finite(probability) .or. probability <= 0.0_dp) then
            probability = 0.0_dp
            dpdz = 0.0_dp
            d2pdz2 = 0.0_dp
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ordinal cumulative-logit loss: class probability is not positive")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine ordinal_probability_terms

    subroutine validate_categorical_inputs(logits, labels, status)
        real(dp), intent(in) :: logits(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(logits, 1) < 1 .or. size(logits, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy: logits must be a nonempty matrix")
            return
        end if
        if (size(labels) /= size(logits, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy: label shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy: logits must be finite")
            return
        end if
        if (any(labels < 1) .or. any(labels > size(logits, 2))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "softmax cross entropy: labels are outside the class range")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_categorical_inputs

    subroutine validate_regression_inputs(prediction, targets, delta, status, name)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), delta
        type(fortnum_status_t), intent(out) :: status
        character(len=*), intent(in) :: name

        if (size(prediction, 1) < 1 .or. size(prediction, 2) < 1 .or. &
            any(shape(targets) /= shape(prediction)) .or. &
            .not. ieee_is_finite(delta) .or. delta <= 0.0_dp .or. &
            any(.not. ieee_is_finite(prediction)) .or. &
            any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(name)//": inputs or delta are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_regression_inputs

    subroutine validate_quantile_inputs(prediction, targets, quantile, status)
        real(dp), intent(in) :: prediction(:, :), targets(:, :), quantile
        type(fortnum_status_t), intent(out) :: status

        if (size(prediction, 1) < 1 .or. size(prediction, 2) < 1 .or. &
            any(shape(targets) /= shape(prediction)) .or. &
            .not. ieee_is_finite(quantile) .or. quantile <= 0.0_dp .or. &
            quantile >= 1.0_dp .or. any(.not. ieee_is_finite(prediction)) .or. &
            any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "Quantile loss: inputs or quantile are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_quantile_inputs

end module fortml_losses
