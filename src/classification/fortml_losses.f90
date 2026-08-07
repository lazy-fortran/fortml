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
    public :: softmax_value, softmax_jvp, softmax_vjp
    public :: log_softmax_value, log_softmax_jvp, log_softmax_vjp
    public :: softmax_cross_entropy_value
    public :: softmax_cross_entropy_jvp
    public :: softmax_cross_entropy_vjp

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

    pure real(dp) function binary_loss(logit, target) result(value)
        real(dp), intent(in) :: logit, target

        value = log(1.0_dp + exp(-abs(logit)))
        if (logit >= 0.0_dp) then
            value = value + (1.0_dp - target)*logit
        else
            value = value - target*logit
        end if
    end function binary_loss

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

end module fortml_losses
