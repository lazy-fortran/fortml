module fortml_regression_metrics
    !! Core regression metrics with explicit row weights and refusal contracts.
    !!
    !! Samples occupy rows and outputs occupy columns.  Metrics that average
    !! errors use a uniform output average, so a row weight is applied once to
    !! every output and the denominator is ``sum(weight)*n_outputs``.  Inputs,
    !! weights, and quantiles must be finite.  Degenerate R2 and explained
    !! variance cases are refused instead of silently changing the definition.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    public :: regression_mean_squared_error
    public :: regression_root_mean_squared_error
    public :: regression_mean_absolute_error
    public :: regression_median_absolute_error
    public :: regression_max_error
    public :: regression_r2_score
    public :: regression_explained_variance
    public :: regression_mean_squared_log_error
    public :: regression_mean_absolute_percentage_error
    public :: regression_mean_pinball_loss

contains

    subroutine regression_mean_squared_error(target, prediction, value, status, &
            sample_weight)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator

        value = 0.0_dp
        call validate_inputs(target, prediction, sample_weight, denominator, &
            status, "mean squared error")
        if (status%code /= FORTNUM_OK) return
        value = weighted_sum(target, prediction, sample_weight, squared_error) / &
            (denominator*real(size(target, 2), dp))
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_mean_squared_error

    subroutine regression_root_mean_squared_error(target, prediction, value, &
            status, sample_weight)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)

        call regression_mean_squared_error(target, prediction, value, status, &
            sample_weight)
        if (status%code /= FORTNUM_OK) return
        value = sqrt(value)
    end subroutine regression_root_mean_squared_error

    subroutine regression_mean_absolute_error(target, prediction, value, status, &
            sample_weight)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator

        value = 0.0_dp
        call validate_inputs(target, prediction, sample_weight, denominator, &
            status, "mean absolute error")
        if (status%code /= FORTNUM_OK) return
        value = weighted_sum(target, prediction, sample_weight, absolute_error) / &
            (denominator*real(size(target, 2), dp))
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_mean_absolute_error

    subroutine regression_median_absolute_error(target, prediction, value, &
            status, sample_weight)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: errors(:), weights(:)
        real(dp) :: denominator
        integer :: i, j, k, count

        value = 0.0_dp
        call validate_inputs(target, prediction, sample_weight, denominator, &
            status, "median absolute error")
        if (status%code /= FORTNUM_OK) return
        count = size(target)
        allocate(errors(count), weights(count))
        k = 0
        do j = 1, size(target, 2)
            do i = 1, size(target, 1)
                k = k + 1
                errors(k) = abs(target(i, j) - prediction(i, j))
                weights(k) = 1.0_dp
                if (present(sample_weight)) weights(k) = sample_weight(i)
            end do
        end do
        call sort_pairs(errors, weights)
        value = weighted_median(errors, weights, sum(weights))
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_median_absolute_error

    subroutine regression_max_error(target, prediction, value, status)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        call validate_inputs(target, prediction, status=status, &
            context="max error")
        if (status%code /= FORTNUM_OK) return
        value = maxval(abs(target - prediction))
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_max_error

    subroutine regression_r2_score(target, prediction, value, status, sample_weight)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator, mean_target, ss_res, ss_tot
        integer :: i, j

        value = 0.0_dp
        call validate_inputs(target, prediction, sample_weight, denominator, &
            status, "R2 score")
        if (status%code /= FORTNUM_OK) return
        value = 0.0_dp
        do j = 1, size(target, 2)
            mean_target = weighted_mean(target(:, j), sample_weight, denominator)
            ss_res = 0.0_dp
            ss_tot = 0.0_dp
            do i = 1, size(target, 1)
                ss_res = ss_res + row_weight(i, sample_weight)* &
                    (target(i, j) - prediction(i, j))**2
                ss_tot = ss_tot + row_weight(i, sample_weight)* &
                    (target(i, j) - mean_target)**2
            end do
            if (ss_tot <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "R2 score: target variance must be positive")
                return
            end if
            value = value + 1.0_dp - ss_res/ss_tot
        end do
        value = value/real(size(target, 2), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_r2_score

    subroutine regression_explained_variance(target, prediction, value, status, &
            sample_weight)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator, target_mean, residual_mean, target_var, residual_var
        integer :: i, j

        value = 0.0_dp
        call validate_inputs(target, prediction, sample_weight, denominator, &
            status, "explained variance")
        if (status%code /= FORTNUM_OK) return
        do j = 1, size(target, 2)
            target_mean = weighted_mean(target(:, j), sample_weight, denominator)
            residual_mean = weighted_mean(target(:, j) - prediction(:, j), &
                sample_weight, denominator)
            target_var = 0.0_dp
            residual_var = 0.0_dp
            do i = 1, size(target, 1)
                target_var = target_var + row_weight(i, sample_weight)* &
                    (target(i, j) - target_mean)**2
                residual_var = residual_var + row_weight(i, sample_weight)* &
                    (target(i, j) - prediction(i, j) - residual_mean)**2
            end do
            if (target_var <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "explained variance: target variance must be positive")
                return
            end if
            value = value + 1.0_dp - residual_var/target_var
        end do
        value = value/real(size(target, 2), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_explained_variance

    subroutine regression_mean_squared_log_error(target, prediction, value, &
            status, sample_weight)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator

        value = 0.0_dp
        call validate_inputs(target, prediction, sample_weight, denominator, &
            status, "mean squared log error")
        if (status%code /= FORTNUM_OK) return
        if (any(target < 0.0_dp) .or. any(prediction < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "mean squared log error: values must be nonnegative")
            return
        end if
        value = weighted_sum_log(target, prediction, sample_weight) / &
            (denominator*real(size(target, 2), dp))
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_mean_squared_log_error

    subroutine regression_mean_absolute_percentage_error(target, prediction, &
            value, status, sample_weight)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator
        integer :: i, j

        value = 0.0_dp
        call validate_inputs(target, prediction, sample_weight, denominator, &
            status, "mean absolute percentage error")
        if (status%code /= FORTNUM_OK) return
        if (any(target == 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "mean absolute percentage error: target values must be nonzero")
            return
        end if
        do j = 1, size(target, 2)
            do i = 1, size(target, 1)
                value = value + row_weight(i, sample_weight)* &
                    abs((target(i, j) - prediction(i, j))/target(i, j))
            end do
        end do
        value = value/(denominator*real(size(target, 2), dp))
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_mean_absolute_percentage_error

    subroutine regression_mean_pinball_loss(target, prediction, quantile, value, &
            status, sample_weight)
        real(dp), intent(in) :: target(:, :), prediction(:, :), quantile
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: denominator, error
        integer :: i, j

        value = 0.0_dp
        call validate_inputs(target, prediction, sample_weight, denominator, &
            status, "mean pinball loss")
        if (status%code /= FORTNUM_OK) return
        if (.not. ieee_is_finite(quantile) .or. quantile < 0.0_dp .or. &
            quantile > 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "mean pinball loss: quantile must lie in [0, 1]")
            return
        end if
        do j = 1, size(target, 2)
            do i = 1, size(target, 1)
                error = target(i, j) - prediction(i, j)
                value = value + row_weight(i, sample_weight)*max(quantile*error, &
                    (quantile - 1.0_dp)*error)
            end do
        end do
        value = value/(denominator*real(size(target, 2), dp))
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_mean_pinball_loss

    subroutine validate_inputs(target, prediction, sample_weight, denominator, &
            status, context)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(out), optional :: denominator
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: context
        real(dp) :: weight_sum

        if (size(target, 1) < 1 .or. size(target, 2) < 1 .or. &
            any(shape(prediction) /= shape(target))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": input shapes are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(target)) .or. &
            any(.not. ieee_is_finite(prediction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                trim(context)//": values must be finite")
            return
        end if
        weight_sum = real(size(target, 1), dp)
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(target, 1) .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    trim(context)//": weights must be finite and nonnegative")
                return
            end if
            weight_sum = sum(sample_weight)
            if (weight_sum <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    trim(context)//": weights must have positive mass")
                return
            end if
        end if
        if (present(denominator)) denominator = weight_sum
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_inputs

    real(dp) function weighted_sum(target, prediction, sample_weight, metric) &
            result(total)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(in), optional :: sample_weight(:)
        interface
            pure real(dp) function metric(a, b)
                import dp
                real(dp), intent(in) :: a, b
            end function metric
        end interface
        integer :: i, j

        total = 0.0_dp
        do j = 1, size(target, 2)
            do i = 1, size(target, 1)
                total = total + row_weight(i, sample_weight)* &
                    metric(target(i, j), prediction(i, j))
            end do
        end do
    end function weighted_sum

    real(dp) function weighted_sum_log(target, prediction, sample_weight) &
            result(total)
        real(dp), intent(in) :: target(:, :), prediction(:, :)
        real(dp), intent(in), optional :: sample_weight(:)
        integer :: i, j

        total = 0.0_dp
        do j = 1, size(target, 2)
            do i = 1, size(target, 1)
                total = total + row_weight(i, sample_weight)* &
                    (log(1.0_dp + target(i, j)) - &
                    log(1.0_dp + prediction(i, j)))**2
            end do
        end do
    end function weighted_sum_log

    pure real(dp) function row_weight(i, sample_weight) result(weight)
        integer, intent(in) :: i
        real(dp), intent(in), optional :: sample_weight(:)

        weight = 1.0_dp
        if (present(sample_weight)) weight = sample_weight(i)
    end function row_weight

    pure real(dp) function weighted_mean(values, sample_weight, denominator) &
            result(mean)
        real(dp), intent(in) :: values(:), denominator
        real(dp), intent(in), optional :: sample_weight(:)
        integer :: i

        mean = 0.0_dp
        do i = 1, size(values)
            mean = mean + row_weight(i, sample_weight)*values(i)
        end do
        mean = mean/denominator
    end function weighted_mean

    pure real(dp) function squared_error(a, b) result(value)
        real(dp), intent(in) :: a, b
        value = (a - b)**2
    end function squared_error

    pure real(dp) function absolute_error(a, b) result(value)
        real(dp), intent(in) :: a, b
        value = abs(a - b)
    end function absolute_error

    subroutine sort_pairs(values, weights)
        real(dp), intent(inout) :: values(:), weights(:)
        real(dp) :: value_tmp, weight_tmp
        integer :: i, j

        do i = 2, size(values)
            value_tmp = values(i)
            weight_tmp = weights(i)
            j = i - 1
            do while (j >= 1)
                if (values(j) <= value_tmp) exit
                values(j + 1) = values(j)
                weights(j + 1) = weights(j)
                j = j - 1
            end do
            values(j + 1) = value_tmp
            weights(j + 1) = weight_tmp
        end do
    end subroutine sort_pairs

    real(dp) function weighted_median(values, weights, total_weight) result(value)
        real(dp), intent(in) :: values(:), weights(:), total_weight
        real(dp) :: cumulative
        integer :: i

        value = values(size(values))
        cumulative = 0.0_dp
        do i = 1, size(values)
            cumulative = cumulative + weights(i)
            if (cumulative >= 0.5_dp*total_weight) then
                value = values(i)
                return
            end if
        end do
    end function weighted_median

end module fortml_regression_metrics
