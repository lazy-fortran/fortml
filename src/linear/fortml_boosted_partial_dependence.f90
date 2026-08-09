!> Brute-force partial-dependence and ICE diagnostics for boosted trees.
module fortml_boosted_partial_dependence
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t
    use fortml_lightgbm, only: lightgbm_t
    implicit none
    private

    integer, parameter, public :: FORTML_TREE_RESPONSE_PREDICTION = 0
    integer, parameter, public :: FORTML_TREE_RESPONSE_MARGIN = 1

    public :: boosted_partial_dependence
    public :: boosted_partial_dependence_device_supported

    interface boosted_partial_dependence
        module procedure xgboost_partial_dependence
        module procedure lightgbm_partial_dependence
    end interface boosted_partial_dependence

contains

    !> Return whether the diagnostic has an implementation for `device_kind`.
    pure logical function boosted_partial_dependence_device_supported(device_kind)
        integer, intent(in) :: device_kind

        boosted_partial_dependence_device_supported = device_kind == FORTML_DEVICE_CPU
    end function boosted_partial_dependence_device_supported

    !> Compute weighted one-feature partial dependence and optional ICE values.
    subroutine xgboost_partial_dependence(model, x, feature_index, grid, &
            average, status, sample_weight, individual, response, device)
        type(xgboost_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), grid(:)
        integer, intent(in) :: feature_index
        real(dp), intent(inout) :: average(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(inout), optional :: individual(:, :)
        integer, intent(in), optional :: response
        type(fortml_device_t), intent(in), optional :: device

        real(dp), allocatable :: work(:, :), prediction(:), pending_average(:)
        real(dp), allocatable :: pending_individual(:, :), weight(:)
        integer :: grid_index, response_code
        real(dp) :: weight_sum

        call validate_request(model%fitted(), model%feature_count(), x, &
            feature_index, grid, average, status, sample_weight, individual, &
            response, device, response_code)
        if (status%code /= FORTNUM_OK) return

        allocate(work, source=x)
        allocate(prediction(size(x, 1)), pending_average(size(grid)))
        allocate(weight(size(x, 1)))
        if (present(individual)) allocate(pending_individual(size(x, 1), size(grid)))
        if (present(sample_weight)) then
            weight = sample_weight
        else
            weight = 1.0_dp
        end if
        weight_sum = sum(weight)

        do grid_index = 1, size(grid)
            work(:, feature_index) = grid(grid_index)
            if (response_code == FORTML_TREE_RESPONSE_MARGIN) then
                call model%predict_margin(work, prediction, status)
            else
                call model%predict(work, prediction, status)
            end if
            if (status%code /= FORTNUM_OK) return
            pending_average(grid_index) = dot_product(weight, prediction)/weight_sum
            if (present(individual)) then
                pending_individual(:, grid_index) = prediction
            end if
        end do

        average = pending_average
        if (present(individual)) individual = pending_individual
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgboost_partial_dependence

    !> Compute weighted one-feature partial dependence and optional ICE values.
    subroutine lightgbm_partial_dependence(model, x, feature_index, grid, &
            average, status, sample_weight, individual, response, device)
        type(lightgbm_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), grid(:)
        integer, intent(in) :: feature_index
        real(dp), intent(inout) :: average(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(inout), optional :: individual(:, :)
        integer, intent(in), optional :: response
        type(fortml_device_t), intent(in), optional :: device

        real(dp), allocatable :: work(:, :), prediction(:), pending_average(:)
        real(dp), allocatable :: pending_individual(:, :), weight(:)
        integer :: grid_index, response_code
        real(dp) :: weight_sum

        call validate_request(model%fitted(), model%feature_count(), x, &
            feature_index, grid, average, status, sample_weight, individual, &
            response, device, response_code)
        if (status%code /= FORTNUM_OK) return

        allocate(work, source=x)
        allocate(prediction(size(x, 1)), pending_average(size(grid)))
        allocate(weight(size(x, 1)))
        if (present(individual)) allocate(pending_individual(size(x, 1), size(grid)))
        if (present(sample_weight)) then
            weight = sample_weight
        else
            weight = 1.0_dp
        end if
        weight_sum = sum(weight)

        do grid_index = 1, size(grid)
            work(:, feature_index) = grid(grid_index)
            if (response_code == FORTML_TREE_RESPONSE_MARGIN) then
                call model%predict_margin(work, prediction, status)
            else
                call model%predict(work, prediction, status)
            end if
            if (status%code /= FORTNUM_OK) return
            pending_average(grid_index) = dot_product(weight, prediction)/weight_sum
            if (present(individual)) then
                pending_individual(:, grid_index) = prediction
            end if
        end do

        average = pending_average
        if (present(individual)) individual = pending_individual
        call status_set(status, FORTNUM_OK, "")
    end subroutine lightgbm_partial_dependence

    subroutine validate_request(fitted, feature_count, x, feature_index, grid, &
            average, status, sample_weight, individual, response, device, &
            response_code)
        logical, intent(in) :: fitted
        integer, intent(in) :: feature_count, feature_index
        real(dp), intent(in) :: x(:, :), grid(:), average(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: individual(:, :)
        integer, intent(in), optional :: response
        type(fortml_device_t), intent(in), optional :: device
        integer, intent(out) :: response_code

        response_code = FORTML_TREE_RESPONSE_PREDICTION
        if (.not. fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted partial dependence: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= feature_count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted partial dependence: input shape is invalid")
            return
        end if
        if (feature_index < 1 .or. feature_index > feature_count) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted partial dependence: feature index is invalid")
            return
        end if
        if (size(grid) < 1 .or. size(average) /= size(grid)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted partial dependence: grid or average shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(grid))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted partial dependence: grid values must be finite")
            return
        end if
        if (present(individual)) then
            if (size(individual, 1) /= size(x, 1) .or. &
                size(individual, 2) /= size(grid)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "boosted partial dependence: ICE output shape is invalid")
                return
            end if
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "boosted partial dependence: weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "boosted partial dependence: weights must be finite")
                return
            end if
            if (any(sample_weight < 0.0_dp) .or. sum(sample_weight) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "boosted partial dependence: weights must have positive mass")
                return
            end if
        end if
        if (present(response)) response_code = response
        if (response_code /= FORTML_TREE_RESPONSE_PREDICTION .and. &
            response_code /= FORTML_TREE_RESPONSE_MARGIN) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "boosted partial dependence: response selector is invalid")
            return
        end if
        if (present(device)) then
            if (.not. device%selected .or. .not. device%available) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "boosted partial dependence: device is not selected")
                return
            end if
            if (device%kind == FORTML_DEVICE_CUDA) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "boosted partial dependence: no resident CUDA implementation")
                return
            end if
            if (device%kind /= FORTML_DEVICE_CPU) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "boosted partial dependence: device kind is invalid")
                return
            end if
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_request

end module fortml_boosted_partial_dependence
