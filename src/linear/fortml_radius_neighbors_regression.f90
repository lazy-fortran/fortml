module fortml_radius_neighbors_regression
    !! Dense closed-radius nearest-neighbor regression.
    !!
    !! Training rows and scalar targets are retained verbatim.  A query uses
    !! every training row whose squared Euclidean distance is at most
    !! ``radius**2``.  Uniform and inverse-distance averaging, nonnegative
    !! sample weights, and an explicit value for empty neighborhoods are
    !! supported.  Radius membership is discrete; input JVP/VJP products
    !! therefore return a typed refusal at selection boundaries.  CUDA is an
    !! explicit refusal until a resident radius-search kernel is linked.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    integer, parameter, public :: RADIUS_REGRESSION_WEIGHTS_UNIFORM = 1
    integer, parameter, public :: RADIUS_REGRESSION_WEIGHTS_DISTANCE = 2
    integer, parameter, public :: RADIUS_REGRESSION_WEIGHT_UNIFORM = &
        RADIUS_REGRESSION_WEIGHTS_UNIFORM
    integer, parameter, public :: RADIUS_REGRESSION_WEIGHT_DISTANCE = &
        RADIUS_REGRESSION_WEIGHTS_DISTANCE

    type, public :: radius_neighbors_regressor_t
        private
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: target_train(:)
        real(dp), allocatable :: sample_weight(:)
        real(dp) :: radius_value = 0.0_dp
        integer :: weighting_code = RADIUS_REGRESSION_WEIGHTS_UNIFORM
        real(dp) :: outlier_value = 0.0_dp
        integer :: n_features = 0
        integer :: n_samples = 0
        logical :: has_outlier = .false.
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => radius_neighbors_regression_fit
        procedure, public :: predict => radius_neighbors_regression_predict
        procedure, public :: predict_device => &
            radius_neighbors_regression_predict_device
        procedure, public :: predict_jvp => &
            radius_neighbors_regression_predict_jvp
        procedure, public :: predict_vjp => &
            radius_neighbors_regression_predict_vjp
        procedure, public :: device_supported => &
            radius_neighbors_regression_device_supported
        procedure, public :: radius => radius_neighbors_regression_radius
        procedure, public :: weighting => radius_neighbors_regression_weighting
        procedure, public :: feature_count => &
            radius_neighbors_regression_feature_count
        procedure, public :: sample_count => &
            radius_neighbors_regression_sample_count
        procedure, public :: fitted => radius_neighbors_regression_fitted
    end type radius_neighbors_regressor_t

    public :: radius_neighbors_regression_fit
    public :: radius_neighbors_regression_predict
    public :: radius_neighbors_regression_predict_device

contains

    subroutine radius_neighbors_regression_fit(self, x, targets, status, radius, &
            weights, sample_weight, outlier_value)
        class(radius_neighbors_regressor_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), targets(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: radius, sample_weight(:), outlier_value
        integer, intent(in), optional :: weights
        real(dp) :: requested_radius, total_weight, radius_squared
        integer :: requested_weight

        self%is_fitted = .false.
        requested_radius = 1.0_dp
        if (present(radius)) requested_radius = radius
        requested_weight = RADIUS_REGRESSION_WEIGHTS_UNIFORM
        if (present(weights)) requested_weight = weights

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(targets) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression fit: inputs must be finite")
            return
        end if
        if (.not. ieee_is_finite(requested_radius) .or. requested_radius <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression fit: radius must be finite and positive")
            return
        end if
        radius_squared = requested_radius*requested_radius
        if (.not. ieee_is_finite(radius_squared)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression fit: radius is too large")
            return
        end if
        if (requested_weight /= RADIUS_REGRESSION_WEIGHTS_UNIFORM .and. &
            requested_weight /= RADIUS_REGRESSION_WEIGHTS_DISTANCE) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression fit: weights must be uniform or distance")
            return
        end if

        allocate(self%x_train(size(x, 1), size(x, 2)), &
            self%target_train(size(targets)), self%sample_weight(size(targets)))
        self%x_train = x
        self%target_train = targets
        self%sample_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(targets)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor regression fit: sample-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor regression fit: sample weights must be finite and nonnegative")
                return
            end if
            total_weight = sum(sample_weight)
            if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor regression fit: sample weights need positive total mass")
                return
            end if
            self%sample_weight = sample_weight
        end if
        self%radius_value = requested_radius
        self%weighting_code = requested_weight
        self%has_outlier = .false.
        self%outlier_value = 0.0_dp
        if (present(outlier_value)) then
            if (.not. ieee_is_finite(outlier_value)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor regression fit: outlier value must be finite")
                return
            end if
            self%has_outlier = .true.
            self%outlier_value = outlier_value
        end if
        self%n_features = size(x, 2)
        self%n_samples = size(x, 1)
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine radius_neighbors_regression_fit

    subroutine radius_neighbors_regression_predict(self, x, targets, status)
        class(radius_neighbors_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: targets(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: d2, delta, denominator, numerator, weight, radius_squared
        integer :: i, j, k, selected
        logical :: exact_neighbor

        targets = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression predict: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(targets) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression predict: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression predict: inputs must be finite")
            return
        end if

        radius_squared = self%radius_value*self%radius_value
        do i = 1, size(x, 1)
            selected = 0
            exact_neighbor = .false.
            do j = 1, self%n_samples
                d2 = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                end do
                if (.not. ieee_is_finite(d2)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "radius-neighbor regression predict: distance overflow")
                    return
                end if
                if (d2 <= radius_squared) then
                    selected = selected + 1
                    if (d2 == 0.0_dp) exact_neighbor = .true.
                end if
            end do
            if (selected == 0) then
                if (.not. self%has_outlier) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "radius-neighbor regression predict: query has no neighbors")
                    return
                end if
                targets(i) = self%outlier_value
                cycle
            end if

            numerator = 0.0_dp
            denominator = 0.0_dp
            do j = 1, self%n_samples
                d2 = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                end do
                if (d2 <= radius_squared) then
                    if (self%weighting_code == RADIUS_REGRESSION_WEIGHTS_UNIFORM) then
                        weight = self%sample_weight(j)
                    else if (exact_neighbor) then
                        weight = 0.0_dp
                        if (d2 == 0.0_dp) weight = self%sample_weight(j)
                    else
                        weight = self%sample_weight(j)/sqrt(d2)
                    end if
                    numerator = numerator + weight*self%target_train(j)
                    denominator = denominator + weight
                end if
            end do
            if (.not. ieee_is_finite(denominator) .or. denominator <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor regression predict: selected neighbors have zero weight")
                return
            end if
            targets(i) = numerator/denominator
            if (.not. ieee_is_finite(targets(i))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "radius-neighbor regression predict: prediction is not finite")
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine radius_neighbors_regression_predict

    subroutine radius_neighbors_regression_predict_device(self, device, x, targets, &
            status)
        class(radius_neighbors_regressor_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: targets(:)
        type(fortnum_status_t), intent(out) :: status

        targets = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, targets, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "radius-neighbor regression device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "radius-neighbor regression device prediction: device kind is invalid")
        end select
    end subroutine radius_neighbors_regression_predict_device

    subroutine radius_neighbors_regression_predict_jvp(self, x, x_dot, targets, &
            targets_dot, status)
        class(radius_neighbors_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: targets(:), targets_dot(:)
        type(fortnum_status_t), intent(out) :: status

        targets = 0.0_dp
        targets_dot = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "radius-neighbor regression input JVP is undefined across selection boundaries")
    end subroutine radius_neighbors_regression_predict_jvp

    subroutine radius_neighbors_regression_predict_vjp(self, x, targets_bar, x_bar, &
            status)
        class(radius_neighbors_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), targets_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        x_bar = 0.0_dp
        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "radius-neighbor regression input VJP is undefined across selection boundaries")
    end subroutine radius_neighbors_regression_predict_vjp

    logical function radius_neighbors_regression_device_supported(self, device_kind) &
            result(supported)
        class(radius_neighbors_regressor_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%is_fitted
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function radius_neighbors_regression_device_supported

    real(dp) function radius_neighbors_regression_radius(self) result(value)
        class(radius_neighbors_regressor_t), intent(in) :: self
        value = self%radius_value
    end function radius_neighbors_regression_radius

    integer function radius_neighbors_regression_weighting(self) result(value)
        class(radius_neighbors_regressor_t), intent(in) :: self
        value = self%weighting_code
    end function radius_neighbors_regression_weighting

    integer function radius_neighbors_regression_feature_count(self) result(value)
        class(radius_neighbors_regressor_t), intent(in) :: self
        value = self%n_features
    end function radius_neighbors_regression_feature_count

    integer function radius_neighbors_regression_sample_count(self) result(value)
        class(radius_neighbors_regressor_t), intent(in) :: self
        value = self%n_samples
    end function radius_neighbors_regression_sample_count

    logical function radius_neighbors_regression_fitted(self) result(value)
        class(radius_neighbors_regressor_t), intent(in) :: self
        value = self%is_fitted
    end function radius_neighbors_regression_fitted

end module fortml_radius_neighbors_regression
