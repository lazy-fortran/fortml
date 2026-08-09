!> Independent multi-output regression adapters for the deterministic tree lanes.
module fortml_xgboost_multioutput
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t
    use fortml_lightgbm, only: lightgbm_t, lightgbm_options_t
    implicit none
    private

    !> One deterministic regression booster per target column.
    !>
    !> Samples remain rows and outputs remain columns.  Every child is fitted
    !> independently with the same options and optional row weights.  Fit and
    !> prediction are transactional: malformed input or a child failure leaves
    !> the destination state/output untouched.  The fitted tree structure is
    !> discrete; input JVP/VJP products therefore use each child tree's
    !> piecewise-constant contract, while leaf-coordinate products remain
    !> available for fixed-state parameter sensitivity.
    type, public :: xgboost_multioutput_t
        private
        type(xgboost_t), allocatable :: models(:)
        integer :: n_inputs = 0
        integer :: n_outputs = 0
        logical :: initialized = .false.
    contains
        procedure, public :: fit => xgb_multi_fit
        procedure, public :: predict => xgb_multi_predict
        procedure, public :: predict_margin => xgb_multi_predict_margin
        procedure, public :: predict_staged_margin => xgb_multi_predict_staged_margin
        procedure, public :: predict_jvp => xgb_multi_predict_jvp
        procedure, public :: predict_vjp => xgb_multi_predict_vjp
        procedure, public :: predict_leaf_jvp => xgb_multi_predict_leaf_jvp
        procedure, public :: predict_leaf_vjp => xgb_multi_predict_leaf_vjp
        procedure, public :: predict_device => xgb_multi_predict_device
        procedure, public :: predict_device_margin => xgb_multi_predict_device_margin
        procedure, public :: device_supported => xgb_multi_device_supported
        procedure, public :: feature_count => xgb_multi_feature_count
        procedure, public :: output_count => xgb_multi_output_count
        procedure, public :: estimator_count => xgb_multi_estimator_count
        procedure, public :: best_iteration => xgb_multi_best_iteration
        procedure, public :: best_validation_loss => xgb_multi_best_validation_loss
        procedure, public :: early_stopped => xgb_multi_early_stopped
        procedure, public :: parameter_count => xgb_multi_parameter_count
        procedure, public :: leaf_parameter_count => xgb_multi_parameter_count
        procedure, public :: leaf_parameters => xgb_multi_leaf_parameters
        procedure, public :: fitted => xgb_multi_fitted
    end type xgboost_multioutput_t

    !> One deterministic LightGBM-style regression booster per target column.
    !> It has the same row/output and transactional contracts as
    !> `xgboost_multioutput_t`, while retaining LightGBM's leaf-wise growth.
    type, public :: lightgbm_multioutput_t
        private
        type(lightgbm_t), allocatable :: models(:)
        integer :: n_inputs = 0
        integer :: n_outputs = 0
        logical :: initialized = .false.
    contains
        procedure, public :: fit => lgbm_multi_fit
        procedure, public :: predict => lgbm_multi_predict
        procedure, public :: predict_margin => lgbm_multi_predict_margin
        procedure, public :: predict_staged_margin => lgbm_multi_predict_staged_margin
        procedure, public :: predict_jvp => lgbm_multi_predict_jvp
        procedure, public :: predict_vjp => lgbm_multi_predict_vjp
        procedure, public :: predict_leaf_jvp => lgbm_multi_predict_leaf_jvp
        procedure, public :: predict_leaf_vjp => lgbm_multi_predict_leaf_vjp
        procedure, public :: predict_device => lgbm_multi_predict_device
        procedure, public :: predict_device_margin => lgbm_multi_predict_device_margin
        procedure, public :: device_supported => lgbm_multi_device_supported
        procedure, public :: feature_count => lgbm_multi_feature_count
        procedure, public :: output_count => lgbm_multi_output_count
        procedure, public :: estimator_count => lgbm_multi_estimator_count
        procedure, public :: best_iteration => lgbm_multi_best_iteration
        procedure, public :: best_validation_loss => lgbm_multi_best_validation_loss
        procedure, public :: early_stopped => lgbm_multi_early_stopped
        procedure, public :: parameter_count => lgbm_multi_parameter_count
        procedure, public :: leaf_parameter_count => lgbm_multi_parameter_count
        procedure, public :: leaf_parameters => lgbm_multi_leaf_parameters
        procedure, public :: fitted => lgbm_multi_fitted
    end type lightgbm_multioutput_t

contains

    subroutine xgb_multi_fit(self, x, targets, status, options, sample_weight, &
            validation_x, validation_targets, validation_weight)
        class(xgboost_multioutput_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), targets(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_targets(:, :), &
            validation_weight(:)
        type(xgboost_multioutput_t) :: candidate
        integer :: j

        if (.not. valid_fit_shape(x, targets, sample_weight, validation_x, &
            validation_targets, validation_weight)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output fit: dimensions or weights are invalid")
            return
        end if
        if (present(validation_x)) then
            if (size(validation_x, 2) /= size(x, 2) .or. &
                size(validation_targets, 2) /= size(targets, 2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost multi-output fit: validation feature/output shape is invalid")
                return
            end if
        end if
        candidate%n_inputs = size(x, 2)
        candidate%n_outputs = size(targets, 2)
        allocate(candidate%models(candidate%n_outputs))
        do j = 1, candidate%n_outputs
            if (present(validation_x)) then
                if (present(sample_weight)) then
                    if (present(validation_weight)) then
                        call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                            options, sample_weight, validation_x, validation_targets(:, j), &
                            validation_weight)
                    else
                        call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                            options, sample_weight, validation_x, validation_targets(:, j))
                    end if
                else if (present(validation_weight)) then
                    call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                        options, validation_x=validation_x, &
                        validation_y=validation_targets(:, j), &
                        validation_weight=validation_weight)
                else
                    call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                        options, validation_x=validation_x, validation_y=validation_targets(:, j))
                end if
            else if (present(sample_weight)) then
                call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                    options, sample_weight)
            else
                call candidate%models(j)%fit_regression(x, targets(:, j), status, options)
            end if
            if (status%code /= FORTNUM_OK) return
        end do
        candidate%initialized = .true.
        self%n_inputs = candidate%n_inputs
        self%n_outputs = candidate%n_outputs
        self%models = candidate%models
        self%initialized = candidate%initialized
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multi_fit

    subroutine xgb_multi_predict(self, x, values, status)
        class(xgboost_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :)
        integer :: j

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output predict: model or output shape is invalid")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)))
        do j = 1, self%n_outputs
            call self%models(j)%predict(x, candidate(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        values = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multi_predict

    subroutine xgb_multi_predict_margin(self, x, values, status)
        class(xgboost_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :)
        integer :: j

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output predict_margin: model or output shape is invalid")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)))
        do j = 1, self%n_outputs
            call self%models(j)%predict_margin(x, candidate(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        values = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multi_predict_margin

    subroutine xgb_multi_predict_staged_margin(self, x, staged, status)
        class(xgboost_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: staged(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :, :), child(:, :)
        integer :: j, n_estimators

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(staged, 1) /= size(x, 1) .or. size(staged, 3) /= self%n_outputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output staged margin: model or output shape is invalid")
            return
        end if
        n_estimators = self%models(1)%estimator_count()
        if (n_estimators < 1 .or. size(staged, 2) /= n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output staged margin: stage count is invalid")
            return
        end if
        do j = 2, self%n_outputs
            if (self%models(j)%estimator_count() /= n_estimators) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost multi-output staged margin: child stage counts differ")
                return
            end if
        end do
        allocate(candidate(size(staged, 1), size(staged, 2), size(staged, 3)))
        allocate(child(size(x, 1), n_estimators))
        do j = 1, self%n_outputs
            call self%models(j)%predict_staged_margin(x, child, status)
            if (status%code /= FORTNUM_OK) return
            candidate(:, :, j) = child
        end do
        staged = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multi_predict_staged_margin

    subroutine xgb_multi_predict_jvp(self, x, x_dot, values, values_dot, status)
        class(xgboost_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(inout) :: values(:, :), values_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), candidate_dot(:, :)
        integer :: j

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values) .or. &
            any(shape(x_dot) /= shape(x)) .or. any(shape(values_dot) /= shape(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output predict_jvp: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output predict_jvp: tangent is not finite")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)), &
            candidate_dot(size(values_dot, 1), size(values_dot, 2)))
        do j = 1, self%n_outputs
            call self%models(j)%predict_jvp(x, x_dot, candidate(:, j), &
                candidate_dot(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        values = candidate
        values_dot = candidate_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multi_predict_jvp

    subroutine xgb_multi_predict_vjp(self, x, output_bar, x_bar, status)
        class(xgboost_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(inout) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), child_bar(:, :)
        integer :: j

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(output_bar, 1) /= size(x, 1) .or. size(output_bar, 2) /= self%n_outputs .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output predict_vjp: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output predict_vjp: cotangent is not finite")
            return
        end if
        allocate(candidate(size(x, 1), size(x, 2)), child_bar(size(x, 1), size(x, 2)))
        candidate = 0.0_dp
        do j = 1, self%n_outputs
            call self%models(j)%predict_vjp(x, output_bar(:, j), child_bar, status)
            if (status%code /= FORTNUM_OK) return
            candidate = candidate + child_bar
        end do
        x_bar = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multi_predict_vjp

    subroutine xgb_multi_predict_leaf_jvp(self, x, parameter_dot, values, values_dot, status)
        class(xgboost_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), parameter_dot(:)
        real(dp), intent(inout) :: values(:, :), values_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), candidate_dot(:, :), p(:)
        integer :: j, first, last, count

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values) .or. &
            any(shape(values_dot) /= shape(values)) .or. &
            size(parameter_dot) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(parameter_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output predict_leaf_jvp: shape or values are invalid")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)), &
            candidate_dot(size(values_dot, 1), size(values_dot, 2)))
        first = 1
        do j = 1, self%n_outputs
            count = self%models(j)%leaf_parameter_count()
            last = first + count - 1
            allocate(p(count))
            p = parameter_dot(first:last)
            call self%models(j)%predict_leaf_jvp(x, p, candidate(:, j), &
                candidate_dot(:, j), status)
            deallocate(p)
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        values = candidate
        values_dot = candidate_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multi_predict_leaf_jvp

    subroutine xgb_multi_predict_leaf_vjp(self, x, output_bar, parameter_bar, status)
        class(xgboost_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:), p(:)
        integer :: j, first, last, count

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(output_bar, 1) /= size(x, 1) .or. size(output_bar, 2) /= self%n_outputs .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output predict_leaf_vjp: shape or values are invalid")
            return
        end if
        allocate(candidate(size(parameter_bar)))
        candidate = 0.0_dp
        first = 1
        do j = 1, self%n_outputs
            count = self%models(j)%leaf_parameter_count()
            last = first + count - 1
            allocate(p(count))
            call self%models(j)%predict_leaf_vjp(x, output_bar(:, j), p, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(p)
                return
            end if
            candidate(first:last) = p
            deallocate(p)
            first = last + 1
        end do
        parameter_bar = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multi_predict_leaf_vjp

    subroutine xgb_multi_predict_device(self, device, x, values, status)
        class(xgboost_multioutput_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :)
        integer :: j

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output device prediction: model or shape is invalid")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output device prediction: device is not selected")
            return
        end if
        if (device%kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "xgboost multi-output device prediction: no resident CUDA tree kernel is linked")
            return
        end if
        if (device%kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output device prediction: device kind is invalid")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)))
        do j = 1, self%n_outputs
            call self%models(j)%predict_device(device, x, candidate(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        values = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multi_predict_device

    subroutine xgb_multi_predict_device_margin(self, device, x, values, status)
        class(xgboost_multioutput_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output device margin: model or shape is invalid")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output device margin: device is not selected")
            return
        end if
        if (device%kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "xgboost multi-output device margin: no resident CUDA tree kernel is linked")
            return
        end if
        if (device%kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output device margin: device kind is invalid")
            return
        end if
        call xgb_multi_predict_margin(self, x, values, status)
    end subroutine xgb_multi_predict_device_margin

    logical function xgb_multi_device_supported(self, device_kind) result(supported)
        class(xgboost_multioutput_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%initialized .and. device_kind == FORTML_DEVICE_CPU
    end function xgb_multi_device_supported

    integer function xgb_multi_feature_count(self) result(value)
        class(xgboost_multioutput_t), intent(in) :: self
        value = self%n_inputs
    end function xgb_multi_feature_count

    integer function xgb_multi_output_count(self) result(value)
        class(xgboost_multioutput_t), intent(in) :: self
        value = self%n_outputs
    end function xgb_multi_output_count

    integer function xgb_multi_estimator_count(self) result(value)
        class(xgboost_multioutput_t), intent(in) :: self
        value = 0
        if (self%initialized .and. allocated(self%models)) value = self%models(1)%estimator_count()
    end function xgb_multi_estimator_count

    integer function xgb_multi_parameter_count(self) result(value)
        class(xgboost_multioutput_t), intent(in) :: self
        integer :: j
        value = 0
        if (.not. self%initialized .or. .not. allocated(self%models)) return
        do j = 1, size(self%models)
            value = value + self%models(j)%leaf_parameter_count()
        end do
    end function xgb_multi_parameter_count

    function xgb_multi_leaf_parameters(self, status) result(parameters)
        class(xgboost_multioutput_t), intent(in) :: self
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: parameters(:), p(:)
        integer :: j, first, last, count

        allocate(parameters(max(0, self%parameter_count())))
        if (.not. self%initialized .or. .not. allocated(self%models)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost multi-output leaf_parameters: model is not initialized")
            return
        end if
        first = 1
        do j = 1, self%n_outputs
            p = self%models(j)%leaf_parameters(status)
            if (status%code /= FORTNUM_OK) return
            count = size(p)
            last = first + count - 1
            parameters(first:last) = p
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end function xgb_multi_leaf_parameters

    logical function xgb_multi_fitted(self) result(value)
        class(xgboost_multioutput_t), intent(in) :: self
        value = self%initialized .and. self%n_inputs > 0 .and. self%n_outputs > 0 .and. &
            allocated(self%models)
    end function xgb_multi_fitted

    function xgb_multi_best_iteration(self) result(values)
        class(xgboost_multioutput_t), intent(in) :: self
        integer, allocatable :: values(:)
        integer :: j

        allocate(values(max(0, self%n_outputs)))
        values = 0
        if (.not. self%initialized .or. .not. allocated(self%models)) return
        do j = 1, self%n_outputs
            values(j) = self%models(j)%best_iteration()
        end do
    end function xgb_multi_best_iteration

    function xgb_multi_best_validation_loss(self) result(values)
        class(xgboost_multioutput_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: j

        allocate(values(max(0, self%n_outputs)))
        values = huge(1.0_dp)
        if (.not. self%initialized .or. .not. allocated(self%models)) return
        do j = 1, self%n_outputs
            values(j) = self%models(j)%best_validation_loss()
        end do
    end function xgb_multi_best_validation_loss

    function xgb_multi_early_stopped(self) result(values)
        class(xgboost_multioutput_t), intent(in) :: self
        logical, allocatable :: values(:)
        integer :: j

        allocate(values(max(0, self%n_outputs)))
        values = .false.
        if (.not. self%initialized .or. .not. allocated(self%models)) return
        do j = 1, self%n_outputs
            values(j) = self%models(j)%early_stopped()
        end do
    end function xgb_multi_early_stopped

    subroutine lgbm_multi_fit(self, x, targets, status, options, sample_weight, &
            validation_x, validation_targets, validation_weight)
        class(lightgbm_multioutput_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), targets(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_targets(:, :), &
            validation_weight(:)
        type(lightgbm_multioutput_t) :: candidate
        integer :: j

        if (.not. valid_fit_shape(x, targets, sample_weight, validation_x, &
            validation_targets, validation_weight)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output fit: dimensions or weights are invalid")
            return
        end if
        if (present(validation_x)) then
            if (size(validation_x, 2) /= size(x, 2) .or. &
                size(validation_targets, 2) /= size(targets, 2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm multi-output fit: validation feature/output shape is invalid")
                return
            end if
        end if
        candidate%n_inputs = size(x, 2)
        candidate%n_outputs = size(targets, 2)
        allocate(candidate%models(candidate%n_outputs))
        do j = 1, candidate%n_outputs
            if (present(validation_x)) then
                if (present(sample_weight)) then
                    if (present(validation_weight)) then
                        call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                            options, sample_weight, validation_x, validation_targets(:, j), &
                            validation_weight)
                    else
                        call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                            options, sample_weight, validation_x, validation_targets(:, j))
                    end if
                else if (present(validation_weight)) then
                    call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                        options, validation_x=validation_x, &
                        validation_y=validation_targets(:, j), &
                        validation_weight=validation_weight)
                else
                    call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                        options, validation_x=validation_x, validation_y=validation_targets(:, j))
                end if
            else if (present(sample_weight)) then
                call candidate%models(j)%fit_regression(x, targets(:, j), status, &
                    options, sample_weight)
            else
                call candidate%models(j)%fit_regression(x, targets(:, j), status, options)
            end if
            if (status%code /= FORTNUM_OK) return
        end do
        candidate%initialized = .true.
        self%n_inputs = candidate%n_inputs
        self%n_outputs = candidate%n_outputs
        self%models = candidate%models
        self%initialized = candidate%initialized
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multi_fit

    subroutine lgbm_multi_predict(self, x, values, status)
        class(lightgbm_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :)
        integer :: j

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output predict: model or output shape is invalid")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)))
        do j = 1, self%n_outputs
            call self%models(j)%predict(x, candidate(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        values = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multi_predict

    subroutine lgbm_multi_predict_margin(self, x, values, status)
        class(lightgbm_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :)
        integer :: j

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output predict_margin: model or output shape is invalid")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)))
        do j = 1, self%n_outputs
            call self%models(j)%predict_margin(x, candidate(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        values = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multi_predict_margin

    subroutine lgbm_multi_predict_staged_margin(self, x, staged, status)
        class(lightgbm_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: staged(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :, :), child(:, :)
        integer :: j, n_estimators

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(staged, 1) /= size(x, 1) .or. size(staged, 3) /= self%n_outputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output staged margin: model or output shape is invalid")
            return
        end if
        n_estimators = self%models(1)%estimator_count()
        if (n_estimators < 1 .or. size(staged, 2) /= n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output staged margin: stage count is invalid")
            return
        end if
        do j = 2, self%n_outputs
            if (self%models(j)%estimator_count() /= n_estimators) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm multi-output staged margin: child stage counts differ")
                return
            end if
        end do
        allocate(candidate(size(staged, 1), size(staged, 2), size(staged, 3)))
        allocate(child(size(x, 1), n_estimators))
        do j = 1, self%n_outputs
            call self%models(j)%predict_staged_margin(x, child, status)
            if (status%code /= FORTNUM_OK) return
            candidate(:, :, j) = child
        end do
        staged = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multi_predict_staged_margin

    subroutine lgbm_multi_predict_jvp(self, x, x_dot, values, values_dot, status)
        class(lightgbm_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(inout) :: values(:, :), values_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), candidate_dot(:, :)
        integer :: j

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values) .or. &
            any(shape(x_dot) /= shape(x)) .or. any(shape(values_dot) /= shape(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output predict_jvp: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output predict_jvp: tangent is not finite")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)), &
            candidate_dot(size(values_dot, 1), size(values_dot, 2)))
        do j = 1, self%n_outputs
            call self%models(j)%predict_jvp(x, x_dot, candidate(:, j), &
                candidate_dot(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        values = candidate
        values_dot = candidate_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multi_predict_jvp

    subroutine lgbm_multi_predict_vjp(self, x, output_bar, x_bar, status)
        class(lightgbm_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(inout) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), child_bar(:, :)
        integer :: j

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(output_bar, 1) /= size(x, 1) .or. size(output_bar, 2) /= self%n_outputs .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output predict_vjp: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output predict_vjp: cotangent is not finite")
            return
        end if
        allocate(candidate(size(x, 1), size(x, 2)), child_bar(size(x, 1), size(x, 2)))
        candidate = 0.0_dp
        do j = 1, self%n_outputs
            call self%models(j)%predict_vjp(x, output_bar(:, j), child_bar, status)
            if (status%code /= FORTNUM_OK) return
            candidate = candidate + child_bar
        end do
        x_bar = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multi_predict_vjp

    subroutine lgbm_multi_predict_leaf_jvp(self, x, parameter_dot, values, values_dot, status)
        class(lightgbm_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), parameter_dot(:)
        real(dp), intent(inout) :: values(:, :), values_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), candidate_dot(:, :), p(:)
        integer :: j, first, last, count

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values) .or. &
            any(shape(values_dot) /= shape(values)) .or. &
            size(parameter_dot) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(parameter_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output predict_leaf_jvp: shape or values are invalid")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)), &
            candidate_dot(size(values_dot, 1), size(values_dot, 2)))
        first = 1
        do j = 1, self%n_outputs
            count = self%models(j)%leaf_parameter_count()
            last = first + count - 1
            allocate(p(count))
            p = parameter_dot(first:last)
            call self%models(j)%predict_leaf_jvp(x, p, candidate(:, j), &
                candidate_dot(:, j), status)
            deallocate(p)
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        values = candidate
        values_dot = candidate_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multi_predict_leaf_jvp

    subroutine lgbm_multi_predict_leaf_vjp(self, x, output_bar, parameter_bar, status)
        class(lightgbm_multioutput_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(inout) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:), p(:)
        integer :: j, first, last, count

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(output_bar, 1) /= size(x, 1) .or. size(output_bar, 2) /= self%n_outputs .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output predict_leaf_vjp: shape or values are invalid")
            return
        end if
        allocate(candidate(size(parameter_bar)))
        candidate = 0.0_dp
        first = 1
        do j = 1, self%n_outputs
            count = self%models(j)%leaf_parameter_count()
            last = first + count - 1
            allocate(p(count))
            call self%models(j)%predict_leaf_vjp(x, output_bar(:, j), p, status)
            if (status%code /= FORTNUM_OK) then
                deallocate(p)
                return
            end if
            candidate(first:last) = p
            deallocate(p)
            first = last + 1
        end do
        parameter_bar = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multi_predict_leaf_vjp

    subroutine lgbm_multi_predict_device(self, device, x, values, status)
        class(lightgbm_multioutput_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :)
        integer :: j

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output device prediction: model or shape is invalid")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output device prediction: device is not selected")
            return
        end if
        if (device%kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "lightgbm multi-output device prediction: no resident CUDA tree kernel is linked")
            return
        end if
        if (device%kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output device prediction: device kind is invalid")
            return
        end if
        allocate(candidate(size(values, 1), size(values, 2)))
        do j = 1, self%n_outputs
            call self%models(j)%predict_device(device, x, candidate(:, j), status)
            if (status%code /= FORTNUM_OK) return
        end do
        values = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_multi_predict_device

    subroutine lgbm_multi_predict_device_margin(self, device, x, values, status)
        class(lightgbm_multioutput_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: values(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. valid_query(self%initialized, self%n_inputs, self%n_outputs, x, values)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output device margin: model or shape is invalid")
            return
        end if
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output device margin: device is not selected")
            return
        end if
        if (device%kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "lightgbm multi-output device margin: no resident CUDA histogram kernel is linked")
            return
        end if
        if (device%kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output device margin: device kind is invalid")
            return
        end if
        call lgbm_multi_predict_margin(self, x, values, status)
    end subroutine lgbm_multi_predict_device_margin

    logical function lgbm_multi_device_supported(self, device_kind) result(supported)
        class(lightgbm_multioutput_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%initialized .and. device_kind == FORTML_DEVICE_CPU
    end function lgbm_multi_device_supported

    integer function lgbm_multi_feature_count(self) result(value)
        class(lightgbm_multioutput_t), intent(in) :: self
        value = self%n_inputs
    end function lgbm_multi_feature_count

    integer function lgbm_multi_output_count(self) result(value)
        class(lightgbm_multioutput_t), intent(in) :: self
        value = self%n_outputs
    end function lgbm_multi_output_count

    integer function lgbm_multi_estimator_count(self) result(value)
        class(lightgbm_multioutput_t), intent(in) :: self
        value = 0
        if (self%initialized .and. allocated(self%models)) value = self%models(1)%estimator_count()
    end function lgbm_multi_estimator_count

    function lgbm_multi_best_iteration(self) result(values)
        class(lightgbm_multioutput_t), intent(in) :: self
        integer, allocatable :: values(:)
        integer :: j

        allocate(values(max(0, self%n_outputs)))
        values = 0
        if (.not. self%initialized .or. .not. allocated(self%models)) return
        do j = 1, self%n_outputs
            values(j) = self%models(j)%best_iteration()
        end do
    end function lgbm_multi_best_iteration

    function lgbm_multi_best_validation_loss(self) result(values)
        class(lightgbm_multioutput_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: j

        allocate(values(max(0, self%n_outputs)))
        values = huge(1.0_dp)
        if (.not. self%initialized .or. .not. allocated(self%models)) return
        do j = 1, self%n_outputs
            values(j) = self%models(j)%best_validation_loss()
        end do
    end function lgbm_multi_best_validation_loss

    function lgbm_multi_early_stopped(self) result(values)
        class(lightgbm_multioutput_t), intent(in) :: self
        logical, allocatable :: values(:)
        integer :: j

        allocate(values(max(0, self%n_outputs)))
        values = .false.
        if (.not. self%initialized .or. .not. allocated(self%models)) return
        do j = 1, self%n_outputs
            values(j) = self%models(j)%early_stopped()
        end do
    end function lgbm_multi_early_stopped

    integer function lgbm_multi_parameter_count(self) result(value)
        class(lightgbm_multioutput_t), intent(in) :: self
        integer :: j
        value = 0
        if (.not. self%initialized .or. .not. allocated(self%models)) return
        do j = 1, size(self%models)
            value = value + self%models(j)%leaf_parameter_count()
        end do
    end function lgbm_multi_parameter_count

    function lgbm_multi_leaf_parameters(self, status) result(parameters)
        class(lightgbm_multioutput_t), intent(in) :: self
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: parameters(:), p(:)
        integer :: j, first, last, count

        allocate(parameters(max(0, self%parameter_count())))
        if (.not. self%initialized .or. .not. allocated(self%models)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm multi-output leaf_parameters: model is not initialized")
            return
        end if
        first = 1
        do j = 1, self%n_outputs
            p = self%models(j)%leaf_parameters(status)
            if (status%code /= FORTNUM_OK) return
            count = size(p)
            last = first + count - 1
            parameters(first:last) = p
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end function lgbm_multi_leaf_parameters

    logical function lgbm_multi_fitted(self) result(value)
        class(lightgbm_multioutput_t), intent(in) :: self
        value = self%initialized .and. self%n_inputs > 0 .and. self%n_outputs > 0 .and. &
            allocated(self%models)
    end function lgbm_multi_fitted

    logical function valid_fit_shape(x, targets, sample_weight, validation_x, &
            validation_targets, validation_weight) result(valid)
        real(dp), intent(in) :: x(:, :), targets(:, :)
        real(dp), intent(in), optional :: sample_weight(:), validation_x(:, :), &
            validation_targets(:, :), validation_weight(:)

        valid = size(x, 1) >= 2 .and. size(x, 2) >= 1 .and. size(targets, 1) == size(x, 1) .and. &
            size(targets, 2) >= 1 .and. all(ieee_is_finite(x)) .and. all(ieee_is_finite(targets))
        if (present(sample_weight)) valid = valid .and. size(sample_weight) == size(x, 1) .and. &
            all(ieee_is_finite(sample_weight)) .and. all(sample_weight > 0.0_dp)
        if (present(validation_x)) then
            if (.not. present(validation_targets)) then
                valid = .false.
            else
                valid = valid .and. size(validation_x, 1) >= 1 .and. &
                    size(validation_targets, 1) == size(validation_x, 1) .and. &
                    all(ieee_is_finite(validation_x)) .and. all(ieee_is_finite(validation_targets))
            end if
        else if (present(validation_targets)) then
            valid = .false.
        end if
        if (present(validation_weight)) then
            if (.not. present(validation_x)) then
                valid = .false.
            else
                valid = valid .and. size(validation_weight) == size(validation_x, 1) .and. &
                    all(ieee_is_finite(validation_weight)) .and. all(validation_weight > 0.0_dp)
            end if
        end if
    end function valid_fit_shape

    logical function valid_query(initialized, n_inputs, n_outputs, x, values) result(valid)
        logical, intent(in) :: initialized
        integer, intent(in) :: n_inputs, n_outputs
        real(dp), intent(in) :: x(:, :), values(:, :)

        valid = initialized .and. size(x, 2) == n_inputs .and. &
            size(values, 1) == size(x, 1) .and. size(values, 2) == n_outputs
    end function valid_query

end module fortml_xgboost_multioutput
