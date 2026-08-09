module fortml_linear_sgd
    !! Deterministic mini-batch linear SGD estimators.
    !!
    !! This module deliberately owns a stochastic objective separate from the
    !! exact logistic/softmax estimators.  A call to `fit` starts from zero and
    !! consumes a reproducible epoch schedule.  `partial_fit` consumes exactly
    !! one epoch and retains parameters, the shuffle stream, and Polyak sums,
    !! making streaming continuation explicit rather than implicit.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use, intrinsic :: iso_fortran_env, only: int64
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_losses, only: stable_sigmoid
    implicit none
    private

    integer, parameter, public :: FORTML_SGD_SCHEDULE_CONSTANT = 1
    integer, parameter, public :: FORTML_SGD_SCHEDULE_INVSCALING = 2

    type, public :: linear_sgd_options_t
        integer :: epochs = 100
        integer :: batch_size = 1
        integer :: shuffle_seed = 17
        integer :: schedule = FORTML_SGD_SCHEDULE_CONSTANT
        logical :: shuffle = .false.
        logical :: average = .false.
        logical :: fit_intercept = .true.
        real(dp) :: learning_rate = 1.0e-2_dp
        real(dp) :: power_t = 0.5_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: l1 = 0.0_dp
    end type linear_sgd_options_t

    type, public :: linear_sgd_regression_t
        private
        real(dp), allocatable :: coefficient(:, :)
        real(dp), allocatable :: intercept(:)
        real(dp), allocatable :: average_coefficient(:, :)
        real(dp), allocatable :: average_intercept(:)
        integer :: n_features_value = 0
        integer :: n_outputs_value = 0
        integer :: updates_value = 0
        integer(int64) :: shuffle_state = 17_int64
        real(dp) :: average_count = 0.0_dp
        type(linear_sgd_options_t) :: options_value
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit_matrix => sgd_regression_fit_matrix
        procedure, public :: fit_vector => sgd_regression_fit_vector
        generic, public :: fit => fit_matrix, fit_vector
        procedure, public :: partial_fit_matrix => sgd_regression_partial_fit_matrix
        procedure, public :: partial_fit_vector => sgd_regression_partial_fit_vector
        generic, public :: partial_fit => partial_fit_matrix, partial_fit_vector
        procedure, public :: predict_matrix => sgd_regression_predict_matrix
        procedure, public :: predict_vector => sgd_regression_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_device => sgd_regression_predict_device
        procedure, public :: coefficients => sgd_regression_coefficients
        procedure, public :: parameters => sgd_regression_parameters
        procedure, public :: set_parameters => sgd_regression_set_parameters
        procedure, public :: feature_count => sgd_regression_feature_count
        procedure, public :: output_count => sgd_regression_output_count
        procedure, public :: update_count => sgd_regression_update_count
        procedure, public :: fitted => sgd_regression_fitted
        procedure, public :: device_supported => sgd_regression_device_supported
    end type linear_sgd_regression_t

    type, public :: linear_sgd_classifier_t
        private
        real(dp), allocatable :: coefficient(:)
        real(dp) :: intercept = 0.0_dp
        real(dp), allocatable :: average_coefficient(:)
        real(dp) :: average_intercept = 0.0_dp
        integer :: class_label(2) = 0
        integer :: n_features_value = 0
        integer :: updates_value = 0
        integer(int64) :: shuffle_state = 17_int64
        real(dp) :: average_count = 0.0_dp
        type(linear_sgd_options_t) :: options_value
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => sgd_classifier_fit
        procedure, public :: partial_fit => sgd_classifier_partial_fit
        procedure, public :: decision_function => sgd_classifier_decision
        procedure, public :: predict_proba => sgd_classifier_predict_proba
        procedure, public :: predict => sgd_classifier_predict
        procedure, public :: predict_device => sgd_classifier_predict_device
        procedure, public :: coefficients => sgd_classifier_coefficients
        procedure, public :: parameters => sgd_classifier_parameters
        procedure, public :: set_parameters => sgd_classifier_set_parameters
        procedure, public :: classes => sgd_classifier_classes
        procedure, public :: feature_count => sgd_classifier_feature_count
        procedure, public :: update_count => sgd_classifier_update_count
        procedure, public :: fitted => sgd_classifier_fitted
        procedure, public :: device_supported => sgd_classifier_device_supported
    end type linear_sgd_classifier_t

contains

    subroutine sgd_regression_fit_matrix(self, x, y, status, options, sample_weight)
        class(linear_sgd_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(linear_sgd_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        type(linear_sgd_options_t) :: settings
        type(linear_sgd_options_t) :: linear_sgd_options_t_default
        integer :: epoch

        settings = linear_sgd_options_t_default
        if (present(options)) settings = options
        call validate_options(settings, status)
        if (.not. status_ok(status)) return
        call initialize_regression(self, x, y, settings, status)
        if (.not. status_ok(status)) return
        do epoch = 1, settings%epochs
            call regression_epoch(self, x, y, status, sample_weight)
            if (.not. status_ok(status)) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sgd_regression_fit_matrix

    subroutine sgd_regression_fit_vector(self, x, y, status, options, sample_weight)
        class(linear_sgd_regression_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(linear_sgd_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: target(:, :)

        allocate(target(size(y), 1))
        target(:, 1) = y
        call sgd_regression_fit_matrix(self, x, target, status, options, sample_weight)
    end subroutine sgd_regression_fit_vector

    subroutine sgd_regression_partial_fit_matrix(self, x, y, status, options, sample_weight)
        class(linear_sgd_regression_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        type(linear_sgd_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        type(linear_sgd_options_t) :: settings
        type(linear_sgd_options_t) :: linear_sgd_options_t_default

        settings = linear_sgd_options_t_default
        if (present(options)) settings = options
        if (self%fitted_value) then
            if (present(options)) then
                if (.not. same_options(self%options_value, settings)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "linear SGD partial_fit: options cannot change after initialization")
                    return
                end if
            end if
            settings = self%options_value
        else
            call validate_options(settings, status)
            if (.not. status_ok(status)) return
            call initialize_regression(self, x, y, settings, status)
            if (.not. status_ok(status)) return
        end if
        call regression_epoch(self, x, y, status, sample_weight)
    end subroutine sgd_regression_partial_fit_matrix

    subroutine sgd_regression_partial_fit_vector(self, x, y, status, options, sample_weight)
        class(linear_sgd_regression_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(linear_sgd_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: target(:, :)

        allocate(target(size(y), 1))
        target(:, 1) = y
        call sgd_regression_partial_fit_matrix(self, x, target, status, options, sample_weight)
    end subroutine sgd_regression_partial_fit_vector

    subroutine initialize_regression(self, x, y, options, status)
        class(linear_sgd_regression_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(linear_sgd_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status
        integer :: n_features, n_outputs

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(y, 1) /= size(x, 1) .or. &
            size(y, 2) < 1 .or. any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD regression: data dimensions or values are invalid")
            return
        end if
        n_features = size(x, 2)
        n_outputs = size(y, 2)
        allocate(self%coefficient(n_features, n_outputs), self%intercept(n_outputs), &
            self%average_coefficient(n_features, n_outputs), self%average_intercept(n_outputs))
        self%coefficient = 0.0_dp
        self%intercept = 0.0_dp
        self%average_coefficient = 0.0_dp
        self%average_intercept = 0.0_dp
        self%n_features_value = n_features
        self%n_outputs_value = n_outputs
        self%updates_value = 0
        self%average_count = 0.0_dp
        self%shuffle_state = int(options%shuffle_seed, int64)
        if (self%shuffle_state <= 0_int64) self%shuffle_state = 17_int64
        self%options_value = options
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine initialize_regression

    subroutine regression_epoch(self, x, y, status, sample_weight)
        class(linear_sgd_regression_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: order(:)
        real(dp), allocatable :: gradient(:, :), gradient_intercept(:), weights(:)
        real(dp) :: mass, rate, prediction, residual, weight
        integer :: n, batch_start, batch_end, batch_size, i, j, k, index

        if (.not. validate_regression_data(self, x, y, sample_weight, status)) return
        n = size(x, 1)
        allocate(order(n), weights(n), gradient(self%n_features_value, self%n_outputs_value), &
            gradient_intercept(self%n_outputs_value))
        order = [(i, i=1, n)]
        if (self%options_value%shuffle) call shuffle_order(order, self%shuffle_state)
        weights = 1.0_dp
        if (present(sample_weight)) weights = sample_weight
        batch_size = min(self%options_value%batch_size, n)
        batch_start = 1
        do while (batch_start <= n)
            batch_end = min(n, batch_start + batch_size - 1)
            mass = sum(weights(order(batch_start:batch_end)))
            if (mass <= 0.0_dp) then
                batch_start = batch_end + 1
                cycle
            end if
            gradient = 0.0_dp
            gradient_intercept = 0.0_dp
            do index = batch_start, batch_end
                i = order(index)
                weight = weights(i)
                do k = 1, self%n_outputs_value
                    prediction = self%intercept(k) + dot_product(x(i, :), self%coefficient(:, k))
                    residual = prediction-y(i, k)
                    gradient(:, k) = gradient(:, k) + weight*residual*x(i, :)
                    gradient_intercept(k) = gradient_intercept(k) + weight*residual
                end do
            end do
            gradient = gradient/mass
            gradient_intercept = gradient_intercept/mass
            rate = update_rate(self%options_value, self%updates_value+1)
            gradient = gradient + self%options_value%l2*self%coefficient
            self%coefficient = self%coefficient-rate*gradient
            if (self%options_value%l1 > 0.0_dp) then
                do k = 1, self%n_outputs_value
                    do j = 1, self%n_features_value
                        self%coefficient(j, k) = soft_threshold(self%coefficient(j, k), &
                            rate*self%options_value%l1)
                    end do
                end do
            end if
            if (self%options_value%fit_intercept) then
                self%intercept = self%intercept-rate*gradient_intercept
            else
                self%intercept = 0.0_dp
            end if
            self%updates_value = self%updates_value + 1
            if (self%options_value%average) then
                self%average_count = self%average_count + 1.0_dp
                self%average_coefficient = self%average_coefficient + self%coefficient
                self%average_intercept = self%average_intercept + self%intercept
            end if
            batch_start = batch_end + 1
        end do
        deallocate(order, weights, gradient, gradient_intercept)
        call status_set(status, FORTNUM_OK, "")
    end subroutine regression_epoch

    subroutine sgd_regression_predict_matrix(self, x, y, status)
        class(linear_sgd_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coefficient(:, :), intercept(:)

        if (.not. self%fitted_value .or. size(x, 2) /= self%n_features_value .or. &
            any(shape(y) /= [size(x, 1), self%n_outputs_value]) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD regression predict: model or shape is invalid")
            return
        end if
        call regression_effective_parameters(self, coefficient, intercept)
        y = matmul(x, coefficient)
        y = y + spread(intercept, 1, size(x, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine sgd_regression_predict_matrix

    subroutine sgd_regression_predict_vector(self, x, y, status)
        class(linear_sgd_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD regression predict: output shape is invalid")
            return
        end if
        allocate(values(size(y), 1))
        call sgd_regression_predict_matrix(self, x, values, status)
        if (status_ok(status)) y = values(:, 1)
    end subroutine sgd_regression_predict_vector

    subroutine regression_effective_parameters(self, coefficient, intercept)
        class(linear_sgd_regression_t), intent(in) :: self
        real(dp), allocatable, intent(out) :: coefficient(:, :), intercept(:)

        allocate(coefficient(self%n_features_value, self%n_outputs_value), &
            intercept(self%n_outputs_value))
        if (self%options_value%average .and. self%average_count > 0.0_dp) then
            coefficient = self%average_coefficient/self%average_count
            intercept = self%average_intercept/self%average_count
        else
            coefficient = self%coefficient
            intercept = self%intercept
        end if
    end subroutine regression_effective_parameters

    subroutine sgd_regression_predict_device(self, device, x, y, status)
        class(linear_sgd_regression_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (device%kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "linear SGD regression predict: resident CUDA kernel is unavailable")
            return
        end if
        call sgd_regression_predict_matrix(self, x, y, status)
    end subroutine sgd_regression_predict_device

    function sgd_regression_coefficients(self) result(values)
        class(linear_sgd_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        real(dp), allocatable :: coefficient(:, :), intercept(:)

        call regression_effective_parameters(self, coefficient, intercept)
        allocate(values(self%n_features_value+1, self%n_outputs_value))
        values(1, :) = intercept
        values(2:, :) = coefficient
    end function sgd_regression_coefficients

    function sgd_regression_parameters(self) result(values)
        class(linear_sgd_regression_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        real(dp), allocatable :: coefficients(:, :)

        coefficients = self%coefficients()
        allocate(values(size(coefficients)))
        values = reshape(coefficients, [size(values)])
    end function sgd_regression_parameters

    subroutine sgd_regression_set_parameters(self, values, status)
        class(linear_sgd_regression_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted_value .or. size(values) /= (self%n_features_value+1)*self%n_outputs_value .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD regression set_parameters: shape is invalid")
            return
        end if
        block
            real(dp) :: packed(self%n_features_value+1, self%n_outputs_value)
            packed = reshape(values, shape(packed))
            self%intercept = packed(1, :)
            self%coefficient = packed(2:, :)
        end block
        self%average_count = 0.0_dp
        self%average_coefficient = 0.0_dp
        self%average_intercept = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine sgd_regression_set_parameters

    integer function sgd_regression_feature_count(self) result(value)
        class(linear_sgd_regression_t), intent(in) :: self
        value = self%n_features_value
    end function sgd_regression_feature_count

    integer function sgd_regression_output_count(self) result(value)
        class(linear_sgd_regression_t), intent(in) :: self
        value = self%n_outputs_value
    end function sgd_regression_output_count

    integer function sgd_regression_update_count(self) result(value)
        class(linear_sgd_regression_t), intent(in) :: self
        value = self%updates_value
    end function sgd_regression_update_count

    logical function sgd_regression_fitted(self) result(value)
        class(linear_sgd_regression_t), intent(in) :: self
        value = self%fitted_value .and. allocated(self%coefficient)
    end function sgd_regression_fitted

    logical function sgd_regression_device_supported(self, device_kind) result(value)
        class(linear_sgd_regression_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = device_kind == FORTML_DEVICE_CPU .and. self%fitted()
    end function sgd_regression_device_supported

    subroutine sgd_classifier_fit(self, x, labels, status, options, classes, sample_weight)
        class(linear_sgd_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(linear_sgd_options_t), intent(in), optional :: options
        integer, intent(in), optional :: classes(:)
        real(dp), intent(in), optional :: sample_weight(:)
        type(linear_sgd_options_t) :: settings
        type(linear_sgd_options_t) :: linear_sgd_options_t_default
        integer :: epoch

        settings = linear_sgd_options_t_default
        if (present(options)) settings = options
        call validate_options(settings, status)
        if (.not. status_ok(status)) return
        call initialize_classifier(self, x, labels, settings, classes, status)
        if (.not. status_ok(status)) return
        do epoch = 1, settings%epochs
            call classifier_epoch(self, x, labels, status, sample_weight)
            if (.not. status_ok(status)) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sgd_classifier_fit

    subroutine sgd_classifier_partial_fit(self, x, labels, status, options, classes, sample_weight)
        class(linear_sgd_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(linear_sgd_options_t), intent(in), optional :: options
        integer, intent(in), optional :: classes(:)
        real(dp), intent(in), optional :: sample_weight(:)
        type(linear_sgd_options_t) :: settings
        type(linear_sgd_options_t) :: linear_sgd_options_t_default

        settings = linear_sgd_options_t_default
        if (present(options)) settings = options
        if (.not. self%fitted_value) then
            call validate_options(settings, status)
            if (.not. status_ok(status)) return
            call initialize_classifier(self, x, labels, settings, classes, status)
            if (.not. status_ok(status)) return
        else
            if (present(options)) then
                if (.not. same_options(self%options_value, settings)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "linear SGD classifier partial_fit: options cannot change")
                    return
                end if
            end if
            settings = self%options_value
            if (present(classes)) then
                if (size(classes) /= 2 .or. any(classes /= self%class_label)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "linear SGD classifier partial_fit: classes cannot change")
                    return
                end if
            end if
        end if
        call classifier_epoch(self, x, labels, status, sample_weight)
    end subroutine sgd_classifier_partial_fit

    subroutine initialize_classifier(self, x, labels, options, classes, status)
        class(linear_sgd_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(linear_sgd_options_t), intent(in) :: options
        integer, intent(in), optional :: classes(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: low, high, i

        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(labels) /= size(x, 1) .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD classifier: data dimensions or values are invalid")
            return
        end if
        if (present(classes)) then
            if (size(classes) /= 2 .or. classes(1) >= classes(2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SGD classifier: classes must be two sorted labels")
                return
            end if
            self%class_label = classes
        else
            low = minval(labels)
            high = maxval(labels)
            if (low == high) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SGD classifier: two classes are required")
                return
            end if
            self%class_label = [low, high]
        end if
        do i = 1, size(labels)
            if (labels(i) /= self%class_label(1) .and. labels(i) /= self%class_label(2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SGD classifier: label is outside the declared classes")
                return
            end if
        end do
        allocate(self%coefficient(size(x, 2)), self%average_coefficient(size(x, 2)))
        self%coefficient = 0.0_dp
        self%intercept = 0.0_dp
        self%average_coefficient = 0.0_dp
        self%average_intercept = 0.0_dp
        self%n_features_value = size(x, 2)
        self%updates_value = 0
        self%average_count = 0.0_dp
        self%shuffle_state = int(options%shuffle_seed, int64)
        if (self%shuffle_state <= 0_int64) self%shuffle_state = 17_int64
        self%options_value = options
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine initialize_classifier

    subroutine classifier_epoch(self, x, labels, status, sample_weight)
        class(linear_sgd_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: order(:)
        real(dp), allocatable :: gradient(:), weights(:)
        real(dp) :: mass, rate, score, residual, weight, gradient_intercept
        integer :: n, batch_start, batch_end, i, j, index, batch_size

        if (.not. validate_classifier_data(self, x, labels, sample_weight, status)) return
        n = size(x, 1)
        allocate(order(n), weights(n), gradient(self%n_features_value))
        order = [(i, i=1, n)]
        if (self%options_value%shuffle) call shuffle_order(order, self%shuffle_state)
        weights = 1.0_dp
        if (present(sample_weight)) weights = sample_weight
        batch_size = min(self%options_value%batch_size, n)
        batch_start = 1
        do while (batch_start <= n)
            batch_end = min(n, batch_start + batch_size - 1)
            mass = sum(weights(order(batch_start:batch_end)))
            if (mass <= 0.0_dp) then
                batch_start = batch_end + 1
                cycle
            end if
            gradient = 0.0_dp
            gradient_intercept = 0.0_dp
            do index = batch_start, batch_end
                i = order(index)
                weight = weights(i)
                score = self%intercept + dot_product(x(i, :), self%coefficient)
                residual = stable_sigmoid(score) - merge(1.0_dp, 0.0_dp, &
                    labels(i) == self%class_label(2))
                gradient = gradient + weight*residual*x(i, :)
                gradient_intercept = gradient_intercept + weight*residual
            end do
            gradient = gradient/mass + self%options_value%l2*self%coefficient
            gradient_intercept = gradient_intercept/mass
            rate = update_rate(self%options_value, self%updates_value+1)
            self%coefficient = self%coefficient-rate*gradient
            if (self%options_value%l1 > 0.0_dp) then
                do j = 1, self%n_features_value
                    self%coefficient(j) = soft_threshold(self%coefficient(j), &
                        rate*self%options_value%l1)
                end do
            end if
            if (self%options_value%fit_intercept) then
                self%intercept = self%intercept-rate*gradient_intercept
            else
                self%intercept = 0.0_dp
            end if
            self%updates_value = self%updates_value + 1
            if (self%options_value%average) then
                self%average_count = self%average_count + 1.0_dp
                self%average_coefficient = self%average_coefficient + self%coefficient
                self%average_intercept = self%average_intercept + self%intercept
            end if
            batch_start = batch_end + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classifier_epoch

    subroutine sgd_classifier_decision(self, x, scores, status)
        class(linear_sgd_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coefficient(:)
        real(dp) :: intercept
        integer :: i

        if (.not. self%fitted_value .or. size(x, 2) /= self%n_features_value .or. &
            size(scores) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD classifier decision: model or shape is invalid")
            return
        end if
        call classifier_effective_parameters(self, coefficient, intercept)
        scores = matmul(x, coefficient)
        scores = scores + [(intercept, i=1, size(x, 1))]
        call status_set(status, FORTNUM_OK, "")
    end subroutine sgd_classifier_decision

    subroutine sgd_classifier_predict_proba(self, x, probabilities, status)
        class(linear_sgd_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)

        if (any(shape(probabilities) /= [size(x, 1), 2])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD classifier probabilities: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        probabilities(:, 2) = stable_sigmoid(scores)
        probabilities(:, 1) = 1.0_dp-probabilities(:, 2)
        call status_set(status, FORTNUM_OK, "")
    end subroutine sgd_classifier_predict_proba

    subroutine sgd_classifier_predict(self, x, labels, status)
        class(linear_sgd_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD classifier predict: output shape is invalid")
            return
        end if
        allocate(scores(size(labels)))
        call self%decision_function(x, scores, status)
        if (.not. status_ok(status)) return
        do i = 1, size(labels)
            labels(i) = merge(self%class_label(2), self%class_label(1), scores(i) >= 0.0_dp)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sgd_classifier_predict

    subroutine classifier_effective_parameters(self, coefficient, intercept)
        class(linear_sgd_classifier_t), intent(in) :: self
        real(dp), allocatable, intent(out) :: coefficient(:)
        real(dp), intent(out) :: intercept

        allocate(coefficient(self%n_features_value))
        if (self%options_value%average .and. self%average_count > 0.0_dp) then
            coefficient = self%average_coefficient/self%average_count
            intercept = self%average_intercept/self%average_count
        else
            coefficient = self%coefficient
            intercept = self%intercept
        end if
    end subroutine classifier_effective_parameters

    subroutine sgd_classifier_predict_device(self, device, x, probabilities, status)
        class(linear_sgd_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (device%kind == FORTML_DEVICE_CUDA) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "linear SGD classifier predict: resident CUDA kernel is unavailable")
            return
        end if
        call sgd_classifier_predict_proba(self, x, probabilities, status)
    end subroutine sgd_classifier_predict_device

    function sgd_classifier_coefficients(self) result(values)
        class(linear_sgd_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        real(dp), allocatable :: coefficient(:)
        real(dp) :: intercept

        call classifier_effective_parameters(self, coefficient, intercept)
        allocate(values(self%n_features_value+1))
        values(1) = intercept
        values(2:) = coefficient
    end function sgd_classifier_coefficients

    function sgd_classifier_parameters(self) result(values)
        class(linear_sgd_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        values = self%coefficients()
    end function sgd_classifier_parameters

    subroutine sgd_classifier_set_parameters(self, values, status)
        class(linear_sgd_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted_value .or. size(values) /= self%n_features_value+1 .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD classifier set_parameters: shape is invalid")
            return
        end if
        self%intercept = values(1)
        self%coefficient = values(2:)
        self%average_count = 0.0_dp
        self%average_coefficient = 0.0_dp
        self%average_intercept = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine sgd_classifier_set_parameters

    function sgd_classifier_classes(self) result(values)
        class(linear_sgd_classifier_t), intent(in) :: self
        integer :: values(2)
        values = self%class_label
    end function sgd_classifier_classes

    integer function sgd_classifier_feature_count(self) result(value)
        class(linear_sgd_classifier_t), intent(in) :: self
        value = self%n_features_value
    end function sgd_classifier_feature_count

    integer function sgd_classifier_update_count(self) result(value)
        class(linear_sgd_classifier_t), intent(in) :: self
        value = self%updates_value
    end function sgd_classifier_update_count

    logical function sgd_classifier_fitted(self) result(value)
        class(linear_sgd_classifier_t), intent(in) :: self
        value = self%fitted_value .and. allocated(self%coefficient)
    end function sgd_classifier_fitted

    logical function sgd_classifier_device_supported(self, device_kind) result(value)
        class(linear_sgd_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = device_kind == FORTML_DEVICE_CPU .and. self%fitted()
    end function sgd_classifier_device_supported

    logical function validate_regression_data(self, x, y, sample_weight, status) result(valid)
        class(linear_sgd_regression_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), y(:, :)
        real(dp), intent(in), optional :: sample_weight(:)
        type(fortnum_status_t), intent(out) :: status
        valid = .false.
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features_value .or. &
            size(y, 1) /= size(x, 1) .or. size(y, 2) /= self%n_outputs_value .or. &
            any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD regression partial_fit: shape or values are invalid")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1) .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SGD regression partial_fit: weights are invalid")
                return
            end if
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function validate_regression_data

    logical function validate_classifier_data(self, x, labels, sample_weight, status) result(valid)
        class(linear_sgd_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(in), optional :: sample_weight(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i
        valid = .false.
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features_value .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear SGD classifier partial_fit: shape or values are invalid")
            return
        end if
        do i = 1, size(labels)
            if (labels(i) /= self%class_label(1) .and. labels(i) /= self%class_label(2)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SGD classifier partial_fit: unknown label")
                return
            end if
        end do
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1) .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "linear SGD classifier partial_fit: weights are invalid")
                return
            end if
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function validate_classifier_data

    subroutine validate_options(options, status)
        type(linear_sgd_options_t), intent(in) :: options
        type(fortnum_status_t), intent(out) :: status
        if (options%epochs < 1 .or. options%batch_size < 1 .or. options%learning_rate <= 0.0_dp .or. &
            .not. ieee_is_finite(options%learning_rate) .or. options%l1 < 0.0_dp .or. &
            options%l2 < 0.0_dp .or. .not. ieee_is_finite(options%l1) .or. &
            .not. ieee_is_finite(options%l2) .or. options%schedule < FORTML_SGD_SCHEDULE_CONSTANT .or. &
            options%schedule > FORTML_SGD_SCHEDULE_INVSCALING .or. &
            options%power_t < 0.0_dp .or. .not. ieee_is_finite(options%power_t)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "linear SGD: options are invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine validate_options

    real(dp) function update_rate(options, update) result(rate)
        type(linear_sgd_options_t), intent(in) :: options
        integer, intent(in) :: update
        rate = options%learning_rate
        if (options%schedule == FORTML_SGD_SCHEDULE_INVSCALING) then
            rate = rate*real(update, dp)**(-options%power_t)
        end if
    end function update_rate

    logical function same_options(left, right) result(equal)
        type(linear_sgd_options_t), intent(in) :: left, right
        equal = left%epochs == right%epochs .and. left%batch_size == right%batch_size .and. &
            left%shuffle_seed == right%shuffle_seed .and. left%schedule == right%schedule .and. &
            left%shuffle .eqv. right%shuffle .and. left%average .eqv. right%average .and. &
            left%fit_intercept .eqv. right%fit_intercept .and. left%learning_rate == right%learning_rate .and. &
            left%power_t == right%power_t .and. left%l1 == right%l1 .and. left%l2 == right%l2
    end function same_options

    subroutine shuffle_order(order, generator)
        integer, intent(inout) :: order(:)
        integer(int64), intent(inout) :: generator
        integer :: i, j, temporary
        do i = size(order), 2, -1
            generator = mod(48271_int64*generator, 2147483647_int64)
            j = 1 + int(mod(generator, int(i, int64)))
            temporary = order(i)
            order(i) = order(j)
            order(j) = temporary
        end do
    end subroutine shuffle_order

    pure real(dp) function soft_threshold(value, threshold) result(output)
        real(dp), intent(in) :: value, threshold
        if (value > threshold) then
            output = value-threshold
        else if (value < -threshold) then
            output = value+threshold
        else
            output = 0.0_dp
        end if
    end function soft_threshold

end module fortml_linear_sgd
