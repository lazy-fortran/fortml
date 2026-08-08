module fortml_classifier_chain
    !! Deterministic binary classifier chain built from logistic heads.
    !!
    !! Each output column has two arbitrary integer labels.  Heads are fitted
    !! in output order; head ``j`` receives the original features followed by
    !! the observed indicator columns ``1:j-1``.  Prediction uses the previous
    !! positive probabilities as the smooth chain features, while ``predict``
    !! uses thresholded labels.  Packed parameters are concatenated in head
    !! order, including the growing chain-feature blocks.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_logistic_regression, only: logistic_regression_t
    implicit none
    private

    type, public :: classifier_chain_t
        private
        type(logistic_regression_t), allocatable :: models(:)
        integer, allocatable :: class_label(:, :)
        real(dp), allocatable :: decision_threshold(:)
        integer, allocatable :: parameter_sizes(:)
        integer :: n_outputs = 0
        integer :: n_features = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => classifier_chain_fit
        procedure, public :: decision_function => classifier_chain_decision
        procedure, public :: decision_function_device => &
            classifier_chain_decision_device
        procedure, public :: predict_proba => classifier_chain_predict_proba
        procedure, public :: predict_proba_device => &
            classifier_chain_predict_proba_device
        procedure, public :: predict_proba_jvp => classifier_chain_predict_proba_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            classifier_chain_predict_proba_parameter_jvp
        procedure, public :: predict_proba_vjp => classifier_chain_predict_proba_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            classifier_chain_predict_proba_parameter_vjp
        procedure, public :: predict => classifier_chain_predict
        procedure, public :: predict_device => classifier_chain_predict_device
        procedure, public :: classes => classifier_chain_classes
        procedure, public :: output_count => classifier_chain_output_count
        procedure, public :: label_count => classifier_chain_output_count
        procedure, public :: feature_count => classifier_chain_feature_count
        procedure, public :: parameter_count => classifier_chain_parameter_count
        procedure, public :: parameters => classifier_chain_parameters
        procedure, public :: set_parameters => classifier_chain_set_parameters
        procedure, public :: thresholds => classifier_chain_thresholds
        procedure, public :: set_thresholds => classifier_chain_set_thresholds
        procedure, public :: fitted => classifier_chain_fitted
        procedure, public :: device_supported => classifier_chain_device_supported
    end type classifier_chain_t

    public :: classifier_chain_fit
    public :: classifier_chain_decision
    public :: classifier_chain_predict_proba
    public :: classifier_chain_predict_proba_jvp
    public :: classifier_chain_predict_proba_parameter_jvp
    public :: classifier_chain_predict_proba_vjp
    public :: classifier_chain_predict_proba_parameter_vjp
    public :: classifier_chain_predict

contains

    subroutine classifier_chain_fit(self, x, labels, status, l2, fit_intercept, &
            max_iterations, tolerance, sample_weight, class_weight, thresholds)
        class(classifier_chain_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: l2, tolerance
        logical, intent(in), optional :: fit_intercept
        integer, intent(in), optional :: max_iterations
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:, :)
        real(dp), intent(in), optional :: thresholds(:)
        integer, allocatable :: binary_labels(:), pair(:)
        real(dp), allocatable :: chain_x(:, :)
        integer :: i, j, n_samples, n_features, n_outputs, feature_count
        character(256) :: failure_message

        self%is_fitted = .false.
        self%n_outputs = 0
        self%n_features = 0
        if (allocated(self%models)) deallocate(self%models)
        if (allocated(self%class_label)) deallocate(self%class_label)
        if (allocated(self%decision_threshold)) deallocate(self%decision_threshold)
        if (allocated(self%parameter_sizes)) deallocate(self%parameter_sizes)
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels, 1) /= size(x, 1) .or. size(labels, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain fit: inputs must be finite")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_outputs = size(labels, 2)
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp) .or. sum(sample_weight) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "classifier chain fit: sample weights must be finite, "// &
                    "nonnegative, and have positive mass")
                return
            end if
        end if
        if (present(class_weight)) then
            if (any(shape(class_weight) /= [2, n_outputs]) .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "classifier chain fit: class_weight must have shape "// &
                    "(2,n_outputs) and contain positive finite values")
                return
            end if
        end if
        allocate(self%models(n_outputs), self%class_label(2, n_outputs), &
            self%decision_threshold(n_outputs), self%parameter_sizes(n_outputs), &
            binary_labels(n_samples), pair(2))
        self%decision_threshold = 0.5_dp
        if (present(thresholds)) then
            if (size(thresholds) /= n_outputs .or. &
                any(.not. ieee_is_finite(thresholds)) .or. &
                any(thresholds <= 0.0_dp) .or. any(thresholds >= 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "classifier chain fit: thresholds must be finite in (0,1)")
                return
            end if
            self%decision_threshold = thresholds
        end if
        self%n_features = n_features
        self%n_outputs = n_outputs
        do j = 1, n_outputs
            call sorted_binary_labels(labels(:, j), pair, status)
            if (status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "classifier chain fit: each output must have exactly two "// &
                    "distinct integer labels")
                return
            end if
            self%class_label(:, j) = pair
        end do
        allocate(chain_x(n_samples, n_features+n_outputs-1))
        chain_x = 0.0_dp
        chain_x(:, :n_features) = x
        do j = 1, n_outputs
            feature_count = n_features + j - 1
            if (j > 1) then
                do i = 1, j - 1
                    chain_x(:, n_features+i) = 0.0_dp
                    where (labels(:, i) == self%class_label(2, i)) &
                        chain_x(:, n_features+i) = 1.0_dp
                end do
            end if
            binary_labels = 0
            where (labels(:, j) == self%class_label(2, j)) binary_labels = 1
            if (present(class_weight)) then
                call self%models(j)%fit(chain_x(:, :feature_count), binary_labels, &
                    status, l2=l2, fit_intercept=fit_intercept, &
                    max_iterations=max_iterations, tolerance=tolerance, &
                    sample_weight=sample_weight, class_weight=class_weight(:, j))
            else
                call self%models(j)%fit(chain_x(:, :feature_count), binary_labels, &
                    status, l2=l2, fit_intercept=fit_intercept, &
                    max_iterations=max_iterations, tolerance=tolerance, &
                    sample_weight=sample_weight)
            end if
            if (status%code /= FORTNUM_OK) then
                write (failure_message, '(a,i0,a,a)') &
                    "classifier chain fit: logistic head ", j, " failed: ", &
                    trim(status%msg)
                call status_set(status, FORTNUM_DOMAIN_ERROR, trim(failure_message))
                return
            end if
        end do
        do j = 1, n_outputs
            self%parameter_sizes(j) = self%models(j)%parameter_count()
        end do
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine classifier_chain_fit

    subroutine classifier_chain_decision(self, x, scores, status)
        class(classifier_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: augmented(:, :), probabilities(:, :)

        if (.not. valid_query(self, x, scores, status, "classifier chain decision")) return
        allocate(probabilities(size(x, 1), self%n_outputs))
        call forward_values(self, x, augmented, scores, probabilities, status)
    end subroutine classifier_chain_decision

    subroutine classifier_chain_predict_proba(self, x, probabilities, status)
        class(classifier_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: augmented(:, :), scores(:, :)

        if (.not. valid_query(self, x, probabilities, status, &
            "classifier chain probability")) return
        allocate(scores(size(x, 1), self%n_outputs))
        call forward_values(self, x, augmented, scores, probabilities, status)
    end subroutine classifier_chain_predict_proba

    subroutine classifier_chain_decision_device(self, device, x, scores, status)
        class(classifier_chain_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "classifier chain device decision: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain device decision: device kind is invalid")
        end select
    end subroutine classifier_chain_decision_device

    subroutine classifier_chain_predict_proba_device(self, device, x, probabilities, status)
        class(classifier_chain_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "classifier chain device probability: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain device probability: device kind is invalid")
        end select
    end subroutine classifier_chain_predict_proba_device

    subroutine classifier_chain_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(classifier_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: augmented(:, :), augmented_dot(:, :)
        real(dp), allocatable :: binary_probabilities(:, :), binary_dot(:, :)
        real(dp), allocatable :: theta_dot(:)
        integer :: j, feature_count

        if (.not. valid_query(self, x, probabilities, status, &
            "classifier chain probability JVP")) return
        if (any(shape(x_dot) /= shape(x)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain probability JVP: tangent or output shape is invalid")
            return
        end if
        allocate(augmented(size(x, 1), self%n_features+self%n_outputs-1), &
            augmented_dot(size(x, 1), self%n_features+self%n_outputs-1), &
            binary_probabilities(size(x, 1), 2), binary_dot(size(x, 1), 2))
        augmented = 0.0_dp
        augmented_dot = 0.0_dp
        augmented(:, :self%n_features) = x
        augmented_dot(:, :self%n_features) = x_dot
        do j = 1, self%n_outputs
            feature_count = self%n_features + j - 1
            allocate(theta_dot(self%parameter_sizes(j)))
            theta_dot = 0.0_dp
            call self%models(j)%predict_proba_jvp(augmented(:, :feature_count), theta_dot, &
                augmented_dot(:, :feature_count), binary_probabilities, binary_dot, status)
            deallocate(theta_dot)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
            probabilities_dot(:, j) = binary_dot(:, 2)
            if (j < self%n_outputs) then
                augmented(:, self%n_features+j) = probabilities(:, j)
                augmented_dot(:, self%n_features+j) = probabilities_dot(:, j)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classifier_chain_predict_proba_jvp

    subroutine classifier_chain_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(classifier_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: augmented(:, :), augmented_dot(:, :)
        real(dp), allocatable :: binary_probabilities(:, :), binary_dot(:, :)
        real(dp), allocatable :: theta_slice(:)
        integer :: j, feature_count, first, last

        if (.not. valid_query(self, x, probabilities, status, &
            "classifier chain parameter JVP")) return
        if (size(theta_dot) /= self%parameter_count() .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain parameter JVP: parameter tangent or output shape is invalid")
            return
        end if
        allocate(augmented(size(x, 1), self%n_features+self%n_outputs-1), &
            augmented_dot(size(x, 1), self%n_features+self%n_outputs-1), &
            binary_probabilities(size(x, 1), 2), binary_dot(size(x, 1), 2))
        augmented = 0.0_dp
        augmented_dot = 0.0_dp
        augmented(:, :self%n_features) = x
        do j = 1, self%n_outputs
            feature_count = self%n_features + j - 1
            first = packed_first(self, j)
            last = packed_last(self, j)
            allocate(theta_slice(self%parameter_sizes(j)))
            theta_slice = theta_dot(first:last)
            call self%models(j)%predict_proba_jvp(augmented(:, :feature_count), theta_slice, &
                augmented_dot(:, :feature_count), binary_probabilities, binary_dot, status)
            deallocate(theta_slice)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
            probabilities_dot(:, j) = binary_dot(:, 2)
            if (j < self%n_outputs) then
                augmented(:, self%n_features+j) = probabilities(:, j)
                augmented_dot(:, self%n_features+j) = probabilities_dot(:, j)
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classifier_chain_predict_proba_parameter_jvp

    subroutine classifier_chain_predict_proba_vjp(self, x, probabilities_bar, x_bar, status)
        class(classifier_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: augmented(:, :), scores(:, :), probabilities(:, :)
        real(dp), allocatable :: probability_bar_work(:, :), binary_bar(:, :)
        real(dp), allocatable :: head_theta_bar(:), head_augmented_bar(:, :)
        integer :: j, feature_count

        x_bar = 0.0_dp
        if (.not. valid_query(self, x, probabilities_bar, status, &
            "classifier chain probability VJP")) return
        if (any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain probability VJP: cotangent or output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%n_outputs), &
            probabilities(size(x, 1), self%n_outputs))
        call forward_values(self, x, augmented, scores, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        allocate(probability_bar_work(size(x, 1), self%n_outputs), &
            binary_bar(size(x, 1), 2))
        probability_bar_work = probabilities_bar
        do j = self%n_outputs, 1, -1
            feature_count = self%n_features + j - 1
            binary_bar(:, 1) = 0.0_dp
            binary_bar(:, 2) = probability_bar_work(:, j)
            allocate(head_theta_bar(self%parameter_sizes(j)))
            allocate(head_augmented_bar(size(x, 1), feature_count))
            call self%models(j)%predict_proba_vjp(augmented(:, :feature_count), binary_bar, &
                head_theta_bar, head_augmented_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + head_augmented_bar(:, :self%n_features)
            if (j > 1) probability_bar_work(:, :j-1) = probability_bar_work(:, :j-1) + &
                head_augmented_bar(:, self%n_features+1:self%n_features+j-1)
            deallocate(head_theta_bar)
            deallocate(head_augmented_bar)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classifier_chain_predict_proba_vjp

    subroutine classifier_chain_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(classifier_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: augmented(:, :), scores(:, :), probabilities(:, :)
        real(dp), allocatable :: probability_bar_work(:, :), binary_bar(:, :)
        real(dp), allocatable :: head_theta_bar(:)
        real(dp), allocatable :: head_augmented_bar(:, :)
        integer :: j, feature_count, first, last

        theta_bar = 0.0_dp
        if (.not. valid_query(self, x, probabilities_bar, status, &
            "classifier chain parameter VJP")) return
        if (size(theta_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain parameter VJP: cotangent or parameter shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%n_outputs), &
            probabilities(size(x, 1), self%n_outputs))
        call forward_values(self, x, augmented, scores, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        allocate(probability_bar_work(size(x, 1), self%n_outputs), &
            binary_bar(size(x, 1), 2))
        probability_bar_work = probabilities_bar
        do j = self%n_outputs, 1, -1
            feature_count = self%n_features + j - 1
            first = packed_first(self, j)
            last = packed_last(self, j)
            allocate(head_theta_bar(self%parameter_sizes(j)))
            allocate(head_augmented_bar(size(x, 1), feature_count))
            binary_bar(:, 1) = 0.0_dp
            binary_bar(:, 2) = probability_bar_work(:, j)
            call self%models(j)%predict_proba_vjp(augmented(:, :feature_count), binary_bar, &
                head_theta_bar, head_augmented_bar, status)
            if (status%code /= FORTNUM_OK) return
            theta_bar(first:last) = head_theta_bar
            deallocate(head_theta_bar)
            if (j > 1) probability_bar_work(:, :j-1) = probability_bar_work(:, :j-1) + &
                head_augmented_bar(:, self%n_features+1:self%n_features+j-1)
            deallocate(head_augmented_bar)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classifier_chain_predict_proba_parameter_vjp

    subroutine classifier_chain_predict(self, x, labels, status)
        class(classifier_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, j

        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(labels) /= [size(x, 1), self%n_outputs])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain predict: model, input, or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_outputs))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_outputs
            do i = 1, size(x, 1)
                if (probabilities(i, j) >= self%decision_threshold(j)) then
                    labels(i, j) = self%class_label(2, j)
                else
                    labels(i, j) = self%class_label(1, j)
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classifier_chain_predict

    subroutine classifier_chain_predict_device(self, device, x, labels, status)
        class(classifier_chain_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain device predict: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "classifier chain device predict: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain device predict: device kind is invalid")
        end select
    end subroutine classifier_chain_predict_device

    function classifier_chain_classes(self) result(labels)
        class(classifier_chain_t), intent(in) :: self
        integer, allocatable :: labels(:, :)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(2, 0))
        end if
    end function classifier_chain_classes

    integer function classifier_chain_output_count(self) result(count)
        class(classifier_chain_t), intent(in) :: self
        count = self%n_outputs
    end function classifier_chain_output_count

    integer function classifier_chain_feature_count(self) result(count)
        class(classifier_chain_t), intent(in) :: self
        count = self%n_features
    end function classifier_chain_feature_count

    integer function classifier_chain_parameter_count(self) result(count)
        class(classifier_chain_t), intent(in) :: self
        count = 0
        if (allocated(self%parameter_sizes)) count = sum(self%parameter_sizes)
    end function classifier_chain_parameter_count

    function classifier_chain_parameters(self) result(values)
        class(classifier_chain_t), intent(in) :: self
        real(dp), allocatable :: values(:), head_values(:)
        integer :: j, first, last

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        do j = 1, self%n_outputs
            head_values = self%models(j)%parameters()
            first = packed_first(self, j)
            last = packed_last(self, j)
            values(first:last) = head_values
        end do
    end function classifier_chain_parameters

    subroutine classifier_chain_set_parameters(self, values, status)
        class(classifier_chain_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, first, last

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain set_parameters: model, shape, or values are invalid")
            return
        end if
        do j = 1, self%n_outputs
            first = packed_first(self, j)
            last = packed_last(self, j)
            call self%models(j)%set_parameters(values(first:last), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine classifier_chain_set_parameters

    function classifier_chain_thresholds(self) result(values)
        class(classifier_chain_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%decision_threshold)) then
            values = self%decision_threshold
        else
            allocate(values(0))
        end if
    end function classifier_chain_thresholds

    subroutine classifier_chain_set_thresholds(self, values, status)
        class(classifier_chain_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%is_fitted .or. size(values) /= self%n_outputs .or. &
            any(.not. ieee_is_finite(values)) .or. any(values <= 0.0_dp) .or. &
            any(values >= 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "classifier chain set_thresholds: values must be finite and in (0,1)")
            return
        end if
        self%decision_threshold = values
        call status_set(status, FORTNUM_OK, "")
    end subroutine classifier_chain_set_thresholds

    logical function classifier_chain_fitted(self) result(value)
        class(classifier_chain_t), intent(in) :: self
        value = self%is_fitted
    end function classifier_chain_fitted

    logical function classifier_chain_device_supported(self, device_kind) result(value)
        class(classifier_chain_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = self%is_fitted .and. device_kind == FORTML_DEVICE_CPU
    end function classifier_chain_device_supported

    logical function valid_query(self, x, values, status, operation) result(valid)
        class(classifier_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), values(:, :)
        type(fortnum_status_t), intent(out) :: status
        character(*), intent(in) :: operation

        valid = .false.
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)//": model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(values) /= [size(x, 1), self%n_outputs])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)// &
                ": model, input, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, trim(operation)//": inputs must be finite")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_query

    subroutine forward_values(self, x, augmented, scores, probabilities, status)
        class(classifier_chain_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: augmented(:, :)
        real(dp), intent(out) :: scores(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :)
        integer :: j, feature_count

        allocate(augmented(size(x, 1), self%n_features+self%n_outputs-1), &
            binary_probabilities(size(x, 1), 2))
        augmented = 0.0_dp
        augmented(:, :self%n_features) = x
        do j = 1, self%n_outputs
            feature_count = self%n_features + j - 1
            call self%models(j)%decision_function(augmented(:, :feature_count), &
                scores(:, j), status)
            if (status%code /= FORTNUM_OK) return
            call self%models(j)%predict_proba(augmented(:, :feature_count), &
                binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
            if (j < self%n_outputs) augmented(:, self%n_features+j) = probabilities(:, j)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine forward_values

    integer function packed_first(self, head) result(first)
        class(classifier_chain_t), intent(in) :: self
        integer, intent(in) :: head
        first = 1
        if (head > 1) first = 1 + sum(self%parameter_sizes(:head-1))
    end function packed_first

    integer function packed_last(self, head) result(last)
        class(classifier_chain_t), intent(in) :: self
        integer, intent(in) :: head
        last = sum(self%parameter_sizes(:head))
    end function packed_last

    subroutine sorted_binary_labels(values, pair, status)
        integer, intent(in) :: values(:)
        integer, intent(out) :: pair(2)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n
        logical :: seen

        pair = 0
        n = 0
        do i = 1, size(values)
            seen = .false.
            do j = 1, n
                if (values(i) == pair(j)) seen = .true.
            end do
            if (.not. seen) then
                if (n < 2) then
                    n = n + 1
                    pair(n) = values(i)
                else
                    call status_set(status, FORTNUM_DOMAIN_ERROR, "not binary")
                    return
                end if
            end if
        end do
        if (n /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "not binary")
            return
        end if
        do i = 2, 2
            j = i
            if (pair(j) < pair(j-1)) then
                n = pair(j)
                pair(j) = pair(j-1)
                pair(j-1) = n
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sorted_binary_labels

end module fortml_classifier_chain
