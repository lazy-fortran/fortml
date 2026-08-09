module fortml_rbf_svm_multiclass
    !! Deterministic one-vs-rest multiclass RBF kernel SVM.
    !!
    !! Each class owns an independent finite-basis ``rbf_svm_classifier_t``.
    !! Positive sigmoid margins are normalized row-wise, which gives the
    !! arbitrary-integer-label simplex contract shared by the other FortML
    !! multiclass estimators.  Fitting is transactional: child models are
    !! assembled in a candidate and published only after every child has
    !! succeeded.  Fit and split-boundary derivatives are deliberately not
    !! exposed; fixed-state query and packed-parameter products are analytic.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_rbf_svm_classifier, only: rbf_svm_classifier_t
    implicit none
    private

    type, public :: rbf_svm_multiclass_t
        private
        type(rbf_svm_classifier_t), allocatable :: models(:)
        integer, allocatable :: class_label(:)
        integer :: n_classes = 0
        integer :: n_features = 0
        integer :: n_parameters_per_model = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => rbf_svm_multiclass_fit
        procedure, public :: decision_function => rbf_svm_multiclass_decision
        procedure, public :: decision_function_device => &
            rbf_svm_multiclass_decision_device
        procedure, public :: predict_proba => rbf_svm_multiclass_predict_proba
        procedure, public :: predict_proba_device => &
            rbf_svm_multiclass_predict_proba_device
        procedure, public :: predict => rbf_svm_multiclass_predict
        procedure, public :: predict_device => rbf_svm_multiclass_predict_device
        procedure, public :: predict_proba_jvp => &
            rbf_svm_multiclass_predict_proba_jvp
        procedure, public :: predict_proba_vjp => &
            rbf_svm_multiclass_predict_proba_vjp
        procedure, public :: predict_proba_parameter_jvp => &
            rbf_svm_multiclass_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            rbf_svm_multiclass_predict_proba_parameter_vjp
        procedure, public :: predict_proba_jvp_device => &
            rbf_svm_multiclass_predict_proba_jvp_device
        procedure, public :: predict_proba_vjp_device => &
            rbf_svm_multiclass_predict_proba_vjp_device
        procedure, public :: predict_proba_parameter_jvp_device => &
            rbf_svm_multiclass_predict_proba_parameter_jvp_device
        procedure, public :: predict_proba_parameter_vjp_device => &
            rbf_svm_multiclass_predict_proba_parameter_vjp_device
        procedure, public :: classes => rbf_svm_multiclass_classes
        procedure, public :: class_count => rbf_svm_multiclass_class_count
        procedure, public :: feature_count => rbf_svm_multiclass_feature_count
        procedure, public :: parameter_count => rbf_svm_multiclass_parameter_count
        procedure, public :: parameters => rbf_svm_multiclass_parameters
        procedure, public :: set_parameters => rbf_svm_multiclass_set_parameters
        procedure, public :: fitted => rbf_svm_multiclass_fitted
        procedure, public :: device_supported => rbf_svm_multiclass_device_supported
    end type rbf_svm_multiclass_t

    public :: rbf_svm_multiclass_fit
    public :: rbf_svm_multiclass_decision
    public :: rbf_svm_multiclass_predict_proba
    public :: rbf_svm_multiclass_predict

contains

    subroutine rbf_svm_multiclass_fit(self, x, labels, status, c, gamma, &
            max_iterations, tolerance, sample_weight)
        class(rbf_svm_multiclass_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: c, gamma, tolerance, sample_weight(:)
        integer, intent(in), optional :: max_iterations
        type(rbf_svm_multiclass_t) :: candidate
        integer, allocatable :: classes(:), binary_labels(:)
        real(dp) :: requested_c, requested_gamma, requested_tolerance
        integer :: requested_iterations, n_samples, n_features, i, n_classes

        requested_c = 1.0_dp
        if (present(c)) requested_c = c
        requested_gamma = 1.0_dp
        if (present(gamma)) requested_gamma = gamma
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        requested_iterations = 500
        if (present(max_iterations)) requested_iterations = max_iterations

        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 2 .or. n_features < 1 .or. size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            .not. ieee_is_finite(requested_c) .or. requested_c <= 0.0_dp .or. &
            .not. ieee_is_finite(requested_gamma) .or. requested_gamma <= 0.0_dp .or. &
            .not. ieee_is_finite(requested_tolerance) .or. requested_tolerance <= 0.0_dp .or. &
            requested_iterations < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass fit: finite positive options are required")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM multiclass fit: sample weights are invalid")
                return
            end if
            if (sum(sample_weight) <= 0.0_dp .or. &
                .not. ieee_is_finite(sum(sample_weight))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM multiclass fit: sample weights have no positive mass")
                return
            end if
        end if

        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass fit: at least two classes are required")
            return
        end if

        allocate(candidate%models(n_classes), candidate%class_label(n_classes), &
            binary_labels(n_samples))
        candidate%class_label = classes
        candidate%n_classes = n_classes
        candidate%n_features = n_features
        do i = 1, n_classes
            binary_labels = 0
            where (labels == classes(i)) binary_labels = 1
            if (present(sample_weight)) then
                call candidate%models(i)%fit(x, binary_labels, status, &
                    c=requested_c, gamma=requested_gamma, &
                    max_iterations=requested_iterations, tolerance=requested_tolerance, &
                    sample_weight=sample_weight)
            else
                call candidate%models(i)%fit(x, binary_labels, status, &
                    c=requested_c, gamma=requested_gamma, &
                    max_iterations=requested_iterations, tolerance=requested_tolerance)
            end if
            if (status%code /= FORTNUM_OK) then
                call status_set(status, status%code, &
                    "RBF SVM multiclass fit: child estimator failed")
                return
            end if
        end do
        candidate%n_parameters_per_model = candidate%models(1)%parameter_count()
        candidate%is_fitted = .true.
        call publish_candidate(self, candidate)
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_multiclass_fit

    subroutine rbf_svm_multiclass_decision(self, x, scores, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local(:)
        integer :: i

        scores = 0.0_dp
        if (.not. prediction_shapes(self, x, scores)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass decision: model, input, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass decision: inputs must be finite")
            return
        end if
        allocate(local(size(x, 1)))
        do i = 1, self%n_classes
            call self%models(i)%decision_function(x, local, status)
            if (status%code /= FORTNUM_OK) return
            scores(:, i) = local
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_multiclass_decision

    subroutine rbf_svm_multiclass_predict_proba(self, x, probabilities, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:, :)
        integer :: i, j
        real(dp) :: total

        probabilities = 0.0_dp
        if (.not. prediction_shapes(self, x, probabilities)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass probability: model, input, or output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1), self%n_classes))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            total = 0.0_dp
            do j = 1, self%n_classes
                probabilities(i, j) = stable_sigmoid(scores(i, j))
                total = total + probabilities(i, j)
            end do
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM multiclass probability: normalization is invalid")
                return
            end if
            probabilities(i, :) = probabilities(i, :)/total
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_multiclass_predict_proba

    subroutine rbf_svm_multiclass_predict(self, x, labels, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, j, best

        labels = 0
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            best = 1
            do j = 2, self%n_classes
                if (probabilities(i, j) > probabilities(i, best)) best = j
            end do
            labels(i) = self%class_label(best)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_multiclass_predict

    subroutine rbf_svm_multiclass_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: raw(:, :), raw_dot(:, :), local(:, :), local_dot(:, :)
        real(dp), allocatable :: theta_dot(:), x_zero(:, :)
        real(dp) :: total, total_dot
        integer :: i, j

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. prediction_shapes(self, x, probabilities) .or. &
            any(shape(x_dot) /= shape(x)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass probability JVP: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass probability JVP: tangent is not finite")
            return
        end if
        allocate(raw(size(x, 1), self%n_classes), raw_dot(size(x, 1), self%n_classes), &
            local(size(x, 1), 2), local_dot(size(x, 1), 2), &
            theta_dot(self%n_parameters_per_model))
        do j = 1, self%n_classes
            theta_dot = 0.0_dp
            call self%models(j)%predict_proba_jvp(x, theta_dot, x_dot, local, local_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = local(:, 2)
            raw_dot(:, j) = local_dot(:, 2)
        end do
        do i = 1, size(x, 1)
            total = sum(raw(i, :))
            total_dot = sum(raw_dot(i, :))
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM multiclass probability JVP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/total
            probabilities_dot(i, :) = (raw_dot(i, :)*total - raw(i, :)*total_dot)/ &
                (total*total)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_multiclass_predict_proba_jvp

    subroutine rbf_svm_multiclass_predict_proba_parameter_jvp(self, x, direction, &
            probabilities, probabilities_dot, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: raw(:, :), raw_dot(:, :), local(:, :), local_dot(:, :)
        real(dp), allocatable :: x_zero(:, :)
        real(dp) :: total, total_dot
        integer :: i, j, first, last

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. prediction_shapes(self, x, probabilities) .or. &
            any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass parameter JVP: shape or values are invalid")
            return
        end if
        allocate(raw(size(x, 1), self%n_classes), raw_dot(size(x, 1), self%n_classes), &
            local(size(x, 1), 2), local_dot(size(x, 1), 2), &
            x_zero(size(x, 1), self%n_features))
        x_zero = 0.0_dp
        do j = 1, self%n_classes
            first = (j - 1)*self%n_parameters_per_model + 1
            last = j*self%n_parameters_per_model
            call self%models(j)%predict_proba_jvp(x, direction(first:last), x_zero, &
                local, local_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = local(:, 2)
            raw_dot(:, j) = local_dot(:, 2)
        end do
        do i = 1, size(x, 1)
            total = sum(raw(i, :))
            total_dot = sum(raw_dot(i, :))
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM multiclass parameter JVP: invalid normalization")
                return
            end if
            probabilities(i, :) = raw(i, :)/total
            probabilities_dot(i, :) = (raw_dot(i, :)*total - raw(i, :)*total_dot)/ &
                (total*total)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_multiclass_predict_proba_parameter_jvp

    subroutine rbf_svm_multiclass_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: raw(:, :), raw_bar(:, :), local(:, :)
        real(dp), allocatable :: local_probability_bar(:, :), local_theta_bar(:), &
            local_x_bar(:, :)
        real(dp) :: total, projection
        integer :: i, j

        x_bar = 0.0_dp
        if (.not. prediction_shapes(self, x, probabilities_bar) .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass probability VJP: shape or values are invalid")
            return
        end if
        allocate(raw(size(x, 1), self%n_classes), raw_bar(size(x, 1), self%n_classes), &
            local(size(x, 1), 2), local_probability_bar(size(x, 1), 2), &
            local_theta_bar(self%n_parameters_per_model), &
            local_x_bar(size(x, 1), self%n_features))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, local, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = local(:, 2)
        end do
        do i = 1, size(x, 1)
            total = sum(raw(i, :))
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM multiclass probability VJP: invalid normalization")
                return
            end if
            projection = sum(probabilities_bar(i, :)*raw(i, :))/total
            raw_bar(i, :) = (probabilities_bar(i, :) - projection)/total
        end do
        do j = 1, self%n_classes
            local_probability_bar = 0.0_dp
            local_probability_bar(:, 2) = raw_bar(:, j)
            call self%models(j)%predict_proba_vjp(x, local_probability_bar, &
                local_theta_bar, local_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + local_x_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_multiclass_predict_proba_vjp

    subroutine rbf_svm_multiclass_predict_proba_parameter_vjp(self, x, &
            probabilities_bar, parameter_bar, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: raw(:, :), raw_bar(:, :), local(:, :)
        real(dp), allocatable :: local_probability_bar(:, :), local_theta_bar(:), &
            local_x_bar(:, :)
        real(dp) :: total, projection
        integer :: i, j, first, last

        parameter_bar = 0.0_dp
        if (.not. prediction_shapes(self, x, probabilities_bar) .or. &
            size(parameter_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass parameter VJP: shape or values are invalid")
            return
        end if
        allocate(raw(size(x, 1), self%n_classes), raw_bar(size(x, 1), self%n_classes), &
            local(size(x, 1), 2), local_probability_bar(size(x, 1), 2), &
            local_theta_bar(self%n_parameters_per_model), &
            local_x_bar(size(x, 1), self%n_features))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, local, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = local(:, 2)
        end do
        do i = 1, size(x, 1)
            total = sum(raw(i, :))
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM multiclass parameter VJP: invalid normalization")
                return
            end if
            projection = sum(probabilities_bar(i, :)*raw(i, :))/total
            raw_bar(i, :) = (probabilities_bar(i, :) - projection)/total
        end do
        do j = 1, self%n_classes
            local_probability_bar = 0.0_dp
            local_probability_bar(:, 2) = raw_bar(:, j)
            call self%models(j)%predict_proba_vjp(x, local_probability_bar, &
                local_theta_bar, local_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            first = (j - 1)*self%n_parameters_per_model + 1
            last = j*self%n_parameters_per_model
            parameter_bar(first:last) = local_theta_bar
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_multiclass_predict_proba_parameter_vjp

    subroutine rbf_svm_multiclass_decision_device(self, device, x, scores, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        scores = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM multiclass device decision: resident CUDA kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass device decision: device kind is invalid")
        end select
    end subroutine rbf_svm_multiclass_decision_device

    subroutine rbf_svm_multiclass_predict_proba_device(self, device, x, probabilities, &
            status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM multiclass device probability: resident CUDA kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass device probability: device kind is invalid")
        end select
    end subroutine rbf_svm_multiclass_predict_proba_device

    subroutine rbf_svm_multiclass_predict_device(self, device, x, labels, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        labels = 0
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM multiclass device prediction: resident CUDA kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass device prediction: device kind is invalid")
        end select
    end subroutine rbf_svm_multiclass_predict_device

    subroutine rbf_svm_multiclass_predict_proba_jvp_device(self, device, x, x_dot, &
            probabilities, probabilities_dot, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass probability JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_jvp(x, x_dot, probabilities, probabilities_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM multiclass probability JVP device: resident CUDA kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass probability JVP device: device kind is invalid")
        end select
    end subroutine rbf_svm_multiclass_predict_proba_jvp_device

    subroutine rbf_svm_multiclass_predict_proba_vjp_device(self, device, x, &
            probabilities_bar, x_bar, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        x_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass probability VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_vjp(x, probabilities_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM multiclass probability VJP device: resident CUDA kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass probability VJP device: device kind is invalid")
        end select
    end subroutine rbf_svm_multiclass_predict_proba_vjp_device

    subroutine rbf_svm_multiclass_predict_proba_parameter_jvp_device(self, device, x, &
            direction, probabilities, probabilities_dot, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass parameter JVP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_jvp(x, direction, probabilities, &
                probabilities_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM multiclass parameter JVP device: resident CUDA kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass parameter JVP device: device kind is invalid")
        end select
    end subroutine rbf_svm_multiclass_predict_proba_parameter_jvp_device

    subroutine rbf_svm_multiclass_predict_proba_parameter_vjp_device(self, device, x, &
            probabilities_bar, parameter_bar, status)
        class(rbf_svm_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass parameter VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_vjp(x, probabilities_bar, parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "RBF SVM multiclass parameter VJP device: resident CUDA kernel is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass parameter VJP device: device kind is invalid")
        end select
    end subroutine rbf_svm_multiclass_predict_proba_parameter_vjp_device

    function rbf_svm_multiclass_classes(self) result(labels)
        class(rbf_svm_multiclass_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function rbf_svm_multiclass_classes

    integer function rbf_svm_multiclass_class_count(self) result(count)
        class(rbf_svm_multiclass_t), intent(in) :: self
        count = self%n_classes
    end function rbf_svm_multiclass_class_count

    integer function rbf_svm_multiclass_feature_count(self) result(count)
        class(rbf_svm_multiclass_t), intent(in) :: self
        count = self%n_features
    end function rbf_svm_multiclass_feature_count

    integer function rbf_svm_multiclass_parameter_count(self) result(count)
        class(rbf_svm_multiclass_t), intent(in) :: self
        count = self%n_classes*self%n_parameters_per_model
        if (.not. self%is_fitted) count = 0
    end function rbf_svm_multiclass_parameter_count

    function rbf_svm_multiclass_parameters(self) result(values)
        class(rbf_svm_multiclass_t), intent(in) :: self
        real(dp), allocatable :: values(:), local(:)
        integer :: i, first, last

        if (.not. self%is_fitted) then
            allocate(values(0))
            return
        end if
        allocate(values(self%parameter_count()))
        values = 0.0_dp
        do i = 1, self%n_classes
            local = self%models(i)%parameters()
            first = (i - 1)*self%n_parameters_per_model + 1
            last = i*self%n_parameters_per_model
            values(first:last) = local
        end do
    end function rbf_svm_multiclass_parameters

    subroutine rbf_svm_multiclass_set_parameters(self, values, status)
        class(rbf_svm_multiclass_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        type(rbf_svm_classifier_t), allocatable :: backup(:)
        integer :: i, first, last
        real(dp) :: gamma

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "RBF SVM multiclass set_parameters: model, shape, or values are invalid")
            return
        end if
        do i = 1, self%n_classes
            first = (i - 1)*self%n_parameters_per_model + 1
            last = i*self%n_parameters_per_model
            gamma = exp(values(first + self%n_parameters_per_model - 1))
            if (.not. ieee_is_finite(gamma) .or. gamma <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "RBF SVM multiclass set_parameters: log-gamma is invalid")
                return
            end if
        end do
        backup = self%models
        do i = 1, self%n_classes
            first = (i - 1)*self%n_parameters_per_model + 1
            last = i*self%n_parameters_per_model
            call self%models(i)%set_parameters(values(first:last), status)
            if (status%code /= FORTNUM_OK) then
                self%models = backup
                return
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rbf_svm_multiclass_set_parameters

    subroutine publish_candidate(self, candidate)
        class(rbf_svm_multiclass_t), intent(inout) :: self
        type(rbf_svm_multiclass_t), intent(inout) :: candidate

        if (allocated(self%models)) deallocate(self%models)
        if (allocated(self%class_label)) deallocate(self%class_label)
        call move_alloc(candidate%models, self%models)
        call move_alloc(candidate%class_label, self%class_label)
        self%n_classes = candidate%n_classes
        self%n_features = candidate%n_features
        self%n_parameters_per_model = candidate%n_parameters_per_model
        self%is_fitted = candidate%is_fitted
    end subroutine publish_candidate

    logical function rbf_svm_multiclass_fitted(self) result(value)
        class(rbf_svm_multiclass_t), intent(in) :: self
        value = self%is_fitted .and. allocated(self%models) .and. &
            allocated(self%class_label)
    end function rbf_svm_multiclass_fitted

    logical function rbf_svm_multiclass_device_supported(self, device_kind) result(value)
        class(rbf_svm_multiclass_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = self%fitted() .and. device_kind == FORTML_DEVICE_CPU
    end function rbf_svm_multiclass_device_supported

    logical function prediction_shapes(self, x, output) result(valid)
        class(rbf_svm_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(in) :: output(:, :)
        valid = self%fitted()
        if (.not. valid) return
        valid = size(x, 1) >= 1 .and. size(x, 2) == self%n_features
        if (.not. valid) return
        valid = all(shape(output) == [size(x, 1), self%n_classes])
    end function prediction_shapes

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, n, value
        logical :: found

        allocate(work(size(labels)))
        n = 0
        do i = 1, size(labels)
            found = .false.
            do j = 1, n
                if (work(j) == labels(i)) then
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) then
                n = n + 1
                work(n) = labels(i)
            end if
        end do
        do i = 2, n
            value = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= value) exit
                work(j + 1) = work(j)
                j = j - 1
            end do
            work(j + 1) = value
        end do
        allocate(classes(n))
        classes = work(:n)
    end subroutine sorted_unique_labels

    pure real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

end module fortml_rbf_svm_multiclass
