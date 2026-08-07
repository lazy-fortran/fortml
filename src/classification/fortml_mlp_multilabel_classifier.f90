module fortml_mlp_multilabel_classifier
    !! Differentiable one-vs-rest multilabel MLP classifier.
    !!
    !! The classifier composes one tested binary MLP head per indicator column.
    !! Targets are finite zero/one values, probabilities are independent
    !! positive-class sigmoid probabilities, and all packed derivative products
    !! concatenate heads in label order.  CUDA requests are typed refusals.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_mlp, only: MLP_TANH
    use fortml_mlp_binary_classifier, only: mlp_binary_classifier_t, &
        mlp_binary_classifier_options_t, mlp_binary_classifier_state_t
    implicit none
    private

    type, public :: mlp_multilabel_classifier_options_t
        integer :: max_epochs = 1000
        integer :: batch_size = 0
        integer :: patience = 0
        integer :: shuffle_seed = 17
        integer :: initialization_seed = 17
        integer :: hidden_activation = MLP_TANH
        logical :: shuffle = .false.
        logical :: restore_best = .true.
        real(dp) :: learning_rate = 1.0e-3_dp
        real(dp) :: beta1 = 0.9_dp
        real(dp) :: beta2 = 0.999_dp
        real(dp) :: epsilon = 1.0e-8_dp
        real(dp) :: l2 = 0.0_dp
        real(dp) :: tolerance = 1.0e-6_dp
        real(dp) :: min_delta = 0.0_dp
    end type mlp_multilabel_classifier_options_t

    type, public :: mlp_multilabel_classifier_state_t
        integer :: epochs = 0
        integer :: updates = 0
        integer :: best_epoch = 0
        logical :: converged = .false.
        logical :: early_stopped = .false.
        real(dp) :: initial_loss = huge(1.0_dp)
        real(dp) :: final_loss = huge(1.0_dp)
        real(dp) :: best_loss = huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type mlp_multilabel_classifier_state_t

    type, public :: mlp_multilabel_classifier_t
        private
        type(mlp_binary_classifier_t), allocatable :: heads(:)
    contains
        procedure, public :: fit => mlp_multilabel_fit
        procedure, public :: decision_function => mlp_multilabel_decision
        procedure, public :: decision_function_device => &
            mlp_multilabel_decision_device
        procedure, public :: decision_function_jvp => mlp_multilabel_decision_jvp
        procedure, public :: decision_function_vjp => mlp_multilabel_decision_vjp
        procedure, public :: predict_proba => mlp_multilabel_predict_proba
        procedure, public :: predict_proba_device => &
            mlp_multilabel_predict_proba_device
        procedure, public :: predict_proba_jvp => mlp_multilabel_predict_proba_jvp
        procedure, public :: predict_proba_vjp => mlp_multilabel_predict_proba_vjp
        procedure, public :: predict => mlp_multilabel_predict
        procedure, public :: predict_device => mlp_multilabel_predict_device
        procedure, public :: label_count => mlp_multilabel_label_count
        procedure, public :: feature_count => mlp_multilabel_feature_count
        procedure, public :: parameter_count => mlp_multilabel_parameter_count
        procedure, public :: parameters => mlp_multilabel_parameters
        procedure, public :: set_parameters => mlp_multilabel_set_parameters
        procedure, public :: loss_gradient => mlp_multilabel_loss_gradient
        procedure, public :: loss_hvp => mlp_multilabel_loss_hvp
        procedure, public :: fitted => mlp_multilabel_fitted
        procedure, public :: device_supported => mlp_multilabel_device_supported
    end type mlp_multilabel_classifier_t

    public :: mlp_multilabel_fit
    public :: mlp_multilabel_decision
    public :: mlp_multilabel_decision_device
    public :: mlp_multilabel_decision_jvp
    public :: mlp_multilabel_decision_vjp
    public :: mlp_multilabel_predict_proba
    public :: mlp_multilabel_predict_proba_device
    public :: mlp_multilabel_predict_proba_jvp
    public :: mlp_multilabel_predict_proba_vjp
    public :: mlp_multilabel_predict
    public :: mlp_multilabel_predict_device

contains

    subroutine mlp_multilabel_fit(self, x, targets, status, hidden_layer_sizes, &
            options, state, sample_weight, class_weight)
        class(mlp_multilabel_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), targets(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: hidden_layer_sizes(:)
        type(mlp_multilabel_classifier_options_t), intent(in), optional :: options
        type(mlp_multilabel_classifier_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:), class_weight(:, :)
        type(mlp_multilabel_classifier_options_t) :: config
        type(mlp_binary_classifier_options_t) :: binary_options
        type(mlp_binary_classifier_state_t) :: head_state
        integer, allocatable :: labels(:)
        integer :: j, n_labels
        real(dp) :: loss_sum, norm_sum

        if (present(options)) config = options
        if (.not. valid_options(config)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel fit: invalid optimizer options")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(targets, 1) /= size(x, 1) &
                .or. size(targets, 2) < 1 .or. any(.not. ieee_is_finite(x)) .or. &
                any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel fit: input dimensions or values are invalid")
            return
        end if
        if (any(targets < 0.0_dp) .or. any(targets > 1.0_dp) .or. &
                any(targets /= real(nint(targets), dp))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel fit: targets must be zero or one")
            return
        end if
        n_labels = size(targets, 2)
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(x, 1) .or. &
                    any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel fit: sample weights must be finite and nonnegative")
                return
            end if
        end if
        if (present(class_weight)) then
            if (any(shape(class_weight) /= [2, n_labels]) .or. &
                    any(.not. ieee_is_finite(class_weight)) .or. &
                    any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MLP multilabel fit: class_weight must have shape (2,n_labels)")
                return
            end if
        end if

        binary_options = to_binary_options(config)
        allocate(self%heads(n_labels), labels(size(x, 1)))
        loss_sum = 0.0_dp
        norm_sum = 0.0_dp
        if (present(state)) state = mlp_multilabel_classifier_state_t()
        do j = 1, n_labels
            labels = nint(targets(:, j))
            if (present(class_weight)) then
                call self%heads(j)%fit(x, labels, status, hidden_layer_sizes, &
                    binary_options, head_state, sample_weight, class_weight(:, j))
            else if (present(sample_weight)) then
                call self%heads(j)%fit(x, labels, status, hidden_layer_sizes, &
                    binary_options, head_state, sample_weight=sample_weight)
            else if (present(hidden_layer_sizes)) then
                call self%heads(j)%fit(x, labels, status, hidden_layer_sizes, &
                    binary_options, head_state)
            else
                call self%heads(j)%fit(x, labels, status, options=binary_options, &
                    state=head_state)
            end if
            if (.not. status_ok(status)) return
            loss_sum = loss_sum + head_state%final_loss
            norm_sum = norm_sum + head_state%gradient_norm**2
            if (present(state)) then
                state%epochs = max(state%epochs, head_state%epochs)
                state%updates = state%updates + head_state%updates
                state%best_epoch = max(state%best_epoch, head_state%best_epoch)
                state%converged = state%converged .or. head_state%converged
                state%early_stopped = state%early_stopped .or. head_state%early_stopped
            end if
        end do
        if (present(state)) then
            state%initial_loss = loss_sum / real(n_labels, dp)
            state%final_loss = state%initial_loss
            state%best_loss = state%final_loss
            state%gradient_norm = sqrt(norm_sum)
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_fit

    subroutine mlp_multilabel_decision(self, x, scores, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j

        scores = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision_function: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
                any(shape(scores) /= [size(x, 1), self%label_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision_function: shape is invalid")
            return
        end if
        do j = 1, self%label_count()
            call self%heads(j)%decision_function(x, scores(:, j), status)
            if (.not. status_ok(status)) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_decision

    subroutine mlp_multilabel_decision_device(self, device, x, scores, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP multilabel device decision: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device decision: device kind is invalid")
        end select
    end subroutine mlp_multilabel_decision_device

    subroutine mlp_multilabel_decision_jvp(self, x, theta_dot, x_dot, scores, &
            scores_dot, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: scores(:, :), scores_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, first, last, count

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision JVP: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%feature_count() .or. &
                any(shape(x_dot) /= shape(x)) .or. any(shape(scores) /= &
                [size(x, 1), self%label_count()]) .or. any(shape(scores_dot) /= &
                shape(scores)) .or. size(theta_dot) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot)) .or. &
                any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision JVP: shape or value is invalid")
            return
        end if
        first = 1
        do j = 1, self%label_count()
            count = self%heads(j)%parameter_count()
            last = first + count - 1
            call self%heads(j)%decision_function_jvp(x, theta_dot(first:last), x_dot, &
                scores(:, j), scores_dot(:, j), status)
            if (.not. status_ok(status)) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_decision_jvp

    subroutine mlp_multilabel_decision_vjp(self, x, scores_bar, theta_bar, x_bar, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: theta_part(:), x_part(:, :)
        integer :: j, first, last, count

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision VJP: model is not fitted")
            return
        end if
        if (any(shape(scores_bar) /= [size(x, 1), self%label_count()]) .or. &
                size(theta_bar) /= self%parameter_count() .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel decision VJP: shape is invalid")
            return
        end if
        allocate(x_part(size(x, 1), size(x, 2)))
        first = 1
        do j = 1, self%label_count()
            count = self%heads(j)%parameter_count()
            last = first + count - 1
            allocate(theta_part(count))
            call self%heads(j)%decision_function_vjp(x, scores_bar(:, j), theta_part, &
                x_part, status)
            if (.not. status_ok(status)) return
            theta_bar(first:last) = theta_part
            x_bar = x_bar + x_part
            deallocate(theta_part)
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_decision_vjp

    subroutine mlp_multilabel_predict_proba(self, x, probabilities, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: head_prob(:, :)
        integer :: j

        probabilities = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel predict_proba: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%label_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel predict_proba: shape is invalid")
            return
        end if
        allocate(head_prob(size(x, 1), 2))
        do j = 1, self%label_count()
            call self%heads(j)%predict_proba(x, head_prob, status)
            if (.not. status_ok(status)) return
            probabilities(:, j) = head_prob(:, 2)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_predict_proba

    subroutine mlp_multilabel_predict_proba_device(self, device, x, probabilities, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device probability: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP multilabel device probability: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device probability: device kind is invalid")
        end select
    end subroutine mlp_multilabel_predict_proba_device

    subroutine mlp_multilabel_predict_proba_jvp(self, x, theta_dot, x_dot, &
            probabilities, probabilities_dot, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: head_prob(:, :), head_dot(:, :)
        integer :: j, first, last, count

        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel probability JVP: model is not fitted")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%label_count()]) .or. &
                any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel probability JVP: output shape is invalid")
            return
        end if
        allocate(head_prob(size(x, 1), 2), head_dot(size(x, 1), 2))
        first = 1
        do j = 1, self%label_count()
            count = self%heads(j)%parameter_count()
            last = first + count - 1
            call self%heads(j)%predict_proba_jvp(x, theta_dot(first:last), x_dot, &
                head_prob, head_dot, status)
            if (.not. status_ok(status)) return
            probabilities(:, j) = head_prob(:, 2)
            probabilities_dot(:, j) = head_dot(:, 2)
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_predict_proba_jvp

    subroutine mlp_multilabel_predict_proba_vjp(self, x, probabilities_bar, theta_bar, &
            x_bar, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: head_bar(:, :), theta_part(:), x_part(:, :)
        integer :: j, first, last, count

        theta_bar = 0.0_dp
        x_bar = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel probability VJP: model is not fitted")
            return
        end if
        if (any(shape(probabilities_bar) /= [size(x, 1), self%label_count()]) .or. &
                size(theta_bar) /= self%parameter_count() .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel probability VJP: shape is invalid")
            return
        end if
        allocate(head_bar(size(x, 1), 2), x_part(size(x, 1), size(x, 2)))
        first = 1
        do j = 1, self%label_count()
            count = self%heads(j)%parameter_count()
            last = first + count - 1
            head_bar(:, 1) = -probabilities_bar(:, j)
            head_bar(:, 2) = probabilities_bar(:, j)
            allocate(theta_part(count))
            call self%heads(j)%predict_proba_vjp(x, head_bar, theta_part, x_part, status)
            if (.not. status_ok(status)) return
            theta_bar(first:last) = theta_part
            x_bar = x_bar + x_part
            deallocate(theta_part)
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_predict_proba_vjp

    subroutine mlp_multilabel_predict(self, x, labels, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, allocatable :: head_labels(:)
        integer :: j

        labels = 0
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel predict: model is not fitted")
            return
        end if
        if (any(shape(labels) /= [size(x, 1), self%label_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel predict: shape is invalid")
            return
        end if
        allocate(head_labels(size(x, 1)))
        do j = 1, self%label_count()
            call self%heads(j)%predict(x, head_labels, status)
            if (.not. status_ok(status)) return
            labels(:, j) = head_labels
        end do
        where (labels < 0 .or. labels > 1) labels = 0
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_predict

    subroutine mlp_multilabel_predict_device(self, device, x, labels, status)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "MLP multilabel device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel device prediction: device kind is invalid")
        end select
    end subroutine mlp_multilabel_predict_device

    integer function mlp_multilabel_label_count(self) result(count)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        count = 0
        if (allocated(self%heads)) count = size(self%heads)
    end function mlp_multilabel_label_count

    integer function mlp_multilabel_feature_count(self) result(count)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        count = 0
        if (self%fitted()) count = self%heads(1)%feature_count()
    end function mlp_multilabel_feature_count

    integer function mlp_multilabel_parameter_count(self) result(count)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        integer :: j
        count = 0
        if (.not. self%fitted()) return
        do j = 1, size(self%heads)
            count = count + self%heads(j)%parameter_count()
        end do
    end function mlp_multilabel_parameter_count

    function mlp_multilabel_parameters(self) result(values)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:), part(:)
        integer :: j, first, last, count

        if (.not. self%fitted()) then
            allocate(values(0))
            return
        end if
        allocate(values(self%parameter_count()))
        first = 1
        do j = 1, self%label_count()
            part = self%heads(j)%parameters()
            count = size(part)
            last = first + count - 1
            values(first:last) = part
            first = last + 1
        end do
    end function mlp_multilabel_parameters

    subroutine mlp_multilabel_set_parameters(self, values, status)
        class(mlp_multilabel_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: j, first, last, count

        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel set_parameters: model is not fitted")
            return
        end if
        if (size(values) /= self%parameter_count() .or. any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel set_parameters: shape or values are invalid")
            return
        end if
        first = 1
        do j = 1, self%label_count()
            count = self%heads(j)%parameter_count()
            last = first + count - 1
            call self%heads(j)%set_parameters(values(first:last), status)
            if (.not. status_ok(status)) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_set_parameters

    subroutine mlp_multilabel_loss_gradient(self, x, targets, l2, value, gradient, status, &
            sample_weight)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), targets(:, :), l2
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: labels(:)
        real(dp), allocatable :: part(:)
        real(dp) :: value_part
        integer :: j, first, last, count

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss_gradient: model is not fitted")
            return
        end if
        if (size(targets, 1) /= size(x, 1) .or. size(targets, 2) /= self%label_count() .or. &
                size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss_gradient: shape is invalid")
            return
        end if
        if (.not. valid_targets(targets)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss_gradient: targets must be zero or one")
            return
        end if
        allocate(labels(size(x, 1)), part(0))
        first = 1
        do j = 1, self%label_count()
            labels = nint(targets(:, j))
            count = self%heads(j)%parameter_count()
            last = first + count - 1
            deallocate(part)
            allocate(part(count))
            if (present(sample_weight)) then
                call self%heads(j)%loss_gradient(x, labels, l2, value_part, part, status, &
                    sample_weight)
            else
                call self%heads(j)%loss_gradient(x, labels, l2, value_part, part, status)
            end if
            if (.not. status_ok(status)) return
            value = value + value_part
            gradient(first:last) = part
            first = last + 1
        end do
        value = value / real(self%label_count(), dp)
        gradient = gradient / real(self%label_count(), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_loss_gradient

    subroutine mlp_multilabel_loss_hvp(self, x, targets, l2, theta_dot, hvp, status, &
            sample_weight)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), targets(:, :), l2, theta_dot(:)
        real(dp), intent(out) :: hvp(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer, allocatable :: labels(:)
        real(dp), allocatable :: part(:)
        integer :: j, first, last, count

        hvp = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss_hvp: model is not fitted")
            return
        end if
        if (size(targets, 1) /= size(x, 1) .or. size(targets, 2) /= self%label_count() .or. &
                size(theta_dot) /= self%parameter_count() .or. size(hvp) /= size(theta_dot) .or. &
                .not. valid_targets(targets)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MLP multilabel loss_hvp: shape or targets are invalid")
            return
        end if
        allocate(labels(size(x, 1)), part(0))
        first = 1
        do j = 1, self%label_count()
            labels = nint(targets(:, j))
            count = self%heads(j)%parameter_count()
            last = first + count - 1
            deallocate(part)
            allocate(part(count))
            if (present(sample_weight)) then
                call self%heads(j)%loss_hvp(x, labels, l2, theta_dot(first:last), part, &
                    status, sample_weight)
            else
                call self%heads(j)%loss_hvp(x, labels, l2, theta_dot(first:last), part, status)
            end if
            if (.not. status_ok(status)) return
            hvp(first:last) = part
            first = last + 1
        end do
        hvp = hvp / real(self%label_count(), dp)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_multilabel_loss_hvp

    logical function mlp_multilabel_fitted(self) result(is_fitted)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        integer :: j
        if (.not. allocated(self%heads)) then
            is_fitted = .false.
            return
        end if
        is_fitted = size(self%heads) > 0
        if (.not. is_fitted) return
        do j = 1, size(self%heads)
            if (.not. self%heads(j)%fitted()) then
                is_fitted = .false.
                return
            end if
        end do
    end function mlp_multilabel_fitted

    logical function mlp_multilabel_device_supported(self, device_kind) result(supported)
        class(mlp_multilabel_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        supported = device_kind == FORTML_DEVICE_CPU .and. self%fitted()
    end function mlp_multilabel_device_supported

    logical function valid_targets(targets) result(valid)
        real(dp), intent(in) :: targets(:, :)
        valid = size(targets, 1) > 0 .and. size(targets, 2) > 0 .and. &
            all(ieee_is_finite(targets)) .and. all(targets >= 0.0_dp) .and. &
            all(targets <= 1.0_dp) .and. all(targets == real(nint(targets), dp))
    end function valid_targets

    logical function valid_options(options) result(valid)
        type(mlp_multilabel_classifier_options_t), intent(in) :: options
        valid = options%max_epochs >= 1 .and. options%batch_size >= 0 .and. &
            options%patience >= 0 .and. options%learning_rate > 0.0_dp .and. &
            options%beta1 >= 0.0_dp .and. options%beta1 < 1.0_dp .and. &
            options%beta2 >= 0.0_dp .and. options%beta2 < 1.0_dp .and. &
            options%epsilon > 0.0_dp .and. options%l2 >= 0.0_dp .and. &
            options%tolerance >= 0.0_dp .and. options%min_delta >= 0.0_dp .and. &
            options%initialization_seed >= 0
        if (options%shuffle) valid = valid .and. options%shuffle_seed > 0
        valid = valid .and. all([ieee_is_finite(options%learning_rate), &
            ieee_is_finite(options%beta1), ieee_is_finite(options%beta2), &
            ieee_is_finite(options%epsilon), ieee_is_finite(options%l2), &
            ieee_is_finite(options%tolerance), ieee_is_finite(options%min_delta)])
    end function valid_options

    function to_binary_options(options) result(binary)
        type(mlp_multilabel_classifier_options_t), intent(in) :: options
        type(mlp_binary_classifier_options_t) :: binary
        binary%max_epochs = options%max_epochs
        binary%batch_size = options%batch_size
        binary%patience = options%patience
        binary%shuffle_seed = options%shuffle_seed
        binary%initialization_seed = options%initialization_seed
        binary%hidden_activation = options%hidden_activation
        binary%shuffle = options%shuffle
        binary%restore_best = options%restore_best
        binary%learning_rate = options%learning_rate
        binary%beta1 = options%beta1
        binary%beta2 = options%beta2
        binary%epsilon = options%epsilon
        binary%l2 = options%l2
        binary%tolerance = options%tolerance
        binary%min_delta = options%min_delta
    end function to_binary_options

end module fortml_mlp_multilabel_classifier
