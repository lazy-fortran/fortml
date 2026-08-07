module fortml_gp_variational_multiclass_classification
    !! One-vs-rest multiclass variational GP classification.
    !!
    !! The wrapper owns one bounded Bernoulli variational GP per sorted class.
    !! Every class therefore has an independent inducing posterior and packed
    !! variational vector, while prediction normalizes the positive columns
    !! onto a probability simplex.  The objective is intentionally explicit:
    !! callers can pass ``elbo_gradient``/``elbo_jvp`` to FortOpt or to the
    !! generic FortML trainer.  No hidden optimizer or host fallback is used.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t
    use fortml_gp_variational_classification, only: &
        gp_variational_classification_t, GP_VARIATIONAL_LOGISTIC, &
        GP_VARIATIONAL_PROBIT
    implicit none
    private

    type, public :: gp_variational_multiclass_options_t
        integer :: likelihood = GP_VARIATIONAL_LOGISTIC
        integer :: n_mc_samples = 16
        integer :: seed = 1
        real(dp) :: jitter = 1.0e-8_dp
    end type gp_variational_multiclass_options_t

    type, public :: gp_variational_multiclass_classification_t
        private
        type(gp_variational_classification_t), allocatable :: models(:)
        integer, allocatable :: class_label(:)
        integer :: n_classes = 0
        integer :: n_features = 0
        logical :: is_initialized = .false.
    contains
        procedure, public :: initialize => gvmc_initialize
        procedure, public :: set_parameters => gvmc_set_parameters
        procedure, public :: parameters => gvmc_parameters
        procedure, public :: parameter_count => gvmc_parameter_count
        procedure, public :: elbo => gvmc_elbo
        procedure, public :: elbo_gradient => gvmc_elbo_gradient
        procedure, public :: elbo_jvp => gvmc_elbo_jvp
        procedure, public :: predict_latent => gvmc_predict_latent
        procedure, public :: predict_proba => gvmc_predict_proba
        procedure, public :: predict_proba_parameter_jvp => &
            gvmc_predict_proba_parameter_jvp
        procedure, public :: predict => gvmc_predict
        procedure, public :: elbo_device => gvmc_elbo_device
        procedure, public :: predict_proba_device => gvmc_predict_proba_device
        procedure, public :: classes => gvmc_classes
        procedure, public :: class_count => gvmc_class_count
        procedure, public :: feature_count => gvmc_feature_count
        procedure, public :: initialized => gvmc_initialized
        procedure, public :: device_supported => gvmc_device_supported
    end type gp_variational_multiclass_classification_t

contains

    subroutine gvmc_initialize(self, inducing_points, classes, kernel, &
            n_mc_samples, seed, status, likelihood, jitter)
        class(gp_variational_multiclass_classification_t), intent(out) :: self
        real(dp), intent(in) :: inducing_points(:, :)
        integer, intent(in) :: classes(:)
        type(kernel_t), intent(in) :: kernel
        integer, intent(in) :: n_mc_samples, seed
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: likelihood
        real(dp), intent(in), optional :: jitter
        integer, allocatable :: sorted_classes(:)
        integer :: i, requested_likelihood
        real(dp) :: requested_jitter

        requested_likelihood = GP_VARIATIONAL_LOGISTIC
        if (present(likelihood)) requested_likelihood = likelihood
        requested_jitter = 1.0e-8_dp
        if (present(jitter)) requested_jitter = jitter
        if (size(classes) < 2 .or. size(inducing_points, 1) < 1 .or. &
            size(inducing_points, 2) /= kernel%input_dim .or. n_mc_samples < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: initialization arguments are invalid")
            return
        end if
        if (requested_jitter <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: initialization arguments are invalid")
            return
        end if
        if (.not. ieee_is_finite(requested_jitter)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: initialization arguments are invalid")
            return
        end if
        if (requested_likelihood /= GP_VARIATIONAL_LOGISTIC .and. &
            requested_likelihood /= GP_VARIATIONAL_PROBIT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: initialization arguments are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(inducing_points))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: initialization arguments are invalid")
            return
        end if
        call sorted_unique(classes, sorted_classes)
        if (size(sorted_classes) /= size(classes)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: class labels must be unique")
            return
        end if
        self%n_classes = size(sorted_classes)
        self%n_features = kernel%input_dim
        allocate(self%class_label(self%n_classes), self%models(self%n_classes))
        self%class_label = sorted_classes
        do i = 1, self%n_classes
            call self%models(i)%initialize(inducing_points, kernel, n_mc_samples, &
                seed + i - 1, status, likelihood=requested_likelihood, jitter=requested_jitter)
            if (status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP multiclass: class initialization failed")
                return
            end if
        end do
        self%is_initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvmc_initialize

    integer function gvmc_parameter_count(self) result(count)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. self%initialized()) return
        do i = 1, self%n_classes
            count = count + self%models(i)%parameter_count()
        end do
    end function gvmc_parameter_count

    function gvmc_parameters(self) result(parameters)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:), local(:)
        integer :: i, first, last

        if (.not. self%initialized()) then
            allocate(parameters(0))
            return
        end if
        allocate(parameters(self%parameter_count()))
        first = 1
        do i = 1, self%n_classes
            local = self%models(i)%parameters()
            last = first + size(local) - 1
            parameters(first:last) = local
            first = last + 1
        end do
    end function gvmc_parameters

    subroutine gvmc_set_parameters(self, parameters, status)
        class(gp_variational_multiclass_classification_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last, local_count

        if (.not. self%initialized() .or. size(parameters) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: packed parameter vector is invalid")
            return
        end if
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            last = first + local_count - 1
            call self%models(i)%set_parameters(parameters(first:last), status)
            if (status%code /= FORTNUM_OK) return
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvmc_set_parameters

    subroutine gvmc_elbo(self, x, labels, value, status, scale)
        class(gp_variational_multiclass_classification_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale
        integer, allocatable :: binary_labels(:)
        real(dp) :: local_value, multiplier
        integer :: i

        value = 0.0_dp
        if (.not. valid_data(self, x, labels, status)) return
        multiplier = 1.0_dp
        if (present(scale)) multiplier = scale
        if (multiplier <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: likelihood scale is invalid")
            return
        end if
        if (.not. ieee_is_finite(multiplier)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: likelihood scale is invalid")
            return
        end if
        allocate(binary_labels(size(labels)))
        do i = 1, self%n_classes
            call encode_labels(labels, self%class_label(i), binary_labels, status)
            if (status%code /= FORTNUM_OK) return
            call self%models(i)%elbo(x, binary_labels, local_value, status, scale=multiplier)
            if (status%code /= FORTNUM_OK) return
            value = value + local_value
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvmc_elbo

    subroutine gvmc_elbo_gradient(self, x, labels, value, gradient, status, scale)
        class(gp_variational_multiclass_classification_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale
        integer, allocatable :: binary_labels(:)
        real(dp), allocatable :: local_gradient(:)
        real(dp) :: local_value, multiplier
        integer :: i, first, last, local_count

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. valid_data(self, x, labels, status)) return
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: gradient shape is invalid")
            return
        end if
        multiplier = 1.0_dp
        if (present(scale)) multiplier = scale
        if (multiplier <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: likelihood scale is invalid")
            return
        end if
        if (.not. ieee_is_finite(multiplier)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: likelihood scale is invalid")
            return
        end if
        allocate(binary_labels(size(labels)))
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            last = first + local_count - 1
            allocate(local_gradient(local_count))
            call encode_labels(labels, self%class_label(i), binary_labels, status)
            if (status%code /= FORTNUM_OK) return
            call self%models(i)%elbo_gradient(x, binary_labels, local_value, &
                local_gradient, status, scale=multiplier)
            if (status%code /= FORTNUM_OK) return
            value = value + local_value
            gradient(first:last) = local_gradient
            deallocate(local_gradient)
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvmc_elbo_gradient

    subroutine gvmc_elbo_jvp(self, x, labels, direction, value, tangent, status, scale)
        class(gp_variational_multiclass_classification_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale
        integer, allocatable :: binary_labels(:)
        real(dp), allocatable :: local_direction(:)
        real(dp) :: local_value, local_tangent, multiplier
        integer :: i, first, last, local_count

        value = 0.0_dp
        tangent = 0.0_dp
        if (.not. valid_data(self, x, labels, status)) return
        if (size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: JVP direction is invalid")
            return
        end if
        multiplier = 1.0_dp
        if (present(scale)) multiplier = scale
        if (multiplier <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: likelihood scale is invalid")
            return
        end if
        if (.not. ieee_is_finite(multiplier)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: likelihood scale is invalid")
            return
        end if
        allocate(binary_labels(size(labels)))
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            last = first + local_count - 1
            allocate(local_direction(local_count))
            local_direction = direction(first:last)
            call encode_labels(labels, self%class_label(i), binary_labels, status)
            if (status%code /= FORTNUM_OK) return
            call self%models(i)%elbo_jvp(x, binary_labels, local_direction, &
                local_value, local_tangent, status, scale=multiplier)
            if (status%code /= FORTNUM_OK) return
            value = value + local_value
            tangent = tangent + local_tangent
            deallocate(local_direction)
            first = last + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvmc_elbo_jvp

    subroutine gvmc_predict_latent(self, x, margins, variances, status)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :), variances(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_mean(:), local_variance(:)
        integer :: i

        if (.not. prediction_valid(self, x, margins, variances, status)) return
        allocate(local_mean(size(x, 1)), local_variance(size(x, 1)))
        do i = 1, self%n_classes
            call self%models(i)%predict_latent(x, local_mean, local_variance, status)
            if (status%code /= FORTNUM_OK) return
            margins(:, i) = local_mean
            variances(:, i) = local_variance
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvmc_predict_latent

    subroutine gvmc_predict_proba(self, x, probabilities, status)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_matrix(:, :)
        real(dp) :: total
        integer :: i, j

        if (.not. prediction_probability_valid(self, x, probabilities, status)) return
        allocate(local_matrix(size(x, 1), 2))
        do i = 1, self%n_classes
            call self%models(i)%predict_proba(x, local_matrix, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i) = local_matrix(:, 2)
        end do
        do j = 1, size(x, 1)
            total = sum(probabilities(j, :))
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP multiclass: invalid probability normalization")
                return
            end if
            probabilities(j, :) = probabilities(j, :)/total
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvmc_predict_proba

    subroutine gvmc_predict_proba_parameter_jvp(self, x, direction, probabilities, &
            probabilities_dot, status)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: raw(:, :), raw_dot(:, :), local_matrix(:, :), &
            local_matrix_dot(:, :)
        real(dp) :: total, total_dot
        integer :: i, j, first, last, local_count

        if (.not. prediction_probability_valid(self, x, probabilities, status)) return
        if (any(shape(probabilities_dot) /= shape(probabilities)) .or. &
            size(direction) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass probability JVP: shape is invalid")
            return
        end if
        allocate(raw(size(x, 1), self%n_classes), raw_dot(size(x, 1), self%n_classes), &
            local_matrix(size(x, 1), 2), local_matrix_dot(size(x, 1), 2))
        first = 1
        do i = 1, self%n_classes
            local_count = self%models(i)%parameter_count()
            last = first + local_count - 1
            call self%models(i)%predict_proba_parameter_jvp(x, direction(first:last), &
                local_matrix, local_matrix_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, i) = local_matrix(:, 2)
            raw_dot(:, i) = local_matrix_dot(:, 2)
            first = last + 1
        end do
        do j = 1, size(x, 1)
            total = sum(raw(j, :))
            total_dot = sum(raw_dot(j, :))
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP multiclass probability JVP: invalid normalization")
                return
            end if
            probabilities(j, :) = raw(j, :)/total
            probabilities_dot(j, :) = (raw_dot(j, :)*total - raw(j, :)*total_dot)/ &
                (total*total)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvmc_predict_proba_parameter_jvp

    subroutine gvmc_predict(self, x, labels, status)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. prediction_input_valid(self, x, status) .or. size(labels) /= size(x, 1)) then
            if (status%code == FORTNUM_OK) call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: label output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvmc_predict

    subroutine gvmc_elbo_device(self, device, x, labels, value, status, scale)
        class(gp_variational_multiclass_classification_t), intent(inout) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale

        value = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            if (present(scale)) then
                call self%elbo(x, labels, value, status, scale=scale)
            else
                call self%elbo(x, labels, value, status)
            end if
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "variational GP multiclass device: resident CUDA OVR graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass device: device kind is invalid")
        end select
    end subroutine gvmc_elbo_device

    subroutine gvmc_predict_proba_device(self, device, x, probabilities, status)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "variational GP multiclass device: resident CUDA OVR graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass device: device kind is invalid")
        end select
    end subroutine gvmc_predict_proba_device

    function gvmc_classes(self) result(classes)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function gvmc_classes

    integer function gvmc_class_count(self) result(count)
        class(gp_variational_multiclass_classification_t), intent(in) :: self

        count = self%n_classes
    end function gvmc_class_count

    integer function gvmc_feature_count(self) result(count)
        class(gp_variational_multiclass_classification_t), intent(in) :: self

        count = self%n_features
    end function gvmc_feature_count

    logical function gvmc_initialized(self) result(initialized)
        class(gp_variational_multiclass_classification_t), intent(in) :: self

        initialized = self%is_initialized .and. allocated(self%models) .and. &
            allocated(self%class_label) .and. self%n_classes >= 2
    end function gvmc_initialized

    logical function gvmc_device_supported(self, device_kind) result(supported)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = self%initialized()
        case default
            supported = .false.
        end select
    end function gvmc_device_supported

    logical function valid_data(self, x, labels, status) result(valid)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j
        logical :: known

        valid = .false.
        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: data shape or values are invalid")
            return
        end if
        do i = 1, size(labels)
            known = .false.
            do j = 1, self%n_classes
                if (labels(i) == self%class_label(j)) known = .true.
            end do
            if (.not. known) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "variational GP multiclass: label is not in initialized classes")
                return
            end if
        end do
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function valid_data

    logical function prediction_input_valid(self, x, status) result(valid)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = self%initialized() .and. size(x, 1) >= 1 .and. &
            size(x, 2) == self%n_features .and. .not. any(.not. ieee_is_finite(x))
        if (.not. valid) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: prediction input is invalid")
        else
            call status_set(status, FORTNUM_OK, "")
        end if
    end function prediction_input_valid

    logical function prediction_valid(self, x, margins, variances, status) result(valid)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), margins(:, :), variances(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = prediction_input_valid(self, x, status)
        if (.not. valid) return
        if (any(shape(margins) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(variances) /= shape(margins))) then
            valid = .false.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: latent output shape is invalid")
        end if
    end function prediction_valid

    logical function prediction_probability_valid(self, x, probabilities, status) &
            result(valid)
        class(gp_variational_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = prediction_input_valid(self, x, status)
        if (.not. valid) return
        if (any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            valid = .false.
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: probability output shape is invalid")
        end if
    end function prediction_probability_valid

    subroutine encode_labels(labels, positive, encoded, status)
        integer, intent(in) :: labels(:), positive
        integer, intent(out) :: encoded(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (size(encoded) /= size(labels)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "variational GP multiclass: encoded label shape is invalid")
            return
        end if
        do i = 1, size(labels)
            encoded(i) = merge(1, 0, labels(i) == positive)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine encode_labels

    subroutine sorted_unique(input, output)
        integer, intent(in) :: input(:)
        integer, allocatable, intent(out) :: output(:)
        integer, allocatable :: work(:)
        integer :: i, j, n, temp
        logical :: found

        allocate(work(size(input)))
        n = 0
        do i = 1, size(input)
            found = .false.
            do j = 1, n
                if (work(j) == input(i)) found = .true.
            end do
            if (.not. found) then
                n = n + 1
                work(n) = input(i)
            end if
        end do
        do i = 1, n - 1
            do j = i + 1, n
                if (work(j) < work(i)) then
                    temp = work(i)
                    work(i) = work(j)
                    work(j) = temp
                end if
            end do
        end do
        allocate(output(n))
        output = work(:n)
    end subroutine sorted_unique

end module fortml_gp_variational_multiclass_classification
