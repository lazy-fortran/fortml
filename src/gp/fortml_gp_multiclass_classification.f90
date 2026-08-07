module fortml_gp_multiclass_classification
    !! One-vs-rest multiclass Gaussian-process classification.
    !!
    !! Each class owns an independent Laplace GP classifier from
    !! ``fortml_gp_classification``.  The positive columns are normalized over
    !! classes after prediction, giving a deterministic probability simplex
    !! while preserving the binary likelihood and kernel contracts.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortml_kernels, only: kernel_t
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_options_t, gp_classification_state_t, &
        GP_LIKELIHOOD_LOGISTIC, GP_LIKELIHOOD_PROBIT
    implicit none
    private

    type, public :: gp_multiclass_classification_options_t
        integer :: likelihood = GP_LIKELIHOOD_LOGISTIC
        integer :: max_iterations = 50
        real(dp) :: tolerance = 1.0e-8_dp
        real(dp) :: jitter = 1.0e-8_dp
        real(dp) :: damping = 1.0_dp
    end type gp_multiclass_classification_options_t

    type, public :: gp_multiclass_classification_state_t
        integer :: class_count = 0
        integer :: total_iterations = 0
        real(dp) :: log_posterior = -huge(1.0_dp)
        logical :: converged = .false.
    end type gp_multiclass_classification_state_t

    type, public :: gp_multiclass_classification_t
        private
        type(gp_classification_t), allocatable :: models(:)
        integer, allocatable :: class_label(:)
        integer :: n_classes = 0
        integer :: n_features = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => gp_multiclass_classification_fit
        procedure, public :: predict_proba => &
            gp_multiclass_classification_predict_proba
        procedure, public :: predict_proba_jvp => &
            gp_multiclass_classification_predict_proba_jvp
        procedure, public :: predict => gp_multiclass_classification_predict
        procedure, public :: classes => gp_multiclass_classification_classes
        procedure, public :: class_count => gp_multiclass_classification_class_count
        procedure, public :: feature_count => &
            gp_multiclass_classification_feature_count
        procedure, public :: parameter_count => &
            gp_multiclass_classification_parameter_count
        procedure, public :: parameters => gp_multiclass_classification_parameters
        procedure, public :: hyperparameter_gradient => &
            gp_multiclass_classification_hyperparameter_gradient
        procedure, public :: fitted => gp_multiclass_classification_fitted
    end type gp_multiclass_classification_t

    public :: gp_multiclass_classification_fit
    public :: gp_multiclass_classification_predict_proba
    public :: gp_multiclass_classification_predict_proba_jvp
    public :: gp_multiclass_classification_predict

contains

    subroutine gp_multiclass_classification_fit( &
            self, x, labels, kernel, status, options, state)
        class(gp_multiclass_classification_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), intent(in) :: kernel
        type(fortnum_status_t), intent(out) :: status
        type(gp_multiclass_classification_options_t), intent(in), optional :: options
        type(gp_multiclass_classification_state_t), intent(out), optional :: state
        type(gp_multiclass_classification_options_t) :: requested
        type(gp_classification_options_t) :: binary_options
        type(gp_classification_state_t) :: binary_state
        type(gp_multiclass_classification_state_t) :: result
        integer, allocatable :: unique_labels(:), binary_labels(:)
        integer :: i, class_count

        result = gp_multiclass_classification_state_t()
        if (present(state)) state = result
        requested = gp_multiclass_classification_options_t()
        if (present(options)) requested = options
        if (.not. valid_options(requested)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: options are invalid")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. &
            size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: inputs must be finite")
            return
        end if
        call sorted_unique_labels(labels, unique_labels)
        class_count = size(unique_labels)
        if (class_count < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: at least two classes are required")
            return
        end if
        if (kernel%input_dim /= size(x, 2) .or. kernel%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification fit: kernel dimension is invalid")
            return
        end if

        allocate(self%models(class_count), self%class_label(class_count))
        self%class_label = unique_labels
        self%n_classes = class_count
        self%n_features = size(x, 2)
        self%is_fitted = .false.
        allocate(binary_labels(size(labels)))
        binary_options%likelihood = requested%likelihood
        binary_options%max_iterations = requested%max_iterations
        binary_options%tolerance = requested%tolerance
        binary_options%jitter = requested%jitter
        binary_options%damping = requested%damping
        result%class_count = class_count
        result%log_posterior = 0.0_dp
        do i = 1, class_count
            binary_labels = 0
            where (labels == unique_labels(i)) binary_labels = 1
            call self%models(i)%fit(x, binary_labels, kernel, status, &
                binary_options, binary_state)
            if (status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass classification fit: one-vs-rest model failed")
                return
            end if
            result%total_iterations = result%total_iterations + binary_state%iterations
            result%log_posterior = result%log_posterior + binary_state%log_posterior
        end do
        result%converged = .true.
        self%is_fitted = .true.
        if (present(state)) state = result
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_fit

    subroutine gp_multiclass_classification_predict_proba( &
            self, x, probabilities, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :)
        real(dp) :: normalization
        integer :: i, j

        if (.not. prediction_shapes(self, x, probabilities, status)) return
        allocate(binary_probabilities(size(x, 1), 2))
        probabilities = 0.0_dp
        do j = 1, self%n_classes
            call self%models(j)%predict_proba(x, binary_probabilities, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, j) = binary_probabilities(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(probabilities(i, :))
            if (.not. ieee_is_finite(normalization) .or. normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass classification prediction: invalid probability sum")
                return
            end if
            probabilities(i, :) = probabilities(i, :)/normalization
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_proba

    subroutine gp_multiclass_classification_predict_proba_jvp( &
            self, x, x_dot, probabilities, probabilities_dot, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), binary_probability_dot(:, :)
        real(dp), allocatable :: raw(:, :), raw_dot(:, :)
        real(dp) :: normalization, normalization_dot
        integer :: i, j

        if (.not. prediction_shapes(self, x, probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. &
            any(.not. ieee_is_finite(x_dot)) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification probability JVP: shape is invalid")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2))
        allocate(binary_probability_dot(size(x, 1), 2))
        allocate(raw(size(x, 1), self%n_classes), raw_dot(size(x, 1), self%n_classes))
        do j = 1, self%n_classes
            call self%models(j)%predict_proba_jvp(x, x_dot, binary_probabilities, &
                binary_probability_dot, status)
            if (status%code /= FORTNUM_OK) return
            raw(:, j) = binary_probabilities(:, 2)
            raw_dot(:, j) = binary_probability_dot(:, 2)
        end do
        do i = 1, size(x, 1)
            normalization = sum(raw(i, :))
            normalization_dot = sum(raw_dot(i, :))
            if (.not. ieee_is_finite(normalization) .or. &
                normalization <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GP multiclass classification probability JVP: invalid sum")
                return
            end if
            probabilities(i, :) = raw(i, :)/normalization
            probabilities_dot(i, :) = (raw_dot(i, :)*normalization - &
                raw(i, :)*normalization_dot)/(normalization*normalization)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict_proba_jvp

    subroutine gp_multiclass_classification_predict(self, x, labels, status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. prediction_input_valid(self, x, status)) return
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification prediction: label shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gp_multiclass_classification_predict

    function gp_multiclass_classification_classes(self) result(classes)
        class(gp_multiclass_classification_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function gp_multiclass_classification_classes

    integer function gp_multiclass_classification_class_count(self) result(count)
        class(gp_multiclass_classification_t), intent(in) :: self

        count = self%n_classes
    end function gp_multiclass_classification_class_count

    integer function gp_multiclass_classification_feature_count(self) result(count)
        class(gp_multiclass_classification_t), intent(in) :: self

        count = self%n_features
    end function gp_multiclass_classification_feature_count

    integer function gp_multiclass_classification_parameter_count(self) result(count)
        class(gp_multiclass_classification_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. self%fitted()) return
        do i = 1, self%n_classes
            count = count + self%models(i)%parameter_count()
        end do
    end function gp_multiclass_classification_parameter_count

    function gp_multiclass_classification_parameters(self) result(parameters)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        real(dp), allocatable :: model_parameters(:)
        integer :: i, first, last

        if (.not. self%fitted()) then
            allocate(parameters(0))
            return
        end if
        allocate(parameters(self%parameter_count()))
        first = 1
        do i = 1, self%n_classes
            model_parameters = self%models(i)%parameters()
            last = first + size(model_parameters) - 1
            parameters(first:last) = model_parameters
            first = last + 1
        end do
    end function gp_multiclass_classification_parameters

    subroutine gp_multiclass_classification_hyperparameter_gradient(self, gradient, &
            status)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(out) :: gradient(:)
        type(fortnum_status_t), intent(out) :: status

        gradient = 0.0_dp
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass hyperparameter gradient: model is not fitted")
            return
        end if
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass hyperparameter gradient: output shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            "GP multiclass hyperparameter gradient: Laplace parameter products are not implemented")
    end subroutine gp_multiclass_classification_hyperparameter_gradient

    logical function gp_multiclass_classification_fitted(self) result(fitted)
        class(gp_multiclass_classification_t), intent(in) :: self

        fitted = self%is_fitted .and. allocated(self%models) .and. &
            allocated(self%class_label) .and. self%n_classes >= 2
    end function gp_multiclass_classification_fitted

    logical function prediction_shapes(self, x, probabilities, status) result(valid)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. prediction_input_valid(self, x, status)) return
        if (any(shape(probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification prediction: probability shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_shapes

    logical function prediction_input_valid(self, x, status) result(valid)
        class(gp_multiclass_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = .false.
        if (.not. self%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification prediction: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GP multiclass classification prediction: input shape is invalid")
            return
        end if
        valid = .true.
        call status_set(status, FORTNUM_OK, "")
    end function prediction_input_valid

    subroutine sorted_unique_labels(labels, unique_labels)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: unique_labels(:)
        integer, allocatable :: work(:)
        integer :: i, j, count, candidate
        logical :: found

        allocate(work(size(labels)))
        count = 0
        do i = 1, size(labels)
            candidate = labels(i)
            found = .false.
            do j = 1, count
                if (work(j) == candidate) then
                    found = .true.
                    exit
                end if
            end do
            if (.not. found) then
                count = count + 1
                work(count) = candidate
            end if
        end do
        do i = 1, count - 1
            do j = i + 1, count
                if (work(j) < work(i)) then
                    candidate = work(i)
                    work(i) = work(j)
                    work(j) = candidate
                end if
            end do
        end do
        allocate(unique_labels(count))
        unique_labels = work(:count)
    end subroutine sorted_unique_labels

    logical function valid_options(options) result(valid)
        type(gp_multiclass_classification_options_t), intent(in) :: options

        valid = (options%likelihood == GP_LIKELIHOOD_LOGISTIC .or. &
            options%likelihood == GP_LIKELIHOOD_PROBIT) .and. &
            options%max_iterations >= 1 .and. &
            ieee_is_finite(options%tolerance) .and. options%tolerance > 0.0_dp .and. &
            ieee_is_finite(options%jitter) .and. options%jitter >= 0.0_dp .and. &
            ieee_is_finite(options%damping) .and. options%damping > 0.0_dp .and. &
            options%damping <= 1.0_dp
    end function valid_options

end module fortml_gp_multiclass_classification
