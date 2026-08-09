module fortml_bernoulli_naive_bayes
    !! Weighted, differentiable Bernoulli Naive Bayes classification.
    !!
    !! Samples are rows and features are columns.  Features may be binary or
    !! relaxed Bernoulli values in [0,1]; the latter gives a smooth extension
    !! useful for input derivatives.  Classes are sorted in ascending integer
    !! order.  The packed parameter order is feature probabilities followed by
    !! class priors, each in Fortran column-major order.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    real(dp), parameter :: DEFAULT_ALPHA = 1.0_dp

    type, public :: bernoulli_naive_bayes_t
        private
        real(dp), allocatable :: feature_probability(:, :)
        real(dp), allocatable :: prior(:)
        real(dp), allocatable :: weighted_count(:)
        integer, allocatable :: class_label(:)
        integer :: n_features = 0
        integer :: n_classes = 0
        real(dp) :: alpha = DEFAULT_ALPHA
        logical :: is_fitted = .false.
        real(dp), allocatable :: partial_x(:, :), partial_weight(:)
        integer, allocatable :: partial_labels(:), partial_classes(:)
        real(dp) :: partial_alpha = DEFAULT_ALPHA
        integer :: partial_sample_count = 0, partial_batch_count = 0
        logical :: partial_initialized = .false.
    contains
        procedure, public :: fit => bernoulli_nb_fit
        procedure, public :: partial_fit => bernoulli_nb_partial_fit
        procedure, public :: warm_start => bernoulli_nb_partial_fit
        procedure, public :: partial_fit_device => bernoulli_nb_partial_fit_device
        procedure, public :: predict_log_proba => bernoulli_nb_predict_log_proba
        procedure, public :: predict_proba => bernoulli_nb_predict_proba
        procedure, public :: predict => bernoulli_nb_predict
        procedure, public :: predict_log_proba_jvp => bernoulli_nb_predict_log_proba_jvp
        procedure, public :: predict_proba_jvp => bernoulli_nb_predict_proba_jvp
        procedure, public :: predict_log_proba_parameter_jvp => &
            bernoulli_nb_predict_log_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            bernoulli_nb_predict_proba_parameter_jvp
        procedure, public :: predict_log_proba_vjp => bernoulli_nb_predict_log_proba_vjp
        procedure, public :: predict_proba_vjp => bernoulli_nb_predict_proba_vjp
        procedure, public :: predict_log_proba_parameter_vjp => &
            bernoulli_nb_predict_log_proba_parameter_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            bernoulli_nb_predict_proba_parameter_vjp
        procedure, public :: classes => bernoulli_nb_classes
        procedure, public :: feature_probabilities => bernoulli_nb_feature_probabilities
        procedure, public :: class_prior => bernoulli_nb_class_prior
        procedure, public :: weighted_class_counts => bernoulli_nb_weighted_counts
        procedure, public :: alpha_value => bernoulli_nb_alpha
        procedure, public :: feature_count => bernoulli_nb_feature_count
        procedure, public :: class_count => bernoulli_nb_class_count
        procedure, public :: parameter_count => bernoulli_nb_parameter_count
        procedure, public :: parameters => bernoulli_nb_parameters
        procedure, public :: set_parameters => bernoulli_nb_set_parameters
        procedure, public :: fitted => bernoulli_nb_fitted
        procedure, public :: partial_fit_initialized => bernoulli_nb_partial_initialized
        procedure, public :: sample_count => bernoulli_nb_sample_count
        procedure, public :: batch_count => bernoulli_nb_batch_count
        procedure, public :: device_supported => bernoulli_nb_device_supported
    end type bernoulli_naive_bayes_t

    public :: bernoulli_nb_fit
    public :: bernoulli_nb_partial_fit
    public :: bernoulli_nb_partial_fit_device
    public :: bernoulli_nb_predict_log_proba
    public :: bernoulli_nb_predict_proba
    public :: bernoulli_nb_predict

contains

    subroutine bernoulli_nb_fit(self, x, labels, status, alpha, priors, &
            sample_weight, class_weight)
        class(bernoulli_naive_bayes_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha, priors(:), sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:)
        integer, allocatable :: classes(:)
        real(dp), allocatable :: weights(:), class_factors(:)
        real(dp) :: requested_alpha, total_weight, prior_sum, success_mass
        integer :: i, j, c, n_samples, n_features, n_classes

        self%is_fitted = .false.
        requested_alpha = DEFAULT_ALPHA
        if (present(alpha)) requested_alpha = alpha
        if (.not. ieee_is_finite(requested_alpha) .or. requested_alpha <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: alpha must be finite and strictly positive")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: input must have at least one sample and feature")
            return
        end if
        if (size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: label shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: inputs must be finite")
            return
        end if
        if (any(x < 0.0_dp) .or. any(x > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: features must lie in [0,1]")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: at least two classes are required")
            return
        end if

        allocate(weights(n_samples), class_factors(n_classes))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB fit: sample-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB fit: class-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB fit: class weights must be finite and positive")
                return
            end if
            class_factors = class_weight
            do i = 1, n_samples
                do c = 1, n_classes
                    if (labels(i) == classes(c)) then
                        weights(i) = weights(i)*class_factors(c)
                        exit
                    end if
                end do
            end do
        end if
        if (any(.not. ieee_is_finite(weights))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: effective weights are not finite")
            return
        end if
        total_weight = sum(weights)
        if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: effective weights must have positive mass")
            return
        end if

        allocate(self%feature_probability(n_features, n_classes), &
            self%prior(n_classes), self%weighted_count(n_classes), &
            self%class_label(n_classes))
        self%feature_probability = 0.0_dp
        self%weighted_count = 0.0_dp
        self%class_label = classes
        do c = 1, n_classes
            do i = 1, n_samples
                if (labels(i) == classes(c)) self%weighted_count(c) = &
                    self%weighted_count(c) + weights(i)
            end do
            if (.not. ieee_is_finite(self%weighted_count(c)) .or. &
                self%weighted_count(c) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB fit: every class needs positive effective weight")
                return
            end if
            do j = 1, n_features
                success_mass = 0.0_dp
                do i = 1, n_samples
                    if (labels(i) == classes(c)) success_mass = success_mass + &
                        weights(i)*x(i, j)
                end do
                self%feature_probability(j, c) = (success_mass + requested_alpha)/ &
                    (self%weighted_count(c) + 2.0_dp*requested_alpha)
                if (.not. ieee_is_finite(self%feature_probability(j, c)) .or. &
                    self%feature_probability(j, c) <= 0.0_dp .or. &
                    self%feature_probability(j, c) >= 1.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "BernoulliNB fit: smoothed feature probability is invalid")
                    return
                end if
            end do
        end do

        if (present(priors)) then
            if (size(priors) /= n_classes) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB fit: prior shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(priors)) .or. any(priors <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB fit: priors must be finite and positive")
                return
            end if
            self%prior = priors
        else
            self%prior = self%weighted_count
        end if
        prior_sum = sum(self%prior)
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: prior mass must be finite and positive")
            return
        end if
        self%prior = self%prior/prior_sum
        if (any(.not. ieee_is_finite(self%prior)) .or. any(self%prior <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB fit: normalized priors are invalid")
            return
        end if
        self%n_features = n_features
        self%n_classes = n_classes
        self%alpha = requested_alpha
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_fit

    subroutine bernoulli_nb_partial_fit(self, x, labels, status, classes, alpha, &
            sample_weight)
        !! Transactionally append a validated Bernoulli batch.
        !!
        !! The first call declares the complete sorted class vocabulary.  The
        !! replay buffer keeps the sufficient statistics exact, including when
        !! a class is absent from an early batch.  Any rejected batch leaves
        !! both the stream metadata and fitted parameters unchanged.
        class(bernoulli_naive_bayes_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: classes(:)
        real(dp), intent(in), optional :: alpha, sample_weight(:)
        type(bernoulli_naive_bayes_t) :: candidate
        type(fortnum_status_t) :: candidate_status
        integer, allocatable :: candidate_classes(:), candidate_labels(:)
        real(dp), allocatable :: candidate_x(:, :), candidate_weight(:), batch_weight(:)
        real(dp) :: requested_alpha
        integer :: n_samples, n_features, n_old, n_classes, i, c
        logical :: first_call, complete

        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1 .or. size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB partial_fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. any(x > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB partial_fit: features must be finite and in [0,1]")
            return
        end if
        allocate(batch_weight(n_samples)); batch_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB partial_fit: sample weights are invalid")
                return
            end if
            batch_weight = sample_weight
        end if
        if (.not. ieee_is_finite(sum(batch_weight)) .or. sum(batch_weight) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB partial_fit: batch weight mass must be positive")
            return
        end if
        requested_alpha = DEFAULT_ALPHA
        if (present(alpha)) requested_alpha = alpha
        first_call = .not. self%partial_initialized
        if (first_call) then
            if (self%is_fitted .or. .not. present(classes)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB partial_fit: fresh model and classes are required")
                return
            end if
            if (size(classes) < 2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB partial_fit: at least two classes are required")
                return
            end if
            do i = 2, size(classes)
                if (classes(i) <= classes(i - 1)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "BernoulliNB partial_fit: classes must be strictly increasing")
                    return
                end if
            end do
            if (.not. ieee_is_finite(requested_alpha) .or. requested_alpha <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB partial_fit: alpha must be finite and positive")
                return
            end if
            allocate(candidate_classes(size(classes))); candidate_classes = classes
        else
            if (n_features /= self%n_features) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB partial_fit: feature count changed")
                return
            end if
            if (present(classes)) then
                if (size(classes) /= size(self%partial_classes) .or. &
                    any(classes /= self%partial_classes)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "BernoulliNB partial_fit: class vocabulary changed")
                    return
                end if
            end if
            requested_alpha = self%partial_alpha
            if (present(alpha)) then
                if (.not. ieee_is_finite(alpha) .or. alpha /= self%partial_alpha) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "BernoulliNB partial_fit: alpha cannot change")
                    return
                end if
            end if
            allocate(candidate_classes(size(self%partial_classes)))
            candidate_classes = self%partial_classes
        end if
        if (.not. ieee_is_finite(requested_alpha) .or. requested_alpha <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB partial_fit: alpha must be finite and positive")
            return
        end if
        n_classes = size(candidate_classes)
        do i = 1, n_samples
            if (.not. any(labels(i) == candidate_classes)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB partial_fit: batch contains an unseen class")
                return
            end if
        end do
        n_old = 0
        if (allocated(self%partial_labels)) n_old = size(self%partial_labels)
        allocate(candidate_x(n_old + n_samples, n_features), &
            candidate_labels(n_old + n_samples), candidate_weight(n_old + n_samples))
        if (n_old > 0) then
            candidate_x(:n_old, :) = self%partial_x
            candidate_labels(:n_old) = self%partial_labels
            candidate_weight(:n_old) = self%partial_weight
        end if
        candidate_x(n_old + 1:, :) = x
        candidate_labels(n_old + 1:) = labels
        candidate_weight(n_old + 1:) = batch_weight
        complete = .true.
        do c = 1, n_classes
            if (.not. any((candidate_labels == candidate_classes(c)) .and. &
                    (candidate_weight > 0.0_dp))) complete = .false.
        end do
        if (complete) then
            call candidate%fit(candidate_x, candidate_labels, candidate_status, &
                alpha=requested_alpha, sample_weight=candidate_weight)
            if (candidate_status%code /= FORTNUM_OK) then
                call status_set(status, candidate_status%code, trim(candidate_status%msg))
                return
            end if
        end if
        self%partial_initialized = .true.
        if (first_call) self%partial_alpha = requested_alpha
        self%partial_sample_count = n_old + n_samples
        self%partial_batch_count = self%partial_batch_count + 1
        if (allocated(self%partial_x)) deallocate(self%partial_x)
        if (allocated(self%partial_labels)) deallocate(self%partial_labels)
        if (allocated(self%partial_weight)) deallocate(self%partial_weight)
        if (allocated(self%partial_classes)) deallocate(self%partial_classes)
        call move_alloc(candidate_x, self%partial_x)
        call move_alloc(candidate_labels, self%partial_labels)
        call move_alloc(candidate_weight, self%partial_weight)
        call move_alloc(candidate_classes, self%partial_classes)
        self%n_features = n_features; self%n_classes = n_classes
        if (allocated(self%class_label)) deallocate(self%class_label)
        allocate(self%class_label(n_classes)); self%class_label = self%partial_classes
        if (complete) then
            self%feature_probability = candidate%feature_probability
            self%prior = candidate%prior
            self%weighted_count = candidate%weighted_count
            self%alpha = candidate%alpha
            self%is_fitted = .true.
        else
            self%is_fitted = .false.
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_partial_fit

    subroutine bernoulli_nb_partial_fit_device(self, device, x, labels, status, &
            classes, alpha, sample_weight)
        class(bernoulli_naive_bayes_t), intent(inout) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: classes(:)
        real(dp), intent(in), optional :: alpha, sample_weight(:)

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB partial_fit device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%partial_fit(x, labels, status, classes, alpha, sample_weight)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "BernoulliNB partial_fit device: resident CUDA sufficient statistics are not implemented")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB partial_fit device: device kind is invalid")
        end select
    end subroutine bernoulli_nb_partial_fit_device

    subroutine bernoulli_nb_predict_log_proba(self, x, log_probabilities, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_joint(self%n_classes), maximum, normalizer
        integer :: i, j, c

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB log probability: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB log probability: input feature shape is invalid")
            return
        end if
        if (any(shape(log_probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB log probability: output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. &
            any(x > 1.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB log probability: inputs must lie in [0,1]")
            return
        end if
        do i = 1, size(x, 1)
            do c = 1, self%n_classes
                log_joint(c) = log(self%prior(c))
                do j = 1, self%n_features
                    log_joint(c) = log_joint(c) + x(i, j)* &
                        log(self%feature_probability(j, c)) + (1.0_dp - x(i, j))* &
                        log(1.0_dp - self%feature_probability(j, c))
                end do
            end do
            maximum = maxval(log_joint)
            normalizer = sum(exp(log_joint - maximum))
            if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "BernoulliNB log probability: normalization is invalid")
                return
            end if
            log_probabilities(i, :) = log_joint - maximum - log(normalizer)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_predict_log_proba

    subroutine bernoulli_nb_predict_proba(self, x, probabilities, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        call self%predict_log_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(probabilities)
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_predict_proba

    subroutine bernoulli_nb_predict(self, x, labels, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :)
        integer :: i, c

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB predict: output shape is invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes))
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            c = maxloc(log_probabilities(i, :), dim=1)
            labels(i) = self%class_label(c)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_predict

    subroutine bernoulli_nb_predict_log_proba_jvp(self, x, x_dot, log_probabilities, &
            log_probabilities_dot, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: joint_dot(self%n_classes), probabilities(self%n_classes)
        real(dp) :: weighted_dot
        integer :: i, j, c

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB input JVP: tangent shape is invalid")
            return
        end if
        if (any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB input JVP: output tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB input JVP: tangent must be finite")
            return
        end if
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            do c = 1, self%n_classes
                joint_dot(c) = 0.0_dp
                do j = 1, self%n_features
                    joint_dot(c) = joint_dot(c) + x_dot(i, j)* &
                        (log(self%feature_probability(j, c)) - &
                        log(1.0_dp - self%feature_probability(j, c)))
                end do
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_predict_log_proba_jvp

    subroutine bernoulli_nb_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB probability JVP: output shape is invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes), &
            log_dot(size(x, 1), self%n_classes))
        call self%predict_log_proba_jvp(x, x_dot, log_probabilities, log_dot, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(log_probabilities)
        probabilities_dot = probabilities*log_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_predict_proba_jvp

    subroutine bernoulli_nb_predict_log_proba_parameter_jvp(self, x, theta_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probability_dot(:, :), prior_dot(:)
        real(dp) :: joint_dot(self%n_classes), probabilities(self%n_classes)
        real(dp) :: weighted_dot, prior_tangent_sum, q, q_dot
        integer :: i, j, c, offset

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB parameter JVP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(log_probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB parameter JVP: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. &
            any(x > 1.0_dp) .or. any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB parameter JVP: inputs and tangents are invalid")
            return
        end if
        offset = self%n_features*self%n_classes
        allocate(probability_dot(self%n_features, self%n_classes), &
            prior_dot(self%n_classes))
        probability_dot = reshape(theta_dot(:offset), shape(probability_dot))
        prior_dot = theta_dot(offset + 1:)
        prior_tangent_sum = sum(prior_dot)
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            do c = 1, self%n_classes
                joint_dot(c) = prior_dot(c)/self%prior(c) - prior_tangent_sum
                do j = 1, self%n_features
                    q = self%feature_probability(j, c)
                    q_dot = probability_dot(j, c)
                    joint_dot(c) = joint_dot(c) + q_dot*(x(i, j)/q - &
                        (1.0_dp - x(i, j))/(1.0_dp - q))
                end do
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_predict_log_proba_parameter_jvp

    subroutine bernoulli_nb_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB parameter probability JVP: output shape is invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes), &
            log_dot(size(x, 1), self%n_classes))
        call self%predict_log_proba_parameter_jvp(x, theta_dot, log_probabilities, &
            log_dot, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(log_probabilities)
        probabilities_dot = probabilities*log_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_predict_proba_parameter_jvp

    subroutine bernoulli_nb_predict_log_proba_vjp(self, x, log_probabilities_bar, &
            x_bar, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :)
        real(dp) :: probabilities(self%n_classes), joint_bar(self%n_classes)
        real(dp) :: cotangent_sum
        integer :: i, j, c

        x_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB log probability VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. any(shape(x_bar) /= shape(x)) .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB log probability VJP: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. &
            any(x > 1.0_dp) .or. any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB log probability VJP: inputs are invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes))
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            cotangent_sum = sum(log_probabilities_bar(i, :))
            do c = 1, self%n_classes
                joint_bar(c) = log_probabilities_bar(i, c) - cotangent_sum*probabilities(c)
                do j = 1, self%n_features
                    x_bar(i, j) = x_bar(i, j) + joint_bar(c)* &
                        (log(self%feature_probability(j, c)) - &
                        log(1.0_dp - self%feature_probability(j, c)))
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_predict_log_proba_vjp

    subroutine bernoulli_nb_predict_proba_vjp(self, x, probabilities_bar, x_bar, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. self%is_fitted .or. size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB probability VJP: model or cotangent shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_log_proba_vjp(x, probabilities_bar*probabilities, x_bar, status)
    end subroutine bernoulli_nb_predict_proba_vjp

    subroutine bernoulli_nb_predict_log_proba_parameter_vjp(self, x, &
            log_probabilities_bar, theta_bar, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), probability_bar(:, :)
        real(dp) :: probabilities(self%n_classes), joint_bar(self%n_classes)
        real(dp) :: prior_bar(self%n_classes), cotangent_sum, q
        integer :: i, j, c, offset

        theta_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB parameter VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(theta_bar) /= self%parameter_count() &
            .or. any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB parameter VJP: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. any(x > 1.0_dp) &
            .or. any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB parameter VJP: inputs are invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes), &
            probability_bar(self%n_features, self%n_classes))
        probability_bar = 0.0_dp
        prior_bar = 0.0_dp
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            cotangent_sum = sum(log_probabilities_bar(i, :))
            do c = 1, self%n_classes
                joint_bar(c) = log_probabilities_bar(i, c) - cotangent_sum*probabilities(c)
                prior_bar(c) = prior_bar(c) + joint_bar(c)/self%prior(c)
                do j = 1, self%n_features
                    q = self%feature_probability(j, c)
                    probability_bar(j, c) = probability_bar(j, c) + joint_bar(c)* &
                        (x(i, j)/q - (1.0_dp - x(i, j))/(1.0_dp - q))
                end do
            end do
        end do
        offset = self%n_features*self%n_classes
        theta_bar(:offset) = reshape(probability_bar, [offset])
        ! The log-softmax pullback has zero class sum, so the projection term
        ! from normalizing the packed prior block vanishes exactly.
        theta_bar(offset + 1:) = prior_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_predict_log_proba_parameter_vjp

    subroutine bernoulli_nb_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. self%is_fitted .or. size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB parameter probability VJP: model or shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_log_proba_parameter_vjp(x, probabilities_bar*probabilities, &
            theta_bar, status)
    end subroutine bernoulli_nb_predict_proba_parameter_vjp

    function bernoulli_nb_classes(self) result(labels)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function bernoulli_nb_classes

    function bernoulli_nb_feature_probabilities(self) result(values)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%feature_probability)) then
            values = self%feature_probability
        else
            allocate(values(0, 0))
        end if
    end function bernoulli_nb_feature_probabilities

    function bernoulli_nb_class_prior(self) result(values)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%prior)) then
            values = self%prior
        else
            allocate(values(0))
        end if
    end function bernoulli_nb_class_prior

    function bernoulli_nb_weighted_counts(self) result(values)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%weighted_count)) then
            values = self%weighted_count
        else
            allocate(values(0))
        end if
    end function bernoulli_nb_weighted_counts

    real(dp) function bernoulli_nb_alpha(self) result(value)
        class(bernoulli_naive_bayes_t), intent(in) :: self

        value = self%alpha
    end function bernoulli_nb_alpha

    integer function bernoulli_nb_feature_count(self) result(count)
        class(bernoulli_naive_bayes_t), intent(in) :: self

        count = self%n_features
    end function bernoulli_nb_feature_count

    integer function bernoulli_nb_class_count(self) result(count)
        class(bernoulli_naive_bayes_t), intent(in) :: self

        count = self%n_classes
    end function bernoulli_nb_class_count

    integer function bernoulli_nb_parameter_count(self) result(count)
        class(bernoulli_naive_bayes_t), intent(in) :: self

        count = self%n_features*self%n_classes + self%n_classes
    end function bernoulli_nb_parameter_count

    function bernoulli_nb_parameters(self) result(values)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: offset

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        offset = self%n_features*self%n_classes
        values(:offset) = reshape(self%feature_probability, [offset])
        values(offset + 1:) = self%prior
    end function bernoulli_nb_parameters

    subroutine bernoulli_nb_set_parameters(self, values, status)
        class(bernoulli_naive_bayes_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: offset
        real(dp) :: prior_sum

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB set_parameters: model is not fitted")
            return
        end if
        if (size(values) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB set_parameters: parameter shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB set_parameters: values must be finite")
            return
        end if
        offset = self%n_features*self%n_classes
        if (any(values(:offset) <= 0.0_dp) .or. any(values(:offset) >= 1.0_dp) &
            .or. any(values(offset + 1:) <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB set_parameters: probabilities must be interior")
            return
        end if
        prior_sum = sum(values(offset + 1:))
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "BernoulliNB set_parameters: prior mass is invalid")
            return
        end if
        self%feature_probability = reshape(values(:offset), shape(self%feature_probability))
        self%prior = values(offset + 1:)/prior_sum
        call status_set(status, FORTNUM_OK, "")
    end subroutine bernoulli_nb_set_parameters

    logical function bernoulli_nb_fitted(self) result(is_fitted)
        class(bernoulli_naive_bayes_t), intent(in) :: self

        is_fitted = self%is_fitted
    end function bernoulli_nb_fitted

    logical function bernoulli_nb_partial_initialized(self) result(value)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        value = self%partial_initialized
    end function bernoulli_nb_partial_initialized

    integer function bernoulli_nb_sample_count(self) result(value)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        value = self%partial_sample_count
    end function bernoulli_nb_sample_count

    integer function bernoulli_nb_batch_count(self) result(value)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        value = self%partial_batch_count
    end function bernoulli_nb_batch_count

    logical function bernoulli_nb_device_supported(self, device_kind) result(value)
        class(bernoulli_naive_bayes_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = device_kind == FORTML_DEVICE_CPU
    end function bernoulli_nb_device_supported

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, n, temporary

        allocate(work, source=labels)
        do i = 2, size(work)
            temporary = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= temporary) exit
                work(j + 1) = work(j)
                j = j - 1
            end do
            work(j + 1) = temporary
        end do
        n = 1
        do i = 2, size(work)
            if (work(i) /= work(n)) then
                n = n + 1
                work(n) = work(i)
            end if
        end do
        allocate(classes(n))
        classes = work(:n)
    end subroutine sorted_unique_labels

end module fortml_bernoulli_naive_bayes
