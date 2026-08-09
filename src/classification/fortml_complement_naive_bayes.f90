module fortml_complement_naive_bayes
    !! Weighted, differentiable Complement Naive Bayes classification.
    !!
    !! Samples are rows and nonnegative feature counts are columns.  The
    !! complement distribution for class c is formed from all weighted feature
    !! counts except those belonging to c.  Its smoothed probabilities are
    !! packed before the normalized class prior.  Prediction uses the usual
    !! ComplementNB positive weights (-log complement probability); the prior
    !! is included as a differentiable intercept so that log-probabilities are
    !! normalized and the estimator has the same contract as the other FortML
    !! Naive Bayes estimators.  ``norm=.true.`` applies the second weight
    !! normalization described by Rennie et al. (2003).
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    real(dp), parameter :: DEFAULT_ALPHA = 1.0_dp

    type, public :: complement_naive_bayes_t
        private
        real(dp), allocatable :: complement_probability(:, :)
        real(dp), allocatable :: feature_weight(:, :)
        real(dp), allocatable :: feature_mass(:, :)
        real(dp), allocatable :: prior(:)
        real(dp), allocatable :: weighted_count(:)
        integer, allocatable :: class_label(:)
        integer :: n_features = 0
        integer :: n_classes = 0
        real(dp) :: alpha = DEFAULT_ALPHA
        logical :: norm = .false.
        logical :: is_fitted = .false.
        real(dp), allocatable :: partial_x(:, :), partial_weight(:)
        integer, allocatable :: partial_labels(:), partial_classes(:)
        real(dp) :: partial_alpha = DEFAULT_ALPHA
        integer :: partial_sample_count = 0, partial_batch_count = 0
        logical :: partial_initialized = .false.
    contains
        procedure, public :: fit => complement_nb_fit
        procedure, public :: partial_fit => complement_nb_partial_fit
        procedure, public :: warm_start => complement_nb_partial_fit
        procedure, public :: partial_fit_device => complement_nb_partial_fit_device
        procedure, public :: predict_log_proba => complement_nb_predict_log_proba
        procedure, public :: predict_proba => complement_nb_predict_proba
        procedure, public :: predict => complement_nb_predict
        procedure, public :: predict_log_proba_jvp => complement_nb_predict_log_proba_jvp
        procedure, public :: predict_proba_jvp => complement_nb_predict_proba_jvp
        procedure, public :: predict_log_proba_parameter_jvp => &
            complement_nb_predict_log_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            complement_nb_predict_proba_parameter_jvp
        procedure, public :: predict_log_proba_vjp => complement_nb_predict_log_proba_vjp
        procedure, public :: predict_proba_vjp => complement_nb_predict_proba_vjp
        procedure, public :: predict_log_proba_parameter_vjp => &
            complement_nb_predict_log_proba_parameter_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            complement_nb_predict_proba_parameter_vjp
        procedure, public :: classes => complement_nb_classes
        procedure, public :: feature_probabilities => &
            complement_nb_feature_probabilities
        procedure, public :: feature_weights => complement_nb_feature_weights
        procedure, public :: feature_counts => complement_nb_feature_counts
        procedure, public :: class_prior => complement_nb_class_prior
        procedure, public :: weighted_class_counts => complement_nb_weighted_counts
        procedure, public :: alpha_value => complement_nb_alpha
        procedure, public :: norm_enabled => complement_nb_norm
        procedure, public :: feature_count => complement_nb_feature_count
        procedure, public :: class_count => complement_nb_class_count
        procedure, public :: parameter_count => complement_nb_parameter_count
        procedure, public :: parameters => complement_nb_parameters
        procedure, public :: set_parameters => complement_nb_set_parameters
        procedure, public :: fitted => complement_nb_fitted
        procedure, public :: partial_fit_initialized => complement_nb_partial_initialized
        procedure, public :: sample_count => complement_nb_sample_count
        procedure, public :: batch_count => complement_nb_batch_count
        procedure, public :: device_supported => complement_nb_device_supported
    end type complement_naive_bayes_t

    public :: complement_nb_fit
    public :: complement_nb_partial_fit
    public :: complement_nb_partial_fit_device
    public :: complement_nb_predict_log_proba
    public :: complement_nb_predict_proba
    public :: complement_nb_predict

contains

    subroutine complement_nb_fit(self, x, labels, status, alpha, priors, &
            sample_weight, class_weight, norm)
        class(complement_naive_bayes_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha, priors(:), sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:)
        logical, intent(in), optional :: norm
        integer, allocatable :: classes(:)
        real(dp), allocatable :: weights(:), class_factors(:)
        real(dp) :: requested_alpha, total_weight, prior_sum
        real(dp) :: complement_total, complement_mass
        logical :: requested_norm, valid_weights
        integer :: i, j, c, n_samples, n_features, n_classes

        self%is_fitted = .false.
        requested_alpha = DEFAULT_ALPHA
        if (present(alpha)) requested_alpha = alpha
        requested_norm = .false.
        if (present(norm)) requested_norm = norm
        if (.not. ieee_is_finite(requested_alpha) .or. requested_alpha <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: alpha must be finite and strictly positive")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: input must have at least one sample and feature")
            return
        end if
        if (size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: label shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: features must be finite and nonnegative")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: at least two classes are required")
            return
        end if

        allocate(weights(n_samples), class_factors(n_classes))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB fit: sample-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB fit: class-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB fit: class weights must be finite and positive")
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
                "ComplementNB fit: effective weights are not finite")
            return
        end if
        total_weight = sum(weights)
        if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: effective weights must have positive mass")
            return
        end if

        allocate(self%feature_mass(n_features, n_classes), &
            self%complement_probability(n_features, n_classes), &
            self%feature_weight(n_features, n_classes), self%prior(n_classes), &
            self%weighted_count(n_classes), self%class_label(n_classes))
        self%feature_mass = 0.0_dp
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
                    "ComplementNB fit: every class needs positive effective weight")
                return
            end if
            do j = 1, n_features
                do i = 1, n_samples
                    if (labels(i) == classes(c)) self%feature_mass(j, c) = &
                        self%feature_mass(j, c) + weights(i)*x(i, j)
                end do
            end do
        end do
        if (any(.not. ieee_is_finite(self%feature_mass))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: feature count mass is not finite")
            return
        end if

        do c = 1, n_classes
            complement_total = 0.0_dp
            do j = 1, n_features
                complement_mass = sum(self%feature_mass(j, :)) - &
                    self%feature_mass(j, c)
                if (.not. ieee_is_finite(complement_mass)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "ComplementNB fit: complement feature mass is not finite")
                    return
                end if
                ! Nonnegative input masses can only produce a tiny negative
                ! value here through roundoff; clamp that harmless residual.
                complement_mass = max(0.0_dp, complement_mass)
                complement_total = complement_total + complement_mass
                self%complement_probability(j, c) = complement_mass + requested_alpha
            end do
            complement_total = complement_total + requested_alpha*real(n_features, dp)
            if (.not. ieee_is_finite(complement_total) .or. &
                complement_total <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB fit: smoothed complement mass is invalid")
                return
            end if
            self%complement_probability(:, c) = &
                self%complement_probability(:, c)/complement_total
        end do
        call build_feature_weights(self%complement_probability, requested_norm, &
            self%feature_weight, valid_weights)
        if (.not. valid_weights) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: complement weights are invalid")
            return
        end if

        if (present(priors)) then
            if (size(priors) /= n_classes) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB fit: prior shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(priors)) .or. any(priors <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB fit: priors must be finite and positive")
                return
            end if
            self%prior = priors
        else
            self%prior = self%weighted_count
        end if
        prior_sum = sum(self%prior)
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: prior mass must be finite and positive")
            return
        end if
        self%prior = self%prior/prior_sum
        if (any(.not. ieee_is_finite(self%prior)) .or. any(self%prior <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB fit: normalized priors are invalid")
            return
        end if
        self%n_features = n_features
        self%n_classes = n_classes
        self%alpha = requested_alpha
        self%norm = requested_norm
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_fit

    subroutine complement_nb_partial_fit(self, x, labels, status, classes, alpha, &
            sample_weight, norm)
        !! Transactionally append a nonnegative-count ComplementNB batch.
        class(complement_naive_bayes_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: classes(:)
        real(dp), intent(in), optional :: alpha, sample_weight(:)
        logical, intent(in), optional :: norm
        type(complement_naive_bayes_t) :: candidate
        type(fortnum_status_t) :: candidate_status
        integer, allocatable :: candidate_classes(:), candidate_labels(:)
        real(dp), allocatable :: candidate_x(:, :), candidate_weight(:), batch_weight(:)
        real(dp) :: requested_alpha
        logical :: requested_norm, first_call, complete
        integer :: n_samples, n_features, n_old, n_classes, i, c

        n_samples = size(x, 1); n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1 .or. size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB partial_fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB partial_fit: features must be finite and nonnegative")
            return
        end if
        allocate(batch_weight(n_samples)); batch_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB partial_fit: sample weights are invalid")
                return
            end if
            batch_weight = sample_weight
        end if
        if (.not. ieee_is_finite(sum(batch_weight)) .or. sum(batch_weight) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB partial_fit: batch weight mass must be positive")
            return
        end if
        requested_alpha = DEFAULT_ALPHA
        if (present(alpha)) requested_alpha = alpha
        requested_norm = .false.
        if (present(norm)) requested_norm = norm
        first_call = .not. self%partial_initialized
        if (first_call) then
            if (self%is_fitted .or. .not. present(classes)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB partial_fit: fresh model and classes are required")
                return
            end if
            if (size(classes) < 2) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB partial_fit: at least two classes are required")
                return
            end if
            do i = 2, size(classes)
                if (classes(i) <= classes(i - 1)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "ComplementNB partial_fit: classes must be strictly increasing")
                    return
                end if
            end do
            if (.not. ieee_is_finite(requested_alpha) .or. requested_alpha <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB partial_fit: alpha must be finite and positive")
                return
            end if
            allocate(candidate_classes(size(classes))); candidate_classes = classes
        else
            if (n_features /= self%n_features) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB partial_fit: feature count changed")
                return
            end if
            if (present(classes)) then
                if (size(classes) /= size(self%partial_classes) .or. &
                    any(classes /= self%partial_classes)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "ComplementNB partial_fit: class vocabulary changed")
                    return
                end if
            end if
            requested_alpha = self%partial_alpha
            if (present(alpha)) then
                if (.not. ieee_is_finite(alpha) .or. alpha /= self%partial_alpha) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "ComplementNB partial_fit: alpha cannot change")
                    return
                end if
            end if
            requested_norm = self%norm
            if (present(norm)) then
                if (norm .neqv. self%norm) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "ComplementNB partial_fit: norm cannot change")
                    return
                end if
            end if
            allocate(candidate_classes(size(self%partial_classes)))
            candidate_classes = self%partial_classes
        end if
        if (.not. ieee_is_finite(requested_alpha) .or. requested_alpha <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB partial_fit: alpha must be finite and positive")
            return
        end if
        n_classes = size(candidate_classes)
        do i = 1, n_samples
            if (.not. any(labels(i) == candidate_classes)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB partial_fit: batch contains an unseen class")
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
                alpha=requested_alpha, sample_weight=candidate_weight, norm=requested_norm)
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
            self%complement_probability = candidate%complement_probability
            self%feature_weight = candidate%feature_weight
            self%feature_mass = candidate%feature_mass
            self%prior = candidate%prior
            self%weighted_count = candidate%weighted_count
            self%alpha = candidate%alpha; self%norm = candidate%norm
            self%is_fitted = .true.
        else
            self%is_fitted = .false.
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_partial_fit

    subroutine complement_nb_partial_fit_device(self, device, x, labels, status, &
            classes, alpha, sample_weight, norm)
        class(complement_naive_bayes_t), intent(inout) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: classes(:)
        real(dp), intent(in), optional :: alpha, sample_weight(:)
        logical, intent(in), optional :: norm

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB partial_fit device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%partial_fit(x, labels, status, classes, alpha, sample_weight, norm)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "ComplementNB partial_fit device: resident CUDA sufficient statistics are not implemented")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB partial_fit device: device kind is invalid")
        end select
    end subroutine complement_nb_partial_fit_device

    subroutine complement_nb_predict_log_proba(self, x, log_probabilities, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_joint(self%n_classes), maximum, normalizer
        integer :: i, c

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB log probability: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB log probability: input feature shape is invalid")
            return
        end if
        if (any(shape(log_probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB log probability: output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB log probability: inputs must be finite and nonnegative")
            return
        end if
        do i = 1, size(x, 1)
            do c = 1, self%n_classes
                log_joint(c) = log(self%prior(c)) + &
                    dot_product(x(i, :), self%feature_weight(:, c))
            end do
            maximum = maxval(log_joint)
            normalizer = sum(exp(log_joint - maximum))
            if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "ComplementNB log probability: normalization is invalid")
                return
            end if
            log_probabilities(i, :) = log_joint - maximum - log(normalizer)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_predict_log_proba

    subroutine complement_nb_predict_proba(self, x, probabilities, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        call self%predict_log_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(probabilities)
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_predict_proba

    subroutine complement_nb_predict(self, x, labels, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :)
        integer :: i, c

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB predict: output shape is invalid")
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
    end subroutine complement_nb_predict

    subroutine complement_nb_predict_log_proba_jvp(self, x, x_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: joint_dot(self%n_classes), probabilities(self%n_classes)
        real(dp) :: weighted_dot
        integer :: i, c

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB input JVP: tangent shape is invalid")
            return
        end if
        if (any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB input JVP: output tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB input JVP: tangent must be finite")
            return
        end if
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            do c = 1, self%n_classes
                joint_dot(c) = dot_product(x_dot(i, :), self%feature_weight(:, c))
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_predict_log_proba_jvp

    subroutine complement_nb_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB probability JVP: output shape is invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes), &
            log_dot(size(x, 1), self%n_classes))
        call self%predict_log_proba_jvp(x, x_dot, log_probabilities, log_dot, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(log_probabilities)
        probabilities_dot = probabilities*log_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_predict_proba_jvp

    subroutine complement_nb_predict_log_proba_parameter_jvp(self, x, theta_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probability_dot(:, :), weight_dot(:, :)
        real(dp), allocatable :: prior_dot(:)
        real(dp) :: joint_dot(self%n_classes), probabilities(self%n_classes)
        real(dp) :: weighted_dot, prior_tangent_sum, log_sum, log_dot_sum
        integer :: i, j, c, offset

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter JVP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter JVP: input shape is invalid")
            return
        end if
        if (any(shape(log_probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter JVP: output shape is invalid")
            return
        end if
        if (size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter JVP: tangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter JVP: inputs and tangents are invalid")
            return
        end if
        offset = self%n_features*self%n_classes
        allocate(probability_dot(self%n_features, self%n_classes), &
            weight_dot(self%n_features, self%n_classes), prior_dot(self%n_classes))
        probability_dot = reshape(theta_dot(:offset), shape(probability_dot))
        prior_dot = theta_dot(offset + 1:)
        prior_tangent_sum = sum(prior_dot)
        if (self%norm) then
            do c = 1, self%n_classes
                log_sum = sum(log(self%complement_probability(:, c)))
                log_dot_sum = sum(probability_dot(:, c)/ &
                    self%complement_probability(:, c))
                do j = 1, self%n_features
                    weight_dot(j, c) = (probability_dot(j, c)/ &
                        self%complement_probability(j, c)*log_sum - &
                        log(self%complement_probability(j, c))*log_dot_sum)/ &
                        (log_sum*log_sum)
                end do
            end do
        else
            weight_dot = -probability_dot/self%complement_probability
        end if
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            do c = 1, self%n_classes
                joint_dot(c) = prior_dot(c)/self%prior(c) - prior_tangent_sum + &
                    dot_product(x(i, :), weight_dot(:, c))
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_predict_log_proba_parameter_jvp

    subroutine complement_nb_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter probability JVP: output shape is invalid")
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
    end subroutine complement_nb_predict_proba_parameter_jvp

    subroutine complement_nb_predict_log_proba_vjp(self, x, log_probabilities_bar, &
            x_bar, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :)
        real(dp) :: probabilities(self%n_classes), joint_bar(self%n_classes)
        real(dp) :: cotangent_sum
        integer :: i, c

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB log probability VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB log probability VJP: input shape is invalid")
            return
        end if
        if (any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB log probability VJP: cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB log probability VJP: inputs are invalid")
            return
        end if
        x_bar = 0.0_dp
        allocate(log_probabilities(size(x, 1), self%n_classes))
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            cotangent_sum = sum(log_probabilities_bar(i, :))
            do c = 1, self%n_classes
                joint_bar(c) = log_probabilities_bar(i, c) - &
                    cotangent_sum*probabilities(c)
                x_bar(i, :) = x_bar(i, :) + joint_bar(c)*self%feature_weight(:, c)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_predict_log_proba_vjp

    subroutine complement_nb_predict_proba_vjp(self, x, probabilities_bar, x_bar, &
            status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB probability VJP: model is not fitted")
            return
        end if
        if (size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB probability VJP: model or cotangent shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_log_proba_vjp(x, probabilities_bar*probabilities, x_bar, &
            status)
    end subroutine complement_nb_predict_proba_vjp

    subroutine complement_nb_predict_log_proba_parameter_vjp(self, x, &
            log_probabilities_bar, theta_bar, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), weight_bar(:, :)
        real(dp), allocatable :: probability_bar(:,:), prior_bar(:)
        real(dp) :: probabilities(self%n_classes), joint_bar(self%n_classes)
        real(dp) :: cotangent_sum, log_sum, log_weight_bar_sum
        integer :: i, j, c, offset

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            size(theta_bar) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter VJP: input or cotangent shape is invalid")
            return
        end if
        if (any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter VJP: output cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter VJP: inputs are invalid")
            return
        end if
        theta_bar = 0.0_dp
        allocate(log_probabilities(size(x, 1), self%n_classes), &
            weight_bar(self%n_features, self%n_classes), &
            probability_bar(self%n_features, self%n_classes), prior_bar(self%n_classes))
        weight_bar = 0.0_dp
        prior_bar = 0.0_dp
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            cotangent_sum = sum(log_probabilities_bar(i, :))
            do c = 1, self%n_classes
                joint_bar(c) = log_probabilities_bar(i, c) - &
                    cotangent_sum*probabilities(c)
                prior_bar(c) = prior_bar(c) + joint_bar(c)/self%prior(c)
                do j = 1, self%n_features
                    weight_bar(j, c) = weight_bar(j, c) + joint_bar(c)*x(i, j)
                end do
            end do
        end do
        if (self%norm) then
            do c = 1, self%n_classes
                log_sum = sum(log(self%complement_probability(:, c)))
                log_weight_bar_sum = dot_product(weight_bar(:, c), &
                    log(self%complement_probability(:, c)))
                do j = 1, self%n_features
                    probability_bar(j, c) = (weight_bar(j, c)/log_sum - &
                        log_weight_bar_sum/(log_sum*log_sum))/ &
                        self%complement_probability(j, c)
                end do
            end do
        else
            probability_bar = -weight_bar/self%complement_probability
        end if
        offset = self%n_features*self%n_classes
        theta_bar(:offset) = reshape(probability_bar, [offset])
        prior_bar = prior_bar - self%prior*dot_product(prior_bar, self%prior)
        theta_bar(offset + 1:) = prior_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_predict_log_proba_parameter_vjp

    subroutine complement_nb_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter probability VJP: model is not fitted")
            return
        end if
        if (size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB parameter probability VJP: model or shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_log_proba_parameter_vjp(x, probabilities_bar*probabilities, &
            theta_bar, status)
    end subroutine complement_nb_predict_proba_parameter_vjp

    function complement_nb_classes(self) result(labels)
        class(complement_naive_bayes_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function complement_nb_classes

    function complement_nb_feature_probabilities(self) result(values)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%complement_probability)) then
            values = self%complement_probability
        else
            allocate(values(0, 0))
        end if
    end function complement_nb_feature_probabilities

    function complement_nb_feature_weights(self) result(values)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%feature_weight)) then
            values = self%feature_weight
        else
            allocate(values(0, 0))
        end if
    end function complement_nb_feature_weights

    function complement_nb_feature_counts(self) result(values)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%feature_mass)) then
            values = self%feature_mass
        else
            allocate(values(0, 0))
        end if
    end function complement_nb_feature_counts

    function complement_nb_class_prior(self) result(values)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%prior)) then
            values = self%prior
        else
            allocate(values(0))
        end if
    end function complement_nb_class_prior

    function complement_nb_weighted_counts(self) result(values)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%weighted_count)) then
            values = self%weighted_count
        else
            allocate(values(0))
        end if
    end function complement_nb_weighted_counts

    real(dp) function complement_nb_alpha(self) result(value)
        class(complement_naive_bayes_t), intent(in) :: self

        value = self%alpha
    end function complement_nb_alpha

    logical function complement_nb_norm(self) result(value)
        class(complement_naive_bayes_t), intent(in) :: self

        value = self%norm
    end function complement_nb_norm

    integer function complement_nb_feature_count(self) result(count)
        class(complement_naive_bayes_t), intent(in) :: self

        count = self%n_features
    end function complement_nb_feature_count

    integer function complement_nb_class_count(self) result(count)
        class(complement_naive_bayes_t), intent(in) :: self

        count = self%n_classes
    end function complement_nb_class_count

    integer function complement_nb_parameter_count(self) result(count)
        class(complement_naive_bayes_t), intent(in) :: self

        count = self%n_features*self%n_classes + self%n_classes
    end function complement_nb_parameter_count

    function complement_nb_parameters(self) result(values)
        class(complement_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: offset

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        offset = self%n_features*self%n_classes
        values(:offset) = reshape(self%complement_probability, [offset])
        values(offset + 1:) = self%prior
    end function complement_nb_parameters

    subroutine complement_nb_set_parameters(self, values, status)
        class(complement_naive_bayes_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate_probability(:, :), candidate_weight(:, :)
        real(dp), allocatable :: candidate_prior(:)
        real(dp) :: prior_sum
        logical :: valid_weights
        integer :: offset

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB set_parameters: model is not fitted")
            return
        end if
        if (size(values) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB set_parameters: parameter shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB set_parameters: values must be finite")
            return
        end if
        offset = self%n_features*self%n_classes
        if (any(values(:offset) <= 0.0_dp) .or. any(values(offset + 1:) <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB set_parameters: probabilities must be positive")
            return
        end if
        allocate(candidate_probability(self%n_features, self%n_classes), &
            candidate_weight(self%n_features, self%n_classes), &
            candidate_prior(self%n_classes))
        candidate_probability = reshape(values(:offset), shape(candidate_probability))
        call build_feature_weights(candidate_probability, self%norm, candidate_weight, &
            valid_weights)
        if (.not. valid_weights) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB set_parameters: complement weights are invalid")
            return
        end if
        candidate_prior = values(offset + 1:)
        prior_sum = sum(candidate_prior)
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ComplementNB set_parameters: prior mass is invalid")
            return
        end if
        self%complement_probability = candidate_probability
        self%feature_weight = candidate_weight
        self%prior = candidate_prior/prior_sum
        call status_set(status, FORTNUM_OK, "")
    end subroutine complement_nb_set_parameters

    logical function complement_nb_fitted(self) result(is_fitted)
        class(complement_naive_bayes_t), intent(in) :: self

        is_fitted = self%is_fitted
    end function complement_nb_fitted

    logical function complement_nb_partial_initialized(self) result(value)
        class(complement_naive_bayes_t), intent(in) :: self
        value = self%partial_initialized
    end function complement_nb_partial_initialized

    integer function complement_nb_sample_count(self) result(value)
        class(complement_naive_bayes_t), intent(in) :: self
        value = self%partial_sample_count
    end function complement_nb_sample_count

    integer function complement_nb_batch_count(self) result(value)
        class(complement_naive_bayes_t), intent(in) :: self
        value = self%partial_batch_count
    end function complement_nb_batch_count

    logical function complement_nb_device_supported(self, device_kind) result(value)
        class(complement_naive_bayes_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = device_kind == FORTML_DEVICE_CPU
    end function complement_nb_device_supported

    subroutine build_feature_weights(probability, norm, weight, valid)
        real(dp), intent(in) :: probability(:, :)
        logical, intent(in) :: norm
        real(dp), intent(out) :: weight(:, :)
        logical, intent(out) :: valid
        real(dp) :: log_sum
        integer :: j, c

        valid = .false.
        if (any(shape(weight) /= shape(probability))) return
        if (size(probability, 1) < 1 .or. size(probability, 2) < 1) return
        if (any(.not. ieee_is_finite(probability)) .or. any(probability <= 0.0_dp)) then
            return
        end if
        do c = 1, size(probability, 2)
            log_sum = sum(log(probability(:, c)))
            if (.not. ieee_is_finite(log_sum)) return
            if (norm) then
                if (abs(log_sum) <= tiny(1.0_dp)) return
                do j = 1, size(probability, 1)
                    weight(j, c) = log(probability(j, c))/log_sum
                end do
            else
                do j = 1, size(probability, 1)
                    weight(j, c) = -log(probability(j, c))
                end do
            end if
        end do
        if (any(.not. ieee_is_finite(weight))) return
        valid = .true.
    end subroutine build_feature_weights

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

end module fortml_complement_naive_bayes
