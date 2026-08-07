module fortml_multinomial_naive_bayes
    !! Weighted, differentiable Multinomial Naive Bayes classification.
    !!
    !! Samples are rows and nonnegative feature counts are columns.  Counts
    !! may be real-valued, which provides a smooth extension useful for input
    !! derivatives.  Classes are sorted in ascending integer order.  The
    !! packed parameter order is smoothed feature probabilities followed by
    !! class priors, each in Fortran column-major order.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    real(dp), parameter :: DEFAULT_ALPHA = 1.0_dp

    type, public :: multinomial_naive_bayes_t
        private
        real(dp), allocatable :: feature_probability(:, :)
        real(dp), allocatable :: feature_mass(:, :)
        real(dp), allocatable :: prior(:)
        real(dp), allocatable :: weighted_count(:)
        integer, allocatable :: class_label(:)
        integer :: n_features = 0
        integer :: n_classes = 0
        real(dp) :: alpha = DEFAULT_ALPHA
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => multinomial_nb_fit
        procedure, public :: predict_log_proba => multinomial_nb_predict_log_proba
        procedure, public :: predict_proba => multinomial_nb_predict_proba
        procedure, public :: predict => multinomial_nb_predict
        procedure, public :: predict_log_proba_jvp => &
            multinomial_nb_predict_log_proba_jvp
        procedure, public :: predict_proba_jvp => multinomial_nb_predict_proba_jvp
        procedure, public :: predict_log_proba_parameter_jvp => &
            multinomial_nb_predict_log_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            multinomial_nb_predict_proba_parameter_jvp
        procedure, public :: predict_log_proba_vjp => multinomial_nb_predict_log_proba_vjp
        procedure, public :: predict_proba_vjp => multinomial_nb_predict_proba_vjp
        procedure, public :: predict_log_proba_parameter_vjp => &
            multinomial_nb_predict_log_proba_parameter_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            multinomial_nb_predict_proba_parameter_vjp
        procedure, public :: classes => multinomial_nb_classes
        procedure, public :: feature_probabilities => &
            multinomial_nb_feature_probabilities
        procedure, public :: feature_counts => multinomial_nb_feature_counts
        procedure, public :: class_prior => multinomial_nb_class_prior
        procedure, public :: weighted_class_counts => multinomial_nb_weighted_counts
        procedure, public :: alpha_value => multinomial_nb_alpha
        procedure, public :: feature_count => multinomial_nb_feature_count
        procedure, public :: class_count => multinomial_nb_class_count
        procedure, public :: parameter_count => multinomial_nb_parameter_count
        procedure, public :: parameters => multinomial_nb_parameters
        procedure, public :: set_parameters => multinomial_nb_set_parameters
        procedure, public :: fitted => multinomial_nb_fitted
    end type multinomial_naive_bayes_t

    public :: multinomial_nb_fit
    public :: multinomial_nb_predict_log_proba
    public :: multinomial_nb_predict_proba
    public :: multinomial_nb_predict

contains

    subroutine multinomial_nb_fit(self, x, labels, status, alpha, priors, &
            sample_weight, class_weight)
        class(multinomial_naive_bayes_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: alpha, priors(:), sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:)
        integer, allocatable :: classes(:)
        real(dp), allocatable :: weights(:), class_factors(:)
        real(dp) :: requested_alpha, total_weight, prior_sum, token_mass
        integer :: i, j, c, n_samples, n_features, n_classes

        self%is_fitted = .false.
        requested_alpha = DEFAULT_ALPHA
        if (present(alpha)) requested_alpha = alpha
        if (.not. ieee_is_finite(requested_alpha) .or. requested_alpha <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB fit: alpha must be finite and strictly positive")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1 .or. size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB fit: features must be finite and nonnegative")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB fit: at least two classes are required")
            return
        end if

        allocate(weights(n_samples), class_factors(n_classes))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MultinomialNB fit: sample-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MultinomialNB fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MultinomialNB fit: class-weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MultinomialNB fit: class weights must be finite and positive")
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
                "MultinomialNB fit: effective weights are not finite")
            return
        end if
        total_weight = sum(weights)
        if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB fit: effective weights must have positive mass")
            return
        end if

        allocate(self%feature_probability(n_features, n_classes), &
            self%feature_mass(n_features, n_classes), self%prior(n_classes), &
            self%weighted_count(n_classes), self%class_label(n_classes))
        self%feature_probability = 0.0_dp
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
                    "MultinomialNB fit: every class needs positive effective weight")
                return
            end if
            do j = 1, n_features
                do i = 1, n_samples
                    if (labels(i) == classes(c)) self%feature_mass(j, c) = &
                        self%feature_mass(j, c) + weights(i)*x(i, j)
                end do
            end do
            token_mass = sum(self%feature_mass(:, c))
            if (.not. ieee_is_finite(token_mass) .or. token_mass < 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MultinomialNB fit: feature count mass is invalid")
                return
            end if
            do j = 1, n_features
                self%feature_probability(j, c) = &
                    (self%feature_mass(j, c) + requested_alpha)/ &
                    (token_mass + requested_alpha*real(n_features, dp))
                if (.not. ieee_is_finite(self%feature_probability(j, c)) .or. &
                    self%feature_probability(j, c) <= 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "MultinomialNB fit: smoothed feature probability is invalid")
                    return
                end if
            end do
        end do

        if (present(priors)) then
            if (size(priors) /= n_classes) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MultinomialNB fit: prior shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(priors)) .or. any(priors <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MultinomialNB fit: priors must be finite and positive")
                return
            end if
            self%prior = priors
        else
            self%prior = self%weighted_count
        end if
        prior_sum = sum(self%prior)
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB fit: prior mass must be finite and positive")
            return
        end if
        self%prior = self%prior/prior_sum
        if (any(.not. ieee_is_finite(self%prior)) .or. any(self%prior <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB fit: normalized priors are invalid")
            return
        end if
        self%n_features = n_features
        self%n_classes = n_classes
        self%alpha = requested_alpha
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine multinomial_nb_fit

    subroutine multinomial_nb_predict_log_proba(self, x, log_probabilities, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_joint(self%n_classes), maximum, normalizer
        integer :: i, j, c

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB log probability: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(log_probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB log probability: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB log probability: inputs must be finite and nonnegative")
            return
        end if
        do i = 1, size(x, 1)
            do c = 1, self%n_classes
                log_joint(c) = log(self%prior(c))
                do j = 1, self%n_features
                    log_joint(c) = log_joint(c) + x(i, j)* &
                        log(self%feature_probability(j, c))
                end do
            end do
            maximum = maxval(log_joint)
            normalizer = sum(exp(log_joint - maximum))
            if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "MultinomialNB log probability: normalization is invalid")
                return
            end if
            log_probabilities(i, :) = log_joint - maximum - log(normalizer)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multinomial_nb_predict_log_proba

    subroutine multinomial_nb_predict_proba(self, x, probabilities, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        call self%predict_log_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(probabilities)
        call status_set(status, FORTNUM_OK, "")
    end subroutine multinomial_nb_predict_proba

    subroutine multinomial_nb_predict(self, x, labels, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :)
        integer :: i, c

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB predict: output shape is invalid")
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
    end subroutine multinomial_nb_predict

    subroutine multinomial_nb_predict_log_proba_jvp(self, x, x_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: joint_dot(self%n_classes), probabilities(self%n_classes)
        real(dp) :: weighted_dot
        integer :: i, j, c

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB input JVP: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB input JVP: tangent must be finite")
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
                        log(self%feature_probability(j, c))
                end do
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multinomial_nb_predict_log_proba_jvp

    subroutine multinomial_nb_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB probability JVP: output shape is invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes), &
            log_dot(size(x, 1), self%n_classes))
        call self%predict_log_proba_jvp(x, x_dot, log_probabilities, log_dot, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(log_probabilities)
        probabilities_dot = probabilities*log_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine multinomial_nb_predict_proba_jvp

    subroutine multinomial_nb_predict_log_proba_parameter_jvp(self, x, theta_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probability_dot(:, :), prior_dot(:)
        real(dp) :: joint_dot(self%n_classes), probabilities(self%n_classes)
        real(dp) :: weighted_dot, prior_tangent_sum
        integer :: i, j, c, offset

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB parameter JVP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. &
            any(shape(log_probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB parameter JVP: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB parameter JVP: inputs and tangents are invalid")
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
                    joint_dot(c) = joint_dot(c) + probability_dot(j, c)*x(i, j)/ &
                        self%feature_probability(j, c)
                end do
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multinomial_nb_predict_log_proba_parameter_jvp

    subroutine multinomial_nb_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB parameter probability JVP: output shape is invalid")
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
    end subroutine multinomial_nb_predict_proba_parameter_jvp

    subroutine multinomial_nb_predict_log_proba_vjp(self, x, log_probabilities_bar, &
            x_bar, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
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
                "MultinomialNB log probability VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. any(shape(x_bar) /= shape(x)) .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB log probability VJP: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB log probability VJP: inputs are invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes))
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            cotangent_sum = sum(log_probabilities_bar(i, :))
            do c = 1, self%n_classes
                joint_bar(c) = log_probabilities_bar(i, c) - &
                    cotangent_sum*probabilities(c)
                do j = 1, self%n_features
                    x_bar(i, j) = x_bar(i, j) + joint_bar(c)* &
                        log(self%feature_probability(j, c))
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multinomial_nb_predict_log_proba_vjp

    subroutine multinomial_nb_predict_proba_vjp(self, x, probabilities_bar, x_bar, &
            status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. self%is_fitted .or. size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB probability VJP: model or cotangent shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_log_proba_vjp(x, probabilities_bar*probabilities, x_bar, &
            status)
    end subroutine multinomial_nb_predict_proba_vjp

    subroutine multinomial_nb_predict_log_proba_parameter_vjp(self, x, &
            log_probabilities_bar, theta_bar, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), probability_bar(:, :)
        real(dp) :: probabilities(self%n_classes), joint_bar(self%n_classes)
        real(dp) :: prior_bar(self%n_classes), cotangent_sum
        integer :: i, j, c, offset

        theta_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB parameter VJP: model is not fitted")
            return
        end if
        if (size(x, 2) /= self%n_features .or. size(theta_bar) /= &
            self%parameter_count() .or. any(shape(log_probabilities_bar) /= &
            [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB parameter VJP: shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(x < 0.0_dp) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB parameter VJP: inputs are invalid")
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
                joint_bar(c) = log_probabilities_bar(i, c) - &
                    cotangent_sum*probabilities(c)
                prior_bar(c) = prior_bar(c) + joint_bar(c)/self%prior(c)
                do j = 1, self%n_features
                    probability_bar(j, c) = probability_bar(j, c) + joint_bar(c)* &
                        x(i, j)/self%feature_probability(j, c)
                end do
            end do
        end do
        offset = self%n_features*self%n_classes
        theta_bar(:offset) = reshape(probability_bar, [offset])
        ! set_parameters normalizes the prior block.  Project its pullback
        ! through that simplex map even when roundoff makes its sum nonzero.
        prior_bar = prior_bar - self%prior*dot_product(prior_bar, self%prior)
        theta_bar(offset + 1:) = prior_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine multinomial_nb_predict_log_proba_parameter_vjp

    subroutine multinomial_nb_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. self%is_fitted .or. size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB parameter probability VJP: model or shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_log_proba_parameter_vjp(x, probabilities_bar*probabilities, &
            theta_bar, status)
    end subroutine multinomial_nb_predict_proba_parameter_vjp

    function multinomial_nb_classes(self) result(labels)
        class(multinomial_naive_bayes_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function multinomial_nb_classes

    function multinomial_nb_feature_probabilities(self) result(values)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%feature_probability)) then
            values = self%feature_probability
        else
            allocate(values(0, 0))
        end if
    end function multinomial_nb_feature_probabilities

    function multinomial_nb_feature_counts(self) result(values)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%feature_mass)) then
            values = self%feature_mass
        else
            allocate(values(0, 0))
        end if
    end function multinomial_nb_feature_counts

    function multinomial_nb_class_prior(self) result(values)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%prior)) then
            values = self%prior
        else
            allocate(values(0))
        end if
    end function multinomial_nb_class_prior

    function multinomial_nb_weighted_counts(self) result(values)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%weighted_count)) then
            values = self%weighted_count
        else
            allocate(values(0))
        end if
    end function multinomial_nb_weighted_counts

    real(dp) function multinomial_nb_alpha(self) result(value)
        class(multinomial_naive_bayes_t), intent(in) :: self

        value = self%alpha
    end function multinomial_nb_alpha

    integer function multinomial_nb_feature_count(self) result(count)
        class(multinomial_naive_bayes_t), intent(in) :: self

        count = self%n_features
    end function multinomial_nb_feature_count

    integer function multinomial_nb_class_count(self) result(count)
        class(multinomial_naive_bayes_t), intent(in) :: self

        count = self%n_classes
    end function multinomial_nb_class_count

    integer function multinomial_nb_parameter_count(self) result(count)
        class(multinomial_naive_bayes_t), intent(in) :: self

        count = self%n_features*self%n_classes + self%n_classes
    end function multinomial_nb_parameter_count

    function multinomial_nb_parameters(self) result(values)
        class(multinomial_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: offset

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        offset = self%n_features*self%n_classes
        values(:offset) = reshape(self%feature_probability, [offset])
        values(offset + 1:) = self%prior
    end function multinomial_nb_parameters

    subroutine multinomial_nb_set_parameters(self, values, status)
        class(multinomial_naive_bayes_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: offset
        real(dp) :: prior_sum

        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB set_parameters: model is not fitted")
            return
        end if
        if (size(values) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB set_parameters: parameter shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB set_parameters: values must be finite")
            return
        end if
        offset = self%n_features*self%n_classes
        if (any(values(:offset) <= 0.0_dp) .or. &
            any(values(offset + 1:) <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB set_parameters: probabilities must be positive")
            return
        end if
        prior_sum = sum(values(offset + 1:))
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "MultinomialNB set_parameters: prior mass is invalid")
            return
        end if
        self%feature_probability = reshape(values(:offset), &
            shape(self%feature_probability))
        self%prior = values(offset + 1:)/prior_sum
        call status_set(status, FORTNUM_OK, "")
    end subroutine multinomial_nb_set_parameters

    logical function multinomial_nb_fitted(self) result(is_fitted)
        class(multinomial_naive_bayes_t), intent(in) :: self

        is_fitted = self%is_fitted
    end function multinomial_nb_fitted

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

end module fortml_multinomial_naive_bayes
