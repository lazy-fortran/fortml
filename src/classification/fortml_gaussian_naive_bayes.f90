module fortml_gaussian_naive_bayes
    !! Deterministic, weighted Gaussian Naive Bayes classification.
    !!
    !! The fitted state follows the usual GaussianNB parameterization: every
    !! class owns a weighted feature mean and a strictly positive feature
    !! variance, while class priors are stored on the probability simplex.
    !! Samples are rows and features are columns.  Classes are sorted in
    !! ascending integer order, so prediction columns are stable for arbitrary
    !! integer labels.  The packed parameter order is means, variances, and
    !! priors, each in Fortran column-major order.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    real(dp), parameter :: DEFAULT_VAR_SMOOTHING = 1.0e-9_dp
    real(dp), parameter :: LOG_TWO_PI = 1.837877066409345483560659472811_dp

    type, public :: gaussian_naive_bayes_t
        private
        real(dp), allocatable :: mean(:, :)
        real(dp), allocatable :: variance(:, :)
        real(dp), allocatable :: prior(:)
        real(dp), allocatable :: weighted_count(:)
        integer, allocatable :: class_label(:)
        integer :: n_features = 0
        integer :: n_classes = 0
        real(dp) :: var_smoothing = DEFAULT_VAR_SMOOTHING
        real(dp) :: epsilon = 0.0_dp
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => gaussian_nb_fit
        procedure, public :: predict_log_proba => gaussian_nb_predict_log_proba
        procedure, public :: predict_proba => gaussian_nb_predict_proba
        procedure, public :: predict => gaussian_nb_predict
        procedure, public :: predict_log_proba_jvp => gaussian_nb_predict_log_proba_jvp
        procedure, public :: predict_proba_jvp => gaussian_nb_predict_proba_jvp
        procedure, public :: predict_log_proba_parameter_jvp => &
            gaussian_nb_predict_log_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            gaussian_nb_predict_proba_parameter_jvp
        procedure, public :: predict_log_proba_vjp => gaussian_nb_predict_log_proba_vjp
        procedure, public :: predict_proba_vjp => gaussian_nb_predict_proba_vjp
        procedure, public :: predict_log_proba_parameter_vjp => &
            gaussian_nb_predict_log_proba_parameter_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            gaussian_nb_predict_proba_parameter_vjp
        procedure, public :: classes => gaussian_nb_classes
        procedure, public :: means => gaussian_nb_means
        procedure, public :: variances => gaussian_nb_variances
        procedure, public :: class_prior => gaussian_nb_class_prior
        procedure, public :: weighted_class_counts => gaussian_nb_weighted_counts
        procedure, public :: var_smoothing_value => gaussian_nb_var_smoothing
        procedure, public :: epsilon_value => gaussian_nb_epsilon
        procedure, public :: feature_count => gaussian_nb_feature_count
        procedure, public :: class_count => gaussian_nb_class_count
        procedure, public :: parameter_count => gaussian_nb_parameter_count
        procedure, public :: parameters => gaussian_nb_parameters
        procedure, public :: set_parameters => gaussian_nb_set_parameters
        procedure, public :: fitted => gaussian_nb_fitted
    end type gaussian_naive_bayes_t

    public :: gaussian_nb_fit
    public :: gaussian_nb_predict_log_proba
    public :: gaussian_nb_predict_proba
    public :: gaussian_nb_predict

contains

    subroutine gaussian_nb_fit(self, x, labels, status, var_smoothing, priors, &
            sample_weight, class_weight)
        class(gaussian_naive_bayes_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: var_smoothing, priors(:), sample_weight(:)
        real(dp), intent(in), optional :: class_weight(:)
        integer, allocatable :: classes(:)
        real(dp), allocatable :: weights(:), class_factors(:)
        real(dp) :: requested_smoothing, weight_sum, epsilon, global_mean
        real(dp) :: global_variance, mass, prior_sum, centered
        integer :: i, j, c, n_samples, n_features, n_classes

        self%is_fitted = .false.
        requested_smoothing = DEFAULT_VAR_SMOOTHING
        if (present(var_smoothing)) requested_smoothing = var_smoothing
        if (.not. ieee_is_finite(requested_smoothing) .or. &
            requested_smoothing < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB fit: var_smoothing must be finite and nonnegative")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1 .or. size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB fit: inputs must be finite")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB fit: at least two classes are required")
            return
        end if

        allocate(weights(n_samples), class_factors(n_classes))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GaussianNB fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        class_factors = 1.0_dp
        if (present(class_weight)) then
            if (size(class_weight) /= n_classes .or. &
                any(.not. ieee_is_finite(class_weight)) .or. &
                any(class_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GaussianNB fit: class weights must be finite and positive "// &
                    "in sorted class order")
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
                "GaussianNB fit: effective weights are not finite")
            return
        end if
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB fit: effective weights must have positive mass")
            return
        end if

        allocate(self%mean(n_features, n_classes), &
            self%variance(n_features, n_classes), self%prior(n_classes), &
            self%weighted_count(n_classes), self%class_label(n_classes))
        self%mean = 0.0_dp
        self%variance = 0.0_dp
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
                    "GaussianNB fit: every class needs positive effective weight")
                return
            end if
            do j = 1, n_features
                do i = 1, n_samples
                    if (labels(i) == classes(c)) self%mean(j, c) = &
                        self%mean(j, c) + weights(i)*x(i, j)
                end do
                self%mean(j, c) = self%mean(j, c)/self%weighted_count(c)
            end do
        end do

        epsilon = 0.0_dp
        do j = 1, n_features
            global_mean = sum(weights*x(:, j))/weight_sum
            global_variance = 0.0_dp
            do i = 1, n_samples
                centered = x(i, j) - global_mean
                global_variance = global_variance + weights(i)*centered*centered
            end do
            global_variance = global_variance/weight_sum
            if (.not. ieee_is_finite(global_variance)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GaussianNB fit: weighted variance is not finite")
                return
            end if
            epsilon = max(epsilon, requested_smoothing*global_variance)
        end do
        if (.not. ieee_is_finite(epsilon)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB fit: variance smoothing overflowed")
            return
        end if
        ! A Gaussian density requires a positive variance even for a constant
        ! feature.  The tiny floor only affects this degenerate case.
        epsilon = max(epsilon, tiny(1.0_dp))
        do c = 1, n_classes
            do j = 1, n_features
                do i = 1, n_samples
                    if (labels(i) == classes(c)) then
                        centered = x(i, j) - self%mean(j, c)
                        self%variance(j, c) = self%variance(j, c) + &
                            weights(i)*centered*centered
                    end if
                end do
                self%variance(j, c) = self%variance(j, c)/ &
                    self%weighted_count(c) + epsilon
                if (.not. ieee_is_finite(self%variance(j, c)) .or. &
                    self%variance(j, c) <= 0.0_dp) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "GaussianNB fit: fitted variance is invalid")
                    return
                end if
            end do
        end do

        if (present(priors)) then
            if (size(priors) /= n_classes .or. any(.not. ieee_is_finite(priors)) &
                .or. any(priors <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GaussianNB fit: priors must be finite and positive")
                return
            end if
            self%prior = priors
        else
            self%prior = self%weighted_count
        end if
        prior_sum = sum(self%prior)
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB fit: priors must have positive finite mass")
            return
        end if
        self%prior = self%prior/prior_sum
        self%n_features = n_features
        self%n_classes = n_classes
        self%var_smoothing = requested_smoothing
        self%epsilon = epsilon
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nb_fit

    subroutine gaussian_nb_predict_log_proba(self, x, log_probabilities, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_joint(self%n_classes), maximum, normalizer
        integer :: i, j, c

        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(log_probabilities) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB log probability: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB log probability: inputs must be finite")
            return
        end if
        do i = 1, size(x, 1)
            do c = 1, self%n_classes
                log_joint(c) = log(self%prior(c))
                do j = 1, self%n_features
                    log_joint(c) = log_joint(c) - 0.5_dp*(LOG_TWO_PI + &
                        log(self%variance(j, c)) + (x(i, j) - self%mean(j, c))**2/ &
                        self%variance(j, c))
                end do
            end do
            maximum = maxval(log_joint)
            normalizer = sum(exp(log_joint - maximum))
            if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "GaussianNB log probability: density normalization is invalid")
                return
            end if
            log_probabilities(i, :) = log_joint - maximum - log(normalizer)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nb_predict_log_proba

    subroutine gaussian_nb_predict_proba(self, x, probabilities, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        call self%predict_log_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(probabilities)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nb_predict_proba

    subroutine gaussian_nb_predict(self, x, labels, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :)
        integer :: i, c

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB predict: output shape is invalid")
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
    end subroutine gaussian_nb_predict

    subroutine gaussian_nb_predict_log_proba_jvp(self, x, x_dot, log_probabilities, &
            log_probabilities_dot, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: joint_dot(self%n_classes), probabilities(self%n_classes)
        real(dp) :: weighted_dot
        integer :: i, j, c

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities)) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB log probability JVP: tangent or output shape is invalid")
            return
        end if
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            do c = 1, self%n_classes
                joint_dot(c) = 0.0_dp
                do j = 1, self%n_features
                    joint_dot(c) = joint_dot(c) - (x(i, j) - self%mean(j, c))/ &
                        self%variance(j, c)*x_dot(i, j)
                end do
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nb_predict_log_proba_jvp

    subroutine gaussian_nb_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB probability JVP: output shape is invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes), &
            log_dot(size(x, 1), self%n_classes))
        call self%predict_log_proba_jvp(x, x_dot, log_probabilities, log_dot, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(log_probabilities)
        probabilities_dot = probabilities*log_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nb_predict_proba_jvp

    subroutine gaussian_nb_predict_log_proba_parameter_jvp(self, x, theta_dot, &
            log_probabilities, log_probabilities_dot, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean_dot(:, :), variance_dot(:, :), prior_dot(:)
        real(dp) :: joint_dot(self%n_classes), probabilities(self%n_classes)
        real(dp) :: weighted_dot, delta, prior_tangent_sum
        integer :: i, j, c, offset

        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(log_probabilities) /= [size(x, 1), self%n_classes]) .or. &
            any(shape(log_probabilities_dot) /= shape(log_probabilities)) .or. &
            size(theta_dot) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB parameter JVP: model, parameter, or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB parameter JVP: inputs and tangents must be finite")
            return
        end if
        offset = self%n_features*self%n_classes
        allocate(mean_dot(self%n_features, self%n_classes), &
            variance_dot(self%n_features, self%n_classes), prior_dot(self%n_classes))
        mean_dot = reshape(theta_dot(:offset), shape(mean_dot))
        variance_dot = reshape(theta_dot(offset + 1:2*offset), shape(variance_dot))
        prior_dot = theta_dot(2*offset + 1:)
        prior_tangent_sum = sum(prior_dot)
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            do c = 1, self%n_classes
                joint_dot(c) = prior_dot(c)/self%prior(c) - prior_tangent_sum
                do j = 1, self%n_features
                    delta = x(i, j) - self%mean(j, c)
                    joint_dot(c) = joint_dot(c) + delta/self%variance(j, c)* &
                        mean_dot(j, c) + (-0.5_dp/self%variance(j, c) + &
                        0.5_dp*delta**2/self%variance(j, c)**2)*variance_dot(j, c)
                end do
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nb_predict_log_proba_parameter_jvp

    subroutine gaussian_nb_predict_proba_parameter_jvp(self, x, theta_dot, &
            probabilities, probabilities_dot, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB parameter probability JVP: output shape is invalid")
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
    end subroutine gaussian_nb_predict_proba_parameter_jvp

    subroutine gaussian_nb_predict_log_proba_vjp(self, x, log_probabilities_bar, &
            x_bar, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :)
        real(dp) :: probabilities(self%n_classes), joint_dot(self%n_classes)
        real(dp) :: centered, cotangent_sum
        integer :: i, j, c

        x_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB log probability VJP: model or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB log probability VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes))
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            cotangent_sum = sum(log_probabilities_bar(i, :))
            do c = 1, self%n_classes
                joint_dot(c) = log_probabilities_bar(i, c) - &
                    cotangent_sum*probabilities(c)
                do j = 1, self%n_features
                    centered = x(i, j) - self%mean(j, c)
                    x_bar(i, j) = x_bar(i, j) - joint_dot(c)*centered/ &
                        self%variance(j, c)
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nb_predict_log_proba_vjp

    subroutine gaussian_nb_predict_proba_vjp(self, x, probabilities_bar, x_bar, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB probability VJP: cotangent shape or values are invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_log_proba_vjp(x, probabilities_bar*probabilities, x_bar, status)
    end subroutine gaussian_nb_predict_proba_vjp

    subroutine gaussian_nb_predict_log_proba_parameter_vjp(self, x, &
            log_probabilities_bar, theta_bar, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), mean_bar(:, :), variance_bar(:, :)
        real(dp), allocatable :: prior_bar(:)
        real(dp) :: probabilities(self%n_classes), cotangent_sum, alpha, centered
        integer :: i, j, c, offset

        theta_bar = 0.0_dp
        if (.not. self%is_fitted .or. size(x, 2) /= self%n_features .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), self%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB parameter VJP: model, parameter, or cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB parameter VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(log_probabilities(size(x, 1), self%n_classes), &
            mean_bar(self%n_features, self%n_classes), &
            variance_bar(self%n_features, self%n_classes), prior_bar(self%n_classes))
        mean_bar = 0.0_dp
        variance_bar = 0.0_dp
        prior_bar = 0.0_dp
        call self%predict_log_proba(x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            cotangent_sum = sum(log_probabilities_bar(i, :))
            do c = 1, self%n_classes
                alpha = log_probabilities_bar(i, c) - cotangent_sum*probabilities(c)
                prior_bar(c) = prior_bar(c) + alpha/self%prior(c)
                do j = 1, self%n_features
                    centered = x(i, j) - self%mean(j, c)
                    mean_bar(j, c) = mean_bar(j, c) + alpha*centered/ &
                        self%variance(j, c)
                    variance_bar(j, c) = variance_bar(j, c) + alpha*( &
                        -0.5_dp/self%variance(j, c) + &
                        0.5_dp*centered**2/self%variance(j, c)**2)
                end do
            end do
        end do
        offset = self%n_features*self%n_classes
        theta_bar(:offset) = reshape(mean_bar, [offset])
        theta_bar(offset + 1:2*offset) = reshape(variance_bar, [offset])
        prior_bar = prior_bar - sum(prior_bar)
        theta_bar(2*offset + 1:) = prior_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nb_predict_log_proba_parameter_vjp

    subroutine gaussian_nb_predict_proba_parameter_vjp(self, x, probabilities_bar, &
            theta_bar, status)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. self%is_fitted .or. size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= self%n_classes .or. &
            size(theta_bar) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB parameter probability VJP: model or shapes are invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call self%predict_log_proba_parameter_vjp(x, probabilities_bar*probabilities, &
            theta_bar, status)
    end subroutine gaussian_nb_predict_proba_parameter_vjp

    function gaussian_nb_classes(self) result(labels)
        class(gaussian_naive_bayes_t), intent(in) :: self
        integer, allocatable :: labels(:)

        if (allocated(self%class_label)) then
            labels = self%class_label
        else
            allocate(labels(0))
        end if
    end function gaussian_nb_classes

    function gaussian_nb_means(self) result(values)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%mean)) then
            values = self%mean
        else
            allocate(values(0, 0))
        end if
    end function gaussian_nb_means

    function gaussian_nb_variances(self) result(values)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%variance)) then
            values = self%variance
        else
            allocate(values(0, 0))
        end if
    end function gaussian_nb_variances

    function gaussian_nb_class_prior(self) result(values)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%prior)) then
            values = self%prior
        else
            allocate(values(0))
        end if
    end function gaussian_nb_class_prior

    function gaussian_nb_weighted_counts(self) result(values)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%weighted_count)) then
            values = self%weighted_count
        else
            allocate(values(0))
        end if
    end function gaussian_nb_weighted_counts

    real(dp) function gaussian_nb_var_smoothing(self) result(value)
        class(gaussian_naive_bayes_t), intent(in) :: self

        value = self%var_smoothing
    end function gaussian_nb_var_smoothing

    real(dp) function gaussian_nb_epsilon(self) result(value)
        class(gaussian_naive_bayes_t), intent(in) :: self

        value = self%epsilon
    end function gaussian_nb_epsilon

    integer function gaussian_nb_feature_count(self) result(count)
        class(gaussian_naive_bayes_t), intent(in) :: self

        count = self%n_features
    end function gaussian_nb_feature_count

    integer function gaussian_nb_class_count(self) result(count)
        class(gaussian_naive_bayes_t), intent(in) :: self

        count = self%n_classes
    end function gaussian_nb_class_count

    integer function gaussian_nb_parameter_count(self) result(count)
        class(gaussian_naive_bayes_t), intent(in) :: self

        count = 2*self%n_features*self%n_classes + self%n_classes
    end function gaussian_nb_parameter_count

    function gaussian_nb_parameters(self) result(values)
        class(gaussian_naive_bayes_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        integer :: offset

        allocate(values(self%parameter_count()))
        values = 0.0_dp
        if (.not. self%is_fitted) return
        offset = self%n_features*self%n_classes
        values(:offset) = reshape(self%mean, [offset])
        values(offset + 1:2*offset) = reshape(self%variance, [offset])
        values(2*offset + 1:) = self%prior
    end function gaussian_nb_parameters

    subroutine gaussian_nb_set_parameters(self, values, status)
        class(gaussian_naive_bayes_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: offset
        real(dp) :: prior_sum

        if (.not. self%is_fitted .or. size(values) /= self%parameter_count() .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB set_parameters: model, shape, or values are invalid")
            return
        end if
        offset = self%n_features*self%n_classes
        if (any(values(offset + 1:2*offset) <= 0.0_dp) .or. &
            any(values(2*offset + 1:) <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB set_parameters: variances and priors must be positive")
            return
        end if
        prior_sum = sum(values(2*offset + 1:))
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "GaussianNB set_parameters: prior mass is invalid")
            return
        end if
        self%mean = reshape(values(:offset), shape(self%mean))
        self%variance = reshape(values(offset + 1:2*offset), shape(self%variance))
        self%prior = values(2*offset + 1:)/prior_sum
        call status_set(status, FORTNUM_OK, "")
    end subroutine gaussian_nb_set_parameters

    logical function gaussian_nb_fitted(self) result(is_fitted)
        class(gaussian_naive_bayes_t), intent(in) :: self

        is_fitted = self%is_fitted
    end function gaussian_nb_fitted

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, n
        integer :: temporary

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

end module fortml_gaussian_naive_bayes
