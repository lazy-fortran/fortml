module fortml_discriminant_analysis
    !! Weighted linear and quadratic discriminant analysis classifiers.
    !!
    !! Samples are rows and features are columns.  Both estimators accept
    !! arbitrary integer labels (stored in sorted order), sample weights, and
    !! optional class priors.  LDA uses one pooled covariance; QDA uses one
    !! covariance per class.  Covariances are regularized with a convex
    !! diagonal ridge controlled by ``reg_param`` and are factorized by the
    !! FortNum Cholesky implementation.
    !!
    !! The packed parameter layout is means (column-major), followed by the
    !! lower triangle of each covariance matrix (class-major), followed by
    !! class-prior masses.  Lower-triangle covariance parameters reconstruct a
    !! symmetric matrix, so parameter products are well-defined on the SPD
    !! manifold.  Prediction JVP/VJP products hold the fitted state fixed;
    !! parameter products cover all smooth mean, covariance, and prior
    !! coordinates.  Integer labels and argmax prediction are discrete and do
    !! not expose derivatives.  CUDA calls return a typed refusal until a
    !! resident discriminant-analysis kernel is linked; no host fallback is
    !! hidden behind an accelerator request.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    real(dp), parameter :: DEFAULT_REG_PARAM = 1.0e-9_dp
    real(dp), parameter :: LOG_TWO_PI = 1.837877066409345483560659472811_dp

    type :: discriminant_state_t
        real(dp), allocatable :: mean(:, :)
        real(dp), allocatable :: covariance(:, :, :)
        real(dp), allocatable :: precision(:, :, :)
        real(dp), allocatable :: prior(:)
        real(dp), allocatable :: weighted_count(:)
        integer, allocatable :: class_label(:)
        type(cholesky_factorization_t), allocatable :: factor(:)
        integer :: n_features = 0
        integer :: n_classes = 0
        integer :: covariance_count = 0
        real(dp) :: reg_param = DEFAULT_REG_PARAM
        logical :: qda = .false.
        logical :: is_fitted = .false.
    end type discriminant_state_t

    type, public :: lda_classifier_t
        private
        type(discriminant_state_t) :: state
    contains
        procedure, public :: fit => lda_fit
        procedure, public :: predict_log_proba => lda_predict_log_proba
        procedure, public :: predict_proba => lda_predict_proba
        procedure, public :: predict => lda_predict
        procedure, public :: predict_proba_device => lda_predict_proba_device
        procedure, public :: predict_device => lda_predict_device
        procedure, public :: predict_log_proba_jvp => lda_predict_log_proba_jvp
        procedure, public :: predict_proba_jvp => lda_predict_proba_jvp
        procedure, public :: predict_log_proba_vjp => lda_predict_log_proba_vjp
        procedure, public :: predict_proba_vjp => lda_predict_proba_vjp
        procedure, public :: predict_log_proba_parameter_jvp => &
            lda_predict_log_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            lda_predict_proba_parameter_jvp
        procedure, public :: predict_log_proba_parameter_vjp => &
            lda_predict_log_proba_parameter_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            lda_predict_proba_parameter_vjp
        procedure, public :: classes => lda_classes
        procedure, public :: means => lda_means
        procedure, public :: covariance => lda_covariance
        procedure, public :: class_prior => lda_class_prior
        procedure, public :: weighted_class_counts => lda_weighted_counts
        procedure, public :: regularization => lda_regularization
        procedure, public :: feature_count => lda_feature_count
        procedure, public :: class_count => lda_class_count
        procedure, public :: parameter_count => lda_parameter_count
        procedure, public :: parameters => lda_parameters
        procedure, public :: set_parameters => lda_set_parameters
        procedure, public :: fitted => lda_fitted
        procedure, public :: device_supported => lda_device_supported
    end type lda_classifier_t

    type, public :: qda_classifier_t
        private
        type(discriminant_state_t) :: state
    contains
        procedure, public :: fit => qda_fit
        procedure, public :: predict_log_proba => qda_predict_log_proba
        procedure, public :: predict_proba => qda_predict_proba
        procedure, public :: predict => qda_predict
        procedure, public :: predict_proba_device => qda_predict_proba_device
        procedure, public :: predict_device => qda_predict_device
        procedure, public :: predict_log_proba_jvp => qda_predict_log_proba_jvp
        procedure, public :: predict_proba_jvp => qda_predict_proba_jvp
        procedure, public :: predict_log_proba_vjp => qda_predict_log_proba_vjp
        procedure, public :: predict_proba_vjp => qda_predict_proba_vjp
        procedure, public :: predict_log_proba_parameter_jvp => &
            qda_predict_log_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_jvp => &
            qda_predict_proba_parameter_jvp
        procedure, public :: predict_log_proba_parameter_vjp => &
            qda_predict_log_proba_parameter_vjp
        procedure, public :: predict_proba_parameter_vjp => &
            qda_predict_proba_parameter_vjp
        procedure, public :: classes => qda_classes
        procedure, public :: means => qda_means
        procedure, public :: covariances => qda_covariances
        procedure, public :: class_prior => qda_class_prior
        procedure, public :: weighted_class_counts => qda_weighted_counts
        procedure, public :: regularization => qda_regularization
        procedure, public :: feature_count => qda_feature_count
        procedure, public :: class_count => qda_class_count
        procedure, public :: parameter_count => qda_parameter_count
        procedure, public :: parameters => qda_parameters
        procedure, public :: set_parameters => qda_set_parameters
        procedure, public :: fitted => qda_fitted
        procedure, public :: device_supported => qda_device_supported
    end type qda_classifier_t

    public :: lda_fit, lda_predict_log_proba, lda_predict_proba, lda_predict
    public :: qda_fit, qda_predict_log_proba, qda_predict_proba, qda_predict

contains

    subroutine lda_fit(self, x, labels, status, reg_param, priors, sample_weight)
        class(lda_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: reg_param, priors(:), sample_weight(:)

        call discriminant_fit(self%state, x, labels, status, .false., reg_param, &
            priors, sample_weight)
    end subroutine lda_fit

    subroutine qda_fit(self, x, labels, status, reg_param, priors, sample_weight)
        class(qda_classifier_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: reg_param, priors(:), sample_weight(:)

        call discriminant_fit(self%state, x, labels, status, .true., reg_param, &
            priors, sample_weight)
    end subroutine qda_fit

    subroutine discriminant_fit(state, x, labels, status, qda, reg_param, priors, &
            sample_weight)
        type(discriminant_state_t), intent(out) :: state
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in) :: qda
        real(dp), intent(in), optional :: reg_param, priors(:), sample_weight(:)
        integer, allocatable :: classes(:), class_index_values(:)
        real(dp), allocatable :: weights(:)
        real(dp) :: requested_reg, total_weight, prior_sum, centered_i, centered_j
        integer :: n_samples, n_features, n_classes, i, j, c, k

        state%is_fitted = .false.
        requested_reg = DEFAULT_REG_PARAM
        if (present(reg_param)) requested_reg = reg_param
        if (.not. ieee_is_finite(requested_reg) .or. requested_reg < 0.0_dp .or. &
            requested_reg > 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant fit: reg_param must be finite in [0,1]")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 1 .or. n_features < 1 .or. size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant fit: input dimensions are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant fit: inputs must be finite")
            return
        end if
        call sorted_unique_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant fit: at least two classes are required")
            return
        end if
        allocate(weights(n_samples), class_index_values(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "discriminant fit: sample weights must be finite and nonnegative")
                return
            end if
            weights = sample_weight
        end if
        total_weight = sum(weights)
        if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant fit: sample weights need positive total mass")
            return
        end if
        do i = 1, n_samples
            class_index_values(i) = class_index(classes, labels(i))
        end do

        allocate(state%mean(n_features, n_classes), state%prior(n_classes), &
            state%weighted_count(n_classes), state%class_label(n_classes))
        state%mean = 0.0_dp
        state%weighted_count = 0.0_dp
        state%class_label = classes
        do c = 1, n_classes
            do i = 1, n_samples
                if (class_index_values(i) == c) then
                    state%weighted_count(c) = state%weighted_count(c) + weights(i)
                    state%mean(:, c) = state%mean(:, c) + weights(i)*x(i, :)
                end if
            end do
            if (.not. ieee_is_finite(state%weighted_count(c)) .or. &
                state%weighted_count(c) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "discriminant fit: every class needs positive effective weight")
                return
            end if
            state%mean(:, c) = state%mean(:, c)/state%weighted_count(c)
        end do
        if (present(priors)) then
            if (size(priors) /= n_classes .or. any(.not. ieee_is_finite(priors)) &
                .or. any(priors <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "discriminant fit: priors must be finite and positive")
                return
            end if
            state%prior = priors
        else
            state%prior = state%weighted_count
        end if
        prior_sum = sum(state%prior)
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant fit: prior mass is invalid")
            return
        end if
        state%prior = state%prior/prior_sum
        state%qda = qda
        state%covariance_count = merge(n_classes, 1, qda)
        allocate(state%covariance(n_features, n_features, state%covariance_count))
        state%covariance = 0.0_dp
        if (qda) then
            do c = 1, n_classes
                do i = 1, n_samples
                    if (class_index_values(i) /= c) cycle
                    do j = 1, n_features
                        centered_j = x(i, j) - state%mean(j, c)
                        do k = 1, n_features
                            centered_i = x(i, k) - state%mean(k, c)
                            state%covariance(k, j, c) = &
                                state%covariance(k, j, c) + weights(i)*centered_i*centered_j
                        end do
                    end do
                end do
                state%covariance(:, :, c) = state%covariance(:, :, c) / &
                    state%weighted_count(c)
            end do
        else
            do i = 1, n_samples
                c = class_index_values(i)
                do j = 1, n_features
                    centered_j = x(i, j) - state%mean(j, c)
                    do k = 1, n_features
                        centered_i = x(i, k) - state%mean(k, c)
                        state%covariance(k, j, 1) = state%covariance(k, j, 1) + &
                            weights(i)*centered_i*centered_j
                    end do
                end do
            end do
            state%covariance(:, :, 1) = state%covariance(:, :, 1)/total_weight
        end if
        do c = 1, state%covariance_count
            do j = 1, n_features
                do k = j + 1, n_features
                    state%covariance(k, j, c) = 0.5_dp*(state%covariance(k, j, c) + &
                        state%covariance(j, k, c))
                    state%covariance(j, k, c) = state%covariance(k, j, c)
                end do
                state%covariance(j, j, c) = (1.0_dp - requested_reg)* &
                    state%covariance(j, j, c) + requested_reg
                do k = 1, j - 1
                    state%covariance(k, j, c) = (1.0_dp - requested_reg)* &
                        state%covariance(k, j, c)
                    state%covariance(j, k, c) = state%covariance(k, j, c)
                end do
            end do
        end do
        state%n_features = n_features
        state%n_classes = n_classes
        state%reg_param = requested_reg
        call rebuild_factorizations(state, status)
        if (status%code /= FORTNUM_OK) return
        state%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine discriminant_fit

    subroutine rebuild_factorizations(state, status)
        type(discriminant_state_t), intent(inout) :: state
        type(fortnum_status_t), intent(out) :: status
        integer :: c, i

        if (allocated(state%factor)) deallocate(state%factor)
        if (allocated(state%precision)) deallocate(state%precision)
        allocate(state%factor(state%covariance_count), &
            state%precision(state%n_features, state%n_features, state%covariance_count))
        do c = 1, state%covariance_count
            call state%factor(c)%factorize(state%covariance(:, :, c), status)
            if (status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "discriminant covariance is not positive definite")
                return
            end if
            state%precision(:, :, c) = 0.0_dp
            do i = 1, state%n_features
                state%precision(i, i, c) = 1.0_dp
            end do
            call state%factor(c)%solve(state%precision(:, :, c), status)
            if (status%code /= FORTNUM_OK) return
            state%precision(:, :, c) = 0.5_dp*(state%precision(:, :, c) + &
                transpose(state%precision(:, :, c)))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine rebuild_factorizations

    subroutine disc_predict_log_proba(state, x, log_probabilities, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: log_probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: log_joint(state%n_classes), delta(state%n_features)
        real(dp) :: maximum, normalizer, logdet
        integer :: i, c, covariance_index

        if (.not. state%is_fitted .or. size(x, 2) /= state%n_features .or. &
            any(shape(log_probabilities) /= [size(x, 1), state%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant log probability: model or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant log probability: inputs must be finite")
            return
        end if
        do i = 1, size(x, 1)
            do c = 1, state%n_classes
                delta = x(i, :) - state%mean(:, c)
                covariance_index = merge(c, 1, state%qda)
                call state%factor(covariance_index)%log_determinant(logdet, status)
                if (status%code /= FORTNUM_OK) return
                log_joint(c) = log(state%prior(c)) - 0.5_dp*(logdet + &
                    dot_product(delta, matmul(state%precision(:, :, covariance_index), delta)) + &
                    real(state%n_features, dp)*LOG_TWO_PI)
            end do
            maximum = maxval(log_joint)
            normalizer = sum(exp(log_joint - maximum))
            if (.not. ieee_is_finite(normalizer) .or. normalizer <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "discriminant log probability: normalization is invalid")
                return
            end if
            log_probabilities(i, :) = log_joint - maximum - log(normalizer)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_predict_log_proba

    subroutine disc_predict_proba(state, x, probabilities, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        call disc_predict_log_proba(state, x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(probabilities)
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_predict_proba

    subroutine disc_predict(state, x, labels, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (.not. state%is_fitted .or. size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant predict: model or output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), state%n_classes))
        call disc_predict_proba(state, x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = state%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_predict

    subroutine disc_input_products(state, x, x_dot, log_probabilities, &
            log_probabilities_dot, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:)
        real(dp) :: delta(state%n_features), gradient(state%n_features)
        real(dp) :: joint_dot(state%n_classes), weighted_dot
        integer :: i, c, covariance_index

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2) .or. &
            any(shape(log_probabilities_dot) /= [size(x, 1), state%n_classes]) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant log probability JVP: tangent or output shape is invalid")
            return
        end if
        call disc_predict_log_proba(state, x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        allocate(probabilities(state%n_classes))
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            do c = 1, state%n_classes
                covariance_index = merge(c, 1, state%qda)
                delta = x(i, :) - state%mean(:, c)
                gradient = -matmul(state%precision(:, :, covariance_index), delta)
                joint_dot(c) = dot_product(gradient, x_dot(i, :))
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_input_products

    subroutine disc_input_vjp(state, x, log_probabilities_bar, x_bar, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        real(dp) :: delta(state%n_features), gradient(state%n_features)
        real(dp) :: alpha, cotangent_sum
        integer :: i, c, covariance_index

        x_bar = 0.0_dp
        if (.not. state%is_fitted .or. size(x, 2) /= state%n_features .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), state%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant log probability VJP: model or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant log probability VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(probabilities(size(x, 1), state%n_classes))
        call disc_predict_proba(state, x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            cotangent_sum = sum(log_probabilities_bar(i, :))
            do c = 1, state%n_classes
                alpha = log_probabilities_bar(i, c) - cotangent_sum*probabilities(i, c)
                covariance_index = merge(c, 1, state%qda)
                delta = x(i, :) - state%mean(:, c)
                gradient = -matmul(state%precision(:, :, covariance_index), delta)
                x_bar(i, :) = x_bar(i, :) + alpha*gradient
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_input_vjp

    subroutine disc_parameter_products(state, x, theta_dot, log_probabilities, &
            log_probabilities_dot, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: log_probabilities(:, :), log_probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: mean_dot(:, :), covariance_dot(:, :, :), prior_dot(:)
        real(dp), allocatable :: probabilities(:)
        real(dp) :: delta(state%n_features), gradient(state%n_features)
        real(dp) :: joint_dot(state%n_classes), weighted_dot, covariance_term
        integer :: i, c, j, k, covariance_index

        if (size(theta_dot) /= disc_parameter_count(state) .or. &
            any(.not. ieee_is_finite(theta_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant parameter JVP: parameter tangent is invalid")
            return
        end if
        call unpack_parameter_tangent(state, theta_dot, mean_dot, covariance_dot, &
            prior_dot, status)
        if (status%code /= FORTNUM_OK) return
        call disc_predict_log_proba(state, x, log_probabilities, status)
        if (status%code /= FORTNUM_OK) return
        allocate(probabilities(state%n_classes))
        do i = 1, size(x, 1)
            probabilities = exp(log_probabilities(i, :))
            do c = 1, state%n_classes
                covariance_index = merge(c, 1, state%qda)
                delta = x(i, :) - state%mean(:, c)
                gradient = matmul(state%precision(:, :, covariance_index), delta)
                joint_dot(c) = dot_product(gradient, mean_dot(:, c)) + &
                    prior_dot(c)/state%prior(c)
                covariance_term = 0.0_dp
                do j = 1, state%n_features
                    do k = 1, state%n_features
                        covariance_term = covariance_term + 0.5_dp*( &
                            gradient(j)*gradient(k) - state%precision(j, k, covariance_index))* &
                            covariance_dot(j, k, covariance_index)
                    end do
                end do
                joint_dot(c) = joint_dot(c) + covariance_term
            end do
            weighted_dot = dot_product(probabilities, joint_dot)
            log_probabilities_dot(i, :) = joint_dot - weighted_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_parameter_products

    subroutine disc_parameter_vjp(state, x, log_probabilities_bar, theta_bar, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :), log_probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :), mean_bar(:, :)
        real(dp), allocatable :: covariance_bar(:, :, :), prior_bar(:)
        real(dp) :: delta(state%n_features), gradient(state%n_features)
        real(dp) :: alpha, cotangent_sum, coefficient
        integer :: i, c, j, k, covariance_index, offset, tri_offset

        theta_bar = 0.0_dp
        if (.not. state%is_fitted .or. size(x, 2) /= state%n_features .or. &
            size(theta_bar) /= disc_parameter_count(state) .or. &
            any(shape(log_probabilities_bar) /= [size(x, 1), state%n_classes])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant parameter VJP: model, parameter, or cotangent shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(log_probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant parameter VJP: inputs and cotangents must be finite")
            return
        end if
        allocate(probabilities(size(x, 1), state%n_classes), &
            mean_bar(state%n_features, state%n_classes), &
            covariance_bar(state%n_features, state%n_features, state%covariance_count), &
            prior_bar(state%n_classes))
        mean_bar = 0.0_dp
        covariance_bar = 0.0_dp
        prior_bar = 0.0_dp
        call disc_predict_proba(state, x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            cotangent_sum = sum(log_probabilities_bar(i, :))
            do c = 1, state%n_classes
                alpha = log_probabilities_bar(i, c) - cotangent_sum*probabilities(i, c)
                covariance_index = merge(c, 1, state%qda)
                delta = x(i, :) - state%mean(:, c)
                gradient = matmul(state%precision(:, :, covariance_index), delta)
                mean_bar(:, c) = mean_bar(:, c) + alpha*gradient
                prior_bar(c) = prior_bar(c) + alpha/state%prior(c)
                do j = 1, state%n_features
                    do k = 1, j
                        coefficient = 0.5_dp*(gradient(j)*gradient(k) - &
                            state%precision(j, k, covariance_index))
                        if (j /= k) coefficient = 2.0_dp*coefficient
                        covariance_bar(j, k, covariance_index) = &
                            covariance_bar(j, k, covariance_index) + alpha*coefficient
                    end do
                end do
            end do
        end do
        offset = state%n_features*state%n_classes
        theta_bar(:offset) = reshape(mean_bar, [offset])
        tri_offset = offset
        do covariance_index = 1, state%covariance_count
            do j = 1, state%n_features
                do k = 1, j
                    tri_offset = tri_offset + 1
                    theta_bar(tri_offset) = covariance_bar(j, k, covariance_index)
                end do
            end do
        end do
        prior_bar = prior_bar - sum(prior_bar)/real(state%n_classes, dp)
        theta_bar(tri_offset + 1:) = prior_bar
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_parameter_vjp

    subroutine unpack_parameter_tangent(state, values, mean_dot, covariance_dot, &
            prior_dot, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: values(:)
        real(dp), allocatable, intent(out) :: mean_dot(:, :), covariance_dot(:, :, :)
        real(dp), allocatable, intent(out) :: prior_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: offset, tri_offset, c, j, k

        allocate(mean_dot(state%n_features, state%n_classes), &
            covariance_dot(state%n_features, state%n_features, state%covariance_count), &
            prior_dot(state%n_classes))
        mean_dot = reshape(values(:state%n_features*state%n_classes), shape(mean_dot))
        offset = state%n_features*state%n_classes
        covariance_dot = 0.0_dp
        tri_offset = offset
        do c = 1, state%covariance_count
            do j = 1, state%n_features
                do k = 1, j
                    tri_offset = tri_offset + 1
                    covariance_dot(j, k, c) = values(tri_offset)
                    covariance_dot(k, j, c) = values(tri_offset)
                end do
            end do
        end do
        prior_dot = values(tri_offset + 1:)
        call status_set(status, FORTNUM_OK, "")
    end subroutine unpack_parameter_tangent

    subroutine disc_parameter_jvp(state, x, theta_dot, probabilities, probabilities_dot, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant probability parameter JVP: output shape is invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), state%n_classes), &
            log_dot(size(x, 1), state%n_classes))
        call disc_parameter_products(state, x, theta_dot, log_probabilities, log_dot, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(log_probabilities)
        probabilities_dot = probabilities*log_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_parameter_jvp

    subroutine disc_probability_jvp(state, x, x_dot, probabilities, probabilities_dot, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: log_probabilities(:, :), log_dot(:, :)

        if (any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant probability JVP: output shape is invalid")
            return
        end if
        allocate(log_probabilities(size(x, 1), state%n_classes), &
            log_dot(size(x, 1), state%n_classes))
        call disc_input_products(state, x, x_dot, log_probabilities, log_dot, status)
        if (status%code /= FORTNUM_OK) return
        probabilities = exp(log_probabilities)
        probabilities_dot = probabilities*log_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_probability_jvp

    subroutine disc_probability_vjp(state, x, probabilities_bar, x_bar, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= state%n_classes .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant probability VJP: cotangent shape or values are invalid")
            return
        end if
        allocate(probabilities(size(x, 1), state%n_classes))
        call disc_predict_proba(state, x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call disc_input_vjp(state, x, probabilities_bar*probabilities, x_bar, status)
    end subroutine disc_probability_vjp

    subroutine disc_parameter_vjp_probability(state, x, probabilities_bar, theta_bar, status)
        type(discriminant_state_t), intent(in) :: state
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)

        if (.not. state%is_fitted .or. size(probabilities_bar, 1) /= size(x, 1) .or. &
            size(probabilities_bar, 2) /= state%n_classes .or. &
            size(theta_bar) /= disc_parameter_count(state) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant probability parameter VJP: model or shapes are invalid")
            return
        end if
        allocate(probabilities(size(x, 1), state%n_classes))
        call disc_predict_proba(state, x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call disc_parameter_vjp(state, x, probabilities_bar*probabilities, theta_bar, status)
    end subroutine disc_parameter_vjp_probability

    subroutine disc_set_parameters(state, values, status)
        type(discriminant_state_t), intent(inout) :: state
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: covariance(:, :, :), means(:, :), prior(:)
        integer :: offset, tri_offset, c, j, k
        real(dp) :: prior_sum

        if (.not. state%is_fitted .or. size(values) /= disc_parameter_count(state) .or. &
            any(.not. ieee_is_finite(values))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant set_parameters: model, shape, or values are invalid")
            return
        end if
        allocate(means(state%n_features, state%n_classes), &
            covariance(state%n_features, state%n_features, state%covariance_count), &
            prior(state%n_classes))
        offset = state%n_features*state%n_classes
        means = reshape(values(:offset), shape(means))
        covariance = 0.0_dp
        tri_offset = offset
        do c = 1, state%covariance_count
            do j = 1, state%n_features
                do k = 1, j
                    tri_offset = tri_offset + 1
                    covariance(j, k, c) = values(tri_offset)
                    covariance(k, j, c) = values(tri_offset)
                end do
            end do
        end do
        prior = values(tri_offset + 1:)
        if (any(prior <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant set_parameters: priors must be positive")
            return
        end if
        prior_sum = sum(prior)
        if (.not. ieee_is_finite(prior_sum) .or. prior_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant set_parameters: prior mass is invalid")
            return
        end if
        state%mean = means
        state%covariance = covariance
        state%prior = prior/prior_sum
        call rebuild_factorizations(state, status)
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine disc_set_parameters

    integer function disc_parameter_count(state) result(count)
        type(discriminant_state_t), intent(in) :: state
        count = state%n_features*state%n_classes + &
            state%covariance_count*state%n_features*(state%n_features + 1)/2 + &
            state%n_classes
    end function disc_parameter_count

    integer function covariance_parameter_count(state) result(count)
        type(discriminant_state_t), intent(in) :: state
        count = state%covariance_count*state%n_features*(state%n_features + 1)/2
    end function covariance_parameter_count

    function disc_parameters(state) result(values)
        type(discriminant_state_t), intent(in) :: state
        real(dp), allocatable :: values(:)
        integer :: offset, tri_offset, c, j, k

        allocate(values(disc_parameter_count(state)))
        values = 0.0_dp
        if (.not. state%is_fitted) return
        offset = state%n_features*state%n_classes
        values(:offset) = reshape(state%mean, [offset])
        tri_offset = offset
        do c = 1, state%covariance_count
            do j = 1, state%n_features
                do k = 1, j
                    tri_offset = tri_offset + 1
                    values(tri_offset) = state%covariance(j, k, c)
                end do
            end do
        end do
        values(tri_offset + 1:) = state%prior
    end function disc_parameters

    subroutine disc_device_proba(state, device, x, probabilities, status)
        type(discriminant_state_t), intent(in) :: state
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call disc_predict_proba(state, x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "discriminant device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant device prediction: device kind is invalid")
        end select
    end subroutine disc_device_proba

    subroutine disc_device_labels(state, device, x, labels, status)
        type(discriminant_state_t), intent(in) :: state
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call disc_predict(state, x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "discriminant device prediction: no resident CUDA kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "discriminant device prediction: device kind is invalid")
        end select
    end subroutine disc_device_labels

    logical function disc_device_supported(state, device_kind) result(supported)
        type(discriminant_state_t), intent(in) :: state
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_DEVICE_CPU)
            supported = state%is_fitted
        case (FORTML_DEVICE_CUDA)
            supported = .false.
        case default
            supported = .false.
        end select
    end function disc_device_supported

    ! LDA method wrappers -------------------------------------------------
    subroutine lda_predict_log_proba(self, x, output, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_predict_log_proba(self%state, x, output, status)
    end subroutine lda_predict_log_proba

    subroutine lda_predict_proba(self, x, output, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_predict_proba(self%state, x, output, status)
    end subroutine lda_predict_proba

    subroutine lda_predict(self, x, labels, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_predict(self%state, x, labels, status)
    end subroutine lda_predict

    subroutine lda_predict_proba_device(self, device, x, output, status)
        class(lda_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_device_proba(self%state, device, x, output, status)
    end subroutine lda_predict_proba_device

    subroutine lda_predict_device(self, device, x, labels, status)
        class(lda_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_device_labels(self%state, device, x, labels, status)
    end subroutine lda_predict_device

    subroutine lda_predict_log_proba_jvp(self, x, x_dot, output, output_dot, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: output(:, :), output_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_input_products(self%state, x, x_dot, output, output_dot, status)
    end subroutine lda_predict_log_proba_jvp

    subroutine lda_predict_proba_jvp(self, x, x_dot, output, output_dot, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: output(:, :), output_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_probability_jvp(self%state, x, x_dot, output, output_dot, status)
    end subroutine lda_predict_proba_jvp

    subroutine lda_predict_log_proba_vjp(self, x, output_bar, x_bar, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_input_vjp(self%state, x, output_bar, x_bar, status)
    end subroutine lda_predict_log_proba_vjp

    subroutine lda_predict_proba_vjp(self, x, output_bar, x_bar, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_probability_vjp(self%state, x, output_bar, x_bar, status)
    end subroutine lda_predict_proba_vjp

    subroutine lda_predict_log_proba_parameter_jvp(self, x, theta_dot, output, &
            output_dot, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: output(:, :), output_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_parameter_products(self%state, x, theta_dot, output, output_dot, status)
    end subroutine lda_predict_log_proba_parameter_jvp

    subroutine lda_predict_proba_parameter_jvp(self, x, theta_dot, output, &
            output_dot, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: output(:, :), output_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_parameter_jvp(self%state, x, theta_dot, output, output_dot, status)
    end subroutine lda_predict_proba_parameter_jvp

    subroutine lda_predict_log_proba_parameter_vjp(self, x, output_bar, theta_bar, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_parameter_vjp(self%state, x, output_bar, theta_bar, status)
    end subroutine lda_predict_log_proba_parameter_vjp

    subroutine lda_predict_proba_parameter_vjp(self, x, output_bar, theta_bar, status)
        class(lda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_parameter_vjp_probability(self%state, x, output_bar, theta_bar, status)
    end subroutine lda_predict_proba_parameter_vjp

    function lda_classes(self) result(values)
        class(lda_classifier_t), intent(in) :: self
        integer, allocatable :: values(:)
        values = self%state%class_label
    end function lda_classes

    function lda_means(self) result(values)
        class(lda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        values = self%state%mean
    end function lda_means

    function lda_covariance(self) result(values)
        class(lda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        if (self%state%covariance_count == 1) then
            values = self%state%covariance(:, :, 1)
        else
            allocate(values(0, 0))
        end if
    end function lda_covariance

    function lda_class_prior(self) result(values)
        class(lda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        values = self%state%prior
    end function lda_class_prior

    function lda_weighted_counts(self) result(values)
        class(lda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        values = self%state%weighted_count
    end function lda_weighted_counts

    real(dp) function lda_regularization(self) result(value)
        class(lda_classifier_t), intent(in) :: self
        value = self%state%reg_param
    end function lda_regularization

    integer function lda_feature_count(self) result(value)
        class(lda_classifier_t), intent(in) :: self
        value = self%state%n_features
    end function lda_feature_count

    integer function lda_class_count(self) result(value)
        class(lda_classifier_t), intent(in) :: self
        value = self%state%n_classes
    end function lda_class_count

    integer function lda_parameter_count(self) result(value)
        class(lda_classifier_t), intent(in) :: self
        value = disc_parameter_count(self%state)
    end function lda_parameter_count

    function lda_parameters(self) result(values)
        class(lda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        values = disc_parameters(self%state)
    end function lda_parameters

    subroutine lda_set_parameters(self, values, status)
        class(lda_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_set_parameters(self%state, values, status)
    end subroutine lda_set_parameters

    logical function lda_fitted(self) result(value)
        class(lda_classifier_t), intent(in) :: self
        value = self%state%is_fitted
    end function lda_fitted

    logical function lda_device_supported(self, device_kind) result(value)
        class(lda_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = disc_device_supported(self%state, device_kind)
    end function lda_device_supported

    ! QDA method wrappers -------------------------------------------------
    subroutine qda_predict_log_proba(self, x, output, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_predict_log_proba(self%state, x, output, status)
    end subroutine qda_predict_log_proba

    subroutine qda_predict_proba(self, x, output, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_predict_proba(self%state, x, output, status)
    end subroutine qda_predict_proba

    subroutine qda_predict(self, x, labels, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_predict(self%state, x, labels, status)
    end subroutine qda_predict

    subroutine qda_predict_proba_device(self, device, x, output, status)
        class(qda_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: output(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_device_proba(self%state, device, x, output, status)
    end subroutine qda_predict_proba_device

    subroutine qda_predict_device(self, device, x, labels, status)
        class(qda_classifier_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_device_labels(self%state, device, x, labels, status)
    end subroutine qda_predict_device

    subroutine qda_predict_log_proba_jvp(self, x, x_dot, output, output_dot, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: output(:, :), output_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_input_products(self%state, x, x_dot, output, output_dot, status)
    end subroutine qda_predict_log_proba_jvp

    subroutine qda_predict_proba_jvp(self, x, x_dot, output, output_dot, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: output(:, :), output_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_probability_jvp(self%state, x, x_dot, output, output_dot, status)
    end subroutine qda_predict_proba_jvp

    subroutine qda_predict_log_proba_vjp(self, x, output_bar, x_bar, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_input_vjp(self%state, x, output_bar, x_bar, status)
    end subroutine qda_predict_log_proba_vjp

    subroutine qda_predict_proba_vjp(self, x, output_bar, x_bar, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_probability_vjp(self%state, x, output_bar, x_bar, status)
    end subroutine qda_predict_proba_vjp

    subroutine qda_predict_log_proba_parameter_jvp(self, x, theta_dot, output, &
            output_dot, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: output(:, :), output_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_parameter_products(self%state, x, theta_dot, output, output_dot, status)
    end subroutine qda_predict_log_proba_parameter_jvp

    subroutine qda_predict_proba_parameter_jvp(self, x, theta_dot, output, &
            output_dot, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), theta_dot(:)
        real(dp), intent(out) :: output(:, :), output_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        call disc_parameter_jvp(self%state, x, theta_dot, output, output_dot, status)
    end subroutine qda_predict_proba_parameter_jvp

    subroutine qda_predict_log_proba_parameter_vjp(self, x, output_bar, theta_bar, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_parameter_vjp(self%state, x, output_bar, theta_bar, status)
    end subroutine qda_predict_log_proba_parameter_vjp

    subroutine qda_predict_proba_parameter_vjp(self, x, output_bar, theta_bar, status)
        class(qda_classifier_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:, :)
        real(dp), intent(out) :: theta_bar(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_parameter_vjp_probability(self%state, x, output_bar, theta_bar, status)
    end subroutine qda_predict_proba_parameter_vjp

    function qda_classes(self) result(values)
        class(qda_classifier_t), intent(in) :: self
        integer, allocatable :: values(:)
        values = self%state%class_label
    end function qda_classes

    function qda_means(self) result(values)
        class(qda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)
        values = self%state%mean
    end function qda_means

    function qda_covariances(self) result(values)
        class(qda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:, :, :)
        values = self%state%covariance
    end function qda_covariances

    function qda_class_prior(self) result(values)
        class(qda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        values = self%state%prior
    end function qda_class_prior

    function qda_weighted_counts(self) result(values)
        class(qda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        values = self%state%weighted_count
    end function qda_weighted_counts

    real(dp) function qda_regularization(self) result(value)
        class(qda_classifier_t), intent(in) :: self
        value = self%state%reg_param
    end function qda_regularization

    integer function qda_feature_count(self) result(value)
        class(qda_classifier_t), intent(in) :: self
        value = self%state%n_features
    end function qda_feature_count

    integer function qda_class_count(self) result(value)
        class(qda_classifier_t), intent(in) :: self
        value = self%state%n_classes
    end function qda_class_count

    integer function qda_parameter_count(self) result(value)
        class(qda_classifier_t), intent(in) :: self
        value = disc_parameter_count(self%state)
    end function qda_parameter_count

    function qda_parameters(self) result(values)
        class(qda_classifier_t), intent(in) :: self
        real(dp), allocatable :: values(:)
        values = disc_parameters(self%state)
    end function qda_parameters

    subroutine qda_set_parameters(self, values, status)
        class(qda_classifier_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        call disc_set_parameters(self%state, values, status)
    end subroutine qda_set_parameters

    logical function qda_fitted(self) result(value)
        class(qda_classifier_t), intent(in) :: self
        value = self%state%is_fitted
    end function qda_fitted

    logical function qda_device_supported(self, device_kind) result(value)
        class(qda_classifier_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = disc_device_supported(self%state, device_kind)
    end function qda_device_supported

    subroutine sorted_unique_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer, allocatable :: work(:)
        integer :: i, j, count, value

        allocate(work, source=labels)
        do i = 2, size(work)
            value = work(i)
            j = i - 1
            do while (j >= 1)
                if (work(j) <= value) exit
                work(j + 1) = work(j)
                j = j - 1
            end do
            work(j + 1) = value
        end do
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(i - 1)) count = count + 1
        end do
        allocate(classes(count))
        classes(1) = work(1)
        count = 1
        do i = 2, size(work)
            if (work(i) /= work(i - 1)) then
                count = count + 1
                classes(count) = work(i)
            end if
        end do
    end subroutine sorted_unique_labels

    integer function class_index(classes, label) result(index)
        integer, intent(in) :: classes(:), label
        integer :: i
        index = 0
        do i = 1, size(classes)
            if (classes(i) == label) then
                index = i
                return
            end if
        end do
    end function class_index

end module fortml_discriminant_analysis
